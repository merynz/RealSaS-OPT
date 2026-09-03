import itertools
import math
from types import SimpleNamespace

import numpy as np
import pytest

from compiler.realsas_compiler_core.directional_binding import (
    DirectionalBindingPolicyV1,
    assert_directional_binding_for_product,
    joint_pivot,
    qualify_directional_joint_view_binding,
)
from compiler.realsas_compiler_core.types import RiggingSurfaceIR, SurfaceNode, SurfaceSupportBinding, QualificationError
from compiler.realsas_compiler_services.proof.directional_motion_evaluator import (
    evaluate_clip_to_qualification_bake,
    make_qualified_motion_bake_provider,
)
from compiler.realsas_compiler_services.proof.directional_motion_provider import (
    QualifiedDirectionalMotionBakeProviderV1,
)
from compiler.realsas_compiler_services.proof.motion_frame_metrics import (
    evaluate_motion_bake_metrics,
    measure_motion_bake_geometry,
)


def _affine(view):
    t = 2.0 * math.pi * view / 8.0
    c, s = math.cos(t), math.sin(t)
    return np.asarray([
        [120.0 * c, -120.0 * s, 20.0, 300.0],
        [15.0 * s, 15.0 * c, 100.0, 280.0],
    ], dtype=np.float64)


def _xy(view, p):
    return tuple((_affine(view) @ np.asarray((*p, 1.0), dtype=np.float64)).tolist())


def _triangle_for_view(points, view):
    best = None
    for tri in itertools.combinations(range(len(points)), 3):
        a, b, c = (np.asarray(_xy(view, points[i])) for i in tri)
        area2 = abs(float(np.cross(b - a, c - a)))
        candidate = (area2, tri)
        if best is None or candidate > best:
            best = candidate
    assert best is not None and best[0] > 1.0
    return best[1]


def _product(*, translation=(0.0, 0.0)):
    points = (
        (-1.0, -1.0, -1.0), (-1.0, -1.0, 1.0), (-1.0, 1.0, -1.0), (-1.0, 1.0, 1.0),
        (1.0, -1.0, -1.0), (1.0, -1.0, 1.0), (1.0, 1.0, -1.0), (1.0, 1.0, 1.0),
    )
    nodes = []
    for i, p in enumerate(points):
        nodes.append(SurfaceNode(
            f"S:{i}", p, tuple(range(8)), ("SYNTH",), (f"OBS:{i}",),
            tuple((view, _xy(view, p)) for view in range(8)), f"G:{i}",
        ))
    surface = RiggingSurfaceIR(tuple(nodes), geometry_lineage_hash="surface-lineage")
    joint = SimpleNamespace(canonical_joint_id="J:root", position=(0.15, -0.10, 0.20), parent_canonical_id=None)
    skeleton = SimpleNamespace(joints=(joint,), skeleton_lineage_hash="skeleton-lineage")
    mechanical = SimpleNamespace(surface=surface, skeleton=skeleton)

    directions = []
    for view in range(8):
        tri = _triangle_for_view(points, view)
        vertices = []
        skin_rows = []
        for local_i, point_i in enumerate(tri):
            sid = f"S:{point_i}"
            vid = f"MV:{view}:{local_i}"
            vertices.append(SimpleNamespace(
                canonical_mesh_vertex_id=vid,
                P=points[point_i],
                support_binding=SurfaceSupportBinding("IDENTITY_SURFACE_NODE", ((sid, 1.0),)),
            ))
            skin_rows.append(SimpleNamespace(canonical_mesh_vertex_id=vid, influences=(("J:root", 1.0),)))
        mesh = SimpleNamespace(vertices=tuple(vertices), faces=((vertices[0].canonical_mesh_vertex_id, vertices[1].canonical_mesh_vertex_id, vertices[2].canonical_mesh_vertex_id),))
        mesh_skin = SimpleNamespace(rows=tuple(skin_rows))
        component = SimpleNamespace(
            component_id="body", mesh=mesh, mesh_skin=mesh_skin,
            setup_order=0, default_visible=True,
        )
        directions.append(SimpleNamespace(view_index=view, camera_binding_hash=f"CAM:{view}", components=(component,)))
    directional = SimpleNamespace(directions=tuple(directions), directional_visual_state_hash="directional-lineage")

    clip = SimpleNamespace(clip_id="idle", clip_kind="PRESET", duration_sec=1.0, loop=True)
    keys = (
        SimpleNamespace(time_sec=0.0, translation_xy=translation, rotation_deg=0.0, scale_xy=(1.0, 1.0), depth_offset=0.0),
        SimpleNamespace(time_sec=0.25, translation_xy=translation, rotation_deg=4.0, scale_xy=(1.0, 1.0), depth_offset=0.0),
        SimpleNamespace(time_sec=0.75, translation_xy=translation, rotation_deg=-4.0, scale_xy=(1.0, 1.0), depth_offset=0.0),
        SimpleNamespace(time_sec=1.0, translation_xy=translation, rotation_deg=0.0, scale_xy=(1.0, 1.0), depth_offset=0.0),
    )
    track = SimpleNamespace(clip_id="idle", canonical_joint_id="J:root", keys=keys)
    motion = SimpleNamespace(clips=(clip,), joint_tracks=(track,), order_tracks=(), visibility_tracks=())
    return SimpleNamespace(
        product_state_hash="a" * 64,
        mechanical_state=mechanical,
        directional_renderables=directional,
        motion_state=motion,
        motion_state_hash="motion-lineage",
    ), clip


def _planar_binding_product(*, joint_z=0.0):
    points = ((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0))
    nodes = tuple(
        SurfaceNode(
            f"PS:{i}", p, tuple(range(8)), ("SYNTH_PLANAR",), (f"POBS:{i}",),
            tuple((view, _xy(view, p)) for view in range(8)), f"PG:{i}",
        )
        for i, p in enumerate(points)
    )
    surface = RiggingSurfaceIR(nodes, geometry_lineage_hash="planar-surface-lineage")
    joint = SimpleNamespace(canonical_joint_id="J:planar", position=(0.1, -0.2, float(joint_z)), parent_canonical_id=None)
    skeleton = SimpleNamespace(joints=(joint,), skeleton_lineage_hash="planar-skeleton-lineage")
    mechanical = SimpleNamespace(surface=surface, skeleton=skeleton)
    directions = tuple(SimpleNamespace(view_index=v, camera_binding_hash=f"PCAM:{v}", components=()) for v in range(8))
    directional = SimpleNamespace(directions=directions, directional_visual_state_hash="planar-directional-lineage")
    return SimpleNamespace(
        product_state_hash="d" * 64,
        mechanical_state=mechanical,
        directional_renderables=directional,
        motion_state_hash="planar-motion-lineage",
    )


def test_affine_surface_correspondence_qualifies_all_joint_view_pivots():
    product, _ = _product()
    binding = qualify_directional_joint_view_binding(product, policy=DirectionalBindingPolicyV1())
    assert binding.source_product_state_hash == product.product_state_hash
    assert len(binding.projections) == 8
    assert len(binding.joint_pivots) == 8
    for view in range(8):
        expected = _xy(view, product.mechanical_state.skeleton.joints[0].position)
        actual = joint_pivot(binding, view, "J:root")
        assert np.allclose(actual, expected, atol=1e-8, rtol=0.0)
        assert binding.projections[view].p95_residual01 < 1e-10


def test_rank3_planar_projection_is_qualified_only_for_joint_on_affine_hull():
    product = _planar_binding_product(joint_z=0.0)
    policy = DirectionalBindingPolicyV1(min_correspondences=4, required_affine_rank=3)
    binding = qualify_directional_joint_view_binding(product, policy=policy)
    assert {row.affine_rank for row in binding.projections} == {3}
    assert all(math.isfinite(row.qualification_report["effective_condition_number"]) for row in binding.projections)
    assert max(row.max_joint_affine_hull_residual01 for row in binding.projections) < 1e-12


def test_rank3_planar_projection_rejects_joint_outside_identified_affine_hull():
    product = _planar_binding_product(joint_z=0.25)
    policy = DirectionalBindingPolicyV1(min_correspondences=4, required_affine_rank=3, max_joint_affine_hull_residual01=0.02)
    with pytest.raises(QualificationError, match="JOINT_OUTSIDE_AFFINE_HULL"):
        qualify_directional_joint_view_binding(product, policy=policy)


def test_stale_product_binding_is_rejected():
    product, _ = _product()
    binding = qualify_directional_joint_view_binding(product)
    stale = SimpleNamespace(**{**product.__dict__, "product_state_hash": "b" * 64})
    with pytest.raises(QualificationError, match="STALE_DIRECTIONAL_BINDING_PRODUCT"):
        assert_directional_binding_for_product(stale, binding)


def test_compatibility_provider_factory_returns_typed_hashed_authority():
    product, _ = _product()
    binding = qualify_directional_joint_view_binding(product)
    provider = make_qualified_motion_bake_provider(binding)
    assert isinstance(provider, QualifiedDirectionalMotionBakeProviderV1)
    provider.assert_for_product(product)
    assert provider.directional_binding_set_hash == binding.binding_set_hash
    assert provider.provider_hash == provider.expected_provider_hash()
    assert len(provider.provider_hash) == 64


def test_rotation_only_directional_evaluator_produces_measured_rigid_loop():
    product, clip = _product()
    binding = qualify_directional_joint_view_binding(product)
    plan = SimpleNamespace(source_product_state_hash=product.product_state_hash, proof_domain="MOTION", proof_plan_hash="c" * 64)
    bake = evaluate_clip_to_qualification_bake(product, plan, clip, binding)
    measured = evaluate_motion_bake_metrics(measure_motion_bake_geometry(bake))
    assert measured["passed"] is True
    assert measured["max_motion01"] > 1e-4
    assert measured["max_edge_stretch_ratio"] == pytest.approx(1.0, abs=1e-9)
    assert measured["max_area_change_ratio"] == pytest.approx(1.0, abs=1e-9)
    assert measured["loop_seam_error01"] == pytest.approx(0.0, abs=1e-10)
    assert bake.metadata["mechanical_xy_used_as_raster_xy"] is False


def test_unqualified_translation_semantics_fail_closed():
    product, clip = _product(translation=(1.0, 0.0))
    binding = qualify_directional_joint_view_binding(product)
    plan = SimpleNamespace(source_product_state_hash=product.product_state_hash, proof_domain="MOTION", proof_plan_hash="c" * 64)
    with pytest.raises(QualificationError, match="TRANSLATION_UNITS_NOT_QUALIFIED"):
        evaluate_clip_to_qualification_bake(product, plan, clip, binding)
