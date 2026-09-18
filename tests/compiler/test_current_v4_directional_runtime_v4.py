from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.playback_directional_assembly_cert_v1 import (
    _source_topology_boundary_edges,
    certify_directional_runtime_v4_assembly_v1,
)
from compiler.realsas_compiler_core.playback_directional_motion_cert_v1 import (
    _boundary_edges,
)
from compiler.realsas_compiler_core.playback_full_surface_v3 import (
    CameraProjectionV3,
    project_points_xyz_v3,
)
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_services.export.current_v4_directional_runtime_v4 import (
    project_directional_product_bakes_to_runtime_v4,
)
from compiler.realsas_compiler_services.export.current_v4_runtime_v2 import (
    RuntimeTexturePayloadV1,
)
from compiler.realsas_compiler_services.proof.motion_bake import (
    bind_qualification_owned_motion_bake,
)


VIEWS = ("V0", "V1")
BODY = "BODY_UNDERLAY"
FG = "HAT_FOREGROUND"


def _camera(view_id: str, index: int) -> CameraProjectionV3:
    return CameraProjectionV3(
        view_id=view_id,
        view_index=index,
        origin=(0.0, 0.0, -5.0),
        right=(1.0, 0.0, 0.0),
        screen_up=(0.0, 1.0, 0.0),
        forward=(0.0, 0.0, 1.0),
        half_extent=32.0,
        resolution=64,
    )


def _texture(index: int) -> RuntimeTexturePayloadV1:
    image_hash = ("%x" % (index + 1)) * 64
    relpath = f"textures/V{index}.png"
    return RuntimeTexturePayloadV1(
        view_index=index,
        image_relpath=relpath,
        image_sha256=image_hash,
        image_crc32=index + 10,
        width=64,
        height=64,
        atlas_payload_hash=content_sha256({
            "image_sha256": image_hash,
            "image_relpath": relpath,
        }),
    )


def _component(view: int, component_id: str, setup_order: int, atlas_hash: str):
    xy = ((10.0, 10.0), (22.0, 10.0), (10.0, 22.0))
    vertices = tuple(
        SimpleNamespace(
            canonical_mesh_vertex_id=f"{component_id}:V{view}:{i}",
            P=(float(i == 1), 0.0, float(i == 2)),
            metadata={"raster_xy": p, "source_raster_xy": p},
        )
        for i, p in enumerate(xy)
    )
    face = tuple(v.canonical_mesh_vertex_id for v in vertices)
    mesh = SimpleNamespace(
        vertices=vertices,
        faces=(face,),
        view_index=view,
        mesh_lineage_hash=content_sha256({"mesh": component_id, "view": view}),
        qualification_report={
            "source_alpha_recall": 1.0,
            "precision_inside_alpha": 1.0,
        },
    )
    mesh_skin = SimpleNamespace(
        mesh_skin_lineage_hash=content_sha256({"skin": component_id, "view": view}),
        metadata={"parent_joint_id": "root"} if component_id != BODY else {},
    )
    corners = tuple(
        SimpleNamespace(
            face_index=0,
            corner_index=i,
            material_uv=(0.0, 0.0),
            donor_view_index=view,
            donor_raster_xy=xy[i],
            authority_class="OBSERVED_LOCAL",
        )
        for i in range(3)
    )
    appearance = SimpleNamespace(
        target_view_index=view,
        atlas_payload_hash=atlas_hash,
        appearance_lineage_hash=content_sha256({"appearance": component_id, "view": view}),
        corner_bindings=corners,
    )
    return SimpleNamespace(
        component_id=component_id,
        view_index=view,
        mesh=mesh,
        mesh_skin=mesh_skin,
        appearance=appearance,
        setup_order=setup_order,
        component_state_hash=content_sha256({"component": component_id, "view": view}),
    )


def _fixture(*, body_first: bool = True):
    textures = tuple(_texture(i) for i in range(2))
    directions = []
    for view in range(2):
        body = _component(view, BODY, 0, textures[view].atlas_payload_hash)
        fg = _component(view, FG, 1, textures[view].atlas_payload_hash)
        directions.append(SimpleNamespace(
            view_index=view,
            components=(body, fg),
        ))

    continuity_views = tuple(
        {
            "view_index": i,
            "substrate_component_id": BODY,
            "metadata": {
                "new_pixels_generated": False,
                "topology_mutated": False,
                "weights_mutated": False,
                "pixel_authority": "EXACT_SUBSTRATE_APPEARANCE_REUSE_ONLY",
            },
        }
        for i in range(2)
    )
    continuity_payload = {
        "views": continuity_views,
        "schema_version": "RealSaS.QualifiedContinuityUnderlaySetIR.v1",
    }
    renderables = SimpleNamespace(
        directions=tuple(directions),
        metadata={
            "continuity_underlay_qualified": True,
            "continuity_underlay_set_hash": "c" * 64,
            "mechanical_continuity_underlay": continuity_payload,
        },
    )
    product = SimpleNamespace(
        product_state_hash="a" * 64,
        directional_renderables=renderables,
        mechanical_state=SimpleNamespace(
            skeleton=SimpleNamespace(
                joints=(SimpleNamespace(
                    canonical_joint_id="root",
                    parent_canonical_id=None,
                ),)
            )
        ),
    )

    rest = {}
    triangles = {}
    frames = []
    for view in range(2):
        for component_id in (BODY, FG):
            mesh_id = f"V{view}:{component_id}"
            offset = 0.0 if component_id == BODY else 4.0
            pts = (
                (10.0 + offset, 10.0),
                (22.0 + offset, 10.0),
                (10.0 + offset, 22.0),
            )
            rest[mesh_id] = pts
            triangles[mesh_id] = ((0, 1, 2),)

    for time_seconds, delta in ((0.0, 0.0), (1.0, 1.0)):
        meshes = {
            mesh_id: tuple((x + delta, y + 0.25 * delta) for x, y in points)
            for mesh_id, points in rest.items()
        }
        orders = {
            f"V{view}": (
                (f"V{view}:{BODY}", f"V{view}:{FG}")
                if body_first
                else (f"V{view}:{FG}", f"V{view}:{BODY}")
            )
            for view in range(2)
        }
        frames.append({
            "time_seconds": time_seconds,
            "mesh_vertices_by_id": meshes,
            "render_order_by_view": orders,
        })

    bake = bind_qualification_owned_motion_bake(
        source_product_state_hash=product.product_state_hash,
        proof_plan_hash="b" * 64,
        clip_id="idle",
        duration_seconds=1.0,
        fps=1.0,
        loop=True,
        evaluator_semantic_version="TEST_DIRECTIONAL_ASSEMBLY_EVALUATOR.v1",
        evaluator_binding_hash="d" * 64,
        sampling_policy="TEST_TWO_ENDPOINTS",
        rest_mesh_vertices_by_id=rest,
        triangles_by_mesh_id=triangles,
        frame_rows=tuple(frames),
    )
    cameras = {view_id: _camera(view_id, i) for i, view_id in enumerate(VIEWS)}
    return product, (bake,), textures, cameras


def test_source_topology_boundary_collapses_runtime_uv_seam_vertex_splits():
    # Runtime appearance vertices 0 and 3 are the same source/mechanical vertex.
    # Vertex 2 remains shared, so raw runtime topology sees a degree-4 branch.
    triangles = np.asarray(((0, 1, 2), (3, 2, 4)), dtype=np.uint32)
    runtime_source_indices = (0, 1, 2, 0, 3)

    with pytest.raises(
        QualificationError,
        match="DIRECTIONAL_BODY_OPEN_OR_BRANCHING_BOUNDARY",
    ):
        _boundary_edges(triangles)

    boundary = _source_topology_boundary_edges(
        triangles,
        runtime_source_indices,
    )
    assert set(tuple(sorted(edge)) for edge in boundary) == {
        (0, 1),
        (0, 4),
        (1, 2),
        (2, 4),
    }


def test_directional_assembly_bridge_roundtrips_qualification_bake_xy_exactly():
    product, bakes, textures, cameras = _fixture()
    projection = project_directional_product_bakes_to_runtime_v4(
        product=product,
        motion_bakes=bakes,
        texture_bindings=textures,
        cameras=cameras,
        required_view_ids=VIEWS,
        body_component_id=BODY,
    )
    assert len(projection.contract.assets) == 4
    assert len(projection.contract.slots) == 2
    assert projection.body_component_id == BODY

    bake_frames = bakes[0].frames
    for runtime_frame, bake_frame in zip(projection.clips[0].frames, bake_frames):
        mesh_xy = dict(bake_frame.mesh_vertices_by_id)
        for asset in projection.contract.assets:
            owner = projection.owner_view_by_asset[asset.asset_id]
            component_id = projection.component_id_by_asset[asset.asset_id]
            source_indices = projection.runtime_source_indices_by_asset[asset.asset_id]
            expected = np.asarray(
                [mesh_xy[f"V{owner}:{component_id}"][i] for i in source_indices],
                dtype=np.float64,
            )
            actual = project_points_xyz_v3(
                runtime_frame.canonical_posed_xyz_by_asset[asset.asset_id],
                projection.contract.views[owner].camera,
            )[:, :2]
            assert np.allclose(actual, expected, atol=2e-5, rtol=0.0)


def test_full_directional_assembly_certificate_passes_source_backed_continuous_fixture():
    product, bakes, textures, cameras = _fixture()
    projection = project_directional_product_bakes_to_runtime_v4(
        product=product,
        motion_bakes=bakes,
        texture_bindings=textures,
        cameras=cameras,
        required_view_ids=VIEWS,
        body_component_id=BODY,
    )
    cert = certify_directional_runtime_v4_assembly_v1(
        product=product,
        projection=projection,
        clip_id="idle",
        required_view_ids=VIEWS,
        body_component_id=BODY,
        min_body_source_alpha_recall=0.99,
        min_body_source_precision=0.99,
    )
    assert cert.source_backed_active_faces_certified
    assert cert.continuity_underlay_certified
    assert cert.continuous_embedding_certified
    assert cert.body_underlay_always_active_first
    assert cert.active_asset_interval_count == 4


def test_full_directional_assembly_certificate_rejects_foreground_before_body():
    product, bakes, textures, cameras = _fixture(body_first=False)
    projection = project_directional_product_bakes_to_runtime_v4(
        product=product,
        motion_bakes=bakes,
        texture_bindings=textures,
        cameras=cameras,
        required_view_ids=VIEWS,
        body_component_id=BODY,
    )
    with pytest.raises(QualificationError, match="DIRECTIONAL_ASSEMBLY_CERT_BODY_NOT_FIRST"):
        certify_directional_runtime_v4_assembly_v1(
            product=product,
            projection=projection,
            clip_id="idle",
            required_view_ids=VIEWS,
            body_component_id=BODY,
        )


def test_full_directional_assembly_certificate_rejects_underlay_that_generates_pixels():
    product, bakes, textures, cameras = _fixture()
    metadata = dict(product.directional_renderables.metadata)
    payload = dict(metadata["mechanical_continuity_underlay"])
    rows = [dict(row) for row in payload["views"]]
    rows[0] = dict(rows[0])
    rows[0]["metadata"] = {
        **dict(rows[0]["metadata"]),
        "new_pixels_generated": True,
    }
    payload["views"] = tuple(rows)
    metadata["mechanical_continuity_underlay"] = payload
    bad_product = SimpleNamespace(
        product_state_hash=product.product_state_hash,
        directional_renderables=SimpleNamespace(
            directions=product.directional_renderables.directions,
            metadata=metadata,
        ),
        mechanical_state=product.mechanical_state,
    )
    projection = project_directional_product_bakes_to_runtime_v4(
        product=bad_product,
        motion_bakes=bakes,
        texture_bindings=textures,
        cameras=cameras,
        required_view_ids=VIEWS,
        body_component_id=BODY,
    )
    with pytest.raises(
        QualificationError,
        match="DIRECTIONAL_ASSEMBLY_CERT_UNDERLAY_NEW_PIXELS_FORBIDDEN",
    ):
        certify_directional_runtime_v4_assembly_v1(
            product=bad_product,
            projection=projection,
            clip_id="idle",
            required_view_ids=VIEWS,
            body_component_id=BODY,
        )
