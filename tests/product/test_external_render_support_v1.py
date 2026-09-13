from __future__ import annotations

from dataclasses import replace

import pytest

from compiler.realsas_compiler_core.mesh.direct_model_skin import DIRECT_MODEL_TRANSFER_METHOD
from compiler.realsas_compiler_core.mesh_binding import mesh_lineage_hash, mesh_skin_lineage_hash
from compiler.realsas_compiler_core.product_external_render import (
    assemble_product_v3_with_external_render_support,
    build_external_directional_renderable,
    build_external_directional_renderable_set,
    build_external_renderable_component,
)
from compiler.realsas_compiler_core.types import (
    QualifiedEditableMeshIR,
    QualifiedJoint,
    QualifiedMeshSkinIR,
    QualifiedMeshSkinRow,
    QualifiedMeshVertex,
    QualifiedSkinIR,
    QualifiedSkinRow,
    QualificationError,
    RiggingSurfaceIR,
    SurfaceNode,
    SurfaceSupportBinding,
)
from compiler.realsas_compiler_core.v4 import (
    build_appearance_binding,
    build_capability_contract,
    build_mechanical_state,
    qualified_skeleton_v2_lineage_hash,
)
from compiler.realsas_compiler_core.v4_types import (
    AppearanceCornerBinding,
    CapabilityRequirement,
    QualifiedSkeletonIRV2,
)
from compiler.realsas_compiler_core.motion import build_deterministic_preset_motion


def _mechanical():
    nodes = (
        SurfaceNode("FIT1:S0", (0.0, 0.0, 0.0), (0,), ("FIT1",), (), raster_bindings=((0, (10.0, 10.0)),)),
        SurfaceNode("FIT1:S1", (0.3, 0.0, 0.0), (0,), ("FIT1",), (), raster_bindings=((0, (20.0, 10.0)),)),
    )
    surface = RiggingSurfaceIR(nodes, geometry_lineage_hash="FIT1:SCIENTIFIC:S")
    joints = (
        QualifiedJoint("J:ROOT", (0.0, 0.0, 0.0), None),
        QualifiedJoint("J:ARM", (0.3, 0.0, 0.0), "J:ROOT"),
    )
    skeleton = QualifiedSkeletonIRV2(joints, ("J:ROOT",), {}, {"status": "PASS"}, "")
    skeleton = replace(skeleton, skeleton_lineage_hash=qualified_skeleton_v2_lineage_hash(skeleton))
    skin = QualifiedSkinIR(
        (
            QualifiedSkinRow("FIT1:S0", (("J:ROOT", 1.0),), 0.0, 0.0),
            QualifiedSkinRow("FIT1:S1", (("J:ARM", 1.0),), 0.0, 0.0),
        ),
        surface.geometry_lineage_hash,
        skeleton.skeleton_lineage_hash,
        {"status": "PASS"},
        "FIT1:SOURCE:SKIN",
    )
    return build_mechanical_state(surface, skeleton, skin)


def _p1_mesh(view: int):
    verts = (
        QualifiedMeshVertex("MV:0", (0.0, 0.0, 0.0), SurfaceSupportBinding("LOCAL_CONVEX_INTERPOLATION", (("P1:S0", 1.0),)), metadata={"raster_xy": (10.0, 10.0)}),
        QualifiedMeshVertex("MV:1", (0.3, 0.0, 0.0), SurfaceSupportBinding("LOCAL_CONVEX_INTERPOLATION", (("P1:S1", 1.0),)), metadata={"raster_xy": (20.0, 10.0)}),
        QualifiedMeshVertex("MV:2", (0.15, 0.0, 0.3), SurfaceSupportBinding("LOCAL_CONVEX_INTERPOLATION", (("P1:S2", 1.0),)), metadata={"raster_xy": (15.0, 20.0)}),
    )
    mesh = QualifiedEditableMeshIR(
        verts,
        (("MV:0", "MV:1", "MV:2"),),
        (("MV:0", "MV:1"), ("MV:1", "MV:2"), ("MV:2", "MV:0")),
        "P1:RENDER:SUPPORT:S",
        view,
        f"CAM:{view}",
        {"status": "PASS_EXTERNAL_SUPPORT"},
        "",
        support_coverage_classification="P1_FULL_SUBJECT",
    )
    return replace(mesh, mesh_lineage_hash=mesh_lineage_hash(mesh))


def _p1_skin(mechanical, mesh):
    rows = (
        QualifiedMeshSkinRow("MV:0", (("J:ROOT", 1.0),), (("P1:S0", 1.0),), 0.0, 0.0),
        QualifiedMeshSkinRow("MV:1", (("J:ARM", 1.0),), (("P1:S1", 1.0),), 0.0, 0.0),
        QualifiedMeshSkinRow("MV:2", (("J:ROOT", 0.5), ("J:ARM", 0.5)), (("P1:S2", 1.0),), 0.0, 0.0),
    )
    value = QualifiedMeshSkinIR(
        rows,
        mesh.surface_binding_hash,
        mechanical.skeleton.skeleton_lineage_hash,
        mechanical.skin.skin_lineage_hash,
        mesh.mesh_lineage_hash,
        DIRECT_MODEL_TRANSFER_METHOD,
        {"status": "PASS_DIRECT_MODEL_EXACT_MESH_SKIN_QUALIFICATION"},
        "",
        metadata={"direct_model_query": True, "historical_weight_transfer_used": False},
    )
    return replace(value, mesh_skin_lineage_hash=mesh_skin_lineage_hash(value))


def _appearance(mesh):
    by_id = {v.canonical_mesh_vertex_id: v for v in mesh.vertices}
    corners = []
    for fi, face in enumerate(mesh.faces):
        for ci, vid in enumerate(face):
            xy = by_id[vid].metadata["raster_xy"]
            corners.append(AppearanceCornerBinding(fi, ci, (xy[0] / 31.0, xy[1] / 31.0), mesh.view_index, xy, f"OBS:{mesh.view_index}", "OBSERVED_LOCAL"))
    return build_appearance_binding(
        target_view_index=mesh.view_index,
        mesh_binding_hash=mesh.mesh_lineage_hash,
        camera_binding_hash=mesh.camera_binding_hash,
        corner_bindings=tuple(corners),
    )


def test_external_render_support_does_not_relabel_scientific_surface():
    mechanical = _mechanical()
    mesh = _p1_mesh(0)
    mesh_skin = _p1_skin(mechanical, mesh)
    component = build_external_renderable_component(
        component_id="BODY",
        view_index=0,
        mesh=mesh,
        mesh_skin=mesh_skin,
        mechanical=mechanical,
        appearance=_appearance(mesh),
        setup_order=0,
        coverage_classification="P1_FULL_SUBJECT",
        materialization_manifest_sha256="sha256:materialized-p1",
        direct_binding_manifest_sha256="sha256:direct-v5",
    )
    q = component.metadata["external_render_support_qualification"]
    assert mechanical.surface.geometry_lineage_hash == "FIT1:SCIENTIFIC:S"
    assert component.mesh.surface_binding_hash == "P1:RENDER:SUPPORT:S"
    assert q["render_support_surface_hash"] == "P1:RENDER:SUPPORT:S"
    assert q["metadata"]["scientific_mechanical_surface_relabelled"] is False
    assert q["metadata"]["historical_weight_transfer_used"] is False


def test_external_product_assembles_eight_views_with_current_motion_contract():
    mechanical = _mechanical()
    directions = []
    for view in range(8):
        mesh = _p1_mesh(view)
        mesh_skin = _p1_skin(mechanical, mesh)
        component = build_external_renderable_component(
            component_id="BODY", view_index=view, mesh=mesh, mesh_skin=mesh_skin,
            mechanical=mechanical, appearance=_appearance(mesh), setup_order=0,
            coverage_classification="P1_FULL_SUBJECT",
            materialization_manifest_sha256="sha256:materialized-p1",
            direct_binding_manifest_sha256="sha256:direct-v5",
        )
        directions.append(build_external_directional_renderable(
            view_index=view, camera_binding_hash=mesh.camera_binding_hash,
            components=(component,), mechanical=mechanical,
        ))
    render_set = build_external_directional_renderable_set(tuple(directions), mechanical)
    motion = build_deterministic_preset_motion(mechanical)
    contract = build_capability_contract(
        "P1_TEST",
        (
            CapabilityRequirement("VISUAL_8_DIRECTION", "REQUIRED", "visual", "policy", ("DIRECTIONAL_VISUAL",)),
            CapabilityRequirement("PRESET_MOTION", "REQUIRED", "motion", "policy", ("MOTION",)),
        ),
    )
    product = assemble_product_v3_with_external_render_support(mechanical, render_set, contract, motion)
    assert product.schema_version == "RealSaS.CanonicalPuppetGraph.v3"
    assert product.mechanical_state.surface.geometry_lineage_hash == "FIT1:SCIENTIFIC:S"
    assert product.runtime_policy["external_render_support_allowed"] is True
    assert product.runtime_policy["scientific_mechanical_surface_relabelled"] is False


def test_external_support_rejects_historical_transfer():
    mechanical = _mechanical()
    mesh = _p1_mesh(0)
    good = _p1_skin(mechanical, mesh)
    bad = replace(good, metadata={"direct_model_query": True, "historical_weight_transfer_used": True}, mesh_skin_lineage_hash="")
    bad = replace(bad, mesh_skin_lineage_hash=mesh_skin_lineage_hash(bad))
    with pytest.raises(QualificationError, match="HISTORICAL_WEIGHT_TRANSFER_FORBIDDEN"):
        build_external_renderable_component(
            component_id="BODY", view_index=0, mesh=mesh, mesh_skin=bad,
            mechanical=mechanical, appearance=_appearance(mesh), setup_order=0,
            coverage_classification="P1_FULL_SUBJECT",
            materialization_manifest_sha256="sha256:materialized-p1",
            direct_binding_manifest_sha256="sha256:direct-v5",
        )
