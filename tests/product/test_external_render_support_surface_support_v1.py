from __future__ import annotations

from dataclasses import replace

import pytest

from compiler.realsas_compiler_core.mesh.mwb2_skin import (
    SURFACE_SUPPORT_CONVEX_TRANSFER_METHOD,
    bind_mwb2_mesh_skin,
)
from compiler.realsas_compiler_core.mesh_binding import mesh_lineage_hash, mesh_skin_lineage_hash
from compiler.realsas_compiler_core.product_external_render import build_external_renderable_component
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
    build_mechanical_state,
    qualified_skeleton_v2_lineage_hash,
)
from compiler.realsas_compiler_core.v4_types import (
    AppearanceCornerBinding,
    QualifiedSkeletonIRV2,
)


def _fixture():
    surface = RiggingSurfaceIR(
        (
            SurfaceNode("S0", (0.0, 0.0, 0.0), (0,), ("p",), (), raster_bindings=((0, (0.0, 0.0)),)),
            SurfaceNode("S1", (1.0, 0.0, 0.0), (0,), ("p",), (), raster_bindings=((0, (1.0, 0.0)),)),
        ),
        geometry_lineage_hash="SURFACE",
    )
    skeleton = QualifiedSkeletonIRV2(
        (
            QualifiedJoint("J0", (0.0, 0.0, 0.0), None),
            QualifiedJoint("J1", (1.0, 0.0, 0.0), "J0"),
        ),
        ("J0",),
        {},
        {"status": "PASS"},
        "",
    )
    skeleton = replace(skeleton, skeleton_lineage_hash=qualified_skeleton_v2_lineage_hash(skeleton))
    skin = QualifiedSkinIR(
        (
            QualifiedSkinRow("S0", (("J0", 1.0),), 0.0, 0.0),
            QualifiedSkinRow("S1", (("J1", 1.0),), 0.0, 0.0),
        ),
        surface.geometry_lineage_hash,
        skeleton.skeleton_lineage_hash,
        {"status": "PASS"},
        "SKIN",
    )
    vertices = (
        QualifiedMeshVertex("V0", (0.0, 0.0, 0.0), SurfaceSupportBinding("IDENTITY_SURFACE_NODE", (("S0", 1.0),)), metadata={"raster_xy": (0.0, 0.0)}),
        QualifiedMeshVertex("V1", (1.0, 0.0, 0.0), SurfaceSupportBinding("IDENTITY_SURFACE_NODE", (("S1", 1.0),)), metadata={"raster_xy": (1.0, 0.0)}),
        QualifiedMeshVertex("V2", (0.5, 0.0, 0.0), SurfaceSupportBinding("LOCAL_CONVEX_INTERPOLATION", (("S0", 0.5), ("S1", 0.5))), metadata={"raster_xy": (0.5, 1.0)}),
    )
    mesh = QualifiedEditableMeshIR(
        vertices,
        (("V0", "V1", "V2"),),
        (("V0", "V1"), ("V1", "V2"), ("V2", "V0")),
        surface.geometry_lineage_hash,
        0,
        "CAM:0",
        {"status": "PASS"},
        "",
        support_coverage_classification="P1_FULL_SUBJECT",
    )
    mesh = replace(mesh, mesh_lineage_hash=mesh_lineage_hash(mesh))
    mechanical = build_mechanical_state(surface, skeleton, skin)
    mesh_skin = bind_mwb2_mesh_skin(surface, skeleton, skin, mesh)
    corners = tuple(
        AppearanceCornerBinding(0, i, (0.0, 0.0), 0, (0.0, 0.0), "OBS:0", "OBSERVED_LOCAL")
        for i in range(3)
    )
    appearance = build_appearance_binding(
        target_view_index=0,
        mesh_binding_hash=mesh.mesh_lineage_hash,
        camera_binding_hash=mesh.camera_binding_hash,
        corner_bindings=corners,
    )
    return mechanical, mesh, mesh_skin, appearance


def test_external_render_support_accepts_current_surface_support_convex_transfer():
    mechanical, mesh, mesh_skin, appearance = _fixture()
    assert mesh_skin.transfer_method == SURFACE_SUPPORT_CONVEX_TRANSFER_METHOD
    component = build_external_renderable_component(
        component_id="BODY",
        view_index=0,
        mesh=mesh,
        mesh_skin=mesh_skin,
        mechanical=mechanical,
        appearance=appearance,
        setup_order=0,
        coverage_classification="P1_FULL_SUBJECT",
        materialization_manifest_sha256="sha256:p1",
        direct_binding_manifest_sha256="sha256:fit2-bind",
    )
    q = component.metadata["external_render_support_qualification"]
    assert q["transfer_method"] == SURFACE_SUPPORT_CONVEX_TRANSFER_METHOD
    assert q["metadata"]["surface_support_convex_transfer"] is True
    assert q["metadata"]["surface_support_convex_replay_verified"] is True
    assert q["metadata"]["exact_mesh_vertex_direct_model_query"] is False


def test_external_render_support_rejects_tampered_surface_support_convex_weights():
    mechanical, mesh, good, appearance = _fixture()
    rows = list(good.rows)
    row = rows[2]
    rows[2] = QualifiedMeshSkinRow(
        row.canonical_mesh_vertex_id,
        (("J0", 0.6), ("J1", 0.4)),
        row.source_support_coefficients,
        row.simplex_residual_before,
        row.correction_l1,
    )
    bad = QualifiedMeshSkinIR(
        tuple(rows),
        good.surface_binding_hash,
        good.skeleton_binding_hash,
        good.skin_binding_hash,
        good.mesh_binding_hash,
        good.transfer_method,
        good.qualification_report,
        "",
        metadata=dict(good.metadata),
    )
    bad = replace(bad, mesh_skin_lineage_hash=mesh_skin_lineage_hash(bad))
    with pytest.raises(QualificationError, match="SURFACE_CONVEX_REPLAY_MISMATCH"):
        build_external_renderable_component(
            component_id="BODY",
            view_index=0,
            mesh=mesh,
            mesh_skin=bad,
            mechanical=mechanical,
            appearance=appearance,
            setup_order=0,
            coverage_classification="P1_FULL_SUBJECT",
            materialization_manifest_sha256="sha256:p1",
            direct_binding_manifest_sha256="sha256:fit2-bind",
        )
