from __future__ import annotations

from dataclasses import replace

from compiler.realsas_compiler_core.mechanical_component_partition import (
    derive_mechanical_component_partition,
)
from compiler.realsas_compiler_core.mesh.component_partition import (
    project_mesh_to_mechanical_components,
)
from compiler.realsas_compiler_core.mesh.mesh_binding import (
    mesh_lineage_hash,
    mesh_skin_lineage_hash,
)
from compiler.realsas_compiler_core.types import (
    QualifiedEditableMeshIR,
    QualifiedJoint,
    QualifiedMeshSkinIR,
    QualifiedMeshSkinRow,
    QualifiedMeshVertex,
    QualifiedSkinIR,
    QualifiedSkinRow,
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
    nodes = tuple(
        SurfaceNode(
            sid,
            P,
            tuple(range(8)),
            ("obs",),
            (sid,),
            tuple((view, (P[0] * 10.0 + 20.0, P[1] * 10.0 + 20.0)) for view in range(8)),
        )
        for sid, P in (
            ("B0", (0.0, 0.0, 0.0)),
            ("B1", (1.0, 0.0, 0.0)),
            ("B2", (0.0, 1.0, 0.0)),
            ("H0", (2.0, 0.0, 0.0)),
            ("H1", (3.0, 0.0, 0.0)),
            ("H2", (2.0, 1.0, 0.0)),
        )
    )
    surface = RiggingSurfaceIR(nodes, geometry_lineage_hash="SURFACE")
    provisional = QualifiedSkeletonIRV2(
        joints=(
            QualifiedJoint("J_ROOT", (0.0, 0.0, 0.0), None),
            QualifiedJoint("J_HEAD", (0.0, 2.0, 0.0), "J_ROOT"),
        ),
        deform_root_ids=("J_ROOT",),
        assembly_root_binding={},
        qualification_report={"passed": True},
        skeleton_lineage_hash="",
    )
    skeleton = replace(
        provisional,
        skeleton_lineage_hash=qualified_skeleton_v2_lineage_hash(provisional),
    )
    body_weights = (("J_ROOT", 0.5), ("J_HEAD", 0.5))
    rows = tuple(
        QualifiedSkinRow(
            sid,
            (("J_HEAD", 1.0),) if sid.startswith("H") else body_weights,
            0.0,
            0.0,
        )
        for sid in ("B0", "B1", "B2", "H0", "H1", "H2")
    )
    skin = QualifiedSkinIR(
        rows,
        "SURFACE",
        skeleton.skeleton_lineage_hash,
        {"passed": True},
        "SKIN",
    )
    mechanical = build_mechanical_state(surface, skeleton, skin)
    partition = derive_mechanical_component_partition(
        surface=surface,
        skeleton=skeleton,
        skin=skin,
        attachment_role_joints={"HEAD": "J_HEAD"},
        require_attachment_roles=("HEAD",),
    )

    vertices = tuple(
        QualifiedMeshVertex(
            "M_" + sid,
            next(node.P for node in nodes if node.surface_id == sid),
            SurfaceSupportBinding("IDENTITY_SURFACE_NODE", ((sid, 1.0),)),
        )
        for sid in ("B0", "B1", "B2", "H0", "H1", "H2")
    )
    faces = (
        ("M_B0", "M_B1", "M_B2"),
        ("M_H0", "M_H1", "M_H2"),
        ("M_B0", "M_B1", "M_H0"),
    )
    edges = tuple(
        sorted(
            {
                tuple(sorted((face[i], face[(i + 1) % 3])))
                for face in faces
                for i in range(3)
            }
        )
    )
    mesh0 = QualifiedEditableMeshIR(
        vertices,
        faces,
        edges,
        "SURFACE",
        0,
        "CAM",
        {"passed": True},
        "",
        support_coverage_classification="TEST",
    )
    mesh = replace(mesh0, mesh_lineage_hash=mesh_lineage_hash(mesh0))
    skin_rows = tuple(
        QualifiedMeshSkinRow(
            vertex.canonical_mesh_vertex_id,
            (("J_HEAD", 1.0),)
            if vertex.canonical_mesh_vertex_id.startswith("M_H")
            else body_weights,
            tuple(vertex.support_binding.coefficients),
            0.0,
            0.0,
        )
        for vertex in vertices
    )
    mesh_skin0 = QualifiedMeshSkinIR(
        skin_rows,
        "SURFACE",
        skeleton.skeleton_lineage_hash,
        "SKIN",
        mesh.mesh_lineage_hash,
        "SURFACE_CONVEX_INTERPOLATION",
        {"passed": True},
        "",
        metadata={
            "historical_weight_transfer_used": False,
            "direct_model_query": False,
            "surface_support_convex_transfer": True,
            "source_skin_rows_consumed": True,
            "semantic_skin_synthesis": False,
        },
    )
    mesh_skin = replace(
        mesh_skin0,
        mesh_skin_lineage_hash=mesh_skin_lineage_hash(mesh_skin0),
    )
    corners = []
    for face_index, face in enumerate(faces):
        for corner_index, vid in enumerate(face):
            vertex = next(v for v in vertices if v.canonical_mesh_vertex_id == vid)
            x, y = vertex.P[:2]
            corners.append(
                AppearanceCornerBinding(
                    face_index,
                    corner_index,
                    (0.25 + 0.05 * x, 0.25 + 0.05 * y),
                    0,
                    (20.0 + x, 20.0 + y),
                    "a" * 64,
                    "OBSERVED_LOCAL",
                    "",
                    1.0,
                )
            )
    appearance = build_appearance_binding(
        target_view_index=0,
        mesh_binding_hash=mesh.mesh_lineage_hash,
        camera_binding_hash="CAM",
        corner_bindings=tuple(corners),
        atlas_payload_hash="ATLAS",
    )
    return mechanical, partition, mesh, mesh_skin, appearance


def test_partition_uses_only_qualified_mechanics_and_accounts_every_surface():
    mechanical, partition, *_ = _fixture()
    assert partition.qualification_report["teacher_truth_used"] is False
    assert partition.component_surface_ids["BODY_UNDERLAY"] == ("B0", "B1", "B2")
    assert partition.component_surface_ids["RIGID_HEAD"] == ("H0", "H1", "H2")
    assert partition.component_parent_joint_ids["RIGID_HEAD"] == "J_HEAD"
    assert len(partition.assignments) == len(mechanical.surface.surface_nodes)


def test_mesh_projection_keeps_full_underlay_and_adds_exact_rigid_overlay():
    mechanical, partition, mesh, mesh_skin, appearance = _fixture()
    projection = project_mesh_to_mechanical_components(
        source_mesh=mesh,
        source_mesh_skin=mesh_skin,
        source_appearance=appearance,
        partition=partition,
        mechanical=mechanical,
    )
    assert projection.cross_component_face_indices == (2,)
    assert projection.qualification_report["underlay_face_count"] == len(mesh.faces)
    assert projection.qualification_report["source_face_coverage_fraction"] == 1.0
    assert projection.qualification_report["boundary_faces_deleted"] is False
    by_id = {row.component_id: row for row in projection.components}
    assert set(by_id) == {"BODY_UNDERLAY", "RIGID_HEAD"}

    underlay = by_id["BODY_UNDERLAY"]
    assert underlay.mesh is mesh
    assert underlay.mesh_skin is mesh_skin
    assert underlay.appearance is appearance
    assert underlay.source_face_indices == tuple(range(len(mesh.faces)))
    assert len(underlay.mesh.faces) == 3

    head = by_id["RIGID_HEAD"]
    assert head.source_face_indices == (1,)
    assert len(head.mesh.faces) == 1
    assert (
        head.mesh_skin.rows[0].influences
        == next(row.influences for row in mesh_skin.rows if row.canonical_mesh_vertex_id == "M_H0")
    )
    assert head.appearance.corner_bindings[0].source_observation_hash == "a" * 64
