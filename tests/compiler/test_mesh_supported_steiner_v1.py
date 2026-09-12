from __future__ import annotations

from dataclasses import replace

from compiler.realsas_compiler_core.mesh.mesh_binding import (
    mesh_candidate_lineage_hash,
    qualify_supported_mesh,
)
from compiler.realsas_compiler_core.mesh.mwb2_skin import bind_mwb2_mesh_skin
from compiler.realsas_compiler_core.types import (
    MeshDiscretizationCandidateIR,
    MeshVertexCandidate,
    QualifiedJoint,
    QualifiedSkinIR,
    QualifiedSkinRow,
    RiggingSurfaceIR,
    SurfaceNode,
    SurfaceSupportBinding,
)
from compiler.realsas_compiler_core.v4 import qualified_skeleton_v2_lineage_hash
from compiler.realsas_compiler_core.v4_types import QualifiedSkeletonIRV2


def test_local_convex_inserted_vertex_is_qualified_and_skin_bound_without_new_semantics():
    surface = RiggingSurfaceIR(
        (
            SurfaceNode("S0", (0.0, 0.0, 0.0), (0,), ("p",), ("o0",), ((0, (0.0, 0.0)),)),
            SurfaceNode("S1", (2.0, 0.0, 0.0), (0,), ("p",), ("o1",), ((0, (2.0, 0.0)),)),
            SurfaceNode("S2", (0.0, 2.0, 0.0), (0,), ("p",), ("o2",), ((0, (0.0, 2.0)),)),
        ),
        geometry_lineage_hash="SURFACE",
    )
    verts = (
        MeshVertexCandidate("C0", (0.0, 0.0, 0.0), SurfaceSupportBinding("IDENTITY_SURFACE_NODE", (("S0", 1.0),)), {"raster_xy": (0.0, 0.0)}),
        MeshVertexCandidate("C1", (2.0, 0.0, 0.0), SurfaceSupportBinding("IDENTITY_SURFACE_NODE", (("S1", 1.0),)), {"raster_xy": (2.0, 0.0)}),
        MeshVertexCandidate("C2", (0.0, 2.0, 0.0), SurfaceSupportBinding("IDENTITY_SURFACE_NODE", (("S2", 1.0),)), {"raster_xy": (0.0, 2.0)}),
        MeshVertexCandidate(
            "CM",
            (1.0, 0.0, 0.0),
            SurfaceSupportBinding("LOCAL_CONVEX_INTERPOLATION", (("S0", 0.5), ("S1", 0.5))),
            {"raster_xy": (1.0, 0.0), "generated_geometry": True, "steiner_role": "BOUNDARY_REFINEMENT"},
        ),
    )
    candidate = MeshDiscretizationCandidateIR(
        verts,
        (("C0", "CM", "C2"), ("CM", "C1", "C2")),
        (("C0", "CM"), ("CM", "C2"), ("C2", "C0"), ("CM", "C1"), ("C1", "C2")),
        "SURFACE",
        0,
        "CAMERA",
        "",
        coverage_classification="OBSERVATION_DOMAIN_CDT",
        residual_report={"source_alpha_recall": 1.0, "precision_inside_alpha": 1.0},
    )
    candidate = replace(candidate, candidate_lineage_hash=mesh_candidate_lineage_hash(candidate))
    mesh = qualify_supported_mesh(surface, candidate)
    assert mesh.qualification_report["local_convex_interpolation_vertex_count"] == 1

    sk0 = QualifiedSkeletonIRV2(
        joints=(QualifiedJoint("J0", (0.0, 0.0, 0.0), None, ("S0", "S1", "S2"), "P0"),),
        deform_root_ids=("J0",),
        assembly_root_binding={},
        qualification_report={"passed": True},
        skeleton_lineage_hash="",
    )
    skeleton = replace(sk0, skeleton_lineage_hash=qualified_skeleton_v2_lineage_hash(sk0))
    skin = QualifiedSkinIR(
        rows=(
            QualifiedSkinRow("S0", (("J0", 1.0),), 0.0, 0.0),
            QualifiedSkinRow("S1", (("J0", 1.0),), 0.0, 0.0),
            QualifiedSkinRow("S2", (("J0", 1.0),), 0.0, 0.0),
        ),
        surface_binding_hash="SURFACE",
        skeleton_binding_hash=skeleton.skeleton_lineage_hash,
        qualification_report={"passed": True},
        skin_lineage_hash="SKIN",
    )
    mesh_skin = bind_mwb2_mesh_skin(surface, skeleton, skin, mesh)
    midpoint = next(v for v in mesh.vertices if v.source_candidate_vertex_id == "CM")
    row = next(r for r in mesh_skin.rows if r.canonical_mesh_vertex_id == midpoint.canonical_mesh_vertex_id)
    assert row.source_support_coefficients == (("S0", 0.5), ("S1", 0.5))
    assert row.influences == (("J0", 1.0),)
