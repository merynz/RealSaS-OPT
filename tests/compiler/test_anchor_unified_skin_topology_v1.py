from __future__ import annotations

import pytest

from compiler.realsas_compiler_core.mesh.anchor_unified import (
    ANCHOR_UNIFIED_TOPOLOGY_METHOD,
    anchor_unified_topology_report,
    require_anchor_unified_topology,
)
from compiler.realsas_compiler_core.mesh.mesh_binding import (
    mesh_candidate_lineage_hash,
    qualify_supported_mesh,
)
from compiler.realsas_compiler_core.mesh.mwb2_skin import bind_mwb2_mesh_skin
from compiler.realsas_compiler_core.types import (
    MeshDiscretizationCandidateIR,
    MeshVertexCandidate,
    QualifiedJoint,
    QualifiedSkeletonIR,
    QualifiedSkinIR,
    QualifiedSkinRow,
    QualificationError,
    RiggingSurfaceIR,
    SurfaceNode,
    SurfaceRelation,
    SurfaceSupportBinding,
)


def _surface():
    nodes = (
        SurfaceNode("A", (0.0, 0.0, 0.0), (0,), (), (), ((0, (0.0, 0.0)),)),
        SurfaceNode("B", (1.0, 0.0, 0.0), (0,), (), (), ((0, (1.0, 0.0)),)),
        SurfaceNode("C", (0.0, 1.0, 0.0), (0,), (), (), ((0, (0.0, 1.0)),)),
        SurfaceNode("D", (1.0, 1.0, 0.0), (0,), (), (), ((0, (1.0, 1.0)),)),
    )
    relations = tuple(
        SurfaceRelation(f"R:{a}:{b}", a, b, "LOCAL", 1.0)
        for a, b in (("A", "B"), ("A", "C"), ("B", "C"), ("B", "D"), ("C", "D"))
    )
    return RiggingSurfaceIR(nodes, relations, "s" * 64)


def _vertex(candidate_id, P, coefficients):
    mode = "IDENTITY_SURFACE_NODE" if len(coefficients) == 1 else "LOCAL_CONVEX_INTERPOLATION"
    return MeshVertexCandidate(
        candidate_id,
        P,
        SurfaceSupportBinding(mode, tuple(coefficients)),
        metadata={"raster_xy": (float(P[0]), float(P[1]))},
    )


def _candidate(surface, vertices, faces):
    edge_set = set()
    for a, b, c in faces:
        edge_set.update({tuple(sorted((a, b))), tuple(sorted((b, c))), tuple(sorted((c, a)))})
    value = MeshDiscretizationCandidateIR(
        tuple(vertices),
        tuple(faces),
        tuple(sorted(edge_set)),
        surface.geometry_lineage_hash,
        0,
        "camera-v0",
        "",
        coverage_classification="TEST",
    )
    return MeshDiscretizationCandidateIR(
        **{**value.__dict__, "candidate_lineage_hash": mesh_candidate_lineage_hash(value)}
    )


def _mechanics(surface):
    skeleton = QualifiedSkeletonIR(
        (QualifiedJoint("J", (0.0, 0.0, 0.0), None, ("A",)),),
        "J",
        {"status": "PASS"},
        "k" * 64,
    )
    rows = tuple(QualifiedSkinRow(node.surface_id, (("J", 1.0),), 0.0, 0.0) for node in surface.surface_nodes)
    skin = QualifiedSkinIR(rows, surface.geometry_lineage_hash, skeleton.skeleton_lineage_hash, {"status": "PASS"}, "w" * 64)
    return skeleton, skin


def test_common_anchor_accepts_relation_supported_face_and_convex_skin_binding():
    surface = _surface()
    vertices = (
        _vertex("a", (0.0, 0.0, 0.0), (("A", 1.0),)),
        _vertex("b", (1.0, 0.0, 0.0), (("B", 1.0),)),
        _vertex("e", (0.25, 0.25, 0.0), (("A", 0.5), ("B", 0.25), ("C", 0.25))),
    )
    mesh = qualify_supported_mesh(surface, _candidate(surface, vertices, (("a", "b", "e"),)))
    report = require_anchor_unified_topology(surface, mesh)
    assert report["passed"] is True
    assert report["method"] == ANCHOR_UNIFIED_TOPOLOGY_METHOD
    assert report["face_without_common_parent_count"] == 0

    skeleton, skin = _mechanics(surface)
    mesh_skin = bind_mwb2_mesh_skin(
        surface,
        skeleton,
        skin,
        mesh,
        require_anchor_unified_topology=True,
    )
    assert mesh_skin.qualification_report["anchor_unified_topology_passed"] is True
    assert mesh_skin.metadata["anchor_unified_topology"] is True


def test_common_anchor_rejects_view_triangle_that_has_no_single_safe_gsa_parent():
    surface = _surface()
    vertices = (
        _vertex("a", (0.0, 0.0, 0.0), (("A", 1.0),)),
        _vertex("b", (1.0, 0.0, 0.0), (("B", 1.0),)),
        _vertex("d", (1.0, 1.0, 0.0), (("D", 1.0),)),
    )
    mesh = qualify_supported_mesh(surface, _candidate(surface, vertices, (("a", "b", "d"),)))
    report = anchor_unified_topology_report(surface, mesh)
    assert report["passed"] is False
    assert report["face_without_common_parent_count"] == 1
    assert report["face_violation_examples"][0]["support_union"] == ("A", "B", "D")

    with pytest.raises(QualificationError, match="ANCHOR_UNIFIED_TOPOLOGY_REQUIRED"):
        require_anchor_unified_topology(surface, mesh)

    skeleton, skin = _mechanics(surface)
    with pytest.raises(QualificationError, match="ANCHOR_UNIFIED_TOPOLOGY_REQUIRED"):
        bind_mwb2_mesh_skin(
            surface,
            skeleton,
            skin,
            mesh,
            require_anchor_unified_topology=True,
        )
