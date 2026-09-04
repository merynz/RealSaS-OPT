from __future__ import annotations

from dataclasses import replace

import pytest

from compiler.realsas_compiler_core.skin import qualify_skin
from compiler.realsas_compiler_core.mesh.mesh_binding import mesh_candidate_lineage_hash, qualify_identity_subset_mesh
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
    SkinInfluenceProposal,
    SkinProposalIR,
    SurfaceNode,
    SurfaceSupportBinding,
)


def _surface(n: int = 2) -> RiggingSurfaceIR:
    nodes = tuple(
        SurfaceNode(f"S{i}", (float(i), 0.0, 0.0), (0, 1), (f"P{i}",), ())
        for i in range(n)
    )
    return RiggingSurfaceIR(nodes, geometry_lineage_hash="SURFACE")


def _skeleton() -> QualifiedSkeletonIR:
    joints = (
        QualifiedJoint("J0", (0.0, 0.0, 0.0), None),
        QualifiedJoint("J1", (1.0, 0.0, 0.0), "J0"),
    )
    return QualifiedSkeletonIR(joints, "J0", {"status": "TEST"}, "SKELETON")


def _proposal(surface: RiggingSurfaceIR, skeleton: QualifiedSkeletonIR, rows: dict[str, tuple[float, float]]) -> SkinProposalIR:
    influences = []
    for sid in sorted(rows):
        a, b = rows[sid]
        influences.extend((SkinInfluenceProposal(sid, "J0", a), SkinInfluenceProposal(sid, "J1", b)))
    return SkinProposalIR(tuple(influences), surface.geometry_lineage_hash, skeleton.skeleton_lineage_hash)


def test_topk_repair_counts_one_row_once_and_reports_full_vector_l1() -> None:
    surface = _surface(1); skeleton = _skeleton()
    q = qualify_skin(surface, skeleton, _proposal(surface, skeleton, {"S0": (0.99, 0.01)}), max_influences=1)
    report = q.qualification_report
    assert report["row_count"] == 1
    assert report["corrected_row_count"] == 1
    assert abs(report["total_sparsification_discarded_mass"] - 0.01) < 1e-12
    # Full-vector L1 is 0.01 removed + 0.01 added to the retained component.
    assert abs(report["total_correction_l1"] - 0.02) < 1e-12


def test_surface_skin_aggregate_budget_cannot_be_spent_independently_per_row() -> None:
    surface = _surface(2); skeleton = _skeleton()
    proposal = _proposal(surface, skeleton, {"S0": (0.994, 0.006), "S1": (0.994, 0.006)})
    with pytest.raises(QualificationError, match="aggregate correction"):
        qualify_skin(surface, skeleton, proposal, max_influences=1, max_simplex_repair_l1=0.02, max_total_correction_l1=0.02)


def test_duplicate_influence_pair_is_rejected_not_silently_summed() -> None:
    surface = _surface(1); skeleton = _skeleton()
    proposal = SkinProposalIR(
        (
            SkinInfluenceProposal("S0", "J0", 0.5),
            SkinInfluenceProposal("S0", "J0", 0.5),
        ),
        surface.geometry_lineage_hash,
        skeleton.skeleton_lineage_hash,
    )
    with pytest.raises(QualificationError, match="duplicate skin influence pair"):
        qualify_skin(surface, skeleton, proposal)


def test_tiny_negative_clipping_is_explicitly_accounted() -> None:
    surface = _surface(1); skeleton = _skeleton()
    q = qualify_skin(surface, skeleton, _proposal(surface, skeleton, {"S0": (-5e-9, 1.000000005)}))
    report = q.qualification_report
    assert report["tiny_negative_clipped_count"] == 1
    assert abs(report["tiny_negative_clipped_mass"] - 5e-9) < 1e-15
    assert report["corrected_row_count"] == 1
    assert report["total_correction_l1"] > 0.0


def _identity_mesh(surface: RiggingSurfaceIR):
    vertices = tuple(
        MeshVertexCandidate(
            f"C{i}",
            node.P,
            SurfaceSupportBinding("IDENTITY_SURFACE_NODE", ((node.surface_id, 1.0),)),
        )
        for i, node in enumerate(surface.surface_nodes)
    )
    candidate = MeshDiscretizationCandidateIR(vertices, (), (), surface.geometry_lineage_hash, 0, "CAM", "")
    candidate = replace(candidate, candidate_lineage_hash=mesh_candidate_lineage_hash(candidate))
    return qualify_identity_subset_mesh(surface, candidate)


def test_mesh_skin_aggregate_normalization_budget_is_fail_closed() -> None:
    surface = _surface(2); skeleton = _skeleton(); mesh = _identity_mesh(surface)
    # Each source row is only 6e-10 off simplex, below the 1e-9 per-row transfer
    # budget, but two such repairs exceed the independent 1e-9 total budget.
    skin = QualifiedSkinIR(
        (
            QualifiedSkinRow("S0", (("J0", 0.5000000003), ("J1", 0.5000000003)), 6e-10, 0.0),
            QualifiedSkinRow("S1", (("J0", 0.5000000003), ("J1", 0.5000000003)), 6e-10, 0.0),
        ),
        surface.geometry_lineage_hash,
        skeleton.skeleton_lineage_hash,
        {"status": "TEST_MALFORMED_QUALIFIED_INPUT"},
        "SKIN",
    )
    with pytest.raises(QualificationError, match="TOTAL_TRANSFER_CORRECTION_BUDGET"):
        bind_mwb2_mesh_skin(surface, skeleton, skin, mesh)
