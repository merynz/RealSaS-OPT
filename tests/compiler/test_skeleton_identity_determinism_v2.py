from __future__ import annotations

from compiler.realsas_compiler_core.rig import (
    qualify_skeleton,
    qualify_skeleton_v2,
)
from compiler.realsas_compiler_core.types import (
    RiggingSurfaceIR,
    SkeletonProposalEdge,
    SkeletonProposalIR,
    SkeletonProposalJoint,
    SurfaceNode,
)


def _fixture():
    nodes = (
        SurfaceNode("S0", (0.0, 0.0, 0.0), (0, 1), (), ()),
        SurfaceNode("S1", (0.0, 0.5, 0.0), (0, 1), (), ()),
        SurfaceNode("S2", (0.0, 1.0, 0.0), (0, 1), (), ()),
    )
    surface = RiggingSurfaceIR(
        nodes,
        (),
        "fixture-surface-lineage",
        metadata={"fixture": "SKELETON_IDENTITY_DETERMINISM_V2"},
    )
    joints = (
        SkeletonProposalJoint(
            "P0",
            (0.0, 0.0, 0.0),
            root_score=1.0,
            confidence=1.0,
            support_surface_ids=("S0",),
        ),
        SkeletonProposalJoint(
            "P1",
            (0.0, 0.5, 0.0),
            root_score=0.2,
            confidence=1.0,
            support_surface_ids=("S1",),
        ),
        SkeletonProposalJoint(
            "P2",
            (0.0, 1.0, 0.0),
            root_score=0.1,
            confidence=1.0,
            support_surface_ids=("S2",),
        ),
    )
    edges = (
        SkeletonProposalEdge("E01", "P0", "P1", 1.0, 1.0),
        SkeletonProposalEdge("E12", "P1", "P2", 1.0, 1.0),
        SkeletonProposalEdge("E10", "P1", "P0", 0.1, 1.0),
        SkeletonProposalEdge("E21", "P2", "P1", 0.1, 1.0),
        SkeletonProposalEdge("E02", "P0", "P2", 0.2, 1.0),
        SkeletonProposalEdge("E20", "P2", "P0", 0.1, 1.0),
    )
    proposal = SkeletonProposalIR(
        joints,
        edges,
        surface.geometry_lineage_hash,
        "DETERMINISM_FIXTURE",
    )
    return surface, proposal


def _joint_rows(value):
    return tuple(
        (
            row.canonical_joint_id,
            row.parent_canonical_id,
            row.source_proposal_id,
        )
        for row in value.joints
    )


def test_repeated_legacy_qualification_mints_byte_stable_identity():
    surface, proposal = _fixture()
    rows = [qualify_skeleton(surface, proposal, run_ilp_shadow=False) for _ in range(4)]
    assert len({row.skeleton_lineage_hash for row in rows}) == 1
    assert len({_joint_rows(row) for row in rows}) == 1
    assert all(
        row.qualification_report["optimizer_identity_contract"]
        == "GRAPH_SELECTION_SEMANTICS_V1"
        for row in rows
    )
    assert all(
        row.qualification_report["optimizer_runtime_telemetry_in_identity"] is False
        for row in rows
    )


def test_repeated_v2_qualification_mints_byte_stable_identity():
    surface, proposal = _fixture()
    rows = [qualify_skeleton_v2(surface, proposal, run_ilp_shadow=False) for _ in range(4)]
    assert len({row.skeleton_lineage_hash for row in rows}) == 1
    assert len({_joint_rows(row) for row in rows}) == 1


def test_ilp_shadow_diagnostics_cannot_change_canonical_identity():
    surface, proposal = _fixture()
    without_shadow = qualify_skeleton(surface, proposal, run_ilp_shadow=False)
    with_shadow = qualify_skeleton(surface, proposal, run_ilp_shadow=True)
    assert without_shadow.skeleton_lineage_hash == with_shadow.skeleton_lineage_hash
    assert _joint_rows(without_shadow) == _joint_rows(with_shadow)

    without_shadow_v2 = qualify_skeleton_v2(
        surface, proposal, run_ilp_shadow=False
    )
    with_shadow_v2 = qualify_skeleton_v2(
        surface, proposal, run_ilp_shadow=True
    )
    assert without_shadow_v2.skeleton_lineage_hash == with_shadow_v2.skeleton_lineage_hash
    assert _joint_rows(without_shadow_v2) == _joint_rows(with_shadow_v2)
