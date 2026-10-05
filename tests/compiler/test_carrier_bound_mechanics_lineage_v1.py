from compiler.realsas_compiler_core.rig import qualify_skeleton
from compiler.realsas_compiler_core.skin import qualify_skin
from compiler.realsas_compiler_core.types import (
    RiggingSurfaceIR,
    SkeletonProposalEdge,
    SkeletonProposalIR,
    SkeletonProposalJoint,
    SkinInfluenceProposal,
    SkinProposalIR,
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
        metadata={"fixture": "CARRIER_BOUND_MECHANICS"},
    )
    proposal = SkeletonProposalIR(
        (
            SkeletonProposalJoint(
                "P0", (0.0, 0.0, 0.0), 1.0, 1.0, ("S0",)
            ),
            SkeletonProposalJoint(
                "P1", (0.0, 0.5, 0.0), 0.2, 1.0, ("S1",)
            ),
            SkeletonProposalJoint(
                "P2", (0.0, 1.0, 0.0), 0.1, 1.0, ("S2",)
            ),
        ),
        (
            SkeletonProposalEdge("E01", "P0", "P1", 1.0, 1.0),
            SkeletonProposalEdge("E12", "P1", "P2", 1.0, 1.0),
            SkeletonProposalEdge("E10", "P1", "P0", 0.1, 1.0),
            SkeletonProposalEdge("E21", "P2", "P1", 0.1, 1.0),
            SkeletonProposalEdge("E02", "P0", "P2", 0.2, 1.0),
            SkeletonProposalEdge("E20", "P2", "P0", 0.1, 1.0),
        ),
        surface.geometry_lineage_hash,
        "CARRIER_BOUND_FIXTURE",
    )
    return surface, proposal


def _joint_semantics(skeleton):
    return tuple(
        (
            row.canonical_joint_id,
            row.position,
            row.parent_canonical_id,
            row.source_proposal_id,
        )
        for row in skeleton.joints
    )


def _skin_proposal(surface, skeleton):
    joints = tuple(j.canonical_joint_id for j in skeleton.joints)
    return SkinProposalIR(
        tuple(
            SkinInfluenceProposal(sid, joints[min(i, len(joints) - 1)], 1.0)
            for i, sid in enumerate(("S0", "S1", "S2"))
        ),
        surface.geometry_lineage_hash,
        skeleton.skeleton_lineage_hash,
        "CARRIER_BOUND_SKIN_FIXTURE",
    )


def test_same_atlas_semantics_on_different_carriers_mint_distinct_qualified_lineage():
    surface, proposal = _fixture()
    common = {
        "mechanical_carrier_geometry": "GEOM",
        "candidate_mesh": "MESH",
        "static_mesh_qualification": "STATIC",
    }
    a = qualify_skeleton(
        surface,
        proposal,
        authority_bindings={
            **common,
            "mechanical_carrier_evidence": "CARRIER:A",
            "mechanical_carrier_topology": "TOPO:A",
        },
    )
    b = qualify_skeleton(
        surface,
        proposal,
        authority_bindings={
            **common,
            "mechanical_carrier_evidence": "CARRIER:B",
            "mechanical_carrier_topology": "TOPO:B",
        },
    )

    assert _joint_semantics(a) == _joint_semantics(b)
    assert a.skeleton_lineage_hash != b.skeleton_lineage_hash
    assert (
        a.qualification_report["authority_bindings"][
            "mechanical_carrier_evidence"
        ]
        == "CARRIER:A"
    )
    assert (
        b.qualification_report["authority_bindings"][
            "mechanical_carrier_topology"
        ]
        == "TOPO:B"
    )


def test_same_skin_semantics_on_same_skeleton_but_different_carrier_binding_changes_lineage():
    surface, skeleton_proposal = _fixture()
    skeleton = qualify_skeleton(surface, skeleton_proposal)
    proposal = _skin_proposal(surface, skeleton)

    a = qualify_skin(
        surface,
        skeleton,
        proposal,
        authority_bindings={
            "mechanical_carrier_evidence": "CARRIER:A",
            "mechanical_carrier_topology": "TOPO:A",
        },
    )
    b = qualify_skin(
        surface,
        skeleton,
        proposal,
        authority_bindings={
            "mechanical_carrier_evidence": "CARRIER:B",
            "mechanical_carrier_topology": "TOPO:B",
        },
    )

    assert a.rows == b.rows
    assert a.skin_lineage_hash != b.skin_lineage_hash
    assert (
        a.qualification_report["authority_bindings"][
            "mechanical_carrier_evidence"
        ]
        == "CARRIER:A"
    )


def test_legacy_unbound_qualifier_calls_remain_deterministic():
    surface, proposal = _fixture()
    a = qualify_skeleton(surface, proposal)
    b = qualify_skeleton(surface, proposal)
    assert a == b
    assert "authority_bindings" not in a.qualification_report

    skin_proposal = _skin_proposal(surface, a)
    sa = qualify_skin(surface, a, skin_proposal)
    sb = qualify_skin(surface, a, skin_proposal)
    assert sa == sb
    assert "authority_bindings" not in sa.qualification_report
