from __future__ import annotations

import pytest

from compiler.realsas_compiler_core.canonical_mesh_candidate_v1 import (
    build_canonical_relation_candidate,
)
from compiler.realsas_compiler_core.mechanical_partition_v1 import build_structural_partition
from compiler.realsas_compiler_core.product_authority_v1 import (
    ComponentCarrierDecisionIR,
    build_component_carrier_policy,
    validate_canonical_mesh_candidate,
)
from compiler.realsas_compiler_core.types import (
    QualificationError,
    RiggingSurfaceIR,
    SurfaceNode,
    SurfaceRelation,
)


def _surface():
    nodes = (
        SurfaceNode("s0", (0.0,0.0,0.0), (0,), ("src",), ("o0",)),
        SurfaceNode("s1", (1.0,0.0,0.0), (0,), ("src",), ("o1",)),
        SurfaceNode("s2", (0.5,0.8,0.0), (0,), ("src",), ("o2",)),
        SurfaceNode("s3", (3.0,0.0,0.0), (0,), ("src",), ("o3",)),
        SurfaceNode("s4", (4.0,0.0,0.0), (0,), ("src",), ("o4",)),
        SurfaceNode("s5", (3.5,0.8,0.0), (0,), ("src",), ("o5",)),
    )
    relations = (
        SurfaceRelation("r01","s0","s1","LOCAL_NEIGHBOR",1.0),
        SurfaceRelation("r12","s1","s2","LOCAL_NEIGHBOR",1.0),
        SurfaceRelation("r02","s0","s2","LOCAL_NEIGHBOR",1.0),
        SurfaceRelation("r34","s3","s4","LOCAL_NEIGHBOR",1.0),
        SurfaceRelation("r45","s4","s5","LOCAL_NEIGHBOR",1.0),
        SurfaceRelation("r35","s3","s5","LOCAL_NEIGHBOR",1.0),
        SurfaceRelation("bridge","s2","s3","UNKNOWN_BRIDGE",1.0,{"crosses_unknown":True}),
    )
    return RiggingSurfaceIR(nodes, relations, "surface-hash")


def _context():
    surface = _surface()
    partition = build_structural_partition(surface)
    # Override the ambiguous bridge into a real structural separation for this
    # baseline fixture by rebuilding through explicit evidence.
    from compiler.realsas_compiler_core.product_authority_v1 import ComponentBoundaryConstraintIR
    partition = build_structural_partition(
        surface,
        boundary_overrides=(
            ComponentBoundaryConstraintIR(
                "cut:s2:s3","s2","s3","SEPARATE",("qualified-boundary-evidence",),1.0
            ),
        ),
    )
    component_ids = sorted(component.component_id for component in partition.components)
    carrier = build_component_carrier_policy(
        partition=partition,
        decisions=tuple(
            ComponentCarrierDecisionIR(cid, "MESH", ("test-evidence",))
            for cid in component_ids
        ),
    )
    return surface, partition, carrier


def test_relation_baseline_is_view_independent_identity_support_only():
    surface, partition, carrier = _context()
    candidate = build_canonical_relation_candidate(
        surface, partition, carrier, producer_policy_hash="producer-policy"
    )
    assert candidate.metadata["view_independent"] is True
    assert candidate.metadata["camera_authority_used"] is False
    assert candidate.metadata["generated_vertex_count"] == 0
    assert len(candidate.faces) == 2
    assert len(candidate.vertices) == 6
    assert all(v.support_binding.mode == "IDENTITY_SURFACE_NODE" for v in candidate.vertices)
    validate_canonical_mesh_candidate(
        candidate, surface=surface, partition=partition, carrier_policy=carrier
    )


def test_candidate_never_emits_cross_component_face():
    surface, partition, carrier = _context()
    candidate = build_canonical_relation_candidate(
        surface, partition, carrier, producer_policy_hash="producer-policy"
    )
    owner = {
        v.candidate_vertex_id: v.component_id
        for v in candidate.vertices
    }
    assert all(len({owner[vid] for vid in face}) == 1 for face in candidate.faces)


def test_no_local_relation_triangle_fails_closed_instead_of_inventing_topology():
    nodes = (
        SurfaceNode("a",(0.0,0.0,0.0),(0,),("src",),("o0",)),
        SurfaceNode("b",(1.0,0.0,0.0),(0,),("src",),("o1",)),
        SurfaceNode("c",(0.0,1.0,0.0),(0,),("src",),("o2",)),
    )
    surface = RiggingSurfaceIR(nodes, (), "surface-empty-rel")
    partition = build_structural_partition(surface)
    carrier = build_component_carrier_policy(
        partition=partition,
        decisions=tuple(
            ComponentCarrierDecisionIR(component.component_id, "MESH", ("test",))
            for component in partition.components
        ),
    )
    with pytest.raises(QualificationError, match="NO_FACE"):
        build_canonical_relation_candidate(
            surface, partition, carrier, producer_policy_hash="producer-policy"
        )
