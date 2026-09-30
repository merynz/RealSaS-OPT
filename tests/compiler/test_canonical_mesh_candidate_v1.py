from __future__ import annotations

import pytest

from compiler.realsas_compiler_core.canonical_mesh_candidate_v1 import (
    build_canonical_relation_candidate,
    build_holeless_partitioned_dense_candidate,
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


def test_explicit_dense_face_provenance_forbids_three_edge_clique_minting():
    # Three real dense faces can support AB, AC and BC independently without
    # providing an ABC face. A relation-only 3-clique reconstruction cannot
    # distinguish that case and therefore invents ABC; explicit face provenance must.
    nodes = (
        SurfaceNode("a",(0.0,0.0,0.0),(0,),("src",),("o0",)),
        SurfaceNode("b",(1.0,0.0,0.0),(0,),("src",),("o1",)),
        SurfaceNode("c",(0.2,1.0,0.0),(0,),("src",),("o2",)),
        SurfaceNode("d",(0.2,0.2,1.0),(0,),("src",),("o3",)),
    )
    edge_pairs = (("a","b"),("a","c"),("b","c"),("a","d"),("b","d"),("c","d"))
    relations = tuple(
        SurfaceRelation(f"r:{a}:{b}",a,b,"LOCAL_NEIGHBOR",1.0)
        for a,b in edge_pairs
    )
    surface = RiggingSurfaceIR(nodes, relations, "surface-k4")
    partition = build_structural_partition(surface)
    carrier = build_component_carrier_policy(
        partition=partition,
        decisions=tuple(
            ComponentCarrierDecisionIR(component.component_id, "MESH", ("test",))
            for component in partition.components
        ),
    )

    relation_only = build_canonical_relation_candidate(
        surface, partition, carrier, producer_policy_hash="relation-policy"
    )
    explicit = build_canonical_relation_candidate(
        surface,
        partition,
        carrier,
        producer_policy_hash="dense-face-policy",
        explicit_face_provenance=(
            ("a","b","d"),
            ("a","c","d"),
            ("b","c","d"),
        ),
    )

    def surface_faces(candidate):
        sid_by_vid = {
            str(v.candidate_vertex_id): str(v.support_binding.coefficients[0][0])
            for v in candidate.vertices
        }
        return {
            tuple(sorted(sid_by_vid[str(vid)] for vid in face))
            for face in candidate.faces
        }

    assert ("a","b","c") in surface_faces(relation_only)
    assert ("a","b","c") not in surface_faces(explicit)
    assert surface_faces(explicit) == {
        ("a","b","d"), ("a","c","d"), ("b","c","d")
    }
    assert explicit.metadata["face_provenance_mode"] == "EXACT_COMPACTED_DENSE_FACE_REPLAY"
    assert explicit.metadata["three_clique_face_minting_allowed"] is False
    validate_canonical_mesh_candidate(
        explicit, surface=surface, partition=partition, carrier_policy=carrier
    )


def test_holeless_harmonic_skin_support_is_component_local_simplex_and_deterministic():
    from compiler.realsas_compiler_core.product_authority_v1 import (
        ComponentBoundaryConstraintIR,
    )

    nodes = (
        SurfaceNode("a",(0.0,0.0,0.0),(0,),("src",),("oa",)),
        SurfaceNode("b",(1.0,0.0,0.0),(0,),("src",),("ob",)),
        SurfaceNode("c",(0.3,1.0,0.0),(0,),("src",),("oc",)),
    )
    relations = (
        SurfaceRelation("ab","a","b","LOCAL_NEIGHBOR",1.0),
        SurfaceRelation("bc","b","c","LOCAL_NEIGHBOR",1.0),
        SurfaceRelation("ca","c","a","LOCAL_NEIGHBOR",1.0),
    )
    surface = RiggingSurfaceIR(nodes, relations, "surface-harmonic-seam")
    partition = build_structural_partition(
        surface,
        boundary_overrides=(
            ComponentBoundaryConstraintIR(
                "cut:ac","a","c","SEPARATE",("test-cut",),1.0
            ),
            ComponentBoundaryConstraintIR(
                "cut:bc","b","c","SEPARATE",("test-cut",),1.0
            ),
        ),
    )
    carrier = build_component_carrier_policy(
        partition=partition,
        decisions=tuple(
            ComponentCarrierDecisionIR(component.component_id, "MESH", ("test",))
            for component in partition.components
        ),
    )
    kwargs = dict(
        surface=surface,
        partition=partition,
        carrier_policy=carrier,
        producer_policy_hash="harmonic-test-policy",
        explicit_face_provenance=(("a","b","c"),),
    )
    owner_copy = build_holeless_partitioned_dense_candidate(
        **kwargs,
        mechanical_skin_transfer="OWNER_COPY",
    )
    harmonic = build_holeless_partitioned_dense_candidate(
        **kwargs,
        mechanical_skin_transfer="COMPONENT_HARMONIC_DIRICHLET_V1",
    )
    harmonic_repeat = build_holeless_partitioned_dense_candidate(
        **kwargs,
        mechanical_skin_transfer="COMPONENT_HARMONIC_DIRICHLET_V1",
    )

    owner_by_sid = {
        str(sid): str(component.component_id)
        for component in partition.components
        for sid in component.surface_ids
    }
    seam = [
        v for v in harmonic.vertices
        if v.support_binding.mode == "SEAM_GEOMETRY_INTERPOLATION"
    ]
    assert seam
    assert harmonic.metadata["mechanical_skin_transfer"] == "COMPONENT_HARMONIC_DIRICHLET_V1"
    assert harmonic.candidate_lineage_hash == harmonic_repeat.candidate_lineage_hash
    assert harmonic.candidate_lineage_hash != owner_copy.candidate_lineage_hash

    support_counts = []
    for vertex in seam:
        support = tuple(
            vertex.support_binding.metadata["skin_support_coefficients"]
        )
        support_counts.append(len(support))
        assert abs(sum(float(w) for _, w in support) - 1.0) < 1e-12
        assert all(
            owner_by_sid[str(sid)] == str(vertex.component_id)
            for sid, _ in support
        )
        assert (
            vertex.support_binding.metadata["mechanical_skin_transfer"]
            == "COMPONENT_HARMONIC_DIRICHLET_V1"
        )

    # The two-owner side must no longer collapse every seam vertex to owner-copy.
    assert max(support_counts) >= 2

    owner_seam = [
        v for v in owner_copy.vertices
        if v.support_binding.mode == "SEAM_GEOMETRY_INTERPOLATION"
    ]
    assert all(
        len(tuple(v.support_binding.metadata["skin_support_coefficients"])) == 1
        for v in owner_seam
    )
