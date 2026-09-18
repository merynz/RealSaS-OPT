from __future__ import annotations

from compiler.realsas_compiler_core.canonical_cdt_adapter_v1 import (
    build_canonical_cdt_candidate,
)
from compiler.realsas_compiler_core.mechanical_partition_v1 import build_structural_partition
from compiler.realsas_compiler_core.product_authority_v1 import (
    CarrierCoverageThresholdIR,
    ComponentCarrierDecisionIR,
    build_component_carrier_policy,
    build_mesh_qualification_policy,
    validate_canonical_mesh_candidate,
)
from compiler.realsas_compiler_core.types import RiggingSurfaceIR, SurfaceNode, SurfaceRelation


def _triangle_surface():
    nodes = (
        SurfaceNode("s0",(0.0,0.0,0.0),(0,),("src",),("o0",)),
        SurfaceNode("s1",(1.0,0.0,0.0),(0,),("src",),("o1",)),
        SurfaceNode("s2",(0.5,0.9,0.0),(0,),("src",),("o2",)),
    )
    relations = (
        SurfaceRelation("r01","s0","s1","LOCAL_NEIGHBOR",1.0),
        SurfaceRelation("r12","s1","s2","LOCAL_NEIGHBOR",1.0),
        SurfaceRelation("r02","s0","s2","LOCAL_NEIGHBOR",1.0),
    )
    return RiggingSurfaceIR(nodes,relations,"surface-hash")


def _context():
    surface = _triangle_surface()
    partition = build_structural_partition(surface)
    cid = partition.components[0].component_id
    carrier = build_component_carrier_policy(
        partition=partition,
        decisions=(ComponentCarrierDecisionIR(cid,"MESH",("structural",)),),
    )
    policy = build_mesh_qualification_policy(
        g1_max_normal_refinement_ratio=0.125,
        g1_max_tangential_to_normal_ratio=0.25,
        g3_min_angle_deg=7.5,
        g3_max_aspect_longest_over_min_altitude=16.0,
        coverage_thresholds=(
            CarrierCoverageThresholdIR("MESH",0.97,0.995,0.005,0.005),
            CarrierCoverageThresholdIR("PLANAR",0.99,0.995,0.0025,0.0025),
        ),
    )
    return surface, partition, carrier, policy


def test_cdt_local_triangle_adapter_emits_current_view_independent_candidate():
    surface, partition, carrier, policy = _context()
    candidate = build_canonical_cdt_candidate(
        surface,
        partition,
        carrier,
        policy,
        relation_baseline_policy_hash="relation-baseline-policy",
        max_quality_iterations=0,
    )
    assert candidate.metadata["view_independent"] is True
    assert candidate.metadata["camera_authority_used"] is False
    assert candidate.metadata["source_mesh_used"] is False
    assert candidate.metadata["boundary_split_policy"] == "FAIL_CLOSED"
    assert candidate.metadata["patch_count"] == 1
    assert candidate.faces
    assert candidate.vertices
    validate_canonical_mesh_candidate(
        candidate,
        surface=surface,
        partition=partition,
        carrier_policy=carrier,
    )


def test_cdt_adapter_rebinds_every_output_vertex_to_admitted_surface_support():
    surface, partition, carrier, policy = _context()
    candidate = build_canonical_cdt_candidate(
        surface,
        partition,
        carrier,
        policy,
        relation_baseline_policy_hash="relation-baseline-policy",
        max_quality_iterations=0,
    )
    known = {node.surface_id for node in surface.surface_nodes}
    for vertex in candidate.vertices:
        assert vertex.support_binding.mode in {"IDENTITY_SURFACE_NODE","LOCAL_CONVEX_INTERPOLATION"}
        assert abs(sum(weight for _,weight in vertex.support_binding.coefficients)-1.0) < 1e-9
        assert all(sid in known and weight >= 0.0 for sid,weight in vertex.support_binding.coefficients)


def test_cdt_adapter_is_deterministic_for_same_inputs():
    surface, partition, carrier, policy = _context()
    kwargs=dict(
        relation_baseline_policy_hash="relation-baseline-policy",
        max_quality_iterations=0,
    )
    a=build_canonical_cdt_candidate(surface,partition,carrier,policy,**kwargs)
    b=build_canonical_cdt_candidate(surface,partition,carrier,policy,**kwargs)
    assert a.candidate_lineage_hash == b.candidate_lineage_hash
