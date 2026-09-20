from __future__ import annotations

from types import SimpleNamespace

from compiler.realsas_compiler_core.canonical_mesh_candidate_v1 import (
    build_canonical_relation_candidate,
)
from compiler.realsas_compiler_core.mechanical_partition_v1 import build_structural_partition
from compiler.realsas_compiler_core.product_authority_v1 import (
    ComponentCarrierDecisionIR,
    build_component_carrier_policy,
)
from compiler.realsas_compiler_core.types import RiggingSurfaceIR, SurfaceNode, SurfaceRelation
from compiler.realsas_compiler_services.orchestrator.adapters.product_mesh_v1 import (
    _relation_parent_quality_report,
)


def _candidate(points):
    nodes=tuple(
        SurfaceNode(f"s{i}",tuple(map(float,p)),(0,),("src",),(f"o{i}",))
        for i,p in enumerate(points)
    )
    rel=(
        SurfaceRelation("r01","s0","s1","LOCAL_NEIGHBOR",1.0),
        SurfaceRelation("r12","s1","s2","LOCAL_NEIGHBOR",1.0),
        SurfaceRelation("r02","s0","s2","LOCAL_NEIGHBOR",1.0),
    )
    surface=RiggingSurfaceIR(nodes,rel,"surface")
    partition=build_structural_partition(surface)
    carrier=build_component_carrier_policy(
        partition=partition,
        decisions=tuple(
            ComponentCarrierDecisionIR(c.component_id,"MESH",("test",))
            for c in partition.components
        ),
    )
    return build_canonical_relation_candidate(
        surface,partition,carrier,producer_policy_hash="policy"
    )


def _policy():
    return SimpleNamespace(
        g3_min_angle_deg=7.5,
        g3_max_aspect_longest_over_min_altitude=16.0,
    )


def test_relation_parent_redteam_detects_no_boundary_split_min_angle_obstruction():
    candidate=_candidate(((0.0,0.0,0.0),(1.0,0.0,0.0),(0.999,0.01,0.0)))
    report=_relation_parent_quality_report(candidate,_policy())
    assert report["face_count"]==1
    assert report["below_min_angle_face_count"]==1
    assert report["policy_violating_face_count"]==1
    assert report["cdt_v1_min_angle_repairability"]=="UNREPAIRABLE_IF_ANY_PARENT_CORNER_IS_BELOW_TARGET"
    assert report["not_raw_marching_cubes_parent_faces"] is True


def test_relation_parent_redteam_accepts_well_conditioned_parent():
    candidate=_candidate(((0.0,0.0,0.0),(1.0,0.0,0.0),(0.5,0.8660254,0.0)))
    report=_relation_parent_quality_report(candidate,_policy())
    assert report["below_min_angle_face_count"]==0
    assert report["above_max_aspect_face_count"]==0
    assert report["policy_violating_face_count"]==0
    assert report["cdt_v1_min_angle_repairability"]=="NO_PARENT_MIN_ANGLE_OBSTRUCTION_DETECTED"
