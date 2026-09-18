from __future__ import annotations

from dataclasses import replace
import json

from compiler.realsas_compiler_core.product_artifact_codec_v1 import (
    camera_projection_from_dict,
    canonical_mesh_candidate_from_dict,
    component_carrier_policy_from_dict,
    mechanical_partition_from_dict,
    mesh_policy_from_dict,
    rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.canonical_mesh_candidate_v1 import build_canonical_relation_candidate
from compiler.realsas_compiler_core.mechanical_partition_v1 import build_structural_partition
from compiler.realsas_compiler_core.playback_full_surface_v3 import CameraProjectionV3
from compiler.realsas_compiler_core.product_authority_v1 import (
    CarrierCoverageThresholdIR,
    ComponentCarrierDecisionIR,
    build_component_carrier_policy,
    build_mesh_qualification_policy,
)
from compiler.realsas_compiler_core.types import RiggingSurfaceIR, SurfaceNode, SurfaceRelation


def _surface():
    nodes=(
        SurfaceNode("s0",(0.0,0.0,0.0),(0,1),("p0",),("o0",),((0,(1.0,2.0)),)),
        SurfaceNode("s1",(1.0,0.0,0.0),(0,1),("p1",),("o1",),((0,(2.0,2.0)),)),
        SurfaceNode("s2",(0.0,1.0,0.0),(0,1),("p2",),("o2",),((0,(1.0,1.0)),)),
    )
    rel=(
        SurfaceRelation("r01","s0","s1","LOCAL",1.0),
        SurfaceRelation("r12","s1","s2","LOCAL",1.0),
        SurfaceRelation("r02","s0","s2","LOCAL",1.0),
    )
    return RiggingSurfaceIR(nodes,rel,"surface-hash")


def test_surface_partition_carrier_candidate_roundtrip_preserves_lineage_fields():
    surface=_surface()
    surface2=rigging_surface_from_dict(surface.to_dict())
    assert surface2.to_dict()==surface.to_dict()

    partition=build_structural_partition(surface)
    partition2=mechanical_partition_from_dict(partition.to_dict())
    assert partition2.to_dict()==partition.to_dict()

    carrier=build_component_carrier_policy(
        partition=partition,
        decisions=tuple(
            ComponentCarrierDecisionIR(c.component_id,"MESH",("evidence",))
            for c in partition.components
        ),
    )
    carrier2=component_carrier_policy_from_dict(carrier.to_dict())
    assert carrier2.to_dict()==carrier.to_dict()

    candidate=build_canonical_relation_candidate(
        surface,partition,carrier,producer_policy_hash="producer-policy"
    )
    candidate2=canonical_mesh_candidate_from_dict(candidate.to_dict())
    assert candidate2.to_dict()==candidate.to_dict()


def test_mesh_policy_roundtrip_keeps_frozen_thresholds():
    policy=build_mesh_qualification_policy(
        g1_max_normal_refinement_ratio=0.125,
        g1_max_tangential_to_normal_ratio=0.25,
        g3_min_angle_deg=7.5,
        g3_max_aspect_longest_over_min_altitude=16.0,
        coverage_thresholds=(
            CarrierCoverageThresholdIR("MESH",0.97,0.995,0.005,0.005),
            CarrierCoverageThresholdIR("PLANAR",0.99,0.995,0.0025,0.0025),
        ),
    )
    assert mesh_policy_from_dict(policy.to_dict()).to_dict()==policy.to_dict()


def test_camera_projection_roundtrip_is_exact():
    camera=CameraProjectionV3(
        "V0",0,(0.0,0.0,-2.0),(1.0,0.0,0.0),(0.0,1.0,0.0),(0.0,0.0,1.0),1.0,1024
    )
    assert camera_projection_from_dict({
        **camera.__dict__,
        "schema_version":camera.schema_version,
    })==camera
