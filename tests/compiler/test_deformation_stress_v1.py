from __future__ import annotations

from dataclasses import replace

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.mesh.deformation_stress_v1 import (
    expected_g3_probe_plan_hash,
    run_g3_deformation_stress,
)
from compiler.realsas_compiler_core.product_authority_v1 import (
    CanonicalMeshCandidateIR,
    CanonicalMeshVertexCandidateIR,
    CarrierCoverageThresholdIR,
    DeformationCapabilityEnvelopeIR,
    JointCapabilityRangeIR,
    build_mesh_qualification_policy,
    canonical_mesh_candidate_lineage_hash,
    deformation_envelope_lineage_hash,
)
from compiler.realsas_compiler_core.types import (
    QualifiedJoint,
    QualifiedSkeletonIR,
    QualifiedSkinIR,
    QualifiedSkinRow,
    RiggingSurfaceIR,
    SurfaceNode,
    SurfaceSupportBinding,
)


def _fixture():
    surface=RiggingSurfaceIR(
        (
            SurfaceNode("s0",(0.0,0.0,0.0),(0,),("src",),("o0",)),
            SurfaceNode("s1",(1.0,0.0,0.0),(0,),("src",),("o1",)),
            SurfaceNode("s2",(0.0,1.0,0.0),(0,),("src",),("o2",)),
        ),
        (),
        "surface-hash",
    )
    skeleton=QualifiedSkeletonIR(
        (
            QualifiedJoint("j0",(0.0,0.0,0.0),None,("s0","s2"),"p0"),
            QualifiedJoint("j1",(1.0,0.0,0.0),"j0",("s1",),"p1"),
        ),
        "j0",
        {"status":"PASS"},
        "skeleton-hash",
    )
    skin=QualifiedSkinIR(
        (
            QualifiedSkinRow("s0",(("j0",1.0),),0.0,0.0),
            QualifiedSkinRow("s1",(("j1",1.0),),0.0,0.0),
            QualifiedSkinRow("s2",(("j0",0.5),("j1",0.5)),0.0,0.0),
        ),
        surface.geometry_lineage_hash,
        skeleton.skeleton_lineage_hash,
        {"status":"PASS"},
        "skin-hash",
    )
    vertices=tuple(
        CanonicalMeshVertexCandidateIR(
            f"v{i}",
            SurfaceSupportBinding("IDENTITY_SURFACE_NODE",((f"s{i}",1.0),)),
            "c0",
            surface.surface_nodes[i].P,
        )
        for i in range(3)
    )
    candidate=CanonicalMeshCandidateIR(
        vertices,(("v0","v1","v2"),),(("v0","v1"),("v1","v2"),("v0","v2")),
        surface.geometry_lineage_hash,"partition-hash","carrier-hash",
        "test-producer","producer-policy","",
    )
    candidate=replace(candidate,candidate_lineage_hash=canonical_mesh_candidate_lineage_hash(candidate))
    axis_contract={
        "schema":"RealSaS.TestAxisContract.v1",
        "status":"FROZEN",
        "joint_axes":[
            {"canonical_joint_id":"j0","axis_xyz":[0,0,1],"legacy_scalar_to_semantic_sign":1.0,"role":"root"},
            {"canonical_joint_id":"j1","axis_xyz":[0,0,1],"legacy_scalar_to_semantic_sign":1.0,"role":"child"},
        ],
    }
    from compiler.realsas_compiler_core.motion_3d_adapter_v1 import parse_axis_contract_v1
    _,axis_hash=parse_axis_contract_v1(axis_contract)
    ranges=(
        JointCapabilityRangeIR("j0",0.0,0.0),
        JointCapabilityRangeIR("j1",-45.0,45.0),
    )
    plan_hash=expected_g3_probe_plan_hash(
        skeleton_lineage_hash=skeleton.skeleton_lineage_hash,
        axis_contract_hash=axis_hash,
        joint_ranges=ranges,
        camera_binding_hashes=tuple(f"cam{i}" for i in range(8)),
        allowed_attachment_state_hashes=(),
    )
    envelope=DeformationCapabilityEnvelopeIR(
        skeleton.skeleton_lineage_hash,
        ranges,
        tuple(f"cam{i}" for i in range(8)),
        (),
        axis_hash,
        plan_hash,
        "",
    )
    envelope=replace(envelope,envelope_lineage_hash=deformation_envelope_lineage_hash(envelope))
    return surface,skeleton,skin,candidate,axis_contract,envelope


def _policy(**kwargs):
    return build_mesh_qualification_policy(
        g1_max_normal_refinement_ratio=0.125,
        g1_max_tangential_to_normal_ratio=0.25,
        g3_min_angle_deg=7.5,
        g3_max_aspect_longest_over_min_altitude=16.0,
        coverage_thresholds=(
            CarrierCoverageThresholdIR("MESH",0.97,0.995,0.005,0.005),
            CarrierCoverageThresholdIR("PLANAR",0.99,0.995,0.0025,0.0025),
        ),
        **kwargs,
    )


def test_g3_rotation_envelope_probe_is_exactly_hash_bound_and_finite():
    surface,skeleton,skin,candidate,axis,envelope=_fixture()
    report=run_g3_deformation_stress(
        candidate,
        surface=surface,
        skeleton=skeleton,
        skin=skin,
        envelope=envelope,
        axis_contract=axis,
        policy=_policy(),
    )
    assert report.probe_plan_hash == envelope.probe_plan_hash
    assert report.probe_count >= 3
    assert report.face_count == 1
    assert report.minimum_area_ratio > 0.0
    assert report.maximum_condition_number >= 1.0
    assert len(report.report_hash) == 64


def test_g3_conditioning_policy_can_fail_a_distorted_lbs_probe_without_changing_motion_semantics():
    surface,skeleton,skin,candidate,axis,envelope=_fixture()
    report=run_g3_deformation_stress(
        candidate,
        surface=surface,
        skeleton=skeleton,
        skin=skin,
        envelope=envelope,
        axis_contract=axis,
        policy=_policy(g3_max_dynamic_condition_number=1.01),
    )
    assert report.passed is False
    assert "DYNAMIC_CONDITION_NUMBER_ABOVE_MAX" in report.failure_invariants


def test_g3_nondefault_translation_scale_fails_closed_until_semantics_exist():
    surface,skeleton,skin,candidate,axis,envelope=_fixture()
    ranges=list(envelope.joint_ranges)
    ranges[1]=replace(ranges[1],translation_radius=0.1)
    plan=expected_g3_probe_plan_hash(
        skeleton_lineage_hash=envelope.skeleton_lineage_hash,
        axis_contract_hash=envelope.axis_contract_hash,
        joint_ranges=tuple(ranges),
        camera_binding_hashes=envelope.camera_binding_hashes,
        allowed_attachment_state_hashes=envelope.allowed_attachment_state_hashes,
    )
    bad=replace(envelope,joint_ranges=tuple(ranges),probe_plan_hash=plan,envelope_lineage_hash="")
    bad=replace(bad,envelope_lineage_hash=deformation_envelope_lineage_hash(bad))
    import pytest
    from compiler.realsas_compiler_core.types import QualificationError
    with pytest.raises(QualificationError,match="TRANSLATION_SCALE_STRESS_NOT_IMPLEMENTED"):
        run_g3_deformation_stress(
            candidate,surface=surface,skeleton=skeleton,skin=skin,
            envelope=bad,axis_contract=axis,policy=_policy(),
        )
