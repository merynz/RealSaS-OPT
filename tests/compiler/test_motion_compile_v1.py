from __future__ import annotations

from dataclasses import replace
import pytest

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.motion_compile_v1 import build_qualified_motion
from compiler.realsas_compiler_core.motion_source_v1 import (
    build_motion_source_asset, build_motion_source_set, QualifiedMotionSourceSealIR,
    motion_source_seal_hash,
)
from compiler.realsas_compiler_core.product_authority_v1 import (
    DeformationCapabilityEnvelopeIR, JointCapabilityRangeIR, QualifiedPresentationGraphIR,
    deformation_envelope_lineage_hash,
)
from compiler.realsas_compiler_core.canonical_puppet_state_v1 import CanonicalPuppetStateIR
from compiler.realsas_compiler_core.types import QualifiedJoint, QualifiedSkeletonIR, QualificationError


def _authorities():
    skeleton=QualifiedSkeletonIR(
        (QualifiedJoint("root",(0.0,0.0,0.0),None,(),"p0"),),
        "root",{"status":"PASS"},"skeleton-hash",
    )
    envelope=DeformationCapabilityEnvelopeIR(
        skeleton_lineage_hash=skeleton.skeleton_lineage_hash,
        joint_ranges=(JointCapabilityRangeIR("root",-30.0,30.0,1.0,0.8,1.2),),
        camera_binding_hashes=tuple(f"camera-{i}" for i in range(8)),
        allowed_attachment_state_hashes=(),
        axis_contract_hash="axis-hash",
        probe_plan_hash="probe-hash",
        envelope_lineage_hash="",
    )
    envelope=replace(envelope,envelope_lineage_hash=deformation_envelope_lineage_hash(envelope))
    state=CanonicalPuppetStateIR(
        "surface",skeleton.skeleton_lineage_hash,"skin","partition","carrier",
        envelope.envelope_lineage_hash,"policy","mesh","mesh-skin",(),"",
        metadata={},
    )
    state=replace(state,product_state_hash=content_sha256({k:v for k,v in state.to_dict().items() if k!="product_state_hash"}))
    presentation=QualifiedPresentationGraphIR(
        slots=(),attachments=(),view_overlays=(),decisions=(),
        skeleton_binding_hash=skeleton.skeleton_lineage_hash,
        mesh_binding_hash="mesh",partition_binding_hash="partition",
        carrier_policy_binding_hash="carrier",product_state_binding_hash=state.product_state_hash,
        presentation_structure_binding_hash="structure",appearance_set_binding_hash="appearance",
        composition_set_binding_hash="composition",qualification_report={"status":"PASS"},
        presentation_lineage_hash="presentation",
    )
    return skeleton,envelope,state,presentation


def _seal(asset,state):
    source_set=build_motion_source_set((asset,))
    seal=QualifiedMotionSourceSealIR(
        source_set.source_set_hash,state.product_state_hash,"rest-hash",source_set.assets,
        {
            "status":"PASS_SOURCE_IDENTITY_AND_PRECONDITION_ONLY",
            "retargeting_performed":False,
            "motion_compilation_performed":False,
            "motion_quality_claimed":False,
            "stage34_compile_required":True,
            "stage35_dynamic_proof_required":True,
        },"",
    )
    seal=replace(seal,motion_source_seal_hash=motion_source_seal_hash(seal))
    return source_set,seal


def test_source_rig_tracks_fail_closed_without_explicit_retarget_map():
    skeleton,envelope,state,presentation=_authorities()
    payload={
        "schema":"RealSaS.MotionSourceClip.v1",
        "clip_id":"run","clip_kind":"RUN","source_space":"SOURCE_RIG_TRACKS_V1",
        "duration_seconds":1.0,"loop":True,"channel_contract":["ROTATION_DEG"],
        "tracks":[{"source_joint_id":"hips","keyframes":[
            {"time_seconds":0.0,"rotation_deg":0.0},
            {"time_seconds":1.0,"rotation_deg":10.0},
        ]}],
    }
    asset=build_motion_source_asset(
        clip_id="run",clip_kind="RUN",source_kind="EXTERNAL_ARTIST_CLIP_V1",
        source_space="SOURCE_RIG_TRACKS_V1",duration_seconds=1.0,loop=True,
        channel_contract=("ROTATION_DEG",),source_payload=payload,source_ref="file:run.json",
    )
    source_set,seal=_seal(asset,state)
    with pytest.raises(QualificationError,match="MOTION_COMPILE_SOURCE_RIG_RETARGET_MISSING"):
        build_qualified_motion(
            source_set=source_set,source_seal=seal,source_payloads={"run":payload},
            product_state=state,skeleton=skeleton,envelope=envelope,presentation=presentation,
            compiler_config={"root_trajectory_modes":{"run":"IN_PLACE"}},
        )


def test_artist_source_retargets_exactly_and_stage34_never_claims_quality():
    skeleton,envelope,state,presentation=_authorities()
    payload={
        "schema":"RealSaS.MotionSourceClip.v1",
        "clip_id":"run","clip_kind":"RUN","source_space":"SOURCE_RIG_TRACKS_V1",
        "duration_seconds":1.0,"loop":True,"channel_contract":["ROTATION_DEG"],
        "tracks":[{"source_joint_id":"hips","keyframes":[
            {"time_seconds":0.0,"rotation_deg":0.0},
            {"time_seconds":1.0,"rotation_deg":10.0},
        ]}],
    }
    asset=build_motion_source_asset(
        clip_id="run",clip_kind="RUN",source_kind="EXTERNAL_ARTIST_CLIP_V1",
        source_space="SOURCE_RIG_TRACKS_V1",duration_seconds=1.0,loop=True,
        channel_contract=("ROTATION_DEG",),source_payload=payload,source_ref="file:run.json",
    )
    source_set,seal=_seal(asset,state)
    constraints,motion=build_qualified_motion(
        source_set=source_set,source_seal=seal,source_payloads={"run":payload},
        product_state=state,skeleton=skeleton,envelope=envelope,presentation=presentation,
        compiler_config={
            "root_trajectory_modes":{"run":"IN_PLACE"},
            "retarget_maps":{"run":{"hips":"root"}},
            "contacts":{},
        },
    )
    assert dict(constraints.root_trajectory_modes)=={"run":"IN_PLACE"}
    assert motion.clips[0].classification=="ARTIST_SOURCE"
    assert motion.clips[0].tracks[0].source_joint_id=="hips"
    assert motion.clips[0].tracks[0].canonical_joint_id=="root"
    assert motion.qualification_report["motion_quality_claimed"] is False
    assert motion.qualification_report["dynamic_proof_passed"] is False


def test_envelope_is_hard_authority_not_a_soft_hint():
    skeleton,envelope,state,presentation=_authorities()
    payload={"preset_id":"MECHANICAL_SWAY_V1","amplitude_deg":45.0}
    asset=build_motion_source_asset(
        clip_id="probe",clip_kind="IDLE",source_kind="INLINE_PRESET_SPEC_V1",
        source_space="PROCEDURAL_PARAMETER_SPACE_V1",duration_seconds=1.0,loop=True,
        channel_contract=("ROTATION_DEG",),source_payload=payload,source_ref="manifest:inline:probe",
    )
    source_set,seal=_seal(asset,state)
    with pytest.raises(QualificationError,match="MOTION_COMPILE_ROTATION_OUTSIDE_ENVELOPE"):
        build_qualified_motion(
            source_set=source_set,source_seal=seal,source_payloads={"probe":payload},
            product_state=state,skeleton=skeleton,envelope=envelope,presentation=presentation,
            compiler_config={},
        )
