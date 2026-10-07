from __future__ import annotations

"""Carrier-native Stage34-36 mechanics for the AXIS V5.4.1 / MIRA V5.5 line.

The exact Stage19 carrier M is immutable across MIRA prediction, G3/G3B proof,
Stage35 canonical-id minting and Stage36 mesh-skin binding. Any repair that changes
M invalidates W_M and requires a new MIRA attempt; semantic skin transfer is never
used to hide basis drift.

Stage34 is intentionally pre-bind and skin-independent. Stage35 owns post-bind
mechanical-observability qualification of coincident controls using final W_M.
"""

from compiler.realsas_compiler_core.artifact_codec_v2 import qualified_mesh_from_dict
from compiler.realsas_compiler_core.carrier_skin_v1 import qualified_carrier_skin_from_dict
from compiler.realsas_compiler_core.deformation_envelope_derivation_v2 import derive_deformation_envelope_v2
from compiler.realsas_compiler_core.deformation_stress_carrier_v1 import run_g3_carrier_native_v1
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.joint_frames_v2 import derive_joint_frames_post_bind_v2
from compiler.realsas_compiler_core.mechanical_carrier_evidence_v1 import mechanical_carrier_evidence_from_dict
from compiler.realsas_compiler_core.product_mesh_skin_carrier_v1 import bind_carrier_native_mesh_skin_v1
from compiler.realsas_compiler_core.qualified_mesh_v2 import qualify_canonical_mesh_candidate_v2
from compiler.realsas_compiler_core.skin_topology_compatibility_carrier_v1 import run_skin_topology_compatibility_carrier_v1
from compiler.realsas_compiler_core.surface_addressing_v1 import static_mesh_qualification_from_dict
from compiler.realsas_compiler_core.tessa_candidate_bridge_v1 import tessa_candidate_bridge_evidence_from_dict_v1
from compiler.realsas_compiler_core.tessa_geometry_qualification_v1 import tessa_static_carrier_binding_from_dict_v1
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import stage_output_payload
from compiler.realsas_compiler_services.orchestrator.adapters.mesh_legacy_v2 import (
    _artifact_root,_axis_contract,_component_observations,_load_camera_set,_load_candidate_and_policy,
    _load_partition_and_carrier,_load_skeleton,_load_surface,_write_ir,_write_json,
)
from compiler.realsas_compiler_core.mesh.product_coverage_v1 import build_g5_coverage_matrix,g5_coverage_evidence_hash


def _load_carrier(ctx):
    return mechanical_carrier_evidence_from_dict(stage_output_payload(ctx,"19_STATIC_CANONICAL_MESH_QUALIFIED","RealSaS.MechanicalCarrierEvidenceIR.v1"))


def _load_carrier_skin(ctx):
    return qualified_carrier_skin_from_dict(stage_output_payload(ctx,"32_SKIN_QUALIFIED","RealSaS.QualifiedCarrierSkinIR.v1"))


def seal_deformation_capability_envelope(ctx:dict)->dict:
    skeleton=_load_skeleton(ctx); carrier=_load_carrier(ctx)
    cfg=dict(ctx["run_manifest"].get("deformation_envelope") or {})
    if cfg:
        return {"status":"BLOCKED","blockers":["PER_CHARACTER_DEFORMATION_AUTHORING_FORBIDDEN"],"diagnostics":{"unsupported_keys":sorted(cfg)}}
    camera_set=_load_camera_set(ctx)
    axis_payload,envelope=derive_deformation_envelope_v2(skeleton=skeleton,camera_set=camera_set)
    root=_artifact_root(ctx,"34_DEFORMATION_CAPABILITY_ENVELOPE")
    return {"status":"PASS","outputs":[
        _write_json(root/"derived_axis_contract.json",axis_payload,authority_class="QUALIFIED_DERIVED_AXIS_CONTRACT",schema="RealSaS.DerivedAxisContract.v1"),
        _write_ir(root/"deformation_envelope.json",envelope,authority_class="QUALIFIED_DEFORMATION_CAPABILITY_ENVELOPE"),
    ],"diagnostics":{
        "joint_range_count":len(envelope.joint_ranges),"camera_count":len(camera_set.cameras),
        "probe_plan_hash":envelope.probe_plan_hash,"axis_contract_hash":envelope.axis_contract_hash,
        "carrier_evidence_hash":carrier.carrier_evidence_hash,
        "joint_frame_semantics":"PRE_BIND_PROVISIONAL__POST_BIND_STAGE35_REQUIRED",
        "post_bind_qualification_required":True,
        "subject_specific_code_used":False,
    }}


def qualify_canonical_mesh_stage(ctx:dict)->dict:
    surface=_load_surface(ctx); skeleton=_load_skeleton(ctx); skin=_load_carrier_skin(ctx); carrier_evidence=_load_carrier(ctx)
    partition,carrier_policy=_load_partition_and_carrier(ctx)
    candidate,policy=_load_candidate_and_policy(ctx); camera_set=_load_camera_set(ctx); cameras=camera_set.cameras
    from compiler.realsas_compiler_core.artifact_codec_v2 import deformation_envelope_from_dict
    envelope=deformation_envelope_from_dict(stage_output_payload(ctx,"34_DEFORMATION_CAPABILITY_ENVELOPE","RealSaS.DeformationCapabilityEnvelopeIR.v1"))
    axis_payload,axis_hash=_axis_contract(ctx)
    if tuple(camera_set.camera_binding_hashes)!=tuple(envelope.camera_binding_hashes):
        return {"status":"FAIL","blockers":["STAGE35_CAMERA_SET_DRIFT"],"diagnostics":{}}
    if axis_hash!=envelope.axis_contract_hash:
        return {"status":"FAIL","blockers":["STAGE35_AXIS_CONTRACT_DRIFT"],"diagnostics":{}}
    if skin.carrier_evidence_hash!=carrier_evidence.carrier_evidence_hash or skin.carrier_topology_hash!=carrier_evidence.topology_hash or skin.carrier_geometry_hash!=carrier_evidence.geometry_hash:
        raise QualificationError("STAGE35_EXACT_CARRIER_SKIN_BINDING_DRIFT")
    if candidate.candidate_lineage_hash!=carrier_evidence.candidate_mesh_binding_hash:
        raise QualificationError("STAGE35_CARRIER_CANDIDATE_BINDING_DRIFT")

    # Final W_M now exists. Qualify every provisional coincident frame generically.
    post_frames,post_bind_report=derive_joint_frames_post_bind_v2(
        skeleton,carrier_skin=skin,cameras=cameras
    )
    axis_rows={str(row["canonical_joint_id"]):row for row in axis_payload.get("joint_axes",())}
    if set(axis_rows)!=set(post_frames):
        raise QualificationError("STAGE35_POST_BIND_FRAME_JOINT_SET_DRIFT")
    max_axis_drift=0.0
    for jid,frame in post_frames.items():
        R=frame.rotation_matrix
        post_axis=(float(R[0][0]),float(R[1][0]),float(R[2][0]))
        pre_axis=tuple(map(float,axis_rows[jid]["axis_xyz"]))
        max_axis_drift=max(max_axis_drift,max(abs(a-b) for a,b in zip(pre_axis,post_axis)))
    if max_axis_drift>1e-12:
        raise QualificationError(f"STAGE35_POST_BIND_FRAME_DRIFT:{max_axis_drift}")
    post_bind_report_hash=content_sha256(post_bind_report)

    g3=run_g3_carrier_native_v1(carrier_evidence,skeleton=skeleton,skin=skin,envelope=envelope,cameras=cameras,policy=policy)
    compatibility=run_skin_topology_compatibility_carrier_v1(carrier_evidence,skeleton=skeleton,skin=skin,envelope=envelope,cameras=cameras,policy=policy)
    root=_artifact_root(ctx,"35_DYNAMIC_MECHANICAL_MESH_QUALIFIED")
    frame_artifact=_write_json(root/"post_bind_joint_frame_qualification.json",post_bind_report,authority_class="QUALIFIED_POST_BIND_JOINT_FRAME_EVIDENCE",schema=post_bind_report["schema"])
    compatibility_artifact=_write_json(root/"skin_topology_compatibility.json",compatibility,authority_class="DIAGNOSTIC_SKIN_TOPOLOGY_COMPATIBILITY",schema=compatibility["schema"])
    _write_ir(root/"g3_deformation_stress.json",g3,authority_class="DIAGNOSTIC_G3_EVIDENCE")
    if not compatibility["passed"]:
        return {"status":"FAIL","blockers":["G3B_CARRIER_REPAIR_REQUIRES_NEW_MIRA_ATTEMPT"],"diagnostics":{
            "g3_report_hash":g3.report_hash,"skin_topology_compatibility_report_hash":compatibility["report_hash"],
            "unsafe_face_count":compatibility["unsafe_face_count"],"weight_mutation":False,
            "repair_semantics":"NEW_MECHANICAL_CARRIER_GENERATION_INVALIDATES_W_M__RERUN_MIRA_ON_NEW_M",
            "auto_skin_transfer_allowed":False,"compatibility_artifact_sha256":compatibility_artifact["sha256"],
            "post_bind_joint_frame_qualification_hash":post_bind_report_hash,
        }}

    observations,source_foreground_masks,observation_set=_component_observations(ctx,surface=surface,partition=partition,carrier=carrier_policy,cameras=cameras)
    g5_rows=build_g5_coverage_matrix(candidate,surface=surface,partition=partition,carrier_policy=carrier_policy,mesh_policy=policy,observations=observations,cameras=cameras,observation_set=observation_set,source_foreground_masks=source_foreground_masks)
    unknown_rows=tuple({"constraint_id":r.constraint_id,"a_surface_id":r.a_surface_id,"b_surface_id":r.b_surface_id,"decision":r.decision,"classification":"CONSEQUENTIAL_UNTIL_EXPLICITLY_RESOLVED"} for r in partition.boundary_constraints if r.decision=="UNKNOWN")
    unknown_report={"schema":"RealSaS.UnknownBoundaryAnalysis.v1","partition_lineage_hash":partition.partition_lineage_hash,"candidate_lineage_hash":candidate.candidate_lineage_hash,"conservative_rule":"ALL_ADMITTED_UNKNOWN_BOUNDARIES_ARE_CONSEQUENTIAL_V1","rows":unknown_rows,"consequential_unknown_boundary_count":len(unknown_rows)}
    unknown_hash=content_sha256(unknown_report)
    g5_payload={"schema":"RealSaS.G5CoverageEvidence.v1","evidence_hash":g5_coverage_evidence_hash(g5_rows),"rows":g5_rows}
    _write_json(root/"g5_coverage_evidence.json",g5_payload,authority_class="DIAGNOSTIC_G5_EVIDENCE",schema=g5_payload["schema"])
    _write_json(root/"unknown_boundary_analysis.json",unknown_report,authority_class="DIAGNOSTIC_G4_EVIDENCE",schema=unknown_report["schema"])
    blockers=[]
    if not g3.passed: blockers.append("G3_DEFORMATION_STRESS_FAIL")
    failed_g5=[row for row in g5_rows if row["status"]!="PASS"]
    if failed_g5: blockers.append("G5_MULTIVIEW_COMPONENT_COVERAGE_FAIL")
    if unknown_rows: blockers.append("G4_CONSEQUENTIAL_UNKNOWN_BOUNDARY")
    if blockers:
        return {"status":"FAIL","blockers":blockers,"diagnostics":{
            "g3_report_hash":g3.report_hash,"g3_failures":g3.failure_invariants,
            "g5_failed_cell_count":len(failed_g5),"g5_evidence_hash":g5_payload["evidence_hash"],
            "consequential_unknown_boundary_count":len(unknown_rows),"unknown_boundary_analysis_hash":unknown_hash,
            "carrier_native":True,"semantic_skin_transfer_performed":False,
            "post_bind_joint_frame_qualification_hash":post_bind_report_hash,
        }}

    qualification_report={
        "gates":{"G1_SUPPORT_LINEAGE":"PASS","G2_TOPOLOGY":"PASS","G3_DEFORMATION":"PASS","G3B_SKIN_TOPOLOGY_COMPATIBILITY":"PASS","G4_COMPONENT_BOUNDARY":"PASS","G5_MULTIVIEW_COVERAGE":"PASS"},
        "single_aggregate_score_authority":False,"view_component_coverage_matrix_complete":True,
        "consequential_unknown_boundary_count":0,"unknown_boundary_analysis_hash":unknown_hash,
        "g3_envelope_binding_hash":envelope.envelope_lineage_hash,"g3_motion_capability_claimed":False,
        "g3_role":"LOCAL_3D_NUMERICAL_CONDITIONING_ONLY","g3_stress_probe_hash":g3.report_hash,"g3_stress_probe_status":"PASS",
        "skin_topology_compatibility_report_hash":compatibility["report_hash"],"skin_topology_compatibility_status":"PASS","skin_topology_weight_mutation":False,
        "carrier_policy_hash":carrier_policy.carrier_policy_lineage_hash,"g5_evidence_hash":g5_payload["evidence_hash"],"view_component_coverage":g5_rows,
        "mechanical_skin_domain":"EXACT_STAGE19_CARRIER_M","semantic_skin_transfer_performed":False,
        "post_bind_joint_frame_qualification_hash":post_bind_report_hash,
        "post_bind_frame_max_axis_drift":max_axis_drift,
    }
    tessa_evidence={}
    if candidate.producer_id == "RealSaS.TESSALearnedMechanicalCarrierProposal.v1":
        tessa_evidence={
            "bridge_evidence":tessa_candidate_bridge_evidence_from_dict_v1(stage_output_payload(ctx,"18_CANONICAL_MESH_ADDRESSING_BUILD","RealSaS.TESSACandidateBridgeEvidenceIR.v1")),
            "static_mesh":static_mesh_qualification_from_dict(stage_output_payload(ctx,"19_STATIC_CANONICAL_MESH_QUALIFIED","RealSaS.StaticCanonicalMeshQualificationIR.v1")),
            "static_binding":tessa_static_carrier_binding_from_dict_v1(stage_output_payload(ctx,"19_STATIC_CANONICAL_MESH_QUALIFIED","RealSaS.TESSAStaticCarrierBindingIR.v1")),
        }
    mesh=qualify_canonical_mesh_candidate_v2(candidate,surface=surface,partition=partition,carrier_policy=carrier_policy,envelope=envelope,policy=policy,qualification_report=qualification_report,**tessa_evidence)
    return {"status":"PASS","outputs":[
        _write_ir(root/"qualified_mesh.json",mesh,authority_class="QUALIFIED_PRODUCT_GEOMETRY"),
        _write_ir(root/"g3_deformation_stress.json",g3,authority_class="QUALIFIED_G3_EVIDENCE"),
        frame_artifact,
        compatibility_artifact,
        _write_json(root/"g5_coverage_evidence.json",g5_payload,authority_class="QUALIFIED_G5_EVIDENCE",schema=g5_payload["schema"]),
        _write_json(root/"unknown_boundary_analysis.json",unknown_report,authority_class="QUALIFIED_G4_EVIDENCE",schema=unknown_report["schema"]),
    ],"diagnostics":{
        "mesh_lineage_hash":mesh.mesh_lineage_hash,"g3_report_hash":g3.report_hash,
        "skin_topology_compatibility_report_hash":compatibility["report_hash"],"g5_evidence_hash":g5_payload["evidence_hash"],
        "coverage_cell_count":len(g5_rows),"carrier_evidence_hash":carrier_evidence.carrier_evidence_hash,
        "semantic_skin_transfer_performed":False,"post_bind_joint_frame_qualification_hash":post_bind_report_hash,
        "post_bind_frame_max_axis_drift":max_axis_drift,
    }}


def bind_qualified_mesh_skin_stage(ctx:dict)->dict:
    surface=_load_surface(ctx); skeleton=_load_skeleton(ctx); skin=_load_carrier_skin(ctx); carrier_evidence=_load_carrier(ctx)
    partition,carrier_policy=_load_partition_and_carrier(ctx)
    from compiler.realsas_compiler_core.artifact_codec_v2 import deformation_envelope_from_dict
    envelope=deformation_envelope_from_dict(stage_output_payload(ctx,"34_DEFORMATION_CAPABILITY_ENVELOPE","RealSaS.DeformationCapabilityEnvelopeIR.v1"))
    _,policy=_load_candidate_and_policy(ctx)
    mesh=qualified_mesh_from_dict(stage_output_payload(ctx,"35_DYNAMIC_MECHANICAL_MESH_QUALIFIED","RealSaS.QualifiedMeshIR.v1"))
    bound=bind_carrier_native_mesh_skin_v1(carrier=carrier_evidence,skeleton=skeleton,carrier_skin=skin,mesh=mesh,surface=surface,partition=partition,carrier_policy=carrier_policy,envelope=envelope,policy=policy)
    root=_artifact_root(ctx,"36_QUALIFIED_MESH_SKIN_TRANSFER")
    return {"status":"PASS","outputs":[_write_ir(root/"qualified_mesh_skin.json",bound,authority_class="QUALIFIED_PRODUCT_MESH_SKIN")],"diagnostics":{
        "mesh_skin_lineage_hash":bound.mesh_skin_lineage_hash,"mesh_binding_hash":bound.mesh_binding_hash,"skin_binding_hash":bound.skin_binding_hash,
        "row_count":len(bound.rows),"transfer_method":bound.transfer_method,"semantic_skin_transfer_performed":False,
    }}

__all__=["seal_deformation_capability_envelope","qualify_canonical_mesh_stage","bind_qualified_mesh_skin_stage"]
