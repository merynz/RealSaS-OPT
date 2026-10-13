"""Pure validation of a shared external IRIS payload; never issues qualification."""
from __future__ import annotations
import hashlib
import json
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import load_file_ref

SHARED_PROFILE="SHARED_IRIS_IMAGE_DERIVED_FSTAR_V1"
JOINT_SAMPLING="JOINT_COHORT_GRADIENT_MEAN__KNIGHT_V5_STAGED_C_LINEAGE"
JOINT_UPDATE="MEAN_GRADIENT_OVER_ALL_SUBJECTS_PER_GLOBAL_STEP"
STAGE_IDS=("DIRECT_FSTAR_COLD","C_OBJECTIVE","C_LONG_HORIZON")


def _validate_joint_policy(policy: dict) -> None:
    if (
        policy.get("teacher_inputs_in_forward") is not False
        or int(policy.get("subject_count",0)) < 1
        or policy.get("sampling") != JOINT_SAMPLING
        or policy.get("optimizer_update_semantics") != JOINT_UPDATE
        or int(policy.get("subject_exposures_per_global_step",0)) != int(policy.get("subject_count",0))
        or policy.get("fit8_pass_from_training_loss_allowed") is not False
    ):
        raise QualificationError("SHARED_IRIS_PROFILE_POLICY_INVALID")
    stages=list(policy.get("training_stages") or ())
    if tuple(row.get("id") for row in stages) != STAGE_IDS:
        raise QualificationError("SHARED_IRIS_PROFILE_STAGE_SCHEDULE_INVALID")
    if sum(int(row.get("steps",0)) for row in stages) != int(policy.get("steps",0)):
        raise QualificationError("SHARED_IRIS_PROFILE_STAGE_TOTAL_DRIFT")


def shared_profile_policy(ctx,prereg=None):
    cfg=ctx["run_manifest"].get("iris_fit") or {}
    profile=cfg.get("training_profile")
    if profile is None:
        if prereg is not None and prereg.qualification_policy.get("shared_cohort_required"):
            raise QualificationError("SHARED_IRIS_PROFILE_REQUIRED")
        return None
    if profile!=SHARED_PROFILE:
        raise QualificationError("SHARED_IRIS_PROFILE_UNKNOWN")
    reference=cfg.get("shared_training_policy") or {}
    policy=load_file_ref(reference,expected_schema="RealSaS.SharedIrisTrainingPolicy.v1")
    _validate_joint_policy(policy)
    if prereg is not None:
        q=prereg.qualification_policy
        if (q.get("shared_cohort_required") is not True
                or q.get("shared_training_policy_sha256")!=reference["sha256"]
                or q.get("training_profile")!=profile):
            raise QualificationError("SHARED_IRIS_PROFILE_PREREG_BINDING")
    return {"training_profile":profile,"shared_training_policy_sha256":reference["sha256"],
            "shared_cohort_required":True,"geometry_role":"SIGNED_SURFACE_EVIDENCE_NOT_APPEARANCE_AUTHORITY"}


def validate_shared_subject_input(ctx):
    from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import stage_output_payload
    audit=stage_output_payload(ctx,"03_SOURCE_MECHANICAL_AUDIT","RealSaS.SourceMechanicalAuditIR.v1")
    binding=(audit.get("metadata") or {}).get("iris_observation_binding")
    if not binding or not binding.get("observation_binding"):
        raise QualificationError("SHARED_IRIS_QUALIFIED_OBSERVATION_BINDING_REQUIRED")
    cfg=ctx["run_manifest"]["iris_fit"]
    manifest=load_file_ref(cfg.get("shared_cohort_manifest") or {},expected_schema="RealSaS.SharedIrisCohortShards.v1")
    policy=load_file_ref(cfg.get("shared_training_policy") or {},expected_schema="RealSaS.SharedIrisTrainingPolicy.v1")
    _validate_joint_policy(policy)
    ids=[r["subject_id"] for r in manifest["subjects"]]
    if (len(ids)!=policy["subject_count"] or len(set(ids))!=len(ids)
            or manifest.get("policy_sha256")!=value_hash(policy)):
        raise QualificationError("SHARED_IRIS_MANIFEST_POLICY_BINDING")
    rows=[row for row in manifest["subjects"] if row["subject_id"]==binding["subject_key"]]
    if len(rows)!=1 or rows[0]["observation_binding"]!=binding["observation_binding"]:
        raise QualificationError("SHARED_IRIS_SUBJECT_OBSERVATION_DRIFT")
    return manifest


def value_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def validate_shared_iris_payload(payload, manifest, policy, *, checkpoint_sha256):
    def require(condition, code):
        if not condition:
            raise QualificationError(code)
    _validate_joint_policy(policy)
    require(payload.get("schema")=="RealSaS.SharedIrisFitExecutionPayload.v1","SHARED_IRIS_PAYLOAD_SCHEMA")
    require(payload.get("status")=="TRAINING_COMPLETE__COMPILER_QUALIFICATION_PENDING","SHARED_IRIS_TRAINING_INCOMPLETE")
    require(payload.get("shared_model_count")==1 and payload.get("shared_optimizer_count")==1,"SHARED_IRIS_NOT_SHARED")
    require(not any(payload.get(k,False) for k in ("fit8_pass_claimed","product_pass_claimed","unseen_claimed")),
            "SHARED_IRIS_UNAUTHORIZED_CLAIM")
    ids=[row["subject_id"] for row in manifest["subjects"]]
    require(len(ids)==policy["subject_count"] and len(set(ids))==len(ids),"SHARED_IRIS_SUBJECT_CARDINALITY")
    binding=payload["binding"]
    require(binding["subject_ids"]==ids and binding["manifest_sha256"]==value_hash(manifest)
            and binding["policy_sha256"]==value_hash(policy),"SHARED_IRIS_INPUT_POLICY_BINDING")
    require(binding.get("optimizer_update_semantics")==JOINT_UPDATE,"SHARED_IRIS_BINDING_UPDATE_SEMANTICS")
    require(payload["binding_sha256"]==value_hash(binding),"SHARED_IRIS_BINDING_HASH")
    require(payload["selected_checkpoint_sha256"]==checkpoint_sha256,"SHARED_IRIS_CHECKPOINT_BINDING")
    counts=payload["subject_exposures"]
    require(set(counts)==set(ids) and all(isinstance(x,int) and x>=0 for x in counts.values()),
            "SHARED_IRIS_EXPOSURE_SUBJECTS")
    global_steps=int(policy["steps"])
    require(payload.get("actual_optimizer_steps")==global_steps,"SHARED_IRIS_OPTIMIZER_STEP_COUNT")
    require(all(int(counts[s])==global_steps for s in ids),"SHARED_IRIS_EXPOSURE_IMBALANCE")
    require(sum(counts.values())==global_steps*len(ids),"SHARED_IRIS_TOTAL_EXPOSURE_COUNT")
    require(payload.get("total_subject_exposures")==global_steps*len(ids),"SHARED_IRIS_TOTAL_EXPOSURE_PAYLOAD")
    receipts=list(payload.get("training_stage_receipts") or ())
    require(tuple(r.get("stage_id") for r in receipts)==STAGE_IDS,"SHARED_IRIS_STAGE_RECEIPTS")
    evaluations=payload["per_subject_evaluations"]
    require(bool(evaluations) and evaluations[-1]["step"]==global_steps,"SHARED_IRIS_FINAL_EVALUATION")
    require([row["subject_id"] for row in evaluations[-1]["subjects"]]==ids,"SHARED_IRIS_HIDDEN_FAILED_SUBJECT")
    require(manifest.get("teacher_inputs_in_forward") is False,"SHARED_IRIS_TEACHER_FORWARD_INPUT")
    return {"schema":"RealSaS.SharedIrisExecutionBindingValidation.v1",
            "binding_sha256":payload["binding_sha256"],"subject_ids":ids,
            "global_optimizer_steps":global_steps,
            "total_subject_exposures":global_steps*len(ids),
            "shared_checkpoint_sha256":checkpoint_sha256,
            "scope":"EXTERNAL_EXECUTION_BINDING_ONLY__DOWNSTREAM_COURTS_STILL_REQUIRED"}


def shared_execution_binding(cfg, prereg, lane, checkpoint_sha256):
    reference=cfg.get("shared_cohort")
    required=bool(prereg.qualification_policy.get("shared_cohort_required",False))
    if not reference:
        if required:
            raise QualificationError("SHARED_IRIS_EXECUTION_BINDING_REQUIRED")
        return None
    if lane!="IRIS" or not required:
        raise QualificationError("SHARED_IRIS_UNDECLARED_SCOPE")
    bundle=load_file_ref(reference,expected_schema="RealSaS.SharedIrisExecutionImportBundle.v1")
    if bundle["training_policy"]["sha256"]!=prereg.qualification_policy.get("shared_training_policy_sha256"):
        raise QualificationError("SHARED_IRIS_EXECUTION_POLICY_DRIFT")
    payload=load_file_ref(bundle["execution"],expected_schema="RealSaS.SharedIrisFitExecutionPayload.v1")
    manifest=load_file_ref(bundle["cohort_manifest"],expected_schema="RealSaS.SharedIrisCohortShards.v1")
    policy=load_file_ref(bundle["training_policy"],expected_schema="RealSaS.SharedIrisTrainingPolicy.v1")
    return validate_shared_iris_payload(payload,manifest,policy,checkpoint_sha256=checkpoint_sha256)
