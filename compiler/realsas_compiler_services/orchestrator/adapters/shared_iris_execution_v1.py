"""Pure validation of a shared external IRIS payload; never issues qualification."""
from __future__ import annotations
import hashlib
import json
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import load_file_ref

def value_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def validate_shared_iris_payload(payload, manifest, policy, *, checkpoint_sha256):
    def require(condition, code):
        if not condition:
            raise QualificationError(code)
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
    require(payload["binding_sha256"]==value_hash(binding),"SHARED_IRIS_BINDING_HASH")
    require(payload["selected_checkpoint_sha256"]==checkpoint_sha256,"SHARED_IRIS_CHECKPOINT_BINDING")
    counts=payload["subject_exposures"]
    require(set(counts)==set(ids) and all(isinstance(x,int) and x>=0 for x in counts.values()),
            "SHARED_IRIS_EXPOSURE_SUBJECTS")
    require(sum(counts.values())==policy["steps"]==payload["actual_optimizer_steps"]
            and max(counts.values())-min(counts.values())<=1,"SHARED_IRIS_EXPOSURE_IMBALANCE")
    evaluations=payload["per_subject_evaluations"]
    require(bool(evaluations) and evaluations[-1]["step"]==policy["steps"],"SHARED_IRIS_FINAL_EVALUATION")
    require([row["subject_id"] for row in evaluations[-1]["subjects"]]==ids,"SHARED_IRIS_HIDDEN_FAILED_SUBJECT")
    require(manifest.get("teacher_inputs_in_forward") is False,"SHARED_IRIS_TEACHER_FORWARD_INPUT")
    return {"schema":"RealSaS.SharedIrisExecutionBindingValidation.v1",
            "binding_sha256":payload["binding_sha256"],"subject_ids":ids,
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
    payload=load_file_ref(bundle["execution"],expected_schema="RealSaS.SharedIrisFitExecutionPayload.v1")
    manifest=load_file_ref(bundle["cohort_manifest"],expected_schema="RealSaS.SharedIrisCohortShards.v1")
    policy=load_file_ref(bundle["training_policy"],expected_schema="RealSaS.SharedIrisTrainingPolicy.v1")
    return validate_shared_iris_payload(payload,manifest,policy,checkpoint_sha256=checkpoint_sha256)
