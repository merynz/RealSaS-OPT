from __future__ import annotations

"""Shared exact-hash external fit receipt helpers for IRIS/Geppetto/Arachne stages."""

from dataclasses import replace
import json
from pathlib import Path

from compiler.realsas_compiler_core.preproduct_authority_v1 import (
    ModelCheckpointSealIR, ModelFitExecutionIR, ModelFitPreregistrationIR,
    model_checkpoint_seal_hash, model_fit_execution_hash, model_fit_preregistration_hash,
)
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_services.orchestrator.adapters.product_mesh_v1 import (
    _load_file_ref, _resolved_path, _sha256, _write_ir,
)

_ALLOWED_EXECUTORS={"EXTERNAL_GPU_JOB_V1","LOCAL_PINNED_GPU_JOB_V1"}


def _json_ref(ref:dict,expected_schema:str):
    path=_resolved_path(str(ref.get("path") or ""))
    expected=str(ref.get("sha256") or "")
    payload=_load_file_ref(ref,expected_schema=expected_schema)
    return path,expected,payload


def build_fit_preregistration(ctx:dict,*,lane:str,section_key:str,upstream_bindings:dict[str,str],stage_id:str):
    cfg=dict(ctx["run_manifest"].get(section_key) or {})
    ref=dict(cfg.get("preregistration") or {})
    path,digest,spec=_json_ref(ref,"RealSaS.ModelFitPreregistrationSpec.v1")
    if str(spec.get("lane") or "")!=lane:
        raise QualificationError("MODEL_PREREG_LANE_DRIFT")
    architecture=str(spec.get("architecture_id") or "")
    executor=str(spec.get("executor_kind") or "")
    if not architecture or executor not in _ALLOWED_EXECUTORS:
        raise QualificationError("MODEL_PREREG_ARCHITECTURE_OR_EXECUTOR_INVALID")
    if bool(spec.get("teacher_inference_inputs_allowed",False)):
        raise QualificationError("MODEL_PREREG_TEACHER_INFERENCE_INPUTS_FORBIDDEN")
    source_ref=dict(spec.get("model_source") or {})
    source_path=_load_file_ref(source_ref,json_required=False)
    source_sha=str(source_ref.get("sha256") or "")
    declared=dict(spec.get("upstream_bindings") or {})
    if declared and declared!={str(k):str(v) for k,v in upstream_bindings.items()}:
        raise QualificationError("MODEL_PREREG_UPSTREAM_BINDING_DRIFT")
    expected=tuple(map(str,spec.get("expected_output_contract") or ()))
    if not expected:
        raise QualificationError("MODEL_PREREG_OUTPUT_CONTRACT_EMPTY")
    value=ModelFitPreregistrationIR(
        lane,architecture,executor,int(spec.get("random_seed",0)),str(source_path),source_sha,
        str(path),digest,tuple(sorted((str(k),str(v)) for k,v in upstream_bindings.items())),
        expected,dict(spec.get("qualification_policy") or {}),"",
        metadata={
            "teacher_inference_inputs_allowed":False,
            "external_execution_is_not_product_authority":True,
            "preregistration_document_schema":"RealSaS.ModelFitPreregistrationSpec.v1",
        },
    )
    value=replace(value,preregistration_hash=model_fit_preregistration_hash(value))
    root=ctx["run_root"]/"artifacts"/stage_id
    return value,_write_ir(root/"model_fit_preregistration.json",value,authority_class=f"{lane}_FIT_PREREGISTRATION")


def build_fit_execution(ctx:dict,*,lane:str,section_key:str,prereg:ModelFitPreregistrationIR,stage_id:str,proposal_required:bool):
    cfg=dict(ctx["run_manifest"].get(section_key) or {})
    receipt_path,receipt_sha,receipt=_json_ref(dict(cfg.get("execution_receipt") or {}),"RealSaS.ModelFitExecutionReceipt.v1")
    checkpoint=_load_file_ref(dict(cfg.get("checkpoint") or {}),json_required=False)
    result=_load_file_ref(dict(cfg.get("result") or {}),json_required=False)
    checkpoint_sha=_sha256(checkpoint); result_sha=_sha256(result)
    proposal_path=None; proposal_sha=None
    if proposal_required:
        proposal_ref=dict(cfg.get("proposal") or {})
        proposal_path=_load_file_ref(proposal_ref,json_required=False)
        proposal_sha=_sha256(proposal_path)

    if str(receipt.get("lane") or "")!=lane or str(receipt.get("status") or "")!="PASS":
        raise QualificationError("MODEL_EXECUTION_RECEIPT_STATUS_OR_LANE_INVALID")
    if str(receipt.get("architecture_id") or "")!=prereg.architecture_id:
        raise QualificationError("MODEL_EXECUTION_ARCHITECTURE_DRIFT")
    if str(receipt.get("preregistration_binding_hash") or "")!=prereg.preregistration_hash:
        raise QualificationError("MODEL_EXECUTION_PREREG_BINDING_DRIFT")
    if str(receipt.get("model_source_sha256") or "")!=prereg.model_source_sha256:
        raise QualificationError("MODEL_EXECUTION_SOURCE_DRIFT")
    if str(receipt.get("checkpoint_sha256") or "")!=checkpoint_sha:
        raise QualificationError("MODEL_EXECUTION_CHECKPOINT_DRIFT")
    if str(receipt.get("result_sha256") or "")!=result_sha:
        raise QualificationError("MODEL_EXECUTION_RESULT_DRIFT")
    if proposal_required and str(receipt.get("proposal_sha256") or "")!=proposal_sha:
        raise QualificationError("MODEL_EXECUTION_PROPOSAL_DRIFT")
    if bool(receipt.get("teacher_inference_inputs_used",False)):
        raise QualificationError("MODEL_EXECUTION_TEACHER_INFERENCE_INPUTS_USED")
    declared=dict(receipt.get("upstream_bindings") or {})
    actual=dict(prereg.upstream_bindings)
    if declared and declared!=actual:
        raise QualificationError("MODEL_EXECUTION_UPSTREAM_BINDING_DRIFT")

    execution=ModelFitExecutionIR(
        lane,prereg.architecture_id,prereg.preregistration_hash,prereg.model_source_sha256,
        prereg.upstream_bindings,str(receipt_path),receipt_sha,str(checkpoint),checkpoint_sha,
        str(result),result_sha,None if proposal_path is None else str(proposal_path),proposal_sha,
        dict(prereg.qualification_policy),
        {
            "status":"PASS",
            "executor_kind":prereg.executor_kind,
            "teacher_inference_inputs_used":False,
            "receipt_payload_sha256":receipt_sha,
            "external_fit_executed":True,
            "product_authority_minted":False,
        },"",
        metadata={"receipt_metadata":dict(receipt.get("metadata") or {})},
    )
    execution=replace(execution,execution_hash=model_fit_execution_hash(execution))
    root=ctx["run_root"]/"artifacts"/stage_id
    return execution,_write_ir(root/"model_fit_execution.json",execution,authority_class=f"{lane}_FIT_EXECUTION_RECEIPT")


def seal_model_checkpoint(ctx:dict,*,execution:ModelFitExecutionIR,qualified_output_binding_hash:str|None,stage_id:str):
    path=_resolved_path(execution.checkpoint_path)
    if not path.is_file() or _sha256(path)!=execution.checkpoint_sha256:
        raise QualificationError("MODEL_CHECKPOINT_BYTES_DRIFT")
    result=_resolved_path(execution.result_path)
    if not result.is_file() or _sha256(result)!=execution.result_sha256:
        raise QualificationError("MODEL_CHECKPOINT_RESULT_BYTES_DRIFT")
    value=ModelCheckpointSealIR(
        execution.lane,execution.execution_hash,str(path),execution.checkpoint_sha256,
        execution.result_sha256,qualified_output_binding_hash,"",
        metadata={
            "architecture_id":execution.architecture_id,
            "external_fit_execution_binding_hash":execution.execution_hash,
            "checkpoint_filename_is_not_authority":True,
        },
    )
    value=replace(value,checkpoint_seal_hash=model_checkpoint_seal_hash(value))
    root=ctx["run_root"]/"artifacts"/stage_id
    return value,_write_ir(root/"model_checkpoint_seal.json",value,authority_class=f"{execution.lane}_CHECKPOINT_SEAL")
