from __future__ import annotations

"""CAS-native product.rig V2 executor.

One AXIS V6 causal execution emits both the control-ready RigProgram proposal and
its joint-coupled SkinToken proposal. Compiler only validates/materializes the rig;
MIRA decoding remains product.skin responsibility.
"""

import json
import os
from time import perf_counter
from typing import Any, Callable

from compiler.realsas_compiler_services.platform_worker.model_inputs import materialize_model_releases

RIG_EXECUTOR_CONTRACT_V2 = "RealSaS.ProductRigInference.v2"
AXIS_V6_ARCHITECTURE_ID = "RealSaS.AXIS.RigProgram.TypedCausal.ControlSkin.v6_0"


def _one(request: dict[str, Any], artifact_type: str, schema_version: str) -> dict[str, Any]:
    rows = [dict(r) for r in request.get("input_artifacts") or ()
            if str(r.get("artifact_type") or "") == artifact_type
            and str(r.get("schema_version") or "") == schema_version]
    if len(rows) != 1:
        raise RuntimeError(f"RIG_V2_INPUT_ARTIFACT_CARDINALITY:{artifact_type}:{schema_version}:{len(rows)}")
    return rows[0]


def _json_payload(read_object, ref):
    try:
        value = json.loads(read_object(ref).decode("utf-8"))
    except Exception as exc:
        raise RuntimeError("RIG_V2_TYPED_JSON_DECODE_FAILED") from exc
    if not isinstance(value, dict):
        raise RuntimeError("RIG_V2_TYPED_JSON_OBJECT_REQUIRED")
    return value


def _json_bytes(value: Any) -> bytes:
    payload = value.to_dict() if hasattr(value, "to_dict") else value
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def _seal(value: Any, *, role: str, artifact_type: str, schema_version: str,
          authority_class: str, put_bytes: Callable[[bytes], dict[str, Any]]) -> dict[str, Any]:
    sealed = put_bytes(_json_bytes(value))
    return {
        "role": role, "artifact_type": artifact_type, "schema_version": schema_version,
        "storage_key": sealed["storage_key"], "content_sha256": sealed["content_sha256"],
        "size_bytes": sealed["size_bytes"], "authority_class": authority_class,
    }


def _device():
    import torch
    raw = os.environ.get("REALSAS_INFERENCE_DEVICE", "cpu").strip() or "cpu"
    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("RIG_V2_CUDA_REQUESTED_BUT_UNAVAILABLE")
    return device


def execute_rig_capability_v2(
    request: dict[str, Any], *, read_object: Callable[[dict[str, Any]], bytes],
    put_bytes: Callable[[bytes], dict[str, Any]],
) -> dict[str, Any]:
    metadata = dict(request.get("capability_metadata") or {})
    if metadata.get("executor_contract") != RIG_EXECUTOR_CONTRACT_V2:
        raise RuntimeError("RIG_V2_EXECUTOR_CONTRACT_DRIFT")
    if str(request.get("kind") or "") != "MODULE":
        raise RuntimeError("RIG_V2_CAPABILITY_KIND_DRIFT")
    execution_id = str(request.get("execution_id") or "")
    if not execution_id:
        raise RuntimeError("RIG_V2_EXECUTION_ID_REQUIRED")
    if metadata.get("compiler_parent_reselection_allowed") is not False:
        raise RuntimeError("RIG_V2_COMPILER_PARENT_RESELECTION_POLICY_DRIFT")
    if metadata.get("compiler_control_synthesis_allowed") is not False:
        raise RuntimeError("RIG_V2_COMPILER_CONTROL_SYNTHESIS_POLICY_DRIFT")
    if metadata.get("compiler_constraint_synthesis_allowed") is not False:
        raise RuntimeError("RIG_V2_COMPILER_CONSTRAINT_SYNTHESIS_POLICY_DRIFT")

    from compiler.realsas_compiler_core.artifact_codec_v2 import rigging_surface_from_dict
    from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import canonical_mesh_from_dict_v1
    from compiler.realsas_compiler_core.canonical_rig_ir_v2 import mint_canonical_rig_v2
    from compiler.realsas_compiler_core.rig_program_ir_v1 import qualify_rig_program_v1
    from compiler.realsas_compiler_core.skin_token_ir_v1 import validate_skin_token_proposal_v1
    from models.axis.rig_program_decode_v6 import AXISV6CodecBinding, decode_axis_v6_events
    from models.axis.runtime_checkpoint_v2 import load_axis_runtime_model_v2
    from models.axis.scene_conditioning_v6 import condition_axis_v6_from_rigging_surface

    mesh_ref = _one(request, "RealSaS.CanonicalMeshIR", "v1")
    surface_ref = _one(request, "RealSaS.RiggingSurfaceIR", "v1")
    mesh = canonical_mesh_from_dict_v1(_json_payload(read_object, mesh_ref))
    surface = rigging_surface_from_dict(_json_payload(read_object, surface_ref))
    if surface.geometry_lineage_hash != mesh.source_evidence_binding_hash:
        raise RuntimeError("RIG_V2_FEATURE_EVIDENCE_MESH_BINDING_DRIFT")

    manifest: dict[str, Any] = {"run_id": "capability-" + execution_id}
    started = perf_counter()
    materialized = materialize_model_releases(
        manifest=manifest, models=request.get("model_releases") or (), read_object=read_object
    )
    materialize_seconds = perf_counter() - started
    if len(materialized) != 1 or str(materialized[0].get("family") or "") != "AXIS":
        raise RuntimeError("RIG_V2_MODEL_FAMILY_SET_DRIFT")
    release = materialized[0]
    if str(release.get("architecture_id") or "") != AXIS_V6_ARCHITECTURE_ID:
        raise RuntimeError("RIG_V2_AXIS_ARCHITECTURE_DRIFT")
    checkpoint = dict(release.get("checkpoint") or {})
    if str(checkpoint.get("artifact_type") or "") != "RealSaS.AXISRuntimeCheckpoint":
        raise RuntimeError("RIG_V2_AXIS_CHECKPOINT_TYPE_DRIFT")
    if str(checkpoint.get("schema_version") or "") != "v2":
        raise RuntimeError("RIG_V2_AXIS_CHECKPOINT_SCHEMA_DRIFT")

    device = _device()
    load_started = perf_counter()
    model, runtime = load_axis_runtime_model_v2(str(checkpoint.get("path") or ""), device=device)
    load_seconds = perf_counter() - load_started
    codec_raw = runtime.get("skin_codec_binding")
    if not isinstance(codec_raw, dict):
        codec_raw = dict(codec_raw or {})
    codec = AXISV6CodecBinding(
        architecture_id=str(codec_raw.get("architecture_id") or ""),
        checkpoint_hash=str(codec_raw.get("checkpoint_sha256") or ""),
        codebook_size=int(codec_raw.get("codebook_size") or 0),
        tokens_per_joint=int(codec_raw.get("tokens_per_joint") or 0),
    )
    if not codec.architecture_id or len(codec.checkpoint_hash) != 64:
        raise RuntimeError("RIG_V2_SKIN_CODEC_BINDING_INVALID")

    conditioning = condition_axis_v6_from_rigging_surface(surface, device=device)
    resource_limit = int(metadata.get("maximum_new_tokens") or model.config.max_sequence_length)
    resource_limit = min(resource_limit, model.config.max_sequence_length)
    inference_started = perf_counter()
    events = model.generate_events(
        conditioning.scene_memory,
        tokens_per_joint=codec.tokens_per_joint,
        max_steps=resource_limit,
    )
    inference_seconds = perf_counter() - inference_started
    proposal, skin_tokens = decode_axis_v6_events(
        events,
        config=model.config,
        normalization_center=conditioning.surface_tensor.normalization_center,
        normalization_scale=conditioning.surface_tensor.normalization_scale,
        mesh_lineage_hash=mesh.mesh_lineage_hash,
        feature_evidence_binding_hash=surface.geometry_lineage_hash,
        carrier_evidence_hash=mesh.carrier_evidence_hash,
        carrier_topology_hash=mesh.carrier_topology_hash,
        carrier_geometry_hash=mesh.carrier_geometry_hash,
        model_provenance=str(release.get("model_release_id") or "") + ":" + str(checkpoint.get("content_sha256") or ""),
        codec_binding=codec,
    )
    validate_skin_token_proposal_v1(
        skin_tokens,
        expected_proposal_joint_ids=tuple(j.proposal_joint_id for j in proposal.joints),
        expected_rig_program_proposal_hash=proposal.proposal_lineage_hash,
    )
    qualified = qualify_rig_program_v1(proposal)
    rig = mint_canonical_rig_v2(mesh, qualified_program=qualified)

    outputs = [
        _seal(rig, role="canonical_rig", artifact_type="RealSaS.CanonicalRigIR", schema_version="v2",
              authority_class="COMPILER_CANONICAL_RIG", put_bytes=put_bytes),
        _seal(proposal, role="rig_program_proposal", artifact_type="RealSaS.RigProgramProposalIR", schema_version="v1",
              authority_class="LEARNED_RIG_PROGRAM_PROPOSAL_ONLY", put_bytes=put_bytes),
        _seal(qualified, role="rig_program_qualification", artifact_type="RealSaS.QualifiedRigProgramIR", schema_version="v1",
              authority_class="QUALIFIED_RIG_PROGRAM", put_bytes=put_bytes),
        _seal(skin_tokens, role="skin_token_proposal", artifact_type="RealSaS.SkinTokenProposalIR", schema_version="v1",
              authority_class="LEARNED_SKIN_TOKEN_PROPOSAL_ONLY", put_bytes=put_bytes),
    ]
    report = dict(qualified.qualification_report or {})
    diagnostics = {
        "schema": "RealSaS.ProductRigInferenceDiagnostics.v2",
        "rig_lineage_hash": rig.rig_lineage_hash,
        "deform_structure_hash": rig.deform_structure_hash,
        "control_program_hash": rig.control_program_hash,
        "mesh_lineage_hash": mesh.mesh_lineage_hash,
        "axis_architecture_id": AXIS_V6_ARCHITECTURE_ID,
        "axis_model_release_id": str(release.get("model_release_id") or ""),
        "axis_checkpoint_content_sha256": str(checkpoint.get("content_sha256") or ""),
        "skin_codec_architecture_id": codec.architecture_id,
        "skin_codec_checkpoint_hash": codec.checkpoint_hash,
        "joint_count": len(rig.joints), "control_count": len(rig.controls),
        "chain_count": len(rig.chains), "constraint_count": len(rig.constraints),
        "anchor_count": len(rig.attachment_anchors),
        "skin_token_count": sum(len(r.token_ids) for r in skin_tokens.rows),
        "parent_reselection_performed": report.get("parent_reselection_performed"),
        "control_synthesis_performed": report.get("control_synthesis_performed"),
        "constraint_synthesis_performed": report.get("constraint_synthesis_performed"),
        "constraint_solver_executed": report.get("constraint_solver_executed"),
        "model_materialization_seconds": float(materialize_seconds),
        "model_load_seconds": float(load_seconds),
        "inference_seconds": float(inference_seconds),
        "inference_device": str(device),
        "fit_performed": False,
        "product_revision_pass_claimed": False,
    }
    return {"capability_id": str(request.get("capability_id") or ""), "status": "PASS",
            "outputs": outputs, "diagnostics": diagnostics, "failure": None}


__all__ = ["RIG_EXECUTOR_CONTRACT_V2", "execute_rig_capability_v2"]
