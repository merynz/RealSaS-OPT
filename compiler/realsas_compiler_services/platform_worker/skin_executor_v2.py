from __future__ import annotations

"""CAS-native product.skin V2 executor.

Consumes AXIS-generated SkinTokenProposal + CanonicalRig.v2. It does not rerun
AXIS and does not search token space. The exact bound MIRA codec decodes tokens to
dense carrier weights, then the existing bounded simplex court qualifies them.
"""

import json
import os
from time import perf_counter
from typing import Any, Callable

from compiler.realsas_compiler_services.platform_worker.model_inputs import materialize_model_releases

SKIN_EXECUTOR_CONTRACT_V2 = "RealSaS.ProductSkinInference.v2"
MIRA_V6_ARCHITECTURE_ID = "RealSaS.MIRA.CarrierNativeFSQSkinCodec.v6_0"


def _one(request: dict[str, Any], artifact_type: str, schema_version: str) -> dict[str, Any]:
    rows = [dict(r) for r in request.get("input_artifacts") or ()
            if str(r.get("artifact_type") or "") == artifact_type
            and str(r.get("schema_version") or "") == schema_version]
    if len(rows) != 1:
        raise RuntimeError(f"SKIN_V2_INPUT_ARTIFACT_CARDINALITY:{artifact_type}:{schema_version}:{len(rows)}")
    return rows[0]


def _json_payload(read_object, ref):
    try:
        value = json.loads(read_object(ref).decode("utf-8"))
    except Exception as exc:
        raise RuntimeError("SKIN_V2_TYPED_JSON_DECODE_FAILED") from exc
    if not isinstance(value, dict):
        raise RuntimeError("SKIN_V2_TYPED_JSON_OBJECT_REQUIRED")
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
        raise RuntimeError("SKIN_V2_CUDA_REQUESTED_BUT_UNAVAILABLE")
    return device


def _qualification_policy(metadata: dict[str, Any]) -> dict[str, Any]:
    raw = dict(metadata.get("carrier_skin_qualification") or {})
    required = {"max_simplex_repair_l1", "max_total_correction_l1", "negative_tolerance", "max_influences"}
    if set(raw) != required:
        raise RuntimeError("SKIN_V2_QUALIFICATION_POLICY_DRIFT")
    out = {
        "max_simplex_repair_l1": float(raw["max_simplex_repair_l1"]),
        "max_total_correction_l1": float(raw["max_total_correction_l1"]),
        "negative_tolerance": float(raw["negative_tolerance"]),
        "max_influences": int(raw["max_influences"]),
    }
    if min(out.values()) < 0 or out["max_influences"] < 1:
        raise RuntimeError("SKIN_V2_QUALIFICATION_POLICY_INVALID")
    return out


def _skin_token_from_dict(payload: dict[str, Any]):
    from compiler.realsas_compiler_core.skin_token_ir_v1 import JointSkinTokenRowIR, SkinTokenProposalIR
    schema = str(payload.get("schema_version") or payload.get("schema") or "")
    if schema != "RealSaS.SkinTokenProposalIR.v1":
        raise RuntimeError("SKIN_V2_TOKEN_PROPOSAL_SCHEMA_DRIFT")
    return SkinTokenProposalIR(
        rows=tuple(JointSkinTokenRowIR(
            proposal_joint_id=str(r["proposal_joint_id"]),
            token_ids=tuple(map(int, r.get("token_ids") or ())),
            metadata=dict(r.get("metadata") or {}),
        ) for r in payload.get("rows") or ()),
        rig_program_proposal_hash=str(payload["rig_program_proposal_hash"]),
        carrier_evidence_hash=str(payload["carrier_evidence_hash"]),
        carrier_topology_hash=str(payload["carrier_topology_hash"]),
        carrier_geometry_hash=str(payload["carrier_geometry_hash"]),
        codec_architecture_id=str(payload["codec_architecture_id"]),
        codec_checkpoint_hash=str(payload["codec_checkpoint_hash"]),
        codebook_size=int(payload["codebook_size"]),
        tokens_per_joint=int(payload["tokens_per_joint"]),
        model_provenance=str(payload.get("model_provenance") or ""),
        proposal_lineage_hash=str(payload["proposal_lineage_hash"]),
        schema_version=schema,
        metadata=dict(payload.get("metadata") or {}),
    )


def execute_skin_capability_v2(
    request: dict[str, Any], *, read_object: Callable[[dict[str, Any]], bytes],
    put_bytes: Callable[[bytes], dict[str, Any]],
) -> dict[str, Any]:
    metadata = dict(request.get("capability_metadata") or {})
    if metadata.get("executor_contract") != SKIN_EXECUTOR_CONTRACT_V2:
        raise RuntimeError("SKIN_V2_EXECUTOR_CONTRACT_DRIFT")
    if str(request.get("kind") or "") != "MODULE":
        raise RuntimeError("SKIN_V2_CAPABILITY_KIND_DRIFT")
    execution_id = str(request.get("execution_id") or "")
    if not execution_id:
        raise RuntimeError("SKIN_V2_EXECUTION_ID_REQUIRED")
    if metadata.get("compiler_token_search_allowed") is not False or metadata.get("compiler_token_repair_allowed") is not False:
        raise RuntimeError("SKIN_V2_COMPILER_TOKEN_AUTHORITY_DRIFT")

    import torch
    from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import canonical_mesh_from_dict_v1
    from compiler.realsas_compiler_core.canonical_rig_ir_v2 import canonical_rig_from_dict_v2
    from compiler.realsas_compiler_core.canonical_skin_ir_v2 import mint_canonical_skin_v2
    from compiler.realsas_compiler_core.carrier_skin_v1 import qualify_carrier_skin_v1
    from compiler.realsas_compiler_core.skin_token_ir_v1 import validate_skin_token_proposal_v1
    from models.mira.conditioning_v6 import (
        MIRA_V6_JOINT_FEATURE_DIM, MIRA_V6_VERTEX_FEATURE_DIM,
        build_mira_conditioning_v6, carrier_skin_proposal_from_decoded_v6,
    )
    from models.mira.runtime_checkpoint_v2 import load_mira_runtime_model_v2

    mesh_ref = _one(request, "RealSaS.CanonicalMeshIR", "v1")
    rig_ref = _one(request, "RealSaS.CanonicalRigIR", "v2")
    token_ref = _one(request, "RealSaS.SkinTokenProposalIR", "v1")
    mesh = canonical_mesh_from_dict_v1(_json_payload(read_object, mesh_ref))
    rig = canonical_rig_from_dict_v2(_json_payload(read_object, rig_ref))
    token_proposal = _skin_token_from_dict(_json_payload(read_object, token_ref))
    if rig.mesh_lineage_hash != mesh.mesh_lineage_hash:
        raise RuntimeError("SKIN_V2_RIG_MESH_PARENT_DRIFT")
    expected_proposal_ids = tuple(str(j.source_proposal_joint_id) for j in rig.joints)
    validate_skin_token_proposal_v1(
        token_proposal,
        expected_proposal_joint_ids=expected_proposal_ids,
        expected_rig_program_proposal_hash=rig.proposal_evidence_hash,
    )
    if token_proposal.carrier_evidence_hash != mesh.carrier_evidence_hash:
        raise RuntimeError("SKIN_V2_TOKEN_CARRIER_EVIDENCE_DRIFT")
    if token_proposal.carrier_topology_hash != mesh.carrier_topology_hash:
        raise RuntimeError("SKIN_V2_TOKEN_CARRIER_TOPOLOGY_DRIFT")
    if token_proposal.carrier_geometry_hash != mesh.carrier_geometry_hash:
        raise RuntimeError("SKIN_V2_TOKEN_CARRIER_GEOMETRY_DRIFT")

    manifest: dict[str, Any] = {"run_id": "capability-" + execution_id}
    started = perf_counter()
    materialized = materialize_model_releases(
        manifest=manifest, models=request.get("model_releases") or (), read_object=read_object
    )
    materialize_seconds = perf_counter() - started
    if len(materialized) != 1 or str(materialized[0].get("family") or "") != "MIRA":
        raise RuntimeError("SKIN_V2_MODEL_FAMILY_SET_DRIFT")
    release = materialized[0]
    if str(release.get("architecture_id") or "") != MIRA_V6_ARCHITECTURE_ID:
        raise RuntimeError("SKIN_V2_MIRA_ARCHITECTURE_DRIFT")
    checkpoint = dict(release.get("checkpoint") or {})
    if str(checkpoint.get("artifact_type") or "") != "RealSaS.MIRARuntimeCheckpoint":
        raise RuntimeError("SKIN_V2_MIRA_CHECKPOINT_TYPE_DRIFT")
    if str(checkpoint.get("schema_version") or "") != "v2":
        raise RuntimeError("SKIN_V2_MIRA_CHECKPOINT_SCHEMA_DRIFT")
    checkpoint_hash = str(checkpoint.get("content_sha256") or "")
    if token_proposal.codec_architecture_id != MIRA_V6_ARCHITECTURE_ID:
        raise RuntimeError("SKIN_V2_TOKEN_CODEC_ARCHITECTURE_DRIFT")
    if token_proposal.codec_checkpoint_hash != checkpoint_hash:
        raise RuntimeError("SKIN_V2_TOKEN_CODEC_CHECKPOINT_DRIFT")

    device = _device()
    load_started = perf_counter()
    codec, runtime = load_mira_runtime_model_v2(str(checkpoint.get("path") or ""), device=device)
    load_seconds = perf_counter() - load_started
    if codec.config.vertex_feature_dim != MIRA_V6_VERTEX_FEATURE_DIM:
        raise RuntimeError("SKIN_V2_VERTEX_FEATURE_DIM_DRIFT")
    if codec.config.joint_feature_dim != MIRA_V6_JOINT_FEATURE_DIM:
        raise RuntimeError("SKIN_V2_JOINT_FEATURE_DIM_DRIFT")
    if codec.config.vocab_size != token_proposal.codebook_size:
        raise RuntimeError("SKIN_V2_TOKEN_CODEBOOK_DRIFT")
    if codec.config.tokens_per_joint != token_proposal.tokens_per_joint:
        raise RuntimeError("SKIN_V2_TOKEN_COUNT_DRIFT")

    conditioning = build_mira_conditioning_v6(mesh, rig, device=device)
    token_by_proposal = {str(r.proposal_joint_id): tuple(map(int, r.token_ids)) for r in token_proposal.rows}
    if len(token_by_proposal) != len(token_proposal.rows):
        raise RuntimeError("SKIN_V2_DUPLICATE_TOKEN_JOINT")
    token_tensor = torch.tensor(
        [[token_by_proposal[str(j.source_proposal_joint_id)] for j in rig.joints]],
        device=device, dtype=torch.long,
    )
    inference_started = perf_counter()
    with torch.no_grad():
        weights = codec.decode_tokens(token_tensor, conditioning.vertex_features, conditioning.joint_features)
    inference_seconds = perf_counter() - inference_started

    provenance = str(release.get("model_release_id") or "") + ":" + checkpoint_hash
    dense_proposal = carrier_skin_proposal_from_decoded_v6(weights, conditioning, model_provenance=provenance)
    policy = _qualification_policy(metadata)
    qualified = qualify_carrier_skin_v1(
        conditioning.carrier, conditioning.skeleton, dense_proposal,
        max_simplex_repair_l1=policy["max_simplex_repair_l1"],
        max_total_correction_l1=policy["max_total_correction_l1"],
        negative_tolerance=policy["negative_tolerance"],
        max_influences=policy["max_influences"],
    )
    skin = mint_canonical_skin_v2(
        mesh, rig,
        qualified_skin=qualified,
        skin_token_proposal_hash=token_proposal.proposal_lineage_hash,
        skin_codec_architecture_id=MIRA_V6_ARCHITECTURE_ID,
        skin_codec_checkpoint_hash=checkpoint_hash,
    )

    outputs = [
        _seal(skin, role="canonical_skin", artifact_type="RealSaS.CanonicalSkinIR", schema_version="v2",
              authority_class="COMPILER_CANONICAL_SKIN", put_bytes=put_bytes),
        _seal(dense_proposal, role="decoded_skin_influence_proposal", artifact_type="RealSaS.CarrierSkinProposalIR", schema_version="v1",
              authority_class="LEARNED_SKIN_INFLUENCE_PROPOSAL_ONLY", put_bytes=put_bytes),
        _seal(qualified, role="carrier_skin_qualification", artifact_type="RealSaS.QualifiedCarrierSkinIR", schema_version="v1",
              authority_class="QUALIFIED_CARRIER_SKIN", put_bytes=put_bytes),
    ]
    report = dict(qualified.qualification_report or {})
    diagnostics = {
        "schema": "RealSaS.ProductSkinInferenceDiagnostics.v2",
        "skin_lineage_hash": skin.skin_lineage_hash,
        "mesh_lineage_hash": mesh.mesh_lineage_hash,
        "deform_structure_hash": rig.deform_structure_hash,
        "control_program_hash_is_skin_parent": False,
        "skin_token_proposal_hash": token_proposal.proposal_lineage_hash,
        "mira_architecture_id": MIRA_V6_ARCHITECTURE_ID,
        "mira_model_release_id": str(release.get("model_release_id") or ""),
        "mira_checkpoint_content_sha256": checkpoint_hash,
        "carrier_vertex_count": len(skin.rows), "joint_count": len(rig.joints),
        "simplex_max_abs_residual_before": report.get("simplex_max_abs_residual_before"),
        "max_row_correction_l1": report.get("max_row_correction_l1"),
        "total_correction_l1": report.get("total_correction_l1"),
        "model_materialization_seconds": float(materialize_seconds),
        "model_load_seconds": float(load_seconds),
        "inference_seconds": float(inference_seconds), "inference_device": str(device),
        "axis_rerun_performed": False,
        "compiler_token_search_performed": False,
        "compiler_token_repair_performed": False,
        "fit_performed": False, "product_revision_pass_claimed": False,
    }
    return {"capability_id": str(request.get("capability_id") or ""), "status": "PASS",
            "outputs": outputs, "diagnostics": diagnostics, "failure": None}


__all__ = ["SKIN_EXECUTOR_CONTRACT_V2", "execute_skin_capability_v2"]
