from __future__ import annotations

"""CAS-native executor for the stable product.skin role.

Current implementation is hybrid: MIRA V5.5 predicts W_M directly on the exact
canonical Mesh carrier using replaceable RiggingSurface/material-support evidence.
Compiler validates carrier/Rig bindings and bounded simplex legality, then mints
CanonicalSkinIR. No surface-domain Skin authority or post-hoc Skin transport is
permitted.
"""

import json
import os
from time import perf_counter
from typing import Any, Callable

from compiler.realsas_compiler_services.platform_worker.model_inputs import (
    materialize_model_releases,
)

SKIN_EXECUTOR_CONTRACT_V1 = "RealSaS.ProductSkinInference.v1"
MIRA_V55_ARCHITECTURE_ID = "RealSaS.MIRA.DirectCarrierSimplex.Axis541.v5_5"


def _one(
    request: dict[str, Any], artifact_type: str, schema_version: str
) -> dict[str, Any]:
    matches = [
        dict(ref)
        for ref in request.get("input_artifacts") or ()
        if str(ref.get("artifact_type") or "") == artifact_type
        and str(ref.get("schema_version") or "") == schema_version
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"SKIN_INPUT_ARTIFACT_CARDINALITY:{artifact_type}:{schema_version}:{len(matches)}"
        )
    return matches[0]


def _json_payload(read_object, ref: dict[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(read_object(ref).decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(
            "SKIN_TYPED_JSON_DECODE_FAILED:" + str(ref.get("artifact_type") or "")
        ) from exc
    if not isinstance(value, dict):
        raise RuntimeError("SKIN_TYPED_JSON_OBJECT_REQUIRED")
    return value


def _json_bytes(value: Any) -> bytes:
    payload = value.to_dict() if hasattr(value, "to_dict") else value
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _device_from_environment():
    import torch

    raw = os.environ.get("REALSAS_INFERENCE_DEVICE", "cpu").strip() or "cpu"
    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SKIN_CUDA_REQUESTED_BUT_UNAVAILABLE")
    return device


def _seal(
    value: Any,
    *,
    role: str,
    artifact_type: str,
    schema_version: str,
    authority_class: str,
    put_bytes: Callable[[bytes], dict[str, Any]],
) -> dict[str, Any]:
    sealed = put_bytes(_json_bytes(value))
    return {
        "role": role,
        "artifact_type": artifact_type,
        "schema_version": schema_version,
        "storage_key": sealed["storage_key"],
        "content_sha256": sealed["content_sha256"],
        "size_bytes": sealed["size_bytes"],
        "authority_class": authority_class,
    }


def _qualification_policy(metadata: dict[str, Any]) -> dict[str, Any]:
    policy = dict(metadata.get("carrier_skin_qualification") or {})
    required = {
        "max_simplex_repair_l1",
        "max_total_correction_l1",
        "negative_tolerance",
        "max_influences",
    }
    if set(policy) != required:
        raise RuntimeError(
            "SKIN_QUALIFICATION_POLICY_DRIFT:" + ",".join(sorted(set(policy) ^ required))
        )
    max_row = float(policy["max_simplex_repair_l1"])
    max_total = float(policy["max_total_correction_l1"])
    negative = float(policy["negative_tolerance"])
    max_influences = int(policy["max_influences"])
    if max_row < 0 or max_total < 0 or negative < 0 or max_influences < 1:
        raise RuntimeError("SKIN_QUALIFICATION_POLICY_INVALID")
    return {
        "max_simplex_repair_l1": max_row,
        "max_total_correction_l1": max_total,
        "negative_tolerance": negative,
        "max_influences": max_influences,
    }


def execute_skin_capability_v1(
    request: dict[str, Any],
    *,
    read_object: Callable[[dict[str, Any]], bytes],
    put_bytes: Callable[[bytes], dict[str, Any]],
) -> dict[str, Any]:
    metadata = dict(request.get("capability_metadata") or {})
    if metadata.get("executor_contract") != SKIN_EXECUTOR_CONTRACT_V1:
        raise RuntimeError("SKIN_EXECUTOR_CONTRACT_DRIFT")
    if str(request.get("kind") or "") != "MODULE":
        raise RuntimeError("SKIN_CAPABILITY_KIND_DRIFT")
    execution_id = str(request.get("execution_id") or "")
    if not execution_id:
        raise RuntimeError("SKIN_EXECUTION_ID_REQUIRED")

    # Heavy MIRA/Torch dependencies stay execution-local. Generic Platform and
    # worker discovery must not import Torch merely because Skin exists.
    from compiler.realsas_compiler_core.artifact_codec_v2 import rigging_surface_from_dict
    from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import canonical_mesh_from_dict_v1
    from compiler.realsas_compiler_core.canonical_rig_ir_v1 import canonical_rig_from_dict_v1
    from compiler.realsas_compiler_core.canonical_skin_ir_v1 import mint_canonical_skin_v1
    from compiler.realsas_compiler_core.carrier_skin_v1 import qualify_carrier_skin_v1
    from models.mira.carrier_native_v55 import proposal_from_dense_wm_v55
    from models.mira.inference_v55 import infer_dense_wm_v55
    from models.mira.runtime_checkpoint_v1 import load_mira_runtime_model_v1

    mesh_ref = _one(request, "RealSaS.CanonicalMeshIR", "v1")
    rig_ref = _one(request, "RealSaS.CanonicalRigIR", "v1")
    surface_ref = _one(request, "RealSaS.RiggingSurfaceIR", "v1")
    support_ref = _one(request, "RealSaS.TESSAMaterialSupportFieldIR", "v1")

    mesh = canonical_mesh_from_dict_v1(_json_payload(read_object, mesh_ref))
    rig = canonical_rig_from_dict_v1(_json_payload(read_object, rig_ref))
    surface = rigging_surface_from_dict(_json_payload(read_object, surface_ref))
    material_support = _json_payload(read_object, support_ref)
    if rig.mesh_lineage_hash != mesh.mesh_lineage_hash:
        raise RuntimeError("SKIN_RIG_MESH_PARENT_DRIFT")
    if surface.geometry_lineage_hash != rig.feature_evidence_binding_hash:
        raise RuntimeError("SKIN_FEATURE_EVIDENCE_RIG_BINDING_DRIFT")
    if surface.geometry_lineage_hash != mesh.source_evidence_binding_hash:
        raise RuntimeError("SKIN_FEATURE_EVIDENCE_MESH_BINDING_DRIFT")

    manifest: dict[str, Any] = {"run_id": "capability-" + execution_id}
    materialize_started = perf_counter()
    materialized = materialize_model_releases(
        manifest=manifest,
        models=request.get("model_releases") or (),
        read_object=read_object,
    )
    materialize_seconds = perf_counter() - materialize_started
    if len(materialized) != 1 or str(materialized[0].get("family") or "") != "MIRA":
        raise RuntimeError("SKIN_MODEL_FAMILY_SET_DRIFT")
    model_release = materialized[0]
    if str(model_release.get("architecture_id") or "") != MIRA_V55_ARCHITECTURE_ID:
        raise RuntimeError("SKIN_MIRA_ARCHITECTURE_DRIFT")
    checkpoint = dict(model_release.get("checkpoint") or {})
    if str(checkpoint.get("artifact_type") or "") != "RealSaS.MIRARuntimeCheckpoint":
        raise RuntimeError("SKIN_MIRA_RUNTIME_CHECKPOINT_TYPE_DRIFT")
    if str(checkpoint.get("schema_version") or "") != "v1":
        raise RuntimeError("SKIN_MIRA_RUNTIME_CHECKPOINT_SCHEMA_DRIFT")

    policy = _qualification_policy(metadata)
    decode_chunk = int(metadata.get("decode_chunk") or 512)
    if decode_chunk < 1:
        raise RuntimeError("SKIN_MIRA_DECODE_CHUNK_INVALID")
    device = _device_from_environment()
    load_started = perf_counter()
    backbone, decoder, runtime_checkpoint = load_mira_runtime_model_v1(
        str(checkpoint.get("path") or ""), device=device
    )
    model_load_seconds = perf_counter() - load_started

    inference_started = perf_counter()
    weights, carrier, skeleton, inference_diagnostics = infer_dense_wm_v55(
        backbone=backbone,
        decoder=decoder,
        mesh=mesh,
        rig=rig,
        surface=surface,
        material_support_payload=material_support,
        device=device,
        decode_chunk=decode_chunk,
    )
    inference_seconds = perf_counter() - inference_started

    provenance = (
        str(model_release.get("model_release_id") or "")
        + ":"
        + str(checkpoint.get("content_sha256") or "")
    )
    proposal = proposal_from_dense_wm_v55(
        weights=weights,
        carrier=carrier,
        skeleton=skeleton,
        model_provenance=provenance,
    )
    qualified = qualify_carrier_skin_v1(
        carrier,
        skeleton,
        proposal,
        max_simplex_repair_l1=policy["max_simplex_repair_l1"],
        max_total_correction_l1=policy["max_total_correction_l1"],
        negative_tolerance=policy["negative_tolerance"],
        max_influences=policy["max_influences"],
    )
    skin = mint_canonical_skin_v1(
        mesh,
        rig,
        qualified_skin=qualified,
        proposal_evidence_refs=(
            str(checkpoint.get("content_sha256") or ""),
            str(support_ref.get("content_sha256") or ""),
        ),
    )

    outputs = [
        _seal(
            skin,
            role="canonical_skin",
            artifact_type="RealSaS.CanonicalSkinIR",
            schema_version="v1",
            authority_class="COMPILER_CANONICAL_SKIN",
            put_bytes=put_bytes,
        ),
        _seal(
            proposal,
            role="skin_influence_proposal",
            artifact_type="RealSaS.CarrierSkinProposalIR",
            schema_version="v1",
            authority_class="LEARNED_SKIN_INFLUENCE_PROPOSAL_ONLY",
            put_bytes=put_bytes,
        ),
        _seal(
            qualified,
            role="carrier_skin_qualification",
            artifact_type="RealSaS.QualifiedCarrierSkinIR",
            schema_version="v1",
            authority_class="QUALIFIED_CARRIER_SKIN",
            put_bytes=put_bytes,
        ),
    ]
    report = dict(qualified.qualification_report or {})
    diagnostics = {
        "schema": "RealSaS.ProductSkinInferenceDiagnostics.v1",
        "skin_lineage_hash": skin.skin_lineage_hash,
        "mesh_lineage_hash": mesh.mesh_lineage_hash,
        "rig_lineage_hash": rig.rig_lineage_hash,
        "feature_evidence_binding_hash": surface.geometry_lineage_hash,
        "material_support_content_sha256": str(support_ref.get("content_sha256") or ""),
        "carrier_vertex_count": len(skin.rows),
        "joint_count": len(rig.joints),
        "mira_architecture_id": MIRA_V55_ARCHITECTURE_ID,
        "mira_model_release_id": str(model_release.get("model_release_id") or ""),
        "mira_checkpoint_content_sha256": str(checkpoint.get("content_sha256") or ""),
        "mira_runtime_checkpoint_schema": str(runtime_checkpoint.get("schema") or ""),
        "model_materialization_seconds": float(materialize_seconds),
        "model_load_seconds": float(model_load_seconds),
        "inference_seconds": float(inference_seconds),
        "inference_device": str(device),
        "decode_chunk": decode_chunk,
        "simplex_max_abs_residual_before": report.get("simplex_max_abs_residual_before"),
        "max_row_correction_l1": report.get("max_row_correction_l1"),
        "total_correction_l1": report.get("total_correction_l1"),
        "feature_evidence_is_canonical_parent": False,
        "material_support_is_canonical_parent": False,
        "canonical_parents": ["CANONICAL_MESH", "CANONICAL_RIG"],
        "fit_performed": False,
        "teacher_authority_used": False,
        "source_skin_runtime_authority": False,
        "gsa_skin_field_authority_minted": False,
        "semantic_skin_transfer_performed": False,
        "product_revision_pass_claimed": False,
        **dict(inference_diagnostics),
    }
    return {
        "capability_id": str(request.get("capability_id") or ""),
        "status": "PASS",
        "outputs": outputs,
        "diagnostics": diagnostics,
        "failure": None,
    }


__all__ = ["execute_skin_capability_v1"]
