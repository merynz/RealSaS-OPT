from __future__ import annotations

"""CAS-native executor for the stable product.skin role.

Current implementation consumes the exact FIT1 MIRA V5.5 ModelRelease. MIRA
proposes W directly on the exact CanonicalMesh carrier; the Compiler performs
shared legality/simplex qualification and mints CanonicalSkinIR bound to the
exact CanonicalMesh + CanonicalRig parent set. RiggingSurface is learned feature
evidence only and never becomes a Skin parent authority.
"""

import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from compiler.realsas_compiler_services.platform_worker.model_inputs import materialize_model_releases

SKIN_EXECUTOR_CONTRACT_V1 = "RealSaS.ProductSkinInference.v1"
MIRA_V55_ARCHITECTURE_ID = "RealSaS.MIRA.DirectCarrierSimplex.Axis541.v5_5"


def _one(request: dict[str, Any], artifact_type: str, schema_version: str) -> dict[str, Any]:
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

    # Heavy MIRA/Arachne/Torch imports are local to actual Skin execution.
    from compiler.realsas_compiler_core.artifact_codec_v2 import rigging_surface_from_dict
    from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import canonical_mesh_from_dict_v1
    from compiler.realsas_compiler_core.canonical_rig_ir_v1 import canonical_rig_from_dict_v1
    from compiler.realsas_compiler_core.canonical_skin_ir_v1 import (
        mint_canonical_skin_v1,
        qualify_canonical_skin_v1,
    )
    from models.mira.runtime_checkpoint_v1 import load_mira_runtime_checkpoint_v1
    from models.mira.runtime_inference_v1 import infer_mira_v55_fit1_v1

    mesh = canonical_mesh_from_dict_v1(
        _json_payload(read_object, _one(request, "RealSaS.CanonicalMeshIR", "v1"))
    )
    rig = canonical_rig_from_dict_v1(
        _json_payload(read_object, _one(request, "RealSaS.CanonicalRigIR", "v1"))
    )
    surface = rigging_surface_from_dict(
        _json_payload(read_object, _one(request, "RealSaS.RiggingSurfaceIR", "v1"))
    )
    if rig.mesh_lineage_hash != mesh.mesh_lineage_hash:
        raise RuntimeError("SKIN_RIG_MESH_PARENT_DRIFT")
    if str(rig.feature_evidence_binding_hash) != str(surface.geometry_lineage_hash):
        raise RuntimeError("SKIN_FEATURE_EVIDENCE_BINDING_DRIFT")

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
    runtime_checkpoint = load_mira_runtime_checkpoint_v1(Path(str(checkpoint.get("path") or "")))

    device_name = os.environ.get("REALSAS_INFERENCE_DEVICE", "cpu").strip() or "cpu"
    inference_started = perf_counter()
    inference = infer_mira_v55_fit1_v1(
        mesh=mesh,
        rig=rig,
        surface=surface,
        runtime_checkpoint=runtime_checkpoint,
        device_name=device_name,
    )
    inference_seconds = perf_counter() - inference_started
    proposal = inference.proposal
    if proposal.skeleton_binding_hash != rig.rig_lineage_hash:
        raise RuntimeError("SKIN_MIRA_PROPOSAL_RIG_BINDING_DRIFT")
    if proposal.carrier_evidence_hash != mesh.carrier_evidence_hash:
        raise RuntimeError("SKIN_MIRA_PROPOSAL_CARRIER_BINDING_DRIFT")

    max_row_repair = float(metadata.get("max_simplex_repair_l1") or 1.0e-5)
    max_total_repair = float(metadata.get("max_total_correction_l1") or 1.0e-4)
    negative_tolerance = float(metadata.get("negative_tolerance") or 1.0e-9)
    qualification = qualify_canonical_skin_v1(
        mesh,
        rig,
        proposal,
        max_simplex_repair_l1=max_row_repair,
        max_total_correction_l1=max_total_repair,
        negative_tolerance=negative_tolerance,
        max_influences=None,
    )
    skin = mint_canonical_skin_v1(mesh, rig, qualification)

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
            qualification,
            role="skin_qualification",
            artifact_type="RealSaS.CanonicalSkinQualificationIR",
            schema_version="v1",
            authority_class="COMPILER_SKIN_QUALIFICATION",
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
    ]
    report = dict(qualification.qualification_report or {})
    diagnostics = {
        "schema": "RealSaS.ProductSkinInferenceDiagnostics.v1",
        "skin_lineage_hash": skin.skin_lineage_hash,
        "mesh_lineage_hash": mesh.mesh_lineage_hash,
        "rig_lineage_hash": rig.rig_lineage_hash,
        "feature_evidence_binding_hash": surface.geometry_lineage_hash,
        "mira_architecture_id": MIRA_V55_ARCHITECTURE_ID,
        "mira_model_release_id": str(model_release.get("model_release_id") or ""),
        "mira_checkpoint_content_sha256": str(checkpoint.get("content_sha256") or ""),
        "model_materialization_seconds": float(materialize_seconds),
        "inference_seconds": float(inference_seconds),
        "inference_device": device_name,
        "prediction_carrier_coverage": report.get("prediction_carrier_coverage"),
        "max_row_correction_l1": report.get("max_row_correction_l1"),
        "total_correction_l1": report.get("total_correction_l1"),
        "semantic_skin_transfer_performed": False,
        "gsa_skin_field_authority_minted": False,
        "feature_evidence_is_canonical_parent": False,
        "canonical_parent_set": ["CANONICAL_MESH", "CANONICAL_RIG"],
        "post_bind_frame_observability_claimed": False,
        "fit_performed": False,
        "generalization_claimed": False,
        "product_revision_pass_claimed": False,
        "runtime_inference": dict(inference.diagnostics),
    }
    return {
        "capability_id": str(request.get("capability_id") or ""),
        "status": "PASS",
        "outputs": outputs,
        "diagnostics": diagnostics,
        "failure": None,
    }


__all__ = ["execute_skin_capability_v1"]
