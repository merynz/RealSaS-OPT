from __future__ import annotations

"""CAS-native executor for the stable product.rig role.

The current implementation is hybrid: AXIS V5.4.1 proposes one hard causal
parent tree and deterministic discrete XYZ geometry; the Compiler validates that
required tree without parent reselection and mints CanonicalRigIR bound to the
exact CanonicalMesh parent. RiggingSurface is current feature evidence only.
"""

import json
import os
from time import perf_counter
from typing import Any, Callable

from compiler.realsas_compiler_services.platform_worker.model_inputs import (
    materialize_model_releases,
)

RIG_EXECUTOR_CONTRACT_V1 = "RealSaS.ProductRigInference.v1"
AXIS_V541_ARCHITECTURE_ID = (
    "RealSaS.AXIS41.BFSHardParent.OrderedDiscreteXYZ256.FrozenCausal.v5_4_1"
)


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
            f"RIG_INPUT_ARTIFACT_CARDINALITY:{artifact_type}:{schema_version}:{len(matches)}"
        )
    return matches[0]


def _json_payload(read_object, ref: dict[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(read_object(ref).decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(
            "RIG_TYPED_JSON_DECODE_FAILED:" + str(ref.get("artifact_type") or "")
        ) from exc
    if not isinstance(value, dict):
        raise RuntimeError("RIG_TYPED_JSON_OBJECT_REQUIRED")
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
        raise RuntimeError("RIG_CUDA_REQUESTED_BUT_UNAVAILABLE")
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


def execute_rig_capability_v1(
    request: dict[str, Any],
    *,
    read_object: Callable[[dict[str, Any]], bytes],
    put_bytes: Callable[[bytes], dict[str, Any]],
) -> dict[str, Any]:
    metadata = dict(request.get("capability_metadata") or {})
    if metadata.get("executor_contract") != RIG_EXECUTOR_CONTRACT_V1:
        raise RuntimeError("RIG_EXECUTOR_CONTRACT_DRIFT")
    if str(request.get("kind") or "") != "MODULE":
        raise RuntimeError("RIG_CAPABILITY_KIND_DRIFT")
    execution_id = str(request.get("execution_id") or "")
    if not execution_id:
        raise RuntimeError("RIG_EXECUTION_ID_REQUIRED")

    # Heavy AXIS/Torch dependencies remain local to actual Rig execution.
    from compiler.realsas_compiler_core.artifact_codec_v2 import rigging_surface_from_dict
    from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import canonical_mesh_from_dict_v1
    from compiler.realsas_compiler_core.canonical_rig_ir_v1 import mint_canonical_rig_v1
    from compiler.realsas_compiler_core.required_tree_v1 import qualify_required_skeleton_tree_v1
    from models.axis.runtime_checkpoint_v1 import load_axis_runtime_model_v1
    from models.geppetto.reference_strength_v1.rigging_surface_tensorization_v1 import (
        tensorize_rigging_surface_v1,
    )

    mesh_ref = _one(request, "RealSaS.CanonicalMeshIR", "v1")
    surface_ref = _one(request, "RealSaS.RiggingSurfaceIR", "v1")
    mesh = canonical_mesh_from_dict_v1(_json_payload(read_object, mesh_ref))
    surface = rigging_surface_from_dict(_json_payload(read_object, surface_ref))
    if surface.geometry_lineage_hash != mesh.source_evidence_binding_hash:
        raise RuntimeError("RIG_FEATURE_EVIDENCE_MESH_BINDING_DRIFT")

    manifest: dict[str, Any] = {"run_id": "capability-" + execution_id}
    materialize_started = perf_counter()
    materialized = materialize_model_releases(
        manifest=manifest,
        models=request.get("model_releases") or (),
        read_object=read_object,
    )
    materialize_seconds = perf_counter() - materialize_started
    if len(materialized) != 1 or str(materialized[0].get("family") or "") != "AXIS":
        raise RuntimeError("RIG_MODEL_FAMILY_SET_DRIFT")
    model_release = materialized[0]
    if str(model_release.get("architecture_id") or "") != AXIS_V541_ARCHITECTURE_ID:
        raise RuntimeError("RIG_AXIS_ARCHITECTURE_DRIFT")
    checkpoint = dict(model_release.get("checkpoint") or {})
    if str(checkpoint.get("artifact_type") or "") != "RealSaS.AXISRuntimeCheckpoint":
        raise RuntimeError("RIG_AXIS_RUNTIME_CHECKPOINT_TYPE_DRIFT")
    if str(checkpoint.get("schema_version") or "") != "v1":
        raise RuntimeError("RIG_AXIS_RUNTIME_CHECKPOINT_SCHEMA_DRIFT")

    device = _device_from_environment()
    model, runtime_checkpoint = load_axis_runtime_model_v1(
        str(checkpoint.get("path") or ""), device=device
    )
    tensor = tensorize_rigging_surface_v1(surface, require_scene_first=True)
    if tensor.source_surface_hash != surface.geometry_lineage_hash:
        raise RuntimeError("RIG_AXIS_FEATURE_TENSOR_BINDING_DRIFT")
    max_steps = int(metadata.get("resource_step_limit_max") or 256)
    if max_steps < 1:
        raise RuntimeError("RIG_RESOURCE_STEP_LIMIT_INVALID")
    resource_step_limit = min(max_steps, tensor.node_count)

    inference_started = perf_counter()
    proposal = model.propose(
        tensor,
        resource_step_limit=resource_step_limit,
        generator=None,
        diffusion_sample_steps=None,
    )
    inference_seconds = perf_counter() - inference_started
    if proposal.model_provenance != AXIS_V541_ARCHITECTURE_ID:
        raise RuntimeError("RIG_AXIS_PROPOSAL_ARCHITECTURE_DRIFT")
    if str((proposal.metadata or {}).get("candidate_architecture") or "") != AXIS_V541_ARCHITECTURE_ID:
        raise RuntimeError("RIG_AXIS_PROPOSAL_METADATA_ARCHITECTURE_DRIFT")
    if proposal.surface_binding_hash != surface.geometry_lineage_hash:
        raise RuntimeError("RIG_AXIS_PROPOSAL_FEATURE_BINDING_DRIFT")

    authority_bindings = {
        "canonical_mesh": mesh.mesh_lineage_hash,
        "mechanical_carrier_evidence": mesh.carrier_evidence_hash,
        "mechanical_carrier_topology": mesh.carrier_topology_hash,
        "mechanical_carrier_geometry": mesh.carrier_geometry_hash,
    }
    qualified = qualify_required_skeleton_tree_v1(
        surface,
        proposal,
        authority_bindings=authority_bindings,
    )
    rig = mint_canonical_rig_v1(
        mesh,
        proposal=proposal,
        qualified_skeleton=qualified,
        feature_evidence_binding_hash=surface.geometry_lineage_hash,
    )

    outputs = [
        _seal(
            rig,
            role="canonical_rig",
            artifact_type="RealSaS.CanonicalRigIR",
            schema_version="v1",
            authority_class="COMPILER_CANONICAL_RIG",
            put_bytes=put_bytes,
        ),
        _seal(
            proposal,
            role="rig_structure_proposal",
            artifact_type="RealSaS.SkeletonProposalIR",
            schema_version="v1",
            authority_class="LEARNED_RIG_STRUCTURE_PROPOSAL_ONLY",
            put_bytes=put_bytes,
        ),
    ]
    report = dict(qualified.qualification_report or {})
    diagnostics = {
        "schema": "RealSaS.ProductRigInferenceDiagnostics.v1",
        "rig_lineage_hash": rig.rig_lineage_hash,
        "mesh_lineage_hash": mesh.mesh_lineage_hash,
        "feature_evidence_binding_hash": surface.geometry_lineage_hash,
        "joint_count": len(rig.joints),
        "root_joint_count": len(rig.root_joint_ids),
        "axis_architecture_id": AXIS_V541_ARCHITECTURE_ID,
        "axis_model_release_id": str(model_release.get("model_release_id") or ""),
        "axis_checkpoint_content_sha256": str(checkpoint.get("content_sha256") or ""),
        "axis_runtime_checkpoint_schema": str(runtime_checkpoint.get("schema") or ""),
        "model_materialization_seconds": float(materialize_seconds),
        "inference_seconds": float(inference_seconds),
        "inference_device": str(device),
        "resource_step_limit": int(resource_step_limit),
        "selection_search_performed": report.get("selection_search_performed"),
        "parent_reselection_performed": report.get("parent_reselection_performed"),
        "frame_observability_claimed": False,
        "post_bind_frame_observability_required": True,
        "feature_evidence_is_canonical_parent": False,
        "canonical_parent": "CANONICAL_MESH",
        "fit_performed": False,
        "teacher_authority_used": False,
        "source_rig_runtime_authority": False,
        "source_skin_runtime_authority": False,
        "product_revision_pass_claimed": False,
    }
    return {
        "capability_id": str(request.get("capability_id") or ""),
        "status": "PASS",
        "outputs": outputs,
        "diagnostics": diagnostics,
        "failure": None,
    }


__all__ = ["execute_rig_capability_v1"]
