from __future__ import annotations

"""CAS-native executor for the stable product.rig role.

AXIS is a replaceable learned proposal provider. The Compiler is the sole Rig
authority: it validates the AXIS-owned hard causal tree and mints canonical joint
identities. The learned proposal is intentionally not emitted as an authority
artifact. Final post-bind frame observability remains a Mechanics responsibility
because it requires the final carrier-native Skin field.

The current FIT1 AXIS checkpoint consumes the historical factorized
RiggingSurface tensor contract. That surface is compatibility feature evidence,
not the canonical parent of Rig. CanonicalMesh is the sole product parent; a
future AXIS input adapter may replace the surface dependency without changing the
stable product.rig role or canonical Rig authority contract.
"""

import json
import os
from time import perf_counter
from typing import Any, Callable

from compiler.realsas_compiler_services.platform_worker.model_inputs import (
    materialize_model_releases,
)


RIG_EXECUTOR_CONTRACT_V1 = "RealSaS.ProductRigCompile.v1"
AXIS_V541_ARCHITECTURE_ID = (
    "RealSaS.AXIS41.BFSHardParent.OrderedDiscreteXYZ256.FrozenCausal.v5_4_1"
)
AXIS_V541_CHECKPOINT_SCHEMA = "RealSaS.AXISBFSDiscreteXYZV541.DeployableCheckpoint.v1"
AXIS_V541_INPUT_ADAPTER_CONTRACT = "RealSaS.GeppettoRiggingSurfaceTensorization.v1"


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


def _failure(
    request: dict[str, Any], *, code: str, diagnostics: dict[str, Any]
) -> dict[str, Any]:
    return {
        "capability_id": str(request.get("capability_id") or ""),
        "status": "FAIL",
        "outputs": [],
        "diagnostics": dict(diagnostics),
        "failure": {
            "code": str(code),
            "class": "COMPILER_RIG",
            "reported_owner_module_id": str(request.get("owner_module_id") or ""),
            "diagnostics": dict(diagnostics),
        },
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
    if str(metadata.get("canonical_parent_role") or "") != "mesh":
        raise RuntimeError("RIG_CANONICAL_PARENT_ROLE_DRIFT")
    if str(metadata.get("axis_input_adapter_contract") or "") != AXIS_V541_INPUT_ADAPTER_CONTRACT:
        raise RuntimeError("RIG_AXIS_INPUT_ADAPTER_CONTRACT_DRIFT")
    if metadata.get("axis_input_adapter_is_canonical_authority") is not False:
        raise RuntimeError("RIG_AXIS_INPUT_ADAPTER_AUTHORITY_FORBIDDEN")
    if metadata.get("axis_input_adapter_replaceable") is not True:
        raise RuntimeError("RIG_AXIS_INPUT_ADAPTER_MUST_BE_REPLACEABLE")
    if metadata.get("surface_dependency_is_compatibility_evidence_only") is not True:
        raise RuntimeError("RIG_SURFACE_DEPENDENCY_MUST_BE_COMPATIBILITY_EVIDENCE_ONLY")
    if metadata.get("compiler_parent_reselection_allowed") is not False:
        raise RuntimeError("RIG_COMPILER_PARENT_RESELECTION_MUST_BE_FALSE")
    if metadata.get("proposal_authority_artifact_emitted") is not False:
        raise RuntimeError("RIG_PROPOSAL_AUTHORITY_OUTPUT_MUST_BE_FALSE")
    if metadata.get("canonical_mesh_mutation_allowed") is not False:
        raise RuntimeError("RIG_CANONICAL_MESH_MUTATION_MUST_BE_FALSE")
    if metadata.get("post_bind_frame_observability_claimed") is not False:
        raise RuntimeError("RIG_POST_BIND_FRAME_OBSERVABILITY_FORBIDDEN")
    if metadata.get("frame_observability_deferred_to_mechanics") is not True:
        raise RuntimeError("RIG_FRAME_OBSERVABILITY_DEFERRED_CONTRACT_REQUIRED")
    if str(metadata.get("axis_checkpoint_schema") or "") != AXIS_V541_CHECKPOINT_SCHEMA:
        raise RuntimeError("RIG_AXIS_CHECKPOINT_SCHEMA_DRIFT")

    # Heavy model imports remain capability-local. Generic Platform worker import
    # must not pull Torch/AXIS into unrelated capabilities.
    from compiler.realsas_compiler_core.artifact_codec_v2 import (
        qualified_skeleton_from_dict,
        rigging_surface_from_dict,
    )
    from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import (
        canonical_mesh_from_dict_v1,
    )
    from compiler.realsas_compiler_core.hashing import content_sha256
    from compiler.realsas_compiler_core.required_tree_v1 import (
        qualify_required_skeleton_tree_v1,
    )
    from compiler.realsas_compiler_core.types import QualificationError
    from compiler.realsas_compiler_services.platform_worker.model_binding import (
        require_materialized_model_release,
    )
    from models.axis.deployable_checkpoint_v541 import (
        load_axis_v541_deployable_checkpoint_v1,
    )
    from models.geppetto.reference_strength_v1.rigging_surface_tensorization_v1 import (
        tensorize_rigging_surface_v1,
    )

    manifest: dict[str, Any] = {"run_id": "capability-" + execution_id}
    materialize_started = perf_counter()
    materialized = materialize_model_releases(
        manifest=manifest,
        models=request.get("model_releases") or (),
        read_object=read_object,
    )
    materialize_seconds = perf_counter() - materialize_started
    families = tuple(sorted(str(row["family"]) for row in materialized))
    if families != ("AXIS",):
        raise RuntimeError("RIG_MODEL_FAMILY_SET_DRIFT")
    axis_ref = require_materialized_model_release(
        manifest,
        family="AXIS",
        architecture_id=AXIS_V541_ARCHITECTURE_ID,
    )

    mesh_ref = _one(request, "RealSaS.CanonicalMeshIR", "v1")
    surface_ref = _one(request, "RealSaS.RiggingSurfaceIR", "v1")
    mesh = canonical_mesh_from_dict_v1(_json_payload(read_object, mesh_ref))
    surface = rigging_surface_from_dict(_json_payload(read_object, surface_ref))
    if mesh.source_evidence_binding_hash != surface.geometry_lineage_hash:
        raise RuntimeError("RIG_MESH_SURFACE_BINDING_DRIFT")
    if str(mesh_ref.get("content_sha256") or "") == "":
        raise RuntimeError("RIG_MESH_CONTENT_IDENTITY_REQUIRED")

    surface_tensor = tensorize_rigging_surface_v1(surface, require_scene_first=True)
    if surface_tensor.source_surface_hash != surface.geometry_lineage_hash:
        raise RuntimeError("RIG_SURFACE_TENSORIZATION_BINDING_DRIFT")
    maximum_joint_steps = int(metadata.get("maximum_joint_steps") or 0)
    if maximum_joint_steps <= 0:
        raise RuntimeError("RIG_MAXIMUM_JOINT_STEPS_INVALID")
    resource_step_limit = min(maximum_joint_steps, int(surface_tensor.node_count))
    if resource_step_limit <= 0:
        raise RuntimeError("RIG_SURFACE_EMPTY")

    device = _device_from_environment()
    model, checkpoint = load_axis_v541_deployable_checkpoint_v1(
        axis_ref.checkpoint_path,
        device=device,
    )
    if str(checkpoint.get("schema") or "") != AXIS_V541_CHECKPOINT_SCHEMA:
        raise RuntimeError("RIG_AXIS_CHECKPOINT_SCHEMA_DRIFT")
    proposal = model.propose(
        surface_tensor,
        resource_step_limit=resource_step_limit,
        diffusion_sample_steps=None,
    )
    proposal_metadata = dict(proposal.metadata or {})
    if str(proposal_metadata.get("candidate_architecture") or "") != AXIS_V541_ARCHITECTURE_ID:
        raise RuntimeError("RIG_AXIS_PROPOSAL_ARCHITECTURE_DRIFT")
    if proposal_metadata.get("hard_causal_parent_authority") is not True:
        raise RuntimeError("RIG_AXIS_HARD_PARENT_AUTHORITY_REQUIRED")
    if proposal_metadata.get("compiler_parent_reselection_allowed") is not False:
        raise RuntimeError("RIG_AXIS_PARENT_RESELECTION_FORBIDDEN")
    if proposal_metadata.get("diffusion_runtime_authority") is not False:
        raise RuntimeError("RIG_AXIS_DIFFUSION_AUTHORITY_FORBIDDEN")

    authority_bindings = {
        "canonical_mesh_lineage_hash": mesh.mesh_lineage_hash,
        "canonical_mesh_content_sha256": str(mesh_ref.get("content_sha256") or ""),
        "carrier_evidence_hash": mesh.carrier_evidence_hash,
        "carrier_topology_hash": mesh.carrier_topology_hash,
        "carrier_geometry_hash": mesh.carrier_geometry_hash,
        "static_qualification_hash": mesh.static_qualification_hash,
        "axis_model_release_id": axis_ref.model_release_id,
        "axis_behavior_identity_sha256": axis_ref.behavior_identity_sha256,
        "axis_checkpoint_content_sha256": axis_ref.checkpoint_content_sha256,
        "axis_input_adapter_contract": AXIS_V541_INPUT_ADAPTER_CONTRACT,
        "axis_input_surface_evidence_hash": surface.geometry_lineage_hash,
    }
    proposal_semantic_hash = content_sha256(proposal.to_dict())
    try:
        skeleton = qualify_required_skeleton_tree_v1(
            surface,
            proposal,
            authority_bindings=authority_bindings,
        )
    except QualificationError as exc:
        diagnostics = {
            "schema": "RealSaS.ProductRigCompileDiagnostics.v1",
            "mesh_lineage_hash": mesh.mesh_lineage_hash,
            "surface_lineage_hash": surface.geometry_lineage_hash,
            "axis_model_release_id": axis_ref.model_release_id,
            "axis_checkpoint_content_sha256": axis_ref.checkpoint_content_sha256,
            "axis_proposal_semantic_hash": proposal_semantic_hash,
            "canonical_parent_role": "mesh",
            "axis_input_adapter_contract": AXIS_V541_INPUT_ADAPTER_CONTRACT,
            "axis_input_adapter_is_canonical_authority": False,
            "surface_dependency_is_compatibility_evidence_only": True,
            "compiler_parent_reselection_performed": False,
            "proposal_authority_artifact_emitted": False,
            "canonical_mesh_mutated": False,
            "post_bind_frame_observability_claimed": False,
            "frame_observability_deferred_to_mechanics": True,
        }
        return _failure(request, code=str(exc), diagnostics=diagnostics)

    # Round-trip through the existing canonical Rig codec. This intentionally
    # reuses QualifiedSkeletonIR rather than minting a second CanonicalRigIR truth.
    skeleton = qualified_skeleton_from_dict(skeleton.to_dict())
    sealed = put_bytes(_json_bytes(skeleton))
    diagnostics = {
        "schema": "RealSaS.ProductRigCompileDiagnostics.v1",
        "skeleton_lineage_hash": skeleton.skeleton_lineage_hash,
        "mesh_lineage_hash": mesh.mesh_lineage_hash,
        "mesh_content_sha256": str(mesh_ref.get("content_sha256") or ""),
        "surface_lineage_hash": surface.geometry_lineage_hash,
        "surface_tensorization_hash": surface_tensor.tensorization_hash,
        "axis_model_release_id": axis_ref.model_release_id,
        "axis_behavior_identity_sha256": axis_ref.behavior_identity_sha256,
        "axis_checkpoint_content_sha256": axis_ref.checkpoint_content_sha256,
        "axis_proposal_semantic_hash": proposal_semantic_hash,
        "joint_count": len(skeleton.joints),
        "model_materialization_seconds": float(materialize_seconds),
        "inference_device": str(device),
        "maximum_joint_steps": maximum_joint_steps,
        "resource_step_limit": resource_step_limit,
        "canonical_parent_role": "mesh",
        "axis_input_adapter_contract": AXIS_V541_INPUT_ADAPTER_CONTRACT,
        "axis_input_adapter_is_canonical_authority": False,
        "axis_input_adapter_replaceable": True,
        "surface_dependency_is_compatibility_evidence_only": True,
        "compiler_parent_reselection_performed": False,
        "proposal_authority_artifact_emitted": False,
        "canonical_mesh_mutated": False,
        "post_bind_frame_observability_claimed": False,
        "frame_observability_deferred_to_mechanics": True,
        "skin_authority_claimed": False,
        "mechanics_authority_claimed": False,
        "motion_authority_claimed": False,
        "product_revision_pass_claimed": False,
    }
    return {
        "capability_id": str(request.get("capability_id") or ""),
        "status": "PASS",
        "outputs": [
            {
                "role": "canonical_rig",
                "artifact_type": "RealSaS.QualifiedSkeletonIR",
                "schema_version": "v1",
                "storage_key": sealed["storage_key"],
                "content_sha256": sealed["content_sha256"],
                "size_bytes": sealed["size_bytes"],
                "authority_class": "COMPILER_CANONICAL_RIG",
            }
        ],
        "diagnostics": diagnostics,
        "failure": None,
    }


__all__ = ["execute_rig_capability_v1"]
