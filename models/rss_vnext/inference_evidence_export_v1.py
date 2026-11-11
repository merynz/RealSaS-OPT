from __future__ import annotations

"""Lossless addressed RSS.vNext inference proposal -> Compiler evidence adapter.

Only the *two original* FRONT/BACK frames are hashed as observations.
The model's per-vertex fields and source-ray samples remain aligned to the
same decoded topology IDs. Nothing here mints a canonical asset.
"""

from dataclasses import asdict
from hashlib import sha256
import json
from typing import Any, Mapping

import torch
from torch import Tensor

from compiler.realsas_compiler_core.rss_vnext_inference_evidence_v1 import (
    RSSvNextInferenceEvidenceIRV1,
    seal_inference_evidence_v1,
)
from .animation_ready_mesh import RSSvNextStructuralMeshCandidate


def _source_tensor_digest(value: Tensor) -> str:
    raw = value.detach().to("cpu").contiguous()
    header = json.dumps(
        {"dtype": str(raw.dtype), "shape": list(raw.shape)}, sort_keys=True,
    ).encode("utf-8")
    return sha256(header + b"\x00" + raw.numpy().tobytes()).hexdigest()


def _list(value: Tensor) -> Any:
    return value.detach().to("cpu").tolist()


def _mesh_evidence(mesh: RSSvNextStructuralMeshCandidate) -> dict[str, Any]:
    mesh.validate()
    if mesh.source_authority != "LEARNED_PREDICTED_TOPOLOGY":
        raise ValueError("RSS_VNEXT_EXPORT_TEACHER_SOURCE_FORBIDDEN")
    rays = mesh.source_ray_evidence
    if rays is None:
        raise ValueError("RSS_VNEXT_EXPORT_ORIGINAL_RAYS_REQUIRED")
    rays.validate(batch=1, points=mesh.rest_vertices_iris.shape[1])
    return {
        "topology_tokens": list(mesh.topology_token_ids),
        "rest_xyz": _list(mesh.rest_vertices_iris[0]),
        "base_xyz": _list(mesh.base_vertices_iris[0]),
        "local_features": _list(mesh.local_features[0]),
        "faces": _list(mesh.faces),
        "vertex_components": _list(mesh.vertex_component_ids),
        "face_components": _list(mesh.face_component_ids),
        "normals_xyz": _list(mesh.normals[0]),
        "semantic_logits": _list(mesh.semantic_logits[0]),
        "presentation_role_logits": _list(mesh.presentation_role_logits[0]),
        "presentation_role_schema_id": mesh.presentation_role_schema_id,
        "affordance_logits": _list(mesh.affordance_logits[0]),
        "graphics_logits": _list(mesh.graphics_logits[0]),
        "static_quality_logits": _list(mesh.static_quality_logits[0]),
        "unsigned_distance": _list(mesh.geometry["unsigned_distance"][0]),
        "sign_logits": _list(mesh.geometry["sign_logit"][0]),
        "source_rays": {
            "rgba": _list(rays.rgba[0]),
            "valid": _list(rays.projection_valid[0]),
            "alpha": _list(rays.silhouette_alpha[0]),
            "appearance_authority": rays.appearance_authority,
            "depth_authority": rays.ray_depth_authority,
        },
        "source_authority": mesh.source_authority,
        "signed_solid_authority": mesh.signed_solid_authority,
        "canonical_mesh_authority": mesh.canonical_mesh_authority,
    }


@torch.no_grad()
def export_inference_evidence_v1(
    result: Mapping[str, Any],
    *,
    original_rgba: Tensor,
    original_camera_features: Tensor,
    checkpoint_sha256: str,
) -> RSSvNextInferenceEvidenceIRV1:
    """Seal an explicitly unqualified proposal, never teacher-forced material.

    Caller must separately bind the exact source asset bytes and actual trained
    model checkpoint. Two observed source frames MUST match model state.
    """
    if original_rgba.ndim != 5 or original_rgba.shape[0:3] != (1, 2, 4):
        raise ValueError("RSS_VNEXT_EXPORT_REQUIRES_TWO_RGBA")
    if original_camera_features.shape != (1, 2, 13):
        raise ValueError("RSS_VNEXT_EXPORT_CAMERA_SHAPE")
    if (
        not bool(torch.isfinite(original_rgba).all())
        or not bool(torch.isfinite(original_camera_features).all())
    ):
        raise ValueError("RSS_VNEXT_EXPORT_NONFINITE_SOURCE")
    if not isinstance(result, Mapping) or result.get("qualification_status") != "UNQUALIFIED_NEURAL_PROPOSAL":
        raise ValueError("RSS_VNEXT_EXPORT_NEURAL_AUTHORITY_INVALID")
    if result.get("observed_view_count") != 2 or result.get("auxiliary_view_authority") != "LEARNED_PREDICTED_ONLY":
        raise ValueError("RSS_VNEXT_EXPORT_SOURCE_AUTHORITY_DRIFT")
    state = result.get("state")
    if (
        state is None
        or not torch.equal(state.observed_rgba, original_rgba)
        or not torch.equal(state.camera_features, original_camera_features)
        or state.source_view_authority != ("SOURCE_OBSERVED_FRONT", "SOURCE_OBSERVED_BACK")
    ):
        raise ValueError("RSS_VNEXT_EXPORT_SOURCE_STATE_MISMATCH")
    status = result.get("mesh_status")
    if status not in ("ABSTAIN", "UNQUALIFIED_MESH_CANDIDATE"):
        raise ValueError("RSS_VNEXT_EXPORT_MESH_STATUS_UNKNOWN")
    mesh = result.get("mesh_candidate")
    if (status == "ABSTAIN") != (mesh is None):
        raise ValueError("RSS_VNEXT_EXPORT_MESH_STATUS_MISMATCH")
    events = result.get("control_event_proposal") or ()
    if events and mesh is None:
        raise ValueError("RSS_VNEXT_EXPORT_CONTROLS_WITHOUT_MESH")
    serialized_events = []
    for ev in events:
        row = asdict(ev)
        row["kind"] = int(ev.kind)
        serialized_events.append(row)
    dynamic = result.get("dynamic_mesh_candidate")
    dynamic_payload = None
    if dynamic is not None:
        dynamic.validate()
        if dynamic.source_mesh is not mesh or not events:
            raise ValueError("RSS_VNEXT_EXPORT_DYNAMIC_TOPOLOGY_BINDING_DRIFT")
        dynamic_payload = {
            "vertices": _list(dynamic.vertices[0]),
            "weights": _list(dynamic.weights[0]),
            "displacement": _list(dynamic.displacement[0]),
            "readiness_logits": _list(dynamic.readiness_logits[0]),
            "authority": dynamic.authority,
        }
    value = RSSvNextInferenceEvidenceIRV1(
        checkpoint_sha256=checkpoint_sha256,
        observed_front_sha256=_source_tensor_digest(original_rgba[0, 0]),
        observed_back_sha256=_source_tensor_digest(original_rgba[0, 1]),
        camera_binding_sha256=_source_tensor_digest(original_camera_features),
        mesh_status=status,
        mesh=None if mesh is None else _mesh_evidence(mesh),
        control_events=tuple(serialized_events),
        dynamic=dynamic_payload,
        scene={
            "source_semantic_patch_logits": _list(state.source_semantic_patch_logits[0]),
            "source_semantic_log_uncertainty": _list(state.source_semantic_log_uncertainty[0]),
            # Retain query-space addresses and anonymous membership, not just
            # the pair graph. These are 3D hypotheses (including UNKNOWN),
            # never source segmentation, drawable authority or canonical IDs.
            "region_query_xyz": _list(state.query_xyz[0]),
            "region_membership_logits": _list(state.semantic_logits[0]),
            "region_graphics_logits": _list(state.graphics_logits[0]),
            "region_relation_logits": _list(state.region_relation_logits[0]),
            "region_relation_valid_mask": _list(state.region_relation_valid_mask[0]),
            "presentation_role_probabilities": _list(state.presentation_role_probabilities[0]),
            "constructed_camera_features": (
                None if state.constructed_camera_features is None
                else _list(state.constructed_camera_features[0])
            ),
            "constructed_camera_angles": (
                None if state.constructed_camera_angles is None
                else _list(state.constructed_camera_angles[0])
            ),
            "authority": "PREDICTED_FROM_TWO_ORIGINAL_VIEWS",
        },
        reason=str(result.get("mesh_failure") or "INFERENCE_TOPOLOGY_ABSTAIN") if mesh is None else "",
        metadata={
            "source_input_format": "FLOAT_RGBA_0_TO_1",
            "camera_feature_contract": "ORTHOGRAPHIC_13D",
            "source_pixels_attached": False,
            "per_vertex_evidence_preserved": mesh is not None,
            "product_qualification_minted": False,
        },
    )
    return seal_inference_evidence_v1(value)


__all__ = ["export_inference_evidence_v1"]
