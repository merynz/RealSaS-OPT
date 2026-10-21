from __future__ import annotations

"""Observation-time role-free presentation prediction bridge.

Only IRIS predictions plus optional source observations enter these builders.
Materialized V2 teacher/all-hit payloads, teacher topology, rig/skin truth and
named semantic taxonomies are deliberately absent from every inference signature.

Prediction validity is fail-closed: a neural value is not evidence merely because
a head emitted it. Callers must explicitly provide a qualification mask for every
predicted family. The first Structural Evidence A/B may therefore qualify tie-safe
front and complete-stack visibility while leaving positive-only occlusion and
unqualified boundary/static families at ABSTAIN.
"""

from typing import Mapping

import torch

from .presentation_evidence_v1 import (
    PRESENTATION_KIND_PAIR,
    PRESENTATION_KIND_RAY_HIT,
    PRESENTATION_KIND_SURFACE,
    RoleFreePresentationEvidenceV1,
    build_role_free_presentation_evidence_v1,
)


PREDICTED_PRESENTATION_PROVENANCE_V1 = {
    "schema": "RealSaS.PredictedRoleFreePresentationProvenance.v1",
    "inference_source": "OBSERVATION_TIME_IRIS_PREDICTIONS_AND_SOURCE_OBSERVATIONS_ONLY",
    "prediction_validity": "EXPLICIT_FAIL_CLOSED_QUALIFICATION_MASKS",
    "materialized_v2_truth_in_inference_forward": False,
    "teacher_topology_in_inference_forward": False,
    "teacher_rig_or_skin_in_inference_forward": False,
    "named_semantic_taxonomy_in_inference_forward": False,
}


def _safe_xyz(xyz: torch.Tensor, valid_mask: torch.Tensor, *, name: str) -> torch.Tensor:
    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError(f"IRIS_PREDICTED_PRESENTATION_V1_{name}_XYZ_SHAPE_INVALID")
    if valid_mask.shape != xyz.shape[:2]:
        raise ValueError(f"IRIS_PREDICTED_PRESENTATION_V1_{name}_MASK_SHAPE_INVALID")
    valid = valid_mask.bool()
    return torch.where(valid[..., None], xyz, torch.zeros_like(xyz))


def _qualified_mask(
    mask: torch.Tensor | None,
    *,
    token_valid_mask: torch.Tensor,
    name: str,
) -> torch.Tensor:
    if mask is None:
        return torch.zeros_like(token_valid_mask, dtype=torch.bool)
    if mask.shape != token_valid_mask.shape:
        raise ValueError(f"IRIS_PREDICTED_PRESENTATION_V1_{name}_VALIDITY_SHAPE_INVALID")
    return mask.bool() & token_valid_mask.bool()


def _expand_camera(
    camera_features: torch.Tensor,
    *,
    token_count: int,
    reference: torch.Tensor,
    name: str,
) -> torch.Tensor:
    if camera_features.shape != (reference.shape[0], 13):
        raise ValueError(f"IRIS_PREDICTED_PRESENTATION_V1_{name}_CAMERA_SHAPE_INVALID")
    return camera_features[:, None, :].expand(-1, token_count, -1)


def concat_role_free_presentation_evidence_v1(
    *parts: RoleFreePresentationEvidenceV1,
) -> RoleFreePresentationEvidenceV1:
    """Concatenate already typed evidence families without changing authority."""

    if not parts:
        raise ValueError("IRIS_PREDICTED_PRESENTATION_V1_NO_PARTS")
    reference = parts[0]
    for part in parts[1:]:
        if part.values.shape[0] != reference.values.shape[0]:
            raise ValueError("IRIS_PREDICTED_PRESENTATION_V1_BATCH_MISMATCH")
        if part.values.device != reference.values.device:
            raise ValueError("IRIS_PREDICTED_PRESENTATION_V1_DEVICE_MISMATCH")
        if part.values.dtype != reference.values.dtype:
            raise ValueError("IRIS_PREDICTED_PRESENTATION_V1_DTYPE_MISMATCH")
        if part.schema_version != reference.schema_version:
            raise ValueError("IRIS_PREDICTED_PRESENTATION_V1_SCHEMA_MISMATCH")
    return RoleFreePresentationEvidenceV1(
        values=torch.cat(tuple(part.values for part in parts), dim=1),
        valid_mask=torch.cat(tuple(part.valid_mask for part in parts), dim=1),
        kind_id=torch.cat(tuple(part.kind_id for part in parts), dim=1),
    )


def surface_presentation_from_predictions_v1(
    *,
    xyz: torch.Tensor,
    camera_features: torch.Tensor,
    token_valid_mask: torch.Tensor,
    appearance: Mapping[str, torch.Tensor],
    boundary_valid_mask: torch.Tensor | None = None,
    source_rgba: torch.Tensor | None = None,
    source_rgba_valid: torch.Tensor | None = None,
    source_support: torch.Tensor | None = None,
    source_support_valid: torch.Tensor | None = None,
) -> RoleFreePresentationEvidenceV1:
    """Type per-surface predictions and observations with explicit qualification."""

    required = ("alpha_boundary_probability", "appearance_boundary_probability")
    missing = [name for name in required if name not in appearance]
    if missing:
        raise ValueError(
            "IRIS_PREDICTED_PRESENTATION_V1_APPEARANCE_MISSING:" + ",".join(missing)
        )
    safe_xyz = _safe_xyz(xyz, token_valid_mask, name="SURFACE")
    token_valid = token_valid_mask.bool()
    boundary_valid = _qualified_mask(
        boundary_valid_mask,
        token_valid_mask=token_valid,
        name="SURFACE_BOUNDARY",
    )
    camera = _expand_camera(
        camera_features,
        token_count=xyz.shape[1],
        reference=xyz,
        name="SURFACE",
    )
    return build_role_free_presentation_evidence_v1(
        xyz=safe_xyz,
        kind_id=torch.full(
            xyz.shape[:2], PRESENTATION_KIND_SURFACE, dtype=torch.long, device=xyz.device
        ),
        token_valid_mask=token_valid,
        camera_features=camera,
        camera_valid=token_valid,
        source_rgba=source_rgba,
        source_rgba_valid=source_rgba_valid,
        alpha_boundary=appearance["alpha_boundary_probability"],
        alpha_boundary_valid=boundary_valid,
        appearance_boundary=appearance["appearance_boundary_probability"],
        appearance_boundary_valid=boundary_valid,
        source_support=source_support,
        source_support_valid=source_support_valid,
    )


def pair_presentation_from_predictions_v1(
    *,
    xyz_a: torch.Tensor,
    xyz_b: torch.Tensor,
    camera_features: torch.Tensor,
    token_valid_mask: torch.Tensor,
    relations: Mapping[str, torch.Tensor],
    front_valid_mask: torch.Tensor | None = None,
    occlusion_valid_mask: torch.Tensor | None = None,
    boundary_valid_mask: torch.Tensor | None = None,
) -> RoleFreePresentationEvidenceV1:
    """Type pair predictions without upgrading unsupported families to authority.

    Pairwise outputs never acquire global-visibility authority here. In the first
    V2-tie-safe specialist qualification, front may become valid; positive-only
    occlusion and projected boundaries remain ABSTAIN unless separately qualified.
    """

    required = (
        "front_probability",
        "occlusion_probability",
        "projected_alpha_boundary_probability",
        "projected_appearance_boundary_probability",
    )
    missing = [name for name in required if name not in relations]
    if missing:
        raise ValueError(
            "IRIS_PREDICTED_PRESENTATION_V1_RELATION_MISSING:" + ",".join(missing)
        )
    if xyz_a.shape != xyz_b.shape:
        raise ValueError("IRIS_PREDICTED_PRESENTATION_V1_PAIR_XYZ_SHAPE_MISMATCH")
    safe_a = _safe_xyz(xyz_a, token_valid_mask, name="PAIR_A")
    safe_b = _safe_xyz(xyz_b, token_valid_mask, name="PAIR_B")
    token_valid = token_valid_mask.bool()
    if camera_features.shape != xyz_a.shape[:2] + (13,):
        raise ValueError("IRIS_PREDICTED_PRESENTATION_V1_PAIR_CAMERA_SHAPE_INVALID")
    front_valid = _qualified_mask(
        front_valid_mask, token_valid_mask=token_valid, name="PAIR_FRONT"
    )
    occlusion_valid = _qualified_mask(
        occlusion_valid_mask, token_valid_mask=token_valid, name="PAIR_OCCLUSION"
    )
    boundary_valid = _qualified_mask(
        boundary_valid_mask, token_valid_mask=token_valid, name="PAIR_BOUNDARY"
    )
    return build_role_free_presentation_evidence_v1(
        xyz=safe_a,
        paired_xyz=safe_b,
        paired_xyz_valid=token_valid,
        kind_id=torch.full(
            xyz_a.shape[:2], PRESENTATION_KIND_PAIR, dtype=torch.long, device=xyz_a.device
        ),
        token_valid_mask=token_valid,
        camera_features=camera_features,
        camera_valid=token_valid,
        alpha_boundary=relations["projected_alpha_boundary_probability"],
        alpha_boundary_valid=boundary_valid,
        appearance_boundary=relations["projected_appearance_boundary_probability"],
        appearance_boundary_valid=boundary_valid,
        front_probability=relations["front_probability"],
        front_valid=front_valid,
        occlusion_probability=relations["occlusion_probability"],
        occlusion_valid=occlusion_valid,
    )


def ray_hit_presentation_from_predictions_v1(
    *,
    hit_xyz: torch.Tensor,
    camera_features: torch.Tensor,
    hit_valid_mask: torch.Tensor,
    global_visibility_probability: torch.Tensor,
    global_visibility_valid_mask: torch.Tensor | None = None,
) -> RoleFreePresentationEvidenceV1:
    """Type complete-stack global visibility with explicit qualification."""

    if global_visibility_probability.shape != hit_xyz.shape[:2]:
        raise ValueError("IRIS_PREDICTED_PRESENTATION_V1_RAY_VISIBILITY_SHAPE_INVALID")
    safe_xyz = _safe_xyz(hit_xyz, hit_valid_mask, name="RAY")
    hit_valid = hit_valid_mask.bool()
    visibility_valid = _qualified_mask(
        global_visibility_valid_mask,
        token_valid_mask=hit_valid,
        name="RAY_VISIBILITY",
    )
    camera = _expand_camera(
        camera_features,
        token_count=hit_xyz.shape[1],
        reference=hit_xyz,
        name="RAY",
    )
    return build_role_free_presentation_evidence_v1(
        xyz=safe_xyz,
        kind_id=torch.full(
            hit_xyz.shape[:2], PRESENTATION_KIND_RAY_HIT, dtype=torch.long, device=hit_xyz.device
        ),
        token_valid_mask=hit_valid,
        camera_features=camera,
        camera_valid=hit_valid,
        global_visibility=global_visibility_probability,
        global_visibility_valid=visibility_valid,
    )
