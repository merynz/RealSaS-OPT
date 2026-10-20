from __future__ import annotations

"""Typed role-free presentation evidence for structuralization.

This module contains inference-side evidence only.  It never accepts named
semantic classes or teacher topology.  Each optional scalar family is paired
with explicit validity, so unavailable evidence abstains instead of silently
becoming a negative label.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


PRESENTATION_KIND_SURFACE = 0
PRESENTATION_KIND_PAIR = 1
PRESENTATION_KIND_RAY_HIT = 2
PRESENTATION_KIND_COUNT = 3

PRESENTATION_FEATURE_NAMES_V1 = (
    "xyz_x", "xyz_y", "xyz_z",
    "source_r", "source_g", "source_b", "source_a", "source_rgba_valid",
    "alpha_boundary", "alpha_boundary_valid",
    "appearance_boundary", "appearance_boundary_valid",
    "front_probability", "front_valid",
    "occlusion_probability", "occlusion_valid",
    "global_visibility", "global_visibility_valid",
    "deformation_coherence", "deformation_coherence_valid",
    "attachment_persistence", "attachment_persistence_valid",
    "source_support", "source_support_valid",
    "kind_surface", "kind_pair", "kind_ray_hit",
)


@dataclass(frozen=True)
class RoleFreePresentationEvidenceV1:
    values: torch.Tensor
    valid_mask: torch.Tensor
    kind_id: torch.Tensor
    schema_version: str = "RealSaS.RoleFreePresentationEvidence.v1"

    def __post_init__(self) -> None:
        if self.values.ndim != 3 or self.values.shape[-1] != len(PRESENTATION_FEATURE_NAMES_V1):
            raise ValueError("IRIS_PRESENTATION_V1_VALUES_SHAPE_INVALID")
        if self.valid_mask.shape != self.values.shape[:2]:
            raise ValueError("IRIS_PRESENTATION_V1_VALID_MASK_SHAPE_INVALID")
        if self.kind_id.shape != self.values.shape[:2]:
            raise ValueError("IRIS_PRESENTATION_V1_KIND_SHAPE_INVALID")
        if not torch.is_floating_point(self.values) or not bool(torch.isfinite(self.values).all()):
            raise ValueError("IRIS_PRESENTATION_V1_VALUES_NUMERIC_INVALID")
        if self.kind_id.numel() and (
            int(self.kind_id.min()) < 0 or int(self.kind_id.max()) >= PRESENTATION_KIND_COUNT
        ):
            raise ValueError("IRIS_PRESENTATION_V1_KIND_INVALID")
        if not bool(self.valid_mask.any(dim=1).all()):
            raise ValueError("IRIS_PRESENTATION_V1_SAMPLE_HAS_NO_VALID_TOKEN")

    @property
    def input_dim(self) -> int:
        return self.values.shape[-1]



def _typed_scalar(
    *,
    reference: torch.Tensor,
    value: torch.Tensor | None,
    valid: torch.Tensor | None,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = reference.shape[:2]
    if (value is None) != (valid is None):
        raise ValueError(f"IRIS_PRESENTATION_V1_{name.upper()}_VALUE_VALIDITY_PAIR_REQUIRED")
    if value is None:
        return reference.new_zeros(shape), reference.new_zeros(shape)
    if value.shape != shape or valid.shape != shape:
        raise ValueError(f"IRIS_PRESENTATION_V1_{name.upper()}_SHAPE_INVALID")
    if not torch.is_floating_point(value) or not bool(torch.isfinite(value).all()):
        raise ValueError(f"IRIS_PRESENTATION_V1_{name.upper()}_NUMERIC_INVALID")
    valid_bool = valid.bool()
    typed = torch.where(valid_bool, value.to(dtype=reference.dtype), torch.zeros_like(value, dtype=reference.dtype))
    return typed, valid_bool.to(dtype=reference.dtype)



def build_role_free_presentation_evidence_v1(
    *,
    xyz: torch.Tensor,
    kind_id: torch.Tensor,
    token_valid_mask: torch.Tensor,
    source_rgba: torch.Tensor | None = None,
    source_rgba_valid: torch.Tensor | None = None,
    alpha_boundary: torch.Tensor | None = None,
    alpha_boundary_valid: torch.Tensor | None = None,
    appearance_boundary: torch.Tensor | None = None,
    appearance_boundary_valid: torch.Tensor | None = None,
    front_probability: torch.Tensor | None = None,
    front_valid: torch.Tensor | None = None,
    occlusion_probability: torch.Tensor | None = None,
    occlusion_valid: torch.Tensor | None = None,
    global_visibility: torch.Tensor | None = None,
    global_visibility_valid: torch.Tensor | None = None,
    deformation_coherence: torch.Tensor | None = None,
    deformation_coherence_valid: torch.Tensor | None = None,
    attachment_persistence: torch.Tensor | None = None,
    attachment_persistence_valid: torch.Tensor | None = None,
    source_support: torch.Tensor | None = None,
    source_support_valid: torch.Tensor | None = None,
) -> RoleFreePresentationEvidenceV1:
    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError("IRIS_PRESENTATION_V1_XYZ_SHAPE_INVALID")
    if not torch.is_floating_point(xyz) or not bool(torch.isfinite(xyz).all()):
        raise ValueError("IRIS_PRESENTATION_V1_XYZ_NUMERIC_INVALID")
    if xyz.numel() and float(xyz.detach().abs().max()) > 0.500001:
        raise ValueError("IRIS_PRESENTATION_V1_XYZ_DOMAIN_DRIFT")
    if kind_id.shape != xyz.shape[:2] or token_valid_mask.shape != xyz.shape[:2]:
        raise ValueError("IRIS_PRESENTATION_V1_TOKEN_METADATA_SHAPE_INVALID")
    if kind_id.numel() and (int(kind_id.min()) < 0 or int(kind_id.max()) >= PRESENTATION_KIND_COUNT):
        raise ValueError("IRIS_PRESENTATION_V1_KIND_INVALID")

    shape = xyz.shape[:2]
    if (source_rgba is None) != (source_rgba_valid is None):
        raise ValueError("IRIS_PRESENTATION_V1_SOURCE_RGBA_VALUE_VALIDITY_PAIR_REQUIRED")
    if source_rgba is None:
        rgba = xyz.new_zeros(shape + (4,))
        rgba_valid = xyz.new_zeros(shape)
    else:
        if source_rgba.shape != shape + (4,) or source_rgba_valid.shape != shape:
            raise ValueError("IRIS_PRESENTATION_V1_SOURCE_RGBA_SHAPE_INVALID")
        if not torch.is_floating_point(source_rgba) or not bool(torch.isfinite(source_rgba).all()):
            raise ValueError("IRIS_PRESENTATION_V1_SOURCE_RGBA_NUMERIC_INVALID")
        valid = source_rgba_valid.bool()
        rgba = torch.where(valid[..., None], source_rgba.to(dtype=xyz.dtype), torch.zeros_like(source_rgba, dtype=xyz.dtype))
        rgba_valid = valid.to(dtype=xyz.dtype)

    scalar_pairs = (
        _typed_scalar(reference=xyz, value=alpha_boundary, valid=alpha_boundary_valid, name="alpha_boundary"),
        _typed_scalar(reference=xyz, value=appearance_boundary, valid=appearance_boundary_valid, name="appearance_boundary"),
        _typed_scalar(reference=xyz, value=front_probability, valid=front_valid, name="front"),
        _typed_scalar(reference=xyz, value=occlusion_probability, valid=occlusion_valid, name="occlusion"),
        _typed_scalar(reference=xyz, value=global_visibility, valid=global_visibility_valid, name="global_visibility"),
        _typed_scalar(reference=xyz, value=deformation_coherence, valid=deformation_coherence_valid, name="deformation_coherence"),
        _typed_scalar(reference=xyz, value=attachment_persistence, valid=attachment_persistence_valid, name="attachment_persistence"),
        _typed_scalar(reference=xyz, value=source_support, valid=source_support_valid, name="source_support"),
    )
    kind = F.one_hot(kind_id.to(dtype=torch.long), num_classes=PRESENTATION_KIND_COUNT).to(dtype=xyz.dtype)
    fields = [xyz, rgba, rgba_valid[..., None]]
    for value, valid in scalar_pairs:
        fields.extend((value[..., None], valid[..., None]))
    fields.append(kind)
    values = torch.cat(fields, dim=-1)
    if values.shape[-1] != len(PRESENTATION_FEATURE_NAMES_V1):
        raise AssertionError("IRIS_PRESENTATION_V1_FEATURE_DIM_DRIFT")
    valid_mask = token_valid_mask.bool()
    values = torch.where(valid_mask[..., None], values, torch.zeros_like(values))
    return RoleFreePresentationEvidenceV1(values=values, valid_mask=valid_mask, kind_id=kind_id)



def null_presentation_evidence_v1(
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> RoleFreePresentationEvidenceV1:
    """Matched-capacity control input for the future A arm.

    The presentation memory module still exists and has identical parameters;
    its single admitted source token contains no presentation information.
    """
    if batch_size <= 0:
        raise ValueError("IRIS_PRESENTATION_V1_NULL_BATCH_NONPOSITIVE")
    values = torch.zeros(batch_size, 1, len(PRESENTATION_FEATURE_NAMES_V1), device=device, dtype=dtype)
    mask = torch.ones(batch_size, 1, device=device, dtype=torch.bool)
    kind_id = torch.zeros(batch_size, 1, device=device, dtype=torch.long)
    return RoleFreePresentationEvidenceV1(values=values, valid_mask=mask, kind_id=kind_id)
