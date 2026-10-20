from __future__ import annotations

"""Typed role-free presentation evidence for structuralization.

Inference-side evidence only. Named semantic classes and teacher topology are
not inputs. Optional evidence is always paired with explicit validity so unknown
abstains instead of silently becoming a negative label.

Pair tokens preserve both structural anchors. View-dependent evidence also
preserves the exact admitted 13-D camera feature contract used by Relation V3;
a topology decoder must not be asked to interpret front/visibility evidence
without knowing the view that made the statement meaningful.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


PRESENTATION_KIND_SURFACE = 0
PRESENTATION_KIND_PAIR = 1
PRESENTATION_KIND_RAY_HIT = 2
PRESENTATION_KIND_COUNT = 3

PRESENTATION_FEATURE_NAMES_V1 = (
    "xyz_a_x", "xyz_a_y", "xyz_a_z",
    "xyz_b_x", "xyz_b_y", "xyz_b_z", "xyz_b_valid",
    "camera_origin_x", "camera_origin_y", "camera_origin_z",
    "camera_right_x", "camera_right_y", "camera_right_z",
    "camera_up_x", "camera_up_y", "camera_up_z",
    "camera_forward_x", "camera_forward_y", "camera_forward_z",
    "camera_half_extent_ratio", "camera_valid",
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


def _typed_probability(
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
    if bool(valid_bool.any()):
        admitted = value[valid_bool]
        if bool(((admitted < 0.0) | (admitted > 1.0)).any()):
            raise ValueError(f"IRIS_PRESENTATION_V1_{name.upper()}_PROBABILITY_RANGE_INVALID")
    typed = torch.where(valid_bool, value.to(dtype=reference.dtype), torch.zeros_like(value, dtype=reference.dtype))
    return typed, valid_bool.to(dtype=reference.dtype)


def build_role_free_presentation_evidence_v1(
    *,
    xyz: torch.Tensor,
    kind_id: torch.Tensor,
    token_valid_mask: torch.Tensor,
    paired_xyz: torch.Tensor | None = None,
    paired_xyz_valid: torch.Tensor | None = None,
    camera_features: torch.Tensor | None = None,
    camera_valid: torch.Tensor | None = None,
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
    if (paired_xyz is None) != (paired_xyz_valid is None):
        raise ValueError("IRIS_PRESENTATION_V1_PAIRED_XYZ_VALUE_VALIDITY_PAIR_REQUIRED")
    if paired_xyz is None:
        xyz_b = xyz.new_zeros(shape + (3,))
        xyz_b_valid = xyz.new_zeros(shape)
    else:
        if paired_xyz.shape != shape + (3,) or paired_xyz_valid.shape != shape:
            raise ValueError("IRIS_PRESENTATION_V1_PAIRED_XYZ_SHAPE_INVALID")
        if not torch.is_floating_point(paired_xyz) or not bool(torch.isfinite(paired_xyz).all()):
            raise ValueError("IRIS_PRESENTATION_V1_PAIRED_XYZ_NUMERIC_INVALID")
        valid_b = paired_xyz_valid.bool()
        if bool(valid_b.any()) and float(paired_xyz[valid_b].detach().abs().max()) > 0.500001:
            raise ValueError("IRIS_PRESENTATION_V1_PAIRED_XYZ_DOMAIN_DRIFT")
        xyz_b = torch.where(valid_b[..., None], paired_xyz.to(dtype=xyz.dtype), torch.zeros_like(paired_xyz, dtype=xyz.dtype))
        xyz_b_valid = valid_b.to(dtype=xyz.dtype)

    if (camera_features is None) != (camera_valid is None):
        raise ValueError("IRIS_PRESENTATION_V1_CAMERA_VALUE_VALIDITY_PAIR_REQUIRED")
    if camera_features is None:
        camera = xyz.new_zeros(shape + (13,))
        camera_valid_t = xyz.new_zeros(shape)
        camera_valid_b = torch.zeros(shape, dtype=torch.bool, device=xyz.device)
    else:
        if camera_features.shape != shape + (13,) or camera_valid.shape != shape:
            raise ValueError("IRIS_PRESENTATION_V1_CAMERA_SHAPE_INVALID")
        if not torch.is_floating_point(camera_features) or not bool(torch.isfinite(camera_features).all()):
            raise ValueError("IRIS_PRESENTATION_V1_CAMERA_NUMERIC_INVALID")
        camera_valid_b = camera_valid.bool()
        camera = torch.where(camera_valid_b[..., None], camera_features.to(dtype=xyz.dtype), torch.zeros_like(camera_features, dtype=xyz.dtype))
        camera_valid_t = camera_valid_b.to(dtype=xyz.dtype)

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
        valid_rgba = source_rgba_valid.bool()
        if bool(valid_rgba.any()):
            admitted = source_rgba[valid_rgba]
            if bool(((admitted < 0.0) | (admitted > 1.0)).any()):
                raise ValueError("IRIS_PRESENTATION_V1_SOURCE_RGBA_RANGE_INVALID")
        rgba = torch.where(valid_rgba[..., None], source_rgba.to(dtype=xyz.dtype), torch.zeros_like(source_rgba, dtype=xyz.dtype))
        rgba_valid = valid_rgba.to(dtype=xyz.dtype)

    scalar_specs = (
        (alpha_boundary, alpha_boundary_valid, "alpha_boundary"),
        (appearance_boundary, appearance_boundary_valid, "appearance_boundary"),
        (front_probability, front_valid, "front"),
        (occlusion_probability, occlusion_valid, "occlusion"),
        (global_visibility, global_visibility_valid, "global_visibility"),
        (deformation_coherence, deformation_coherence_valid, "deformation_coherence"),
        (attachment_persistence, attachment_persistence_valid, "attachment_persistence"),
        (source_support, source_support_valid, "source_support"),
    )
    scalar_pairs = tuple(
        _typed_probability(reference=xyz, value=value, valid=valid, name=name)
        for value, valid, name in scalar_specs
    )
    valid_by_name = {
        name: (valid.bool() if valid is not None else torch.zeros(shape, dtype=torch.bool, device=xyz.device))
        for _, valid, name in scalar_specs
    }

    pair_kind = kind_id == PRESENTATION_KIND_PAIR
    if bool((pair_kind & token_valid_mask.bool() & ~paired_xyz_valid.bool()).any()) if paired_xyz_valid is not None else bool((pair_kind & token_valid_mask.bool()).any()):
        raise ValueError("IRIS_PRESENTATION_V1_PAIR_REQUIRES_SECOND_ANCHOR")

    view_dependent = (
        valid_by_name["alpha_boundary"]
        | valid_by_name["appearance_boundary"]
        | valid_by_name["front"]
        | valid_by_name["occlusion"]
        | valid_by_name["global_visibility"]
    )
    if bool((view_dependent & ~camera_valid_b).any()):
        raise ValueError("IRIS_PRESENTATION_V1_VIEW_EVIDENCE_REQUIRES_CAMERA")

    kind = F.one_hot(kind_id.to(dtype=torch.long), num_classes=PRESENTATION_KIND_COUNT).to(dtype=xyz.dtype)
    fields = [xyz, xyz_b, xyz_b_valid[..., None], camera, camera_valid_t[..., None], rgba, rgba_valid[..., None]]
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
    """Matched-capacity control input for the future A arm."""
    if batch_size <= 0:
        raise ValueError("IRIS_PRESENTATION_V1_NULL_BATCH_NONPOSITIVE")
    values = torch.zeros(batch_size, 1, len(PRESENTATION_FEATURE_NAMES_V1), device=device, dtype=dtype)
    mask = torch.ones(batch_size, 1, device=device, dtype=torch.bool)
    kind_id = torch.zeros(batch_size, 1, device=device, dtype=torch.long)
    return RoleFreePresentationEvidenceV1(values=values, valid_mask=mask, kind_id=kind_id)
