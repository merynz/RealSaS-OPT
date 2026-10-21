from __future__ import annotations

"""Typed presentation evidence from admitted source observations.

Unlike neural prediction bridges, this module carries only values observed in the
source views and explicit observation validity. 3-D anchors must already come from
an inference-admissible model surface; this module does not accept teacher geometry.
"""

import torch

from .presentation_evidence_v1 import (
    PRESENTATION_KIND_SURFACE,
    RoleFreePresentationEvidenceV1,
    build_role_free_presentation_evidence_v1,
)


SOURCE_OBSERVATION_PRESENTATION_PROVENANCE_V1 = {
    "schema": "RealSaS.SourceObservationPresentationProvenance.v1",
    "anchor_authority": "MODEL_PRODUCED_SURFACE_ONLY",
    "rgba_authority": "ADMITTED_SOURCE_VIEW",
    "boundary_authority": "DETERMINISTIC_SOURCE_RGBA_DERIVATION",
    "teacher_geometry_allowed": False,
    "teacher_topology_allowed": False,
    "named_semantic_taxonomy_allowed": False,
}


def surface_presentation_from_source_observations_v1(
    *,
    xyz: torch.Tensor,
    camera_features: torch.Tensor,
    token_valid_mask: torch.Tensor,
    source_rgba: torch.Tensor,
    source_rgba_valid: torch.Tensor,
    alpha_boundary: torch.Tensor,
    appearance_boundary: torch.Tensor,
    boundary_valid: torch.Tensor,
    source_support: torch.Tensor,
    source_support_valid: torch.Tensor,
) -> RoleFreePresentationEvidenceV1:
    """Build surface-kind evidence without a neural appearance authority upgrade."""

    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError("IRIS_SOURCE_PRESENTATION_V1_XYZ_SHAPE_INVALID")
    shape = xyz.shape[:2]
    if token_valid_mask.shape != shape:
        raise ValueError("IRIS_SOURCE_PRESENTATION_V1_TOKEN_MASK_SHAPE_INVALID")
    if camera_features.shape != shape + (13,):
        raise ValueError("IRIS_SOURCE_PRESENTATION_V1_CAMERA_SHAPE_INVALID")
    if source_rgba.shape != shape + (4,):
        raise ValueError("IRIS_SOURCE_PRESENTATION_V1_RGBA_SHAPE_INVALID")
    for name, value in (
        ("SOURCE_RGBA", source_rgba_valid),
        ("ALPHA_BOUNDARY", alpha_boundary),
        ("APPEARANCE_BOUNDARY", appearance_boundary),
        ("BOUNDARY_VALID", boundary_valid),
        ("SOURCE_SUPPORT", source_support),
        ("SOURCE_SUPPORT_VALID", source_support_valid),
    ):
        if value.shape != shape:
            raise ValueError(f"IRIS_SOURCE_PRESENTATION_V1_{name}_SHAPE_INVALID")

    token_valid = token_valid_mask.bool()
    rgba_valid = source_rgba_valid.bool() & token_valid
    boundary_valid_t = boundary_valid.bool() & token_valid
    support_valid = source_support_valid.bool() & token_valid
    safe_xyz = torch.where(token_valid[..., None], xyz, torch.zeros_like(xyz))
    safe_camera = torch.where(
        token_valid[..., None], camera_features, torch.zeros_like(camera_features)
    )
    return build_role_free_presentation_evidence_v1(
        xyz=safe_xyz,
        kind_id=torch.full(
            shape,
            PRESENTATION_KIND_SURFACE,
            dtype=torch.long,
            device=xyz.device,
        ),
        token_valid_mask=token_valid,
        camera_features=safe_camera,
        camera_valid=token_valid,
        source_rgba=source_rgba,
        source_rgba_valid=rgba_valid,
        alpha_boundary=alpha_boundary,
        alpha_boundary_valid=boundary_valid_t,
        appearance_boundary=appearance_boundary,
        appearance_boundary_valid=boundary_valid_t,
        source_support=source_support,
        source_support_valid=support_valid,
    )
