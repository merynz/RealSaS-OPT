from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


CANONICAL_VIEW_COUNT_V1 = 8
CANONICAL_VIEW_PROVENANCE_V1 = {
    "AUTHORED_CANONICAL",
    "SOURCE_OBSERVED",
    "GENERATED_CANONICAL",
}

DEFORMATION_MODES_V1 = (
    "SKINNED",
    "RIGID_BONE_ATTACHED",
    "RIGID_CONSTRAINED",
    "RIGID_ROOT",
    "STATIC",
    "UNKNOWN",
)

ATTACHMENT_RELATIONS_V1 = (
    "ATTACHED_TO",
    "HELD_BY",
    "CONSTRAINED_TO",
    "INDEPENDENT",
    "UNKNOWN",
)


@dataclass(frozen=True)
class CanonicalViewEvidenceSetV1:
    """Tensor contract at the Unified IRIS observation boundary.

    FIT8 Phase A uses eight authored canonical views. The future one-image
    product may replace seven entries with generated hypotheses without changing
    the downstream tensor contract. Generated views remain evidence, never source
    authority.
    """

    images: torch.Tensor  # [B, 8, C, H, W]
    camera_features: torch.Tensor  # [B, 8, Ccam]
    confidence: torch.Tensor  # [B, 8]
    provenance: tuple[str, ...]  # length 8, common across batch in V1

    @property
    def observed_source_mask(self) -> torch.Tensor:
        values = [p != "GENERATED_CANONICAL" for p in self.provenance]
        return torch.tensor(values, dtype=torch.bool, device=self.images.device)

    @property
    def generated_mask(self) -> torch.Tensor:
        return ~self.observed_source_mask


@dataclass(frozen=True)
class UnifiedFieldContractV1:
    """Stable output dimensions for the first Unified IRIS field implementation.

    The dimensions are deliberately larger than StdGEN/StdGEN++ categorical
    semantics because RealSaS must represent subject-local ownership, material
    identity and generic attachment/deformation relations across non-human
    morphology.
    """

    owner_embedding_dim: int = 64
    material_embedding_dim: int = 32
    deformation_mode_count: int = len(DEFORMATION_MODES_V1)
    attachment_relation_count: int = len(ATTACHMENT_RELATIONS_V1)
    scene_dim_floor: int = 1024
    scene_layers_floor: int = 16
    scene_heads_floor: int = 16
    triplane_resolution_floor: int = 64
    triplane_channels_floor: int = 80

    def validate(self) -> None:
        values = (
            self.owner_embedding_dim,
            self.material_embedding_dim,
            self.deformation_mode_count,
            self.attachment_relation_count,
            self.scene_dim_floor,
            self.scene_layers_floor,
            self.scene_heads_floor,
            self.triplane_resolution_floor,
            self.triplane_channels_floor,
        )
        if min(values) <= 0:
            raise ValueError("UNIFIED_IRIS_FIELD_CONTRACT_DIMENSION_INVALID")
        if self.scene_dim_floor % self.scene_heads_floor:
            raise ValueError("UNIFIED_IRIS_SCENE_DIM_NOT_DIVISIBLE_BY_HEADS")
        if self.deformation_mode_count != len(DEFORMATION_MODES_V1):
            raise ValueError("UNIFIED_IRIS_DEFORMATION_MODE_COUNT_DRIFT")
        if self.attachment_relation_count != len(ATTACHMENT_RELATIONS_V1):
            raise ValueError("UNIFIED_IRIS_ATTACHMENT_RELATION_COUNT_DRIFT")


def validate_canonical_view_evidence_set_v1(
    value: CanonicalViewEvidenceSetV1,
    *,
    require_all_authored: bool = False,
    require_at_least_one_observed: bool = True,
) -> None:
    images = value.images
    cameras = value.camera_features
    confidence = value.confidence

    if images.ndim != 5:
        raise ValueError("UNIFIED_IRIS_VIEW_IMAGES_SHAPE_INVALID")
    batch, views = images.shape[:2]
    if views != CANONICAL_VIEW_COUNT_V1:
        raise ValueError("UNIFIED_IRIS_CANONICAL_VIEW_COUNT_INVALID")
    if cameras.ndim != 3 or cameras.shape[:2] != (batch, views):
        raise ValueError("UNIFIED_IRIS_CAMERA_FEATURE_SHAPE_INVALID")
    if confidence.shape != (batch, views):
        raise ValueError("UNIFIED_IRIS_VIEW_CONFIDENCE_SHAPE_INVALID")
    if len(value.provenance) != CANONICAL_VIEW_COUNT_V1:
        raise ValueError("UNIFIED_IRIS_VIEW_PROVENANCE_COUNT_INVALID")
    if any(p not in CANONICAL_VIEW_PROVENANCE_V1 for p in value.provenance):
        raise ValueError("UNIFIED_IRIS_VIEW_PROVENANCE_INVALID")
    if require_all_authored and any(p != "AUTHORED_CANONICAL" for p in value.provenance):
        raise ValueError("UNIFIED_IRIS_FIT8_REQUIRES_AUTHORED_CANONICAL_VIEWS")
    if require_at_least_one_observed and all(p == "GENERATED_CANONICAL" for p in value.provenance):
        raise ValueError("UNIFIED_IRIS_REQUIRES_OBSERVED_SOURCE_EVIDENCE")
    if not torch.isfinite(images).all():
        raise ValueError("UNIFIED_IRIS_VIEW_IMAGES_NONFINITE")
    if not torch.isfinite(cameras).all():
        raise ValueError("UNIFIED_IRIS_CAMERA_FEATURES_NONFINITE")
    if not torch.isfinite(confidence).all():
        raise ValueError("UNIFIED_IRIS_VIEW_CONFIDENCE_NONFINITE")
    if torch.any(confidence < 0.0) or torch.any(confidence > 1.0):
        raise ValueError("UNIFIED_IRIS_VIEW_CONFIDENCE_OUT_OF_RANGE")

    # Authored/source-observed views are exact observations in this contract.
    # Confidence may still encode acquisition quality, but generated hypotheses
    # are forbidden from silently becoming source-authority observations.
    if require_all_authored and not torch.all(confidence > 0.0):
        raise ValueError("UNIFIED_IRIS_AUTHORED_VIEW_CONFIDENCE_ZERO")


def validate_preserve_classes_v1(
    preserve_classes: Sequence[int],
    class_count: int,
) -> tuple[int, ...]:
    if class_count <= 0:
        raise ValueError("UNIFIED_IRIS_SEMANTIC_CLASS_COUNT_INVALID")
    preserve = tuple(sorted(set(int(v) for v in preserve_classes)))
    if not preserve:
        raise ValueError("UNIFIED_IRIS_SEMANTIC_PRESERVE_SET_EMPTY")
    if preserve[0] < 0 or preserve[-1] >= class_count:
        raise ValueError("UNIFIED_IRIS_SEMANTIC_PRESERVE_CLASS_OUT_OF_RANGE")
    return preserve


__all__ = [
    "ATTACHMENT_RELATIONS_V1",
    "CANONICAL_VIEW_COUNT_V1",
    "CANONICAL_VIEW_PROVENANCE_V1",
    "DEFORMATION_MODES_V1",
    "CanonicalViewEvidenceSetV1",
    "UnifiedFieldContractV1",
    "validate_canonical_view_evidence_set_v1",
    "validate_preserve_classes_v1",
]
