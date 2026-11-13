from __future__ import annotations

"""Validity-aware role-free Graphics truth with explicit conditioning semantics.

Historical V1-era supervision mixes useful but differently authoritative proxies
and also mixes view-invariant pair relations with raster-view boundary evidence.
V2 makes both distinctions explicit:

- static truth contains only relations that are intended to be view-invariant;
- projected alpha/appearance boundaries live in the view-conditioned truth row;
- every scalar target carries authority, provenance and trainability;
- unavailable truth abstains instead of becoming an implicit negative.

Historical V8 shards are not reinterpreted by this schema.
"""

from dataclasses import dataclass
from typing import Mapping

from .boundary_causality_v1 import EvidenceAuthority


_AUTHORITATIVE = frozenset(
    {
        EvidenceAuthority.RAW_AUTHORED,
        EvidenceAuthority.DETERMINISTIC_DERIVED,
        EvidenceAuthority.QUALIFIED_DERIVED,
    }
)

STATIC_RELATION_KEYS_V2 = (
    "same_drawable_affinity",
    "cut_probability",
    "attachment_affinity",
)

VIEW_RELATION_KEYS_V2 = (
    "front_probability",
    "occlusion_probability",
    "visibility_probability_a",
    "visibility_probability_b",
    "projected_alpha_boundary_probability",
    "projected_appearance_boundary_probability",
)

ALL_RELATION_KEYS_V2 = STATIC_RELATION_KEYS_V2 + VIEW_RELATION_KEYS_V2


@dataclass(frozen=True)
class QualifiedProbabilityTargetV2:
    value: float
    authority: EvidenceAuthority
    provenance: str
    valid_for_training: bool

    def __post_init__(self) -> None:
        value = float(self.value)
        if not 0.0 <= value <= 1.0:
            raise ValueError("GRAPHICS_TRUTH_V2_PROBABILITY_INVALID")
        if not self.provenance or not self.provenance.strip():
            raise ValueError("GRAPHICS_TRUTH_V2_PROVENANCE_REQUIRED")
        if self.valid_for_training and self.authority not in _AUTHORITATIVE:
            raise ValueError("GRAPHICS_TRUTH_V2_DIAGNOSTIC_TARGET_CANNOT_TRAIN")

    @classmethod
    def abstain(cls, *, provenance: str) -> "QualifiedProbabilityTargetV2":
        return cls(
            value=0.0,
            authority=EvidenceAuthority.DIAGNOSTIC_ONLY,
            provenance=provenance,
            valid_for_training=False,
        )


@dataclass(frozen=True)
class GraphicsStaticTruthRowV2:
    """View-invariant structural relation truth only."""

    face_a: int
    face_b: int
    same_drawable_affinity: QualifiedProbabilityTargetV2
    cut_probability: QualifiedProbabilityTargetV2
    attachment_affinity: QualifiedProbabilityTargetV2
    schema_version: str = "RealSaS.GraphicsStaticTruthRow.v2"

    def __post_init__(self) -> None:
        if self.face_a < 0 or self.face_b < 0 or self.face_a == self.face_b:
            raise ValueError("GRAPHICS_TRUTH_V2_FACE_PAIR_INVALID")


@dataclass(frozen=True)
class GraphicsViewTruthRowV2:
    """View-conditioned projected relation truth.

    Projected alpha/appearance boundaries belong here because their historical
    producer is a raster-view neighborhood measurement. They must not be routed
    to a camera-free static prediction branch.
    """

    view_index: int
    face_a: int
    face_b: int
    front_probability: QualifiedProbabilityTargetV2
    occlusion_probability: QualifiedProbabilityTargetV2
    visibility_probability_a: QualifiedProbabilityTargetV2
    visibility_probability_b: QualifiedProbabilityTargetV2
    projected_alpha_boundary_probability: QualifiedProbabilityTargetV2
    projected_appearance_boundary_probability: QualifiedProbabilityTargetV2
    schema_version: str = "RealSaS.GraphicsViewTruthRow.v2"

    def __post_init__(self) -> None:
        if self.view_index < 0 or self.face_a < 0 or self.face_b < 0 or self.face_a == self.face_b:
            raise ValueError("GRAPHICS_TRUTH_V2_VIEW_ROW_INVALID")


def _row_to_training_maps_v2(row, names: tuple[str, ...]) -> tuple[dict[str, float], dict[str, bool]]:
    values: dict[str, float] = {}
    validity: dict[str, bool] = {}
    for name in names:
        target = getattr(row, name)
        values[name] = float(target.value)
        validity[name] = bool(target.valid_for_training)
    return values, validity


def static_row_to_training_maps_v2(
    row: GraphicsStaticTruthRowV2,
) -> tuple[dict[str, float], dict[str, bool]]:
    return _row_to_training_maps_v2(row, STATIC_RELATION_KEYS_V2)


def view_row_to_training_maps_v2(
    row: GraphicsViewTruthRowV2,
) -> tuple[dict[str, float], dict[str, bool]]:
    return _row_to_training_maps_v2(row, VIEW_RELATION_KEYS_V2)


def assert_required_validity_keys_v2(
    truth: Mapping[str, object], validity: Mapping[str, object], *, required: tuple[str, ...]
) -> None:
    missing_truth = [name for name in required if name not in truth]
    missing_validity = [name for name in required if name not in validity]
    if missing_truth:
        raise ValueError("GRAPHICS_TRUTH_V2_VALUE_MISSING:" + ",".join(missing_truth))
    if missing_validity:
        raise ValueError("GRAPHICS_TRUTH_V2_VALIDITY_MISSING:" + ",".join(missing_validity))
