from __future__ import annotations

"""Validity-aware role-free Graphics truth.

V1-era supervision mixes several useful but differently authoritative proxies:
`graphics_truth_v1.py` can derive cut as the complement of anonymous owner
identity, while the active Fit8 shard builder writes a broader owner-or-material
boundary proxy into `relation_cut_probability`.  V2 does not reinterpret either
as causal cut truth.  Every scalar target carries explicit authority, provenance
and trainability; unavailable truth abstains.
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
    face_a: int
    face_b: int
    same_drawable_affinity: QualifiedProbabilityTargetV2
    cut_probability: QualifiedProbabilityTargetV2
    attachment_affinity: QualifiedProbabilityTargetV2
    alpha_boundary_probability: QualifiedProbabilityTargetV2
    appearance_boundary_probability: QualifiedProbabilityTargetV2
    schema_version: str = "RealSaS.GraphicsStaticTruthRow.v2"

    def __post_init__(self) -> None:
        if self.face_a < 0 or self.face_b < 0 or self.face_a == self.face_b:
            raise ValueError("GRAPHICS_TRUTH_V2_FACE_PAIR_INVALID")


@dataclass(frozen=True)
class GraphicsViewTruthRowV2:
    view_index: int
    face_a: int
    face_b: int
    front_probability: QualifiedProbabilityTargetV2
    occlusion_probability: QualifiedProbabilityTargetV2
    visibility_probability_a: QualifiedProbabilityTargetV2
    visibility_probability_b: QualifiedProbabilityTargetV2
    schema_version: str = "RealSaS.GraphicsViewTruthRow.v2"

    def __post_init__(self) -> None:
        if self.view_index < 0 or self.face_a < 0 or self.face_b < 0 or self.face_a == self.face_b:
            raise ValueError("GRAPHICS_TRUTH_V2_VIEW_ROW_INVALID")


def static_row_to_training_maps_v2(
    row: GraphicsStaticTruthRowV2,
) -> tuple[dict[str, float], dict[str, bool]]:
    names = (
        "same_drawable_affinity",
        "cut_probability",
        "attachment_affinity",
        "alpha_boundary_probability",
        "appearance_boundary_probability",
    )
    values: dict[str, float] = {}
    validity: dict[str, bool] = {}
    for name in names:
        target = getattr(row, name)
        values[name] = float(target.value)
        validity[name] = bool(target.valid_for_training)
    return values, validity


def view_row_to_training_maps_v2(
    row: GraphicsViewTruthRowV2,
) -> tuple[dict[str, float], dict[str, bool]]:
    names = (
        "front_probability",
        "occlusion_probability",
        "visibility_probability_a",
        "visibility_probability_b",
    )
    values: dict[str, float] = {}
    validity: dict[str, bool] = {}
    for name in names:
        target = getattr(row, name)
        values[name] = float(target.value)
        validity[name] = bool(target.valid_for_training)
    return values, validity


def assert_required_validity_keys_v2(
    truth: Mapping[str, object], validity: Mapping[str, object], *, required: tuple[str, ...]
) -> None:
    missing_truth = [name for name in required if name not in truth]
    missing_validity = [name for name in required if name not in validity]
    if missing_truth:
        raise ValueError("GRAPHICS_TRUTH_V2_VALUE_MISSING:" + ",".join(missing_truth))
    if missing_validity:
        raise ValueError("GRAPHICS_TRUTH_V2_VALIDITY_MISSING:" + ",".join(missing_validity))
