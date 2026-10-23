from __future__ import annotations

"""Representation-neutral influence/simplex qualification.

This is the single numeric court used by both the historical carrier-native Skin
qualification and the product CanonicalSkin qualification. Callers own identity
mapping and authority semantics; this module owns only domain coverage, joint
legality, non-negativity and bounded simplex normalization.
"""

from dataclasses import dataclass
import math
from typing import Iterable

from .types import QualificationError


@dataclass(frozen=True)
class QualifiedInfluenceRowV1:
    domain_id: str
    influences: tuple[tuple[str, float], ...]
    simplex_residual_before: float
    correction_l1: float


@dataclass(frozen=True)
class InfluenceSimplexQualificationV1:
    rows: tuple[QualifiedInfluenceRowV1, ...]
    max_simplex_residual_before: float
    max_row_correction_l1: float
    total_correction_l1: float


def qualify_influence_simplex_v1(
    *,
    ordered_domain_ids: Iterable[str],
    allowed_joint_ids: Iterable[str],
    influences: Iterable[tuple[str, str, float]],
    max_simplex_repair_l1: float,
    max_total_correction_l1: float,
    negative_tolerance: float,
    max_influences: int | None = None,
    error_prefix: str = "SKIN",
) -> InfluenceSimplexQualificationV1:
    domain_ids = tuple(map(str, ordered_domain_ids))
    if not domain_ids or len(set(domain_ids)) != len(domain_ids):
        raise QualificationError(f"{error_prefix}_EMPTY_OR_DUPLICATE_DOMAIN")
    joint_ids = {str(value) for value in allowed_joint_ids}
    if not joint_ids:
        raise QualificationError(f"{error_prefix}_EMPTY_JOINT_DOMAIN")
    if max_simplex_repair_l1 < 0.0 or max_total_correction_l1 < 0.0 or negative_tolerance < 0.0:
        raise QualificationError(f"{error_prefix}_QUALIFICATION_BUDGET_INVALID")
    if max_influences is not None and int(max_influences) <= 0:
        raise QualificationError(f"{error_prefix}_MAX_INFLUENCES_INVALID")

    domain_set = set(domain_ids)
    grouped: dict[str, dict[str, float]] = {domain_id: {} for domain_id in domain_ids}
    for raw_domain_id, raw_joint_id, raw_weight in influences:
        domain_id = str(raw_domain_id)
        joint_id = str(raw_joint_id)
        weight = float(raw_weight)
        if domain_id not in domain_set:
            raise QualificationError(f"{error_prefix}_UNKNOWN_DOMAIN_ID")
        if joint_id not in joint_ids:
            raise QualificationError(f"{error_prefix}_UNKNOWN_JOINT")
        if joint_id in grouped[domain_id]:
            raise QualificationError(f"{error_prefix}_DUPLICATE_DOMAIN_JOINT")
        if not math.isfinite(weight) or weight < -float(negative_tolerance):
            raise QualificationError(f"{error_prefix}_WEIGHT_INVALID")
        grouped[domain_id][joint_id] = max(0.0, weight)

    rows: list[QualifiedInfluenceRowV1] = []
    total_correction = 0.0
    max_row_correction = 0.0
    max_residual = 0.0
    for domain_id in domain_ids:
        values = grouped[domain_id]
        if not values:
            raise QualificationError(f"{error_prefix}_DOMAIN_UNCOVERED")
        ordered = sorted(values.items())
        raw_sum = sum(weight for _, weight in ordered)
        if not math.isfinite(raw_sum) or raw_sum <= 0.0:
            raise QualificationError(f"{error_prefix}_ROW_ZERO")
        residual = abs(raw_sum - 1.0)
        normalized = tuple((joint_id, float(weight / raw_sum)) for joint_id, weight in ordered)
        correction = sum(
            abs(after - before)
            for (_, before), (_, after) in zip(ordered, normalized)
        )
        if correction > float(max_simplex_repair_l1) + 1e-15:
            raise QualificationError(f"{error_prefix}_ROW_REPAIR_BUDGET_EXCEEDED")
        if max_influences is not None:
            positive = sum(weight > 0.0 for _, weight in normalized)
            if positive > int(max_influences):
                raise QualificationError(f"{error_prefix}_MAX_INFLUENCES_EXCEEDED")
        total_correction += correction
        max_row_correction = max(max_row_correction, correction)
        max_residual = max(max_residual, residual)
        rows.append(
            QualifiedInfluenceRowV1(
                domain_id=domain_id,
                influences=normalized,
                simplex_residual_before=float(residual),
                correction_l1=float(correction),
            )
        )
    if total_correction > float(max_total_correction_l1) + 1e-15:
        raise QualificationError(f"{error_prefix}_TOTAL_REPAIR_BUDGET_EXCEEDED")

    return InfluenceSimplexQualificationV1(
        rows=tuple(rows),
        max_simplex_residual_before=float(max_residual),
        max_row_correction_l1=float(max_row_correction),
        total_correction_l1=float(total_correction),
    )


__all__ = [
    "InfluenceSimplexQualificationV1",
    "QualifiedInfluenceRowV1",
    "qualify_influence_simplex_v1",
]
