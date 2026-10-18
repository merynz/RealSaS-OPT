from __future__ import annotations

"""Court-support qualification for relation-truth V2 manifests.

Truth authority and class support are separate obligations.  A deterministic
producer can be perfectly correct yet still provide only one class (for example,
the current same-ray occlusion gate emits positive rows only).  Such evidence may
be useful for training/telemetry, but it is not eligible for a binary held-out
PASS court until authoritative support for both classes exists.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping

from .graphics_truth_v2 import STATIC_RELATION_KEYS_V2, VIEW_RELATION_KEYS_V2


class BinarySupportStatusV2(str, Enum):
    NO_VALID_SUPPORT = "NO_VALID_SUPPORT"
    SINGLE_CLASS_SUPPORT = "SINGLE_CLASS_SUPPORT"
    BINARY_SUPPORT = "BINARY_SUPPORT"


@dataclass(frozen=True)
class BinarySupportQualificationV2:
    status: BinarySupportStatusV2
    valid: int
    positive: int
    negative: int
    court_eligible: bool


def _validated_counts(stats: Mapping[str, object]) -> tuple[int, int, int]:
    try:
        valid = int(stats["valid"])
        positive = int(stats["positive"])
        negative = int(stats["negative"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("RELATION_TRUTH_V2_SUPPORT_STATS_INCOMPLETE") from exc
    if min(valid, positive, negative) < 0 or positive + negative != valid:
        raise ValueError("RELATION_TRUTH_V2_SUPPORT_STATS_INCONSISTENT")
    return valid, positive, negative


def qualify_binary_support_v2(
    stats: Mapping[str, object],
    *,
    minimum_valid: int = 2,
) -> BinarySupportQualificationV2:
    if int(minimum_valid) < 2:
        raise ValueError("RELATION_TRUTH_V2_MINIMUM_VALID_MUST_BE_AT_LEAST_TWO")
    valid, positive, negative = _validated_counts(stats)
    if valid == 0:
        status = BinarySupportStatusV2.NO_VALID_SUPPORT
    elif positive == 0 or negative == 0:
        status = BinarySupportStatusV2.SINGLE_CLASS_SUPPORT
    else:
        status = BinarySupportStatusV2.BINARY_SUPPORT
    return BinarySupportQualificationV2(
        status=status,
        valid=valid,
        positive=positive,
        negative=negative,
        court_eligible=(
            status is BinarySupportStatusV2.BINARY_SUPPORT
            and valid >= int(minimum_valid)
        ),
    )


def aggregate_binary_stats_v2(rows: Iterable[Mapping[str, object]]) -> dict[str, int | float]:
    valid = positive = negative = total = abstained = 0
    seen = False
    for row in rows:
        seen = True
        row_valid, row_positive, row_negative = _validated_counts(row)
        row_total = int(row.get("total", row_valid))
        row_abstained = int(row.get("abstained", row_total - row_valid))
        if row_total < row_valid or row_abstained < 0 or row_valid + row_abstained != row_total:
            raise ValueError("RELATION_TRUTH_V2_SUPPORT_TOTAL_INCONSISTENT")
        valid += row_valid
        positive += row_positive
        negative += row_negative
        total += row_total
        abstained += row_abstained
    if not seen:
        return {
            "total": 0,
            "valid": 0,
            "abstained": 0,
            "positive": 0,
            "negative": 0,
            "positive_fraction": 0.0,
        }
    return {
        "total": total,
        "valid": valid,
        "abstained": abstained,
        "positive": positive,
        "negative": negative,
        "positive_fraction": float(positive / valid) if valid else 0.0,
    }


def qualify_relation_manifest_support_v2(
    manifest: Mapping[str, object],
    *,
    minimum_valid: int = 2,
) -> dict[str, BinarySupportQualificationV2]:
    """Aggregate producer statistics and classify binary court eligibility.

    This function does not claim target authority or scientific PASS.  It answers
    only whether the emitted manifest has nontrivial two-class support for each
    family.  Authority, held-out splitting and preregistered thresholds remain
    separate courts.
    """

    subjects = manifest.get("subjects")
    if not isinstance(subjects, list) or not subjects:
        raise ValueError("RELATION_TRUTH_V2_MANIFEST_SUBJECTS_MISSING")

    family_rows: dict[str, list[Mapping[str, object]]] = {
        name: [] for name in STATIC_RELATION_KEYS_V2 + VIEW_RELATION_KEYS_V2
    }
    for subject in subjects:
        if not isinstance(subject, Mapping):
            raise ValueError("RELATION_TRUTH_V2_MANIFEST_SUBJECT_INVALID")
        views = subject.get("views")
        if not isinstance(views, list):
            raise ValueError("RELATION_TRUTH_V2_MANIFEST_VIEWS_MISSING")
        for view in views:
            if not isinstance(view, Mapping):
                raise ValueError("RELATION_TRUTH_V2_MANIFEST_VIEW_INVALID")
            static = view.get("static")
            projected = view.get("view")
            if not isinstance(static, Mapping) or not isinstance(projected, Mapping):
                raise ValueError("RELATION_TRUTH_V2_MANIFEST_SUPPORT_SECTIONS_MISSING")
            for name in STATIC_RELATION_KEYS_V2:
                row = static.get(name)
                if not isinstance(row, Mapping):
                    raise ValueError(f"RELATION_TRUTH_V2_MANIFEST_STATIC_STATS_MISSING:{name}")
                family_rows[name].append(row)
            for name in VIEW_RELATION_KEYS_V2:
                row = projected.get(name)
                if not isinstance(row, Mapping):
                    raise ValueError(f"RELATION_TRUTH_V2_MANIFEST_VIEW_STATS_MISSING:{name}")
                family_rows[name].append(row)

    return {
        name: qualify_binary_support_v2(
            aggregate_binary_stats_v2(rows),
            minimum_valid=minimum_valid,
        )
        for name, rows in family_rows.items()
    }
