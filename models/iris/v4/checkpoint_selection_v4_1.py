from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class HardSignCheckpointCandidateV41:
    step: int
    max_foreground_miss_fraction: float
    max_background_zero_crossing_fraction: float
    source_exterior_negative_fraction: float

    def validate(self) -> None:
        if int(self.step) <= 0:
            raise ValueError("checkpoint step must be positive")
        for name in (
            "max_foreground_miss_fraction",
            "max_background_zero_crossing_fraction",
            "source_exterior_negative_fraction",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError(f"invalid hard-sign checkpoint field {name}")

    @property
    def hard_sign_vector(self) -> tuple[float, float, float]:
        self.validate()
        return (
            float(self.max_foreground_miss_fraction),
            float(self.max_background_zero_crossing_fraction),
            float(self.source_exterior_negative_fraction),
        )


def dominates_hard_sign_v41(
    lhs: HardSignCheckpointCandidateV41,
    rhs: HardSignCheckpointCandidateV41,
) -> bool:
    """True only when lhs is no worse on every hard sign axis and better on at least one."""

    a = lhs.hard_sign_vector
    b = rhs.hard_sign_vector
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def pareto_front_hard_sign_v41(
    candidates: Sequence[HardSignCheckpointCandidateV41],
) -> tuple[HardSignCheckpointCandidateV41, ...]:
    if not candidates:
        raise ValueError("at least one checkpoint candidate is required")
    for candidate in candidates:
        candidate.validate()
    if len({int(candidate.step) for candidate in candidates}) != len(candidates):
        raise ValueError("checkpoint steps must be unique")

    front: list[HardSignCheckpointCandidateV41] = []
    for candidate in candidates:
        if any(
            other.step != candidate.step and dominates_hard_sign_v41(other, candidate)
            for other in candidates
        ):
            continue
        front.append(candidate)
    return tuple(sorted(front, key=lambda row: int(row.step)))


@dataclass(frozen=True, order=True)
class Stage13CheckpointSelectionScoreV41:
    """Lower is better; built only from the frozen Stage13-v2 gates.

    This score is a checkpoint-selection device for controlled FIT1. It does not alter
    Stage13 PASS semantics. A candidate that passes all gates always outranks a failing
    candidate. Among failures we first minimize the number of violated per-view rules,
    then the worst normalized gate excess, then the sum of normalized excesses.
    """

    failed_rule_count: int
    worst_normalized_excess: float
    summed_normalized_excess: float
    step: int

    def validate(self) -> None:
        if int(self.failed_rule_count) < 0 or int(self.step) <= 0:
            raise ValueError("invalid Stage13 selection score integers")
        for value in (self.worst_normalized_excess, self.summed_normalized_excess):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError("invalid Stage13 normalized excess")


def _deficit_ratio(value: float, threshold: float) -> float:
    # For lower-bounded metrics close to 1, normalize by the allowed deficit to 1.
    if value >= threshold:
        return 0.0
    denom = 1.0 - threshold
    if denom <= 0.0:
        raise ValueError("lower-bound threshold must be < 1")
    return (threshold - value) / denom


def _excess_ratio(value: float, threshold: float) -> float:
    if value <= threshold:
        return 0.0
    if threshold <= 0.0:
        raise ValueError("upper-bound threshold must be positive")
    return value / threshold - 1.0


def stage13_checkpoint_selection_score_v41(
    *,
    step: int,
    per_view_rows: Sequence[Mapping[str, float]],
    thresholds: Mapping[str, float],
) -> Stage13CheckpointSelectionScoreV41:
    required_thresholds = {
        "min_recall",
        "min_precision",
        "max_largest_coherent_hole_fraction",
        "max_interior_uncovered_fraction",
        "min_component_recall",
        "max_silhouette_edge_p95_px",
    }
    if not required_thresholds.issubset(thresholds):
        missing = sorted(required_thresholds - set(thresholds))
        raise ValueError(f"missing Stage13 thresholds: {missing}")
    if len(per_view_rows) != 8:
        raise ValueError("Stage13 checkpoint selection requires exactly eight views")

    excesses: list[float] = []
    for row in per_view_rows:
        excesses.extend(
            [
                _deficit_ratio(float(row["recall"]), float(thresholds["min_recall"])),
                _deficit_ratio(float(row["precision"]), float(thresholds["min_precision"])),
                _deficit_ratio(
                    float(row["minimum_eligible_component_recall"]),
                    float(thresholds["min_component_recall"]),
                ),
                _excess_ratio(
                    float(row["largest_coherent_hole_fraction"]),
                    float(thresholds["max_largest_coherent_hole_fraction"]),
                ),
                _excess_ratio(
                    float(row["interior_uncovered_fraction"]),
                    float(thresholds["max_interior_uncovered_fraction"]),
                ),
                _excess_ratio(
                    float(row["silhouette_edge_p95_px"]),
                    float(thresholds["max_silhouette_edge_p95_px"]),
                ),
            ]
        )

    failed = sum(1 for value in excesses if value > 0.0)
    score = Stage13CheckpointSelectionScoreV41(
        failed_rule_count=int(failed),
        worst_normalized_excess=float(max(excesses) if excesses else 0.0),
        summed_normalized_excess=float(sum(excesses)),
        step=int(step),
    )
    score.validate()
    return score
