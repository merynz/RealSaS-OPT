from __future__ import annotations

"""Subject-agnostic minimum-sufficient control-basis selection.

This module does not generate rigs and does not contain character-specific joint
counts. It only selects among already-measured candidate control bases.

Every margin is signed and normalized by the owning court:
    margin >= 0  -> the corresponding hard requirement is satisfied
    margin < 0   -> the requirement fails

The selector first requires all hard margins to pass, then minimizes complexity.
Extra proof margin is only a tie-breaker between equally simple admissible bases.
"""

from dataclasses import asdict, dataclass
import math
from typing import Iterable


@dataclass(frozen=True)
class ControlBasisEvaluationV1:
    basis_id: str
    control_ids: tuple[str, ...]
    mechanical_margin: float
    source_fidelity_margin: float
    editability_margin: float
    free_running_margin: float
    complexity_score: float

    @property
    def control_count(self) -> int:
        return len(self.control_ids)

    @property
    def minimum_hard_margin(self) -> float:
        return min(
            float(self.mechanical_margin),
            float(self.source_fidelity_margin),
            float(self.editability_margin),
            float(self.free_running_margin),
        )

    @property
    def admissible(self) -> bool:
        return self.minimum_hard_margin >= 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ControlBasisSelectionV1:
    selected_basis_id: str
    selected_control_ids: tuple[str, ...]
    selected_complexity_score: float
    selected_minimum_hard_margin: float
    admissible_basis_ids: tuple[str, ...]
    rejected_basis_ids: tuple[str, ...]
    semantic_version: str = "RealSaS.MinimumSufficientControlBasisSelector.v1"

    def to_dict(self) -> dict:
        return asdict(self)



def upper_bounded_signed_margin_v1(
    value: float,
    threshold: float,
    *,
    epsilon: float = 1e-12,
) -> float:
    """Return >=0 when an upper-bounded badness metric satisfies value <= threshold."""
    x = float(value)
    t = float(threshold)
    e = float(epsilon)
    if not all(math.isfinite(v) for v in (x, t, e)) or e <= 0.0:
        raise ValueError("CONTROL_BASIS_MARGIN_ARGUMENT_INVALID")
    return (t - x) / max(abs(t), e)


def lower_bounded_signed_margin_v1(
    value: float,
    threshold: float,
    *,
    epsilon: float = 1e-12,
) -> float:
    """Return >=0 when a lower-bounded goodness metric satisfies value >= threshold."""
    x = float(value)
    t = float(threshold)
    e = float(epsilon)
    if not all(math.isfinite(v) for v in (x, t, e)) or e <= 0.0:
        raise ValueError("CONTROL_BASIS_MARGIN_ARGUMENT_INVALID")
    return (x - t) / max(abs(t), e)


def minimum_gate_margin_v1(*margins: float) -> float:
    """Fail-closed aggregation for a gate made of multiple required sub-margins."""
    if not margins:
        raise ValueError("CONTROL_BASIS_GATE_MARGINS_EMPTY")
    values = tuple(float(x) for x in margins)
    if not all(math.isfinite(x) for x in values):
        raise ValueError("CONTROL_BASIS_GATE_MARGIN_NONFINITE")
    return min(values)


def _validate(value: ControlBasisEvaluationV1) -> None:
    if not value.basis_id:
        raise ValueError("CONTROL_BASIS_ID_EMPTY")
    if not value.control_ids:
        raise ValueError("CONTROL_BASIS_EMPTY")
    if len(set(value.control_ids)) != len(value.control_ids):
        raise ValueError("CONTROL_BASIS_DUPLICATE_CONTROL")
    values = (
        value.mechanical_margin,
        value.source_fidelity_margin,
        value.editability_margin,
        value.free_running_margin,
        value.complexity_score,
    )
    if not all(math.isfinite(float(x)) for x in values):
        raise ValueError("CONTROL_BASIS_NONFINITE_METRIC")
    if float(value.complexity_score) < 0.0:
        raise ValueError("CONTROL_BASIS_COMPLEXITY_NEGATIVE")


def select_minimum_sufficient_control_basis_v1(
    evaluations: Iterable[ControlBasisEvaluationV1],
) -> ControlBasisSelectionV1:
    rows = tuple(evaluations)
    if not rows:
        raise ValueError("CONTROL_BASIS_EVALUATIONS_EMPTY")

    for row in rows:
        _validate(row)

    ids = [row.basis_id for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("CONTROL_BASIS_EVALUATION_ID_DUPLICATE")

    admissible = tuple(row for row in rows if row.admissible)
    if not admissible:
        raise ValueError("NO_MECHANICALLY_SUFFICIENT_CONTROL_BASIS")

    # Correct optimization order:
    #   1) all hard requirements already pass,
    #   2) minimize subject-specific complexity,
    #   3) then minimize raw control count,
    #   4) only then prefer larger residual hard margin,
    #   5) deterministic lexical tie-break.
    selected = min(
        admissible,
        key=lambda row: (
            float(row.complexity_score),
            int(row.control_count),
            -float(row.minimum_hard_margin),
            row.basis_id,
        ),
    )

    admitted_ids = tuple(sorted(row.basis_id for row in admissible))
    rejected_ids = tuple(sorted(row.basis_id for row in rows if not row.admissible))
    return ControlBasisSelectionV1(
        selected_basis_id=selected.basis_id,
        selected_control_ids=selected.control_ids,
        selected_complexity_score=float(selected.complexity_score),
        selected_minimum_hard_margin=float(selected.minimum_hard_margin),
        admissible_basis_ids=admitted_ids,
        rejected_basis_ids=rejected_ids,
    )


__all__ = [
    "ControlBasisEvaluationV1",
    "ControlBasisSelectionV1",
    "upper_bounded_signed_margin_v1",
    "lower_bounded_signed_margin_v1",
    "minimum_gate_margin_v1",
    "select_minimum_sufficient_control_basis_v1",
]
