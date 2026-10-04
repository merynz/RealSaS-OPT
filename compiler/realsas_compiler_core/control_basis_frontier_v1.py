from __future__ import annotations

"""Proof-guided, non-monotonic control-basis search frontier.

Control-basis admissibility is not monotonic in control count. A larger basis may
fail while one of its structural children passes. Therefore RealSaS must not use
binary search or one-path greedy pruning as an optimum-rig proof.

This module is deliberately model-agnostic. It maintains a finite, hashable Pareto
frontier over proof defects and effective-control complexity. Learned ATLAS proposal
semantics and Compiler hard proof remain separate owners.
"""

from dataclasses import asdict, dataclass, field
from typing import Iterable

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.types import QualificationError


@dataclass(frozen=True)
class ControlBasisSearchCandidateV1:
    basis_id: str
    control_ids: tuple[str, ...]
    parent_basis_id: str | None
    mechanical_defect: float
    source_defect: float
    editability_defect: float
    secondary_complexity: float
    hard_admissible: bool
    minimum_hard_margin: float
    metadata: dict = field(default_factory=dict)
    schema_version: str = "RealSaS.ControlBasisSearchCandidate.v1"

    @property
    def control_count(self) -> int:
        return len(self.control_ids)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ControlBasisParetoFrontierV1:
    candidates: tuple[ControlBasisSearchCandidateV1, ...]
    minimum_sufficient_basis_ids: tuple[str, ...]
    minimum_sufficient_control_count: int | None
    frontier_hash: str
    schema_version: str = "RealSaS.ControlBasisParetoFrontier.v1"

    def to_dict(self) -> dict:
        return asdict(self)


def _validate(row: ControlBasisSearchCandidateV1) -> None:
    if not row.basis_id:
        raise QualificationError("CONTROL_FRONTIER_BASIS_ID_EMPTY")
    if not row.control_ids or len(set(row.control_ids)) != len(row.control_ids):
        raise QualificationError("CONTROL_FRONTIER_CONTROL_AXIS_INVALID")
    for name in (
        "mechanical_defect",
        "source_defect",
        "editability_defect",
        "secondary_complexity",
    ):
        value = float(getattr(row, name))
        if value < 0.0 or value != value or value == float("inf"):
            raise QualificationError("CONTROL_FRONTIER_DEFECT_INVALID:" + name)
    margin = float(row.minimum_hard_margin)
    if margin != margin or margin in (float("inf"), -float("inf")):
        raise QualificationError("CONTROL_FRONTIER_MARGIN_INVALID")
    if row.hard_admissible:
        if (
            row.mechanical_defect > 0.0
            or row.source_defect > 0.0
            or row.editability_defect > 0.0
        ):
            raise QualificationError("CONTROL_FRONTIER_ADMISSIBLE_WITH_DEFECT")
        if margin < 0.0:
            raise QualificationError("CONTROL_FRONTIER_ADMISSIBLE_NEGATIVE_MARGIN")


def _dominates(
    a: ControlBasisSearchCandidateV1,
    b: ControlBasisSearchCandidateV1,
) -> bool:
    """Weak Pareto dominance across count + proof defect + secondary burden.

    Hard admissibility itself is intentionally not an extra dominance dimension;
    it is implied by zero hard-defect coordinates. This lets a lower-count
    inadmissible candidate remain on the frontier when it trades count for a
    bounded defect that a later owner-scoped repair may close.
    """
    av = (
        a.control_count,
        float(a.mechanical_defect),
        float(a.source_defect),
        float(a.editability_defect),
        float(a.secondary_complexity),
    )
    bv = (
        b.control_count,
        float(b.mechanical_defect),
        float(b.source_defect),
        float(b.editability_defect),
        float(b.secondary_complexity),
    )
    return all(x <= y for x, y in zip(av, bv)) and any(
        x < y for x, y in zip(av, bv)
    )


def build_control_basis_pareto_frontier_v1(
    rows: Iterable[ControlBasisSearchCandidateV1],
) -> ControlBasisParetoFrontierV1:
    values = tuple(rows)
    if not values:
        raise QualificationError("CONTROL_FRONTIER_EMPTY")
    ids = [row.basis_id for row in values]
    if len(set(ids)) != len(ids):
        raise QualificationError("CONTROL_FRONTIER_DUPLICATE_BASIS_ID")
    for row in values:
        _validate(row)

    kept = []
    for row in values:
        if any(
            other.basis_id != row.basis_id and _dominates(other, row)
            for other in values
        ):
            continue
        kept.append(row)

    kept.sort(
        key=lambda row: (
            row.control_count,
            float(row.mechanical_defect),
            float(row.source_defect),
            float(row.editability_defect),
            float(row.secondary_complexity),
            -float(row.minimum_hard_margin),
            row.basis_id,
        )
    )

    admitted = [row for row in values if row.hard_admissible]
    if admitted:
        min_count = min(row.control_count for row in admitted)
        min_ids = tuple(
            sorted(row.basis_id for row in admitted if row.control_count == min_count)
        )
    else:
        min_count = None
        min_ids = ()

    payload = {
        "schema": "RealSaS.ControlBasisParetoFrontier.v1",
        "candidate_ids": tuple(row.basis_id for row in kept),
        "candidate_hashes": tuple(content_sha256(row.to_dict()) for row in kept),
        "minimum_sufficient_basis_ids": min_ids,
        "minimum_sufficient_control_count": min_count,
    }
    return ControlBasisParetoFrontierV1(
        candidates=tuple(kept),
        minimum_sufficient_basis_ids=min_ids,
        minimum_sufficient_control_count=min_count,
        frontier_hash=content_sha256(payload),
    )


__all__ = [
    "ControlBasisSearchCandidateV1",
    "ControlBasisParetoFrontierV1",
    "build_control_basis_pareto_frontier_v1",
]
