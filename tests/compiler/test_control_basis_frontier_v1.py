import pytest

from compiler.realsas_compiler_core.control_basis_frontier_v1 import (
    ControlBasisSearchCandidateV1,
    build_control_basis_pareto_frontier_v1,
)
from compiler.realsas_compiler_core.types import QualificationError


def _row(
    basis_id,
    count,
    *,
    mech,
    source=0.0,
    edit=0.0,
    secondary=0.0,
    admitted=False,
    margin=-1.0,
):
    return ControlBasisSearchCandidateV1(
        basis_id=basis_id,
        control_ids=tuple(f"{basis_id}:J{i}" for i in range(count)),
        parent_basis_id=None,
        mechanical_defect=mech,
        source_defect=source,
        editability_defect=edit,
        secondary_complexity=secondary,
        hard_admissible=admitted,
        minimum_hard_margin=margin,
    )


def test_nonmonotonic_smaller_failing_basis_can_remain_on_frontier():
    rows = (
        _row("R28", 28, mech=0.20, secondary=2.0),
        _row("R27_PASS", 27, mech=0.0, admitted=True, margin=0.05, secondary=2.0),
        _row("R26_NEAR", 26, mech=0.03, secondary=1.5),
    )
    result = build_control_basis_pareto_frontier_v1(rows)
    assert [x.basis_id for x in result.candidates] == ["R26_NEAR", "R27_PASS"]
    assert result.minimum_sufficient_control_count == 27
    assert result.minimum_sufficient_basis_ids == ("R27_PASS",)


def test_dominated_same_or_larger_basis_is_removed():
    rows = (
        _row("A", 10, mech=0.1, source=0.0, edit=0.0, secondary=1.0),
        _row("B", 11, mech=0.2, source=0.0, edit=0.0, secondary=2.0),
    )
    result = build_control_basis_pareto_frontier_v1(rows)
    assert [x.basis_id for x in result.candidates] == ["A"]


def test_minimum_sufficient_is_computed_over_full_sealed_family_not_only_frontier():
    rows = (
        _row("PASS12", 12, mech=0.0, admitted=True, margin=0.02, secondary=2.0),
        _row("PASS10", 10, mech=0.0, admitted=True, margin=0.01, secondary=3.0),
        _row("FAIL9", 9, mech=0.01, admitted=False, margin=-0.1, secondary=1.0),
    )
    result = build_control_basis_pareto_frontier_v1(rows)
    assert result.minimum_sufficient_control_count == 10
    assert result.minimum_sufficient_basis_ids == ("PASS10",)


def test_admissible_candidate_cannot_carry_hard_defect():
    with pytest.raises(
        QualificationError, match="CONTROL_FRONTIER_ADMISSIBLE_WITH_DEFECT"
    ):
        build_control_basis_pareto_frontier_v1(
            (_row("BAD", 4, mech=0.1, admitted=True, margin=0.01),)
        )
