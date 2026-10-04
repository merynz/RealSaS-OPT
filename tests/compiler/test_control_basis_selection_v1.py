import pytest

from compiler.realsas_compiler_core.control_basis_selection_v1 import (
    ControlBasisEvaluationV1,
    lower_bounded_signed_margin_v1,
    minimum_gate_margin_v1,
    select_minimum_sufficient_control_basis_v1,
    upper_bounded_signed_margin_v1,
)


def _row(
    basis_id,
    count,
    *,
    mechanical,
    source=0.2,
    edit=0.2,
    free=0.2,
    complexity=None,
):
    return ControlBasisEvaluationV1(
        basis_id=basis_id,
        control_ids=tuple(f"J{i}" for i in range(count)),
        mechanical_margin=float(mechanical),
        source_fidelity_margin=float(source),
        editability_margin=float(edit),
        free_running_margin=float(free),
        complexity_score=float(count if complexity is None else complexity),
    )


def test_minimum_sufficient_beats_more_complex_higher_margin_basis():
    result = select_minimum_sufficient_control_basis_v1(
        (
            _row("R12", 12, mechanical=0.02, complexity=12.0),
            _row("R18", 18, mechanical=0.80, complexity=18.0),
        )
    )
    assert result.selected_basis_id == "R12"
    assert len(result.selected_control_ids) == 12


def test_different_subjects_can_select_different_optimum_counts():
    shared_ids = ("R8", "R12", "R18")

    subject_a = select_minimum_sufficient_control_basis_v1(
        (
            _row(shared_ids[0], 8, mechanical=-0.20, complexity=8.0),
            _row(shared_ids[1], 12, mechanical=0.05, complexity=12.0),
            _row(shared_ids[2], 18, mechanical=0.60, complexity=18.0),
        )
    )
    subject_b = select_minimum_sufficient_control_basis_v1(
        (
            _row(shared_ids[0], 8, mechanical=-0.50, complexity=8.0),
            _row(shared_ids[1], 12, mechanical=-0.10, complexity=12.0),
            _row(shared_ids[2], 18, mechanical=0.08, complexity=18.0),
        )
    )

    assert subject_a.selected_basis_id == "R12"
    assert len(subject_a.selected_control_ids) == 12

    assert subject_b.selected_basis_id == "R18"
    assert len(subject_b.selected_control_ids) == 18


def test_editability_can_reject_mechanically_good_overfit_basis():
    result = select_minimum_sufficient_control_basis_v1(
        (
            _row("MECH_ONLY", 10, mechanical=0.4, edit=-0.01, complexity=10.0),
            _row("EDITABLE", 12, mechanical=0.1, edit=0.05, complexity=12.0),
        )
    )
    assert result.selected_basis_id == "EDITABLE"
    assert "MECH_ONLY" in result.rejected_basis_ids


def test_equal_complexity_uses_count_before_excess_margin():
    result = select_minimum_sufficient_control_basis_v1(
        (
            _row("SMALL", 10, mechanical=0.01, complexity=20.0),
            _row("LARGE", 14, mechanical=1.0, complexity=20.0),
        )
    )
    assert result.selected_basis_id == "SMALL"


def test_equal_complexity_and_count_uses_margin_as_tiebreaker():
    result = select_minimum_sufficient_control_basis_v1(
        (
            _row("LOW_MARGIN", 10, mechanical=0.01, complexity=10.0),
            _row("HIGH_MARGIN", 10, mechanical=0.40, complexity=10.0),
        )
    )
    assert result.selected_basis_id == "HIGH_MARGIN"


def test_no_admissible_basis_fails_closed():
    with pytest.raises(
        ValueError, match="NO_MECHANICALLY_SUFFICIENT_CONTROL_BASIS"
    ):
        select_minimum_sufficient_control_basis_v1(
            (
                _row("R8", 8, mechanical=-0.1),
                _row("R12", 12, mechanical=-0.01),
            )
        )


def test_signed_normalized_margin_contracts_have_universal_pass_semantics():
    assert upper_bounded_signed_margin_v1(8.0, 10.0) == pytest.approx(0.2)
    assert upper_bounded_signed_margin_v1(12.0, 10.0) == pytest.approx(-0.2)

    assert lower_bounded_signed_margin_v1(0.9, 0.8) == pytest.approx(0.125)
    assert lower_bounded_signed_margin_v1(0.7, 0.8) == pytest.approx(-0.125)

    assert minimum_gate_margin_v1(0.2, 0.125, 0.4) == pytest.approx(0.125)
    assert minimum_gate_margin_v1(0.2, -0.125, 0.4) == pytest.approx(-0.125)


def test_larger_basis_cannot_win_only_because_secondary_complexity_is_lower():
    result = select_minimum_sufficient_control_basis_v1(
        (
            _row("SMALL", 10, mechanical=0.05, complexity=100.0),
            _row("LARGE", 14, mechanical=0.50, complexity=1.0),
        )
    )
    assert result.selected_basis_id == "SMALL"
    assert len(result.selected_control_ids) == 10
