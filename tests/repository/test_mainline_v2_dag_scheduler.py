from __future__ import annotations

import pytest

from compiler.realsas_compiler_services.orchestrator.mainline import (
    _target_closure,
    dependency_failure_ids,
    ready_stage_ids,
    topological_stage_ids,
)


def _stage(ordinal, stage_id, deps=()):
    return {
        "ordinal": ordinal,
        "id": stage_id,
        "depends_on": list(deps),
    }


def _row(ordinal, stage_id, status="PENDING"):
    return {
        "ordinal": ordinal,
        "id": stage_id,
        "status": status,
    }


def test_ready_set_preserves_independent_branch_after_failure():
    plan = {
        "stages": [
            _stage(1, "01_A"),
            _stage(2, "02_B", ("01_A",)),
            _stage(3, "03_C", ("01_A",)),
            _stage(4, "04_D", ("02_B",)),
            _stage(5, "05_E", ("03_C",)),
        ]
    }
    ledger = {
        "stages": [
            _row(1, "01_A", "PASS"),
            _row(2, "02_B", "FAIL"),
            _row(3, "03_C", "PENDING"),
            _row(4, "04_D", "PENDING"),
            _row(5, "05_E", "PENDING"),
        ]
    }
    assert ready_stage_ids(plan, ledger) == ("03_C",)
    assert dependency_failure_ids(plan, ledger, "04_D") == ("02_B",)
    assert dependency_failure_ids(plan, ledger, "05_E") == ()


def test_target_closure_is_dependency_ancestry_not_ordinal_prefix():
    plan = {
        "stages": [
            _stage(1, "01_A"),
            _stage(2, "02_B", ("01_A",)),
            _stage(3, "03_C", ("01_A",)),
            _stage(4, "04_D", ("02_B",)),
            _stage(5, "05_E", ("03_C",)),
        ]
    }
    assert _target_closure(plan, ("05_E",)) == {"01_A", "03_C", "05_E"}
    assert "02_B" not in _target_closure(plan, ("05_E",))
    assert "04_D" not in _target_closure(plan, ("05_E",))


def test_topological_order_detects_cycle():
    plan = {
        "stages": [
            _stage(1, "01_A", ("02_B",)),
            _stage(2, "02_B", ("01_A",)),
        ]
    }
    with pytest.raises(RuntimeError, match="MAINLINE_PLAN_CYCLE"):
        topological_stage_ids(plan)


def test_topological_order_uses_ordinal_only_as_stable_display_tiebreak():
    plan = {
        "stages": [
            _stage(3, "03_C", ("01_A",)),
            _stage(1, "01_A"),
            _stage(2, "02_B", ("01_A",)),
        ]
    }
    assert topological_stage_ids(plan) == ("01_A", "02_B", "03_C")
