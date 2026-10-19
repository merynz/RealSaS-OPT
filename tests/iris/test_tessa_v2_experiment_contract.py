from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from models.tessa.v2.experiment_v2 import learning_rate_at_step_v1, validate_preregistration_v1

PREREG = Path("canonical/TESSA_V2_FIT8_PREREGISTRATION_V1.json")


def load():
    return json.loads(PREREG.read_text(encoding="utf-8"))


def test_preregistration_exact_contract_is_valid():
    value = load()
    validate_preregistration_v1(value)
    assert value["optimizer"]["optimizer_steps"] == 2400
    assert value["architecture"]["d_model"] == 1024
    assert value["free_running"]["teacher_prefix_allowed"] is False
    assert value["interpretation"]["tessa_mandatory_authorized_by_this_experiment"] is False


def test_preregistration_rejects_capacity_reduction_and_authority_escalation():
    value = load(); value["architecture"]["decoder_layers"] = 12
    with pytest.raises(ValueError, match="ARCHITECTURE_DRIFT"):
        validate_preregistration_v1(value)
    value = load(); value["interpretation"]["canonical_m_authorized_by_this_experiment"] = True
    with pytest.raises(ValueError, match="AUTHORITY_ESCALATION"):
        validate_preregistration_v1(value)


def test_lr_schedule_is_locked_and_bounded():
    value = load(); validate_preregistration_v1(value)
    warmup = int(value["optimizer"]["warmup_steps"])
    peak = float(value["optimizer"]["learning_rate"])
    floor = float(value["optimizer"]["minimum_learning_rate"])
    assert 0.0 < learning_rate_at_step_v1(1, value) < peak
    assert learning_rate_at_step_v1(warmup, value) == pytest.approx(peak)
    assert learning_rate_at_step_v1(2400, value) == pytest.approx(floor)
    with pytest.raises(ValueError):
        learning_rate_at_step_v1(0, value)
    with pytest.raises(ValueError):
        learning_rate_at_step_v1(2401, value)
