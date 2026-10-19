from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from models.iris.vnext.cohort_v1 import SUBJECT_IDS
from tools.iris_vnext.hydrate_fit8_direct_surface_inputs_v1 import validate_binding


BINDING = Path("canonical/VNEXT_DIRECT_SURFACE_INPUT_BINDING_V1.json")


def _load():
    return json.loads(BINDING.read_text(encoding="utf-8"))


def test_reviewed_direct_surface_binding_is_valid_and_complete():
    value = _load()
    validate_binding(value)
    assert value["training_allowed"] is False
    assert value["external_storage_write_allowed"] is False
    assert value["scientific_pass_claimed"] is False
    assert tuple(row["subject_id"] for row in value["prepared"]["subjects"]) == SUBJECT_IDS
    assert value["surface_policy"]["base_cells"] == 512
    assert value["surface_policy"]["fine_cells"] == 1024
    assert value["surface_policy"]["tessa_invoked"] is False


def test_binding_cannot_authorize_training_or_remote_write():
    for key in ("training_allowed", "external_storage_write_allowed", "scientific_pass_claimed"):
        value = _load()
        value[key] = True
        with pytest.raises(ValueError):
            validate_binding(value)


def test_binding_cannot_reduce_scientific_surface_resolution():
    value = _load()
    value["surface_policy"]["base_cells"] = 256
    value["surface_policy"]["fine_cells"] = 512
    with pytest.raises(ValueError, match="FULL_RESOLUTION_POLICY_REQUIRED"):
        validate_binding(value)


def test_binding_fails_closed_on_subject_or_checkpoint_drift():
    value = _load()
    value["prepared"]["subjects"] = list(reversed(value["prepared"]["subjects"]))
    with pytest.raises(ValueError, match="SUBJECT_ORDER_DRIFT"):
        validate_binding(value)

    value = _load()
    value["v8"]["checkpoint"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="CHECKPOINT_BINDING_INVALID"):
        validate_binding(value)
