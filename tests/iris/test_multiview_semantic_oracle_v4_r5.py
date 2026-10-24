from __future__ import annotations

import pytest

from tools.research.run_multiview_semantic_oracle_v4_r5 import (
    _DISCOVERY_SCHEMA,
    _LOCALIZATION_SCHEMA,
    _validate_discovery,
    _validate_localization,
)


def test_discovery_preserves_distinct_same_label_instances_with_descriptors():
    payload = {
        "schema": _DISCOVERY_SCHEMA,
        "summary": "knight",
        "instances": [
            {"id": "I00", "label": "hand", "descriptor": "sword-side hand", "confidence": 0.9},
            {"id": "I01", "label": "hand", "descriptor": "shield-side hand", "confidence": 0.88},
            {"id": "I02", "label": "sword", "descriptor": "long held blade", "confidence": 0.95},
        ],
    }
    out = _validate_discovery(payload)
    assert [x["id"] for x in out["instances"]] == ["I00", "I01", "I02"]
    assert out["instances"][0]["descriptor"] != out["instances"][1]["descriptor"]


def test_discovery_rejects_view_direction_as_semantic_label():
    payload = {
        "schema": _DISCOVERY_SCHEMA,
        "summary": "bad retry",
        "instances": [
            {"id": "V0", "label": "front", "descriptor": "front view", "confidence": 0.98},
        ],
    }
    with pytest.raises(ValueError, match="unsupported label"):
        _validate_discovery(payload)


def test_localization_requires_multiview_support():
    instance = {"id": "I00", "label": "sword", "descriptor": "long held blade", "confidence": 0.9}
    payload = {
        "schema": _LOCALIZATION_SCHEMA,
        "id": "I00",
        "label": "sword",
        "v": {
            "0": [100, 200, 300, 700, 210, 430, 0.9],
        },
    }
    with pytest.raises(ValueError, match="insufficient multiview support"):
        _validate_localization(payload, instance=instance)


def test_localization_accepts_two_or_more_visible_views():
    instance = {"id": "I00", "label": "cape", "descriptor": "back flowing cloth", "confidence": 0.92}
    payload = {
        "schema": _LOCALIZATION_SCHEMA,
        "id": "I00",
        "label": "cape",
        "v": {
            "0": [100, 200, 400, 800, 220, 450, 0.9],
            "1": [120, 180, 420, 790, 230, 440, 0.88],
            "4": [180, 140, 700, 850, 420, 450, 0.93],
        },
    }
    out = _validate_localization(payload, instance=instance)
    assert sorted(out["v"]) == ["0", "1", "4"]
