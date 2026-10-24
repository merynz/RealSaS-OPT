from __future__ import annotations

import pytest

from tools.research.run_multiview_semantic_oracle_v4_r6 import (
    _GROUND_SCHEMA,
    _single_view_prompt,
    _validate_single_view,
)


def _instance():
    return {
        "id": "I10",
        "label": "sword",
        "descriptor": "long blade associated with sword-side hand",
        "confidence": 0.98,
    }


def test_single_view_prompt_has_no_numeric_coordinate_template_to_copy():
    prompt = _single_view_prompt(_instance(), 3)
    assert "[100,200,300,700" not in prompt
    assert "there are intentionally no example coordinates to copy" in prompt
    assert "bbox_2d" in prompt
    assert "point_2d" in prompt


def test_single_view_grounding_accepts_visible_bbox_and_interior_point():
    payload = {
        "schema": _GROUND_SCHEMA,
        "id": "I10",
        "label": "sword",
        "view": 3,
        "present": True,
        "bbox_2d": [120, 240, 810, 620],
        "point_2d": [500, 430],
    }
    out = _validate_single_view(payload, instance=_instance(), view=3)
    assert out["present"] is True
    assert out["bbox_2d"] == [120.0, 240.0, 810.0, 620.0]


def test_single_view_grounding_accepts_absence_only_with_null_geometry():
    payload = {
        "schema": _GROUND_SCHEMA,
        "id": "I10",
        "label": "sword",
        "view": 6,
        "present": False,
        "bbox_2d": None,
        "point_2d": None,
    }
    out = _validate_single_view(payload, instance=_instance(), view=6)
    assert out["present"] is False


def test_single_view_grounding_rejects_point_outside_box():
    payload = {
        "schema": _GROUND_SCHEMA,
        "id": "I10",
        "label": "sword",
        "view": 2,
        "present": True,
        "bbox_2d": [100, 200, 300, 400],
        "point_2d": [900, 900],
    }
    with pytest.raises(ValueError, match="point must lie inside"):
        _validate_single_view(payload, instance=_instance(), view=2)


def test_single_view_grounding_rejects_wrong_view_identity():
    payload = {
        "schema": _GROUND_SCHEMA,
        "id": "I10",
        "label": "sword",
        "view": 0,
        "present": False,
        "bbox_2d": None,
        "point_2d": None,
    }
    with pytest.raises(ValueError, match="view mismatch"):
        _validate_single_view(payload, instance=_instance(), view=1)
