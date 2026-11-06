from __future__ import annotations

import pytest

from tools.research.run_multiview_semantic_oracle_v4_r7 import (
    _validate_single_view_r7,
    normalize_view_token_v1,
)


def test_view_token_normalization_accepts_observed_qwen_forms():
    assert normalize_view_token_v1(6) == 6
    assert normalize_view_token_v1("6") == 6
    assert normalize_view_token_v1("V6") == 6
    assert normalize_view_token_v1("view V6") == 6
    assert normalize_view_token_v1("orthographic view V6") == 6


def test_view_token_normalization_rejects_ambiguous_or_out_of_range_forms():
    for value in ["V8", "view 9", "V2/V3", "front view", True]:
        with pytest.raises(ValueError):
            normalize_view_token_v1(value)


def test_r7_validator_accepts_observed_sword_grounding_shape():
    instance = {"id": "10", "label": "sword", "descriptor": "long blade", "confidence": 0.98}
    payload = {
        "schema": "RealSaS.SingleViewInstanceGrounding.v1",
        "id": 10,
        "label": "sword",
        "view": "orthographic view V6",
        "present": True,
        "bbox_2d": [230, 520, 817, 563],
        "point_2d": [500, 540],
    }
    out = _validate_single_view_r7(payload, instance=instance, view=6)
    assert out["view"] == 6
    assert out["bbox_2d"] == [230.0, 520.0, 817.0, 563.0]
    assert out["point_2d"] == [500.0, 540.0]
