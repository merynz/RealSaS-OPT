from pathlib import Path

import pytest

from tools.research.run_multiview_semantic_oracle_v4_r3 import (
    COMPACT_SCHEMA,
    _extract_json_strict,
    expand_compact_oracle_v1,
    normalize_local_image_sources_v1,
)
from tools.research.run_multiview_semantic_oracle_v4_r4 import _unknown, _validate_item


def test_compact_oracle_expands_visible_views_and_relations():
    compact = {
        "schema": COMPACT_SCHEMA,
        "summary": "stylized knight",
        "instances": [
            {
                "id": "I00",
                "label": "sword",
                "confidence": 0.93,
                "v": {
                    "0": [100, 200, 300, 700, 210, 430, 0.91],
                    "4": [120, 210, 330, 720, 225, 440, 0.88],
                },
                "r": [["HELD_BY", "I01", 0.9]],
            },
            {
                "id": "I01",
                "label": "hand",
                "confidence": 0.90,
                "v": {"0": [180, 390, 250, 470, 215, 430, 0.9]},
                "r": [],
            },
        ],
    }
    result = expand_compact_oracle_v1(compact)
    assert result["schema"] == "RealSaS.MultiviewSemanticOracle.v1"
    sword = result["instances"][0]
    assert sword["views"]["0"]["present"] is True
    assert sword["views"]["4"]["present"] is True
    assert sword["views"]["1"]["present"] is False
    assert sword["relations"] == [
        {"type": "HELD_BY", "target_instance_id": "I01", "confidence": 0.9}
    ]


def test_compact_oracle_rejects_invalid_coordinate_contract():
    compact = {
        "schema": COMPACT_SCHEMA,
        "summary": "x",
        "instances": [
            {
                "id": "I00",
                "label": "cape",
                "confidence": 0.8,
                "v": {"0": [500, 500, 400, 700, 450, 600, 0.8]},
                "r": [],
            }
        ],
    }
    with pytest.raises(ValueError, match="positive area"):
        expand_compact_oracle_v1(compact)


def test_strict_json_parser_does_not_repair_malformed_output():
    with pytest.raises(RuntimeError, match="MULTIMODAL_JSON_INVALID"):
        _extract_json_strict('{"schema":"x","instances":[}')


def test_file_uri_normalization_preserves_non_image_fields(tmp_path: Path):
    image = tmp_path / "V0.png"
    image.write_bytes(b"not-read-by-this-test")
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "V0"},
            {"type": "image", "image": image.resolve().as_uri()},
        ],
    }]
    out = normalize_local_image_sources_v1(messages)
    assert out[0]["content"][0]["text"] == "V0"
    assert out[0]["content"][1]["image"] == str(image.resolve())
    assert messages[0]["content"][1]["image"].startswith("file://")


def test_mask_item_validation_and_fail_closed_unknown():
    valid = {
        "schema": "RealSaS.MultiviewMaskAdjudicationItem.v1",
        "instance_id": "I00",
        "label": "cape",
        "verdict": "PASS",
        "semantic_match": True,
        "confidence": 0.88,
        "bad_views": [5],
        "reason_code": "GOOD",
    }
    row = _validate_item(valid, instance_id="I00", label="cape")
    assert row["verdict"] == "PASS"
    assert row["bad_views"] == [5]

    unknown = _unknown("I00", "cape")
    assert unknown["verdict"] == "UNKNOWN"
    assert unknown["semantic_match"] is False
    assert unknown["confidence"] == 0.0
