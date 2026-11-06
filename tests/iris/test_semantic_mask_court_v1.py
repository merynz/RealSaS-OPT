import numpy as np
import pytest

from models.iris.semantic_mask_court_v1 import (
    audit_prompted_mask_v1,
    normalized_box_to_pixels,
    normalized_point_to_pixels,
    validate_multiview_oracle_payload_v1,
)


def _subject():
    arr = np.zeros((64, 64), dtype=bool)
    arr[4:60, 4:60] = True
    return arr


def test_prompted_mask_passes_when_local_coherent_and_centered():
    subject = _subject()
    mask = np.zeros_like(subject)
    mask[20:42, 24:40] = True
    out = audit_prompted_mask_v1(
        mask,
        subject_mask=subject,
        prompt_box=(20, 16, 44, 46),
        prompt_center=(32, 31),
        minimum_pixels=32,
    )
    assert out.status == "PASS"
    assert out.failure_code is None
    assert out.center_retained
    assert out.connected_component_count == 1
    assert out.largest_component_fraction == 1.0


def test_prompted_mask_rejects_fragmented_multi_region_result():
    subject = _subject()
    mask = np.zeros_like(subject)
    mask[18:22, 18:22] = True
    mask[26:30, 26:30] = True
    mask[34:38, 34:38] = True
    mask[42:46, 42:46] = True
    out = audit_prompted_mask_v1(
        mask,
        subject_mask=subject,
        prompt_box=(14, 14, 50, 50),
        prompt_center=(20, 20),
        minimum_pixels=16,
        minimum_largest_component_fraction=0.80,
        maximum_components=3,
    )
    assert out.status == "FAIL"
    assert out.failure_code in {"MASK_FRAGMENTED", "MASK_MULTI_REGION"}


def test_prompted_mask_rejects_dropped_prompt_center():
    subject = _subject()
    mask = np.zeros_like(subject)
    mask[20:35, 20:35] = True
    out = audit_prompted_mask_v1(
        mask,
        subject_mask=subject,
        prompt_box=(18, 18, 46, 46),
        prompt_center=(42, 42),
        minimum_pixels=16,
    )
    assert out.status == "FAIL"
    assert out.failure_code == "MASK_DROPPED_PROMPT_CENTER"


def test_normalized_prompt_conversion_is_bounded():
    box = normalized_box_to_pixels([100, 200, 800, 900], width=200, height=100)
    point = normalized_point_to_pixels([500, 500], width=200, height=100)
    assert box == (20, 20, 160, 90)
    assert point == (100, 50)


def test_oracle_contract_requires_all_eight_views_and_valid_labels():
    payload = {
        "schema": "RealSaS.MultiviewSemanticOracle.v1",
        "instances": [
            {
                "instance_id": "I00",
                "label": "sword",
                "confidence": 0.9,
                "views": {
                    str(v): {
                        "present": v in {0, 1},
                        "bbox_1000": [100, 200, 400, 800] if v in {0, 1} else None,
                        "center_1000": [250, 500] if v in {0, 1} else None,
                        "confidence": 0.8 if v in {0, 1} else 0.0,
                    }
                    for v in range(8)
                },
                "relations": [],
            }
        ],
    }
    assert validate_multiview_oracle_payload_v1(payload) is payload
    broken = {**payload, "instances": [{**payload["instances"][0], "label": "dragon_wing"}]}
    with pytest.raises(ValueError):
        validate_multiview_oracle_payload_v1(broken)
