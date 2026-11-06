from __future__ import annotations

import numpy as np

from tools.research.run_multiview_semantic_oracle_v4_r8 import (
    apply_cross_instance_collision_court,
    collision_action,
)


def _mask(x0: int, y0: int, x1: int, y1: int):
    m = np.zeros((32, 32), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def _payload(*items):
    return {
        "schema": "RealSaS.MultiviewMaskAdjudication.v1",
        "instances": [
            {
                "instance_id": iid,
                "label": label,
                "verdict": "PASS",
                "semantic_match": True,
                "confidence": 0.98,
                "bad_views": [],
                "reason_code": "GOOD",
            }
            for iid, label in items
        ],
    }


def test_collision_action_rejects_same_label_distinct_instances():
    assert collision_action("arm", "arm", iou=0.99) == ("A", "B")


def test_collision_action_prefers_specific_hierarchical_label():
    assert collision_action("head", "helmet", iou=0.99) == ("A",)
    assert collision_action("helmet", "head", iou=0.99) == ("B",)
    assert collision_action("body", "armor", iou=0.95) == ("A",)
    assert collision_action("arm", "hand", iou=0.95) == ("A",)


def test_collision_action_rejects_incompatible_cross_label_pair():
    assert collision_action("arm", "sword", iou=0.99) == ("A", "B")
    assert collision_action("hand", "sword", iou=0.99) == ("A", "B")


def test_collision_action_ignores_below_threshold_pair():
    assert collision_action("arm", "sword", iou=0.50) == ()


def test_r7_observed_arm_hand_sword_collapse_is_fail_closed():
    shared = _mask(4, 10, 28, 14)
    # Slight changes keep IoU above 0.90, mirroring the R7 V4 collapse where
    # arm/hand/sword masks were ~0.99 IoU.
    arm = shared.copy()
    hand = shared.copy()
    sword = shared.copy()
    rows = [
        {"instance_id": "3", "label": "arm", "view_index": 4, "mask": arm},
        {"instance_id": "5", "label": "hand", "view_index": 4, "mask": hand},
        {"instance_id": "10", "label": "sword", "view_index": 4, "mask": sword},
    ]
    out = apply_cross_instance_collision_court(
        _payload(("3", "arm"), ("5", "hand"), ("10", "sword")), rows
    )
    by_id = {x["instance_id"]: x for x in out["instances"]}
    # arm loses to hand under hierarchy, and sword collision makes hand/sword
    # both fail. Net result: none of the three may authorize V4 evidence.
    assert 4 in by_id["3"]["bad_views"]
    assert 4 in by_id["5"]["bad_views"]
    assert 4 in by_id["10"]["bad_views"]
    assert out["cross_instance_collision_court"]["collision_count"] == 3


def test_head_helmet_collision_keeps_helmet_view_only():
    m = _mask(8, 2, 24, 16)
    rows = [
        {"instance_id": "2", "label": "head", "view_index": 3, "mask": m},
        {"instance_id": "9", "label": "helmet", "view_index": 3, "mask": m.copy()},
    ]
    out = apply_cross_instance_collision_court(_payload(("2", "head"), ("9", "helmet")), rows)
    by_id = {x["instance_id"]: x for x in out["instances"]}
    assert 3 in by_id["2"]["bad_views"]
    assert 3 not in by_id["9"]["bad_views"]
