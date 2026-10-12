from __future__ import annotations

import numpy as np

from models.iris.semantic_provider_quality_v1 import (
    audit_semantic_proposals_v1,
    mask_iou,
)


def _row(view, label, mask, score=0.9):
    return {"view_index": view, "label": label, "mask": mask, "score": score}


def test_first_knight_failure_shape_is_rejected_as_provider_mask_collapse():
    subject = np.zeros((32, 32), dtype=bool)
    subject[4:28, 5:27] = True
    proposals = []
    for view in range(8):
        for label in ("body", "cape", "sword", "shield"):
            proposals.append(_row(view, label, subject.copy()))
    report = audit_semantic_proposals_v1(
        proposals,
        subject_masks=[subject] * 8,
        maximum_whole_subject_fraction=1.0,
    )
    assert report.status == "FAIL"
    assert report.failure_code == "PROVIDER_MASK_COLLAPSE"
    assert report.maximum_cross_label_iou == 1.0
    assert report.near_duplicate_cross_label_pair_count > 0


def test_whole_subject_dominance_is_independently_rejected():
    subject = np.zeros((32, 32), dtype=bool)
    subject[3:29, 3:29] = True
    proposals = [
        _row(view, label, subject.copy())
        for view, label in enumerate(("arm", "cape", "sword", "shield", "head", "hand"))
    ]
    report = audit_semantic_proposals_v1(
        proposals,
        subject_masks=[subject] * 8,
        maximum_duplicate_pair_fraction=1.0,
        maximum_whole_subject_fraction=0.25,
    )
    assert report.status == "FAIL"
    assert report.failure_code == "WHOLE_SUBJECT_MASK_DOMINANCE"


def test_localized_distinct_semantics_pass():
    subject = np.ones((40, 40), dtype=bool)
    arm = np.zeros_like(subject); arm[12:28, 5:12] = True
    cape = np.zeros_like(subject); cape[8:34, 14:27] = True
    sword = np.zeros_like(subject); sword[10:36, 30:34] = True
    proposals = []
    for view in range(8):
        proposals.extend([
            _row(view, "arm", arm),
            _row(view, "cape", cape),
            _row(view, "sword", sword),
        ])
    report = audit_semantic_proposals_v1(
        proposals,
        subject_masks=[subject] * 8,
    )
    assert report.status == "PASS"
    assert report.failure_code is None
    assert report.distinct_label_count == 3
    assert report.maximum_cross_label_iou < 0.97
    assert mask_iou(arm, arm) == 1.0
