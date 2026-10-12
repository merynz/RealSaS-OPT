from __future__ import annotations

import numpy as np

from compiler.realsas_compiler_core.semantic_relation_evidence_v2 import (
    calibrate_label_view_scores_v2,
)


def test_single_view_evidence_abstains_even_when_only_label():
    scores = np.zeros((1, 1, 8), dtype=np.float64)
    scores[0, 0, 3] = 0.95
    calibrated, confidence, label_index, support, counts = calibrate_label_view_scores_v2(
        scores,
        minimum_face_confidence=0.45,
        minimum_label_support_views=2,
    )
    assert calibrated[0, 0] == 0.95
    assert confidence[0] == 0.95
    assert label_index.tolist() == [-1]
    assert not support.any()
    assert counts.tolist() == [[1]]


def test_two_view_absolute_evidence_qualifies_without_becoming_one():
    scores = np.zeros((1, 1, 8), dtype=np.float64)
    scores[0, 0, 2] = 0.80
    scores[0, 0, 3] = 0.90
    calibrated, confidence, label_index, support, counts = calibrate_label_view_scores_v2(
        scores,
        minimum_face_confidence=0.45,
        minimum_label_support_views=2,
    )
    assert np.isclose(calibrated[0, 0], 0.85)
    assert np.isclose(confidence[0], 0.85)
    assert label_index.tolist() == [0]
    assert support[0, 2] and support[0, 3]
    assert counts.tolist() == [[2]]


def test_competing_labels_are_penalized_by_dominance():
    scores = np.zeros((1, 2, 8), dtype=np.float64)
    scores[0, 0, 1] = 0.90
    scores[0, 0, 2] = 0.90
    scores[0, 1, 1] = 0.80
    scores[0, 1, 2] = 0.80
    calibrated, confidence, label_index, _, _ = calibrate_label_view_scores_v2(
        scores,
        minimum_face_confidence=0.45,
        minimum_label_support_views=2,
    )
    # label0 dominance = 1.8 / 3.4; absolute = .9 => ~.476
    assert np.isclose(calibrated[0, 0], (1.8 / 3.4) * 0.9)
    assert np.isclose(calibrated[0, 1], (1.6 / 3.4) * 0.8)
    assert label_index.tolist() == [0]
    assert confidence[0] < 0.50


def test_winning_label_support_is_label_specific():
    scores = np.zeros((1, 2, 8), dtype=np.float64)
    # Three semantic views exist in total, but the winning label appears in only one.
    scores[0, 0, 0] = 0.95
    scores[0, 1, 1] = 0.20
    scores[0, 1, 2] = 0.20
    _, _, label_index, support, counts = calibrate_label_view_scores_v2(
        scores,
        minimum_face_confidence=0.45,
        minimum_label_support_views=2,
    )
    assert counts.tolist() == [[1, 2]]
    assert label_index.tolist() == [-1]
    assert not support.any()


def test_invalid_support_requirement_fails_closed():
    scores = np.zeros((1, 1, 8), dtype=np.float64)
    for n in (0, 9):
        try:
            calibrate_label_view_scores_v2(scores, minimum_label_support_views=n)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid minimum_label_support_views must fail")
