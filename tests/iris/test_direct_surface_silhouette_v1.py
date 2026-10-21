from __future__ import annotations

import numpy as np

from tools.corpus.iris_observation_admission_v1 import project_mask
from tools.iris_vnext.audit_fit8_direct_surface_silhouette_v1 import _mask_metrics


def _camera(resolution: int = 64):
    return {
        "origin": [0.0, 0.0, 2.0],
        "right": [1.0, 0.0, 0.0],
        "screen_up": [0.0, 1.0, 0.0],
        "forward": [0.0, 0.0, -1.0],
        "half_extent": 1.0,
        "resolution": resolution,
        "view_index": 0,
    }


def test_mask_metrics_perfect_and_missing_candidate_are_measured_not_promoted():
    truth = np.zeros((8, 8), dtype=bool)
    truth[2:6, 2:6] = True
    perfect = _mask_metrics(truth, truth)
    assert perfect["iou"] == 1.0
    assert perfect["precision"] == 1.0
    assert perfect["recall"] == 1.0
    assert perfect["symmetric_maximum_mask_gap_px"] == 0.0

    missing = _mask_metrics(np.zeros_like(truth), truth)
    assert missing["iou"] == 0.0
    assert missing["recall"] == 0.0
    assert missing["predicted_mask_empty"] is True
    assert missing["symmetric_maximum_mask_gap_px"] > 0.0


def test_world_inverse_and_stage01_raster_semantics_are_exactly_reusable():
    center = np.asarray([0.25, -0.5, 1.25], dtype=np.float64)
    half = 2.0
    canonical = np.asarray(
        [
            [-0.25, -0.25, 0.0],
            [0.25, -0.25, 0.0],
            [0.0, 0.25, 0.0],
        ],
        dtype=np.float64,
    )
    world = center[None, :] + half * canonical
    recovered = (world - center[None, :]) / half
    np.testing.assert_allclose(recovered, canonical, atol=0.0, rtol=0.0)

    camera = _camera()
    # Projection is translation-relative to camera origin; construct the same
    # triangle directly in camera/source world and verify deterministic raster.
    camera["origin"] = center.tolist()
    predicted_a = project_mask(world, np.asarray([[0, 1, 2]], dtype=np.int64), camera)
    predicted_b = project_mask(center[None, :] + half * recovered, np.asarray([[0, 1, 2]], dtype=np.int64), camera)
    assert predicted_a.any()
    assert np.array_equal(predicted_a, predicted_b)
    assert _mask_metrics(predicted_a, predicted_b)["iou"] == 1.0
