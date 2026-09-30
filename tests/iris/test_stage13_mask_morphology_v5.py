from __future__ import annotations

import numpy as np

from models.iris.v5.stage13_mask_morphology_v5 import (
    diagnose_mask_pair_v5,
    error_distance_report_v5,
    overlap_metrics_v5,
)


def _disk(n: int = 129, radius: float = 30.0) -> np.ndarray:
    yy, xx = np.mgrid[:n, :n]
    c = (n - 1) / 2.0
    return ((xx - c) ** 2 + (yy - c) ** 2) <= radius ** 2


def test_identical_masks_have_no_error_and_unit_overlap():
    src = _disk()
    d = diagnose_mask_pair_v5(src, src)
    assert d["overlap"]["recall"] == 1.0
    assert d["overlap"]["precision"] == 1.0
    assert d["overlap"]["predicted_to_source_area_ratio"] == 1.0
    assert d["distance"]["false_positive"]["count"] == 0
    assert d["distance"]["false_negative"]["count"] == 0


def test_local_outward_band_is_localized_by_distance_report():
    src = _disk()
    from scipy import ndimage
    outside = ndimage.distance_transform_edt(~src)
    pred = src | ((~src) & (outside <= 3.0))
    m = overlap_metrics_v5(src, pred)
    assert m["recall"] == 1.0
    assert m["precision"] < 1.0
    r = error_distance_report_v5(src, pred)
    assert r["false_positive"]["count"] > 0
    assert r["false_positive"]["within_4px_fraction"] == 1.0
    assert r["false_positive"]["p95"] <= 3.0


def test_remote_false_positive_is_not_misclassified_as_boundary_band():
    src = _disk()
    pred = src.copy()
    pred[3:8, 3:8] = True
    r = error_distance_report_v5(src, pred)
    assert r["false_positive"]["count"] == 25
    assert r["false_positive"]["within_8px_fraction"] == 0.0
    assert r["false_positive"]["beyond_max_band_fraction"] == 1.0


def test_erosion_sweep_is_diagnostic_and_does_not_mutate_input():
    src = _disk()
    from scipy import ndimage
    outside = ndimage.distance_transform_edt(~src)
    pred = src | ((~src) & (outside <= 2.0))
    before = pred.copy()
    d = diagnose_mask_pair_v5(src, pred)
    assert np.array_equal(pred, before)
    rows = d["prediction_erosion_sweep"]
    assert rows[0]["erosion_radius_px"] == 0.0
    assert rows[0]["precision"] < 1.0
    assert rows[-1]["predicted_foreground_pixels"] < rows[0]["predicted_foreground_pixels"]
