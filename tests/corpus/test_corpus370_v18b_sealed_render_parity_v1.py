"""Adversarial retrospective diagnostic; zero new truth and no threshold relaxation."""
import numpy as np
import pytest

from tools.corpus.diagnose_corpus370_v18b_sealed_render_parity_v1 import (
    SCHEMA, STATUS, TOL, pair_metrics
)


def image():
    frame = np.zeros((192, 192, 4), dtype=np.float32)
    frame[...,:3] = (.4,.3,.2)
    frame[...,3] = 1.
    return frame


def test_sealed_parity_exact_copy():
    a = image()
    x = pair_metrics(a,a.copy())
    assert x["pixel_difference_at_original_threshold_pass"]
    assert x["effect_mask_empty"]
    assert x["alpha_abs_over_tolerance_pixel_count"] == 0
    assert x["premultiplied_rgb_over_tolerance_pixel_count"] == 0


def test_old_99th_percentile_gate_stays_exactly_original():
    a = image();b=a.copy()
    b[:20,:20,0] = .9
    x=pair_metrics(a,b)
    assert not x["pixel_difference_at_original_threshold_pass"]
    assert x["raw_max_channel_p99"] > TOL
    assert x["premultiplied_rgb_over_tolerance_pixel_count"] == 400
    assert not x["effect_mask_empty"]


def test_rgb_only_drift_separated_from_alpha_drift():
    a=image();b=a.copy()
    b[:20,:20,0] = .9
    x=pair_metrics(a,b)
    assert x["alpha_abs_over_tolerance_pixel_count"]==0
    assert x["premultiplied_rgb_over_tolerance_pixel_count"]==400
    assert x["raw_max_channel_over_tolerance_pixel_count"]==400


def test_alpha_only_mismatch_detected():
    a=image();b=a.copy()
    b[:20,:20,3] = .25
    x=pair_metrics(a,b)
    assert x["alpha_abs_over_tolerance_pixel_count"] == 400
    assert not x["pixel_difference_at_original_threshold_pass"]
    assert not x["effect_mask_empty"]


def test_transparent_rgb_transport_only_is_not_false_visual_effect():
    a=image();b=a.copy()
    a[...,3]=0.;b[...,3]=0.
    b[:20,:20,0] = .9
    x=pair_metrics(a,b)
    assert not x["pixel_difference_at_original_threshold_pass"]
    assert x["raw_difference_over_tolerance_but_no_visible_premul_or_alpha_pixel_count"] == 400
    assert x["premultiplied_rgb_over_tolerance_pixel_count"]==0
    assert x["effect_mask_empty"]


def test_rare_spot_cannot_hide_in_pixel_p99_summary():
    a=image();b=a.copy()
    b[0,0,0]=.9
    x=pair_metrics(a,b)
    assert x["pixel_difference_at_original_threshold_pass"]
    assert x["raw_max_channel_over_tolerance_pixel_count"]==1
    assert x["raw_max_channel_max"]>TOL


@pytest.mark.parametrize("attack",["nan","inf","wrong_size","missing_alpha"])
def test_invalid_original_frame_rejected(attack):
    a=image();b=a.copy()
    if attack=="nan": b[0,0,0]=np.nan
    if attack=="inf": b[0,0,0]=np.inf
    if attack=="wrong_size": b=b[:8]
    if attack=="missing_alpha": b=b[...,:3]
    with pytest.raises(ValueError,match="V18B_DIAGNOSTIC_INVALID_RGBA_PAIR"):
        pair_metrics(a,b)


def test_no_authority_and_immutable_original_tolerance():
    assert SCHEMA.endswith(".v1")
    assert "NO_AUTHORITY" in STATUS
    assert TOL==.025
