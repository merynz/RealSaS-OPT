"""No false PASS for transparent Cycles transport or fabricated alpha evidence."""
import numpy as np
import pytest

from tools.corpus.court_corpus370_v18b_transparency_bounce_witness_v1 import (
    ORIGINAL_TOL,VARIANTS,STATUS,image_alpha_metrics
)


def blank():
    return np.zeros((192,192,4),np.float32)


def test_true_hidden_empty_has_zero_alpha_everywhere():
    row=image_alpha_metrics(blank())
    assert row["max_alpha"]==0
    assert row["alpha_over_original_threshold_pixel_count"]==0
    assert not row["alpha_over_original_threshold"]


def test_one_stochastic_path_leak_above_025_is_detected():
    a=blank()
    a[31,40,3]=1/32
    metrics=image_alpha_metrics(a)
    assert metrics["alpha_over_original_threshold_pixel_count"]==1
    assert metrics["alpha_over_original_threshold"]
    assert metrics["alpha_p99"]==0
    assert metrics["max_alpha"]==1/32


def test_low_bounce_alpha_sample_quantization_does_not_become_artist_owner():
    a=blank()
    a[17,20,3]=23/32
    v=image_alpha_metrics(a)
    assert v["max_alpha"]==23/32
    assert v["alpha_over_original_threshold_pixel_count"]==1
    assert "NO_ARTIST_AUTHORITY" in STATUS
    assert VARIANTS==("transparent_shader_original_bounce_limit",
                      "transparent_shader_high_bounce_limit",
                      "physically_hidden_soup")


@pytest.mark.parametrize("attack",["nan","inf","negative_alpha","over_alpha","wrong_shape"])
def test_invalid_raster_is_not_courted(attack):
    x=blank()
    if attack=="nan":x[0,0,0]=np.nan
    if attack=="inf":x[0,0,0]=np.inf
    if attack=="negative_alpha":x[0,0,3]=-.5
    if attack=="over_alpha":x[0,0,3]=1.5
    if attack=="wrong_shape":x=x[:2]
    with pytest.raises(ValueError,match="V18B_BOUNCE_COURT_INVALID_SOURCE_RGBA"):
        image_alpha_metrics(x)


def test_original_threshold_is_not_relaxed():
    assert ORIGINAL_TOL==.025
