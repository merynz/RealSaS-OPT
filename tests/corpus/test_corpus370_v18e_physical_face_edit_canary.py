"""V18e archived mechanics original 3D triangle edit canary redteam, no layer gold."""
from __future__ import annotations
import numpy as np
import pytest
from tools.corpus.court_corpus370_v18e_physical_face_edit_canary import (
    TOL,ARMS,SCHEMA,PASS,image_delta_metrics
)

def image(alpha=1):
    frame=np.zeros((192,192,4),np.float32)
    frame[:,:,:3]=(.3,.45,.2)
    frame[:,:,3]=alpha
    return frame

def test_real_source_noop_zero_effect():
    original=image()
    d=image_delta_metrics(original,original.copy())
    assert d["raw_max_channel_p99"]==0
    assert d["raw_max_channel_over_tol_pixels"]==0
    assert d["source_premultiplied_rgba_effect_pixels"]==0
    assert not d["source_effect_exists"]

def test_physical_face_removal_produces_real_alpha_change():
    original=image()
    removed=original.copy()
    removed[10:30,40:60,3]=0
    d=image_delta_metrics(original,removed)
    assert d["alpha_abs_over_tol_pixels"]==400
    assert d["source_premultiplied_rgba_effect_pixels"]==400
    assert d["source_effect_exists"]

def test_hidden_transparent_rgb_not_mistaken_for_edit():
    a=image(0);b=a.copy()
    b[10:30,40:60,0]=.9
    d=image_delta_metrics(a,b)
    assert d["raw_max_channel_over_tol_pixels"]==400
    assert d["alpha_abs_over_tol_pixels"]==0
    assert d["source_premultiplied_rgba_effect_pixels"]==0

def test_single_bad_pixel_retained_in_metrics_beyond_p99():
    a=image();b=a.copy();b[0,0,0]=1.
    d=image_delta_metrics(a,b)
    assert d["raw_max_channel_p99"]==0
    assert d["raw_max_channel_over_tol_pixels"]==1
    assert d["source_premultiplied_rgba_effect_pixels"]==1

@pytest.mark.parametrize("attack",("nan","inf","wrong_shape","wrong_channel"))
def test_corrupted_images_fail_closed(attack):
    a=image();b=a.copy()
    if attack=="nan":b[0,0,0]=np.nan
    if attack=="inf":b[0,0,0]=np.inf
    if attack=="wrong_shape":b=b[:100]
    if attack=="wrong_channel":b=b[...,:3]
    with pytest.raises(ValueError,match="V18E_COURT_RGBA_ARRAY"):
        image_delta_metrics(a,b)

def test_no_artist_or_training_claims_from_canary():
    assert TOL==.025
    assert len(ARMS)==8 and len(set(ARMS))==8
    assert ARMS[0:4]==("SOURCE_ORIGINAL","SOUP_FULL","SOUP_NOOP","SOUP_EMPTY")
    assert "OWNER_UNKNOWN" in PASS
    assert SCHEMA.endswith(".v18e")
