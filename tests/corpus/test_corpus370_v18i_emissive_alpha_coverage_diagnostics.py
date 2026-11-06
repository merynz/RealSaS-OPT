"""V18i alpha identity / source subpixel coverage adverse examples."""
import numpy as np
import pytest

from tools.corpus.court_corpus370_v18i_emissive_alpha_coverage_diagnostics import (
    alpha_coverage_probe,SCHEMA,STATUS,TOL,
)


def blank():
    return np.zeros((192,192,4),dtype=np.float32)


def patch(rgb,alpha):
    a=blank()
    a[20:30,30:40,:3]=rgb
    a[20:30,30:40,3]=alpha
    return a


def test_over_alpha_identity_is_independent_of_layer_order():
    front=patch((.9,.08,.035),1.0)
    back=patch((.04,.14,.92),1.0)
    full=front.copy()
    out=alpha_coverage_probe(full,front,back)
    assert out["order_invariant_OVER_alpha_mismatch_pixels"]==0
    assert out["per_pixel_two_order_oracle_failed_pixels"]==0
    assert out["order_discriminating_overlap_pixels"]==100


def test_disjoint_opaque_subpixel_samples_falsify_postfilter_over_without_artist_label():
    # At every subpixel exactly one of two opaque colors is present.
    # The full filtered pixel is opaque, but each independently filtered
    # single-part image has alpha 0.5. Filtering then OVER gives 0.75,
    # different from rendering the union then filtering alpha 1.0.
    red=(.9,.08,.035)
    blue=(.04,.14,.92)
    full=patch(tuple((x+y)/2 for x,y in zip(red,blue)),1.0)
    group=patch(red,0.5)
    remainder=patch(blue,0.5)
    out=alpha_coverage_probe(full,group,remainder)
    assert out["order_invariant_OVER_alpha_mismatch_pixels"]==100
    assert out["per_pixel_oracle_fail_with_ALPHA_mismatch_pixels"]==100
    assert out["per_pixel_oracle_fail_without_alpha_mismatch_pixels"]==0
    assert out["inclusion_exclusion_union_bounds_warning_pixels"]==0
    assert out["alpha_mismatch_max"]==pytest.approx(.25)


def test_no_overlap_and_remainder_empty_is_not_drawable_gold():
    only=patch((.9,.08,.035),1.0)
    zero=blank()
    d=alpha_coverage_probe(only,only,zero)
    assert d["order_invariant_OVER_alpha_mismatch_pixels"]==0
    assert d["order_discriminating_overlap_pixels"]==0
    assert d["per_pixel_two_order_oracle_failed_pixels"]==0
    assert d["requires_pixelwise_source_subsample_proof_for_aa_causality"]


def test_rgb_order_failure_with_valid_alpha_identity():
    red=patch((.9,.08,.035),1.0)
    blue=patch((.04,.14,.92),1.0)
    unseen=patch((.5,.5,.5),1.0)
    out=alpha_coverage_probe(unseen,red,blue)
    assert out["order_invariant_OVER_alpha_mismatch_pixels"]==0
    assert out["per_pixel_oracle_fail_without_alpha_mismatch_pixels"]==100


@pytest.mark.parametrize("failure",("nan","inf","negative_alpha","excess_alpha","shape"))
def test_invalid_source_rgba_must_fail_closed(failure):
    full=blank()
    if failure=="nan":full[0,0,0]=np.nan
    if failure=="inf":full[0,0,1]=np.inf
    if failure=="negative_alpha":full[0,0,3]=-.2
    if failure=="excess_alpha":full[0,0,3]=1.2
    if failure=="shape":full=full[:100]
    with pytest.raises(ValueError,match="V18I_INVALID_SEALED_RGBA"):
        alpha_coverage_probe(full,blank(),blank())


def test_no_new_art_authority_and_original_tolerance():
    assert SCHEMA.endswith(".v18i")
    assert "NO_DRAWABLE_TRUTH" in STATUS
    assert TOL==.025
