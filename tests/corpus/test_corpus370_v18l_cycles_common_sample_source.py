"""V18l same-index Blender 1spp source compositor redteam (numpy only)."""
import numpy as np
import pytest

from tools.corpus.court_corpus370_v18l_cycles_common_sample_source import (
    ARMS,INDICES,TOL,SCHEMA,STATUS,over,premul,per_source_view,
)

RED=(.9,.08,.035)
BLUE=(.04,.14,.92)


def frames():
    return np.zeros((8,3,192,192,4),dtype=np.float32)


def stamp(a,kind,y=30,x=40,alpha=1.):
    a[:,y,x,:3]=kind
    a[:,y,x,3]=alpha


def test_full_iso_remove_identical_index_same_source_sample_is_exact():
    f=frames()
    # Eight sample indices; four have only source G, four have only source R.
    for i in range(8):
        c=RED if i<4 else BLUE
        arm=1 if i<4 else 2
        f[i,0,30,40]=(*c,1.)
        f[i,arm,30,40]=(*c,1.)
    result=per_source_view(f)
    assert result["same_index_sample_best_order_failed_pixel_instances"]==0
    assert result["same_index_sample_alpha_invariant_failed_pixel_instances"]==0
    assert result["source_8_sample_full_mean_nonempty_pixels"]==1
    assert result["source_8_sample_premultiplied_joint_pre_filter_vs_postfilter_best_order_pixel_failures"]==1
    assert result["source_8_sample_postfilter_best_order_error_mean"]>0
    assert result["source_8_sample_pre_filter_best_fixed_order_pixel_failures"]==1
    assert result["source_raw_1spp_alpha_nonbinary_pixel_instances"]==0


def test_same_sample_index_can_have_group_and_remainder_at_different_depths():
    f=frames()
    for i in range(8):
        c=RED if i%2==0 else BLUE
        f[i,0,30,40]=(*c,1.)
        f[i,1,30,40]=(*RED,1.)
        f[i,2,30,40]=(*BLUE,1.)
    out=per_source_view(f)
    assert out["same_index_sample_best_order_failed_pixel_instances"]==0
    assert out["source_8_sample_pre_filter_best_fixed_order_pixel_failures"]==1
    assert out["source_8_sample_premultiplied_joint_pre_filter_vs_postfilter_best_order_pixel_failures"]==1


def test_shifted_or_desynchronized_subset_must_not_count_as_parity():
    f=frames()
    stamp(f[:,0],RED)
    stamp(f[:,2],BLUE)
    out=per_source_view(f)
    assert out["same_index_sample_best_order_failed_pixel_instances"]>0
    assert out["same_index_sample_alpha_invariant_failed_pixel_instances"]==0
    # A source-subset checksum alone cannot falsely claim camera-sample match.


def test_separated_opaque_masks_consistent_joint_alpha_but_not_postfiltered():
    f=frames()
    for i in range(8):
        if i%2==0:
            f[i,0,30,40]=(*RED,1.)
            f[i,1,30,40]=(*RED,1.)
        else:
            f[i,0,30,40]=(*BLUE,1.)
            f[i,2,30,40]=(*BLUE,1.)
    out=per_source_view(f)
    assert out["same_index_sample_best_order_failed_pixel_instances"]==0
    # Source alpha 1; independently averaged G and R alpha .5
    # leading to .75 under postfilter OVER (wrong).
    assert out["source_8_sample_premultiplied_joint_pre_filter_vs_postfilter_best_order_pixel_failures"]==1


@pytest.mark.parametrize("bad",("nan","negative_alpha","large_alpha","wrong_shape"))
def test_invalid_original_renderer_samples_fail_closed(bad):
    f=frames()
    if bad=="nan":f[0,0,20,10,0]=np.nan
    if bad=="negative_alpha":f[0,0,20,10,3]=-.3
    if bad=="large_alpha":f[0,0,20,10,3]=1.5
    if bad=="wrong_shape":f=f[:4]
    with pytest.raises(ValueError,match="V18L_"):
        per_source_view(f)


def test_distinguish_blender_subset_measurement_from_artist_gold():
    assert len(INDICES)==8
    assert INDICES==(0,4,8,12,16,20,24,28)
    assert ARMS==("FULL","ISOLATE","REMOVE")
    assert TOL==.025
    assert SCHEMA.endswith(".v18l")
    assert "SOURCE_ARTIST_UNKNOWN" in STATUS
