"""Original 3-color physically edited source sample independent court redteam."""
import numpy as np
import pytest

from tools.corpus.court_corpus370_original_source_k3_cycles_palette_edits_v1 import measure


def synthetic():
    # Actual producer arm ordering is intentionally frozen by this test.
    # Scene is 3 opaque colored planar strips with a second occluded group
    # at some pixels. FULL is independently authored, NOT compositor output.
    arr=np.zeros((8,13,192,192,4),np.float32)
    colors=((.85,.12,.05,1.),(.05,.7,.16,1.),(.13,.12,.87,1.))
    for x in range(192):
        front=x%3
        behind=(front+1)%3
        for k in (front,behind):
            arr[:,1+k,:,x]=colors[k]
            arr[:,4+k,:,x]=(0.2 if k==front else 0.6,0,0,1)
        arr[:,0,:,x]=colors[front]
        for edit in range(3):
            new=tuple(.15 if v>=.5 else .85 for v in colors[edit][:3])+(1.,)
            if edit==front:
                arr[:,7+edit,:,x]=new
            else:
                arr[:,7+edit,:,x]=colors[edit]
            for old in range(3):
                # edited isolate is empty where edit material is absent
                if old!=edit:continue
                if old!=front and old!=behind:
                    arr[:,7+old,:,x]=0
            arr[:,10+edit,:,x]=new if edit==front else colors[front]
    return arr


def test_all_three_original_source_palette_edits_reconstruct_sample_first():
    r=measure(synthetic())
    assert r["source_multi_group_overlapping_sample_count"]>100000
    assert r["source_unknown_depth_tie_sample_count"]==0
    assert all(row["sample_rgba_mismatch_count"]==0 for row in r["metric_rows"])
    assert all(row["eight_sample_mean_rgba_mismatch_count"]==0 for row in r["metric_rows"])
    assert all(row["physical_full_edit_changed_sample_pixel_count"]>0
               for row in r["metric_rows"][1:])


def test_forged_edited_full_target_causes_failure():
    arr=synthetic()
    arr[0,10,10,12,:3]=0
    result=measure(arr)
    assert result["metric_rows"][1]["sample_rgba_mismatch_count"]>=1


def test_source_depth_tie_is_unknown_and_not_force_ordered():
    arr=synthetic()
    arr[0,5,10,12,0]=arr[0,4,10,12,0]+.0001
    result=measure(arr)
    assert result["source_unknown_depth_tie_sample_count"]>=1
    assert result["metric_rows"][0]["unknown_source_samples"]>=1


@pytest.mark.parametrize("case",["alpha","nan","shape"])
def test_invalid_original_source_sample_shapes_fail(case):
    arr=synthetic()
    if case=="alpha":arr[0,4,0,0,3]=.3
    elif case=="nan":arr[0,2,0,0,1]=np.nan
    else:arr=arr[:7]
    with pytest.raises(ValueError,match="SAMPLE_K"):
        measure(arr)
