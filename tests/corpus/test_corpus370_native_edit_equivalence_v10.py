"""Adversarial V10 source edit-equivalence tests, synthetic only."""
import copy
import numpy as np
import pytest
from tools.corpus.court_corpus370_native_edit_equivalence_v10 import evaluate

def fixture(same=False):
    sizes=[8,16]
    arms=["SOURCE_ORIGINAL","SOUP_FULL","SOUP_NOOP","REMOVE_ALL_GROUPS",
          "FINE_REMOVE_T00","FINE_ISOLATE_T00",
          "GROUP_REMOVE_K008_T00","GROUP_ISOLATE_K008_T00",
          "GROUP_REMOVE_K016_T00","GROUP_ISOLATE_K016_T00"]
    names={k:i for i,k in enumerate(arms)}
    data=np.zeros((3,4,len(arms),192,192,4),np.float32)
    baseline=np.zeros((192,192,4),np.float32)
    baseline[30:110,30:110,:]=(.2,.3,.4,1.)
    for name in ("SOURCE_ORIGINAL","SOUP_FULL","SOUP_NOOP"):
        data[:,:,names[name]]=baseline
    fine=baseline.copy()
    fine[50:60,50:60,:]=0.
    coarse=fine.copy()
    coarse[60:65,60:70,:]=0.
    tiny=np.zeros_like(baseline)
    tiny[50:60,50:60,:]=baseline[50:60,50:60,:]
    big=tiny.copy()
    big[60:65,60:70,:]=baseline[60:65,60:70,:]
    if same:coarse=fine.copy();big=tiny.copy()
    for name,arr in [
        ("FINE_REMOVE_T00",fine),("FINE_ISOLATE_T00",tiny),
        ("GROUP_REMOVE_K008_T00",coarse),("GROUP_ISOLATE_K008_T00",big),
        ("GROUP_REMOVE_K016_T00",fine),("GROUP_ISOLATE_K016_T00",tiny),
    ]:data[:,:,names[name]]=arr
    manifest={
        "schema":"RealSaS.Corpus370NativeAnimatedPartitionSourceRender.v9",
        "status":"REAL_AUTHORED_NATIVE_POSE_ANIMATED_RGBA_RENDER__INDEPENDENT_COURT_REQUIRED",
        "source_sha256":"a"*64,"subject_id":"only-synthetic-source",
        "teacher_only":True,"graphics_owner_authorized":False,
        "training_authorized":False,"pose_indices":[0,1,2],
        "view_indices":[0,2,4,6],
        "candidate_sizes":sizes,"arms":arms,
        "plan":{"tasks":[{"task_index":0}],
                "finest_available_reference_size":16,"task_count":1},
    }
    court={
        "schema":"RealSaS.Corpus370NativeAnimatedPartitionRasterCourt.v9",
        "source_sha256":"a"*64,"subject_id":"only-synthetic-source",
        "all_pose_view_source_parity":True,"source_soup_parity_qualified_pose_views":12,
        "unqualified_pose_views":0,
        "graphics_owner_authorized":False,"training_authorized":False,
        "same_drawable_positive_pairs":0,
        "different_drawable_negative_pairs":0,
        "available_group_sizes":sizes,"source_face_proxy_task_count":1,
    }
    return data,manifest,court


def test_direct_pairwise_counterexample_not_finest_truth():
    p,m,c=fixture()
    d=evaluate(p,m,c)
    assert d["source_edit_task_count"]==1
    comparison=d["pairwise_equivalence_classes_scoped_to_tested_source_edits"][0]
    assert comparison["direct_difference_pixel_instances"]>0
    assert comparison["isolated_difference_pixel_instances"]>0
    assert comparison["finite_test_equivalence_class_member"] is False
    assert d["optimal_group_count_authorized"] is False
    assert d["artist_same_drawable_positive_pairs"]==0
    assert d["training_authorized"] is False


def test_identical_edit_response_is_only_finite_scoped_not_drawable_truth():
    p,m,c=fixture(same=True)
    d=evaluate(p,m,c)
    comparison=d["pairwise_equivalence_classes_scoped_to_tested_source_edits"][0]
    assert comparison["finite_test_equivalence_class_member"] is True
    assert comparison["direct_difference_pixel_instances"]==0
    assert comparison["isolated_difference_pixel_instances"]==0
    assert d["exact_response_equality_in_unseen_actions_or_cameras"] is False
    assert d["graphics_owner_authorized"] is False
    assert d["finest_partition_is_artist_truth"] is False


@pytest.mark.parametrize("fault",[
    "forged_training","forged_graphics","forged_source","missing_pose",
    "reference_parity_absent","court_mismatch","fake_owner_positive",
    "wrong_edit_arms","nonfinite_pixel","invalid_noop","nonzero_alpha",
    "no_real_fine_effect","missing_isolation",
])
def test_counterfeit_or_invisible_source_evidence_cannot_claim_equivalence(fault):
    p,m,c=copy.deepcopy(fixture(same=True))
    if fault=="forged_training":m["training_authorized"]=True
    elif fault=="forged_graphics":c["graphics_owner_authorized"]=True
    elif fault=="forged_source":m["source_sha256"]="b"*64
    elif fault=="missing_pose":m["pose_indices"]=[0,1]
    elif fault=="reference_parity_absent":c["source_soup_parity_qualified_pose_views"]=11
    elif fault=="court_mismatch":c["source_edit_task_count"]=19;c["source_face_proxy_task_count"]=19
    elif fault=="fake_owner_positive":c["same_drawable_positive_pairs"]=1
    elif fault=="wrong_edit_arms":m["arms"]=m["arms"][:-1]
    elif fault=="nonfinite_pixel":p[0,0,0,50,50,0]=np.nan
    elif fault=="invalid_noop":p[0,0,2,50,50,0]=.8
    elif fault=="nonzero_alpha":p[0,0,0,50,50,3]=2.
    elif fault=="no_real_fine_effect":
        p[:,:,m["arms"].index("FINE_REMOVE_T00")]=p[:,:,m["arms"].index("SOUP_FULL")]
    elif fault=="missing_isolation":
        p[:,:,m["arms"].index("FINE_ISOLATE_T00")]=0.
    with pytest.raises((ValueError,KeyError)):
        evaluate(p,m,c)
