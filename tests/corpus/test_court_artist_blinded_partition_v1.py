"""Blind real-artist layer benchmark court contracts. Synthetic TEST ONLY.

No synthetic test is a real original artist label / proof; metrics-only invariant.
"""
from __future__ import annotations
import copy
import numpy as np
import pytest
from tools.corpus.court_artist_blinded_partition_v1 import (
    evaluate,array_digest,SCHEMA,SOURCE_KIND,PREDICTION_KIND
)

def make(ids,pred,mask=None,weights=None):
    ids=np.asarray(ids,dtype=np.int64)
    pred=np.asarray(pred,dtype=np.int64)
    if mask is None:mask=np.ones(len(ids),bool)
    if weights is None:weights=np.ones(len(ids),np.float64)
    artist={
        "source_kind":SOURCE_KIND,
        "reference_from_original_artist_project":True,
        "human_authored_not_generated_proxy":True,
        "artist_drawable_ids_hidden_from_algorithm":True,
        "author_project_SHA256":"a"*64,
        "original_merged_surface_SHA256":"b"*64,
        "element_address_SHA256":"c"*64,
        "artist_label_array_SHA256":array_digest(ids),
        "source_composite_parity_qualified":True,
        "layer_semantics_authoritative_only_for_court":True,
        "training_authorized":False
    }
    candidate={
        "prediction_kind":PREDICTION_KIND,
        "original_merged_surface_SHA256":"b"*64,
        "element_address_SHA256":"c"*64,
        "anonymous_partition_array_SHA256":array_digest(pred),
        "author_slot_mesh_material_names_exposed":False,
        "author_layer_count_exposed":False,
        "layer_reference_used_in_training":False,
        "artist_label_access_before_prediction":False,
        "precommitted_before_artist_reference_opened":True,
        "training_authorized":False
    }
    return ids,pred,np.asarray(weights),np.asarray(mask,dtype=bool),artist,candidate

def test_one_to_one_opaque_artist_partition_permutation_unchanged():
    args=make([0,0,1,1,2,2],[3,3,100,100,5,5])
    result=evaluate(*args)
    assert result["hungarian_mean_matched_iou"]==pytest.approx(1)
    assert result["weighted_variation_of_information_bits"]==pytest.approx(0,abs=1e-9)
    assert result["prediction_observable_weight_coverage"]==1
    assert result["author_layers_with_significant_algorithm_splits"]==0
    assert result["predicted_regions_merging_significant_author_layers"]==0
    assert result["graphics_owner_authorized"] is False
    assert result["training_authorized"] is False

def test_excessive_geometric_oversegmentation_is_not_perfect_artist_match():
    r=evaluate(*make([0,0,1,1],[10,11,12,13]))
    assert r["hungarian_mean_matched_iou"]<1.
    assert r["author_layers_with_significant_algorithm_splits"]==2
    assert r["predicted_regions_merging_significant_author_layers"]==0

def test_one_source_mesh_proxy_for_every_layer_is_wrong():
    r=evaluate(*make([0,0,1,1],[99,99,99,99]))
    assert r["hungarian_mean_matched_iou"]<1.
    assert r["predicted_regions_merging_significant_author_layers"]==1

def test_unknown_can_be_accurate_but_must_report_lost_coverage():
    r=evaluate(*make([0,0,1,1],[10,10,-1,-1]))
    assert r["prediction_observable_weight_coverage"]==.5
    assert r["hungarian_mean_matched_iou"]==1.
    assert r["training_authorized"] is False

def test_geometry_area_weighted_evaluation_changes_score():
    equal=evaluate(*make([0,0,1,1],[10,11,20,20]))
    weighted=evaluate(*make([0,0,1,1],[10,11,20,20],weights=[15,1,1,1]))
    assert equal["weighted_variation_of_information_bits"]!=weighted["weighted_variation_of_information_bits"]

@pytest.mark.parametrize("attack",[
    "bone_labels_as_truth","semantic_material_leak",
    "artist_prediction_leak","wrong_source_sha",
    "incorrect_flattening","training_reference","training_prediction",
    "fake_algorithm_precommit","counterfeit_author",
    "untyped_3d_object_proxy","group_count_leaked","forged_artist_array",
    "forged_algorithm_array","mismatched_element_ids"
])
def test_forgeries_and_prior_leakage_fail_closed(attack):
    args=list(make([0,0,1,1],[10,10,20,20]))
    a=args[4];p=args[5]
    if attack=="bone_labels_as_truth":a["source_kind"]="SOURCE_3D_MESH_COMPONENT_PROXY"
    elif attack=="semantic_material_leak":p["author_slot_mesh_material_names_exposed"]=True
    elif attack=="artist_prediction_leak":p["artist_label_access_before_prediction"]=True
    elif attack=="wrong_source_sha":p["original_merged_surface_SHA256"]="d"*64
    elif attack=="incorrect_flattening":a["source_composite_parity_qualified"]=False
    elif attack=="training_reference":a["training_authorized"]=True
    elif attack=="training_prediction":p["training_authorized"]=True
    elif attack=="fake_algorithm_precommit":p["precommitted_before_artist_reference_opened"]=False
    elif attack=="counterfeit_author":a["human_authored_not_generated_proxy"]=False
    elif attack=="untyped_3d_object_proxy":a["reference_from_original_artist_project"]=False
    elif attack=="group_count_leaked":p["author_layer_count_exposed"]=True
    elif attack=="forged_artist_array":args[0]=np.array([0,1,0,1])
    elif attack=="forged_algorithm_array":args[1]=np.array([20,10,10,20])
    elif attack=="mismatched_element_ids":p["element_address_SHA256"]="x"*64
    with pytest.raises(ValueError):
        evaluate(*args)

@pytest.mark.parametrize("invalid",[
    ([0,0],[1,1],[-2.,1.],[True,True]),
    ([0,0],[1,1],[1.,1.],[False,False]),
    ([0,0],[1,1],[float("nan"),1.],[True,True])
])
def test_corrupt_unobservable_weight_domain_fails(invalid):
    ids,labels,weights,mask=invalid
    args=list(make(ids,labels,mask,weights))
    with pytest.raises(ValueError):
        evaluate(*args)

def test_unrelated_artist_layer_k_is_not_fixed_model_parameter():
    ids=[5,5,8,8,8,9,9,14]
    pred=[20,20,40,40,40,100,100,110]
    result=evaluate(*make(ids,pred))
    assert result["author_layer_count_in_evaluated_domain"]==4
    assert result["anonymous_prediction_region_count_in_evaluated_domain"]==4
    assert result["hungarian_mean_matched_iou"]==pytest.approx(1.)
    assert result["artist_layer_count_leaked_to_algorithm"] is False
