"""Original-source topology component leakage tests (not artist gold truth)."""
import copy
import json
import pytest
from tools.corpus.author_spine_blind_flatten_v1 import flatten
from tools.corpus.artist_blind_topology_leakage_audit_v1 import predict,audit

@pytest.fixture()
def source():
    return {
      "bones":[{"name":"root"},{"name":"artist_arm","parent":"root","x":3.}],
      "slots":[{"name":"artist_head","bone":"root","attachment":"hat"},
               {"name":"artist_hand","bone":"artist_arm","attachment":"glove"}],
      "skins":[{"name":"default","attachments":{
        "artist_head":{"hat":{"type":"region","width":10.,"height":8.}},
        "artist_hand":{"glove":{"type":"region","width":3.,"height":4.}},
      }}],
      "animations":{"unused_walk":{"bones":{}}}
    }

def pieces(source):
    return flatten(source,source_bytes=json.dumps(source).encode(),seed=1001)

def test_unwelded_artist_source_disconnected_components_expose_exact_cuts(source):
    m,h,receipt=pieces(source)
    p=predict(m)
    result=audit(m,p,h)
    assert result["tautological_one_to_one_component_match"] is True
    assert result["source_topology_component_face_purity"]==1.
    assert result["source_slot_reference_face_purity"]==1.
    assert result["original_artist_setup_supported_slot_count_PRIVATE_COURT_ONLY"]==2
    assert result["anonymous_topological_component_count"]==2
    assert p["artist_reference_not_read"] is True
    assert p["original_appearance_image_input_used"] is False
    assert result["this_is_true_artist_drawable_recovery"] is False
    assert result["training_authorized"] is False

def test_prediction_files_do_not_contain_artist_names_or_layer_count(source):
    m,h,_=pieces(source)
    p=predict(m)
    serialized=json.dumps(p)
    for forbidden in ("artist_head","artist_hand","hat","glove","artist_arm",
                      "original_artist_setup_supported_slot_count"):
        assert forbidden not in serialized
    assert "private_reference" not in serialized

def test_same_predicted_component_can_merge_artist_source_parts_without_proving_owner(source):
    m,h,_=pieces(source)
    p=predict(m)
    p["face_partition_id"]=[0]*len(p["face_partition_id"])
    from tools.corpus.author_spine_blind_flatten_v1 import canonical_bytes,digest
    p["face_partition_SHA256"]=digest(canonical_bytes(p["face_partition_id"]))
    result=audit(m,p,h)
    assert result["topological_components_mixing_multiple_original_author_slots"]==1
    assert result["tautological_one_to_one_component_match"] is False
    assert result["graphics_owner_authorized"] is False

@pytest.mark.parametrize("attack",[
    "external_merged_sha","forged_reference","invented_source_CAA",
    "pretend_blind_raster","source_label_leaked","forged_predicted_partitions",
    "training_claim","broken_triangle_indices"
])
def test_forgeries_cannot_be_independent_artist_gold_benchmark(source,attack):
    m,h,_=pieces(source); p=predict(m)
    m=copy.deepcopy(m);h=copy.deepcopy(h);p=copy.deepcopy(p)
    if attack=="external_merged_sha":p["anonymous_merged_surface_sha256"]="x"*64
    elif attack=="forged_reference":h["anonymous_merged_surface_sha256"]="x"*64
    elif attack=="invented_source_CAA":h["source_appearance_rgba_alpha_parity_qualified"]=True
    elif attack=="pretend_blind_raster":m["source_composite_parity_qualified"]=True
    elif attack=="source_label_leaked":m["original_slot_attachment_bone_names_present"]=True
    elif attack=="forged_predicted_partitions":p["face_partition_id"][0]=999
    elif attack=="training_claim":p["training_authorized"]=True
    elif attack=="broken_triangle_indices":m["faces"][0][0]=999999
    with pytest.raises((ValueError,IndexError)):
        audit(m,p,h) if attack!="broken_triangle_indices" else predict(m)

def test_only_original_source_masked_parts_remain_unknown(source):
    d=copy.deepcopy(source)
    d["slots"][1]["bone"]="not_a_bone"
    # Invalid source is masked by source adapter, never positive ground truth.
    m,h,r=pieces(d)
    p=predict(m)
    outcome=audit(m,p,h)
    assert r["unsupported_original_attachment_count"]==1
    assert outcome["original_unsupported_source_attachment_count"]==1
    assert outcome["this_is_true_artist_drawable_recovery"] is False
