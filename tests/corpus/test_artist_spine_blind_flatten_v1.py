"""Blind native original Spine human-art source adapter V1 synthetic contracts.

Synthetic samples test file/runtime contracts only. They are NOT evidence of
real source reconstruction, original CAA parity or artist layer accuracy.
"""
from __future__ import annotations
import copy
import hashlib
import json
import numpy as np
import pytest
from tools.corpus.author_spine_blind_flatten_v1 import flatten

def raw(d):
    return json.dumps(d,sort_keys=True,separators=(",",":")).encode()

@pytest.fixture()
def authored():
    return {
       "skeleton":{"spine":"4.3.75-beta","images":"./images/"},
       "bones":[{"name":"original_human_artist_root"},
                {"name":"artist_rear_arm","parent":"original_human_artist_root",
                 "x":10., "y":3.,"rotation":0.}],
       "slots":[
         {"name":"original_artist_hand","bone":"artist_rear_arm","attachment":"original_glove"},
         {"name":"original_artist_torso","bone":"original_human_artist_root","attachment":"body"},
         {"name":"original_artist_eye","bone":"artist_rear_arm","attachment":"eye_smile"},
         {"name":"hidden_slot","bone":"original_human_artist_root"}],
       "skins":[{"name":"default","attachments":{
          "original_artist_hand":{"original_glove":{
              "type":"region","x":2.,"y":0.,"width":8.,"height":6.}},
          "original_artist_torso":{"body":{
              "type":"mesh","uvs":[0,0,1,0,0,1],
              "vertices":[0,0,10,0,0,10],
              "triangles":[0,1,2]}},
          "original_artist_eye":{"eye_smile":{
              "type":"mesh","uvs":[0,0,1,0,0,1],
              "vertices":[1,1,0.,0.,1.,
                          1,1,2.,0.,1.,
                          1,1,0.,2.,1.],
              "triangles":[0,1,2]}}
       }}],
       "animations":{"walk":{"bones":{}}}
    }

def test_merge_contains_no_human_layer_names_ids_or_counts(authored):
    input,hidden,r=flatten(authored,source_bytes=raw(authored))
    assert r["source_triangle_count"]==4
    assert r["source_vertex_count"]==10
    assert r["original_source_slot_count_teacher_only"]==4
    assert r["setup_active_attachment_count_teacher_only"]==3
    assert r["setup_supported_attachment_count_teacher_only"]==3
    assert not r["true_artist_layer_benchmark_PASS"]
    assert not r["native_composite_appearance_CAA_qualified"]
    assert len(hidden["artist_slot_id_teacher_only_per_face"])==4
    assert len(set(hidden["artist_slot_id_teacher_only_per_face"]))==3
    assert set(input["faces"][0]).issubset(range(10))
    encoded=json.dumps(input,sort_keys=True)
    for secret in ("original_artist","original_glove","artist_rear_arm",
                   "eye_smile","walk","hidden_slot","author_layer_count"):
        assert secret not in encoded
    assert input["source_composite_parity_qualified"] is False
    assert input["disconnected_topology_can_leak_original_author_partition"] is True
    assert input["training_authorized"] is False
    assert hidden["training_authorized"] is False

def test_source_local_mesh_unweighted_and_weighted_original_bone_geo(authored):
    merged,hidden,_=flatten(authored,source_bytes=raw(authored),seed=42)
    xyz=np.asarray(merged["vertices_setup_xy"])
    # Original artist rear arm moved +10/+3 from root in setup pose.
    assert np.any(np.all(np.isclose(xyz,[10,3]),axis=1))
    assert np.any(np.all(np.isclose(xyz,[12,3]),axis=1))
    assert np.any(np.all(np.isclose(xyz,[10,5]),axis=1))
    assert np.any(np.all(np.isclose(xyz,[0,0]),axis=1))
    assert sorted(x["method"] for x in hidden["artist_slot_draw_order_teacher_only"])==[
        "MESH_UNWEIGHTED","MESH_WEIGHTED","REGION_QUAD_NO_ATLAS_CAA"]

def test_deterministic_with_seed_without_author_order_label_leak(authored):
    a,h,rec=flatten(authored,source_bytes=raw(authored),seed=112)
    b,j,r=flatten(authored,source_bytes=raw(authored),seed=112)
    assert a==b and h==j and rec==r
    c,_,_=flatten(authored,source_bytes=raw(authored),seed=113)
    assert c["faces"]!=a["faces"]
    assert rec["anonymous_merged_surface_sha256"]!=flatten(authored,source_bytes=raw(authored),seed=113)[2]["anonymous_merged_surface_sha256"]

def test_extra_alternative_author_attachment_never_becomes_setup_drawable(authored):
    d=copy.deepcopy(authored)
    d["skins"][0]["attachments"]["original_artist_hand"]["unselected_artist_alternative"]={
        "type":"region","width":18,"height":12
    }
    merged,hidden,r=flatten(d,source_bytes=raw(d))
    assert r["setup_active_attachment_count_teacher_only"]==3
    assert r["source_triangle_count"]==4
    assert "unselected_artist_alternative" not in json.dumps(merged)
    assert not hidden["artist_same_drawable_reconstruction_qualified"]

def test_source_unknown_nonnormal_bone_marks_unsupported_instead_of_lying(authored):
    d=copy.deepcopy(authored)
    d["bones"][1]["inherit"]="noRotationOrReflection"
    merged,h,r=flatten(d,source_bytes=raw(d))
    assert r["unsupported_original_attachment_count"]==2
    assert r["setup_supported_attachment_count_teacher_only"]==1
    assert h["all_original_artist_active_attachments_supported"] is False
    assert not r["true_artist_layer_benchmark_PASS"]

def test_source_attachment_metadata_invalid_fails_closed(authored):
    d=copy.deepcopy(authored)
    d["slots"][1]["attachment"]="unknown_original"
    with pytest.raises(ValueError,match="SELECTED_ATTACHMENT_NOT_IN_DEFAULT_SKIN"):
        flatten(d,source_bytes=raw(d))

def test_source_missing_default_skin_fails_closed(authored):
    d=copy.deepcopy(authored);d["skins"][0]["name"]="not_default"
    with pytest.raises(ValueError,match="DEFAULT_SKIN_REQUIRED"):
        flatten(d,source_bytes=raw(d))

def test_source_self_parent_bone_cycle_fails_closed(authored):
    d=copy.deepcopy(authored);d["bones"][0]["parent"]="artist_rear_arm"
    with pytest.raises(ValueError,match="BONE_CYCLE"):
        flatten(d,source_bytes=raw(d))

def test_source_non_unit_skin_weights_masks_original_attachment(authored):
    d=copy.deepcopy(authored)
    d["skins"][0]["attachments"]["original_artist_eye"]["eye_smile"]["vertices"][4]=.3
    _,hidden,r=flatten(d,source_bytes=raw(d))
    assert r["setup_supported_attachment_count_teacher_only"]==2
    assert "WEIGHTED_INFLUENCE_NOT_NORMALIZED" in hidden["unsupported_original_attachment_cases_teacher_only"][0]["reason"]
    assert not r["true_artist_layer_benchmark_PASS"]

def test_source_fake_negative_triangle_address_masks_component(authored):
    d=copy.deepcopy(authored)
    d["skins"][0]["attachments"]["original_artist_torso"]["body"]["triangles"][0]=-7
    _,hidden,r=flatten(d,source_bytes=raw(d))
    assert r["setup_supported_attachment_count_teacher_only"]==2
    assert r["unsupported_original_attachment_count"]==1
    assert not hidden["all_original_artist_active_attachments_supported"]

def test_source_unknown_clip_no_misrepresented_as_visible_artist_drawable(authored):
    d=copy.deepcopy(authored)
    d["skins"][0]["attachments"]["original_artist_torso"]["body"]={
        "type":"clipping","vertices":[0,0,3,0,3,2],"end":"original_artist_eye"
    }
    _,h,r=flatten(d,source_bytes=raw(d))
    assert r["unsupported_original_attachment_count"]==1
    assert h["unsupported_original_attachment_cases_teacher_only"][0]["attachment_kind_teacher_only"]=="clipping"
    assert not r["native_composite_appearance_CAA_qualified"]

def test_source_byte_sha_is_original_not_derived_editor_label(authored):
    blob=raw(authored)
    merged,hidden,r=flatten(authored,source_bytes=blob)
    assert r["original_source_SHA256"]==hashlib.sha256(blob).hexdigest()
    assert hidden["human_authored_source_export_sha256"]==r["original_source_SHA256"]
    assert len(hidden["artist_slot_id_teacher_only_per_face"])==len(merged["faces"])
    assert hidden["anonymous_merged_surface_sha256"]==r["anonymous_merged_surface_sha256"]
