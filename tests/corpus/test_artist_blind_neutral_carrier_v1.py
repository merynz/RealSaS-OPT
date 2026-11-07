"""Neutral topology control: source art attachments cannot survive grid merging."""
import copy
import json
import pytest
from tools.corpus.author_spine_blind_flatten_v1 import flatten
from tools.corpus.artist_blind_neutral_carrier_v1 import build,audit_connected

@pytest.fixture()
def source():
    d={
        "bones":[{"name":"source_named_root"}],
        "slots":[{"name":"separate_artist_head","bone":"source_named_root","attachment":"head_piece"},
                 {"name":"separate_artist_torso","bone":"source_named_root","attachment":"torso_piece"}],
        "skins":[{"name":"default","attachments":{
            "separate_artist_head":{"head_piece":{"type":"region","width":20,"height":14,"x":0,"y":12}},
            "separate_artist_torso":{"torso_piece":{"type":"region","width":24,"height":18,"x":0,"y":-3}}
        }}]
    }
    merged,hidden,_=flatten(d,source_bytes=json.dumps(d).encode())
    return merged,hidden

def test_merged_source_original_islands_replaced_by_one_neutral_grid(source):
    merged,_=source
    from tools.corpus.artist_blind_topology_leakage_audit_v1 import predict
    assert predict(merged)["predicted_topological_component_count"]==2
    carrier=build(merged,cells=8)
    receipt=audit_connected(carrier)
    assert receipt["one_connected_component"]
    assert receipt["artist_drawable_recovery_accuracy"] is None
    assert receipt["generic_geometry_only_ownership_proven"] is False
    assert receipt["source_artist_face_labels_comparable_to_grid_faces"] is False
    assert len(carrier["faces"])==8*8*2
    assert len(carrier["vertices_setup_xy"])==9*9
    assert carrier["original_atlas_or_raster_appearance_available"] is False
    assert carrier["training_authorized"] is False

def test_source_layer_count_and_names_not_in_carrier(source):
    carrier=build(source[0])
    raw=json.dumps(carrier,sort_keys=True)
    for forbidden in ("separate_artist","head_piece","torso_piece",
                      "source_named_root","source_original_layer_count"):
        assert forbidden not in raw
    assert carrier["original_author_part_count_visible"] is False

@pytest.mark.parametrize("fault",[
    "source_admits_training","source_author_names_present",
    "source_faked_original_CAA","fake_original_face_correspondence",
    "fake_gold_masks_projected","fake_art_source_triangles",
    "empty_source","bad_cells"
])
def test_gold_ground_truth_cannot_be_forged_from_blank_neutral_carrier(source,fault):
    merged=copy.deepcopy(source[0])
    if fault=="source_admits_training":merged["training_authorized"]=True
    elif fault=="source_author_names_present":merged["original_slot_attachment_bone_names_present"]=True
    elif fault=="source_faked_original_CAA":merged["source_composite_parity_qualified"]=True
    elif fault=="empty_source":merged["vertices_setup_xy"]=[]
    elif fault=="bad_cells":
        with pytest.raises(ValueError):
            build(merged,cells=1)
        return
    if fault in ("fake_original_face_correspondence","fake_gold_masks_projected","fake_art_source_triangles"):
        carrier=build(merged)
        if fault=="fake_original_face_correspondence":carrier["original_source_face_to_grid_face_correspondence"]=True
        elif fault=="fake_gold_masks_projected":carrier["artist_2d_masks_projected_onto_grid"]=True
        else:carrier["original_author_face_incidence_preserved"]=True
        with pytest.raises(ValueError):
            audit_connected(carrier)
    else:
        with pytest.raises(ValueError):
            build(merged)

def test_carrier_deterministic_for_fixed_bbox_not_artist_face_order(source):
    first=build(source[0],cells=10)
    second=copy.deepcopy(source[0])
    second["faces"].reverse()
    a=build(second,cells=10)
    assert first["faces"]==a["faces"]
    assert first["vertices_setup_xy"]==a["vertices_setup_xy"]
    assert first["source_anonymous_merged_surface_SHA256"]!=a["source_anonymous_merged_surface_SHA256"]
