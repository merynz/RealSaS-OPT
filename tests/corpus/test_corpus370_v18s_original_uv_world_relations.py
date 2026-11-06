"""Synthetic V18s original typed relation evidence, no source artist labels."""
import copy
import numpy as np
import pytest

from tools.corpus.compile_corpus370_v18s_original_uv_world_relations import (
    measure_relations,
)


def source_fixture():
    # Face0 + face1 share original indexed edge (0,1);
    # Face2 is disconnected but its vertex4 overlays vertex0.
    faces = np.asarray([[0,1,2],[1,0,3],[4,5,6]], np.int64)
    uv = np.asarray([
        [[0,0],[1,0],[0,1]],
        [[1,0],[0,0],[1,1]],
        [[0,0],[0.8,0.6],[0.9,0.9]],
    ],np.float64)
    pos = np.asarray([(0,0,0),(1,0,0),(0,1,0),(1,1,0),
                      (0,0,0),(0,1,1),(1,0,1)],np.float64)
    poses = np.stack([pos,pos+np.array([0,0,0.1]),pos+np.array([0,0,0.5])])
    local = [{
        "face_a_source_teacher_only": 0,
        "face_b_source_teacher_only": 1,
        "source_mechanical_any_axis_divergent": True,
        "source_mechanical_pooled_divergent": False,
        "fit_consistent_geometric_normal_crease_30deg": False,
    }]
    lifted = [{
        "source_island_a_teacher_only":0,
        "source_island_b_teacher_only":1,
        "source_fit_both_pose_min_vertex_gap_le_0p005":True,
    }]
    return {
        "faces":faces,"face_uv":uv,"baseline_vertices":poses,
        "island_for_face":[0,0,1],"local_rows":local,
        "lifted_rows":lifted,
        "original_source_hash":"1"*64,
    }


def test_same_uv_address_not_same_mechanical_response_or_drawable_owner():
    x=source_fixture()
    r=measure_relations(**x)
    assert r["source_local_relation_census"]["uv_continuous"]==1
    assert r["source_local_relation_census"]["axis_divergent_but_uv_continuous"]==1
    assert r["source_lifted_relation_census"]["world_fit_near_with_coincident_uv"]==1
    assert r["source_lifted_relation_census"]["same_vertex_pair_world_near_both_fit"]==1
    assert r["original_author_drawable_ownership"]=="UNKNOWN"
    assert not r["iris_training_authorized"]
    assert not r["graphics_owner_authorized"]
    assert r["same_drawable_training_pairs_qualified"]==0
    assert r["different_drawable_training_pairs_qualified"]==0
    assert r["source_material_datablock_identification"]=="UNKNOWN_PER_MESH_SLOT_ZERO_NOT_IDENTITY"


def test_source_uv_discontinuity_does_not_create_artist_cannot_link():
    x=source_fixture()
    x["face_uv"][1,1,0]=0.35
    r=measure_relations(**x)
    assert r["source_local_relation_census"].get("uv_endpoint_discontinuous")==1
    assert r["source_local_face_edge_evidence_private"][0]["different_drawable"]=="UNKNOWN"


def test_nonlocal_near_source_can_have_uv_endpoint_discontinuity():
    x=source_fixture()
    x["face_uv"][2,0]=[0.3,0.3]
    r=measure_relations(**x)
    assert r["source_lifted_relation_census"]["world_fit_near_without_coincident_uv"]==1
    assert r["source_lifted_fit_world_uv_evidence_private"][0]["source_artistic_owner_merge"]=="UNKNOWN"


def test_separate_fit_pose_can_reject_a_source_contact_even_with_same_uv():
    x=source_fixture()
    x["baseline_vertices"][1,4]=[1,1,1]
    x["lifted_rows"][0]["source_fit_both_pose_min_vertex_gap_le_0p005"]=False
    r=measure_relations(**x)
    assert r["source_lifted_relation_census"].get("same_vertex_pair_world_near_both_fit",0)==0


def test_absent_original_shared_edge_fails_closed():
    x=source_fixture()
    x["local_rows"][0]["face_b_source_teacher_only"]=2
    with pytest.raises(ValueError,match="SOURCE_LOCAL_EDGE_NOT_SHARED"):
        measure_relations(**x)


def test_original_source_face_uv_address_cannot_be_nan():
    x=source_fixture()
    x["face_uv"][0,0,0]=float("nan")
    with pytest.raises(ValueError,match="UV_FACE_CORNER_DOMAIN"):
        measure_relations(**x)


def test_nonlocal_candidate_not_automatically_added():
    x=source_fixture()
    x["lifted_rows"]=[]
    r=measure_relations(**x)
    assert r["source_lifted_relation_census"]=={}
    assert not r["claimed_IRIS_training_target"]
