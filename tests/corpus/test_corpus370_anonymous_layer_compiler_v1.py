"""Source-only anonymous operational layers, synthetic contract regressions.

Neither proof of true author drawable identity nor a full rendering test.
"""
from __future__ import annotations

import copy
import numpy as np
import pytest

from tools.corpus.compile_corpus370_anonymous_layer_proposals_v1 import compile_candidates
from tools.corpus.court_corpus370_anonymous_layer_edit_proxy_v1 import court

PARENTS={
    "source_sha256":"a"*64,
    "canonical_arrays_sha256":"b"*64,
    "source_pair_receipt_sha256":"c"*64,
}


def fixtures():
    vertices=np.array([
        [0.,0.,0.],[1.,0.,0.],[0.,1.,0.],
        [1.,1.,0.],[2.,0.,0.],[2.,1.,0.],
    ],dtype=np.float64)
    faces=np.array([
        [0,1,2],[2,1,3],[1,4,3],[3,4,5],
    ],dtype=np.int32)
    posed=vertices.copy()
    posed[5,2]=0.5
    poses=np.stack([vertices,posed],axis=0)
    def row(a,b,view,pixel):
        return {
            "schema":"RealSaS.Corpus370GenericPairKinematicDiagnostic.v1",
            "source_face_a_teacher_only":a,
            "source_face_b_teacher_only":b,
            "view_index":view,
            "pixel_x":pixel,"pixel_y":7,
            "source_hit_visibility_owner":"SOURCE_LOCUS_ONLY",
            "same_drawable":"UNKNOWN","different_drawable":"UNKNOWN",
            "training_authorized":False,
        }
    rows=[row(0,3,0,2),row(1,2,4,4)]
    pairs={
        "schema":"RealSaS.Corpus370GenericPairKinematicDiagnosticBatch.v1",
        "parents":{"source_sha256":"a"*64},
        "pair_count":len(rows),"rows":rows,
        "training_authorized":False,
        "qualified_same_drawable_positives":0,
        "qualified_different_drawable_negatives":0,
    }
    return vertices,faces,poses,pairs


def proposal():
    v,f,posed,pairs=fixtures()
    return compile_candidates(v,f,posed,pairs["rows"],parents=PARENTS)


def test_seven_anonymous_baselines_derived_without_semantics():
    result=proposal()
    assert result["candidate_count"]==7
    assert result["face_count"]==4
    assert result["source_semantic_or_material_names_used"] is False
    assert result["training_authorized"] is False
    cs={c["candidate"]:c for c in result["candidates"]}
    assert cs["SINGLE_LAYER"]["slot_count"]==1
    assert cs["ATOMIC_PER_FACE"]["slot_count"]==4
    assert cs["MANIFOLD_COMPONENTS"]["slot_count"]==1
    assert cs["TRAIN_VIEW_RAY_SEEDED"]["slot_count"]==2
    assert all(len(c["face_slot_teacher_only"])==4 for c in cs.values())


def test_consumer_uses_train_only_winner_and_opposing_view_diagnostics():
    v,f,poses,pairs=fixtures()
    p=compile_candidates(v,f,poses,pairs["rows"],parents=PARENTS)
    result=court(p,pairs)
    assert result["selected_candidate_train_only"] in {
        "TRAIN_VIEW_RAY_SEEDED","GEOMETRY_SEAMS",
        "MOTION_SEAMS","GEOMETRY_OR_MOTION_SEAMS"}
    atomic=next(c for c in result["candidate_scores"]
                if c["candidate"]=="ATOMIC_PER_FACE")
    single=next(c for c in result["candidate_scores"]
                if c["candidate"]=="SINGLE_LAYER")
    assert atomic["train_ray_atomic_layer_reveal_hits"]==1
    assert atomic["heldout_ray_atomic_layer_reveal_hits"]==1
    assert single["train_ray_atomic_layer_reveal_hits"]==0
    assert single["heldout_ray_atomic_layer_reveal_hits"]==0
    assert result["qualified_same_drawable_positives"]==0
    assert result["qualified_different_drawable_negatives"]==0
    assert result["graphics_family_authorized"] is False


def test_heldout_source_pairs_never_affect_candidate_generation_or_selection():
    v,f,poses,pairs=fixtures()
    p0=compile_candidates(v,f,poses,pairs["rows"],parents=PARENTS)
    x=court(p0,pairs)
    tampered=copy.deepcopy(pairs)
    tampered["rows"][1]["source_face_a_teacher_only"]=0
    tampered["rows"][1]["source_face_b_teacher_only"]=2
    p1=compile_candidates(v,f,poses,tampered["rows"],parents=PARENTS)
    y=court(p1,tampered)
    assert p1["candidates"]==p0["candidates"]
    assert x["selected_candidate_train_only"]==y["selected_candidate_train_only"]


def test_common_root_motion_not_promoted_as_drawable_identity():
    v,f,poses,pairs=fixtures()
    common=np.stack([v,v+np.array([4.,2.,-1.])])
    p=compile_candidates(v,f,common,pairs["rows"],parents=PARENTS)
    assert p["training_authorized"] is False
    assert court(p,pairs)["author_drawable_partition_identified"] is False


def test_same_source_face_invalid_as_two_editable_loci():
    v,f,poses,pairs=fixtures()
    pairs["rows"][0]["source_face_b_teacher_only"]=0
    with pytest.raises(ValueError,match="FACE_OR_VIEW"):
        compile_candidates(v,f,poses,pairs["rows"],parents=PARENTS)


def test_one_view_not_enough_for_heldout():
    v,f,poses,pairs=fixtures()
    pairs["rows"]=[pairs["rows"][0],copy.deepcopy(pairs["rows"][0])]
    pairs["rows"][1]["pixel_x"]=9
    with pytest.raises(ValueError,match="0_TRAIN_4_HELDOUT"):
        compile_candidates(v,f,poses,pairs["rows"],parents=PARENTS)


def test_privileged_semantic_or_training_forgery_fails_closed():
    v,f,poses,pairs=fixtures()
    p=compile_candidates(v,f,poses,pairs["rows"],parents=PARENTS)
    p["source_semantic_or_material_names_used"]=True
    with pytest.raises(ValueError,match="PROPOSAL_SCOPE"):
        court(p,pairs)
    p=proposal()
    p["training_authorized"]=True
    with pytest.raises(ValueError,match="PROPOSAL_SCOPE"):
        court(p,pairs)


def test_partition_relabel_allowed_but_noncanonical_report_rejected():
    v,f,poses,pairs=fixtures()
    p=proposal()
    c=next(a for a in p["candidates"]
           if a["candidate"]=="TRAIN_VIEW_RAY_SEEDED")
    c["face_slot_teacher_only"]=[7 if x==0 else 9
                                  for x in c["face_slot_teacher_only"]]
    with pytest.raises(ValueError,match="PARTITION_INTEGRITY"):
        court(p,pairs)


def test_bad_face_topology_and_nan_pose_fail():
    v,f,poses,pairs=fixtures()
    bad=f.copy();bad[0]=[0,0,2]
    with pytest.raises(ValueError,match="ANON_LAYER_FACES"):
        compile_candidates(v,bad,poses,pairs["rows"],parents=PARENTS)
    bad_pose=poses.copy();bad_pose[0,0,0]=np.nan
    with pytest.raises(ValueError,match="NATIVE_POSES"):
        compile_candidates(v,f,bad_pose,pairs["rows"],parents=PARENTS)


def test_nonmanifold_edge_does_not_create_manifold_connection():
    v=np.array([[0,0,0],[1,0,0],[0,1,0],
                [0,-1,0],[0,0,1]],np.float64)
    f=np.array([[0,1,2],[1,0,3],[0,1,4]],np.int32)
    pair=fixtures()[3]
    pair["rows"][0]["source_face_a_teacher_only"]=0
    pair["rows"][0]["source_face_b_teacher_only"]=1
    pair["rows"][1]["source_face_a_teacher_only"]=1
    pair["rows"][1]["source_face_b_teacher_only"]=2
    pp=compile_candidates(v,f,np.stack([v,v]),pair["rows"],parents=PARENTS)
    cs={c["candidate"]:c for c in pp["candidates"]}
    assert cs["MANIFOLD_COMPONENTS"]["slot_count"]==3
    assert court(pp,pair)["training_authorized"] is False


def test_visibility_negative_is_not_different_drawable_negative():
    v,f,poses,pairs=fixtures()
    result=court(proposal(),pairs)
    assert result["qualified_different_drawable_negatives"]==0
    assert result["operational_layer_student_targets_minted"] is False
