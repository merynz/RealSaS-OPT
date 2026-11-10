"""Adversarial morphology-neutral group compression; synthetic contract only."""
import copy

import numpy as np
import pytest

from tools.corpus.compile_corpus370_hierarchical_layer_groups_v1 import (
    coalesce,build,
)
from tools.corpus.court_corpus370_hierarchical_layer_groups_v1 import verify

S="a"*64


def fixtures(n=80):
    # Eighty disconnected source triangle islands, intentionally arbitrary.
    # This mimics the worst one-surface-per-component editor output.
    v=np.zeros((3*n,3),np.float64)
    for i in range(n):
        x=float(i%10)*2.
        y=float(i//10)*2.
        v[3*i:3*i+3]=[[x,y,0],[x+0.3,y,0],[x,y+0.3,0]]
    f=np.arange(3*n,dtype=np.int32).reshape(n,3)
    p=np.stack([v,v.copy(),v.copy()])
    p[1,::3,2]+=0.05
    p[2,1::3,2]-=0.03
    def row(view,a,b,px):
        return {
            "schema":"RealSaS.Corpus370GenericPairKinematicDiagnostic.v1",
            "view_index":view,"source_face_a_teacher_only":a,
            "source_face_b_teacher_only":b,
            "pixel_x":px,"pixel_y":7,
            "source_hit_visibility_owner":"SOURCE_LOCUS_ONLY",
            "same_drawable":"UNKNOWN","different_drawable":"UNKNOWN",
            "training_authorized":False,
        }
    pairs={
        "schema":"RealSaS.Corpus370GenericPairKinematicDiagnosticBatch.v1",
        "parents":{"source_sha256":S},
        "rows":[row(0,0,79,2),row(4,10,70,4)],
        "pair_count":2,
        "qualified_same_drawable_positives":0,
        "qualified_different_drawable_negatives":0,
        "training_authorized":False,
    }
    proposal={
        "schema":"RealSaS.Corpus370AnonymousLayerProposal.v1",
        "training_authorized":False,"graphics_owner_minted":False,
        "source_semantic_or_material_names_used":False,
        "parents":{"source_sha256":S,
                   "canonical_arrays_sha256":"b"*64,
                   "source_pair_receipt_sha256":"c"*64},
        "face_count":n,"authored_pose_count":3,
        "candidates":[{
            "candidate":"MANIFOLD_COMPONENTS","slot_count":n,
            "face_slot_teacher_only":list(range(n)),
        }],
    }
    base={
        "schema":"RealSaS.Corpus370AnonymousLayerEditProxyCourt.v1",
        "selected_candidate_train_only":"MANIFOLD_COMPONENTS",
        "selected_slot_count":n,
        "train_view":0,"heldout_view":4,
        "training_authorized":False,
        "graphics_family_authorized":False,
        "qualified_same_drawable_positives":0,
        "qualified_different_drawable_negatives":0,
    }
    return v,f,p,pairs,proposal,base


def test_hierarchy_significant_compression_preserves_atomic_source_addresses():
    v,f,p,pairs,proposal,base=fixtures()
    h=build(v,f,p,proposal,base,[pairs["rows"][0]])
    assert h["base_atomic_patch_count"]==80
    assert {x["slot_count"] for x in h["candidates"]}=={8,16,32,64}
    assert len(h["measured_merges"])>=72
    assert h["heldout_pairs_used_for_grouping"] is False
    x=verify(h,pairs,proposal,base)
    assert x["selected_slot_count"]==8
    assert x["group_compression_ratio"]==10
    assert x["train_ray_count"]==1
    assert x["training_authorized"] is False
    assert x["true_author_drawable_partition_qualified"] is False


def test_no_train_locus_merge_even_at_two_groups():
    v,f,p,pairs,prop,base=fixtures(12)
    report=coalesce(v,f,p,list(range(12)),[(0,11)],
                    target_counts=(2,4,8))
    assert 2 in report["snapshots"]
    labels=report["snapshots"][2]
    assert labels[0]!=labels[11]


def test_no_heldout_data_leak_into_proposals_or_winner():
    v,f,p,pairs,prop,base=fixtures()
    a=build(v,f,p,prop,base,[pairs["rows"][0]])
    result_a=verify(a,pairs,prop,base)
    mutated=copy.deepcopy(pairs)
    mutated["rows"][1]["source_face_a_teacher_only"]=0
    mutated["rows"][1]["source_face_b_teacher_only"]=79
    b=build(v,f,p,prop,base,[mutated["rows"][0]])
    result_b=verify(b,mutated,prop,base)
    assert a["candidates"]==b["candidates"]
    assert result_a["selected_candidate_train_only"]==result_b["selected_candidate_train_only"]


def test_training_violation_fails_even_if_other_edits_work():
    v,f,p,pairs,prop,base=fixtures()
    h=build(v,f,p,prop,base,[pairs["rows"][0]])
    h["candidates"][0]["face_slot_teacher_only"]=[0]*len(f)
    h["candidates"][0]["slot_count"]=1
    with pytest.raises(ValueError,match="COARSEN_ONLY_PARTITION|GROUP_COUNT_NAME"):
        verify(h,pairs,prop,base)


def test_parent_already_merging_required_edit_must_be_rejected():
    v,f,p,pairs,prop,base=fixtures()
    prop["candidates"][0]["face_slot_teacher_only"][79]=0
    with pytest.raises(ValueError,match="PARENT_CONTRADICTS_TRAIN"):
        build(v,f,p,prop,base,[pairs["rows"][0]])


def test_no_training_truth_and_no_semantic_names():
    v,f,p,pairs,prop,base=fixtures()
    prop["source_semantic_or_material_names_used"]=True
    with pytest.raises(ValueError,match="PARENT_PROPOSAL_SCOPE"):
        build(v,f,p,prop,base,[pairs["rows"][0]])
    prop["source_semantic_or_material_names_used"]=False
    h=build(v,f,p,prop,base,[pairs["rows"][0]])
    h["training_authorized"]=True
    with pytest.raises(ValueError,match="PARENT_SCOPE"):
        verify(h,pairs,prop,base)


def test_bad_trajectories_and_nontriangulated_source_rejected():
    v,f,p,pairs,prop,base=fixtures()
    p[0,0,0]=float("nan")
    with pytest.raises(ValueError,match="AUTHORED_POSES_INVALID"):
        build(v,f,p,prop,base,[pairs["rows"][0]])
    p[0,0,0]=v[0,0]
    f=f[:,:2]
    with pytest.raises(ValueError,match="FACES_INVALID"):
        build(v,f,p,prop,base,[pairs["rows"][0]])


def test_coincident_geometry_not_reinterpreted_as_same_drawable_truth():
    v,f,p,pairs,prop,base=fixtures()
    # Co-motion or spatial proximity cannot mint owner relationship.
    p=np.stack([v,v,v])
    h=build(v,f,p,prop,base,[pairs["rows"][0]])
    c=verify(h,pairs,prop,base)
    assert c["same_drawable_positive_pairs"]==0
    assert c["different_drawable_negative_pairs"]==0
    assert c["graphics_family_authorized"] is False
    assert c["full_image_alpha_depth_replay_executed"] is False


def test_source_face_coverage_fail_closed():
    v,f,p,pairs,prop,base=fixtures()
    h=build(v,f,p,prop,base,[pairs["rows"][0]])
    h["candidates"][0]["face_slot_teacher_only"]=[0,1]
    with pytest.raises(ValueError,match="COARSEN_ONLY_PARTITION"):
        verify(h,pairs,prop,base)
