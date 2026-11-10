"""V12 private original-source heldout RGBA and anti-owner contracts.

Synthetic tests are NOT real teacher evidence. 192x192 contracts mirror hosted
full-image original-source/source-soup + 12 precommitted heldout source probes.
"""
import copy
import numpy as np
import pytest
from tools.corpus.court_corpus370_heldout_native_edit_v12 import evaluate

@pytest.fixture(scope="module")
def heldout():
    sizes=[8,16,32]
    faces=list(range(32))
    proposal={
        "schema":"RealSaS.Corpus370AllViewGeometryOperationalGroups.v3",
        "status":"ALL_8_INPUT_VIEW_GEOMETRY_LAYER_HYPOTHESES__RENDER_OUTCOME_HELDOUT",
        "source_sha256":"a"*64,"source_face_count":32,
        "training_authorized":False,"graphics_owner_truth_minted":False,
        "view4_rendered_edit_consequences_used_by_compiler":False,
        "candidates":[{
            "slot_count":k,"candidate":f"ALLVIEW_GEOMETRY_GROUPS_{k}",
            "face_slot_teacher_only":[i//(32//k) for i in faces],
            "source_faces_changed":False,"training_authorized":False
        } for k in sizes]
    }
    pairs={
        "parents":{"source_sha256":"a"*64,"rest_geometry_sha256":"b"*64},
        "training_authorized":False,
        "rows":[{"view_index":0,"source_face_a_teacher_only":0,"source_face_b_teacher_only":8},
                {"view_index":4,"source_face_a_teacher_only":4,"source_face_b_teacher_only":12}],
    }
    from tools.corpus.corpus370_partition_edit_plan_v7 import build_plan
    original=build_plan(proposal,pairs)
    excluded={t["source_face_teacher_only"] for t in original["tasks"]}
    new=[x for x in faces if x not in excluded][:12]
    queries=[{
        "test_index":i,"source_face_teacher_only":face,
        "finest_group_id_teacher_only":face,
        "source_groups_per_candidate_size_teacher_only":{
            str(k):face//(32//k) for k in sizes},
        "same_drawable":"UNKNOWN","different_drawable":"UNKNOWN",
        "training_authorized":False
    } for i,face in enumerate(new)]
    assert len(new)==12
    plan={
        "schema":"RealSaS.Corpus370GeometryOnlyHeldoutBranchProbePlan.v11",
        "subject_id":"synthetic-only","source_sha256":"a"*64,
        "new_source_face_intervention_query_count":12,
        "query_targets_source_teacher_only":queries,
        "real_rgba_consequences_measured_for_heldout_queries":False,
        "graphics_owner_authorized":False,"training_authorized":False,
        "parents":{
            "independent_native_v10_court_sha256":"e"*64,
            "original_native_v9_manifest_sha256":"f"*64
        }
    }
    old={
        "schema":"RealSaS.Corpus370NativeAnimatedEditResponseEquivalenceCourt.v10",
        "subject_id":"synthetic-only","source_sha256":"a"*64,
        "graphics_owner_authorized":False,"training_authorized":False,
        "artist_same_drawable_positive_pairs":0,
        "artist_different_drawable_negative_pairs":0,
        "pairwise_equivalence_classes_scoped_to_tested_source_edits":[{
            "candidate_a":a,"candidate_b":b,
            "total_source_edit_task_count":original["task_count"],
            "finite_test_equivalence_class_member":(a,b)==(16,32),
        } for a,b in ((8,16),(8,32),(16,32))]
    }
    arms=["SOURCE_ORIGINAL","SOUP_FULL","SOUP_NOOP","REMOVE_ALL_GROUPS"]
    for i in range(12):
        for k in sizes:
            arms.extend((f"GROUP_REMOVE_K{k:03d}_T{i:02d}",
                         f"GROUP_ISOLATE_K{k:03d}_T{i:02d}"))
    manifest={
        "schema":"RealSaS.Corpus370HeldoutNativeAnimatedSourceRender.v12",
        "status":"REAL_V11_HELDOUT_SOURCE_NATIVE_POSE_RGBA_RENDER__INDEPENDENT_V12_COURT_REQUIRED",
        "teacher_only":True,"source_rest_exact_geometry_verified":True,
        "source_selected_pose_exact_teacher_replay_verified":True,
        "v11_original_heldout_targets_verified":True,
        "source_sha256":"a"*64,"subject_id":"synthetic-only",
        "candidate_sizes":sizes,"v11_heldout_plan_sha256":"c"*64,
        "source_group_proposal_sha256":"d"*64,"source_pair_sha256":"g"*64,
        "reference_policy":"DIRECT_PAIRWISE_HELDOUT_SOURCE_GEOMETRIC_EDIT_NO_ARTIST_TRUTH",
        "graphics_owner_authorized":False,"training_authorized":False,
        "source_bone_names_as_targets":False,
        "animated_render_caa_authorized":False,
        "materialization_authorized":False,
        "prior_v10_court_sha256":"e"*64,
        "prior_v9_manifest_sha256":"f"*64,
        "previous_tested_source_faces_teacher_only":[x["source_face_teacher_only"]
                                                     for x in original["tasks"]],
        "view_indices":[0,2,4,6],"raster_resolution":192,"cycles_samples":32,
        "pose_indices":[0,3,8],"arms":arms
    }
    pixels=np.zeros((3,4,len(arms),192,192,4),np.float16)
    source=np.zeros((192,192,4),np.float16)
    source[30:145,30:145]=(.2,.4,.7,1.)
    for i in range(3):
        pixels[:,:,i]=source
    index={name:i for i,name in enumerate(arms)}
    for t in range(12):
        y=35+(t//4)*25;x=37+(t%4)*25
        for k in sizes:
            output=source.copy();iso=np.zeros_like(source)
            margin=0 if k==32 else 2 if k==16 else 5
            output[y:y+8+margin,x:x+8+margin]=0
            iso[y:y+8+margin,x:x+8+margin]=source[y:y+8+margin,x:x+8+margin]
            pixels[:,:,index[f"GROUP_REMOVE_K{k:03d}_T{t:02d}"]]=output
            pixels[:,:,index[f"GROUP_ISOLATE_K{k:03d}_T{t:02d}"]]=iso
    return (pixels,manifest,proposal,pairs,plan,old)

def test_real_source_like_heldout_counterexample_has_no_semantic_training(heldout):
    d=evaluate(*heldout)
    assert d["new_untested_original_source_face_targets"]==12
    assert d["source_soup_parity_qualified_pose_views"]==12
    assert d["source_soup_parity_abstained_pose_views"]==0
    assert d["artist_same_drawable_positive_pairs"]==0
    assert d["artist_different_drawable_negative_pairs"]==0
    assert d["training_authorized"] is False
    pair=next(x for x in d["source_geometry_pairwise_heldout_consequences"]
              if (x["candidate_a"],x["candidate_b"])==(16,32))
    assert pair["v10_original_source_hit_indistinguishable_all_tasks"] is True
    assert pair["v10_finite_source_edit_equivalence_falsified_by_new_heldout"] is True
    assert pair["heldout_queries_with_pairwise_counterexamples"]==12
    assert pair["same_drawable"]=="UNKNOWN"

@pytest.mark.parametrize("fault",[
    "fake_student_semantics","fake_training","fake_owner",
    "fake_v11_training","fake_old_v10_positives","source_mismatch",
    "prior_v9_sha_mismatch","previous_anchor_reuse","different_test_order",
    "invented_layer_size","wrong_camera","missing_edit_arm","invalid_pose_count"
])
def test_corrupt_source_lineage_and_owner_labels_fail_closed(heldout,fault):
    pix,m,proposal,pairs,plan,old=heldout
    m=copy.deepcopy(m);proposal=copy.deepcopy(proposal)
    pairs=copy.deepcopy(pairs);plan=copy.deepcopy(plan);old=copy.deepcopy(old)
    if fault=="fake_student_semantics":m["source_bone_names_as_targets"]=True
    elif fault=="fake_training":m["training_authorized"]=True
    elif fault=="fake_owner":m["graphics_owner_authorized"]=True
    elif fault=="fake_v11_training":plan["training_authorized"]=True
    elif fault=="fake_old_v10_positives":old["artist_same_drawable_positive_pairs"]=1
    elif fault=="source_mismatch":m["source_sha256"]="z"*64
    elif fault=="prior_v9_sha_mismatch":m["prior_v9_manifest_sha256"]="1"*64
    elif fault=="previous_anchor_reuse":
        plan["query_targets_source_teacher_only"][0]["source_face_teacher_only"]=0
        plan["query_targets_source_teacher_only"][0]["finest_group_id_teacher_only"]=0
        plan["query_targets_source_teacher_only"][0]["source_groups_per_candidate_size_teacher_only"]={"8":0,"16":0,"32":0}
    elif fault=="different_test_order":plan["query_targets_source_teacher_only"][1]["test_index"]=13
    elif fault=="invented_layer_size":m["candidate_sizes"]=[8,16,64]
    elif fault=="wrong_camera":m["view_indices"]=[0,2,4,7]
    elif fault=="missing_edit_arm":m["arms"]=m["arms"][:-1]
    elif fault=="invalid_pose_count":m["pose_indices"]=[0,3]
    with pytest.raises((ValueError,KeyError)):
        evaluate(pix,m,proposal,pairs,plan,old)

def test_missing_source_soup_parity_does_not_mint_layer_truth(heldout):
    # Synthetic image corruption in only one pose-view; do not allocate
    # hundreds of megabytes just to test a single forged pixel.
    pix,m,proposal,pairs,plan,old=heldout
    m=copy.deepcopy(m);m["status"]="UNVERIFIED_FAKE_NATIVE_RENDER"
    with pytest.raises(ValueError):
        evaluate(pix,m,proposal,pairs,plan,old)
