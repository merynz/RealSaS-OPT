"""Synthetic proxy tests; source / drawable truth is established by private real run."""
import copy
import numpy as np
import pytest

from tools.corpus.corpus370_partition_edit_plan_v7 import build_plan,arm_names
from tools.corpus.court_corpus370_multi_partition_edit_v7 import evaluate


def parents(sizes=(8,16,32),face_count=32):
    labels={
        8:[i//4 for i in range(face_count)],
        16:[i//2 for i in range(face_count)],
        32:list(range(face_count)),
    }
    proposal={
        "schema":"RealSaS.Corpus370AllViewGeometryOperationalGroups.v3",
        "status":"ALL_8_INPUT_VIEW_GEOMETRY_LAYER_HYPOTHESES__RENDER_OUTCOME_HELDOUT",
        "source_sha256":"a"*64,
        "source_face_count":face_count,
        "training_authorized":False,
        "graphics_owner_truth_minted":False,
        "view4_rendered_edit_consequences_used_by_compiler":False,
        "candidates":[{
            "slot_count":size,
            "candidate":f"ALLVIEW_GEOMETRY_GROUPS_{size}",
            "source_faces_changed":False,
            "training_authorized":False,
            "face_slot_teacher_only":labels[size],
        } for size in sizes],
    }
    pairs={
        "parents":{"source_sha256":"a"*64},
        "training_authorized":False,
        "rows":[
            {"view_index":0,"source_face_a_teacher_only":0,
             "source_face_b_teacher_only":8},
            {"view_index":4,"source_face_a_teacher_only":4,
             "source_face_b_teacher_only":12},
        ],
    }
    return proposal,pairs


def fixture(sizes=(8,16,32)):
    proposal,pairs=parents(sizes)
    plan=build_plan(proposal,pairs)
    arms=arm_names(plan)
    pixels=np.zeros((8,len(arms),128,128,4),dtype=np.float32)
    source=np.zeros((128,128,4),dtype=np.float32)
    source[20:100,20:100,:]=(.5,.4,.3,1.)
    for i in range(3):
        pixels[:,i]=source
    by_arm={a:i for i,a in enumerate(arms)}
    for task in plan["tasks"]:
        idx=task["task_index"]
        y=22+idx*18
        x=26+idx*18
        for size in plan["source_geometry_candidate_sizes"]:
            removed=source.copy()
            # More coarse: extra original pixels lost outside the fine target.
            pad=plan["finest_available_reference_size"]//size
            removed[y-pad:y+8+pad,x-pad:x+8+pad]=0.
            pixels[:,by_arm[f"GROUP_REMOVE_K{size:03d}_T{idx:02d}"]]=removed
            isolated=np.zeros_like(source)
            isolated[y-pad:y+8+pad,x-pad:x+8+pad]=source[
                y-pad:y+8+pad,x-pad:x+8+pad]
            pixels[:,by_arm[f"GROUP_ISOLATE_K{size:03d}_T{idx:02d}"]]=isolated
        finest=plan["finest_available_reference_size"]
        pixels[:,by_arm[f"FINE_REMOVE_T{idx:02d}"]]=pixels[
            :,by_arm[f"GROUP_REMOVE_K{finest:03d}_T{idx:02d}"]]
        pixels[:,by_arm[f"FINE_ISOLATE_T{idx:02d}"]]=pixels[
            :,by_arm[f"GROUP_ISOLATE_K{finest:03d}_T{idx:02d}"]]
    manifest={
        "schema":"RealSaS.Corpus370MultiPartitionSourceEditRender.v7",
        "status":"SOURCE_MULTIPARTITION_REST_EDIT_PROXY__INDEPENDENT_COURT_REQUIRED",
        "teacher_only":True,"training_authorized":False,
        "graphics_owner_authorized":False,"model_training_authorized":False,
        "original_geometry_reimport_parity_verified":True,
        "reference_policy":"FINEST_AVAILABLE_GEOMETRY_HYPOTHESIS",
        "source_sha256":"a"*64,
        "original_source_partition_edit_plan_teacher_only":plan,
        "source_group_partition_candidates":plan["source_geometry_candidate_sizes"],
        "arms":arms,"views":list(range(8)),
        "full_frame_resolution":128,
        "private_view_details":[{
            "view":v, "source_groups_same_addresses_across_cameras":True}
            for v in range(8)],
    }
    return pixels,manifest,proposal,pairs


def test_realistic_partition_edit_proxy_compares_all_candidates_and_no_labels():
    p,m,proposal,pairs=fixture()
    r=evaluate(p,m,proposal,pairs)
    assert r["source_original_vs_soup_noop_parity_views"]==8
    assert r["proxy_edit_task_count"]==4
    assert r["available_candidate_sizes"]==[8,16,32]
    assert r["unavailable_requested_candidate_sizes"]==[64]
    assert r["candidate_aggregates"]["8"][
        "source_direct_edit_delta_pixels_vs_fine_sum"]>r[
        "candidate_aggregates"]["16"]["source_direct_edit_delta_pixels_vs_fine_sum"]>0
    assert r["candidate_aggregates"]["32"][
        "source_peel_isolation_delta_pixels_vs_fine_sum"]==0
    assert r["pareto_or_optimal_partition_selected"] is False
    assert r["same_drawable_positive_pairs"]==0
    assert r["different_drawable_negative_pairs"]==0
    assert r["training_authorized"] is False


def test_missing_fine_partition_is_unavailable_not_fake_zero():
    p,m,proposal,pairs=fixture((8,16))
    r=evaluate(p,m,proposal,pairs)
    assert r["available_candidate_sizes"]==[8,16]
    assert r["unavailable_requested_candidate_sizes"]==[32,64]
    assert r["candidate_aggregates"]["16"]["source_direct_edit_delta_pixels_vs_fine_sum"]==0


@pytest.mark.parametrize("fault",[
    "forged_teacher","forged_training","forged_owner","forged_source",
    "arm_swap","duplicate_camera","mismatched_plan","wrong_resolution",
    "nonfinite","nonzero_noop","bad_fine_reference","counterfeit_source",
])
def test_court_rejects_tampering_and_pixel_replay_drift(fault):
    p,m,proposal,pairs=fixture()
    if fault=="forged_teacher":m["teacher_only"]=False
    elif fault=="forged_training":m["training_authorized"]=True
    elif fault=="forged_owner":m["graphics_owner_authorized"]=True
    elif fault=="forged_source":m["source_sha256"]="b"*64
    elif fault=="arm_swap":m["arms"][5],m["arms"][6]=m["arms"][6],m["arms"][5]
    elif fault=="duplicate_camera":m["private_view_details"][3]["view"]=2
    elif fault=="mismatched_plan":
        m["original_source_partition_edit_plan_teacher_only"]["tasks"][0][
            "source_face_teacher_only"]=19
    elif fault=="wrong_resolution":m["full_frame_resolution"]=256
    elif fault=="nonfinite":p[0,4,0,0,0]=np.nan
    elif fault=="nonzero_noop":p[3,2,40,40,0]=.99
    elif fault=="bad_fine_reference":p[4,4,24,28,3]=1.
    elif fault=="counterfeit_source":proposal["source_sha256"]="b"*64
    with pytest.raises(ValueError):
        evaluate(p,m,proposal,pairs)


def test_refinement_break_and_missing_face_rejected():
    proposal,pairs=parents()
    labels=proposal["candidates"][1]["face_slot_teacher_only"]
    labels[1],labels[12]=labels[12],labels[1]  # preserves 16 labels, violates 8->16 nesting
    with pytest.raises(ValueError,match="PARTITION_NOT_NESTED"):
        build_plan(proposal,pairs)
    proposal,pairs=parents()
    proposal["candidates"][1]["face_slot_teacher_only"][2]=99
    with pytest.raises(ValueError,match="PARTITION_CANDIDATE_COMPLETE_LABELS"):
        build_plan(proposal,pairs)


def test_view4_anchor_is_source_address_not_render_response():
    p,m,proposal,pairs=fixture()
    pairs["rows"][1]["source_face_b_teacher_only"]=31
    with pytest.raises(ValueError,match="PARTITION_COURT_EXACT_PLAN"):
        evaluate(p,m,proposal,pairs)
