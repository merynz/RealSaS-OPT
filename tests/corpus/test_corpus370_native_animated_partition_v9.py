"""V9 forged ownership, native pose, source parity, temporal edit tests.

Synthetic images are ONLY contract tests; actual native-source proof is SHA-bound
to hosted Blender original source and separate renderer consumer.
"""
import copy
import numpy as np
import pytest
from tests.corpus.test_corpus370_multi_partition_edit_v7 import parents
from tools.corpus.corpus370_partition_edit_plan_v7 import build_plan,arm_names
from tools.corpus.court_corpus370_native_animated_partition_v9 import evaluate


def scene():
    proposal,pairs=parents()
    plan=build_plan(proposal,pairs)
    arms=arm_names(plan)
    frames=np.zeros((3,4,len(arms),192,192,4),np.float32)
    source=np.zeros((192,192,4),np.float32)
    source[30:160,40:145,:]=(.3,.4,.6,1.)
    arms_index={s:i for i,s in enumerate(arms)}
    for p in range(3):
        for v in range(4):
            for arm in ("SOURCE_ORIGINAL","SOUP_FULL","SOUP_NOOP"):
                frames[p,v,arms_index[arm]]=source
            for task in plan["tasks"]:
                k=task["task_index"]
                y=44+k*18
                x=55+k*16
                for size in plan["source_geometry_candidate_sizes"]:
                    pad=plan["finest_available_reference_size"]//size
                    removed=source.copy()
                    removed[y-pad:y+7+pad,x-pad:x+7+pad,:]=0.
                    isolated=np.zeros_like(source)
                    isolated[y-pad:y+7+pad,x-pad:x+7+pad,:]=source[
                        y-pad:y+7+pad,x-pad:x+7+pad,:]
                    frames[p,v,arms_index[f"GROUP_REMOVE_K{size:03d}_T{k:02d}"]]=removed
                    frames[p,v,arms_index[f"GROUP_ISOLATE_K{size:03d}_T{k:02d}"]]=isolated
                fine=plan["finest_available_reference_size"]
                frames[p,v,arms_index[f"FINE_REMOVE_T{k:02d}"]]=frames[
                    p,v,arms_index[f"GROUP_REMOVE_K{fine:03d}_T{k:02d}"]]
                frames[p,v,arms_index[f"FINE_ISOLATE_T{k:02d}"]]=frames[
                    p,v,arms_index[f"GROUP_ISOLATE_K{fine:03d}_T{k:02d}"]]
    receipt={
        "source":{"sha256":"a"*64},
        "output":{"sha256":"b"*64},
        "sampled_frames":[
            {"action":"idle","frame":0},{"action":"idle","frame":1},
            {"action":"run","frame":0},{"action":"run","frame":2},
            {"action":"jump","frame":0},{"action":"jump","frame":3}
        ]
    }
    indices=[0,3,5]
    manifest={
        "schema":"RealSaS.Corpus370NativeAnimatedPartitionSourceRender.v9",
        "status":"REAL_AUTHORED_NATIVE_POSE_ANIMATED_RGBA_RENDER__INDEPENDENT_COURT_REQUIRED",
        "subject_id":"synthetic_only","source_sha256":"a"*64,
        "native_teacher_arrays_sha256":"b"*64,
        "source_group_proposal_sha256":"c"*64,
        "source_selected_pose_exact_teacher_replay_verified":True,
        "source_rest_exact_geometry_verified":True,
        "reference_policy":"FINEST_AVAILABLE_GEOMETRY_HYPOTHESIS_NOT_AUTHOR_TRUTH",
        "teacher_only":True,"graphics_owner_authorized":False,
        "training_authorized":False,
        "animated_render_caa_authorized":False,
        "materialization_authorized":False,
        "source_bone_names_as_targets":False,
        "plan":plan,"arms":arms,
        "candidate_sizes":plan["source_geometry_candidate_sizes"],
        "view_indices":[0,2,4,6],
        "raster_resolution":192,"cycles_samples":32,
        "pose_indices":indices,
        "selected_pose_details_teacher_only":[{
            "pose_index":p,"source_frame_native_teacher_only":receipt["sampled_frames"][p]["frame"],
            "max_original_evaluated_vs_sealed_teacher_coordinate_error":0.0,
            "source_native_exact_face_order":True,
        } for p in indices],
    }
    return frames,manifest,proposal,pairs,receipt


def test_real_pose_raster_court_qualifies_all_and_keeps_unknown():
    p,m,proposal,pairs,receipt=scene()
    out=evaluate(p,m,proposal,pairs,receipt)
    assert out["total_pose_view_census"]==12
    assert out["source_soup_parity_qualified_pose_views"]==12
    assert out["unqualified_pose_views"]==0
    assert out["candidate_aggregates"]["8"][
        "additional_direct_pixel_instances_vs_fine_proxy"]>0
    assert out["candidate_aggregates"]["32"][
        "additional_direct_pixel_instances_vs_fine_proxy"]==0
    assert out["graphics_owner_authorized"] is False
    assert out["optimal_group_count_authorized"] is False
    assert out["same_drawable_positive_pairs"]==0
    assert out["different_drawable_negative_pairs"]==0
    assert out["training_authorized"] is False


def test_original_source_parity_failure_censors_affected_pose_view():
    p,m,proposal,pairs,receipt=scene()
    p[1,2,0,30:160,40:145,:3]=1.
    out=evaluate(p,m,proposal,pairs,receipt)
    assert out["source_soup_parity_qualified_pose_views"]==11
    assert out["unqualified_pose_views"]==1
    assert out["all_pose_view_source_parity"] is False
    assert "EDIT_AUTHORITY_BLOCKED" in out["status"]
    assert out["graphics_owner_authorized"] is False


def test_zero_occupancy_is_abstain_not_render_pass():
    p,m,proposal,pairs,receipt=scene()
    p[0,0,0]=0.
    out=evaluate(p,m,proposal,pairs,receipt)
    assert out["source_soup_parity_qualified_pose_views"]==11
    assert out["parity_census"][0]["parity_failure_reason"]=="EMPTY_ANIMATED_CAMERA_OCCUPANCY"


@pytest.mark.parametrize("fault",[
    "source_mismatch","teacher_mismatch","altered_plan","wrong_camera",
    "fake_training","fake_owner","fake_caa","fake_student_bones",
    "missing_pose","pose_order","frame_forged","nonfinite",
    "bad_alpha","no_edit_arm","invalid_noop","wrong_resolution",
])
def test_forged_source_motion_render_never_mints_ownership(fault):
    p,m,proposal,pairs,receipt=copy.deepcopy(scene())
    if fault=="source_mismatch":m["source_sha256"]="e"*64
    elif fault=="teacher_mismatch":m["native_teacher_arrays_sha256"]="f"*64
    elif fault=="altered_plan":m["plan"]["tasks"][0]["source_face_teacher_only"]=19
    elif fault=="wrong_camera":m["view_indices"]=[0,2,4,7]
    elif fault=="fake_training":m["training_authorized"]=True
    elif fault=="fake_owner":m["graphics_owner_authorized"]=True
    elif fault=="fake_caa":m["animated_render_caa_authorized"]=True
    elif fault=="fake_student_bones":m["source_bone_names_as_targets"]=True
    elif fault=="missing_pose":m["pose_indices"]=[0,1]
    elif fault=="pose_order":m["selected_pose_details_teacher_only"][0]["pose_index"]=3
    elif fault=="frame_forged":m["selected_pose_details_teacher_only"][0][
        "source_frame_native_teacher_only"]=9
    elif fault=="nonfinite":p[0,0,0,30,40,0]=np.nan
    elif fault=="bad_alpha":p[0,0,0,30,40,3]=1.4
    elif fault=="no_edit_arm":m["arms"]=m["arms"][:-1]
    elif fault=="invalid_noop":m["source_rest_exact_geometry_verified"]=False
    elif fault=="wrong_resolution":m["raster_resolution"]=128
    with pytest.raises(ValueError):
        evaluate(p,m,proposal,pairs,receipt)
