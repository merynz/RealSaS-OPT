"""Adversarial original Cat native authored pose edit physical source decision tests."""
import copy

import numpy as np
import pytest

from tools.corpus.court_drawable_capability_pose_transport_v3 import (
    recount,validate_provenance,FLAGS,POSES,VIEWS,OFFSETS,ARMS,
    SOURCE,ATLAS,TEACHER,TEACHER_RECEIPT,CAMERA,POLICY,PLAN,ATLAS_ASSETS
)

def synthetic(shift=10):
    a=np.zeros((2,4,2,4,192,192,4),dtype=np.float32)
    a[...,3]=1
    a[...,0]=.2
    for pose in range(2):
        for i in range(3):
            x=15+i*40+pose*shift
            a[pose,:,:,i+1,:,x:x+10,1]=.9
    return a

def fixture():
    rows=[{"source_atlas_rgba_sha256":str(i)*64,"arm":arm}
          for i,arm in enumerate(ARMS)]
    teacher={"source":{"sha256":SOURCE},"output":{"sha256":TEACHER},
             "teacher_only":True,"sampled_frame_count":32}
    plan={"source_original_glb_sha256":SOURCE,
          "source_original_texture_sha256":ATLAS,
          "source_uv_coordinate_origin":"GLTF_2_IMAGE_TOP_LEFT__UV_V_INCREASES_DOWN"}
    assets={"source_operation_plan_sha256":PLAN,
            "original_atlas_arm_rows":rows}
    manifest={
        "schema":"RealSaS.Corpus370OriginalCatAuthoredPoseUVPhysicalSamples.v3",
        "status":"PHYSICAL_SOURCE_NATIVE_POSE_UV_EDITS_RENDERED__INDEPENDENT_COURT_REQUIRED",
        "original_source_sha256":SOURCE,
        "original_texture_sha256":ATLAS,
        "original_native_teacher_sha256":TEACHER,
        "original_teacher_receipt_sha256":TEACHER_RECEIPT,
        "original_camera_sha256":CAMERA,
        "original_material_shader_sha256":POLICY,
        "original_source_uv_plan_sha256":PLAN,
        "original_source_atlas_edits_manifest_sha256":ATLAS_ASSETS,
        "source_samples_npz_sha256":"f"*64,
        "source_samples_npz_shape":[2,4,2,4,192,192,4],
        "teacher_pose_sample_ordinals":list(POSES),
        "original_calibrated_view_indices":list(VIEWS),
        "original_same_cycles_sample_offsets":list(OFFSETS),
        "original_source_uv_edit_arms":list(ARMS),
        "original_source_sample_count":64,
        "original_pose_dependent_uv_surface_texture_rendered":True,
        "source_sample_alpha_policy":"ORIGINAL_OPAQUE_TEXTURE__NO_NONBINARY_ALPHA",
        "source_drawable_owner":"UNKNOWN",
        "same_drawable_training_pairs_qualified":0,
        "different_drawable_training_pairs_qualified":0,
        "source_atlas_arm_sha256":[r["source_atlas_rgba_sha256"] for r in rows],
        "source_pose_world_max_displacement":.3,
        "source_pose_tolerance":1e-6,
        "original_source_pose_geometry_replay_receipts":[{
            "teacher_pose_sample_ordinal":ordinal,
            "source_teacher_original_face_order_matches":True,
            "source_pose_original_vertex_count":1204,
            "source_geometry_max_abs_error":1e-8,
            "source_geometry_tolerance":1e-6
        } for ordinal in POSES],
        **{k:False for k in FLAGS}
    }
    return manifest,teacher,plan,assets

def test_source_pose_transport_has_actual_frozen_screen_counterexample():
    r=recount(synthetic(shift=12))
    assert r["fixed_screen_pose14_total_mismatched_sample_incidents"]>0
    assert r["fixed_screen_pose14_max_error"]>.025
    assert all(z["source_edit_support_gained_at_pose14"]>0 for z in r["per_view_edit"])
    assert all(z["source_edit_support_lost_at_pose14"]>0 for z in r["per_view_edit"])
    assert r["source_all_arms_alpha_max_drift"]==0
    assert all(v>=32 for v in r["source_visible_edit_sample_incidents_both_poses_by_edit"])

def test_pose_frozen_candidate_not_forced_to_fail_when_positions_equal():
    r=recount(synthetic(shift=0))
    assert r["fixed_screen_pose14_total_mismatched_sample_incidents"]==0
    assert r["fixed_screen_pose14_max_error"]==0

def test_real_pose_changes_do_not_fake_native_texture_alpha():
    a=synthetic()
    a[1,0,0,2,10,10,3]=.7
    r=recount(a)
    assert r["source_all_arms_alpha_max_drift"]>.25

@pytest.mark.parametrize("shape",[(1,4,2,4,192,192,4),
                                     (2,4,2,3,192,192,4),
                                     (2,3,2,4,192,192,4)])
def test_unknown_source_pose_or_camera_cannot_be_silently_relabelled(shape):
    with pytest.raises(ValueError,match="SOURCE_REAL_CYCLES_SAMPLES_SHAPE"):
        recount(np.zeros(shape,dtype=np.float32))

def test_provenance_original_native_teacher_and_glTF_atlas():
    m,t,p,a=fixture()
    validate_provenance(m,teacher_receipt=t,uvplan=p,assets=a,sample_sha="f"*64)

@pytest.mark.parametrize("attack",[
    "invent_owner","admit_training","fake_pose_geometry","zero_displacement",
    "wrong_v_axis","source_sha","pose_ordinal","camera_scope",
    "sample_offset","leak_source_atlas","unknown_alpha","source_array_sha",
])
def test_original_source_provenance_adversarial_attacks_fail_closed(attack):
    m,t,p,a=fixture()
    if attack=="invent_owner":m["same_drawable_training_pairs_qualified"]=3
    elif attack=="admit_training":m["iris_training_authorized"]=True
    elif attack=="fake_pose_geometry":
        m["original_source_pose_geometry_replay_receipts"][0][
            "source_geometry_max_abs_error"]=.02
    elif attack=="zero_displacement":m["source_pose_world_max_displacement"]=0
    elif attack=="wrong_v_axis":p["source_uv_coordinate_origin"]="BLENDER_BOTTOM_LEFT"
    elif attack=="source_sha":m["original_source_sha256"]="0"*64
    elif attack=="pose_ordinal":m["teacher_pose_sample_ordinals"]=[0,1]
    elif attack=="camera_scope":m["original_calibrated_view_indices"]=[1,3,5,7]
    elif attack=="sample_offset":m["original_same_cycles_sample_offsets"]=[0,8]
    elif attack=="leak_source_atlas":a["source_operation_plan_sha256"]="0"*64
    elif attack=="unknown_alpha":m["source_sample_alpha_policy"]="NATIVE_ALPHA_BLEND"
    elif attack=="source_array_sha":m["source_samples_npz_sha256"]="0"*64
    with pytest.raises(ValueError):
        validate_provenance(m,teacher_receipt=t,uvplan=p,assets=a,sample_sha="f"*64)

def test_fixed_screen_counterexample_records_physical_original_pose_indices():
    r=recount(synthetic(shift=12))
    examples=[v["first_failure_private_source_pixel"] for v in r["per_view_edit"]]
    assert all(e is not None for e in examples)
    assert all(e["sample_offset"] in (0,16) and
               e["source_uv_operation_index"] in (0,1,2) and
               e["max_channel_absolute_difference"]>.025 for e in examples)
    assert all(len(e["actual_pose_edited_source_premultiplied_rgba"])==4 for e in examples)

def test_source_teacher_shape_obligations_require_native_geometry():
    m,t,p,a=fixture()
    t["teacher_only"]=False
    with pytest.raises(ValueError):
        validate_provenance(m,teacher_receipt=t,uvplan=p,assets=a,sample_sha="f"*64)
