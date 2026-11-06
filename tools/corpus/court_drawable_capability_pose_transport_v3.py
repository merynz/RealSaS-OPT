#!/usr/bin/env python3
"""Independent source-private pose-dependent original Cat UV edit capability court.

The original Cat authored glTF supplies the physically rendered source edit
oracle. A restricted representation stores the RGB edit delta on a fixed
camera/sample image plane at original pose ordinal 0, with no warp or source
address lookup. This court independently compares it to original source
authored pose ordinal 14 heldout physical edits. NO artist ownership labels.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA="RealSaS.Corpus370IndependentOriginalCatPoseUVTransportCourt.v3"
PARENT="RealSaS.Corpus370OriginalCatAuthoredPoseUVPhysicalSamples.v3"
TOL=0.025
VIEWS=(0,2,4,6)
OFFSETS=(0,16)
POSES=(0,14)
ARMS=("BASE","SINGLE_0","SINGLE_1","SINGLE_2")
FLAGS=("graphics_owner_authorized","iris_training_authorized",
       "tessa_training_authorized","materialization_authorized","training_authorized")
SOURCE="1e73aaf7d497dd7d2946badf92d189d2333f163e66dced91feb446bc9bdee850"
ATLAS="2f6d0032e51c2d49b13b8e689d59eb39aa3ee6c32f2a1851213e11fc7593177d"
TEACHER="c2792c8efb36c4ccdf9178c6b933351b50364bbd28707e9365a3c115db00d603"
TEACHER_RECEIPT="a0f224019deca20bf5d807ab4d27a7c341a65bc21930fc99ad8e88ca5b4bc023"
CAMERA="300b935a47379f6c44d4b4356554d1c32a86ca3b732a62405c73da9e0ec6adf0"
POLICY="c8fe4e307ddbbaf5cc0892cffc4267bd08e394f2299f4e70155ffc6207190a4b"
PLAN="8be8fccde3d9cc97f7ba452fd50c1a9f4ebea2e34da590a22d30872b2b0f497b"
ATLAS_ASSETS="0e9a2fcf6a2c048eb6e99048793290b8865d802689f59b7c8c20bc6390cf479a"

def require(ok,code):
    if not ok:raise ValueError("INDEPENDENT_CAT_V3_POSE_UV_"+code)

def sha(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file,"sha256").hexdigest()

def read(path):return json.loads(Path(path).read_text())

def premul(samples):
    s=np.asarray(samples,dtype=np.float64)
    return np.concatenate([s[...,:3]*s[...,3:4],s[...,3:4]],axis=-1)

def recount(samples):
    require(samples.shape==(2,4,2,4,192,192,4) and
        np.isfinite(samples).all() and
        np.min(samples[...,3])>=-1e-5 and
        np.max(samples[...,3])<=1.00001,
        "SOURCE_REAL_CYCLES_SAMPLES_SHAPE")
    rgba=premul(samples)
    fixed_ref=rgba[0,:,:,1:]-rgba[0,:,:,0:1]
    pose1_base=rgba[1,:,:,0:1]
    fixed_prediction=pose1_base+fixed_ref
    original_pose1_target=rgba[1,:,:,1:]
    error=np.max(np.abs(fixed_prediction-original_pose1_target),axis=-1)
    incorrect=error>TOL
    results=[]
    totals=[0,0,0]
    mismatches=0
    for vi,view in enumerate(VIEWS):
        for edit_idx in range(3):
            diff0=np.max(np.abs(rgba[0,vi,:,edit_idx+1]-
                                rgba[0,vi,:,0]),axis=-1)
            diff1=np.max(np.abs(rgba[1,vi,:,edit_idx+1]-
                                rgba[1,vi,:,0]),axis=-1)
            n0=int(np.count_nonzero(diff0>TOL))
            n1=int(np.count_nonzero(diff1>TOL))
            totals[edit_idx]+=n0+n1
            failures=int(np.count_nonzero(incorrect[vi,:,edit_idx]))
            mismatches+=failures
            where=np.argwhere(incorrect[vi,:,edit_idx])
            witness=None
            if len(where):
                s,y,x=[int(z) for z in where[0]]
                witness={
                    "original_source_view":view,"sample_offset":OFFSETS[s],
                    "source_uv_operation_index":edit_idx,
                    "pixel_yx":[y,x],
                    "predicted_fixed_screen_premultiplied_rgba":
                        fixed_prediction[vi,s,edit_idx,y,x].tolist(),
                    "actual_pose_edited_source_premultiplied_rgba":
                        original_pose1_target[vi,s,edit_idx,y,x].tolist(),
                    "max_channel_absolute_difference":float(error[vi,s,edit_idx,y,x])
                }
            results.append({
                "view_index":view,"edit_index":edit_idx,
                "source_pose0_observable_sample_incidents":n0,
                "source_pose14_observable_sample_incidents":n1,
                "source_edit_support_gained_at_pose14":int(np.count_nonzero(
                    (diff1>TOL)&(diff0<=TOL))),
                "source_edit_support_lost_at_pose14":int(np.count_nonzero(
                    (diff0>TOL)&(diff1<=TOL))),
                "fixed_screen_pose14_rgba_mismatches":failures,
                "fixed_screen_pose14_max_channel_error":float(
                    np.max(error[vi,:,edit_idx])),
                "first_failure_private_source_pixel":witness,
            })
    alpha_drift=float(np.max(np.abs(samples[...,3]-
                                        samples[:,:,:,0:1,...,3])))
    return {
        "per_view_edit":results,
        "source_visible_edit_sample_incidents_both_poses_by_edit":totals,
        "fixed_screen_pose14_total_mismatched_sample_incidents":mismatches,
        "fixed_screen_pose14_max_error":float(np.max(error)),
        "source_all_arms_alpha_max_drift":alpha_drift,
        "source_edit_target_selects_atlas_pixels_from_original_glb_uv":True,
        "source_is_original_authored_rig_render_not_new_puppet_runtime":True,
        "fixed_screen_uses_pose14_edited_target_to_choose_delta":False,
    }

def validate_provenance(manifest,*,teacher_receipt,uvplan,assets,sample_sha):
    require(manifest.get("schema")==PARENT and
        manifest.get("status")==
            "PHYSICAL_SOURCE_NATIVE_POSE_UV_EDITS_RENDERED__INDEPENDENT_COURT_REQUIRED" and
        manifest.get("original_source_sha256")==SOURCE and
        manifest.get("original_texture_sha256")==ATLAS and
        manifest.get("original_native_teacher_sha256")==TEACHER and
        manifest.get("original_teacher_receipt_sha256")==TEACHER_RECEIPT and
        manifest.get("original_camera_sha256")==CAMERA and
        manifest.get("original_material_shader_sha256")==POLICY and
        manifest.get("original_source_uv_plan_sha256")==PLAN and
        manifest.get("original_source_atlas_edits_manifest_sha256")==ATLAS_ASSETS and
        manifest.get("source_samples_npz_sha256")==sample_sha and
        manifest.get("source_samples_npz_shape")==[2,4,2,4,192,192,4],
        "SOURCE_ORIGINAL_IMMUTABLE_IDENTITY")
    require(manifest.get("teacher_pose_sample_ordinals")==list(POSES) and
        manifest.get("original_calibrated_view_indices")==list(VIEWS) and
        manifest.get("original_same_cycles_sample_offsets")==list(OFFSETS) and
        manifest.get("original_source_uv_edit_arms")==list(ARMS) and
        manifest.get("original_source_sample_count")==64 and
        manifest.get("original_pose_dependent_uv_surface_texture_rendered") is True and
        manifest.get("source_sample_alpha_policy")==
            "ORIGINAL_OPAQUE_TEXTURE__NO_NONBINARY_ALPHA" and
        manifest.get("source_drawable_owner")=="UNKNOWN" and
        manifest.get("same_drawable_training_pairs_qualified")==0 and
        manifest.get("different_drawable_training_pairs_qualified")==0 and
        all(manifest.get(k) is False for k in FLAGS),
        "SOURCE_RENDER_SCOPE_OR_FORGED_AUTHORITY")
    require(teacher_receipt.get("source",{}).get("sha256")==SOURCE and
        teacher_receipt.get("output",{}).get("sha256")==TEACHER and
        teacher_receipt.get("teacher_only") is True and
        teacher_receipt.get("sampled_frame_count")==32 and
        uvplan.get("source_original_glb_sha256")==SOURCE and
        uvplan.get("source_original_texture_sha256")==ATLAS and
        uvplan.get("source_uv_coordinate_origin")==
            "GLTF_2_IMAGE_TOP_LEFT__UV_V_INCREASES_DOWN" and
        assets.get("source_operation_plan_sha256")==PLAN and
        [x.get("arm") for x in assets.get("original_atlas_arm_rows",[])][:4]==list(ARMS) and
        manifest.get("source_atlas_arm_sha256")==[
            x["source_atlas_rgba_sha256"]
            for x in assets.get("original_atlas_arm_rows",[])[:4]],
        "SOURCE_TEACHER_ACTION_AND_ATLAS_ADDRESS")
    rows=manifest.get("original_source_pose_geometry_replay_receipts")
    require(type(rows) is list and len(rows)==2 and
        [x.get("teacher_pose_sample_ordinal") for x in rows]==list(POSES) and
        all(x.get("source_teacher_original_face_order_matches") is True and
            x.get("source_pose_original_vertex_count")==1204 and
            0<=x.get("source_geometry_max_abs_error",float("inf")) <=
                x.get("source_geometry_tolerance",0)
            for x in rows) and
        manifest.get("source_pose_world_max_displacement",0)>
            manifest.get("source_pose_tolerance",float("inf")),
        "ORIGINAL_NATIVE_POSE_GEOMETRY_NOT_OBSERVED")

def run(root,output):
    names={
        "original_glb":("src/animal-cat.glb",SOURCE),
        "atlas":("src/Textures/colormap.png",ATLAS),
        "teacher":("teacher.npz",TEACHER),
        "teacher_receipt":("teacher.json",TEACHER_RECEIPT),
        "cameras":("camera.json",CAMERA),
        "shader":("policy.json",POLICY),
        "uvplan":("plan.json",PLAN),
        "atlas_assets":("atlas-assets.json",ATLAS_ASSETS),
    }
    for code,(name,dig) in names.items():
        require(sha(root/name)==dig,"PARENT_SHA:"+code)
    manifest_path=root/"render/PRIVATE_CAT_NATIVE_POSE_UV_EDIT_CYCLES_V3.json"
    sample_path=root/"render/PRIVATE_CAT_NATIVE_POSE_UV_EDIT_CYCLES_SAMPLES_V3.npz"
    manifest=read(manifest_path)
    parent_teacher=read(root/"teacher.json")
    parent_plan=read(root/"plan.json")
    parent_atlas=read(root/"atlas-assets.json")
    validate_provenance(manifest,teacher_receipt=parent_teacher,
        uvplan=parent_plan,assets=parent_atlas,sample_sha=sha(sample_path))
    with np.load(sample_path,allow_pickle=False) as source:
        data=np.asarray(source["source_native_pose_uv_rgba"],dtype=np.float32)
    measured=recount(data)
    nonvacuity=[0,0,0]
    for row in measured["per_view_edit"]:
        nonvacuity[row["edit_index"]]+=row["source_pose0_observable_sample_incidents"]
    visible14=[0,0,0]
    for row in measured["per_view_edit"]:
        visible14[row["edit_index"]]+=row["source_pose14_observable_sample_incidents"]
    missing=[]
    for i in range(3):
        if nonvacuity[i]<32:missing.append(f"ORIGINAL_POSE0_EDIT{i}_NOT_OBSERVABLE")
        if visible14[i]<32:missing.append(f"ORIGINAL_POSE14_EDIT{i}_NOT_OBSERVABLE")
    if measured["source_all_arms_alpha_max_drift"]>1e-6:
        missing.append("SOURCE_ALPHA_DRIFT")
    status=("PASS_ORIGINAL_CAT_POSE_UV_EDIT_PHYSICAL_SOURCE__"
        "FIXED_SCREEN_CLASS_"+("VIOLATED" if measured[
            "fixed_screen_pose14_total_mismatched_sample_incidents"]>0 else "SATISFIED")
        if not missing else "FAIL_ORIGINAL_CAT_POSE_UV_EDIT_SOURCE_NONVACUITY")
    result={
        "schema":SCHEMA,"status":status,
        "source_original_shader_domain":"ADMITTED_UNLIT_LINKED_TEXTURE_ONLY",
        "source_original_pose_scope":"AUTHORED_TEACHER_ORDINAL_0_AND_14_ONLY",
        "source_original_edited_atlas_operation_scope":"THREE_FIXED_ANONYMOUS_UV_ADDRESSES",
        "source_original_blender_physical_render_manifest_sha256":sha(manifest_path),
        "source_original_blender_physical_sample_sha256":sha(sample_path),
        "source_original_teacher_sha256":TEACHER,
        "source_original_uv_plan_sha256":PLAN,
        "source_original_atlas_edits_sha256":ATLAS_ASSETS,
        "source_original_pose_world_max_displacement":
            manifest["source_pose_world_max_displacement"],
        "source_original_sampler_rgba_premultiplied_tolerance":TOL,
        "actual_source_original_full_physical_renders":64,
        "source_fixed_screen_delta_class_status":
            "VIOLATED" if measured[
                "fixed_screen_pose14_total_mismatched_sample_incidents"]>0
                else "SATISFIED",
        "source_uv_surface_attached_edit_native_source_status":
            "SATISFIED_PHYSICAL_ORIGINAL_SOURCE_WITNESS_ONLY" if not missing else "UNKNOWN",
        "source_missing_evidence_obligations":missing,
        "source_edit_observed_pose0_by_operation":nonvacuity,
        "source_edit_observed_pose14_by_operation":visible14,
        "measured_original_physical_source":measured,
        "original_drawable_artist_partition":"UNKNOWN_NOT_MEASURED",
        "pose_deformation_in_new_puppet_runtime":"UNKNOWN_NOT_MEASURED",
        "attachment_release":"UNKNOWN_NOT_MEASURED",
        "secondary_motion":"UNKNOWN_NOT_MEASURED",
        "same_drawable_pairs":[],"different_drawable_pairs":[],
        **{k:False for k in FLAGS},
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":status,
        "source_pose0_visible_edits":nonvacuity,
        "source_pose14_visible_edits":visible14,
        "frozen_screen_pixel_sample_failures":
            measured["fixed_screen_pose14_total_mismatched_sample_incidents"],
        "new_original_source_renders":64,"owner_labels":0},sort_keys=True),flush=True)
    require(not missing,"SOURCE_EDIT_POSE_NONVACUITY")

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--input-root",type=Path,required=True)
    a.add_argument("--output",type=Path,required=True)
    x=a.parse_args()
    run(x.input_root,x.output)

if __name__=="__main__":main()
