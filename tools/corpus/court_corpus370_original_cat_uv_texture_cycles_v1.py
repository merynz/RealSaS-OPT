#!/usr/bin/env python3
"""Independent original Cat UV texture edit algebra court, source-private.

No Blender producer imports or source-face owner inference. Reconstructs all
source atlas edit pixels independently and checks all original full-scene
Cycles sample pair/joint edits against observed singles + baseline only.
No same/different drawable label or IRIS materialization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

SCHEMA="RealSaS.Corpus370IndependentOriginalCatSourceUVTextureEditCourt.v1"
PARENT="RealSaS.Corpus370OriginalCatSourceTextureSameCyclesEditEvidence.v1"
PLAN="RealSaS.Corpus370OriginalSourceUVAtlasEditPlan.v1"
ATLAS="RealSaS.Corpus370OriginalSourceUVAtlasEditImages.v1"
ARMS=("BASE","SINGLE_0","SINGLE_1","SINGLE_2",
      "PAIR_01","PAIR_02","PAIR_12","JOINT_012")
SETS=((),(0,),(1,),(2,),(0,1),(0,2),(1,2),(0,1,2))
VIEWS=(1,3,5,7)
INDICES=(0,4,8,12,16,20,24,28)
TOL=.025
NONVACUOUS=32
FLAGS=("graphics_owner_authorized","iris_training_authorized",
       "tessa_training_authorized","materialization_authorized","training_authorized")

def req(ok,code):
    if not ok:raise ValueError("INDEPENDENT_CAT_SOURCE_UV_"+code)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))

def image(path):
    with Image.open(path) as img:
        req(img.mode=="RGBA" and img.size==(512,512),
            "ORIGINAL_SOURCE_TEXTURE_IMAGE_FORMAT")
        return np.array(img,dtype=np.uint8)

def verify_original_atlas_variants(original_path,plan,prepared,source_dir):
    src=image(original_path)
    req(plan.get("schema")==PLAN and prepared.get("schema")==ATLAS and
        prepared.get("source_operation_plan_sha256") is not None and
        plan.get("source_original_texture_sha256")==sha(original_path) and
        prepared.get("original_texture_source_sha256")==sha(original_path) and
        plan.get("original_source_uv_operation_count")==3 and
        prepared.get("source_atlas_edit_rule")==
            "ATLAS_TILE_U8_RGB_INVERT__ALPHA_UNCHANGED" and
        np.all(src[...,3]==255),
        "SOURCE_ORIGINAL_TEXTURE_AND_OPERATION_POLICY")
    selections=plan["selected_original_source_uv_edits"]
    rects=[tuple(x["source_rgba_png_pixel_rectangle_xyxy_exclusive"])
           for x in selections]
    masks=[]
    for k,rect in enumerate(rects):
        u,v=selections[k]["uv_tile_u"],selections[k]["uv_tile_v"]
        req(rect==(u*64,(7-v)*64,(u+1)*64,(8-v)*64)
            and selections[k]["source_uv_edit_operation_index"]==k,
            "SOURCE_UV_TILE_PHYSICAL_PNG_RECTANGLE")
        m=np.zeros((512,512),dtype=np.bool_)
        x0,y0,x1,y1=rect
        req(x1-x0==64 and y1-y0==64,"SOURCE_TEXTURE_RECTANGLE_SIZE")
        m[y0:y1,x0:x1]=True
        masks.append(m)
    req(not np.any(masks[0]&masks[1]) and
        not np.any(masks[0]&masks[2]) and
        not np.any(masks[1]&masks[2]),
        "SOURCE_TEXTURE_EDIT_UV_TILE_COLLISION")
    rows=prepared["original_atlas_arm_rows"]
    req([z["arm"] for z in rows]==list(ARMS) and
        [tuple(z["selected_source_uv_operation_indices"]) for z in rows]==list(SETS),
        "SOURCE_TEXTURE_ARM_INDEX_IDENTITY")
    for row,selected in zip(rows,SETS):
        path=(source_dir/row["filename"]).resolve()
        req(path.is_relative_to(source_dir.resolve()) and
            path.is_file() and sha(path)==row["source_atlas_rgba_sha256"],
            "SOURCE_PHYSICAL_TEXTURE_INPUT_SHA")
        got=image(path)
        expected=src.copy()
        for i in selected:
            expected[...,0][masks[i]]=255-src[...,0][masks[i]]
            expected[...,1][masks[i]]=255-src[...,1][masks[i]]
            expected[...,2][masks[i]]=255-src[...,2][masks[i]]
        req(np.array_equal(got,expected) and
            row["original_atlas_edited_rgb_pixel_count"]==4096*len(selected) and
            row.get("source_texture_alpha_preserved_exactly") is True,
            "SOURCE_TEXTURE_EDIT_BYTES_WRONG_OR_MUTATED_OUTSIDE_UV")
    return masks

def premul(x):
    c=np.asarray(x,dtype=np.float64)
    return np.concatenate([c[...,:3]*c[...,3:4],c[...,3:4]],axis=-1)

def score_sample_source_rgba(a):
    req(a.shape==(8,8,192,192,4) and np.isfinite(a).all() and
        np.min(a[...,3])>=-1e-5 and np.max(a[...,3])<=1+1e-5,
        "ORIGINAL_CYCLES_SOURCE_SAMPLE_RGBA_SHAPE")
    c=premul(a)
    base=c[:,0]
    singles=c[:,1:4]
    sample_distinct=[np.max(np.abs(singles[:,i]-base),axis=-1)>TOL
                     for i in range(3)]
    outcomes=[]
    for target_arm,selection in [(4,(0,1)),(5,(0,2)),(6,(1,2)),(7,(0,1,2))]:
        prediction=base.copy()
        for i in selection:
            prediction+=(singles[:,i]-base)
        actual=c[:,target_arm]
        delta=np.max(np.abs(prediction-actual),axis=-1)
        mismatches=np.argwhere(delta>TOL)
        aggregate_delta=np.max(np.abs(
            np.mean(prediction,axis=0)-np.mean(actual,axis=0)),axis=-1)
        # No prediction lookup from pairwise/joint FULL render before comparing.
        outcomes.append({
            "actual_heldout_full_source_arm":ARMS[target_arm],
            "source_single_edited_arm_input_indices":list(selection),
            "source_per_sample_rgba_mismatch_count":len(mismatches),
            "source_per_sample_rgba_max_delta":float(np.max(delta)),
            "source_eight_sample_average_rgba_mismatch_count":
                int(np.count_nonzero(aggregate_delta>TOL)),
            "source_eight_sample_average_rgba_max_delta":
                float(np.max(aggregate_delta)),
            "source_private_first_12_failure_indices_s_y_x":
                mismatches[:12].tolist(),
            "prediction_uses_heldout_pair_or_joint_target":False,
        })
    source_alpha_difference=float(np.max(
        np.abs(a[... ,3]-a[:, :, 0:1, ..., 3]))) if False else 0.0
    # Correct axis: a [S,Arms,H,W,4]. All arms must have source
    # source-native alpha exactly matching baseline (no geometry edit).
    alpha_drifts=np.max(np.abs(a[...,3]-a[:,0:1,...,3]),axis=(0,2,3))
    overlaps={(f"{i}_{j}"):int(np.count_nonzero(
        sample_distinct[i]&sample_distinct[j]))
        for i,j in [(0,1),(0,2),(1,2)]}
    return {
        "edited_single_source_visible_sample_count":[
            int(np.count_nonzero(mask)) for mask in sample_distinct],
        "pairwise_single_edit_image_support_overlaps":overlaps,
        "all_source_arm_alpha_max_drift":float(np.max(alpha_drifts)),
        "source_heldout_pair_joint_court":outcomes,
    }

def run(source,atlas,plan_path,variants_path,variants_dir,
        manifest_path,frames_path,out_path):
    source_plan=read(plan_path)
    edits=read(variants_path)
    manifest=read(manifest_path)
    req(sha(source)==source_plan.get("source_original_glb_sha256") and
        sha(atlas)==source_plan.get("source_original_texture_sha256"),
        "SOURCE_ORIGINAL_RAW_BYTES_SHA")
    req(manifest.get("schema")==PARENT and
        manifest.get("status")==
            "ORIGINAL_SOURCE_CAT_TEXTURE_UV_CYCLES_RENDERED__INDEPENDENT_COURT_REQUIRED"
        and manifest.get("original_source_sha256")==sha(source) and
        manifest.get("original_source_atlas_sha256")==sha(atlas) and
        manifest.get("original_source_uv_edit_plan_sha256")==sha(plan_path) and
        manifest.get("original_source_prepared_atlas_manifest_sha256")==sha(variants_path)
        and manifest.get("source_uv_atlas_edit_rule")==
            "ATLAS_TILE_U8_RGB_INVERT__ALPHA_UNCHANGED" and
        [x["source_atlas_rgba_sha256"] for x in edits["original_atlas_arm_rows"]]
             ==manifest.get("source_atlas_texture_arm_sha256") and
        manifest.get("source_original_sample_npz_sha256")==sha(frames_path) and
        manifest.get("source_original_sample_npz_shape")==[4,8,8,192,192,4],
        "SOURCE_SAME_SAMPLE_MANIFEST_LINEAGE")
    req(manifest.get("original_source_rest_only") is True and
        manifest.get("source_original_pose2_edit_court_passed") is False and
        manifest.get("all_texture_edit_sample_frames_independent_full_source_renders") is True
        and manifest.get("camera_indices")==list(VIEWS) and
        manifest.get("cycles_subset_indices")==list(INDICES) and
        manifest.get("cycles_configured_samples")==32 and
        manifest.get("cycles_seed")==0 and
        manifest.get("cycles_adaptive_sampling") is False and
        manifest.get("cycles_subset_length")==1 and
        manifest.get("cycles_pixel_filter")=="BOX" and
        manifest.get("cycles_filter_width")==.01 and
        manifest.get("cycles_resolution")==192 and
        manifest.get("original_source_uv_texture_arms")==list(ARMS) and
        all(z.get(flag) is False for z in (source_plan,edits,manifest) for flag in FLAGS),
        "ORIGINAL_SOURCE_SAMPLING_POLICY_AND_NONOWNER")
    verify_original_atlas_variants(atlas,source_plan,edits,variants_dir)
    with np.load(frames_path,allow_pickle=False) as archive:
        samples=np.asarray(archive["source_uv_edit_cycles_sample_rgba"],dtype=np.float32)
    req(samples.shape==(4,8,8,192,192,4) and np.isfinite(samples).all(),
        "ORIGINAL_SOURCE_PHYSICAL_SAMPLES")
    per_view=[]
    for ix,v in enumerate(VIEWS):
        metrics=score_sample_source_rgba(samples[ix])
        per_view.append({"original_source_view":v,**metrics})
    total_per_edit=[sum(row["edited_single_source_visible_sample_count"][i]
                      for row in per_view) for i in range(3)]
    all_mismatch=sum(m["source_per_sample_rgba_mismatch_count"]+
          m["source_eight_sample_average_rgba_mismatch_count"]
          for row in per_view for m in row["source_heldout_pair_joint_court"])
    max_alpha=max(x["all_source_arm_alpha_max_drift"] for x in per_view)
    failed=[]
    for i,v in enumerate(total_per_edit):
        if v<NONVACUOUS:failed.append(f"SOURCE_UV_EDIT_{i}_NONVACUITY")
    if all_mismatch:failed.append("SOURCE_TEXTURE_PAIR_JOINT_NONADDITIVE")
    if max_alpha>1e-6:failed.append("SOURCE_RGB_EDIT_INCORRECTLY_CHANGED_ALPHA")
    result={
        "schema":SCHEMA,
        "status":("PASS_ORIGINAL_CAT_LINKED_UV_EDIT_CONSEQUENCE_SUPERPOSITION__OWNER_UNKNOWN"
            if not failed else "FAIL_ORIGINAL_CAT_LINKED_UV_EDIT_CONSEQUENCE"),
        "source_original_glb_sha256":sha(source),
        "original_source_image_sha256":sha(atlas),
        "original_source_uv_edit_plan_sha256":sha(plan_path),
        "original_source_atlas_edit_assets_manifest_sha256":sha(variants_path),
        "original_source_blender_render_manifest_sha256":sha(manifest_path),
        "original_source_blender_sample_npz_sha256":sha(frames_path),
        "original_source_texture_uv_tile_operations":[
            [x["uv_tile_u"],x["uv_tile_v"]] for x in
            source_plan["selected_original_source_uv_edits"]],
        "physical_original_sample_frames":4*8*8,
        "source_single_edit_observable_samples_per_operation":total_per_edit,
        "source_unknown_alpha_drift":max_alpha,
        "independent_court_per_camera":per_view,
        "failed_obligations":failed,
        "source_sampler_tolerance_rgba_premultiplied_max":TOL,
        "source_face_count":source_plan["source_glb_triangle_count"],
        "source_pose_scope":"ORIGINAL_CAT_REST_ONLY",
        "source_one_sample_edit_effect_from_raw_glb_without_single_render":
            "NOT_ESTABLISHED",
        "source_atlas_region_is_art_drawable":False,
        "source_original_artist_drawable_owner":"UNKNOWN",
        "same_drawable_training_pairs_qualified":0,
        "different_drawable_training_pairs_qualified":0,
        **{k:False for k in FLAGS},
    }
    out_path.parent.mkdir(parents=True,exist_ok=True)
    out_path.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":result["status"],
        "single_source_edit_changed_samples":total_per_edit,
        "independent_heldout_source_joint_errors":all_mismatch,
        "failed_obligations":failed,"training_authorized":False},sort_keys=True),
        flush=True)
    req(not failed,"ORIGINAL_SOURCE_UV_PHYSICAL_CONSEQUENCE_FAIL")

def main():
    a=argparse.ArgumentParser(description=__doc__)
    for k in ("source","atlas","source-plan","prepared-atlas-manifest",
              "prepared-atlas-dir","render-manifest","source-samples","output"):
        a.add_argument("--"+k,type=Path,required=True)
    args=a.parse_args()
    run(args.source,args.atlas,args.source_plan,args.prepared_atlas_manifest,
        args.prepared_atlas_dir,args.render_manifest,args.source_samples,args.output)

if __name__=="__main__":main()
