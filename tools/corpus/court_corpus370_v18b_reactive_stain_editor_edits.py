#!/usr/bin/env python3
"""Independent REAL original-source edit-collateral court for V18a stain layers.

Comparison is between (1) original Blender original-source posed scene, (2)
same original UV/material source-face soup and (3) physically rendered group
hide/isolate, single-face source-target edit proxies, across four original
calibrated cameras at UNSEEN pose2. This is an operational edit diagnosis,
not recovery of human-intended artist layer identity.

Ground truth is the *observed source editing consequence*, not an existing
fine K partition or any named bone/material/skin owner.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from tools.corpus.compile_corpus370_v18a_reactive_staining import THRESHOLDS
from tools.corpus.corpus370_native_raster_join_plan_v14 import arms_for_plan

SCHEMA="RealSaS.Corpus370IndependentReactiveStainHeldoutSourceEditCourt.v18b"
RENDER="RealSaS.Corpus370OriginalReactiveStainHeldoutSourceEditRender.v18b"
PASS="REAL_ORIGINAL_HELDOUT_POSE_STAIN_GROUP_EDITS_MEASURED__AUTHOR_DRAWABLE_UNKNOWN"
TOL=.025
VIEWS=(0,2,4,6)
TARGET_INDICES=(0,2,4,6,8,10)


def require(x,msg):
    if not x:raise ValueError(msg)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def rgba_effect(a,b):
    a=np.asarray(a,np.float32);b=np.asarray(b,np.float32)
    require(a.shape==b.shape and a.shape[-1]==4
        and np.isfinite(a).all() and np.isfinite(b).all(),
        "V18B_INDEPENDENT_REAL_RGBA_SOURCE_IMAGE_MALFORMED")
    changed_alpha=np.abs(a[...,3]-b[...,3])
    changed_rgb=np.max(np.abs(
        a[...,:3]*a[...,3:]-b[...,:3]*b[...,3:]),axis=-1)
    return np.maximum(changed_alpha,changed_rgb)>TOL


def effects(ref,group):
    """Pixel-effect comparison, not false same/different drawable labels."""
    ref=np.asarray(ref,bool);group=np.asarray(group,bool)
    require(ref.shape==group.shape and ref.ndim==2,
        "V18B_EDIT_EFFECT_MASK_SHAPE")
    overlap=int((ref&group).sum())
    over=int((group&~ref).sum())
    under=int((ref&~group).sum())
    return {
        "single_face_reference_effect_pixel_instances":int(ref.sum()),
        "source_stain_group_edit_effect_pixel_instances":int(group.sum()),
        "intersection_with_single_face_proxy_pixel_instances":overlap,
        "group_only_edit_collateral_pixel_instances":over,
        "single_face_proxy_not_explained_pixel_instances":under,
        "observable_single_face_reference":bool(ref.any()),
        "candidate_equals_artist_editable_layer":"UNKNOWN"
    }


def expected_arms():
    a=["SOURCE_ORIGINAL","SOUP_FULL","SOUP_NOOP","SOUP_EMPTY"]
    for i in range(len(TARGET_INDICES)):
        a.extend((f"SINGLE_FACE_REMOVE_T{i:02d}",
                  f"SINGLE_FACE_ISOLATE_T{i:02d}"))
        for t in THRESHOLDS:
            key=str(t).replace(".","p")
            a.extend((f"STAIN_REMOVE_{key}_T{i:02d}",
                      f"STAIN_ISOLATE_{key}_T{i:02d}"))
    return a


def evaluate(render_json,edited_npz,stain_report,stain_npz,stain_court,
             v14_job,v14_pixels):
    manifest=load(render_json)
    report=load(stain_report)
    source_court=load(stain_court)
    job=load(v14_job)
    arms=expected_arms()
    require(manifest.get("schema")==RENDER
        and manifest.get("status")==
            "V18B_REAL_ORIGINAL_SOURCE_STAIN_GROUP_EDITS_RENDERED__INDEPENDENT_COURT_REQUIRED"
        and manifest.get("subject_id")==job["subject_id"]==report["subject_id"]
        and manifest.get("original_source_sha256")==job["source"]["sha256"]==
            report["source_sha256"]
        and manifest.get("source_original_v14_job_sha256")==sha(v14_job)
        and manifest.get("original_v14_RGBA_sha256")==sha(v14_pixels)
        and manifest.get("source_reactive_stain_report_sha256")==sha(stain_report)
        and manifest.get("source_reactive_stain_npz_sha256")==sha(stain_npz)
        and manifest.get("independent_source_stain_court_sha256")==sha(stain_court)
        and manifest.get("original_source_heldout_pose_vertices_verified") is True
        and manifest.get("source_rest_triangle_UV_material_authority_reused") is True
        and manifest.get("source_vs_original_v14_RGBA_reference_source_parity_verified") is True
        and manifest.get("new_source_actual_edited_rgba_npz_sha256")==sha(edited_npz)
        and manifest.get("source_edit_arms")==arms
        and manifest.get("heldout_original_authored_pose_ordinal")==2
        and manifest.get("heldout_source_camera_views")==list(VIEWS)
        and manifest.get("artist_original_source_part_names_as_labels") is False
        and manifest.get("source_face_single_triangle_is_artist_gold") is False
        and manifest.get("source_editor_edit_independence_proven") is False
        and manifest.get("full_original_caa_draw_order_qualified") is False
        and manifest.get("graphics_owner_authorized") is False
        and manifest.get("iris_training_authorized") is False
        and manifest.get("tessa_training_authorized") is False
        and manifest.get("materialization_authorized") is False
        and manifest.get("training_authorized") is False
        and source_court.get("status")==
            "ORIGINAL_NATIVE_SOURCE_STAIN_GRAPH_REPRODUCED__ARTIST_EDIT_TRUTH_UNKNOWN"
        and source_court.get("private_stain_report_sha256")==sha(stain_report)
        and source_court.get("private_stain_arrays_sha256")==sha(stain_npz)
        and source_court.get("actual_editor_action_independence_proven") is False,
        "V18B_ORIGINAL_SOURCE_STAIN_TRAINING_AUTHORITY_OR_SOURCE_SHA_FORGERY")
    v11_path=Path(job["v11_private_plan"]["path"])
    require(sha(v11_path)==job["v11_private_plan"]["sha256"]
        and sha(v11_path)==manifest["original_v11_heldout_target_plan_sha256"]
        and sha(job["native_teacher_receipt"]["path"])==
            manifest["original_source_teacher_manifest_sha256"]
        and sha(job["native_teacher"]["path"])==
            manifest["original_source_teacher_vertices_sha256"]
        and sha(job["camera"]["path"])==
            manifest["original_source_camera_sha256"],
        "V18B_SOURCE_ANIMATION_AND_BLIND_EDIT_TARGETS_SHA")
    plan=load(v11_path)
    targets=[plan["query_targets_source_teacher_only"][i]["source_face_teacher_only"]
             for i in TARGET_INDICES]
    require(targets==manifest["source_face_probe_target_ordinals_teacher_only"]
        and len(set(targets))==len(TARGET_INDICES)
        and plan["source_sha256"]==job["source"]["sha256"]
        and plan["real_rgba_consequences_measured_for_heldout_queries"] is False,
        "V18B_UNSEEN_HELDOUT_EDITOR_TARGETS_CHANGED")
    with np.load(stain_npz,allow_pickle=False) as d:
        faces=np.asarray(d["faces"],np.int64)
        policies={
            t:np.asarray(d["labels_"+str(t).replace(".","p")],np.int32)
            for t in THRESHOLDS}
    require(all(len(labels)==len(faces) for labels in policies.values())
        and all(0<=f<len(faces) for f in targets),
        "V18B_REACTIVE_GROUP_ORIGINAL_SOURCE_FACE_ADDRESS")
    with np.load(edited_npz,allow_pickle=False) as d:
        require(d.files==["original_stain_editor_rgba"],
            "V18B_REAL_RGBA_ARRAY_MEMBERS")
        image=np.asarray(d["original_stain_editor_rgba"],np.float32)
    with np.load(v14_pixels,allow_pickle=False) as d:
        require(d.files==["original_native_actuator_rgba"],
            "V18B_PARENT_ORIGINAL_SOURCE_RGBA_ARRAY_MEMBERS")
        original=np.asarray(d["original_native_actuator_rgba"],np.float32)
    original_names,_=arms_for_plan(load(job["selected_source_controls"]["path"]))
    require(image.shape==(4,len(arms),192,192,4)
        and original.shape==(3,4,len(original_names),192,192,4)
        and np.isfinite(image).all()
        and np.min(image[...,3])>=-1e-5
        and np.max(image[...,3])<=1.00001,
        "V18B_REAL_ORIGINAL_SOURCE_EDITOR_RGBA_SHAPE_AND_FINITE")
    origindex=original_names.index("ORIGINAL_BASE")
    idx={x:i for i,x in enumerate(arms)}
    whole=[]
    aggregates={str(t):{
        "group_remove_effect_pixels":0,
        "single_face_proxy_remove_effect_pixels":0,
        "group_remove_collateral_pixels":0,
        "group_remove_missing_single_face_reference_pixels":0,
        "group_isolate_effect_pixels":0,
        "single_face_proxy_isolate_effect_pixels":0,
        "group_isolate_collateral_pixels":0,
        "group_isolate_missing_single_face_reference_pixels":0,
        "observable_single_face_reference_case_count":0,
        "unobservable_single_face_reference_case_count":0,
        "total_heldout_source_edit_cases":0
    } for t in THRESHOLDS}
    for vi,view in enumerate(VIEWS):
        base=image[vi,idx["SOURCE_ORIGINAL"]]
        soup=image[vi,idx["SOUP_FULL"]]
        noop=image[vi,idx["SOUP_NOOP"]]
        empty=image[vi,idx["SOUP_EMPTY"]]
        orig=original[2,vi,origindex]
        require(float(np.percentile(np.max(np.abs(base-orig),axis=-1),99))<=TOL
            and float(np.percentile(np.max(np.abs(base-soup),axis=-1),99))<=TOL
            and not rgba_effect(soup,noop).any()
            and int((empty[...,3]>.025).sum())==0,
            "V18B_NATIVE_ORIGINAL_RGB_SOURCE_SOUP_NOOP_EMPTY_COURT")
        for i,target in enumerate(targets):
            ref_remove=rgba_effect(soup,image[vi,idx[f"SINGLE_FACE_REMOVE_T{i:02d}"]])
            ref_isolate=rgba_effect(empty,image[vi,idx[f"SINGLE_FACE_ISOLATE_T{i:02d}"]])
            for t in THRESHOLDS:
                key=str(t).replace(".","p")
                labels=policies[t]
                assert 0<=labels[target]<=labels.max()
                gm=rgba_effect(soup,image[vi,idx[f"STAIN_REMOVE_{key}_T{i:02d}"]])
                gi=rgba_effect(empty,image[vi,idx[f"STAIN_ISOLATE_{key}_T{i:02d}"]])
                remove=effects(ref_remove,gm)
                isolate=effects(ref_isolate,gi)
                row={
                    "source_camera_view":view,
                    "precommitted_v11_heldout_target_ordinal":TARGET_INDICES[i],
                    "source_target_face_teacher_only":target,
                    "candidate_threshold":t,
                    "group_candidate_face_count":
                        int(np.count_nonzero(labels==labels[target])),
                    "remove":remove,
                    "isolate":isolate,
                    "author_same_drawable":"UNKNOWN",
                    "author_different_drawable":"UNKNOWN",
                    "source_editability_status":
                        "OBSERVED_PROXY_EDIT_RESPONSE" if (ref_remove.any() or ref_isolate.any())
                        else "UNKNOWN_SINGLE_FACE_SOURCE_EDIT_UNOBSERVABLE"
                }
                whole.append(row)
                total=aggregates[str(t)]
                total["group_remove_effect_pixels"]+=int(gm.sum())
                total["single_face_proxy_remove_effect_pixels"]+=int(ref_remove.sum())
                total["group_remove_collateral_pixels"]+=remove[
                    "group_only_edit_collateral_pixel_instances"]
                total["group_remove_missing_single_face_reference_pixels"]+=remove[
                    "single_face_proxy_not_explained_pixel_instances"]
                total["group_isolate_effect_pixels"]+=int(gi.sum())
                total["single_face_proxy_isolate_effect_pixels"]+=int(ref_isolate.sum())
                total["group_isolate_collateral_pixels"]+=isolate[
                    "group_only_edit_collateral_pixel_instances"]
                total["group_isolate_missing_single_face_reference_pixels"]+=isolate[
                    "single_face_proxy_not_explained_pixel_instances"]
                total["observable_single_face_reference_case_count"]+=int(
                    ref_remove.any() or ref_isolate.any())
                total["unobservable_single_face_reference_case_count"]+=int(
                    not ref_remove.any() and not ref_isolate.any())
                total["total_heldout_source_edit_cases"]+=1
    require(len(whole)==len(VIEWS)*len(TARGET_INDICES)*len(THRESHOLDS)
        and all(x["total_heldout_source_edit_cases"]==
                len(VIEWS)*len(TARGET_INDICES)
                for x in aggregates.values()),
        "V18B_REAL_ORIGINAL_EDIT_COLLATERAL_COVERAGE_MISSING")
    return {
        "schema":SCHEMA,
        "status":PASS,
        "original_source_id":job["subject_id"],
        "source_sha256":job["source"]["sha256"],
        "real_v18b_source_editor_manifest_sha256":sha(render_json),
        "real_v18b_source_editor_rgba_npz_sha256":sha(edited_npz),
        "parent_v18a_independent_stain_court_sha256":sha(stain_court),
        "heldout_original_source_authored_pose_ordinal":2,
        "original_camera_views":list(VIEWS),
        "precommitted_heldout_source_target_count":len(targets),
        "frozen_candidate_similarity_thresholds":list(THRESHOLDS),
        "actual_real_blender_source_render_arms_per_camera":len(arms),
        "actual_heldout_source_edit_cases":len(whole),
        "source_edit_consequence_rows_teacher_only":whole,
        "per_candidate_heldout_edit_collateral":aggregates,
        "single_face_proxy_is_not_author_layer":True,
        "observed_source_edit_response_not_artist_owner":True,
        "genuine_independent_edit_collateral_measured":True,
        "artist_same_drawable_positive_pairs":0,
        "artist_different_drawable_negative_pairs":0,
        "graphics_owner_authorized":False,
        "full_alpha_caa_qualified":False,
        "iris_training_authorized":False,
        "tessa_training_authorized":False,
        "materialization_authorized":False,
        "training_authorized":False
    }


def main():
    p=argparse.ArgumentParser()
    for k in ("render_manifest","edited_npz","stain_report","stain_npz",
              "stain_court","v14_job","v14_original_pixels","output"):
        p.add_argument("--"+k.replace("_","-"),required=True,type=Path)
    args=p.parse_args()
    result=evaluate(args.render_manifest,args.edited_npz,args.stain_report,
        args.stain_npz,args.stain_court,args.v14_job,args.v14_original_pixels)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,sort_keys=True,indent=2,
                                      allow_nan=False)+"\n")
    print(json.dumps({
        "status":result["status"],
        "original_source_id":result["original_source_id"],
        "cases":result["actual_heldout_source_edit_cases"],
        "per_threshold":result["per_candidate_heldout_edit_collateral"],
        "artist_labels_minted":0
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
