#!/usr/bin/env python3
"""Independent original-source 3D physical removal canary court.

Runs only on sealed original pose2 renders. The producer physically deletes
triangles (not transparency), and this court recomputes original-source,
full-soup, no-op, empty, single-face and anonymous-group consequences.
No artist grouping or student training authority is produced.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from tools.corpus.court_corpus370_v18b_reactive_stain_editor_edits import (
    TOL,VIEWS,rgba_effect
)
from tools.corpus.corpus370_native_raster_join_plan_v14 import arms_for_plan

RENDER="RealSaS.Corpus370OriginalPhysicalFaceEditCanaryRender.v18e"
SCHEMA="RealSaS.Corpus370IndependentPhysicalFaceEditCanaryCourt.v18e"
PASS="ORIGINAL_SOURCE_PHYSICAL_TRIANGLE_REMOVAL_CANARY_COURTED__OWNER_UNKNOWN"
ARMS=("SOURCE_ORIGINAL","SOUP_FULL","SOUP_NOOP","SOUP_EMPTY",
      "SINGLE_FACE_REMOVE_T00","SINGLE_FACE_ISOLATE_T00",
      "STAIN_REMOVE_0p1_T00","STAIN_ISOLATE_0p1_T00")

def require(ok,code):
    if not ok: raise ValueError(code)

def sha(path):
    with Path(path).open("rb") as p:
        return hashlib.file_digest(p,"sha256").hexdigest()

def load(path):
    return json.loads(Path(path).read_text())

def image_delta_metrics(a,b):
    x=np.asarray(a,np.float32);y=np.asarray(b,np.float32)
    require(x.shape==y.shape and x.shape==(192,192,4)
            and np.isfinite(x).all() and np.isfinite(y).all(),
            "V18E_COURT_RGBA_ARRAY")
    raw=np.max(np.abs(x-y),axis=-1)
    effect=rgba_effect(x,y)
    return {
        "raw_max_channel_p99":float(np.percentile(raw,99)),
        "raw_max_channel_over_tol_pixels":int(np.count_nonzero(raw>TOL)),
        "alpha_abs_over_tol_pixels":int(np.count_nonzero(np.abs(x[...,3]-y[...,3])>TOL)),
        "source_premultiplied_rgba_effect_pixels":int(effect.sum()),
        "source_effect_exists":bool(effect.any()),
    }

def evaluate(manifest_path,edited_path,stain_path,stain_npz_path,stain_court_path,
             v14_job_path,v14_pixels_path):
    manifest=load(manifest_path);v14=load(v14_job_path)
    stain=load(stain_path);stain_court=load(stain_court_path)
    require(manifest.get("schema")==RENDER
        and manifest.get("status")==
            "V18E_REAL_ORIGINAL_PHYSICAL_FACE_EDITS_RENDERED__INDEPENDENT_COURT_REQUIRED"
        and manifest.get("source_edit_arms")==list(ARMS)
        and manifest.get("actual_render_arms_per_camera")==len(ARMS)
        and manifest.get("physical_face_visibility_operator")==
            "ACTUAL_TRIANGLE_REMOVAL_AND_OBJECT_RENDER_HIDE"
        and manifest.get("physical_removal_not_author_drawable_layer") is True
        and manifest.get("precommitted_v11_target_query_indices")==[0]
        and manifest.get("frozen_tested_stain_similarity_thresholds")==[.1]
        and manifest.get("original_full_stain_similarity_thresholds")==[.045,.1,.18]
        and manifest.get("source_original_v14_job_sha256")==sha(v14_job_path)
        and manifest.get("original_v14_RGBA_sha256")==sha(v14_pixels_path)
        and manifest.get("source_reactive_stain_report_sha256")==sha(stain_path)
        and manifest.get("source_reactive_stain_npz_sha256")==sha(stain_npz_path)
        and manifest.get("independent_source_stain_court_sha256")==sha(stain_court_path)
        and manifest.get("new_source_actual_edited_rgba_npz_sha256")==sha(edited_path)
        and v14.get("schema")=="RealSaS.Corpus370SourceNativeMechanicalRasterJointJob.v14"
        and manifest.get("subject_id")==v14.get("subject_id")==stain.get("subject_id")
        and manifest.get("original_source_sha256")==v14["source"]["sha256"]
          ==stain.get("source_sha256")
        and stain_court.get("status")==
            "ORIGINAL_NATIVE_SOURCE_STAIN_GRAPH_REPRODUCED__ARTIST_EDIT_TRUTH_UNKNOWN"
        and manifest.get("original_source_heldout_pose_vertices_verified") is True
        and manifest.get("source_rest_triangle_UV_material_authority_reused") is True
        and manifest.get("artist_original_source_part_names_as_labels") is False
        and manifest.get("source_face_single_triangle_is_artist_gold") is False
        and manifest.get("source_editor_edit_independence_proven") is False
        and manifest.get("full_original_caa_draw_order_qualified") is False
        and all(manifest.get(key) is False for key in (
            "graphics_owner_authorized","iris_training_authorized",
            "tessa_training_authorized","materialization_authorized",
            "training_authorized"))
        and manifest.get("heldout_original_authored_pose_ordinal")==2
        and manifest.get("heldout_source_camera_views")==list(VIEWS),
        "V18E_COURT_SOURCE_SHA_OR_LABEL_AUTHORITY")
    source_face_plan=load(Path(v14["v11_private_plan"]["path"]))
    require(sha(Path(v14["v11_private_plan"]["path"]))==
            manifest["original_v11_heldout_target_plan_sha256"]
            and sha(Path(v14["v11_private_plan"]["path"]))==
            v14["v11_private_plan"]["sha256"],
            "V18E_SOURCE_V11_HELDOUT_PLAN_SHA")
    target=source_face_plan["query_targets_source_teacher_only"][0]["source_face_teacher_only"]
    require([target]==manifest["source_face_probe_target_ordinals_teacher_only"],
            "V18E_SOURCE_BLIND_TARGET_CHANGED")
    with np.load(edited_path,allow_pickle=False) as f:
        require(f.files==["original_stain_editor_rgba"],
                "V18E_COURT_RGBA_MEMBER")
        images=np.asarray(f["original_stain_editor_rgba"],np.float32)
    with np.load(v14_pixels_path,allow_pickle=False) as f:
        require(f.files==["original_native_actuator_rgba"],
                "V18E_COURT_V14_RGBA_MEMBER")
        parent=np.asarray(f["original_native_actuator_rgba"],np.float32)
    with np.load(Path(v14["rest_geometry"]["path"]),allow_pickle=False) as f:
        faces=np.asarray(f["faces"],np.int64)
    with np.load(stain_npz_path,allow_pickle=False) as f:
        stain_faces=np.asarray(f["faces"],np.int64)
        labels=np.asarray(f["labels_0p1"],np.int32)
    require(np.array_equal(faces,stain_faces)
        and labels.shape==(len(faces),)
        and len(labels)>0 and np.all(labels>=0),
        "V18E_COURT_PARENT_STAIN_SOURCE_FACE_ARRAY")
    group_size=int(np.count_nonzero(labels==labels[target]))
    require(images.shape==(4,len(ARMS),192,192,4)
            and parent.ndim==6 and parent.shape[0:2]==(3,4)
            and parent.shape[3:]==(192,192,4)
            and np.isfinite(images).all() and np.isfinite(parent).all(),
            "V18E_COURT_IMAGE_SHAPE_AND_FINITE")
    original_arms,_=arms_for_plan(load(v14["selected_source_controls"]["path"]))
    origindex=original_arms.index("ORIGINAL_BASE")
    n=len(faces)
    # Every physically removed count is independently checked against the
    # sealed V18a source-face partition, not only the producer metadata.
    required_counts=[n,n,n,0,n-1,1,n-group_size,group_size]
    claimed=manifest["actual_visible_original_source_triangles_per_arm_view"]
    require(len(claimed)==4
            and all(len(row)==len(ARMS) for row in claimed),
            "V18E_COURT_VISIBILITY_CENSUS_SHAPE")
    # No source artist owner is known. Compare only operational render behavior.
    result=[]
    total_observable=0
    for i,view in enumerate(VIEWS):
        originals=images[i,0];soup=images[i,1]
        no_op=images[i,2];empty=images[i,3]
        original_delta=image_delta_metrics(originals,parent[2,i,origindex])
        parity=image_delta_metrics(originals,soup)
        noop=image_delta_metrics(soup,no_op)
        require(original_delta["raw_max_channel_p99"]<=TOL
            and parity["raw_max_channel_p99"]<=TOL
            and not noop["source_effect_exists"]
            and int(np.count_nonzero(empty[...,3]>TOL))==0,
            "V18E_COURT_ORIGINAL_SOURCE_PHYSICAL_EMPTY_RENDER_PARITY")
        for idx,expected in enumerate(required_counts):
            if expected is not None:
                require(claimed[i][idx]==expected,
                        "V18E_COURT_PHYSICAL_REMOVAL_FACE_COUNT")
        one_rm=image_delta_metrics(soup,images[i,4])
        one_iso=image_delta_metrics(empty,images[i,5])
        grp_rm=image_delta_metrics(soup,images[i,6])
        grp_iso=image_delta_metrics(empty,images[i,7])
        observed=one_rm["source_effect_exists"] or one_iso["source_effect_exists"]
        total_observable+=int(observed)
        result.append({
            "original_camera_view":view,
            "source_vs_v14_original":original_delta,
            "source_vs_original_soup":parity,
            "soup_noop":noop,
            "physically_empty_alpha_over_tol_pixels":int(np.count_nonzero(empty[...,3]>TOL)),
            "single_source_face_remove":one_rm,
            "single_source_face_isolate":one_iso,
            "anonymous_stain_group_remove":grp_rm,
            "anonymous_stain_group_isolate":grp_iso,
            "single_source_face_reference_observable":bool(observed),
            "artist_same_drawable":"UNKNOWN",
            "artist_different_drawable":"UNKNOWN",
        })
    require(total_observable>0,
            "V18E_COURT_PRECOMMITTED_EDIT_TARGET_NEVER_OBSERVABLE")
    return {
        "schema":SCHEMA,"status":PASS,"original_source_id_private":v14["subject_id"],
        "parent_v14_job_sha256":sha(v14_job_path),
        "parent_v14_rgba_sha256":sha(v14_pixels_path),
        "parent_v18a_stain_report_sha256":sha(stain_path),
        "parent_v18a_stain_npz_sha256":sha(stain_npz_path),
        "parent_v18a_stain_court_sha256":sha(stain_court_path),
        "independently_reopened_v18e_manifest_sha256":sha(manifest_path),
        "independently_reopened_v18e_rgba_sha256":sha(edited_path),
        "physical_edit_arms_per_original_camera":len(ARMS),
        "original_camera_views":list(VIEWS),
        "observable_blind_source_face_references":total_observable,
        "per_view":result,
        "anonymous_group_is_artist_drawable":"UNKNOWN",
        "source_physical_edit_render_parity_qualified":True,
        "graphics_owner_authorized":False,
        "iris_training_authorized":False,
        "tessa_training_authorized":False,
        "materialization_authorized":False,
        "training_authorized":False,
    }

def main():
    p=argparse.ArgumentParser()
    for key in ("render_manifest","edited_npz","stain_report","stain_npz","stain_court",
                "v14_job","v14_original_pixels","output"):
        p.add_argument("--"+key.replace("_","-"),type=Path,required=True)
    a=p.parse_args()
    output=evaluate(a.render_manifest,a.edited_npz,a.stain_report,
                    a.stain_npz,a.stain_court,a.v14_job,a.v14_original_pixels)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(output,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({
        "status":output["status"],
        "observed_real_single_face_edits":output["observable_blind_source_face_references"],
        "views":len(output["original_camera_views"]),
        "physical_edit_arms":output["physical_edit_arms_per_original_camera"],
        "per_view":[{
            "view":v["original_camera_view"],
            "single_remove_pixels":v["single_source_face_remove"]["source_premultiplied_rgba_effect_pixels"],
            "single_isolate_pixels":v["single_source_face_isolate"]["source_premultiplied_rgba_effect_pixels"],
            "group_remove_pixels":v["anonymous_stain_group_remove"]["source_premultiplied_rgba_effect_pixels"],
            "group_isolate_pixels":v["anonymous_stain_group_isolate"]["source_premultiplied_rgba_effect_pixels"],
            "empty_alpha_pixels":v["physically_empty_alpha_over_tol_pixels"],
        } for v in output["per_view"]],
        "training_authorized":False
    },sort_keys=True),flush=True)

if __name__=="__main__":
    main()
