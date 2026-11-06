#!/usr/bin/env python3
"""V18l independently recompute exact per-index Cycles source subset compositing.

It re-opens physically edited original Blender source per-index frames and
preexisting V18h source court. It cannot prove one sample ray is the same
camera point across three separate scenes just from identical indices: it
measures that assumption through per-sample RGB/alpha consistency instead.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA="RealSaS.Corpus370IndependentCyclesCommonSampleSourceCourt.v18l"
SOURCE="RealSaS.Corpus370OriginalCyclesCommonSampleComplement.v18l"
STATUS="ORIGINAL_CYCLES_SUBSET_COMMON_SAMPLE_LAYER_REMEASURED__SOURCE_ARTIST_UNKNOWN"
SOURCE_STATUS="V18L_ORIGINAL_CYCLES_COMMON_SOURCE_SAMPLE_RENDERED__INDEPENDENT_COURT_REQUIRED"
V18H="RealSaS.Corpus370IndependentEmissiveComplementCourt.v18h"
V18H_PASS="ORIGINAL_SOURCE_EMISSIVE_COMPOSITING_CONTRAST_COURTED__NO_DRAWABLE_AUTHORITY"
INDICES=(0,4,8,12,16,20,24,28)
VIEWS=(0,2,4,6)
ARMS=("FULL","ISOLATE","REMOVE")
TOL=.025


def require(ok,error):
    if not ok:raise ValueError(error)


def sha(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def premul(images):
    x=np.asarray(images,dtype=np.float64)
    require(x.shape[-1]==4 and np.isfinite(x).all()
        and np.min(x[...,3])>=-1e-5
        and np.max(x[...,3])<=1.+1e-5,
        "V18L_INDEPENDENT_SOURCE_RGBA_BAD")
    a=np.clip(x[...,3:4],0.,1.)
    return np.concatenate((x[...,:3]*a,a),axis=-1)


def over(top,bottom):
    return top+bottom*(1.-top[...,3:4])


def per_source_view(frames):
    """frames = 8 indexed real-Cycles samples x FULL/G/R x H x W x RGBA.

    The full-render answer at eight *identical claimed sample indices* is
    measured, NOT assumed. A per-sample best-of-2 oracle is only diagnostic;
    no per-pixel art layer ordering or model targets are minted.
    """
    require(frames.shape==(len(INDICES),len(ARMS),192,192,4),
        "V18L_ORIGINAL_CAMERA_SAMPLE_RENDER_SHAPE")
    p=premul(frames)
    full=p[:,0]
    g=p[:,1]
    r=p[:,2]
    sample_a=over(g,r)
    sample_b=over(r,g)
    sample_best=np.minimum(
        np.max(np.abs(full-sample_a),axis=-1),
        np.max(np.abs(full-sample_b),axis=-1))
    sample_alpha=np.abs(full[...,3]-(g[...,3]+r[...,3]-g[...,3]*r[...,3]))
    # Per-sample best-of-order from matched subset outputs.
    # This is an ORACLE diagnostic, not a precommitted drawable sorting plan.
    sample_a_mean=np.mean(sample_a,axis=0)
    sample_b_mean=np.mean(sample_b,axis=0)
    full_mean=np.mean(full,axis=0)
    g_mean=np.mean(g,axis=0)
    r_mean=np.mean(r,axis=0)
    post_a=over(g_mean,r_mean)
    post_b=over(r_mean,g_mean)
    joint_best=np.minimum(
        np.max(np.abs(full_mean-sample_a_mean),axis=-1),
        np.max(np.abs(full_mean-sample_b_mean),axis=-1))
    post_best=np.minimum(
        np.max(np.abs(full_mean-post_a),axis=-1),
        np.max(np.abs(full_mean-post_b),axis=-1))
    sample_error=np.max(np.abs(full-sample_a),axis=-1)
    # Exact source renderer outputs, same subsample index but not
    # guaranteed same ray position until sample-oracle residual checks.
    matched_alpha_fail=sample_alpha>TOL
    avg_alpha=full_mean[...,3]
    # The same source's per-index full mean differs from 32spp V18h
    # by *both* sample subset and filter policy: do not judge it as gold.
    return {
      "rendered_pixel_sample_instances":int(sample_best.size),
      "same_index_sample_best_order_failed_pixel_instances":
        int(np.count_nonzero(sample_best>TOL)),
      "same_index_sample_alpha_invariant_failed_pixel_instances":
        int(np.count_nonzero(matched_alpha_fail)),
      "same_index_sample_best_order_error_max":
        float(sample_best.max()),
      "same_index_sample_best_order_error_mean":
        float(sample_best.mean()),
      "source_rendered_nonempty_full_pixel_instances":
        int(np.count_nonzero(full[...,3]>TOL)),
      "source_raw_1spp_alpha_nonbinary_pixel_instances":
        int(np.count_nonzero((full[...,3]>TOL)&(full[...,3]<1-TOL))),
      "source_8_sample_full_mean_nonempty_pixels":
        int(np.count_nonzero(avg_alpha>TOL)),
      "source_8_sample_premultiplied_joint_pre_filter_vs_postfilter_best_order_pixel_failures":
        int(np.count_nonzero(post_best>TOL)),
      "source_8_sample_pre_filter_best_fixed_order_pixel_failures":
        int(np.count_nonzero(joint_best>TOL)),
      "source_8_sample_postfilter_best_order_error_mean":float(post_best.mean()),
      "source_8_sample_samplefirst_best_fixed_order_error_mean":
        float(joint_best.mean()),
      "source_8_sample_pixelwise_oracle_not_drawable_order":True,
      "_full_mean":full_mean,
      "_joint_best":joint_best,
      "_post_best":post_best,
      "_full_sample_error":sample_error,
    }


def evaluate(source_manifest,source_npz,v18h_court,
             expected_v18h_sha):
    require(isinstance(expected_v18h_sha,str) and len(expected_v18h_sha)==64
        and sha(v18h_court)==expected_v18h_sha,
        "V18L_EXACT_PARENT_V18H_SOURCE_COURT_SHA")
    m=load(source_manifest)
    c=load(v18h_court)
    require(m.get("schema")==SOURCE and m.get("status")==SOURCE_STATUS
        and c.get("schema")==V18H and c.get("status")==V18H_PASS
        and m.get("original_source_sha256")==c.get("original_source_sha256")
        and m.get("subject_id")==c.get("source_subject_id_private")
        and m.get("original_source_v18l_parent_v18h_run")=="42120479356-1"
        and m.get("source_original_v14_job_sha256")
        and m.get("original_source_teacher_vertices_sha256")
        and m.get("original_source_camera_sha256")
        and m.get("parent_v18f_independent_court_sha256")==
            c.get("exact_v18f_original_independent_court_sha256")
        and m.get("parent_v18g_independent_compositing_sha256")==
            c.get("exact_v18g_original_independent_court_sha256")
        and m.get("original_source_cycles_sample_indices")==list(INDICES)
        and m.get("heldout_source_camera_views")==list(VIEWS)
        and m.get("frozen_tested_stain_similarity_thresholds")==[.1]
        and m.get("source_edit_arms")==list(ARMS)
        and m.get("actual_render_arms_per_camera")==len(ARMS)
        and m.get("source_cycles_max_samples")==32
        and m.get("source_cycle_subset_length")==1
        and m.get("source_cycle_sampling_seed")==0
        and m.get("source_cycles_adaptive_sampling") is False
        and m.get("source_cycles_pixel_filter")=="BOX"
        and m.get("source_cycles_pixel_filter_width")==.01
        and m.get("source_cycle_sample_pattern_equals_original_v18h_32spp_film") is False
        and m.get("source_material_policy")==
            "FIXED_TWO_COLOR_OPAQUE_EMISSION_NO_MATERIAL_SHADING"
        and m.get("physical_face_visibility_operator")==
            "ACTUAL_TRIANGLE_REMOVAL_AND_OBJECT_RENDER_HIDE"
        and m.get("v18l_source_nontrivial_group_face_count",0)>0
        and m.get("v18l_source_nontrivial_group_face_count")<
            m.get("v18l_original_source_face_count",0)
        and m.get("new_source_actual_edited_rgba_npz_sha256")==sha(source_npz)
        and m.get("no_artist_layer_labels_minted") is True
        and all(m.get(k) is False for k in (
          "graphics_owner_authorized","iris_training_authorized",
          "tessa_training_authorized","materialization_authorized",
          "training_authorized"))
        and all(c.get(k) is False for k in (
          "graphics_owner_authorized","iris_training_authorized",
          "tessa_training_authorized","materialization_authorized",
          "training_authorized")),
        "V18L_SOURCE_NOT_EXACT_ORIGINAL_PARENT_OR_AUTHORITY")
    with np.load(source_npz,allow_pickle=False) as f:
        require(f.files==["common_cycles_sample_rgba"],
                "V18L_SOURCE_EXACT_CYCLES_SUBSET_ARRAY")
        rgba=np.asarray(f["common_cycles_sample_rgba"],np.float32)
    require(rgba.shape==(4,8,3,192,192,4)
        and np.isfinite(rgba).all()
        and rgba[...,3].min()>=-1e-5
        and rgba[...,3].max()<=1.+1e-5,
        "V18L_ORIGINAL_SOURCE_SUBSET_RGBA_VALIDITY")
    per_view=[]
    full_errors=[]
    for i,view in enumerate(VIEWS):
        report=per_source_view(rgba[i])
        full_errors.append(report.pop("_full_sample_error"))
        report.pop("_full_mean")
        report.pop("_joint_best")
        report.pop("_post_best")
        report["source_original_view"]=int(view)
        report["source_camera_partition"]="LOCKED_EVALUATION" if view in (4,6) else "EXPLORATORY"
        per_view.append(report)
    require(len(per_view)==4 and
            sum(x["rendered_pixel_sample_instances"] for x in per_view)
            ==4*8*192*192,
        "V18L_INDEPENDENT_FULL_SOURCE_SAMPLE_CENSUS")
    return {
      "schema":SCHEMA,"status":STATUS,
      "subject_id_private":m["subject_id"],
      "original_source_sha256":m["original_source_sha256"],
      "source_render_manifest_sha256":sha(source_manifest),
      "source_render_npz_sha256":sha(source_npz),
      "exact_original_v18h_independent_court_sha256":sha(v18h_court),
      "original_source_renderer_policy":{
        "cycles_max_samples":32,"selected_sample_offsets":list(INDICES),
        "subset_sample_count":1,"camera_views":list(VIEWS),
        "pixel_filter":"BOX","filter_width":0.01,
        "sample_alignment_inferred_only_from_measured_composition":True,
        "original_v18h_32spp_filter_equivalence":"NOT_CLAIMED",
      },
      "by_original_source_camera":per_view,
      "same_index_sample_failed_instances_total":sum(
          v["same_index_sample_best_order_failed_pixel_instances"] for v in per_view),
      "same_index_sample_alpha_failures_total":sum(
          v["same_index_sample_alpha_invariant_failed_pixel_instances"] for v in per_view),
      "matched_sample_invariant_qualified":all(
          v["same_index_sample_best_order_failed_pixel_instances"]==0 for v in per_view),
      "not_a_cycles_32_sample_full_frame_parity_test":True,
      "artist_drawable_identity":"UNKNOWN",
      "source_emission_diagnostic_not_original_art_material":True,
      "same_drawable_training_pairs":0,
      "different_drawable_training_pairs":0,
      "graphics_owner_authorized":False,
      "iris_training_authorized":False,
      "tessa_training_authorized":False,
      "materialization_authorized":False,
      "training_authorized":False,
    }


def main():
    p=argparse.ArgumentParser()
    for name in ("manifest","rgba","v18h_court","output"):
        p.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    p.add_argument("--expected-v18h-court-sha",required=True)
    a=p.parse_args()
    result=evaluate(a.manifest,a.rgba,a.v18h_court,
                    a.expected_v18h_court_sha)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({
        "status":result["status"],
        "same_index_sample_failed_instances_total":
            result["same_index_sample_failed_instances_total"],
        "same_index_sample_alpha_failures_total":
            result["same_index_sample_alpha_failures_total"],
        "matched_sample_invariant_qualified":
            result["matched_sample_invariant_qualified"],
        "per_view":[{k:v for k,v in row.items()
            if k not in ()} for row in result["by_original_source_camera"]],
        "training_authorized":False,
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
