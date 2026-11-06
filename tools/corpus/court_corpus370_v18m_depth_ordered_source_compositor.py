#!/usr/bin/env python3
"""V18m: independent oracle-free source-local G/R depth-sort composer court.

Only the isolated G/R color layers and G/R depth layers enter the predictor.
Original FULL pixels enter the independently evaluated court *after*
predictions are frozen. No image-fit order, artist layer or model-target mint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from tools.corpus.court_corpus370_v18l_cycles_common_sample_source import (
    premul,over,per_source_view,
    SCHEMA as V18L_COURT, STATUS as V18L_PASS,
    SOURCE as V18L_MANIFEST, SOURCE_STATUS as V18L_MANIFEST_STATUS,
)
from tools.corpus.court_corpus370_v18l_source_sample_diversity import (
    compare as compare_baseline,
)

SCHEMA="RealSaS.Corpus370IndependentDepthOrderedCyclesSourceCourt.v18m"
STATUS="ORIGINAL_CYCLES_SOURCE_LOCAL_DEPTH_ORDER_COURTED__ARTIST_DRAWABLE_UNKNOWN"
V18M_MANIFEST="RealSaS.Corpus370OriginalCyclesSampleDepthEvidence.v18m"
V18M_STATUS="V18M_ORIGINAL_CYCLES_SOURCE_DEPTH_REPLAYED__INDEPENDENT_COURT_REQUIRED"
TOL=.025
DEPTH_TIE_EPS=.0002
CAMERAS=(0,2,4,6)
INDICES=(0,4,8,12,16,20,24,28)
NO_TRAINING=("graphics_owner_authorized","iris_training_authorized",
             "tessa_training_authorized","materialization_authorized",
             "training_authorized")


def require(ok,msg):
    if not ok:
        raise ValueError(msg)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def predict_source_only(layer_rgba,layer_viewz_rgba,depth_eps=DEPTH_TIE_EPS):
    """Predict without any FULL target or source oracle RGB dependency.

    Layout: 8 Cycles subset indices, two physically edited isolated source
    layers G/R, original 192×192 pixels and RGBA. Depth material is monotonic
    normalized Cycles camera View Z; lower R channel means closer surface.
    """
    colors=np.asarray(layer_rgba,dtype=np.float64)
    depth=np.asarray(layer_viewz_rgba,dtype=np.float64)
    require(colors.shape==(8,2,192,192,4)
        and depth.shape==colors.shape
        and np.isfinite(colors).all() and np.isfinite(depth).all()
        and 0<depth_eps<=.001,
        "V18M_PREDICTOR_SOURCE_RGB_AND_DEPTH_SHAPE")
    cp=premul(colors)
    da=np.clip(depth[...,3],0.,1.)
    ca=np.clip(colors[...,3],0.,1.)
    max_alpha_diff=float(np.max(np.abs(ca-da)))
    # If the depth measurement's G/R coverage does not equal its frozen
    # Cycles color arm, source sample indexing has drifted. Fail before
    # trying to select a source layer by color.
    require(max_alpha_diff<=1e-6,
        "V18M_PREDICTOR_DIFFERENT_CYCLES_RAY_SAMPLE_OR_COVERAGE")
    require(np.all((ca<=1e-6)|(ca>=1-1e-6)),
        "V18M_PREDICTOR_AMBIGUOUS_NONBINARY_1SPP_ALPHA")
    covered=ca>.5
    g=covered[:,0]
    r=covered[:,1]
    z_g=depth[:,0,:,:,0]
    z_r=depth[:,1,:,:,0]
    # Only measured source hits are allowed to assert depth. No fake
    # comparison against non-hit background RGB or zeroed alpha surfaces.
    valid_depth=(np.where(g,(z_g>=.019)&(z_g<=.981),True)
        &np.where(r,(z_r>=.019)&(z_r<=.981),True))
    both=g&r
    ties=both&(np.abs(z_g-z_r)<=depth_eps)
    unknown=(~valid_depth)|ties
    choose_g=~unknown&g&(~r|(z_g+depth_eps<z_r))
    choose_r=~unknown&r&(~g|(z_r+depth_eps<z_g))
    require(np.all(~(choose_g&choose_r)),
        "V18M_PREDICTOR_BOTH_SOURCE_LAYERS_FRONTMOST")
    # All layers are opaque fixed-emission source diagnostic materials.
    # Empty source rays are transparent, and tied rays abstain.
    pred=np.where(choose_g[...,None],cp[:,0],
        np.where(choose_r[...,None],cp[:,1],0.))
    require(pred.shape==(8,192,192,4) and np.isfinite(pred).all(),
        "V18M_PREDICTOR_OUTPUT_FINITE")
    return {
        "predicted_full_premultiplied_samples":pred,
        "sample_valid_mask":~unknown,
        "unknown_depth_tie_mask":ties,
        "invalid_depth_mask":~valid_depth,
        "predicted_g_frontmost_mask":choose_g,
        "predicted_r_frontmost_mask":choose_r,
        "both_physically_present_mask":both,
        "max_exact_v18l_isolate_vs_v18m_depth_coverage_diff":max_alpha_diff,
    }


def measure_view(v18l_frames,depth_frames):
    require(np.asarray(v18l_frames).shape==(8,3,192,192,4)
        and np.asarray(depth_frames).shape==(8,2,192,192,4),
        "V18M_SOURCE_VIEW_SAMPLE_CAMERA_SHAPE")
    # The predictor receives only the original physically isolated color
    # layers, and original independently depth-shaded isolated layers.
    p=predict_source_only(v18l_frames[:,1:],depth_frames)
    full=premul(v18l_frames[:,0])
    predicted=p["predicted_full_premultiplied_samples"]
    valid=p["sample_valid_mask"]
    err=np.max(np.abs(predicted-full),axis=-1)
    qualified=int(np.count_nonzero(valid))
    require(qualified>0,"V18M_ALL_SOURCE_RAYS_UNQUALIFIED")
    complete=np.all(valid,axis=0)
    pred_mean=np.mean(predicted,axis=0)
    true_mean=np.mean(full,axis=0)
    pixel_err=np.max(np.abs(pred_mean-true_mean),axis=-1)
    baseline=compare_baseline(v18l_frames)
    fixed=per_source_view(v18l_frames)
    color=premul(v18l_frames[:,1:])
    ga=np.mean(color[:,0],axis=0)
    ra=np.mean(color[:,1],axis=0)
    fixed_a=np.mean(over(color[:,0],color[:,1]),axis=0)
    fixed_b=np.mean(over(color[:,1],color[:,0]),axis=0)
    baseline_fixed=np.minimum(
        np.max(np.abs(true_mean-fixed_a),axis=-1),
        np.max(np.abs(true_mean-fixed_b),axis=-1))
    baseline_post=np.minimum(
        np.max(np.abs(true_mean-over(ga,ra)),axis=-1),
        np.max(np.abs(true_mean-over(ra,ga)),axis=-1))
    require(int(np.count_nonzero(baseline_fixed>TOL))==
          fixed["source_8_sample_pre_filter_best_fixed_order_pixel_failures"]
        and int(np.count_nonzero(baseline_post>TOL))==
          fixed["source_8_sample_premultiplied_joint_pre_filter_vs_postfilter_best_order_pixel_failures"]
        and baseline["source_8_subset_prefilter_order_fail_pixels"]==
          int(np.count_nonzero(baseline_fixed>TOL)),
        "V18M_ORIGINAL_SOURCE_BASELINE_FIXED_ORDER_CENSUS_CHANGED")
    source_fail=(baseline_fixed>TOL)&complete
    source_post_fail=(baseline_post>TOL)&complete
    source_pred_fail=(pixel_err>TOL)&complete
    delta=pixel_err-baseline_fixed
    return {
       "original_source_pixel_sample_instances":int(valid.size),
       "original_source_qualified_pixel_sample_instances":qualified,
       "original_source_invalid_depth_samples":int(
           np.count_nonzero(p["invalid_depth_mask"])),
       "original_source_depth_tie_abstained_samples":int(
           np.count_nonzero(p["unknown_depth_tie_mask"])),
       "original_source_both_layer_present_samples":int(
           np.count_nonzero(p["both_physically_present_mask"])),
       "original_source_prediction_g_frontmost_samples":int(
           np.count_nonzero(p["predicted_g_frontmost_mask"])),
       "original_source_prediction_r_frontmost_samples":int(
           np.count_nonzero(p["predicted_r_frontmost_mask"])),
       "original_source_sample_color_failures_on_qualified":int(
           np.count_nonzero(valid&(err>TOL))),
       "original_source_sample_max_rgba_error_on_qualified":float(
           np.max(err[valid])),
       "source_all_8_sample_qualified_pixels":int(np.count_nonzero(complete)),
       "source_pixels_unknown_any_sample":int(np.count_nonzero(~complete)),
       "original_source_oracle_free_depth_order_predicted_failures":int(
           np.count_nonzero(source_pred_fail)),
       "original_source_oracle_free_depth_order_predicted_mean_error":float(
           np.mean(pixel_err[complete])) if np.any(complete) else None,
       "source_original_blender_postfilter_failures_full_domain":int(
           np.count_nonzero(baseline_post>TOL)),
       "source_original_blender_fixed_samplefirst_failures_full_domain":int(
           np.count_nonzero(baseline_fixed>TOL)),
       "source_blender_postfilter_failures_in_depth_qualified_domain":int(
           np.count_nonzero(source_post_fail)),
       "source_blender_fixed_samplefirst_failures_in_depth_qualified_domain":int(
           np.count_nonzero(source_fail)),
       "source_fixed_samplefirst_failures_recovered_by_depth_sort":int(
           np.count_nonzero(source_fail&~source_pred_fail)),
       "source_depth_predictor_regresses_previously_fixed_qualified_pixels":int(
           np.count_nonzero(~source_fail&source_pred_fail&complete)),
       "source_depth_prediction_improves_over_fixed_order_qualified_pixels":int(
           np.count_nonzero(complete&(delta<-1e-12))),
       "source_depth_prediction_worse_than_fixed_order_qualified_pixels":int(
           np.count_nonzero(complete&(delta>1e-12))),
       "source_depth_prediction_matches_fixed_order_qualified_pixels":int(
           np.count_nonzero(complete&(np.abs(delta)<=1e-12))),
       "full_target_input_used_to_select_frontmost_surface":False,
       "depth_evidence_is_teacher_private_and_no_drawable_labels":True,
       "source_sample_alignment_is_measured_not_assumed":True,
       "original_cycles_32spp_paint_fidelity_not_claimed":True,
       "source_subset_color_depth_alpha_max_diff":
            p["max_exact_v18l_isolate_vs_v18m_depth_coverage_diff"],
    }


def audit(v18m_manifest,v18m_npz,v18l_manifest,v18l_npz,v18l_court,
          expected_v18l_court_sha):
    require(len(expected_v18l_court_sha)==64 and
        sha(v18l_court)==expected_v18l_court_sha,
        "V18M_INDEPENDENT_ORIGINAL_V18L_COURT_SHA_CHANGED")
    m=load(v18m_manifest)
    lm=load(v18l_manifest)
    lc=load(v18l_court)
    require(m.get("schema")==V18M_MANIFEST and m.get("status")==V18M_STATUS
        and lm.get("schema")==V18L_MANIFEST and
            lm.get("status")==V18L_MANIFEST_STATUS
        and lc.get("schema")==V18L_COURT and lc.get("status")==V18L_PASS
        and lc.get("matched_sample_invariant_qualified") is True
        and lc.get("source_render_manifest_sha256")==sha(v18l_manifest)
        and lc.get("source_render_npz_sha256")==sha(v18l_npz)
        and m.get("source_independent_v18l_court_sha256")==sha(v18l_court)
        and m.get("source_independent_v18l_render_manifest_sha256")==sha(v18l_manifest)
        and m.get("source_independent_v18l_raw_rgba_sha256")==sha(v18l_npz)
        and m.get("new_source_actual_edited_rgba_npz_sha256")==sha(v18m_npz)
        and m.get("original_source_sha256")==lm.get("original_source_sha256")
        and m.get("source_original_v14_job_sha256")==lm.get("source_original_v14_job_sha256")
        and m.get("source_reactive_stain_npz_sha256")==lm.get("source_reactive_stain_npz_sha256")
        and m.get("source_original_camera_sha256")==lm.get("source_original_camera_sha256")
        and m.get("subject_id")==lm.get("subject_id")==lc.get("subject_id_private")
        and m.get("original_source_cycles_sample_indices")==list(INDICES)
        and m.get("heldout_source_camera_views")==list(CAMERAS)
        and m.get("source_cycle_sampling_seed")==0
        and m.get("source_cycles_max_samples")==32
        and m.get("source_cycle_subset_length")==1
        and m.get("source_cycles_pixel_filter")=="BOX"
        and m.get("source_cycles_pixel_filter_width")==.01
        and m.get("source_cycle_depth_material")==
            "CAMERA_DATA_VIEW_Z_DEPTH_TO_LINEAR_EMISSION_R"
        and m.get("source_depth_normalization_to_min")==.02
        and m.get("source_depth_normalization_to_max")==.98
        and m.get("source_face_depth_evidence_is_teacher_private_not_art_layer_label") is True
        and m.get("source_edit_arms")==["GROUP_DEPTH","REMAINDER_DEPTH"]
        and m.get("original_source_v18m_parent_v18l_run")=="42131579613-1"
        and len(m.get("source_depth_per_view_source_geometry_only_normalized_ranges",[]))==4
        and all(len(v)==2 and v[1]>v[0]>0
            for v in m["source_depth_per_view_source_geometry_only_normalized_ranges"])
        and all(m.get(k) is False for k in NO_TRAINING)
        and all(lm.get(k) is False for k in NO_TRAINING)
        and all(lc.get(k) is False for k in NO_TRAINING),
        "V18M_SOURCE_DEPTH_ORIGINAL_PARENT_OR_NONPROMOTION_BINDING")
    with np.load(v18m_npz,allow_pickle=False) as d:
        require(d.files==["source_cycles_depth_rgba"],
                "V18M_COURT_PRIVATE_DEPTH_ARRAY_MEMBERS")
        depth=np.asarray(d["source_cycles_depth_rgba"],np.float32)
    with np.load(v18l_npz,allow_pickle=False) as z:
        require(z.files==["common_cycles_sample_rgba"],
                "V18M_COURT_SEALED_V18L_SAMPLE_ARRAY_MEMBERS")
        colors=np.asarray(z["common_cycles_sample_rgba"],np.float32)
    require(depth.shape==(4,8,2,192,192,4)
        and colors.shape==(4,8,3,192,192,4),
        "V18M_ORIGINAL_SOURCE_CYCLES_COMMON_SAMPLE_DEPTH_SHAPE")
    reports=[]
    for i,view in enumerate(CAMERAS):
        row=measure_view(colors[i],depth[i])
        row["source_original_camera_view"]=view
        row["source_camera_partition"]=(
            "LOCKED_EVALUATION" if view in (4,6) else "EXPLORATORY")
        reports.append(row)
    require(sum(row["original_source_pixel_sample_instances"]
                for row in reports)==4*8*192*192,
        "V18M_INDEPENDENT_FULL_SOURCE_RENDER_SAMPLE_CENSUS")
    return {
        "schema":SCHEMA,"status":STATUS,
        "original_source_subject_id_private":m["subject_id"],
        "original_source_sha256":m["original_source_sha256"],
        "source_original_v18l_color_rgba_sha256":sha(v18l_npz),
        "source_original_v18m_depth_rgba_sha256":sha(v18m_npz),
        "source_original_v18m_manifest_sha256":sha(v18m_manifest),
        "source_original_independent_v18l_court_sha256":sha(v18l_court),
        "original_source_camera_reports":reports,
        "oracle_free_source_depth_order_predictor":True,
        "full_source_rgba_used_only_in_independent_scoring":True,
        "per_sample_source_depth_unknown_remains_unknown":True,
        "same_drawable_truth_minted":0,
        "different_drawable_truth_minted":0,
        "source_artist_graphics_owner_identity":"UNKNOWN",
        "source_artist_original_appearance_recovery":"NOT_EVALUATED",
        "source_common_sample_renderer_is_not_original_artist_32spp":True,
        "graphics_owner_authorized":False,
        "iris_training_authorized":False,
        "tessa_training_authorized":False,
        "materialization_authorized":False,
        "training_authorized":False,
    }


def main():
    p=argparse.ArgumentParser()
    for key in ("v18m_manifest","v18m_npz","v18l_manifest","v18l_npz",
                "v18l_court","output"):
        p.add_argument("--"+key.replace("_","-"),type=Path,required=True)
    p.add_argument("--expected-v18l-court-sha",required=True)
    args=p.parse_args()
    out=audit(args.v18m_manifest,args.v18m_npz,
        args.v18l_manifest,args.v18l_npz,args.v18l_court,
        args.expected_v18l_court_sha)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({
        "status":out["status"],
        "source":out["original_source_subject_id_private"],
        "per_camera":out["original_source_camera_reports"],
        "oracle_free_source_depth_order_predictor":True,
        "graphics_owner_authorized":False,
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
