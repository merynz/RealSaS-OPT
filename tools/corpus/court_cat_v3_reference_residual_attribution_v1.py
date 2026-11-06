#!/usr/bin/env python3
"""Retrospective residual attribution, not a new parity qualification.

On exactly sealed original Cat V3 source samples, A1-corrected origin and
unchanged CAA predictions, count alpha/coverage and opaque-RGB mismatches,
source sample-offset variability, and independent edit consequence errors.
No pixel masks are excluded; no source owner/training/model admission.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from tools.corpus.court_original_cat_v3_blender_pixel_origin_amendment_v1 import (
    source_bottom_first_to_top_first, ORIGINAL_REFERENCE_PREDICTIONS,
    ORIGINAL_PHYSICAL_COURT_V0,
)
from tools.corpus.court_original_cat_v3_puppet_reference_parity_v0 import (
    SOURCE_SAMPLES, TOL, EXPECTED_SOURCE_SHAPE, EXPECTED_PREDICTION_SHAPE,
)

A1_COURT = "ea856eebc796aec4d5dd548948085258aac05f4a53574d1aa2152aff7b914568"
SCHEMA = "RealSaS.CatV3ReferenceResidualAttribution.v1"


def digest(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def require(ok, code):
    if not ok:
        raise ValueError("CAT_V3_REFERENCE_RESIDUAL_" + code)


def classify_residuals_v1(reference_pm, original_rgba_bottom_first, *, tolerance=TOL):
    """Pure complete-pixel accounting, exact geometric noise-floor inequality."""
    p = np.asarray(reference_pm, np.float64)
    raw = np.asarray(original_rgba_bottom_first, np.float64)
    require(p.ndim == 6 and p.shape[:3] == (2,4,4) and p.shape[-1] == 4,
            "PREDICTIONS_INVALID")
    require(raw.shape == (2,4,2,4,*p.shape[-3:-1],4) and
            np.isfinite(raw).all() and np.isfinite(p).all(),
            "SOURCE_SHAPE_OR_FINITE_INVALID")
    require(tolerance == TOL, "TOLERANCE_DRIFT")
    src = source_bottom_first_to_top_first(raw)
    pm = np.concatenate([src[...,:3]*src[...,3:4],src[...,3:4]],axis=-1)
    expanded = p[:,:,None,:,:,:,:]
    diff = np.abs(expanded-pm)
    mismatched = np.max(diff,axis=-1)>tolerance
    alpha_bad = np.abs(expanded[...,3]-pm[...,3])>tolerance
    opaque_both = (expanded[...,3]>.99)&(pm[...,3]>.99)
    rgb_bad = np.max(diff[...,:3],axis=-1)>tolerance
    alpha_class = mismatched & alpha_bad
    opaque_rgb_class = mismatched & ~alpha_bad & opaque_both & rgb_bad
    remaining_class = mismatched & ~(alpha_class|opaque_rgb_class)
    # Two physical source stochastic offsets are tested against ONE reference
    # for same pose/view/edit. By L-infinity triangle inequality any pixel pair
    # separated by more than 2*TOL MUST cause >=1 prediction violation.
    source_offset_difference = np.max(
        np.abs(pm[:,:,0]-pm[:,:,1]),axis=-1)
    impossible_to_satisfy_both = source_offset_difference>2*tolerance
    source_offset_above_tolerance = source_offset_difference>tolerance
    per_arm={}
    for ai,name in enumerate(("BASE","SINGLE_0","SINGLE_1","SINGLE_2")):
        m=mismatched[:,:,:,ai]
        per_arm[name]={
            "pixel_sample_comparisons":int(m.size),
            "mismatched_pixel_samples":int(np.count_nonzero(m)),
            "two_source_offsets_exceed_2tolerance_pairs":int(np.count_nonzero(
                impossible_to_satisfy_both[:,:,ai])),
        }
    edit_effect={}
    for ai,name in enumerate(("SINGLE_0","SINGLE_1","SINGLE_2"),start=1):
        pred_delta=p[:,:,ai]-p[:,:,0]
        phys_delta=pm[:,:,:,ai]-pm[:,:,:,0]
        delta_error=np.max(np.abs(pred_delta[:,:,None]-phys_delta),axis=-1)
        edit_effect[name]={
            "mismatched_source_edit_effect_sample_pixels":int(np.count_nonzero(
                delta_error>tolerance)),
            "total_sample_pixels":int(delta_error.size),
            "max_delta_premultiplied_rgba_error":float(np.max(delta_error)),
        }
    return {
        "status":"RETROSPECTIVE_CAUSAL_DIAGNOSTIC_ONLY",
        "original_source_no_target_tuning":True,
        "orientation_conversion":"BLENDER_BOTTOM_FIRST_TO_CAA_TOP_FIRST_ONLY",
        "frozen_tolerance":tolerance,
        "full_image_comparison_pixel_samples":int(mismatched.size),
        "total_physical_reference_mismatches":int(np.count_nonzero(mismatched)),
        "alpha_or_coverage_mismatched_pixel_samples":int(np.count_nonzero(alpha_class)),
        "opaque_interior_rgb_mismatched_pixel_samples":int(np.count_nonzero(opaque_rgb_class)),
        "unclassified_mismatched_pixel_samples":int(np.count_nonzero(remaining_class)),
        "source_sample_offset_pairs":int(source_offset_difference.size),
        "source_two_offset_pairs_difference_above_tolerance":int(np.count_nonzero(
            source_offset_above_tolerance)),
        "deterministic_prediction_mandatory_miss_lower_bound_pixel_pairs":int(
            np.count_nonzero(impossible_to_satisfy_both)),
        "lower_bound_reason":"L_INFINITY_TRIANGLE_INEQUALITY__SAME_POSE_VIEW_ARM_PIXEL__SOURCE_DIFF_GT_2TOL",
        "per_arm":per_arm,
        "edit_delta_consequence":edit_effect,
        "k4_layer_overflow_spatial_attribution":"UNKNOWN__BINARY_OVERFLOW_MASK_NOT_EXPORTED",
        "source_artist_ownership":"UNKNOWN",
        "original_graphics_decomposition_qualified":False,
        "training_authorized":False,
        "materialization_authorized":False,
    }


def run(root: Path, output: Path):
    root=Path(root);output=Path(output)
    ps=root/"PRIVATE_CAT_V3_PUPPET_REFERENCE_PREDICTIONS_V0.npz"
    ss=root/"PRIVATE_CAT_NATIVE_POSE_UV_EDIT_CYCLES_SAMPLES_V3.npz"
    ai=root/"PRIVATE_CAT_V3_BLENDER_PIXEL_ORIGIN_AMENDMENT_A1_COURT.json"
    hist=root/"PRIVATE_INDEPENDENT_CAT_V3_PUPPET_REFERENCE_PHYSICAL_PARITY_V0.json"
    for path,sha in ((ps,ORIGINAL_REFERENCE_PREDICTIONS),
                     (ss,SOURCE_SAMPLES),(ai,A1_COURT),
                     (hist,ORIGINAL_PHYSICAL_COURT_V0)):
        require(digest(path)==sha,"IMMUTABLE_PARENT_SHA_DRIFT:"+path.name)
    a1=json.loads(ai.read_text());history=json.loads(hist.read_text())
    require(a1.get("corrected_source_readback_comparison",{}).get(
        "total_source_sample_operation_pixel_mismatches")==38167
        and a1.get("original_v0_recomputed",{}).get("total_mismatches")==1004587
        and history["physical_comparison"]["total_source_sample_operation_pixel_mismatches"]==1004587
        and a1.get("correction_source_axis")==-3
        and a1.get("learned_alignment_or_warp") is False,
        "A1_POST_HOC_CORRECTION_PROVENANCE_DRIFT")
    with np.load(ps,allow_pickle=False) as f:
        p=np.asarray(f["puppet_reference_premultiplied_linear_rgba"],np.float32)
    with np.load(ss,allow_pickle=False) as f:
        s=np.asarray(f["source_native_pose_uv_rgba"],np.float32)
    require(p.shape==EXPECTED_PREDICTION_SHAPE and
            s.shape==EXPECTED_SOURCE_SHAPE,
            "SOURCE_PUPPET_ARRAY_SHAPE_DRIFT")
    results=classify_residuals_v1(p,s)
    require(results["total_physical_reference_mismatches"]==38167
            and results["full_image_comparison_pixel_samples"]==2359296,
            "CORRECTED_ORIGIN_RECOUNT_NOT_REPRODUCED")
    report={
        "schema":SCHEMA,"A1_independent_court_sha256":A1_COURT,
        "original_source_v3_samples_sha256":SOURCE_SAMPLES,
        "compiled_reference_predictions_sha256":ORIGINAL_REFERENCE_PREDICTIONS,
        "historic_physical_V0_court_sha256":ORIGINAL_PHYSICAL_COURT_V0,
        "residual_attribution":results,
        "artist_owner":"UNKNOWN",
        "student_inference_authorized":False,
        "training_authorized":False,
        "materialization_authorized":False,
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({
        "status":results["status"],
        "mismatches":results["total_physical_reference_mismatches"],
        "alpha_coverage":results["alpha_or_coverage_mismatched_pixel_samples"],
        "opaque_rgb":results["opaque_interior_rgb_mismatched_pixel_samples"],
        "unclassified":results["unclassified_mismatched_pixel_samples"],
        "source_pair_unavoidable_miss_lower_bound":results[
            "deterministic_prediction_mandatory_miss_lower_bound_pixel_pairs"],
        "source_sample_pair_divergent_gt_tol":results[
            "source_two_offset_pairs_difference_above_tolerance"],
        "k4_spatial_attribution":"UNKNOWN",
        "training_authorized":False,
    },sort_keys=True),flush=True)
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    run(a.root,a.output)


if __name__=="__main__":
    main()
