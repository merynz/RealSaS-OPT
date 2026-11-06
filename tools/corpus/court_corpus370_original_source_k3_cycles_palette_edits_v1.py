#!/usr/bin/env python3
"""Independent original-source three-palette SAME Cycles sample edited compositor.

Never imports Blender source-render producer. Predicts from original physically
isolated source material/depth and *edited isolate only*, then scores source
FULL/edited FULL after prediction. No trained-owner authority.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from tools.corpus.compose_corpus370_source_sample_k_v1 import compose_isolated_original_source_k

SCHEMA="RealSaS.Corpus370IndependentOriginalThreePaletteCyclesEditCourt.v1"
PRODUCER="RealSaS.Corpus370OriginalThreePaletteCommonCyclesSampleEditEvidence.v1"
SOURCE_SHA="ad66b51cc5fc44c9a3429a5b1b98a61621af08d58d102d8a817499bd6a7f5b5a"
SOURCE_PALETTE_SHA="8ff5534e4e8d7549cb9d419e4aea58511e0e042f9113e7d5807a8652a1b76edc"
VIEW_SET=(1,3,5,7)
INDICES=(0,4,8,12,16,20,24,28)
ARM_SET=("SOURCE_FULL","ISOLATE_0","ISOLATE_1","ISOLATE_2",
      "DEPTH_0","DEPTH_1","DEPTH_2",
      "EDITED_ISOLATE_0","EDITED_ISOLATE_1","EDITED_ISOLATE_2",
      "EDITED_SOURCE_FULL_0","EDITED_SOURCE_FULL_1","EDITED_SOURCE_FULL_2")
RGBA_TOL=.025
EDIT_MIN=8
MIN_PHYSICAL_OVERLAP=8
MIN_QUALIFIED=100000

def req(ok,code):
    if not ok:raise ValueError("SAMPLE_K3_ORIGINAL_COURT_"+code)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()

def premul(image):
    x=np.asarray(image,dtype=np.float64)
    return np.concatenate((x[...,:3]*x[...,3:4],x[...,3:4]),axis=-1)

def measure(arr):
    """Return bounded per-view metrics; input shape [8,13,192,192,4]."""
    a=np.asarray(arr,dtype=np.float64)
    req(a.shape==(8,13,192,192,4) and np.isfinite(a).all()
        and np.min(a[...,3])>=-1e-5 and np.max(a[...,3])<=1.00001,
        "ORIGINAL_SOURCE_SAMPLE_RGBA_ARRAY")
    isolated=a[:,1:4]
    depth=a[:,4:7]
    base=compose_isolated_original_source_k(isolated,depth)
    qualified=base["qualified_sample_mask"]
    base_target=premul(a[:,0])
    base_predictions=base["predicted_source_sample_premultiplied_rgba"]
    req(np.array_equal(qualified,base["qualified_sample_mask"]),
        "BASELINE_QUALIFIED_SAMPLE_ORDER")
    outcomes=[]
    metrics=[]
    for index in (-1,0,1,2):
        if index<0:
            prediction=base
            target=base_target
            label="SOURCE_FULL"
        else:
            edited=isolated.copy()
            edited[:,index]=a[:,7+index]
            prediction=compose_isolated_original_source_k(edited,depth)
            target=premul(a[:,10+index])
            label=f"SOURCE_EDITED_FULL_{index}"
        q=prediction["qualified_sample_mask"]
        req(np.array_equal(q,qualified),"EDITED_DEPTH_CHANGED_NON_GEOMETRIC_QUALIFICATION")
        delta=np.max(np.abs(prediction["predicted_source_sample_premultiplied_rgba"]-target),axis=-1)
        count=int(np.count_nonzero((delta>RGBA_TOL)&q))
        max_delta=float(np.max(delta[q])) if np.any(q) else float("inf")
        sample_count=int(np.count_nonzero(q))
        # A mean pixel is qualified only if all eight same Cycles samples valid.
        all_pixels=np.all(q,axis=0)
        predicted_8=np.mean(prediction["predicted_source_sample_premultiplied_rgba"],axis=0)
        target_8=np.mean(target,axis=0)
        aggregate=np.max(np.abs(predicted_8-target_8),axis=-1)
        aggregate_errors=int(np.count_nonzero((aggregate>RGBA_TOL)&all_pixels))
        edits_changed=0
        if index>=0:
            edited_full_delta=np.max(np.abs(base_target-target),axis=-1)
            edits_changed=int(np.count_nonzero((edited_full_delta>RGBA_TOL)&q))
        metrics.append({"arm":label,"qualified_source_sample_pixels":sample_count,
            "unknown_source_samples":int(np.count_nonzero(~q)),
            "sample_rgba_mismatch_count":count,
            "sample_max_rgba_abs_error":max_delta,
            "fully_qualified_eight_sample_pixels":int(np.count_nonzero(all_pixels)),
            "eight_sample_mean_rgba_mismatch_count":aggregate_errors,
            "physical_full_edit_changed_sample_pixel_count":edits_changed})
        outcomes.append(prediction)
    overlap=int(np.count_nonzero((base["physical_source_group_hit_count"]>=2)&qualified))
    return {"metric_rows":metrics,"source_multi_group_overlapping_sample_count":overlap,
            "source_unknown_depth_tie_sample_count":
                int(np.count_nonzero(base["unknown_depth_tie_mask"])),
            "source_invalid_depth_sample_count":
                int(np.count_nonzero(base["invalid_source_depth_mask"])),
            "source_full_alpha_intermediate_single_source_sample_count":
                int(np.count_nonzero((a[:,0,...,3]>1e-6)&(a[:,0,...,3]<1.-1e-6))),
            "source_isolate_nonbinary_alpha_count":
                int(np.count_nonzero((isolated[...,3]>1e-6)&(isolated[...,3]<1.-1e-6)))}

def run(manifest_path,array_path,output):
    p=json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    req(p.get("schema")==PRODUCER
        and p.get("status")=="SOURCE_ORIGINAL_K3_CYCLES_EDIT_SAMPLES_RENDERED__COURT_REQUIRED"
        and p.get("original_source_sha256")==SOURCE_SHA
        and p.get("source_original_material_palette_sha256")==SOURCE_PALETTE_SHA,
        "ORIGINAL_PRIVATE_PRODUCER_SCHEMA_AND_PARENT")
    req(p.get("source_cycle_sample_indices")==list(INDICES)
        and p.get("source_cycle_camera_view_indices")==list(VIEW_SET)
        and p.get("source_cycle_arms")==list(ARM_SET)
        and p.get("source_cycle_samples_max")==32
        and p.get("source_cycle_subset_length")==1
        and p.get("source_cycle_seed")==0
        and p.get("source_cycle_adaptive_sampling") is False
        and p.get("source_cycle_pixel_filter")=="BOX"
        and p.get("source_cycle_filter_width")==.01
        and p.get("source_cycle_resolution")==192,
        "FROZEN_SAME_CYCLES_SAMPLE_POLICY")
    req(p.get("source_original_triangles_per_color") and
        sum(p["source_original_triangles_per_color"])==1334 and
        len(p["source_original_triangles_per_color"])==3
        and all(k>0 for k in p["source_original_triangles_per_color"])
        and p.get("physical_nonselected_faces_removed") is True
        and p.get("source_original_rest_pose_only") is True
        and p.get("source_original_artwork_heldout_pose_test") is False
        and p.get("source_full_and_edited_full_are_court_targets_only") is True
        and p.get("source_color_edit_rule")
          =="RGB_COMPONENT_GE_HALF_TO_0P15_OTHER_TO_0P85_ALPHA_UNCHANGED",
        "ORIGINAL_SOURCE_PHYSICAL_K3_ISOLATION_POLICY")
    req(p.get("source_original_edited_sample_npz_sha256")==sha(array_path)
        and p.get("source_original_edited_sample_npz_shape")==[
            4,8,13,192,192,4],
        "ORIGINAL_SOURCE_PRIVATE_ARRAY_SHA")
    for key in ("graphics_owner_authorized","iris_training_authorized",
                "tessa_training_authorized","materialization_authorized",
                "training_authorized"):
        req(p.get(key) is False,"ORIGINAL_SOURCE_PRODUCER_AUTHORITY_DRIFT_"+key)
    with np.load(array_path,allow_pickle=False) as npz:
        source=np.asarray(npz["source_original_palette_k3_sample_rgba"],dtype=np.float64)
    req(source.shape==(4,8,13,192,192,4) and np.isfinite(source).all(),
        "ORIGINAL_CYCLES_SAMPLE_PAYLOAD")
    reports=[]
    for i,view in enumerate(VIEW_SET):
        result=measure(source[i])
        reports.append({"source_view_index":view,**result})
    failures=[]
    total_valid=sum(r["metric_rows"][0]["qualified_source_sample_pixels"] for r in reports)
    total_multi=sum(r["source_multi_group_overlapping_sample_count"] for r in reports)
    if total_valid<MIN_QUALIFIED:
        failures.append("INSUFFICIENT_K3_ORIGINAL_SAMPLE_COVERAGE")
    if total_multi<MIN_PHYSICAL_OVERLAP:
        failures.append("INSUFFICIENT_ORIGINAL_CYCLES_MULTI_GROUP_OVERLAP")
    for k in range(3):
        changed=sum(r["metric_rows"][k+1]["physical_full_edit_changed_sample_pixel_count"]
                    for r in reports)
        if changed<EDIT_MIN:failures.append(f"PALETTE_{k}_PHYSICAL_EDIT_NONVACUITY")
    for r in reports:
        if r["source_full_alpha_intermediate_single_source_sample_count"]>0 or \
           r["source_isolate_nonbinary_alpha_count"]>0:
            failures.append("NONBINARY_ALPHA_SOURCE_POLICY_UNSUPPORTED")
        for row in r["metric_rows"]:
            if row["sample_rgba_mismatch_count"]>0 or \
               row["eight_sample_mean_rgba_mismatch_count"]>0:
                failures.append(f"VIEW_{r['source_view_index']}_{row['arm']}_RGBA_COURT_FAIL")
    result={"schema":SCHEMA,
        "status":"PASS_ORIGINAL_PHYSICAL_K3_SAME_CYCLES_SAMPLE_EDIT_COURT__OWNER_UNKNOWN"
            if not failures else "FAIL_ORIGINAL_PHYSICAL_K3_SAME_CYCLES_SAMPLE_EDIT_COURT",
        "original_source_sha256":SOURCE_SHA,
        "original_source_renderer_manifest_sha256":sha(manifest_path),
        "original_source_renderer_rgba_npz_sha256":sha(array_path),
        "original_source_palette_facts_sha256":SOURCE_PALETTE_SHA,
        "original_camera_views":list(VIEW_SET),
        "cycles_indices":list(INDICES),
        "source_group_count_K":3,
        "qualified_sample_total":total_valid,
        "original_physical_overlapping_source_sample_count":total_multi,
        "source_per_view_results":reports,
        "failure_reasons":failures,
        "producer_court_source_full_not_used_to_choose_depth":True,
        "source_original_rest_only":True,
        "source_edit_sampling_policy_heldout_vs_previous_v2":False,
        "source_artist_drawable_owner":"UNKNOWN",
        "same_drawable_training_pairs_qualified":0,
        "different_drawable_training_pairs_qualified":0,
        "graphics_owner_authorized":False,"iris_training_authorized":False,
        "tessa_training_authorized":False,"materialization_authorized":False,
        "training_authorized":False}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n",
                      encoding="utf-8")
    print(json.dumps({"status":result["status"],"qualified_source_samples":total_valid,
        "overlap_samples":total_multi,"failure_count":len(failures),
        "training_authorized":False},sort_keys=True),flush=True)
    req(not failures,"ORIGINAL_SOURCE_K3_SAME_INDEX_COURT_FAIL")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for arg in ("manifest","array","output"):parser.add_argument("--"+arg,type=Path,required=True)
    a=parser.parse_args()
    run(a.manifest,a.array,a.output)
