#!/usr/bin/env python3
"""SHA-locked original Triceratops V18n K=2 regression for new variable-K core.

The new predictor sees ONLY original physically isolated material RGBA and
independent same-index V18m depth G/R. Source FULL/NATIVE are court targets.
V18n original raw producer had the known status spelling bug, verified by an
independent pre-existing recovery court. Do not "repair" raw parent bytes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from tools.corpus.compose_corpus370_source_sample_k_v1 import compose_isolated_original_source_k
from tools.corpus.court_corpus370_v18n_original_material_sample_replay import (
    predict_material_from_source_isolates,
)

SCHEMA="RealSaS.Corpus370IndependentOriginalSampleVariableKRegressionCourt.v1"
FROZEN_SOURCE="ad66b51cc5fc44c9a3429a5b1b98a61621af08d58d102d8a817499bd6a7f5b5a"
VIEWS=(0,2,4,6)
INDICES=(0,4,8,12,16,20,24,28)
TOL=.025

def require(x,code):
    if not x:raise ValueError("SAMPLE_K_REAL_SOURCE_"+code)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda:f.read(1<<20),b""):h.update(data)
    return h.hexdigest()

def load(path):
    return json.loads(Path(path).read_text())

def score(a,b,threshold):
    delta=np.max(np.abs(a-b),axis=-1)
    return {"mismatch_count":int(np.count_nonzero(delta>threshold)),
            "max_absolute_channel_difference":float(np.max(delta))}

def run(m_path,m_npz,d_path,d_npz,out):
    m=load(m_path)
    d=load(d_path)
    require(m.get("schema")=="RealSaS.Corpus370OriginalArtistPolicyMaterialSampleReplay.v18n"
        and d.get("schema")=="RealSaS.Corpus370OriginalCyclesSampleDepthEvidence.v18m",
        "ORIGINAL_MANIFEST_SCHEMAS")
    require(m.get("original_source_sha256")==d.get("original_source_sha256")==FROZEN_SOURCE
        and m.get("source_original_v18n_parent_v18m_run")=="42135271057-1"
        and m.get("source_depth_source_parent_v18m_court_sha256")
          =="3981af638b632a512b3fde4ac6a75d5d0e2d8a831a81f5becb017af26233ac99",
        "ORIGINAL_PARENT_LINEAGE")
    require(m.get("new_source_actual_edited_rgba_npz_sha256")==sha(m_npz)
        and d.get("new_source_actual_edited_rgba_npz_sha256")==sha(d_npz),
        "ORIGINAL_ARRAY_SHA")
    for item in (m,d):
        require(item.get("original_source_cycles_sample_indices")==list(INDICES)
            and item.get("heldout_source_camera_views")==list(VIEWS)
            and item.get("heldout_original_authored_pose_ordinal")==2
            and item.get("source_cycle_subset_length")==1
            and item.get("source_cycle_sampling_seed")==0
            and item.get("source_cycles_adaptive_sampling") is False
            and item.get("source_cycles_pixel_filter")=="BOX"
            and item.get("source_cycles_pixel_filter_width")==.01
            and item.get("original_source_camera_sha256")
                =="e73917f650ee2dd925efe68a88e2fafdd5f99dbc5b7c410b7de0532ecf66bb0d",
            "ORIGINAL_CYCLES_SAMPLE_CAMERA_DRIFT")
    require(m.get("source_edit_arms")==[
        "SOURCE_NATIVE","SOUP_FULL","SOUP_ISOLATE","SOUP_REMOVE"]
        and d.get("source_edit_arms")==["GROUP_DEPTH","REMAINDER_DEPTH"],
        "SOURCE_EDIT_ARM_ORDER")
    with np.load(m_npz,allow_pickle=False) as x, np.load(d_npz,allow_pickle=False) as y:
        original=np.asarray(x["source_material_cycles_sample_rgba"],dtype=np.float64)
        depth=np.asarray(y["source_cycles_depth_rgba"],dtype=np.float64)
    require(original.shape==(4,8,4,192,192,4)
        and depth.shape==(4,8,2,192,192,4)
        and np.isfinite(original).all() and np.isfinite(depth).all(),
        "ORIGINAL_SOURCE_SAMPLE_ARRAY_LAYOUT")
    reports=[]
    for vi,view in enumerate(VIEWS):
        colors=original[vi,:,2:]  # physically isolated G/R ONLY
        depths=depth[vi]          # independent same sample G/R geometry
        prediction=compose_isolated_original_source_k(colors,depths)
        prior=predict_material_from_source_isolates(colors,depths)
        pred=prediction["predicted_source_sample_premultiplied_rgba"]
        valid=prediction["qualified_sample_mask"]
        require(np.array_equal(valid,prior["source_valid_sample_mask"]),
                "K2_VALIDITY_NOT_COMPATIBLE_WITH_V18N")
        old=prior["predicted_sample_premultiplied_rgba"]
        parity=score(pred,old,1e-6)
        original_native=original[vi,:,0]
        original_full=original[vi,:,1]
        native=np.concatenate((original_native[...,:3]*original_native[...,3:4],
                               original_native[...,3:4]),axis=-1)
        soup=np.concatenate((original_full[...,:3]*original_full[...,3:4],
                             original_full[...,3:4]),axis=-1)
        source_target=score(pred[valid],native[valid],TOL)
        source_soup=score(pred[valid],soup[valid],TOL)
        report={"view_index":view,"cycles_source_indices":list(INDICES),
            "total_sample_pixels":int(valid.size),
            "qualified_sample_pixels":int(np.count_nonzero(valid)),
            "unknown_sample_pixels":int(np.count_nonzero(~valid)),
            "v18n_k2_predictor_parity":parity,
            "new_k2_to_original_native":source_target,
            "new_k2_to_original_soup":source_soup}
        reports.append(report)
    failures=[x for x in reports
        if x["v18n_k2_predictor_parity"]["mismatch_count"]>0
        or x["new_k2_to_original_native"]["mismatch_count"]>0
        or x["new_k2_to_original_soup"]["mismatch_count"]>0
        or x["unknown_sample_pixels"]>0]
    result={"schema":SCHEMA,
        "status":"PASS_SOURCE_ORIGINAL_K2_REGRESSION_NO_ARTIST_OWNER"
            if not failures else "FAIL_SOURCE_ORIGINAL_K2_REGRESSION",
        "original_source_sha256":FROZEN_SOURCE,
        "original_v18n_source_manifest_sha256":sha(m_path),
        "original_v18n_source_rgba_npz_sha256":sha(m_npz),
        "original_v18m_source_depth_manifest_sha256":sha(d_path),
        "original_v18m_source_depth_npz_sha256":sha(d_npz),
        "original_v18n_raw_status_spelling_repair_performed":False,
        "source_original_subject_private":"TRICERATOPS",
        "source_sample_views":list(VIEWS),
        "source_sample_offsets":list(INDICES),
        "source_K":2,
        "source_sample_total":int(sum(x["total_sample_pixels"] for x in reports)),
        "per_view_source_regression_court":reports,
        "failed_views":[x["view_index"] for x in failures],
        "source_material_isolate_input_only":True,
        "source_full_native_used_as_independent_court_targets_only":True,
        "source_artist_drawable_owner":"UNKNOWN",
        "generic_variable_K3_real_original_source_courted":False,
        "graphics_owner_authorized":False,"iris_training_authorized":False,
        "tessa_training_authorized":False,"materialization_authorized":False,
        "training_authorized":False}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n",
                   encoding="utf-8")
    print(json.dumps({"status":result["status"],
        "source_sample_total":result["source_sample_total"],
        "failed_views":result["failed_views"],"training_authorized":False},sort_keys=True))
    require(not failures,"ORIGINAL_SOURCE_SAMPLE_K2_PARITY_FAILURE")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("v18n-manifest","v18n-npz","v18m-manifest","v18m-npz","output"):
        p.add_argument("--"+k,required=True,type=Path)
    a=p.parse_args()
    run(a.v18n_manifest,a.v18n_npz,a.v18m_manifest,a.v18m_npz,a.output)
