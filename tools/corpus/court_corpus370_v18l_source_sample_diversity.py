#!/usr/bin/env python3
"""V18l follow-up: independently assess whether common Cycles subsets are nonvacuous.

A source worker can successfully validate sha/schema and source edit sample
consistency, but still be scientifically useless if all subset-index frames
hit the same pixel locations. This court counts per-index FULL RGBA variability
over ALL pixels and relates full sample-first compositing to postfilter OVER.
It never grants layer, source artist texture, or student training authority.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from tools.corpus.court_corpus370_v18l_cycles_common_sample_source import (
    premul,over,per_source_view,
    SCHEMA as SOURCE_COURT_SCHEMA,
    STATUS as SOURCE_COURT_STATUS,
    SOURCE as SOURCE_RENDER_SCHEMA,
    SOURCE_STATUS as SOURCE_RENDER_STATUS,
    TOL,
)

SCHEMA="RealSaS.Corpus370IndependentCyclesSubsetNonvacuityCourt.v18l"
PASS="V18L_SOURCE_SAMPLE_DIVERSITY_INDEPENDENTLY_AUDITED__DRAWABLE_UNKNOWN"


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def require(value,err):
    if not value:
        raise ValueError(err)


def compare(frames):
    require(frames.shape==(8,3,192,192,4)
            and np.isfinite(frames).all()
            and np.min(frames[...,3])>=-1e-5
            and np.max(frames[...,3])<=1+1e-5,
            "V18L_DIVERSITY_SOURCE_SUBSET_RGBA_SHAPE")
    # Use premultiplied RGB and alpha: transparent RGB bleed is not evidence.
    p=premul(frames)
    full=p[:,0]
    group=p[:,1]
    rest=p[:,2]
    full_spread=np.max(np.max(full,axis=0)-np.min(full,axis=0),axis=-1)
    group_spread=np.max(np.max(group,axis=0)-np.min(group,axis=0),axis=-1)
    rest_spread=np.max(np.max(rest,axis=0)-np.min(rest,axis=0),axis=-1)
    full_mean=np.mean(full,axis=0)
    ga=np.mean(group,axis=0)
    ra=np.mean(rest,axis=0)
    sample_gr=np.mean(over(group,rest),axis=0)
    sample_rg=np.mean(over(rest,group),axis=0)
    post_gr=over(ga,ra)
    post_rg=over(ra,ga)
    ma=lambda x:np.max(np.abs(full_mean-x),axis=-1)
    sample_error=np.minimum(ma(sample_gr),ma(sample_rg))
    post_error=np.minimum(ma(post_gr),ma(post_rg))
    prior=per_source_view(frames)
    require(int(np.count_nonzero(sample_error>TOL))==
        prior["source_8_sample_pre_filter_best_fixed_order_pixel_failures"]
        and int(np.count_nonzero(post_error>TOL))==
        prior["source_8_sample_premultiplied_joint_pre_filter_vs_postfilter_best_order_pixel_failures"],
        "V18L_DIVERSITY_RECOMPUTED_SAMPLE_LAYER_MATH_CHANGED")
    return {
      "source_original_full_nonvacuous_different_subsets_pixel_count":
        int(np.count_nonzero(full_spread>TOL)),
      "source_group_nonvacuous_pixel_count":
        int(np.count_nonzero(group_spread>TOL)),
      "source_rest_nonvacuous_pixel_count":
        int(np.count_nonzero(rest_spread>TOL)),
      "source_full_8_subset_max_premul_rgba_change":
        float(np.max(full_spread)),
      "source_full_sample_subset_positions_proven_identical_rays":False,
      "real_cycles_subset_index_variability_observed":
        bool(np.any(full_spread>TOL)),
      "source_8_subset_full_mean_nonzero_alpha_pixels":
        int(np.count_nonzero(full_mean[...,3]>TOL)),
      "source_8_subset_prefilter_order_fail_pixels":
        int(np.count_nonzero(sample_error>TOL)),
      "source_8_subset_postfilter_order_fail_pixels":
        int(np.count_nonzero(post_error>TOL)),
      "source_8_subset_postfilter_fail_but_prefilter_pass_pixels":
        int(np.count_nonzero((post_error>TOL)&(sample_error<=TOL))),
      "source_8_subset_postfilter_pass_but_prefilter_fail_pixels":
        int(np.count_nonzero((post_error<=TOL)&(sample_error>TOL))),
      "source_8_subset_prefilter_improves_error_pixels":
        int(np.count_nonzero(sample_error+1e-12<post_error)),
      "source_8_subset_prefilter_worsens_error_pixels":
        int(np.count_nonzero(sample_error>post_error+1e-12)),
      "source_8_subset_prefilter_ties_error_pixels":
        int(np.count_nonzero(np.abs(sample_error-post_error)<=1e-12)),
      "source_render_same_index_sample_oracle_fail_instances":
        prior["same_index_sample_best_order_failed_pixel_instances"],
      "no_original_artist_drawable_or_texture_authority":True,
    }


def audit(manifest_path,raw_npz_path,source_court_path,expected_source_court_sha):
    require(len(expected_source_court_sha)==64
        and sha(source_court_path)==expected_source_court_sha,
        "V18L_DIVERSITY_EXPECTED_INDEPENDENT_PARENT_COURT_SHA")
    m=load(manifest_path)
    c=load(source_court_path)
    require(m.get("schema")==SOURCE_RENDER_SCHEMA
        and m.get("status")==SOURCE_RENDER_STATUS
        and c.get("schema")==SOURCE_COURT_SCHEMA
        and c.get("status")==SOURCE_COURT_STATUS
        and c.get("source_render_manifest_sha256")==sha(manifest_path)
        and c.get("source_render_npz_sha256")==sha(raw_npz_path)
        and c.get("original_source_sha256")==m.get("original_source_sha256")
        and c.get("subject_id_private")==m.get("subject_id")
        and c.get("not_a_cycles_32_sample_full_frame_parity_test") is True
        and m.get("new_source_actual_edited_rgba_npz_sha256")==sha(raw_npz_path)
        and all(o.get(k) is False for o in (m,c) for k in (
            "graphics_owner_authorized","iris_training_authorized",
            "tessa_training_authorized","materialization_authorized",
            "training_authorized")),
        "V18L_DIVERSITY_FROZEN_SOURCE_COURT_AND_NO_AUTHORITY")
    with np.load(raw_npz_path,allow_pickle=False) as f:
        require(f.files==["common_cycles_sample_rgba"],
                "V18L_DIVERSITY_UNEXPECTED_ORIGINAL_SAMPLE_DATA")
        a=np.asarray(f["common_cycles_sample_rgba"],np.float32)
    require(a.shape==(4,8,3,192,192,4),
            "V18L_DIVERSITY_SOURCE_FOUR_CAMERA_SAMPLES_MISSING")
    rows=[]
    for i,view in enumerate((0,2,4,6)):
        z=compare(a[i])
        # The upstream court measured a superset, so source class identity
        # and per-sample discrepancy must exactly re-align with raw bytes.
        orig=c["by_original_source_camera"][i]
        require(orig["source_original_view"]==view
            and orig["same_index_sample_best_order_failed_pixel_instances"]==
                z["source_render_same_index_sample_oracle_fail_instances"]
            and orig["source_8_sample_premultiplied_joint_pre_filter_vs_postfilter_best_order_pixel_failures"]==
                z["source_8_subset_postfilter_order_fail_pixels"]
            and orig["source_8_sample_pre_filter_best_fixed_order_pixel_failures"]==
                z["source_8_subset_prefilter_order_fail_pixels"],
            "V18L_DIVERSITY_UPSTREAM_INDEPENDENT_MEASUREMENT_CHANGED")
        rows.append({"camera_view":view,"camera_holdout_partition":
                     "EVALUATION" if view in (4,6) else "EXPLORATORY",**z})
    total=sum(z["source_original_full_nonvacuous_different_subsets_pixel_count"]
              for z in rows)
    return {
      "schema":SCHEMA,"status":PASS,
      "subject_id_private":c["subject_id_private"],
      "original_source_sha256":c["original_source_sha256"],
      "source_render_manifest_sha256":sha(manifest_path),
      "source_render_rgba_sha256":sha(raw_npz_path),
      "independent_parent_source_court_sha256":sha(source_court_path),
      "original_common_sample_run":c["original_source_renderer_policy"].get(
          "parent_v18l_run","UNKNOWN_RUN_FROM_SOURCE_COURT"),
      "per_original_view":rows,
      "total_nonvacuous_pixel_view_occurrences":total,
      "nonvacuity_gate_passed":bool(total>0),
      "full_source_sample_index_effects_nonvacuous":bool(total>0),
      "same_sample_offset_is_not_guarantee_of_exact_rays":True,
      "no_unseen_character_or_artist_layer_success_claimed":True,
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
    for x in ("manifest","rgba","source_court","output"):
        p.add_argument("--"+x.replace("_","-"),type=Path,required=True)
    p.add_argument("--expected-source-court-sha",required=True)
    a=p.parse_args()
    v=audit(a.manifest,a.rgba,a.source_court,a.expected_source_court_sha)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(v,sort_keys=True,indent=2,
                                   allow_nan=False)+"\n")
    print(json.dumps({
        "status":v["status"],
        "per_original_view":v["per_original_view"],
        "nonvacuity_gate_passed":v["nonvacuity_gate_passed"],
        "training_authorized":False,
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
