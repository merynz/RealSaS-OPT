#!/usr/bin/env python3
"""Independent retrospectively paired original-source V18k error cross-tab.

Requires exact SHA-pinned original independent V18k geometric court whose
recalculated source rays already qualify source geometry and source lineage.
Recomputes full-domain per-pixel three-way errors, with actual heldout camera
and explicit global regression accounting; no positive artist/model authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import fmean

from tools.corpus.measure_corpus370_v18k_sample_aware_source_compositing import (
    SCHEMA as SOURCE_SCHEMA, STATUS as SOURCE_STATUS, THRESHOLDS, VIEWS, TOL,
)
from tools.corpus.court_corpus370_v18k_sample_aware_source_compositing import (
    SCHEMA as COURT_SCHEMA, STATUS as COURT_STATUS,
)

SCHEMA="RealSaS.Corpus370V18kIndependentFullDomainPairedResidualCrosscheck.v1"
STATUS="SOURCE_PAIRS_REEVALUATED__LOCAL_GAINS_AND_GLOBAL_REGRESSIONS_BOTH_REPORTED"


def require(ok,code):
    if not ok:
        raise ValueError(code)


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def load(p):
    return json.loads(Path(p).read_text())


def aggregate(rows):
    """Pair only *same source view, threshold and pixel* residuals."""
    require(rows,"V18K_CROSSTAB_NONEMPTY_SOURCE_ROWS")
    totals={}
    for r in rows:
        comp=r["actual_original_blender_source_comparison"]
        geo=r["geometry_subsample_compositor"]
        layer=comp["real_blender_v18h_postfilter_best_of_two_source_orders_error"]
        joint=comp["source_joint_sample_error"]
        post=comp["geometry_postfilter_best_order_error"]
        if joint is None or post is None:
            validity="UNKNOWN_GEOMETRIC_SAMPLE"
        else:
            validity="QUALIFIED_ALL_4X4"
            require(all(isinstance(q,(int,float)) and q>=0 and q<10
                        for q in (layer,joint,post))
                    and comp["real_blender_v18h_postfilter_both_orders_fail"]==
                        (layer>TOL)
                    and comp["sample_reconstruction_improves_actual_blender_postfilter"]==
                        (joint+1e-12<layer),
                    "V18K_CROSSTAB_PIXEL_METRIC_RECONCILIATION")
        v=int(r["source_camera_view"])
        t=float(r["frozen_v18a_threshold"])
        name=("LOCKED_EVALUATION" if v in (4,6) else "EXPLORATORY")
        require(v in VIEWS and t in THRESHOLDS
            and r["camera_evaluation_partition"]==(
                "LOCKED_EVALUATION_VIEW" if v in (4,6)
                else "EXPLORATORY_VIEW")
            and r["geometry_precommitted_selection"] is True
            and r["rgba_error_never_used_for_pixel_selection"] is True
            and r["source_graphic_drawable_owner"]=="UNKNOWN"
            and r["vacuous_whole_character_group"] is
                (r["source_group_face_count"]==r["whole_source_face_count"]),
            "V18K_CROSSTAB_FROZEN_CAMERA_SOURCE_GROUP_LABEL")
        k=f"{name}:{t}"
        entry=totals.setdefault(k,{
            "source_pixel_count":0,"geometrically_qualified":0,
            "geometry_unknown":0,"vacuous_k1_source_pixels":0,
            "blender_postfilter_failed":0,
            "geometric_joint_failed":0,"geometric_postfilter_failed":0,
            "geometric_joint_beats_blender_postfilter":0,
            "geometric_joint_worse_than_blender_postfilter":0,
            "geometric_joint_tied_blender_postfilter":0,
            "geometric_joint_beats_blender_among_original_blender_failures":0,
            "geometric_joint_worse_among_original_blender_failures":0,
            "geometric_joint_rescues_original_blender_failures_under_tolerance":0,
            "geometric_joint_beats_geometry_postfilter_among_blender_failures":0,
            "geometric_joint_and_geometry_postfilter_diverge_over_tolerance":0,
            "geometric_mixed_nearest_surface_pixels":0,
            "geometric_mixed_nearest_among_blender_failures":0,
            "blender_postfilter_alpha_invariant_failures":0,
            "_blender_errors":[],"_joint_errors":[],"_geometry_post_errors":[],
        })
        entry["source_pixel_count"]+=1
        entry["vacuous_k1_source_pixels"]+=int(r["vacuous_whole_character_group"])
        entry["blender_postfilter_failed"]+=int(layer>TOL)
        entry["blender_postfilter_alpha_invariant_failures"]+=int(
            comp["real_blender_v18h_postfilter_order_invariant_alpha_error"]>TOL)
        if validity=="UNKNOWN_GEOMETRIC_SAMPLE":
            entry["geometry_unknown"]+=1
            continue
        entry["geometrically_qualified"]+=1
        entry["geometric_joint_failed"]+=int(joint>TOL)
        entry["geometric_postfilter_failed"]+=int(post>TOL)
        entry["geometric_joint_and_geometry_postfilter_diverge_over_tolerance"]+=int(
            geo["geometric_opaque_sample_composition"][
                "joint_vs_best_postfilter_order_max_rgba_delta"]>TOL)
        mixed=bool(geo["geometric_opaque_sample_composition"][
            "mixed_source_nearest_G_R_within_pixel"])
        entry["geometric_mixed_nearest_surface_pixels"]+=int(mixed)
        entry["geometric_mixed_nearest_among_blender_failures"]+=int(
            mixed and layer>TOL)
        delta=joint-layer
        entry["geometric_joint_beats_blender_postfilter"]+=int(delta<-1e-12)
        entry["geometric_joint_worse_than_blender_postfilter"]+=int(delta>1e-12)
        entry["geometric_joint_tied_blender_postfilter"]+=int(abs(delta)<=1e-12)
        if layer>TOL:
            entry["geometric_joint_beats_blender_among_original_blender_failures"]+=int(
                delta<-1e-12)
            entry["geometric_joint_worse_among_original_blender_failures"]+=int(
                delta>1e-12)
            entry["geometric_joint_rescues_original_blender_failures_under_tolerance"]+=int(
                joint<=TOL)
            entry["geometric_joint_beats_geometry_postfilter_among_blender_failures"]+=int(
                joint+1e-12<post)
        entry["_blender_errors"].append(layer)
        entry["_joint_errors"].append(joint)
        entry["_geometry_post_errors"].append(post)
    for k,e in totals.items():
        require(e["geometrically_qualified"]+e["geometry_unknown"]==
                e["source_pixel_count"]
            and e["geometric_joint_beats_blender_postfilter"]+
                e["geometric_joint_worse_than_blender_postfilter"]+
                e["geometric_joint_tied_blender_postfilter"]==
                e["geometrically_qualified"],
            "V18K_CROSSTAB_SOURCE_QUALITY_CENSUS")
        if e["geometrically_qualified"]:
            e["mean_blender_postfilter_max_premul_rgba_error"]=fmean(
                e.pop("_blender_errors"))
            e["mean_geometric_joint_max_premul_rgba_error"]=fmean(
                e.pop("_joint_errors"))
            e["mean_geometry_postfilter_max_premul_rgba_error"]=fmean(
                e.pop("_geometry_post_errors"))
        else:
            for item in ("_blender_errors","_joint_errors","_geometry_post_errors"):
                e.pop(item)
    return totals


def audit(proposal,court,expected_court_sha):
    require(isinstance(expected_court_sha,str) and len(expected_court_sha)==64
            and sha(court)==expected_court_sha,
            "V18K_CROSSTAB_FROZEN_INDEPENDENT_SOURCE_COURT_SHA")
    p=load(proposal)
    c=load(court)
    require(p.get("schema")==SOURCE_SCHEMA and p.get("status")==SOURCE_STATUS
        and c.get("schema")==COURT_SCHEMA and c.get("status")==COURT_STATUS
        and c.get("source_original_v18k_proposal_sha256")==sha(proposal)
        and c.get("original_source_sha256")==p.get("original_source_sha256")
        and c.get("original_source_subject_id_private")==p.get("source_subject_id_private")
        and c.get("total_threshold_view_pixel_query_rows")==
          len(p.get("original_source_pixel_rows_private_teacher_only",[]))
        and c.get("source_geometry_only_selection") is True
        and c.get("whole_character_k1_is_vacuous") is True
        and c.get("not_exact_blender_cycles_sample_pattern") is True
        and c.get("original_artist_drawable_and_source_alpha")=="UNKNOWN"
        and all(o.get(key) is False for o in (p,c) for key in (
            "graphics_owner_authorized","iris_training_authorized",
            "tessa_training_authorized","materialization_authorized",
            "training_authorized")),
        "V18K_CROSSTAB_PARENT_COURT_OR_AUTHORITY_INVALID")
    rows=p["original_source_pixel_rows_private_teacher_only"]
    by_bucket=aggregate(rows)
    require(set(by_bucket)=={f"{group}:{t}" for group in
        ("EXPLORATORY","LOCKED_EVALUATION") for t in THRESHOLDS},
        "V18K_CROSSTAB_MISSING_ORIGINAL_CAMERA_THRESHOLD")
    for key,data in by_bucket.items():
        group,t=key.split(":")
        view_ids=(0,2) if group=="EXPLORATORY" else (4,6)
        prior=sum(c["per_threshold_evaluation_partition_metrics"][
                f"{'EXPLORATORY' if group=='EXPLORATORY' else 'EVALUATION'}:{t}"][
                    "original_source_pixel_queries"] for _ in (0,))
        require(data["source_pixel_count"]==prior,
                "V18K_CROSSTAB_INDEPENDENT_COURT_PIXEL_CENSUS")
    return {
        "schema":SCHEMA,"status":STATUS,
        "original_source_subject_id_private":c["original_source_subject_id_private"],
        "original_source_sha256":c["original_source_sha256"],
        "original_v18k_run":"42128783773-1",
        "exact_original_v18k_proposal_sha256":sha(proposal),
        "exact_independent_v18k_source_court_sha256":sha(court),
        "source_pixels_all_reconciled":len(rows),
        "per_frozen_camera_partition_and_threshold":by_bucket,
        "all_source_error_classes_included_not_only_prior_failures":True,
        "same_geometry_sample_domain_still_not_blender_cycles_32spp":True,
        "no_artist_drawable_identity_promoted":True,
        "no_model_training_targets_promoted":True,
        "graphics_owner_authorized":False,
        "iris_training_authorized":False,
        "tessa_training_authorized":False,
        "materialization_authorized":False,
        "training_authorized":False,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--proposal",type=Path,required=True)
    p.add_argument("--parent-court",type=Path,required=True)
    p.add_argument("--expected-parent-court-sha",required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    out=audit(a.proposal,a.parent_court,a.expected_parent_court_sha)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({
        "status":out["status"],
        "reconciled_source_pixels":out["source_pixels_all_reconciled"],
        "per_frozen_partition":out["per_frozen_camera_partition_and_threshold"],
        "training_authorized":False,
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
