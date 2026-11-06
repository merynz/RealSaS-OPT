#!/usr/bin/env python3
"""Independent re-open / remeasurement of sealed source V18j geometry outcomes.

The court verifies all parent SHA-locked original 3D source/camera/edited RGBA
artifacts, recomputes each retrospective selected pixel and every 4x4 ray
instead of trusting submitted ray statuses. It does NOT create graphics truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tools.corpus.measure_corpus370_v18j_source_subpixel_order_attribution import (
    measure, SCHEMA as PROPOSAL, STATUS as PROPOSAL_STATUS,
    STRATA, SAMPLE_COUNT, MAX_PIXELS_PER_VIEW_STRATUM, VIEWS, THRESHOLDS,
)

SCHEMA="RealSaS.Corpus370IndependentSourceSubpixelAttributionCourt.v18j"
PASS="V18J_ORIGINAL_4X4_SOURCE_RAY_STACKS_INDEPENDENTLY_REMEASURED__OWNER_UNKNOWN"


def require(ok,reason):
    if not ok:raise ValueError(reason)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def audit(proposal_file,v14_job,teacher,camera,rest,stain,v18h_manifest,
          v18h_rgba,v18i_court,expected_v18i_sha):
    claimed=load(proposal_file)
    require(claimed.get("schema")==PROPOSAL
        and claimed.get("status")==PROPOSAL_STATUS
        and claimed.get("strata")==list(STRATA)
        and claimed.get("source_camera_views")==list(VIEWS)
        and claimed.get("max_per_source_view_threshold_stratum")==
             MAX_PIXELS_PER_VIEW_STRATUM
        and claimed.get("source_rays_issued")==
             len(claimed.get("source_private_pixel_and_ray_rows",[]))*SAMPLE_COUNT
        and claimed.get("source_pixel_outcome_selected") is True
        and claimed.get("fixed_grid_is_not_blender_cycles_sample_pattern") is True
        and claimed.get("geometric_hit_is_not_source_alpha_truth") is True
        and claimed.get("sample_geometry_is_not_drawable_ownership") is True
        and claimed.get("unknown_ties_or_incomplete_rays_not_forced_negative") is True
        and claimed.get("art_original_layer_same_different")=="UNKNOWN"
        and all(claimed.get(k) is False for k in (
            "graphics_owner_authorized","iris_training_authorized",
            "tessa_training_authorized","materialization_authorized",
            "training_authorized")),
        "V18J_INDEPENDENT_UNTRUSTED_PROPOSAL_OR_AUTHORITY")
    actual=measure(v14_job,teacher,camera,rest,stain,
                   v18h_manifest,v18h_rgba,v18i_court,expected_v18i_sha)
    # Exact remeasurement of all numerical sample stacks, not just counts.
    # No changed tolerances and no producer-to-court trust delegation.
    require(claimed==actual,
            "V18J_INDEPENDENT_ORIGINAL_ALL_SUBPIXEL_RAYS_NOT_REPRODUCED")
    rows=actual["source_private_pixel_and_ray_rows"]
    require(all(z["retrospective_pixel_stratum"] in STRATA
            and z["outcome_selected_not_blind_training"] is True
            and z["source_ray_evidence_private"]["source_geometry_pixel_y_top_down"]+
                z["source_ray_evidence_private"]["source_framebuffer_pixel_y_bottom_up"]==191
            for z in rows),
            "V18J_INDEPENDENT_RAY_PIXEL_COORDINATE_OR_LABEL_FRAUD")
    unique=len(set((
        z["source_camera_view"],z["frozen_threshold"],
        z["retrospective_pixel_stratum"],
        z["source_ray_evidence_private"]["source_framebuffer_pixel_x"],
        z["source_ray_evidence_private"]["source_framebuffer_pixel_y_bottom_up"],
    ) for z in rows))
    require(unique==len(rows),
            "V18J_INDEPENDENT_DUPLICATED_SOURCE_RAY_STRATA")
    sums={}
    for kind in STRATA:
        chosen=[z for z in rows if z["retrospective_pixel_stratum"]==kind]
        rays=[r for z in chosen for r in z["source_ray_evidence_private"]["geometric_subsamples"]]
        sums[kind]={
            "retrospectively_selected_original_pixels":len(chosen),
            "independently_remeasured_source_rays":len(rays),
            "complete_untied_rays":sum(x["qualified_for_geometric_coverage"] for x in rays),
            "unknown_incomplete_or_depth_tie_rays":sum(
                not x["qualified_for_geometric_coverage"] for x in rays),
            "source_pixels_with_opposite_nearest_geometric_sides":sum(
                z["source_ray_evidence_private"]
                 ["opposite_nearest_source_sides_across_subpixels"] for z in chosen),
            "source_pixels_with_some_interleaved_geometric_ray":sum(
                z["source_ray_evidence_private"]
                 ["qualified_with_three_or_more_depth_side_runs"]>0 for z in chosen),
            "fully_qualified_pixels":sum(
                z["source_ray_evidence_private"]
                 ["fully_qualified_4x4_coverage_fractions"] is not None for z in chosen),
            "coverage_classification_is_not_blender_alpha_gold":True,
        }
    require(sum(x["independently_remeasured_source_rays"] for x in sums.values())
            ==actual["source_rays_issued"],
            "V18J_INDEPENDENT_SOURCE_RAY_CENSUS")
    return {
        "schema":SCHEMA,"status":PASS,
        "original_source_subject_id_private":actual["original_source_subject_private"],
        "original_source_sha256":actual["original_source_sha256"],
        "producer_sha256":sha(proposal_file),
        "source_v14_job_sha256":sha(v14_job),
        "source_v14_teacher_sha256":sha(teacher),
        "source_v14_camera_sha256":sha(camera),
        "source_v14_rest_sha256":sha(rest),
        "source_v18a_stain_sha256":sha(stain),
        "source_v18h_rgba_sha256":sha(v18h_rgba),
        "source_v18i_independent_court_sha256":sha(v18i_court),
        "original_source_sample_grid":"V17B_FIXED_4X4",
        "original_heldout_pose_ordinal":2,
        "source_camera_views":list(VIEWS),
        "frozen_stain_thresholds":list(THRESHOLDS),
        "retrospective_mechanism_evidence":sums,
        "complete_private_source_ray_reread":True,
        "image_error_pixel_selection_is_retrospective_not_heldout":True,
        "predicted_cycles_alpha_from_geometry_qualified":False,
        "candidate_group_editability_qualified":False,
        "source_artist_same_different_part_labels":"UNKNOWN",
        "unknown_is_not_a_negative_edge":True,
        "graphics_owner_authorized":False,
        "iris_training_authorized":False,
        "tessa_training_authorized":False,
        "materialization_authorized":False,
        "training_authorized":False,
    }


def main():
    p=argparse.ArgumentParser()
    for name in ("proposal","v14_job","teacher","camera","rest",
                 "stain_npz","v18h_manifest","v18h_rgba","v18i_court","output"):
        p.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    p.add_argument("--expected-v18i-sha",required=True)
    args=p.parse_args()
    result=audit(args.proposal,args.v14_job,args.teacher,args.camera,args.rest,
                 args.stain_npz,args.v18h_manifest,args.v18h_rgba,
                 args.v18i_court,args.expected_v18i_sha)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,sort_keys=True,indent=2,
                                      allow_nan=False)+"\n")
    print(json.dumps({
        "status":result["status"],
        "per_original_pixel_class":result["retrospective_mechanism_evidence"],
        "training_authorized":False,
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
