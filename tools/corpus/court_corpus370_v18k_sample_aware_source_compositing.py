#!/usr/bin/env python3
"""V18k independent subprocess source reread + sample composition invariant court.

The independent Actions job reopens exact V14/V18a/V18h/V18j assets and reruns
source-ray samples under the fixed same geometry oracle, then separately checks
composition identities, camera partitions, K1 vacuity and authority gates.
Not an independently implemented second ray-intersector, and NOT artist gold.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.corpus.measure_corpus370_v18k_sample_aware_source_compositing import (
    measure,SCHEMA as PRODUCER,STATUS as SOURCE_STATUS,VIEWS,THRESHOLDS,
    SAMPLE_COUNT,RED,BLUE,TOL,HELDOUT_VIEWS,
)

SCHEMA="RealSaS.Corpus370OriginalSourceSampleAwareCompositingReevaluationCourt.v18k"
STATUS="SOURCE_GEOMETRY_JOINT_SAMPLE_VS_POSTFILTER_REMEASURED__DRAWABLE_UNKNOWN"


def require(ok,reason):
    if not ok:raise ValueError(reason)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def court(proposal_path,v14_job,teacher,camera,rest,stain_npz,
          v18h_manifest,v18h_rgba,v18h_court,v18j_court,expected_h,expected_j):
    claimed=load(proposal_path)
    require(claimed.get("schema")==PRODUCER
        and claimed.get("status")==SOURCE_STATUS
        and claimed.get("no_rgba_error_in_pixel_selection") is True
        and claimed.get("selection")==
            "SOURCE_GEOMETRY_CENTROIDS_GROUP_AND_REMAINDER_PLUS_FIXED_NEIGHBORS"
        and claimed.get("camera_locked_evaluation_views")==list(HELDOUT_VIEWS)
        and claimed.get("camera_exploratory_views")==[0,2]
        and claimed.get("geometrical_sample_filter_is_not_exact_cycles") is True
        and claimed.get("k1_perfect_reconstruction_is_vacuous") is True
        and claimed.get("heldout_camera_policy_frozen_before_original_source_measurements") is True
        and claimed.get("artist_drawable_identity")=="UNKNOWN"
        and claimed.get("operational_drawable_identity")=="UNKNOWN"
        and all(claimed.get(k) is False for k in
            ("graphics_owner_authorized","iris_training_authorized",
             "tessa_training_authorized","materialization_authorized",
             "training_authorized")),
        "V18K_SOURCE_PROPOSAL_AUTHORITY_OR_SELECTION_INVALID")
    actual=measure(v14_job,teacher,camera,rest,stain_npz,v18h_manifest,
        v18h_rgba,v18h_court,v18j_court,expected_h,expected_j)
    require(actual==claimed,
            "V18K_SOURCE_GEOMETRY_PIXEL_SELECTION_OR_RENDER_RAYS_NOT_REPRODUCED")
    rows=actual["original_source_pixel_rows_private_teacher_only"]
    all_seen=set()
    summary={}
    for row in rows:
        geometry=row["geometry_subsample_compositor"]
        original=row["actual_original_blender_source_comparison"]
        view=row["source_camera_view"]
        t=float(row["frozen_v18a_threshold"])
        x,y=row["source_pixel_topdown"]
        xx,yy=row["source_blender_framebuffer_pixel_bottomup"]
        require((x,191-y)==(xx,yy)
            and row["source_pose_ordinal"]==2
            and row["geometry_precommitted_selection"] is True
            and row["rgba_error_never_used_for_pixel_selection"] is True
            and row["camera_evaluation_partition"]==(
                "LOCKED_EVALUATION_VIEW" if view in HELDOUT_VIEWS
                else "EXPLORATORY_VIEW")
            and row["source_graphic_drawable_owner"]=="UNKNOWN"
            and row["vacuous_whole_character_group"] is
                (row["source_group_face_count"]==row["whole_source_face_count"])
            and geometry["source_sample_count"]==SAMPLE_COUNT
            and original["no_artist_layer_identity"] is True,
            "V18K_ORIGINAL_SOURCE_PIXEL_CAMERA_AND_UNKNOWN_INVARIANTS")
        key=(view,t,x,y)
        require(key not in all_seen,"V18K_SOURCE_PIXEL_QUERY_DOUBLE_COUNT")
        all_seen.add(key)
        if geometry["geometric_opaque_sample_composition"] is not None:
            g=geometry["geometric_opaque_sample_composition"]
            j=np.asarray(g["joint_pre_filter_premultiplied_rgba"],np.float64)
            a=np.asarray(g["postfilter_group_premultiplied_rgba"],np.float64)
            b=np.asarray(g["postfilter_remainder_premultiplied_rgba"],np.float64)
            ab=np.asarray(g["postfilter_g_over_r_premultiplied_rgba"],np.float64)
            ba=np.asarray(g["postfilter_r_over_g_premultiplied_rgba"],np.float64)
            require(all(z.shape==(4,) and np.isfinite(z).all()
                        and -1e-10<=z[3]<=1+1e-10 for z in (j,a,b,ab,ba))
                and np.allclose(ab,a+b*(1.-a[3]),atol=1e-10,rtol=0)
                and np.allclose(ba,b+a*(1.-b[3]),atol=1e-10,rtol=0)
                and abs(ab[3]-ba[3])<=1e-10
                and abs(float(g["joint_alpha_minus_postfilter_alpha"])-
                        (j[3]-ab[3]))<=1e-10
                and abs(float(g["joint_vs_best_postfilter_order_max_rgba_delta"])-
                    min(float(np.max(np.abs(j-ab))),
                        float(np.max(np.abs(j-ba)))))<=1e-10
                and original["source_joint_sample_error"] is not None
                and original["geometry_postfilter_best_order_error"] is not None,
                "V18K_INDEPENDENT_PREMUL_COMPOSITING_MATH_MISMATCH")
        else:
            require(geometry["ray_unknown_sample_count"]>0
                and original["source_joint_sample_error"] is None,
                "V18K_UNKNOWN_GEOMETRY_IMPROPERLY_TREATED_AS_PASSED")
        bucket=("EVALUATION" if view in HELDOUT_VIEWS else "EXPLORATORY",
                 str(t))
        if bucket not in summary:
            summary[bucket]=dict(
                original_source_pixel_queries=0,
                fully_qualified_geometric_pixels=0,
                geometric_unknown_pixels=0,
                mixed_nearest_geometric_pixels=0,
                geometric_joint_vs_postfilter_difference_gt_025=0,
                measured_source_blender_postfilter_both_orders_fail=0,
                joint_4x4_geometry_closer_than_blender_postfilter=0,
                source_alpha_identity_wrong_under_blender_postfilter=0,
                trivially_whole_character_pixel_queries=0,
                source_ray_subsamples_executed=0,
            )
        stats=summary[bucket]
        stats["original_source_pixel_queries"]+=1
        stats["source_ray_subsamples_executed"]+=SAMPLE_COUNT
        stats["trivially_whole_character_pixel_queries"]+=int(
            row["vacuous_whole_character_group"])
        stats["measured_source_blender_postfilter_both_orders_fail"]+=int(
            original["real_blender_v18h_postfilter_both_orders_fail"])
        stats["source_alpha_identity_wrong_under_blender_postfilter"]+=int(
            original["real_blender_v18h_postfilter_order_invariant_alpha_error"]>TOL)
        if geometry["geometric_opaque_sample_composition"] is None:
            stats["geometric_unknown_pixels"]+=1
        else:
            g=geometry["geometric_opaque_sample_composition"]
            stats["fully_qualified_geometric_pixels"]+=1
            stats["mixed_nearest_geometric_pixels"]+=int(
                g["mixed_source_nearest_G_R_within_pixel"])
            stats["geometric_joint_vs_postfilter_difference_gt_025"]+=int(
                g["joint_vs_best_postfilter_order_max_rgba_delta"]>TOL)
            stats["joint_4x4_geometry_closer_than_blender_postfilter"]+=int(
                original["sample_reconstruction_improves_actual_blender_postfilter"])
    require(set(summary)=={(z,str(t)) for z in ("EVALUATION","EXPLORATORY")
                           for t in THRESHOLDS}
        and sum(z["original_source_pixel_queries"] for z in summary.values())
            ==len(rows)
        and all(v["fully_qualified_geometric_pixels"]+
                v["geometric_unknown_pixels"]==
                v["original_source_pixel_queries"] for v in summary.values())
        and len(all_seen)==len(rows),
        "V18K_INDEPENDENT_PER_VIEW_PIXEL_CENSUS")
    return {
        "schema":SCHEMA,"status":STATUS,
        "original_source_subject_id_private":actual["source_subject_id_private"],
        "original_source_sha256":actual["original_source_sha256"],
        "source_original_v18k_proposal_sha256":sha(proposal_path),
        "original_source_v18h_independent_court_sha256":sha(v18h_court),
        "original_source_v18j_independent_court_sha256":sha(v18j_court),
        "original_source_geometry_rays_remeasured":actual["geometric_original_view_pixel_rays_unique"],
        "total_threshold_view_pixel_query_rows":len(rows),
        "per_threshold_evaluation_partition_metrics":{
            f"{a}:{b}":v for (a,b),v in sorted(summary.items())
        },
        "source_independent_remeasurement_of_4x4_query":"YES_REUSED_FROZEN_GEOMETRIC_RAY_KERNEL",
        "original_artist_drawable_and_source_alpha":"UNKNOWN",
        "source_ray_hit_is_not_artist_alpha":True,
        "whole_character_k1_is_vacuous":True,
        "source_geometry_only_selection":True,
        "not_exact_blender_cycles_sample_pattern":True,
        "sample_compositing_is_not_exact_source_rgb_certification":True,
        "source_visual_heldout_is_camera_reserved_not_independent_unseen_subject":True,
        "source_actual_blender_render_authorized_as_graphics_training_gold":False,
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
    for name in ("proposal","v14_job","teacher","camera","rest","stain_npz",
                 "v18h_manifest","v18h_rgba","v18h_court","v18j_court","output"):
        p.add_argument("--"+name.replace("_","-"),required=True,type=Path)
    for name in ("expected_v18h_court_sha","expected_v18j_court_sha"):
        p.add_argument("--"+name.replace("_","-"),required=True)
    a=p.parse_args()
    out=court(a.proposal,a.v14_job,a.teacher,a.camera,a.rest,a.stain_npz,
        a.v18h_manifest,a.v18h_rgba,a.v18h_court,a.v18j_court,
        a.expected_v18h_court_sha,a.expected_v18j_court_sha)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,indent=2,sort_keys=True,
                                   allow_nan=False)+"\n")
    print(json.dumps({
        "status":out["status"],
        "source":out["original_source_subject_id_private"],
        "per_threshold_evaluation_partition_metrics":
           out["per_threshold_evaluation_partition_metrics"],
        "graphics_owner_authorized":False,
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
