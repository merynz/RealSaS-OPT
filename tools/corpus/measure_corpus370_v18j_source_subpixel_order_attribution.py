#!/usr/bin/env python3
"""V18j: source-posed 4x4 geometric ray attribution of V18i raster failures.

Retrospective EXPLANATION ONLY. Outcome-stratified pixel selection (alpha
failure, RGB-only failure, visible pass) must NEVER become blind test, source
artist drawable identity, signed grouping edge, or student training truth.
No Blender render, no semantic labels. The independent court remeasures every
exact original source face stack and recomputes all observed strata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from models.iris.truth.graphics_oracle_v1 import (
    camera_from_mapping, measure_orthographic_ray_stack_v1,
)
from tools.corpus.measure_corpus370_v17b_original_subpixel_coverage_v1 import (
    X_OFFSETS, Y_OFFSETS, offset_camera,
)
from tools.corpus.court_corpus370_v18h_emissive_compositing_witness import (
    THRESHOLDS, VIEWS, arms,
)
from tools.corpus.court_corpus370_v18g_operational_compositing_diagnostics import TOL

SCHEMA="RealSaS.Corpus370OriginalSourceSubpixelOrderAttribution.v18j"
STATUS="ORIGINAL_SOURCE_4X4_SUBPIXEL_RAY_ATTRIBUTION_MEASURED__NO_LAYER_AUTHORITY"
STRATA=("ALPHA_FAIL","RGB_ONLY_FAIL","VISIBLE_PASS")
MAX_PIXELS_PER_VIEW_STRATUM=5
POSE_ORDINAL=2
PIXEL_RESOLUTION=192
SAMPLE_COUNT=len(X_OFFSETS)*len(Y_OFFSETS)
V18I="RealSaS.Corpus370IndependentEmissiveAlphaCoverageDiagnosis.v18i"
V18H="RealSaS.Corpus370OriginalEmissiveComplementRender.v18h"
V14="RealSaS.Corpus370SourceNativeMechanicalRasterJointJob.v14"


def require(ok, reason):
    if not ok:raise ValueError(reason)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def stratum_masks(full,group,remaining):
    """Read-only class split, *not* layer ground truth."""
    for a in (full,group,remaining):
        require(a.shape==(192,192,4) and np.isfinite(a).all()
            and np.min(a[...,3])>=-1e-5
            and np.max(a[...,3])<=1.00001,"V18J_SOURCE_EMISSIVE_RGBA")
    af=np.clip(full[...,3],0,1)
    ag=np.clip(group[...,3],0,1)
    ar=np.clip(remaining[...,3],0,1)
    aover=ag+ar-ag*ar
    fa=np.concatenate((full[...,:3]*af[...,None],af[...,None]),axis=-1)
    ga=np.concatenate((group[...,:3]*ag[...,None],ag[...,None]),axis=-1)
    ra=np.concatenate((remaining[...,:3]*ar[...,None],ar[...,None]),axis=-1)
    forward=ga+ra*(1-ag[...,None])
    reverse=ra+ga*(1-ar[...,None])
    e1=np.max(np.abs(fa-forward),axis=-1)
    e2=np.max(np.abs(fa-reverse),axis=-1)
    fail=np.minimum(e1,e2)>TOL
    amiss=np.abs(af-aover)>TOL
    return {
        "ALPHA_FAIL":fail & amiss,
        "RGB_ONLY_FAIL":fail & ~amiss,
        "VISIBLE_PASS":~fail & (af>TOL),
    }


def pick_pixels(mask, maximum=MAX_PIXELS_PER_VIEW_STRATUM):
    """Fixed deterministic raster-order, evenly-spaced retrospective sample."""
    require(mask.shape==(192,192) and mask.dtype==np.dtype("bool")
            and type(maximum) is int and maximum>=1,
            "V18J_STRATIFIED_PIXEL_SELECTION_INVALID")
    yx=np.argwhere(mask)
    if len(yx)<=maximum:
        return [(int(y),int(x)) for y,x in yx]
    indexes=np.linspace(0,len(yx)-1,maximum,dtype=np.int64)
    return [(int(yx[k,0]),int(yx[k,1])) for k in indexes]


def ray_probe(vertices,faces,camera,frame_y,x,group_labels,group_id,tie):
    """Return all geometric depths in the exact frozen 4x4 screen-pixel grid."""
    # Blender bpy.data.images pixels are BOTTOM-up. Graphics oracle camera
    # coordinates are TOP-down. V17 preserves this exact framebuffer reversal.
    top_y=PIXEL_RESOLUTION-1-frame_y
    measured=[]
    for dy in Y_OFFSETS:
        for dx in X_OFFSETS:
            displaced=offset_camera(camera,dx,dy)
            stack=measure_orthographic_ray_stack_v1(
                vertices,faces,displaced,int(x),int(top_y),tie_epsilon=tie)
            valid=bool(stack.complete and not stack.has_depth_tie)
            seq=[{
                "source_face_teacher_only":int(hit.face_index),
                "anonymized_candidate_side":"G" if
                    int(group_labels[hit.face_index])==group_id else "R",
                "world_ray_depth":float(hit.depth),
            } for hit in stack.hits]
            sides=[z["anonymized_candidate_side"] for z in seq]
            measured.append({
                "offset_x":float(dx),"offset_y":float(dy),
                "stack_complete":bool(stack.complete),
                "has_depth_tie":bool(stack.has_depth_tie),
                "geometric_hit_count":len(seq),
                "qualified_for_geometric_coverage":valid,
                "depth_sorted_source_hits_private":seq,
                "first_visible_geometric_side":
                   (sides[0] if sides else "NONE") if valid else "UNKNOWN",
                "group_present_any_depth":
                   ("G" in sides) if valid else None,
                "remainder_present_any_depth":
                   ("R" in sides) if valid else None,
                "depth_side_switch_count":
                   sum(a!=b for a,b in zip(sides,sides[1:])) if valid else None,
            })
    require(len(measured)==SAMPLE_COUNT,"V18J_FIXED_SUBSAMPLE_COUNT")
    good=[z for z in measured if z["qualified_for_geometric_coverage"]]
    unknown=len(measured)-len(good)
    both_first={"G","R"}<=set(z["first_visible_geometric_side"] for z in good)
    both_stack=sum(bool(z["group_present_any_depth"] and
                            z["remainder_present_any_depth"]) for z in good)
    if not unknown:
        gp=sum(bool(z["group_present_any_depth"]) for z in good)/SAMPLE_COUNT
        rp=sum(bool(z["remainder_present_any_depth"]) for z in good)/SAMPLE_COUNT
        un=sum(bool(z["geometric_hit_count"]) for z in good)/SAMPLE_COUNT
        post=gp+rp-gp*rp
        fraction={
            "geometric_group_subsample_fraction":gp,
            "geometric_remainder_subsample_fraction":rp,
            "geometric_joint_any_subsample_fraction":un,
            "geometric_postfiltered_over_alpha":post,
            "joint_minus_postfiltered_alpha":un-post,
        }
    else:
        fraction=None
    return {
        "source_framebuffer_pixel_x":int(x),
        "source_framebuffer_pixel_y_bottom_up":int(frame_y),
        "source_geometry_pixel_y_top_down":int(top_y),
        "geometric_subsamples":measured,
        "qualified_subsample_count":len(good),
        "unknown_incomplete_or_tied_subsamples":unknown,
        "qualified_first_group_subsamples":sum(z["first_visible_geometric_side"]=="G" for z in good),
        "qualified_first_remainder_subsamples":sum(z["first_visible_geometric_side"]=="R" for z in good),
        "qualified_nohit_subsamples":sum(z["first_visible_geometric_side"]=="NONE" for z in good),
        "qualified_with_both_sides_in_depth_stack":both_stack,
        "qualified_with_three_or_more_depth_side_runs":sum(
            (z["depth_side_switch_count"] or 0)>=2 for z in good),
        "opposite_nearest_source_sides_across_subpixels":bool(both_first),
        "fully_qualified_4x4_coverage_fractions":fraction,
        "source_graphic_part_identity":"UNKNOWN",
    }


def measure(v14_job_path,teacher_path,camera_path,rest_path,stain_path,
            render_manifest_path,rgba_path,v18i_report_path,expected_v18i_sha):
    require(isinstance(expected_v18i_sha,str) and len(expected_v18i_sha)==64
            and sha(v18i_report_path)==expected_v18i_sha,
            "V18J_EXACT_V18I_SOURCE_COURT_SHA_REQUIRED")
    job=load(v14_job_path)
    manifest=load(render_manifest_path)
    parent=load(v18i_report_path)
    camera_bundle=load(camera_path)
    require(job.get("schema")==V14 and job.get("teacher_only") is True
            and job.get("training_authorized") is False
            and job.get("graphics_owner_authorized") is False
            and job.get("view_indices")==list(VIEWS)
            and len(job.get("pose_indices",[]))==3
            and job.get("raster_resolution")==192
            and all(sha(path)==job[ref]["sha256"] for path,ref in (
                (teacher_path,"native_teacher"),(camera_path,"camera"),
                (rest_path,"rest_geometry")))
            and manifest.get("schema")==V18H
            and manifest.get("source_edit_arms")==arms()
            and manifest.get("subject_id")==job["subject_id"]
            and manifest.get("source_original_v14_job_sha256")==sha(v14_job_path)
            and manifest.get("original_source_teacher_vertices_sha256")==sha(teacher_path)
            and manifest.get("original_source_camera_sha256")==sha(camera_path)
            and manifest.get("source_reactive_stain_npz_sha256")==sha(stain_path)
            and manifest.get("new_source_actual_edited_rgba_npz_sha256")==sha(rgba_path)
            and manifest.get("original_source_sha256")==job["source"]["sha256"]
            and manifest.get("heldout_original_authored_pose_ordinal")==POSE_ORDINAL
            and manifest.get("physical_face_visibility_operator")==
                "ACTUAL_TRIANGLE_REMOVAL_AND_OBJECT_RENDER_HIDE"
            and manifest.get("source_material_policy")==
                "FIXED_TWO_COLOR_OPAQUE_EMISSION_NO_MATERIAL_SHADING"
            and parent.get("schema")==V18I
            and parent.get("status")==
                "SOURCE_EMISSIVE_ALPHA_COVERAGE_VS_ORDER_DIAGNOSED__NO_DRAWABLE_TRUTH"
            and parent.get("source_subject_private")==job["subject_id"]
            and parent.get("original_source_sha256")==job["source"]["sha256"]
            and parent.get("parent_v18h_manifest_sha256")==sha(render_manifest_path)
            and parent.get("parent_v18h_render_sha256")==sha(rgba_path)
            and parent.get("original_v18h_run")=="42120479356-1"
            and parent.get("new_blender_renders")==0
            and set(parent.get("per_threshold",{}))==set(str(t) for t in THRESHOLDS)
            and len(parent.get("per_view_threshold",[]))==12
            and all(doc.get(k) is False for doc in (job,manifest,parent)
               for k in ("graphics_owner_authorized","training_authorized"))
            and all(parent.get(k) is False for k in (
              "iris_training_authorized","tessa_training_authorized",
              "materialization_authorized","anti_aliasing_mechanism_proved")),
            "V18J_PARENT_ORIGINAL_SOURCE_BINDING_INVALID")
    with np.load(teacher_path,allow_pickle=False) as z:
        faces=np.asarray(z["faces"],np.int64)
        vertices_all=np.asarray(z["posed_vertices_source_world"],np.float64)
    with np.load(rest_path,allow_pickle=False) as z:
        rest_faces=np.asarray(z["faces"],np.int64)
        rest_vertices=np.asarray(z["rest_vertices_source_world"],np.float64)
    with np.load(stain_path,allow_pickle=False) as z:
        require(set(z.files)=={"faces","stain_channels","graph_edges",
            "selected_source_control_axes_teacher_only"}|
            {"labels_"+str(t).replace(".","p") for t in THRESHOLDS},
            "V18J_V18A_ORIGINAL_ARRAY_AUTHORITY")
        stain_faces=np.asarray(z["faces"],np.int64)
        labels={t:np.asarray(z["labels_"+str(t).replace(".","p")],np.int32)
                for t in THRESHOLDS}
    with np.load(rgba_path,allow_pickle=False) as z:
        require(z.files==["emissive_complement_rgba"],
                "V18J_SOURCE_EMISSIVE_RGBA_MEMBER")
        frames=np.asarray(z["emissive_complement_rgba"],np.float32)
    require(frames.shape==(4,10,192,192,4)
        and np.array_equal(faces,rest_faces)
        and np.array_equal(faces,stain_faces)
        and vertices_all.ndim==3 and vertices_all.shape[1:]==rest_vertices.shape
        and len(vertices_all)>max(job["pose_indices"])
        and faces.min()>=0 and faces.max()<len(rest_vertices)
        and np.isfinite(vertices_all).all()
        and all(v.shape==(len(faces),) and np.all(v>=0)
                for v in labels.values())
        and camera_bundle.get("schema")=="RealSaS.CameraProjectionBundle.v1"
        and len(camera_bundle.get("cameras",[]))==8,
        "V18J_SOURCE_FACE_CAMERA_POSE_ARRAY_BINDING")
    cameras={int(row["view_index"]):camera_from_mapping(
        dict(row,resolution=192)) for row in camera_bundle["cameras"]}
    require(set(cameras)==set(range(8)),"V18J_FROZEN_EIGHT_CAMERA_SET")
    # This is the same *original source pose2* used by V18h's Blender worker,
    # not V16's signed A+B perturbation state.
    xyz=vertices_all[job["pose_indices"][POSE_ORDINAL]]
    extent=max(1e-12,float(np.linalg.norm(np.ptp(xyz,axis=0))))
    tie=max(1e-9,1e-7*extent)
    original_face=int(manifest["source_face_probe_target_ordinals_teacher_only"][0])
    require(0<=original_face<len(faces),"V18J_FROZEN_GROUP_ANCHOR_FACE")
    expected={(int(r["view"]),float(r["threshold"])):r
              for r in parent["per_view_threshold"]}
    require(len(expected)==12,"V18J_V18I_VIEW_THRESHOLD_DUPLICATE")
    indexes={name:i for i,name in enumerate(arms())}
    output=[]
    total={}
    for t in THRESHOLDS:
        group=int(labels[t][original_face])
        group_size=int(np.count_nonzero(labels[t]==group))
        for vi,view in enumerate(VIEWS):
            key=str(t).replace(".","p")
            full=frames[vi,indexes[f"EMISSIVE_{key}_FULL"]]
            iso=frames[vi,indexes[f"EMISSIVE_{key}_ISOLATE"]]
            rem=frames[vi,indexes[f"EMISSIVE_{key}_REMOVE"]]
            mask=stratum_masks(full,iso,rem)
            evidence=expected[(view,float(t))]
            require(evidence["original_source_group_face_count"]==group_size
                and int(evidence["observations"]["order_invariant_OVER_alpha_mismatch_pixels"])==
                    int(mask["ALPHA_FAIL"].sum())
                and int(evidence["observations"]["per_pixel_oracle_fail_without_alpha_mismatch_pixels"])==
                    int(mask["RGB_ONLY_FAIL"].sum()),
                "V18J_V18I_SHA_REMEASURED_PIXEL_CLASSES_CHANGED")
            for kind in STRATA:
                picks=pick_pixels(mask[kind])
                for frame_y,x in picks:
                    ray=ray_probe(xyz,faces,cameras[view],frame_y,x,labels[t],group,tie)
                    output.append({
                        "source_camera_view":int(view),
                        "frozen_threshold":float(t),
                        "source_group_index_teacher_only":group,
                        "group_source_face_count":group_size,
                        "original_full_source_group_count":int(np.unique(labels[t]).size),
                        "retrospective_pixel_stratum":kind,
                        "complete_stratum_population":int(mask[kind].sum()),
                        "source_ray_evidence_private":ray,
                        "outcome_selected_not_blind_training":True,
                    })
                total[f"{view}:{t}:{kind}"]={
                    "population":int(mask[kind].sum()),
                    "selected":len(picks),
                    "complete_subpixel_source_rays_executed":len(picks)*SAMPLE_COUNT,
                    "selection_status":"OUTCOME_SELECTED_RETROSPECTIVE_NOT_HELDOUT",
                }
    require(sum(z["selected"] for z in total.values())==len(output)
        and sum(z["complete_subpixel_source_rays_executed"] for z in total.values())
            ==len(output)*SAMPLE_COUNT,"V18J_UNEXPECTED_RAY_QUERY_COUNT")
    return {
        "schema":SCHEMA,"status":STATUS,
        "original_source_subject_private":job["subject_id"],
        "original_source_sha256":job["source"]["sha256"],
        "parent_v14_job_sha256":sha(v14_job_path),
        "parent_v14_teacher_sha256":sha(teacher_path),
        "parent_v14_camera_sha256":sha(camera_path),
        "parent_v14_rest_sha256":sha(rest_path),
        "parent_v18a_stain_npz_sha256":sha(stain_path),
        "parent_v18h_manifest_sha256":sha(render_manifest_path),
        "parent_v18h_rgba_sha256":sha(rgba_path),
        "parent_v18i_independent_court_sha256":sha(v18i_report_path),
        "pose_ordinal":POSE_ORDINAL,
        "source_authored_pose_index":int(job["pose_indices"][POSE_ORDINAL]),
        "source_camera_views":list(VIEWS),
        "pixel_coordinate_convention":"BLENDER_BOTTOM_UP_TO_GRAPHICS_ORACLE_TOP_DOWN_191_MINUS_Y",
        "strata":list(STRATA),
        "max_per_source_view_threshold_stratum":MAX_PIXELS_PER_VIEW_STRATUM,
        "subsample_offsets_x":list(X_OFFSETS),
        "subsample_offsets_y":list(Y_OFFSETS),
        "tie_epsilon_world":tie,
        "original_source_faces":len(faces),
        "source_rays_issued":len(output)*SAMPLE_COUNT,
        "retrospective_stratum_summary":total,
        "source_private_pixel_and_ray_rows":output,
        "source_pixel_outcome_selected":True,
        "fixed_grid_is_not_blender_cycles_sample_pattern":True,
        "geometric_hit_is_not_source_alpha_truth":True,
        "sample_geometry_is_not_drawable_ownership":True,
        "unknown_ties_or_incomplete_rays_not_forced_negative":True,
        "art_original_layer_same_different":"UNKNOWN",
        "graphics_owner_authorized":False,
        "iris_training_authorized":False,
        "tessa_training_authorized":False,
        "materialization_authorized":False,
        "training_authorized":False,
    }


def main():
    p=argparse.ArgumentParser()
    for name in ("v14_job","teacher","camera","rest","stain_npz",
                 "v18h_manifest","v18h_rgba","v18i_court","output"):
        p.add_argument("--"+name.replace("_","-"),required=True,type=Path)
    p.add_argument("--expected-v18i-sha",required=True)
    a=p.parse_args()
    out=measure(a.v14_job,a.teacher,a.camera,a.rest,a.stain_npz,
                a.v18h_manifest,a.v18h_rgba,a.v18i_court,a.expected_v18i_sha)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({
        "status":out["status"],"original_source":out["original_source_subject_private"],
        "source_rays_issued":out["source_rays_issued"],
        "retrospective_stratum_summary":out["retrospective_stratum_summary"],
        "graphics_owner_authorized":False,
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
