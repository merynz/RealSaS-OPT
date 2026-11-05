#!/usr/bin/env python3
"""Independent V9 source-native animated RGBA intervention consequence court.

A missing native original-vs-soup render parity disqualifies a pose/view from
group consequence comparison. FINEST geometry is not artist truth. No model,
graphics owner, CAA or mechanical authority can be minted by this court.
"""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
import numpy as np
from tools.corpus.corpus370_partition_edit_plan_v7 import build_plan,arm_names

SCHEMA="RealSaS.Corpus370NativeAnimatedPartitionRasterCourt.v9"
RENDER="RealSaS.Corpus370NativeAnimatedPartitionSourceRender.v9"
PARITY_P99_TOL=.025
NOOP_MAX_TOL=.0025
EDIT_TOL=.025


def require(ok,reason):
    if not ok:raise ValueError(reason)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def signal(a,b):
    alpha=np.maximum(a[...,3],b[...,3])
    color=np.max(np.abs(a[...,:3]-b[...,:3]),axis=-1)
    return np.maximum(np.abs(a[...,3]-b[...,3]),color*(alpha>1e-4))


def evaluate(pixels,manifest,proposal,pairs,teacher_receipt):
    plan=build_plan(proposal,pairs)
    require(manifest.get("schema")==RENDER
            and manifest.get("status")==
            "REAL_AUTHORED_NATIVE_POSE_ANIMATED_RGBA_RENDER__INDEPENDENT_COURT_REQUIRED"
            and manifest.get("source_sha256")==proposal["source_sha256"]==
                teacher_receipt["source"]["sha256"]
            and manifest.get("native_teacher_arrays_sha256")==
                teacher_receipt["output"]["sha256"]
            and manifest.get("source_group_proposal_sha256") is not None
            and manifest.get("source_selected_pose_exact_teacher_replay_verified") is True
            and manifest.get("source_rest_exact_geometry_verified") is True
            and manifest.get("reference_policy")==
                "FINEST_AVAILABLE_GEOMETRY_HYPOTHESIS_NOT_AUTHOR_TRUTH"
            and manifest.get("teacher_only") is True
            and manifest.get("graphics_owner_authorized") is False
            and manifest.get("training_authorized") is False
            and manifest.get("animated_render_caa_authorized") is False
            and manifest.get("materialization_authorized") is False
            and manifest.get("source_bone_names_as_targets") is False
            and manifest.get("plan")==plan
            and manifest.get("arms")==arm_names(plan)
            and manifest.get("candidate_sizes")==plan["source_geometry_candidate_sizes"]
            and manifest.get("view_indices")==[0,2,4,6]
            and manifest.get("raster_resolution")==192
            and manifest.get("cycles_samples")==32,
            "V9_INDEPENDENT_COURT_SOURCE_NATIVE_PLAN_AUTHORITY")
    indices=manifest.get("pose_indices")
    details=manifest.get("selected_pose_details_teacher_only")
    sample_frames=teacher_receipt.get("sampled_frames")
    require(isinstance(indices,list) and len(indices)==3
            and all(type(x) is int and 0<=x<len(sample_frames) for x in indices)
            and len(set(indices))==3
            and isinstance(details,list) and len(details)==3
            and [x.get("pose_index") for x in details]==indices
            and all(x.get("source_native_exact_face_order") is True
                    and x.get("max_original_evaluated_vs_sealed_teacher_coordinate_error",1.)<=1e-4
                    and x.get("source_frame_native_teacher_only")==sample_frames[i]["frame"]
                    for x,i in zip(details,indices)),
            "V9_INVALID_OR_FORGED_SOURCE_NATIVE_FRAME_REPLAY")
    arms=arm_names(plan)
    p=np.asarray(pixels)
    require(p.shape==(3,4,len(arms),192,192,4)
            and p.dtype.kind=="f" and np.isfinite(p).all()
            and np.all(p[...,3]>=-1e-5)
            and np.all(p[...,3]<=1.+1e-5),
            "V9_RGBA_PRIVATE_REAL_SOURCE_ARRAY_SHAPE")
    index={name:i for i,name in enumerate(arms)}
    parities=[]
    for pose_ord,pose_index in enumerate(indices):
        for view_ord,view_index in enumerate(manifest["view_indices"]):
            src=p[pose_ord,view_ord,index["SOURCE_ORIGINAL"]]
            full=p[pose_ord,view_ord,index["SOUP_FULL"]]
            noop=p[pose_ord,view_ord,index["SOUP_NOOP"]]
            p99=float(np.quantile(signal(src,full),.99))
            noop_max=float(signal(full,noop).max())
            occupied=int(np.count_nonzero(src[...,3]>.05))
            qualified=occupied>0 and p99<=PARITY_P99_TOL and noop_max<=NOOP_MAX_TOL
            parities.append({
                "pose_index":pose_index,
                "view_index":view_index,
                "source_occupied_pixel_count":occupied,
                "source_vs_soup_rgba_p99":p99,
                "soup_vs_noop_rgba_max":noop_max,
                "qualified_for_edit_consequence":qualified,
                "parity_failure_reason":(
                    "EMPTY_ANIMATED_CAMERA_OCCUPANCY" if occupied==0
                    else "SOURCE_VS_SOUP_PARITY" if p99>PARITY_P99_TOL
                    else "NONZERO_NOOP" if noop_max>NOOP_MAX_TOL
                    else None),
            })
    qualified={(i,j) for i in range(3) for j in range(4)
               if parities[i*4+j]["qualified_for_edit_consequence"]}
    aggregates={}
    task_rows=[]
    for task in plan["tasks"]:
        tid=task["task_index"]
        fine_remove=index[f"FINE_REMOVE_T{tid:02d}"]
        fine_isolate=index[f"FINE_ISOLATE_T{tid:02d}"]
        contributions={}
        for size in plan["source_geometry_candidate_sizes"]:
            removal=index[f"GROUP_REMOVE_K{size:03d}_T{tid:02d}"]
            isolate=index[f"GROUP_ISOLATE_K{size:03d}_T{tid:02d}"]
            additional_direct=additional_isolate=0
            candidate_effect=reference_effect=0
            for t,v in sorted(qualified):
                full=p[t,v,index["SOUP_FULL"]]
                candidate=p[t,v,removal]
                fine=p[t,v,fine_remove]
                iso=p[t,v,isolate]
                fine_iso=p[t,v,fine_isolate]
                candidate_effect+=int(np.count_nonzero(
                    signal(full,candidate)>EDIT_TOL))
                reference_effect+=int(np.count_nonzero(
                    signal(full,fine)>EDIT_TOL))
                additional_direct+=int(np.count_nonzero(
                    signal(candidate,fine)>EDIT_TOL))
                additional_isolate+=int(np.count_nonzero(
                    signal(iso,fine_iso)>EDIT_TOL))
            if size==plan["finest_available_reference_size"]:
                require(additional_direct==0 and additional_isolate==0,
                        "V9_FINE_REFERENCE_SELF_PARITY")
            contributions[str(size)]={
                "source_anonymous_group_count":size,
                "qualified_pose_views":len(qualified),
                "reference_edit_pixel_instances":reference_effect,
                "candidate_edit_pixel_instances":candidate_effect,
                "additional_direct_pixel_instances_vs_fine_proxy":additional_direct,
                "additional_isolate_pixel_instances_vs_fine_proxy":additional_isolate,
                "reference_is_artist_drawable_truth":False,
                "semantic_label_learned":False,
                "same_drawable":"UNKNOWN",
                "different_drawable":"UNKNOWN",
            }
        task_rows.append({
            "source_face_proxy_task_index":tid,
            "source_face_id_teacher_only":task["source_face_teacher_only"],
            "by_candidate_size":contributions,
        })
    for size in plan["source_geometry_candidate_sizes"]:
        rows=[r["by_candidate_size"][str(size)] for r in task_rows]
        aggregates[str(size)]={
            "candidate_count":size,
            "qualified_pose_view_count":len(qualified),
            "tasks":len(rows),
            "additional_direct_pixel_instances_vs_fine_proxy":sum(
                x["additional_direct_pixel_instances_vs_fine_proxy"] for x in rows),
            "additional_isolate_pixel_instances_vs_fine_proxy":sum(
                x["additional_isolate_pixel_instances_vs_fine_proxy"] for x in rows),
            "optimal_partition_minted":False,
        }
    return {
        "schema":SCHEMA,
        "status":("MEASURED_NATIVE_ANIMATED_RGBA_PARITY_AND_EDIT_PROXY__NO_DRAWABLE"
                  if len(qualified)==12 else
                  "MEASURED_NATIVE_ANIMATED_RGBA_PARTIAL_OR_FAIL_PARITY__EDIT_AUTHORITY_BLOCKED"),
        "subject_id":manifest["subject_id"],
        "source_sha256":manifest["source_sha256"],
        "real_native_pose_count":3,
        "real_native_view_count":4,
        "total_pose_view_census":12,
        "source_soup_parity_qualified_pose_views":len(qualified),
        "unqualified_pose_views":12-len(qualified),
        "all_pose_view_source_parity":len(qualified)==12,
        "parity_census":parities,
        "available_group_sizes":plan["source_geometry_candidate_sizes"],
        "unavailable_group_sizes":plan["unavailable_requested_sizes"],
        "source_face_proxy_task_count":plan["task_count"],
        "candidate_aggregates":aggregates,
        "per_task_private_teacher_only":task_rows,
        "source_pose_rig_names_as_model_targets":False,
        "semantic_class_priors_used_to_build_partitions":False,
        "both_semantically_same_and_different_drawables_can_share_geometry":True,
        "source_geom_only_underdetermined_drawable":True,
        "artist_identity_from_fine_reference":False,
        "artist_identity_from_source_motion":False,
        "same_drawable_positive_pairs":0,
        "different_drawable_negative_pairs":0,
        "animated_native_resolution_caa_authorized":False,
        "graphics_owner_authorized":False,
        "mechanical_causality_authorized":False,
        "optimal_group_count_authorized":False,
        "training_authorized":False,
        "materialization_authorized":False,
    }
