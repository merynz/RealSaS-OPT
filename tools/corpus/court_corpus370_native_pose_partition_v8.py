#!/usr/bin/env python3
"""V8: source-native authored-pose correspondence of *anonymous* geometry groups.

This is NOT a render/occlusion/CAA court, author drawable identity, a causal
mechanical intervention, or training admission. The source's face index is
the address and must match the independently court-verified original rest.
"""
from __future__ import annotations

from collections import Counter
import math

import numpy as np

from tools.corpus.corpus370_partition_edit_plan_v7 import build_plan

SCHEMA="RealSaS.Corpus370NativePosePartitionPersistenceCourt.v8"
REST_REL_TOL=1e-6


def require(ok,code):
    if not ok:
        raise ValueError(code)


def _areas(vertices,faces):
    a=vertices[...,faces[:,1],:]-vertices[...,faces[:,0],:]
    b=vertices[...,faces[:,2],:]-vertices[...,faces[:,0],:]
    return np.linalg.norm(np.cross(a,b,axis=-1),axis=-1)*.5


def _rigid_residual(x,y,extent):
    """Kabsch best-fit residual; diagnostic, NOT a source authored rig."""
    x=np.asarray(x,np.float64)
    y=np.asarray(y,np.float64)
    if len(x)<3:
        return float("nan")
    xc=x-x.mean(axis=0)
    yc=y-y.mean(axis=0)
    u,_,vh=np.linalg.svd(xc.T@yc)
    sign=1. if np.linalg.det(u@vh)>=0 else -1.
    m=np.diag([1.,1.,sign])
    rotation=u@m@vh
    residual=np.linalg.norm(xc@rotation-yc,axis=1)
    return float(np.sqrt(np.mean(residual**2))/extent)


def evaluate(original_rest,authored,proposal,pairs,teacher_receipt,teacher_court,
             *,rest_sha256,teacher_sha256,teacher_receipt_sha256,source_sha256):
    plan=build_plan(proposal,pairs)
    source=plan["source_sha256"]
    require(source==source_sha256
            and pairs["parents"]["source_sha256"]==source
            and pairs["parents"]["rest_geometry_sha256"]==rest_sha256
            and pairs["parents"]["authored_teacher_sha256"]==teacher_sha256
            and pairs["parents"]["authored_teacher_receipt_sha256"]==
                teacher_receipt_sha256
            and teacher_receipt["source"]["sha256"]==source
            and teacher_receipt["output"]["sha256"]==teacher_sha256
            and teacher_receipt.get("teacher_only") is True
            and teacher_receipt.get("teacher_rig_skin_motion_allowed_as_inference_input") is False
            and teacher_court.get("source_sha256")==source
            and teacher_court.get("status")==
                "PASS_SOURCE_AUTHORED_TEMPORAL_GEOMETRY_CANDIDATE__CAUSAL_RESPONSE_SEPARATE"
            and teacher_court.get("teacher_arrays",{}).get("sha256")==teacher_sha256
            and teacher_court.get("teacher_manifest",{}).get("sha256")==teacher_receipt_sha256
            and teacher_court.get("training_authorized") is False
            and teacher_court.get("family_authority_minted") is False,
            "V8_SOURCE_NATIVE_AUTHORED_OR_REST_LINEAGE")
    rest=np.asarray(original_rest["rest_vertices_source_world"],dtype=np.float64)
    faces=np.asarray(original_rest["faces"])
    teacher_rest=np.asarray(authored["rest_vertices_source_world"],dtype=np.float64)
    teacher_faces=np.asarray(authored["faces"])
    posed=np.asarray(authored["posed_vertices_source_world"],dtype=np.float64)
    require(rest.ndim==2 and rest.shape[1:]==(3,) and len(rest)>3
            and faces.ndim==2 and faces.shape[1:]==(3,)
            and faces.dtype.kind in "iu"
            and np.all(faces>=0) and np.all(faces<len(rest))
            and len(faces)==plan["source_face_count"]
            and np.all(faces[:,0]!=faces[:,1])
            and np.all(faces[:,1]!=faces[:,2])
            and np.all(faces[:,2]!=faces[:,0]),
            "V8_ORIGINAL_REST_GEOMETRY_FACE_DOMAIN")
    extent=float(np.linalg.norm(np.ptp(rest,axis=0)))
    require(math.isfinite(extent) and extent>1e-10
            and np.isfinite(rest).all()
            and teacher_rest.shape==rest.shape
            and np.isfinite(teacher_rest).all()
            and teacher_faces.shape==faces.shape
            and np.array_equal(teacher_faces,faces)
            and np.max(np.abs(teacher_rest-rest))<=REST_REL_TOL*max(1.,extent),
            "V8_AUTHORED_TEACHER_EXACT_SOURCE_REST_FACE_ORDER")
    frames=teacher_receipt.get("sampled_frames")
    inventory=teacher_receipt.get("native_motion_inventory_exported")
    require(isinstance(frames,list) and isinstance(inventory,list)
            and len(frames)>=2 and len(frames)==len(posed)
            and len(inventory)>=1
            and teacher_receipt.get("sampled_frame_count")==len(frames)
            and teacher_court.get("counts",{}).get("sampled_frame_count")==len(frames)
            and all(type(f.get("frame")) is int
                    and isinstance(f.get("action"),str)
                    and f.get("action") in inventory for f in frames)
            and set(f["action"] for f in frames)==set(inventory)
            and posed.ndim==3 and posed.shape[1:]==rest.shape
            and np.isfinite(posed).all(),
            "V8_NATIVE_ACTION_POSE_TIME_CENSUS")
    ordinal={name:i for i,name in enumerate(inventory)}
    per_action=Counter(x["action"] for x in frames)
    same_take_edges=[
        i for i in range(1,len(frames))
        if frames[i]["action"]==frames[i-1]["action"]
        and frames[i]["frame"]>frames[i-1]["frame"]
    ]
    require(all(frames[i]["frame"]>frames[i-1]["frame"]
                for i in range(1,len(frames))
                if frames[i]["action"]==frames[i-1]["action"])
            and sum(per_action.values())==len(frames),
            "V8_NATIVE_FRAMES_OUT_OF_ORDER")
    original_area=_areas(rest,faces)
    pose_area=_areas(posed,faces)
    require(np.isfinite(original_area).all()
            and np.isfinite(pose_area).all()
            and np.any(original_area>max(1e-15,extent*extent*1e-12)),
            "V8_NATIVE_FACE_AREA_NONFINITE_OR_ALL_ZERO")
    active=original_area>max(1e-15,extent*extent*1e-12)
    collapsed=pose_area<=np.maximum(original_area[None,:]*1e-4,
                                    extent*extent*1e-12)
    displacements=np.linalg.norm(posed-rest[None],axis=-1)/extent
    by_size={}
    for candidate in proposal["candidates"]:
        size=candidate["slot_count"]
        if size not in plan["source_geometry_candidate_sizes"]:
            continue
        labels=np.asarray(candidate["face_slot_teacher_only"],np.int64)
        require(labels.shape==(len(faces),)
                and set(labels.tolist())==set(range(size)),
                "V8_SOURCE_FACE_TO_GROUP_COVERAGE")
        groups=[]
        centroid_tracks=[]
        for group in range(size):
            loc=np.flatnonzero(labels==group)
            require(len(loc)>0,"V8_GROUP_EMPTY")
            vertices=np.unique(faces[loc].reshape(-1))
            rv=rest[vertices]
            pv=posed[:,vertices,:]
            center_ref=rv.mean(axis=0)
            centers=pv.mean(axis=1)
            center_offsets=(centers-center_ref[None,:])/extent
            # Importer-authored node/object/root motion is observation only.
            movement=np.linalg.norm(center_offsets,axis=1)
            residual=[_rigid_residual(rv,pv[t],extent)
                      for t in range(len(posed))]
            rest_area=float(original_area[loc].sum())
            ratio=pose_area[:,loc].sum(axis=1)/max(rest_area,1e-18)
            tracked={
                "anonymous_group_id_teacher_only":group,
                "source_face_count":int(len(loc)),
                "source_vertex_count":int(len(vertices)),
                "rest_surface_area_source_units_squared":rest_area,
                "native_pose_group_face_count_persistent":True,
                "native_pose_group_source_face_ids_persistent":True,
                "any_centroid_motion_over_source_extent":bool(np.max(movement)>1e-8),
                "max_centroid_displacement_over_source_extent":float(np.max(movement)),
                "max_local_best_rigid_fit_rmse_over_source_extent":float(np.max(residual)),
                "median_local_best_rigid_fit_rmse_over_source_extent":float(np.median(residual)),
                "pose_area_ratio_min":float(np.min(ratio)),
                "pose_area_ratio_max":float(np.max(ratio)),
                "collapsed_source_face_pose_instances":int(np.count_nonzero(collapsed[:,loc])),
                "visibility_from_camera_measured":False,
                "animated_draw_order_measured":False,
                "same_drawable":"UNKNOWN",
                "different_drawable":"UNKNOWN",
                "training_authorized":False,
            }
            groups.append(tracked)
            centroid_tracks.append(centers)
        center=np.stack(centroid_tracks,axis=1)
        pairwise_relative=[]
        for i in range(size):
            for j in range(i+1,size):
                d0=float(np.linalg.norm(center[0,i]-center[0,j]))
                cur=np.linalg.norm(center[:,i]-center[:,j],axis=-1)
                pairwise_relative.append(float(np.max(np.abs(cur-d0))/extent))
        # Explicitly observational. Co-motion or relative deformation cannot
        # mint SAME or DIFFERENT drawable ownership.
        by_size[str(size)]={
            "group_count":size,
            "pose_count":len(posed),
            "group_face_domain_unchanged_for_all_poses":True,
            "groups_with_centroid_motion":sum(int(x["any_centroid_motion_over_source_extent"])
                                              for x in groups),
            "max_relative_group_centroid_distance_change_over_source_extent":
                max(pairwise_relative,default=0.),
            "max_group_local_rigid_fit_rmse_over_source_extent":
                max(x["max_local_best_rigid_fit_rmse_over_source_extent"] for x in groups),
            "source_face_pose_collapses":int(np.count_nonzero(collapsed[:,active])),
            "source_face_collapses_are_author_drawable_failures":False,
            "group_observations_teacher_only":groups,
            "stable_group_ids_prove_drawable_owner":False,
            "training_authorized":False,
        }
    require(set(map(int,by_size))==set(plan["source_geometry_candidate_sizes"]),
            "V8_GROUP_PARTITION_CENSUS_MISSING")
    return {
        "schema":SCHEMA,
        "status":"ORIGINAL_SOURCE_NATIVE_POSE_FACE_CORRESPONDENCE_MEASURED__NOT_DRAWABLE_OR_CAA",
        "source_sha256":source_sha256,
        "source_authored_rest_exact_face_order":True,
        "authored_pose_count":int(len(posed)),
        "authored_action_count":len(inventory),
        "sampled_frames_by_action":dict(sorted(per_action.items())),
        "same_take_temporal_edges":len(same_take_edges),
        "source_extent":extent,
        "source_rest_face_count":len(faces),
        "source_rest_vertex_count":len(rest),
        "whole_source_max_vertex_motion_over_extent":float(np.max(displacements)),
        "whole_source_collapsed_face_pose_instances":int(np.count_nonzero(collapsed[:,active])),
        "whole_source_rest_degenerate_face_count":int(np.count_nonzero(~active)),
        "available_candidate_sizes":plan["source_geometry_candidate_sizes"],
        "unavailable_requested_sizes":plan["unavailable_requested_sizes"],
        "by_candidate_size":by_size,
        "source_face_identity_in_pose_domain_verified":True,
        "animated_source_original_vs_group_soup_image_parity_courted":False,
        "camera_visibility_animated_courted":False,
        "native_resolution_caa_authorized":False,
        "graphics_owner_authorized":False,
        "mechanical_causality_authorized":False,
        "same_drawable_positive_pairs":0,
        "different_drawable_negative_pairs":0,
        "training_authorized":False,
        "materialization_authorized":False,
    }
