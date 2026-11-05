#!/usr/bin/env python3
"""Corpus370 V17: pose-aware original source surface rays joined to real RGBA.

GEOMETRY precommit: source triangle+barycentric all-hit ray stacks in the
ACTUAL original native authored pose and four AB joint native source poses.
Ray proposals use geometry/camera ONLY; no original pixels/artist part count.
COURT: freeze proposal SHA, independently replay source rays, then read SHA-
bound original V14/V16 RAW RGBA and report exact original pixel effects.

A geometric first hit is NOT a graphics drawable, z-opacity proof or alpha
contributor. A zero RGBA pixel over a hit is UNKNOWN (texture/alpha/culling).
This is a source teacher observation court, not a model training payload.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from models.iris.truth.graphics_oracle_v1 import (
    camera_from_mapping, face_centroid_candidate_pixels_v1,
    measure_orthographic_ray_stack_v1
)
from tools.corpus.court_corpus370_causal_rgba_isolation_relation_v15 import signal

PRED_SCHEMA="RealSaS.Corpus370PoseAwareSourceRayGeometryPrecommit.v17"
COURT_SCHEMA="RealSaS.Corpus370PoseAwareSourceRayOriginalRGBAIndependentCourt.v17"
RES=192
VIEW_INDICES=(0,2,4,6)
SIGN_KEYS=("PLUS_PLUS","PLUS_MINUS","MINUS_PLUS","MINUS_MINUS")
CANDIDATES_PER_POSE_CAMERA=8
TOL=.025


def require(ok,code):
    if not ok:raise ValueError(code)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as fd:
        for chunk in iter(lambda:fd.read(8<<20),b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+"\n")


def ray_contract(ray):
    require(ray.view_index in VIEW_INDICES,"V17_SOURCE_RAY_VIEW")
    out={
        "view_index":int(ray.view_index),
        "pixel_x":int(ray.pixel_x),"pixel_y":int(ray.pixel_y),
        "complete":bool(ray.complete),
        "has_depth_tie":bool(ray.has_depth_tie),
        "tie_epsilon":float(ray.tie_epsilon),
        "hits":[{
            "face_index_teacher_only":int(hit.face_index),
            "source_barycentric_teacher_only":[float(k) for k in hit.barycentric],
            "depth_source_units":float(hit.depth),
        } for hit in ray.hits]
    }
    require(all(0<=h["face_index_teacher_only"] for h in out["hits"]),
        "V17_GEOMETRIC_RAY_ILLEGAL_FACE_ADDRESS")
    require(all(len(h["source_barycentric_teacher_only"])==3
                and abs(sum(h["source_barycentric_teacher_only"])-1)<=1e-5
                for h in out["hits"]),
        "V17_UNNORMALIZED_SOURCE_BARYCENTRIC")
    return out


def projection_camera(bundle,view):
    require(bundle.get("schema")=="RealSaS.CameraProjectionBundle.v1",
        "V17_EXACT_ORIGINAL_CAMERA_SCHEMA")
    rows=bundle.get("cameras") or []
    require(len(rows)==8 and
            set(int(r["view_index"]) for r in rows)==set(range(8)),
        "V17_ORIGINAL_ALL_EIGHT_SOURCE_CAMERA_COMMITMENT")
    camera=dict(next(r for r in rows if int(r["view_index"])==view))
    # V14 rendered the SAME frozen orthographic basis, origin, forward,
    # clip/half_extent with a 192px raster. Convert only resolution; never
    # change source-camera orientation or the original geometric pixel grid.
    camera["resolution"]=RES
    result=camera_from_mapping(camera)
    require(result.view_index==view and result.resolution==RES,
        "V17_SOURCE_CAMERA_RASTER_BINDING")
    return result


def source_geometry(args):
    job=read(args.v14_job)
    render=read(args.v14_manifest)
    v16=read(args.v16_manifest)
    require(job.get("schema")==
                "RealSaS.Corpus370SourceNativeMechanicalRasterJointJob.v14"
        and render.get("schema")==
                "RealSaS.Corpus370SourceNativeMechanicalRasterJointRender.v14"
        and v16.get("schema")==
                "RealSaS.Corpus370OriginalDualNativeControlCausalInteractionRender.v16"
        and job["source"]["sha256"]==render["source_sha256"]==
            v16["source_sha256"]
        and job["subject_id"]==render["subject_id"]==v16["subject_id"]
        and render["private_job_sha256"]==sha(args.v14_job)
        and v16["parent_v14_job_sha256"]==sha(args.v14_job)
        and v16["parent_v14_render_manifest_sha256"]==sha(args.v14_manifest)
        and v16["original_dual_control_rgba_npz_sha256"]==sha(args.v16_npz)
        and v16["original_v13b_control_vertex_array_sha256"]==sha(args.v13b_npz)
        and job["rest_geometry"]["sha256"]==sha(args.rest_npz)
        and job["camera"]["sha256"]==sha(args.camera_bundle)
        and job["v13b_array"]["sha256"]==sha(args.v13b_npz)
        and render["source_native_perturbed_vertices_verified_before_and_after_render"] is True
        and v16["joint_original_source_to_soup_parity_rechecked"] is True
        and v16["private_original_joint_world_vertices_for_independent_court"] is True
        and job["artist_drawable_identity_inferred"] is False
        and job["training_authorized"] is False
        and render["graphics_owner_authorized"] is False
        and v16["graphics_owner_authorized"] is False
        and v16["training_authorized"] is False,
        "V17_FALSE_ORIGINAL_NATIVE_GEOMETRY_OR_SOURCE_PARENT")
    with np.load(args.rest_npz,allow_pickle=False) as n:
        faces=np.asarray(n["faces"],np.int64)
        rest=np.asarray(n["rest_vertices_source_world"],np.float64)
    with np.load(args.v13b_npz,allow_pickle=False) as n:
        originals=np.asarray(n["baseline_vertices"],np.float64)
    with np.load(args.v16_npz,allow_pickle=False) as n:
        require(set(n.files)=={
            "original_native_dual_control_rgba",
            "original_native_dual_control_world_vertices"
        }, "V17_ORIGINAL_V16_PRIVATE_JOINT_GEOMETRY_ARRAYS")
        joints=np.asarray(n["original_native_dual_control_world_vertices"],np.float64)
    require(faces.ndim==2 and faces.shape[1]==3
        and faces.min()>=0 and faces.max()<len(rest)
        and originals.shape==(3,len(rest),3)
        and joints.shape==(3,4,4,len(rest),3)
        and np.isfinite(originals).all() and np.isfinite(joints).all(),
        "V17_ORIGINAL_POSED_AND_JOINT_MESH_TRIANGLE_ADDRESS")
    return job,render,v16,faces,originals,joints,read(args.camera_bundle)


def sample_geometry(faces,originals,joints,bundle):
    rows=[]
    valid=0;all_hits=0;abstain=0;sample_count=0
    for pose in range(3):
        vertices=originals[pose]
        scale=float(np.linalg.norm(np.ptp(vertices,axis=0)))
        require(np.isfinite(scale) and scale>0,"V17_SOURCE_POSE_GEOMETRY_SCALE")
        tie=max(1e-9,scale*1e-7)
        for vi,view in enumerate(VIEW_INDICES):
            camera=projection_camera(bundle,view)
            candidates=face_centroid_candidate_pixels_v1(
                vertices,faces,camera,maximum=CANDIDATES_PER_POSE_CAMERA)
            require(len(candidates)>=1,"V17_SOURCE_POSE_CAMERA_NO_GEOMETRIC_CANDIDATES")
            for px,py in candidates:
                base=ray_contract(measure_orthographic_ray_stack_v1(
                    vertices,faces,camera,int(px),int(py),tie_epsilon=tie
                ))
                inter={}
                for ki,key in enumerate(SIGN_KEYS):
                    inter[key]=ray_contract(measure_orthographic_ray_stack_v1(
                        joints[pose,vi,ki],faces,camera,
                        int(px),int(py),tie_epsilon=tie
                    ))
                nvalid=sum(int(s["complete"] and not s["has_depth_tie"]
                               and len(s["hits"])>0)
                           for s in [base,*inter.values()])
                valid+=nvalid
                sample_count+=5
                all_hits+=sum(len(x["hits"]) for x in [base,*inter.values()])
                abstain+=5-nvalid
                rows.append({
                    "source_pose_ordinal":pose,
                    "source_camera_view":view,
                    "source_pixel_x":int(px),"source_pixel_y":int(py),
                    "baseline_pose_ray":base,
                    "joint_sign_pair_rays":inter,
                    "geometry_selection_read_rgba":False,
                    "original_artist_drawable_id": "UNKNOWN"
                })
    require(len(rows)<=3*4*CANDIDATES_PER_POSE_CAMERA
        and len(rows)>=3*4
        and len({(r["source_pose_ordinal"],r["source_camera_view"],
                  r["source_pixel_x"],r["source_pixel_y"]) for r in rows})==len(rows),
        "V17_SOURCE_SELECTED_RAYS_INCOMPLETE_OR_DUPLICATED")
    return rows,{
        "source_pixel_proposal_count":len(rows),
        "ray_geometry_state_cases":sample_count,
        "ray_complete_untied_hit_cases":valid,
        "ray_incomplete_tied_or_no_hit_abstain_cases":abstain,
        "original_geometric_hit_count_over_all_states":all_hits
    }


def predict(args):
    job,original,v16,faces,base,joints,bundle=source_geometry(args)
    rows,summary=sample_geometry(faces,base,joints,bundle)
    result={
        "schema":PRED_SCHEMA,
        "status":"V17_ORIGINAL_3D_POSE_AWARE_GEOMETRIC_RAY_STACKS_SHA_FROZEN__NO_RGB_OR_OWNER",
        "subject_id":job["subject_id"],
        "source_sha256":job["source"]["sha256"],
        "parent_original_source_v14_job_sha256":sha(args.v14_job),
        "parent_original_source_v14_manifest_sha256":sha(args.v14_manifest),
        "parent_v16_original_joint_render_manifest_sha256":sha(args.v16_manifest),
        "parent_source_rest_geometry_sha256":sha(args.rest_npz),
        "parent_v13b_native_pose_array_sha256":sha(args.v13b_npz),
        "parent_original_camera_bundle_sha256":sha(args.camera_bundle),
        "parent_v16_private_original_joint_npz_sha256":sha(args.v16_npz),
        "source_camera_views":list(VIEW_INDICES),
        "raster_resolution":RES,
        "source_pose_count":3,
        "source_joint_sign_pairs":list(SIGN_KEYS),
        "source_proposals_per_pose_camera_at_most":CANDIDATES_PER_POSE_CAMERA,
        "image_source_pixels_read_for_proposal":False,
        "source_visible_alpha_read_for_proposal":False,
        "source_original_author_drawable_labels_used":False,
        "pose_aware_ray_stack_before_any_rendered_pixel_court":True,
        "alpha_compositing_proven":False,
        "original_artist_same_drawable_labels_qualified":0,
        "original_artist_different_drawable_labels_qualified":0,
        "graphics_owner_authorized":False,
        "training_authorized":False,
        "materialization_authorized":False,
        "source_hit_ray_rows_teacher_only":rows,
        "summary":summary,
    }
    write(args.prediction,result)
    print(json.dumps({"status":result["status"],"subject_id":result["subject_id"],
        "summary":summary,"owner":False},sort_keys=True),flush=True)


def court(args):
    # Prediction already produced and SHA-locked by *previous* independent CLI.
    pred=read(args.prediction)
    require(sha(args.prediction)==args.sealed_prediction_sha256
        and pred.get("schema")==PRED_SCHEMA
        and pred.get("status")==
            "V17_ORIGINAL_3D_POSE_AWARE_GEOMETRIC_RAY_STACKS_SHA_FROZEN__NO_RGB_OR_OWNER"
        and pred.get("image_source_pixels_read_for_proposal") is False
        and pred.get("source_original_author_drawable_labels_used") is False
        and pred.get("alpha_compositing_proven") is False
        and pred.get("graphics_owner_authorized") is False
        and pred.get("training_authorized") is False
        and pred.get("materialization_authorized") is False
        and pred.get("original_artist_same_drawable_labels_qualified")==0
        and pred.get("original_artist_different_drawable_labels_qualified")==0,
        "V17_GEOMETRY_ONLY_PREDICTION_SEAL_OR_AUTHORITY")
    job,parent,v16,faces,base,joints,bundle=source_geometry(args)
    require(pred["subject_id"]==job["subject_id"]
        and pred["source_sha256"]==job["source"]["sha256"]
        and pred["parent_original_source_v14_job_sha256"]==sha(args.v14_job)
        and pred["parent_original_source_v14_manifest_sha256"]==sha(args.v14_manifest)
        and pred["parent_v16_original_joint_render_manifest_sha256"]==sha(args.v16_manifest)
        and pred["parent_source_rest_geometry_sha256"]==sha(args.rest_npz)
        and pred["parent_v13b_native_pose_array_sha256"]==sha(args.v13b_npz)
        and pred["parent_original_camera_bundle_sha256"]==sha(args.camera_bundle)
        and pred["parent_v16_private_original_joint_npz_sha256"]==sha(args.v16_npz),
        "V17_FROZEN_RAY_GEOMETRIC_PARENT_DRIFT")
    newrows,newsummary=sample_geometry(faces,base,joints,bundle)
    require(pred["source_hit_ray_rows_teacher_only"]==newrows
        and pred["summary"]==newsummary,
        "V17_RAY_GEOMETRY_OR_BLIND_SOURCE_SELECTION_FORGERY")
    public=read(args.canonical_v16)
    require(public.get("schema")==
        "RealSaS.Corpus370V16OriginalDualSourceControlIndependentCausalRGBARealResult.v1"
        and public["real_hosted_run_id"]==42083242419
        and public["real_workflow_conclusion"]=="success"
        and public["source_graphics_owner_authorized"] is False
        and public["training_authorized"] is False
        and public["materialization_authorized"] is False,
        "V17_REAL_V16_CANONICAL_FAMILY_PARENT")
    p=next((s for s in public["source_results"]
            if s["subject_id"]==job["subject_id"]),None)
    require(p is not None and sha(args.v16_court)==p["private_v16_court_sha256"]
        and p["original_v14_independent_court_sha256"]==
            sha(args.v14_court),
        "V17_SAME_ORIGINAL_SOURCE_PUBLIC_PRIVATE_COURT_SHA")
    v16court=read(args.v16_court)
    require(v16court.get("schema")==
        "RealSaS.Corpus370IndependentOriginalDualNativeControlRGBAInteractionCourt.v16"
        and v16court.get("status")==
        "V16_REAL_JOINT_NATIVE_CONTROLS_CAUSAL_NONADDITIVITY_MEASURED__AUTHOR_OWNER_UNKNOWN"
        and v16court["source_v14_parent_rgba_sha256"]==sha(args.v14_rgba)
        and v16court["source_v16_joint_RGBA_npz_sha256"]==sha(args.v16_npz)
        and v16court["source_v16_private_blender_manifest_sha256"]==sha(args.v16_manifest)
        and v16court["independent_original_dual_control_world_vertex_and_v13b_single_mechanics_verified_cases"]==48
        and v16court["independently_replayed_qualifying_source_soup_RGBA_case_count"]==48
        and v16court["same_drawable_truth_authorized"] is False
        and v16court["training_authorized"] is False,
        "V17_NATIVE_V16_ORIGINAL_GRAPHICS_AND_MECHANICAL_COURT_NOT_QUALIFIED")
    with np.load(args.v14_rgba,allow_pickle=False) as d:
        require(d.files==["original_native_actuator_rgba"],
            "V17_V14_REPLAY_ORIGINAL_RGBA_UNEXPECTED_NPZ")
        src=np.asarray(d["original_native_actuator_rgba"],np.float32)
    with np.load(args.v16_npz,allow_pickle=False) as d:
        rgba=np.asarray(d["original_native_dual_control_rgba"],np.float32)
    require(src.shape==(3,4,19,RES,RES,4)
        and rgba.shape==(3,4,14,RES,RES,4)
        and np.isfinite(src).all() and np.isfinite(rgba).all(),
        "V17_PIXEL_RASTER_NOT_192_RGBA_OR_NOT_FINITE")
    from tools.corpus.corpus370_native_raster_join_plan_v14 import arms_for_plan
    old_arms,_=arms_for_plan(parent["selected_source_controls"])
    old_base_index=old_arms.index("ORIGINAL_BASE")
    new_base_index=v16["rendered_arms"].index("ORIGINAL_BASE")
    rows=[]
    counters={
        "source_geometry_sampled_pixels":0,
        "source_baseline_complete_untied_with_hits":0,
        "source_baseline_hit_with_alpha_positive_same_pixel":0,
        "source_baseline_hit_with_alpha_positive_y_mirrored":0,
        "source_baseline_no_geometric_hit_but_alpha_positive":0,
        "joint_original_geometric_ray_cases":0,
        "joint_original_ray_complete_untied_hit_cases":0,
        "original_joint_rgba_effect_positive_pixel_instances":0,
        "original_joint_first_hit_transition_cases":0,
        "original_joint_ray_indeterminate_cases":0
    }
    for r in pred["source_hit_ray_rows_teacher_only"]:
        pi=int(r["source_pose_ordinal"])
        view=int(r["source_camera_view"])
        vi=VIEW_INDICES.index(view)
        px=int(r["source_pixel_x"]);py=int(r["source_pixel_y"])
        oldpixel=src[pi,vi,old_base_index,py,px]
        newpixel=rgba[pi,vi,new_base_index,py,px]
        require(np.max(np.abs(oldpixel-newpixel))<=TOL,
            "V17_V14_AND_V16_SAME_ORIGINAL_BASE_RGBA_PIXEL")
        alpha=float(newpixel[3])
        mirroralpha=float(rgba[pi,vi,new_base_index,RES-1-py,px,3])
        hits=r["baseline_pose_ray"]["hits"]
        geom_ok=(r["baseline_pose_ray"]["complete"] and
                 not r["baseline_pose_ray"]["has_depth_tie"] and bool(hits))
        counters["source_geometry_sampled_pixels"]+=1
        counters["source_baseline_complete_untied_with_hits"]+=int(geom_ok)
        counters["source_baseline_hit_with_alpha_positive_same_pixel"]+=int(
            geom_ok and alpha>.01)
        counters["source_baseline_hit_with_alpha_positive_y_mirrored"]+=int(
            geom_ok and mirroralpha>.01)
        counters["source_baseline_no_geometric_hit_but_alpha_positive"]+=int(
            not hits and alpha>.01)
        baseline_face=hits[0]["face_index_teacher_only"] if geom_ok else None
        joint=[]
        for sign in SIGN_KEYS:
            pairpixel=rgba[
                pi,vi,v16["rendered_arms"].index("SOURCE_AB_"+sign),py,px]
            effect=float(signal(newpixel[None,None,:],pairpixel[None,None,:])[0,0])
            linked=r["joint_sign_pair_rays"][sign]
            joint_ok=linked["complete"] and not linked["has_depth_tie"] and bool(linked["hits"])
            joint_face=linked["hits"][0]["face_index_teacher_only"] if joint_ok else None
            change_face=(baseline_face is not None and joint_face is not None
                         and baseline_face!=joint_face)
            counters["joint_original_geometric_ray_cases"]+=1
            counters["joint_original_ray_complete_untied_hit_cases"]+=int(joint_ok)
            counters["original_joint_rgba_effect_positive_pixel_instances"]+=int(effect>TOL)
            counters["original_joint_first_hit_transition_cases"]+=int(change_face)
            counters["original_joint_ray_indeterminate_cases"]+=int(not joint_ok)
            joint.append({
                "sign":sign,"image_premultiplied_rgba_max_channel_change":effect,
                "visible_rgba_effect_observed":bool(effect>TOL),
                "first_geometric_source_face_transition":bool(change_face),
                "joint_front_face_teacher_only":joint_face,
                "joint_all_geometric_hits":len(linked["hits"]),
                "geometry_rays_complete_untied":bool(joint_ok),
                "gfx_editability_ownership":"UNKNOWN"
            })
        rows.append({
            "native_pose_ordinal":pi,
            "source_camera_view":view,
            "source_pixel_x":px,"source_pixel_y":py,
            "original_source_RGBA_alpha":alpha,
            "original_geometric_first_source_face_teacher_only":baseline_face,
            "original_baseline_ray_complete_untied":bool(geom_ok),
            "source_alpha_zero_does_not_prove_absent_surface":True,
            "joint_counterfactual_cases":joint,
            "artist_source_drawable_identity":"UNKNOWN"
        })
    require(len(rows)==len(pred["source_hit_ray_rows_teacher_only"])
        and counters["joint_original_geometric_ray_cases"]==len(rows)*4
        and counters["source_baseline_complete_untied_with_hits"]>0,
        "V17_NONEMPTY_ORIGINAL_POSED_SOURCE_GEOMETRY_RAY_COVERAGE")
    # Do not elevate potential camera-coordinate disagreement into alpha/order truth.
    observed=(counters["source_baseline_hit_with_alpha_positive_same_pixel"]>0)
    return {
        "schema":COURT_SCHEMA,
        "status":(
            "V17_SOURCE_POSE_AWARE_GEOMETRIC_RAYS_AND_ORIGINAL_RGBA_SAMPLED__OWNER_UNKNOWN"
            if observed else
            "V17_ORIGINAL_CAMERA_OR_ALPHA_ALIGNMENT_UNDETERMINED__OWNER_UNKNOWN"
        ),
        "source":args.source,
        "subject_id":job["subject_id"],
        "source_sha256":job["source"]["sha256"],
        "source_original_v14_job_sha256":sha(args.v14_job),
        "source_original_v16_court_sha256":sha(args.v16_court),
        "source_original_v16_npz_sha256":sha(args.v16_npz),
        "source_geometry_only_original_ray_proposal_sha256":sha(args.prediction),
        "original_pixel_alignment_stats":counters,
        "original_pose_camera_geometry_ray_plus_joint_rgba_effect_cases":rows,
        "original_source_ray_faces_and_barycentric_are_teacher_only":True,
        "camera_bundle_matches_render_on_all_pixels_proven":False,
        "original_native_full_hit_geometry_is_not_alpha_draw_order":True,
        "source_alpha_opacity_per_hit_proven":False,
        "source_graphics_drawable_ownership_proven":False,
        "qualified_same_drawable_pairs":0,
        "qualified_different_drawable_pairs":0,
        "graphics_owner_authorized":False,
        "iris_training_authorized":False,
        "tessa_training_authorized":False,
        "materialization_authorized":False,
        "training_authorized":False
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("predict","court"))
    for name in ("v14_job","v14_manifest","v13b_npz","rest_npz",
                 "camera_bundle","v16_manifest","v16_npz",
                 "prediction"):
        p.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    for name in ("canonical_v16","v16_court","v14_rgba",
                 "v14_court","output"):
        p.add_argument("--"+name.replace("_","-"),type=Path)
    p.add_argument("--sealed-prediction-sha256")
    p.add_argument("--source")
    args=p.parse_args()
    if args.mode=="predict":
        predict(args)
    else:
        require(args.output is not None and args.canonical_v16 is not None
                and args.v16_court is not None and args.v14_rgba is not None
                and args.v14_court is not None
                and args.source in ("CAT","TRICERATOPS","COWFISH")
                and isinstance(args.sealed_prediction_sha256,str)
                and len(args.sealed_prediction_sha256)==64,
                "V17_PRIVATE_INDEPENDENT_COURT_ARGUMENTS_REQUIRED")
        result=court(args)
        write(args.output,result)
        print(json.dumps({"source":args.source,"status":result["status"],
            "pose_aware_pinned_rays":
                result["original_pixel_alignment_stats"]["source_geometry_sampled_pixels"],
            "original_full_source_alpha_hit_alignment":
                result["original_pixel_alignment_stats"]["source_baseline_hit_with_alpha_positive_same_pixel"],
            "joint_rgba_effect_cases":
                result["original_pixel_alignment_stats"]["original_joint_rgba_effect_positive_pixel_instances"],
            "training_authorized":False},sort_keys=True),flush=True)


if __name__=="__main__":
    main()
