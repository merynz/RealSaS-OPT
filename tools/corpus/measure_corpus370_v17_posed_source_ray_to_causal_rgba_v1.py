#!/usr/bin/env python3
"""V17: pose-aware all-hit source ray stacks joined with real V16 causal RGBA.

This consumes exact V14/V16 private original-source bytes, makes *geometry-only*
pixel proposals separately for EACH actual original V16 simultaneous A+B pose,
and queries the frozen original camera. Every hit has original source triangle
address, barycentric coordinate and ordered depth. Exact SAME pixel's original
source / A / B / real A+B RGBA is measured independently. This is *not* an
alpha-composition, draw-order, original artist-layer or training target court.

No original rig, human part name, fixed K, semantic label, original artist
attachment count, or private source image is published into Git.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np

from models.iris.truth.graphics_oracle_v1 import (
    camera_from_mapping, face_centroid_candidate_pixels_v1,
    measure_orthographic_ray_stack_v1,
)
from tools.corpus.court_corpus370_causal_rgba_isolation_relation_v15 import (
    verify_parents as verify_v14,
)
from tools.corpus.corpus370_native_raster_join_plan_v14 import arms_for_plan

SCHEMA="RealSaS.Corpus370OriginalPosedSourceRayToCausalRGBAJoinProposal.v17"
SOURCE_STATUS="V17_REAL_SOURCE_POSED_GEOMETRIC_RAYS_AND_CAUSAL_RGBA_OBSERVED__OWNER_UNKNOWN"
V16_SCHEMA="RealSaS.Corpus370V16OriginalDualSourceControlIndependentCausalRGBARealResult.v1"
V16_STATUS="REAL_THREE_NATIVE_SOURCE_DUAL_CONTROL_MECHANICAL_RGBA_INTERACTION_3_OF_3_PASS__ARTIST_OWNER_UNKNOWN"
V16_COURT="RealSaS.Corpus370IndependentOriginalDualNativeControlRGBAInteractionCourt.v16"
V16_RENDER="RealSaS.Corpus370OriginalDualNativeControlCausalInteractionRender.v16"
SIGNS=("PLUS_PLUS","PLUS_MINUS","MINUS_PLUS","MINUS_MINUS")
VIEWS=(0,2,4,6)
SAMPLES_PER_CASE=12
TOL=.025


def require(ok,reason):
    if not ok: raise ValueError(reason)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(8<<20),b""):h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def premul(image):
    p=np.asarray(image,dtype=np.float32)
    require(p.shape[-1]==4 and np.isfinite(p).all(),"V17_FINITE_RGBA")
    return np.concatenate((p[...,:3]*p[...,3:],p[...,3:]),axis=-1)


def pixel_effect(a,b,threshold=TOL):
    """Original Blender linear RGBA: RGB premultiplied, alpha separate."""
    require(0<threshold<=TOL,"V17_EFFECT_THRESHOLD")
    return np.max(np.abs(premul(a)-premul(b)),axis=-1)>threshold


def validate_and_open(v14_public,v16_public,source,v14_dir,v14_job,
                      v16_manifest,v16_court,v16_pixels):
    p14=load(v14_public)
    p16=load(v16_public)
    v14_paths,source14,manifest14,court14,parent_hashes=verify_v14(
        v14_dir,p14,source)
    require(p16.get("schema")==V16_SCHEMA and p16.get("status")==V16_STATUS
        and p16.get("real_hosted_run_id")==42083242419
        and p16.get("total_original_joint_causal_cases")==144
        and p16.get("source_count")==3
        and p16.get("source_graphics_owner_authorized") is False
        and p16.get("training_authorized") is False
        and p16.get("materialization_authorized") is False,
        "V17_REAL_V16_CANONICAL_WRONG_OR_FORGED_OWNER")
    matches=[x for x in p16["source_results"] if x.get("source")==source]
    require(len(matches)==1,"V17_REAL_ROSTER_SOURCE_MISSING")
    r=matches[0]
    witness=load(v16_court)
    renderer=load(v16_manifest)
    job=load(v14_job)
    require(digest(v16_court)==r["private_v16_court_sha256"]
        and witness.get("schema")==V16_COURT
        and witness.get("status")==
            "V16_REAL_JOINT_NATIVE_CONTROLS_CAUSAL_NONADDITIVITY_MEASURED__AUTHOR_OWNER_UNKNOWN"
        and witness.get("causal_joint_visual_counterfactual_cases")==48
        and witness.get("independent_original_dual_control_world_vertex_and_v13b_single_mechanics_verified_cases")==48
        and witness.get("identical_original_native_A_plus_B_vertices_across_all_camera_views_verified_cases")==48
        and witness.get("independently_replayed_qualifying_source_soup_RGBA_case_count")==48
        and witness.get("original_author_same_drawable_labels")==0
        and witness.get("original_author_different_drawable_labels")==0
        and witness.get("training_authorized") is False
        and witness.get("materialization_authorized") is False
        and witness.get("graphics_owner_authorized") is False,
        "V17_V16_INDEPENDENT_ORIGINAL_PRIVATE_COURT")
    require(renderer.get("schema")==V16_RENDER
        and renderer.get("source_sha256")==source14["source_sha256"]
        and renderer.get("subject_id")==source14["subject_id"]
        and witness["subject_id"]==r["subject_id"]==source14["subject_id"]
        and witness["original_source_sha256"]==r.get("source_sha256",source14["source_sha256"])
        and witness["source_v16_private_blender_manifest_sha256"]==digest(v16_manifest)
        and witness["source_v16_joint_RGBA_npz_sha256"]==digest(v16_pixels)
        and renderer["original_dual_control_rgba_npz_sha256"]==digest(v16_pixels)
        and renderer["parent_v14_render_manifest_sha256"]==parent_hashes["manifest"]
        and renderer["parent_v14_rgba_npz_sha256"]==parent_hashes["pixels"]
        and renderer["parent_v14_job_sha256"]==digest(v14_job)
        and renderer["source_presentation_draw_order_proven"] is False
        and renderer["graphics_owner_authorized"] is False
        and renderer["training_authorized"] is False,
        "V17_SOURCE_RENDERER_SOURCE_OR_V14_V16_PARENT_SHA")
    require(job.get("schema")==
        "RealSaS.Corpus370SourceNativeMechanicalRasterJointJob.v14"
        and job.get("view_indices")==list(VIEWS)
        and job.get("pose_indices")==manifest14["rendered_pose_indices"]
        and job["source"]["sha256"]==source14["source_sha256"]
        and job["camera"]["sha256"]==manifest14["source_camera_bundle_sha256"]
        and job["rest_geometry"]["sha256"]==manifest14["source_rest_geometry_sha256"]
        and digest(job["camera"]["path"])==job["camera"]["sha256"]
        and digest(job["rest_geometry"]["path"])==job["rest_geometry"]["sha256"],
        "V17_CAMERA_REST_SOURCE_AND_REAL_V14_JOB_SHA")
    cam_payload=load(job["camera"]["path"])
    require(cam_payload.get("schema")=="RealSaS.CameraProjectionBundle.v1"
        and len(cam_payload["cameras"])==8
        and {int(x["view_index"]) for x in cam_payload["cameras"]}==set(range(8)),
        "V17_EXACT_EIGHT_SOURCE_CAMERA_BUNDLE")
    with np.load(job["rest_geometry"]["path"],allow_pickle=False) as f:
        faces=np.asarray(f["faces"],np.int64)
        rest=np.asarray(f["rest_vertices_source_world"],np.float64)
    with np.load(v16_pixels,allow_pickle=False) as f:
        require(set(f.files)=={
            "original_native_dual_control_rgba",
            "original_native_dual_control_world_vertices"
        },"V17_EXACT_PRIVATE_V16_PIXEL_AND_WORLD_VERTEX_MEMBERS")
        rgba=np.asarray(f["original_native_dual_control_rgba"],np.float32)
        vertices=np.asarray(f["original_native_dual_control_world_vertices"],np.float64)
    with np.load(v14_paths["pixels"],allow_pickle=False) as f:
        require(f.files==["original_native_actuator_rgba"],
            "V17_EXACT_PRIVATE_V14_SOURCE_RGBA_MEMBERS")
        singles=np.asarray(f["original_native_actuator_rgba"],np.float32)
    old_arms,_=arms_for_plan(manifest14["selected_source_controls"])
    require(old_arms==manifest14["rendered_arms"]
        and singles.shape==(3,4,19,192,192,4)
        and rgba.shape==(3,4,14,192,192,4)
        and vertices.shape==(3,4,4,len(rest),3)
        and np.isfinite(vertices).all() and np.isfinite(rgba).all()
        and np.array_equal(rest.shape, np.asarray((len(rest),3)))
        and len(faces)>0 and faces.min()>=0 and faces.max()<len(rest),
        "V17_REAL_SOURCE_WORLD_VERTEX_FACE_RGBA_DIMENSIONS")
    oldix={key:i for i,key in enumerate(old_arms)}
    newarms=renderer["rendered_arms"]
    require(newarms==["ORIGINAL_BASE","ISOLATED_AB_BASE"]+[
        mode+"_"+sign for sign in SIGNS
        for mode in ("SOURCE_AB","SOUP_AB","ISOLATED_AB")
    ],"V17_ORIGINAL_REAL_V16_ARM_ORDER")
    newix={key:i for i,key in enumerate(newarms)}
    case_lookup={}
    for x in witness["causal_joint_effect_rows"]:
        k=(x["source_pose_ordinal"],x["source_camera_view"],
            x["anonymous_intervention_a_sign"]+"_"+
            x["anonymous_intervention_b_sign"])
        require(k not in case_lookup,"V17_DUPLICATE_V16_INTERVENTION_CASE")
        case_lookup[k]=x
    require(len(case_lookup)==48,"V17_MISSING_V16_48_REAL_INTERVENTION_CASES")
    return (source14,renderer,job,faces,cam_payload,oldix,newix,
            singles,rgba,vertices,case_lookup,
            dict(v14_public_sha256=digest(v14_public),
                 v16_public_sha256=digest(v16_public),
                 v14_job_sha256=digest(v14_job),
                 v14_court_sha256=parent_hashes["court"],
                 v16_court_sha256=digest(v16_court),
                 v16_manifest_sha256=digest(v16_manifest),
                 v16_original_RGBA_and_vertices_sha256=digest(v16_pixels),
                 rest_geometry_sha256=digest(job["rest_geometry"]["path"]),
                 original_camera_sha256=digest(job["camera"]["path"])))


def proposed_rows(v14_public,v16_public,source,v14_dir,v14_job,
                  v16_manifest,v16_court,v16_pixels):
    (identity,manifest,job,faces,cameras,oldix,newix,
     singles,rgba,vertices,ref,parents)=validate_and_open(
         v14_public,v16_public,source,v14_dir,v14_job,
         v16_manifest,v16_court,v16_pixels)
    extent=max(1e-12,float(np.linalg.norm(
        np.ptp(vertices[0,0,0],axis=0))))
    tie=max(1e-9,1e-7*extent)
    camera_rows={int(c["view_index"]):c for c in cameras["cameras"]}
    rows=[]
    stats={
        "original_source_pose_view_pair_cases":0,
        "geometry_selected_rays":0,
        "complete_untied_nonempty_rays":0,
        "multihit_geometric_rays":0,
        "nohit_or_incomplete_or_tied_rays":0,
        "sampled_rgba_joint_effect_pixels":0,
        "sampled_rgba_joint_nonadditive_pixels":0,
        "sampled_full_RGBA_nonzero_alpha_pixels":0,
        "geometric_ray_with_rgba_effect_pixels":0,
        "effect_without_qualified_geometric_ray_pixels":0,
    }
    for pi in range(3):
        for vi,view in enumerate(VIEWS):
            # Product cameras represent top-down pixel coordinates. Blender's
            # readback of image.pixels is bottom-row-first. This conversion is
            # fixed by the two APIs, never tuned to observed source responses.
            cam_row=dict(camera_rows[view])
            cam_row["resolution"]=192
            cam=camera_from_mapping(cam_row)
            base=rgba[pi,vi,newix["ORIGINAL_BASE"]]
            old_base=singles[pi,vi,oldix["ORIGINAL_BASE"]]
            require(float(np.percentile(
                np.max(np.abs(base-old_base),axis=-1),99))<=TOL,
                "V17_SAME_SOURCE_V14_V16_RGBA_BASELINE")
            for si,sign in enumerate(SIGNS):
                xyz=vertices[pi,vi,si]
                # Same original source pose and signed native action is
                # camera-invariant geometry. V16's independent court assures it.
                require(np.max(np.abs(xyz-vertices[pi,0,si]))<=1e-5,
                    "V17_NONINVARIANT_ORIGINAL_SOURCE_POSED_VERTICES")
                ray_pixels=face_centroid_candidate_pixels_v1(
                    xyz,faces,cam,maximum=SAMPLES_PER_CASE)
                require(len(ray_pixels)>0,"V17_NO_GEOMETRIC_PIXEL_PROPOSALS")
                ab=rgba[pi,vi,newix["SOURCE_AB_"+sign]]
                soup=rgba[pi,vi,newix["SOUP_AB_"+sign]]
                require(float(np.percentile(
                    np.max(np.abs(ab-soup),axis=-1),99))<=TOL,
                    "V17_ORIGINAL_JOINT_SOURCE_TO_SOUP_RGBA_PARITY")
                sign_a,sign_b=sign.split("_")
                a=singles[pi,vi,oldix["SOURCE_C00_"+sign_a]]
                b=singles[pi,vi,oldix["SOURCE_C01_"+sign_b]]
                whole=pixel_effect(base,ab)
                nonlinear=(
                    np.max(np.abs(
                        premul(ab)-premul(a)-premul(b)+premul(base)
                    ),axis=-1)>TOL
                )
                independent_case=ref[(pi,view,sign)]
                require(int(whole.sum())==
                        independent_case["joint_ab_effect_pixel_instances"]
                    and int(nonlinear.sum())==
                        independent_case["joint_nonadditive_rgba_response_pixel_instances"],
                    "V17_RECOMPUTE_ORIGINAL_V16_CASE_FULL_RGBA_EFFECT")
                stats["original_source_pose_view_pair_cases"]+=1
                for px,py in ray_pixels:
                    assert 0<=px<192 and 0<=py<192
                    stack=measure_orthographic_ray_stack_v1(
                        xyz,faces,cam,px,py,tie_epsilon=tie)
                    buffer_y=191-py
                    geo=bool(stack.complete and not stack.has_depth_tie
                             and len(stack.hits)>0)
                    effect=bool(whole[buffer_y,px])
                    nonadditive=bool(nonlinear[buffer_y,px])
                    alpha=float(base[buffer_y,px,3])
                    stackhits=[{
                        "face_index_teacher_only":int(h.face_index),
                        "barycentric_teacher_only":[float(z) for z in h.barycentric],
                        "forward_depth_source_world_units":float(h.depth)
                    } for h in stack.hits]
                    rows.append({
                        "source_pose_ordinal":pi,
                        "source_camera_view":view,
                        "native_joint_sign_pair":sign,
                        "ray_image_pixel_topdown_x":int(px),
                        "ray_image_pixel_topdown_y":int(py),
                        "original_blender_framebuffer_y":int(buffer_y),
                        "geometric_stack_complete":bool(stack.complete),
                        "geometric_stack_has_depth_tie":bool(stack.has_depth_tie),
                        "source_geometric_hit_count":len(stackhits),
                        "source_geometric_all_hits_ordered_near_to_far":stackhits,
                        "raster_original_source_alpha":alpha,
                        "real_original_source_joint_ab_visible_rgba_effect":effect,
                        "real_original_source_joint_ab_nonadditive_rgba_effect":nonadditive,
                        "status":("SOURCE_GEOMETRIC_RAY_JOIN_OBSERVED__NOT_ALPHA_OCCLUSION_PROOF"
                                  if geo else "UNKNOWN_INCOMPLETE_TIED_OR_EMPTY_SOURCE_RAY"),
                        "source_material_alpha_per_hit":"UNKNOWN",
                        "actual_artist_drawable_relation":"UNKNOWN",
                        "source_render_order_of_hits_qualified":False,
                    })
                    stats["geometry_selected_rays"]+=1
                    stats["complete_untied_nonempty_rays"]+=int(geo)
                    stats["multihit_geometric_rays"]+=int(geo and len(stackhits)>=2)
                    stats["nohit_or_incomplete_or_tied_rays"]+=int(not geo)
                    stats["sampled_rgba_joint_effect_pixels"]+=int(effect)
                    stats["sampled_rgba_joint_nonadditive_pixels"]+=int(nonadditive)
                    stats["sampled_full_RGBA_nonzero_alpha_pixels"]+=int(alpha>TOL)
                    stats["geometric_ray_with_rgba_effect_pixels"]+=int(geo and effect)
                    stats["effect_without_qualified_geometric_ray_pixels"]+=int(
                        not geo and effect)
    require(stats["original_source_pose_view_pair_cases"]==48
        and stats["geometry_selected_rays"]==len(rows)
        and stats["geometry_selected_rays"]>0
        and stats["complete_untied_nonempty_rays"]>0,
        "V17_REAL_ORIGINAL_POSED_SOURCE_GEOMETRIC_HITS_UNOBSERVABLE")
    return {
        "schema":SCHEMA,
        "status":SOURCE_STATUS,
        "original_source_id":identity["subject_id"],
        "original_source_sha256":identity["source_sha256"],
        "source_label_private_teacher_only":source,
        "source_position_3d": "ORIGINAL_V16_NATIVE_SIMULTANEOUS_SIGNED_CONTROL_WORLD",
        "pixel_proposal_policy":"POSED_SOURCE_TRIANGLE_CENTROID_GEOMETRY_ONLY__NO_RGBA_RESPONSE_SELECTION",
        "original_blender_image_layout":"BOTTOM_ROW_FIRST__CONVERT_TO_TOP_DOWN_FOR_GEOMETRIC_RAY",
        "sample_count_per_pose_view_joint_sign_at_most":SAMPLES_PER_CASE,
        "source_camera_authority":"ORIGINAL_V14_SHA_LOCKED_V5_CALIBRATED_SOURCE_CAMERA",
        "geometric_ray_backend":"IRIS_GRAPHICS_ORACLE_MOLLER_TRUMBORE_FULL_HIT_STACK",
        "parents_sha256":parents,
        "stats":stats,
        "source_ray_rows_private_only":rows,
        "geometric_depth_is_rasterized_order":False,
        "source_alpha_per_hit_qualified":False,
        "source_draw_order_authorized":False,
        "artist_graphics_owner_authorized":False,
        "training_authorized":False,
        "materialization_authorized":False
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True,choices=("CAT","TRICERATOPS","COWFISH"))
    for f in ("v14_public","v16_public","v14_dir","v14_job",
              "v16_manifest","v16_court","v16_pixels","output"):
        p.add_argument("--"+f.replace("_","-"),required=True,type=Path)
    a=p.parse_args()
    out=proposed_rows(a.v14_public,a.v16_public,a.source,
        a.v14_dir,a.v14_job,a.v16_manifest,a.v16_court,a.v16_pixels)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    stage=a.output.with_suffix(a.output.suffix+".partial")
    stage.write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+"\n")
    stage.replace(a.output)
    print(json.dumps({"status":out["status"],"source":a.source,
        "stats":out["stats"],"training_authorized":False},sort_keys=True),flush=True)


if __name__=="__main__":
    main()
