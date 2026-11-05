#!/usr/bin/env python3
"""V17 independent SHA-bound source pose/ray/pixel case court.

Re-opens the exact original V14 camera, original rest faces and V16 joint source
world-space vertices, reconstructs each ray, barycentric triangle world point,
ray-depth ordering and actual source framebuffer premultiplied RGBA effect.
This is a geometric/source-response observation, never alpha ordering or an
artist DRAWABLE teacher label. Source/private per-ray data stay off GitHub.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from models.iris.truth.graphics_oracle_v1 import (
    camera_from_mapping, face_centroid_candidate_pixels_v1,
    measure_orthographic_ray_stack_v1,
)
from tools.corpus.measure_corpus370_v17_posed_source_ray_to_causal_rgba_v1 import (
    SCHEMA as PROPOSAL_SCHEMA, SOURCE_STATUS, SIGNS, VIEWS,
    TOL, SAMPLES_PER_CASE, require, validate_and_open, premul, pixel_effect
)

SCHEMA="RealSaS.Corpus370IndependentOriginalPosedSourceRayCausalRGBAJoinCourt.v17"
PASS="V17_REAL_POSED_SOURCE_GEOMETRIC_RAY_AND_RGBA_LINK_RECALCULATED__ARTIST_OWNER_UNKNOWN"


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(8<<20),b""):h.update(chunk)
    return h.hexdigest()


def verify_hit_world_address(vertices,faces,camera,px,py,hit,extent):
    """Separate barycentric ray-depth world reconstruction, not trusting labels."""
    face=hit["face_index_teacher_only"]
    bary=hit["barycentric_teacher_only"]
    depth=hit["forward_depth_source_world_units"]
    require(type(face) is int and 0<=face<len(faces)
        and isinstance(bary,list) and len(bary)==3
        and all(type(x) is float and np.isfinite(x)
                and -1e-6<=x<=1.000001 for x in bary)
        and abs(sum(bary)-1.)<=1e-5
        and type(depth) is float and np.isfinite(depth) and depth>0,
        "V17_INVALID_POSED_SOURCE_FACE_BARY_OR_DEPTH")
    baryxyz=np.asarray(bary) @ vertices[faces[face]]
    x=((px+.5)/camera.resolution*2.-1.)*camera.half_extent
    y=(1.-(py+.5)/camera.resolution*2.)*camera.half_extent
    projected=(np.asarray(camera.origin)
        +np.asarray(camera.right)*x
        +np.asarray(camera.screen_up)*y
        +np.asarray(camera.forward)*depth)
    require(np.max(np.abs(projected-baryxyz))<=max(1e-6,extent*2e-6),
        "V17_GEOMETRIC_BARYCENTRIC_SOURCE_RAY_WORLD_POINT_FALSIFIED")


def verify_row_geometry_and_rgba(row,xyz,faces,cam,base,a,b,ab,extent,tie):
    px=row["ray_image_pixel_topdown_x"]
    py=row["ray_image_pixel_topdown_y"]
    require(type(px) is int and type(py) is int
        and 0<=px<cam.resolution and 0<=py<cam.resolution,
        "V17_SOURCE_PIXEL_COORDINATE_OUTSIDE_FRAME")
    by=cam.resolution-1-py
    require(row["original_blender_framebuffer_y"]==by,
        "V17_FALSE_BLENDER_ROW_CONVENTION_OR_SOURCE_IMAGE_POSE")
    ray=measure_orthographic_ray_stack_v1(
        xyz,faces,cam,px,py,tie_epsilon=tie)
    original=row["source_geometric_all_hits_ordered_near_to_far"]
    require(isinstance(original,list) and len(original)==len(ray.hits)
        and row["geometric_stack_complete"]==ray.complete
        and row["geometric_stack_has_depth_tie"]==ray.has_depth_tie
        and row["source_geometric_hit_count"]==len(ray.hits),
        "V17_INDEPENDENT_COMPLETE_GEOMETRIC_HIT_LIST_MISMATCH")
    previous=0.
    for stored,measured in zip(original,ray.hits):
        verify_hit_world_address(xyz,faces,cam,px,py,stored,extent)
        require(stored["face_index_teacher_only"]==measured.face_index
            and np.max(np.abs(
                np.asarray(stored["barycentric_teacher_only"])-
                np.asarray(measured.barycentric)))<=1e-7
            and abs(stored["forward_depth_source_world_units"]-
                measured.depth)<=max(1e-7,extent*1e-7)
            and stored["forward_depth_source_world_units"]>=previous,
            "V17_RAY_HIT_ADDRESS_DEPTH_OR_ORDER_INCONSISTENT")
        previous=stored["forward_depth_source_world_units"]
    eff=bool(pixel_effect(base[by,px],ab[by,px]))
    interaction=bool(np.max(np.abs(
        premul(ab[by,px])-premul(a[by,px])-
        premul(b[by,px])+premul(base[by,px])
    ))>TOL)
    qualified=bool(ray.complete and not ray.has_depth_tie and len(ray.hits)>0)
    require(row["real_original_source_joint_ab_visible_rgba_effect"]==eff
        and row["real_original_source_joint_ab_nonadditive_rgba_effect"]==interaction
        and abs(float(row["raster_original_source_alpha"])-
                float(base[by,px,3]))<=1e-6
        and row["source_render_order_of_hits_qualified"] is False
        and row["source_material_alpha_per_hit"]=="UNKNOWN"
        and row["actual_artist_drawable_relation"]=="UNKNOWN"
        and row["status"]==(
            "SOURCE_GEOMETRIC_RAY_JOIN_OBSERVED__NOT_ALPHA_OCCLUSION_PROOF"
            if qualified else
            "UNKNOWN_INCOMPLETE_TIED_OR_EMPTY_SOURCE_RAY"),
        "V17_INDEPENDENT_CAUSAL_RGBA_EFFECT_OR_AUTHORITY_INVALID")
    return qualified,len(ray.hits)>=2,eff,interaction,base[by,px,3]>TOL


def court(proposal_path,v14_public,v16_public,source,v14_dir,v14_job,
          v16_manifest,v16_court,v16_pixels):
    p=json.loads(Path(proposal_path).read_text())
    (identity,render,job,faces,cameras,oi,ni,
     singles,rgba,vertices,ref,parents)=validate_and_open(
        v14_public,v16_public,source,v14_dir,v14_job,
        v16_manifest,v16_court,v16_pixels)
    require(p.get("schema")==PROPOSAL_SCHEMA
        and p.get("status")==SOURCE_STATUS
        and p.get("original_source_id")==identity["subject_id"]
        and p.get("original_source_sha256")==identity["source_sha256"]
        and p.get("source_label_private_teacher_only")==source
        and p.get("parents_sha256")==parents
        and p.get("pixel_proposal_policy")==
            "POSED_SOURCE_TRIANGLE_CENTROID_GEOMETRY_ONLY__NO_RGBA_RESPONSE_SELECTION"
        and p.get("original_blender_image_layout")==
            "BOTTOM_ROW_FIRST__CONVERT_TO_TOP_DOWN_FOR_GEOMETRIC_RAY"
        and p.get("sample_count_per_pose_view_joint_sign_at_most")==SAMPLES_PER_CASE
        and p.get("geometric_depth_is_rasterized_order") is False
        and p.get("source_alpha_per_hit_qualified") is False
        and p.get("source_draw_order_authorized") is False
        and p.get("artist_graphics_owner_authorized") is False
        and p.get("training_authorized") is False
        and p.get("materialization_authorized") is False,
        "V17_FORGED_OR_DRIFTED_SOURCE_RAY_TO_RGBA_PROPOSAL")
    rows=p["source_ray_rows_private_only"]
    require(isinstance(rows,list) and len(rows)>0,
        "V17_NO_SOURCE_RAY_CASES_TO_COURT")
    grouped={}
    for row in rows:
        key=(row["source_pose_ordinal"],row["source_camera_view"],
            row["native_joint_sign_pair"])
        require(key[0] in range(3) and key[1] in VIEWS
            and key[2] in SIGNS,"V17_EXTRA_FOREIGN_POSE_VIEW_CONTROL")
        grouped.setdefault(key,[]).append(row)
    expected={(i,v,s) for i in range(3) for v in VIEWS for s in SIGNS}
    require(set(grouped)==expected,"V17_MISSING_REAL_SOURCE_CASES_OR_FORGED_GROUP")
    camera_rows={int(c["view_index"]):c for c in cameras["cameras"]}
    extent=max(1e-12,float(np.linalg.norm(np.ptp(vertices[0,0,0],axis=0))))
    tie=max(1e-9,1e-7*extent)
    recalculated={
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
            cam_row=dict(camera_rows[view],resolution=192)
            cam=camera_from_mapping(cam_row)
            base=rgba[pi,vi,ni["ORIGINAL_BASE"]]
            for si,sign in enumerate(SIGNS):
                xyz=vertices[pi,vi,si]
                expected_pixels=face_centroid_candidate_pixels_v1(
                    xyz,faces,cam,maximum=SAMPLES_PER_CASE)
                case=grouped[(pi,view,sign)]
                supplied=[(r["ray_image_pixel_topdown_x"],
                           r["ray_image_pixel_topdown_y"]) for r in case]
                require(tuple(supplied)==expected_pixels,
                    "V17_SOURCE_RAY_PROPOSAL_NOT_GEOMETRY_ONLY_OR_TAMPERED")
                sa,sb=sign.split("_")
                a=singles[pi,vi,oi["SOURCE_C00_"+sa]]
                b=singles[pi,vi,oi["SOURCE_C01_"+sb]]
                ab=rgba[pi,vi,ni["SOURCE_AB_"+sign]]
                recalculated["original_source_pose_view_pair_cases"]+=1
                for row in case:
                    qualified,multihit,effect,interaction,alpha=(
                        verify_row_geometry_and_rgba(
                            row,xyz,faces,cam,base,a,b,ab,extent,tie
                        )
                    )
                    recalculated["geometry_selected_rays"]+=1
                    recalculated["complete_untied_nonempty_rays"]+=int(qualified)
                    recalculated["multihit_geometric_rays"]+=int(qualified and multihit)
                    recalculated["nohit_or_incomplete_or_tied_rays"]+=int(not qualified)
                    recalculated["sampled_rgba_joint_effect_pixels"]+=int(effect)
                    recalculated["sampled_rgba_joint_nonadditive_pixels"]+=int(interaction)
                    recalculated["sampled_full_RGBA_nonzero_alpha_pixels"]+=int(alpha)
                    recalculated["geometric_ray_with_rgba_effect_pixels"]+=int(
                        qualified and effect)
                    recalculated["effect_without_qualified_geometric_ray_pixels"]+=int(
                        not qualified and effect)
    require(recalculated==p["stats"]
        and recalculated["original_source_pose_view_pair_cases"]==48
        and recalculated["complete_untied_nonempty_rays"]>0
        and recalculated["geometry_selected_rays"]==len(rows),
        "V17_INDEPENDENT_POSed_SOURCE_RAY_AND_EFFECT_STATS_DRIFT")
    return {
        "schema":SCHEMA,
        "status":PASS,
        "original_source_id":identity["subject_id"],
        "original_source_sha256":identity["source_sha256"],
        "source_label_private_teacher_only":source,
        "parent_v14_private_court_sha256":parents["v14_court_sha256"],
        "parent_v16_private_court_sha256":parents["v16_court_sha256"],
        "parent_v16_joint_original_source_RGBA_and_3D_sha256":
            parents["v16_original_RGBA_and_vertices_sha256"],
        "source_pose_ray_proposal_sha256":sha(proposal_path),
        "recalculated_stats":recalculated,
        "independent_source_barycentric_world_reconstruction_per_hit":True,
        "independent_frozen_3d_pose_ray_visibility_mask_and_RGBA_replay":True,
        "actual_alpha_ordered_layer_render_proved":False,
        "author_same_drawable_positive_pairs":0,
        "author_different_drawable_negative_pairs":0,
        "graphics_owner_authorized":False,
        "iris_training_authorized":False,
        "tessa_training_authorized":False,
        "materialization_authorized":False,
        "training_authorized":False
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True,choices=("CAT","TRICERATOPS","COWFISH"))
    for f in ("proposal","v14_public","v16_public","v14_dir","v14_job",
              "v16_manifest","v16_court","v16_pixels","output"):
        p.add_argument("--"+f.replace("_","-"),required=True,type=Path)
    a=p.parse_args()
    c=court(a.proposal,a.v14_public,a.v16_public,a.source,
            a.v14_dir,a.v14_job,a.v16_manifest,a.v16_court,a.v16_pixels)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(c,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({"status":c["status"],"source":a.source,
                      "stats":c["recalculated_stats"],"training_authorized":False},
                     sort_keys=True),flush=True)


if __name__=="__main__":
    main()
