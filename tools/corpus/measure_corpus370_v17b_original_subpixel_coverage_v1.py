#!/usr/bin/env python3
"""V17b: physically test 4x4 SUBPIXEL source-mesh coverage of V17 UNKNOWN rays.

The original V17 pixel-CENTRE ray may miss thin geometry despite a nonzero
anti-aliased original Blender RGBA pixel. Query 16 deterministic sub-pixel
parallel rays on the EXACT source V16 joint world mesh + frozen V14 camera.
Do not use RGB/effect labels to select offsets; only V17's already sealed
unknown ray list. This proves geometric subpixel coverage only, not exact
Cycles filtering, alpha compositing, actual painted drawable or training truth.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import numpy as np

from models.iris.truth.graphics_oracle_v1 import (
    camera_from_mapping,measure_orthographic_ray_stack_v1
)
from tools.corpus.measure_corpus370_v17_posed_source_ray_to_causal_rgba_v1 import (
    SIGNS,VIEWS,require,validate_and_open
)

SCHEMA="RealSaS.Corpus370SourcePosedSubpixelGeometricCoverageProposal.v17b"
PASS="ORIGINAL_SOURCE_SUBPIXEL_GEOMETRIC_COVERAGE_MEASURED__ALPHA_OWNER_UNKNOWN"
PARENT="RealSaS.Corpus370V17OriginalPosedSourceRayToCausalRGBAIndependentRealResult.v1"
PARENT_COURT="RealSaS.Corpus370IndependentOriginalPosedSourceRayCausalRGBAJoinCourt.v17"
X_OFFSETS=(-.375,-.125,.125,.375)
Y_OFFSETS=(-.375,-.125,.125,.375)
N_SAMPLES=16


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(8<<20),b""):h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def offset_camera(camera,dx,dy):
    """Translate orthographic camera origin by fractional pixel ray offset.

    The input geometric pixel coordinate is TOP-DOWN: positive y offset
    moves the world ray opposite screen_up; x offset moves with camera.right.
    No Blender or original source art is needed to evaluate this transform.
    """
    require(-.5<dx<.5 and -.5<dy<.5,"V17B_SUBPIXEL_OFFSET_IN_PIXEL_BOUND")
    scale=2.*camera.half_extent/camera.resolution
    o=np.asarray(camera.origin,dtype=np.float64)
    xyz=(o
         +np.asarray(camera.right,dtype=np.float64)*(dx*scale)
         -np.asarray(camera.screen_up,dtype=np.float64)*(dy*scale))
    return replace(camera,origin=tuple(float(x) for x in xyz))


def ray_sample(xyz,faces,camera,px,py,dx,dy,tie):
    displaced=offset_camera(camera,dx,dy)
    stack=measure_orthographic_ray_stack_v1(
        xyz,faces,displaced,px,py,tie_epsilon=tie)
    qualified=bool(stack.complete and not stack.has_depth_tie and stack.hits)
    return {
        "subpixel_offset_x":float(dx),
        "subpixel_offset_y":float(dy),
        "complete":bool(stack.complete),
        "depth_tie":bool(stack.has_depth_tie),
        "geometric_hit_count":len(stack.hits),
        "first_source_face_ordinal_teacher_only":
            int(stack.hits[0].face_index) if qualified else None,
        "first_hit_barycentric_teacher_only":
            [float(v) for v in stack.hits[0].barycentric] if qualified else None,
        "first_hit_world_depth":float(stack.hits[0].depth) if qualified else None,
        "qualified_subpixel_geometric_hit":qualified
    }


def propose(v17_public,v17_proposal,v17_court,source,v14_public,v16_public,
            v14_dir,v14_job,v16_manifest,v16_court,v16_pixels):
    public=load(v17_public)
    original=load(v17_proposal)
    qualified=load(v17_court)
    require(public.get("schema")==PARENT
        and public.get("run_id")==42088058238
        and public.get("all_github_hosted_jobs_success")==4
        and public.get("source_graphics_ownership_authorized") is False
        and public.get("training_authorized") is False
        and public.get("materialization_authorized") is False,
        "V17B_MISSING_REAL_V17_ORIGINAL_PUBLIC_COURT")
    refs=[x for x in public["source_results"] if x.get("source")==source]
    require(len(refs)==1,"V17B_SOURCE_NOT_IN_QUALIFIED_V17")
    ref=refs[0]
    require(sha(v17_court)==ref["court_sha256"]
        and qualified.get("schema")==PARENT_COURT
        and qualified.get("status")==
            "V17_REAL_POSED_SOURCE_GEOMETRIC_RAY_AND_RGBA_LINK_RECALCULATED__ARTIST_OWNER_UNKNOWN"
        and qualified.get("source_pose_ray_proposal_sha256")==sha(v17_proposal)
        and qualified.get("original_source_sha256")==ref["original_source_sha256"]
        and qualified.get("graphics_owner_authorized") is False
        and qualified.get("training_authorized") is False
        and qualified.get("materialization_authorized") is False,
        "V17B_ORIGINAL_V17_PRIVATE_COURT_AND_INPUT_SHA_MISMATCH")
    (identity,renderer,job,faces,bundle,oldix,newix,singles,rgba,
     vertices,independent_cases,parents)=validate_and_open(
         v14_public,v16_public,source,v14_dir,v14_job,
         v16_manifest,v16_court,v16_pixels)
    require(identity["subject_id"]==ref["subject_id"]
        and identity["source_sha256"]==ref["original_source_sha256"]
        and original.get("parents_sha256")==parents
        and original.get("schema")==
            "RealSaS.Corpus370OriginalPosedSourceRayToCausalRGBAJoinProposal.v17"
        and original.get("status")==
            "V17_REAL_SOURCE_POSED_GEOMETRIC_RAYS_AND_CAUSAL_RGBA_OBSERVED__OWNER_UNKNOWN"
        and original.get("stats")==qualified.get("recalculated_stats")
        and original["stats"]["geometry_selected_rays"]==
            ref["sampled_original_pose_ray_cases"]
        and original["stats"]["nohit_or_incomplete_or_tied_rays"]==
            ref["unknown_incomplete_tied_or_empty_rays"],
        "V17B_V17_ORIGINAL_PRIVATE_RAY_PIXEL_SOURCE_CHAIN")
    cams={int(z["view_index"]):z for z in bundle["cameras"]}
    extent=max(1e-12,float(np.linalg.norm(np.ptp(vertices[0,0,0],axis=0))))
    tie=max(1e-9,1e-7*extent)
    outputs=[]
    stats=dict(
        parent_v17_qualified_center_pixel_rays=ref["complete_untied_nonempty_rays"],
        parent_v17_unqualified_center_rays=ref["unknown_incomplete_tied_or_empty_rays"],
        original_unqualified_rays_tested=0,
        subpixel_physical_rays_tested=0,
        subpixel_complete_untied_hits=0,
        original_unknown_center_ray_recovered_by_subpixel_coverage=0,
        original_unknown_center_ray_remains_unknown=0,
        original_alpha_nonzero_unknown_center_rays_recovered=0,
        original_visual_effect_unknown_center_rays_recovered=0,
        original_visual_effect_unknown_center_rays_not_recovered=0,
    )
    seen=set()
    for parent_row in original["source_ray_rows_private_only"]:
        if (parent_row["geometric_stack_complete"]
                and not parent_row["geometric_stack_has_depth_tie"]
                and parent_row["source_geometric_hit_count"]>0):
            continue
        pi=parent_row["source_pose_ordinal"]
        view=parent_row["source_camera_view"]
        sign=parent_row["native_joint_sign_pair"]
        px=parent_row["ray_image_pixel_topdown_x"]
        py=parent_row["ray_image_pixel_topdown_y"]
        key=(pi,view,sign,px,py)
        require(key not in seen
            and pi in range(3) and view in VIEWS and sign in SIGNS
            and type(px) is int and type(py) is int
            and 0<=px<192 and 0<=py<192,
            "V17B_SOURCE_UNKNOWN_QUERY_DUPLICATE_OR_TAMPERED")
        seen.add(key)
        vi=VIEWS.index(view)
        si=SIGNS.index(sign)
        xyz=vertices[pi,vi,si]
        cam=camera_from_mapping(dict(cams[view],resolution=192))
        samples=[ray_sample(xyz,faces,cam,px,py,dx,dy,tie)
                 for dy in Y_OFFSETS for dx in X_OFFSETS]
        q=sum(int(s["qualified_subpixel_geometric_hit"]) for s in samples)
        visual=bool(parent_row["real_original_source_joint_ab_visible_rgba_effect"])
        alpha=bool(parent_row["raster_original_source_alpha"]>.025)
        outputs.append({
            "source_pose_ordinal":pi,
            "source_camera_view":view,
            "native_joint_sign_pair":sign,
            "geometric_pixel_x":px,
            "geometric_pixel_y":py,
            "original_framebuffer_pixel_y":parent_row["original_blender_framebuffer_y"],
            "parent_original_rgba_effect_observed":visual,
            "parent_original_source_pixel_alpha_nonzero":alpha,
            "subpixel_rays":samples,
            "valid_subpixel_hit_count":q,
            "status":(
                "SOME_GEOMETRIC_SUBPIXEL_SOURCE_HITS__ALPHA_AND_ARTIST_OWNER_UNKNOWN"
                if q else "NO_QUALIFIED_HITS_IN_FIXED_4X4__UNKNOWN_NOT_ABSENCE"
            ),
            "source_drawable_identity":"UNKNOWN"
        })
        stats["original_unqualified_rays_tested"]+=1
        stats["subpixel_physical_rays_tested"]+=len(samples)
        stats["subpixel_complete_untied_hits"]+=q
        stats["original_unknown_center_ray_recovered_by_subpixel_coverage"]+=int(q>0)
        stats["original_unknown_center_ray_remains_unknown"]+=int(q==0)
        stats["original_alpha_nonzero_unknown_center_rays_recovered"]+=int(
            alpha and q>0)
        stats["original_visual_effect_unknown_center_rays_recovered"]+=int(
            visual and q>0)
        stats["original_visual_effect_unknown_center_rays_not_recovered"]+=int(
            visual and q==0)
    require(stats["original_unqualified_rays_tested"]==
                stats["parent_v17_unqualified_center_rays"]
        and stats["subpixel_physical_rays_tested"]==
            N_SAMPLES*len(outputs)
        and stats["original_unknown_center_ray_recovered_by_subpixel_coverage"]+
            stats["original_unknown_center_ray_remains_unknown"]==len(outputs),
        "V17B_ORIGINAL_UNCERTAIN_RAY_SUBPIXEL_COVERAGE_INCOMPLETE")
    return {
        "schema":SCHEMA,"status":PASS,
        "source_label_private_teacher_only":source,
        "original_source_id":identity["subject_id"],
        "original_source_sha256":identity["source_sha256"],
        "original_v17_private_court_sha256":sha(v17_court),
        "original_v17_ray_proposal_sha256":sha(v17_proposal),
        "original_v16_rgba_vertices_sha256":parents["v16_original_RGBA_and_vertices_sha256"],
        "original_camera_sha256":parents["original_camera_sha256"],
        "rest_geometry_sha256":parents["rest_geometry_sha256"],
        "subpixel_offsets_x":list(X_OFFSETS),
        "subpixel_offsets_y":list(Y_OFFSETS),
        "sample_policy":"4X4_FIXED_GEOMETRIC_PIXEL_CELL_OFFSETS__NO_RGBA_RESPONSE_TUNING",
        "parent_original_v17_source_ray_unknowns_only":True,
        "status_counts":stats,
        "private_subpixel_ray_rows":outputs,
        "observed_subpixel_coverage_is_not_original_painted_alpha":True,
        "blender_cycles_raster_filter_exact_match_authorized":False,
        "artist_drawable_same_pairs_minted":0,
        "artist_drawable_different_pairs_minted":0,
        "alpha_draw_order_authorized":False,
        "graphics_owner_authorized":False,
        "training_authorized":False,
        "materialization_authorized":False
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True,choices=("CAT","TRICERATOPS","COWFISH"))
    for f in ("v17_public","v17_proposal","v17_court","v14_public",
              "v16_public","v14_dir","v14_job","v16_manifest",
              "v16_court","v16_pixels","output"):
        p.add_argument("--"+f.replace("_","-"),type=Path,required=True)
    a=p.parse_args()
    result=propose(a.v17_public,a.v17_proposal,a.v17_court,a.source,
        a.v14_public,a.v16_public,a.v14_dir,a.v14_job,
        a.v16_manifest,a.v16_court,a.v16_pixels)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({"status":result["status"],"source":a.source,
        "counts":result["status_counts"],"training_authorized":False},
        sort_keys=True),flush=True)


if __name__=="__main__":
    main()
