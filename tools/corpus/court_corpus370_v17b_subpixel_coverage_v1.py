#!/usr/bin/env python3
"""Independent V17b court: re-open frozen source and verify real 4x4 subpixels.

Python-only original source geometry, independently SHA-verified. Source-alpha
and artist layer ownership are NEVER established by subpixel ray coverage.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from models.iris.truth.graphics_oracle_v1 import camera_from_mapping
from tools.corpus.measure_corpus370_v17_posed_source_ray_to_causal_rgba_v1 import (
    SIGNS,VIEWS,require,validate_and_open
)
from tools.corpus.measure_corpus370_v17b_original_subpixel_coverage_v1 import (
    SCHEMA as PRODUCER, PASS as SOURCE_MEASURED,
    X_OFFSETS,Y_OFFSETS,N_SAMPLES,propose,offset_camera
)

SCHEMA="RealSaS.Corpus370IndependentOriginalSubpixelRayGeometricCoverageCourt.v17b"
PASS="V17B_REAL_SOURCE_4X4_GEOMETRIC_SUBPIXEL_COVERAGE_REEVALUATED__ALPHA_OWNER_UNKNOWN"


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(8<<20),b""):h.update(chunk)
    return h.hexdigest()


def validate_source_subpixel_world_point(
    world_vertices,faces,camera,px,py,dx,dy,
    front_face,barycentric,depth,extent
):
    require(type(front_face) is int and 0<=front_face<len(faces)
        and isinstance(barycentric,list) and len(barycentric)==3
        and all(type(x) is float and np.isfinite(x)
                and -1e-6<=x<=1.000001 for x in barycentric)
        and abs(sum(barycentric)-1.)<=1e-5
        and type(depth) is float and np.isfinite(depth) and depth>0,
        "V17B_FALSE_SOURCE_FACE_BARYCENTRIC_OR_DEPTH")
    cam=offset_camera(camera,dx,dy)
    xx=((px+.5)/cam.resolution*2.-1.)*cam.half_extent
    yy=(1.-(py+.5)/cam.resolution*2.)*cam.half_extent
    from_ray=(
        np.asarray(cam.origin,dtype=np.float64)
        +np.asarray(cam.right,dtype=np.float64)*xx
        +np.asarray(cam.screen_up,dtype=np.float64)*yy
        +np.asarray(cam.forward,dtype=np.float64)*depth
    )
    from_face=(
        np.asarray(barycentric,dtype=np.float64)
        @ np.asarray(world_vertices[faces[front_face]],dtype=np.float64)
    )
    require(np.max(np.abs(from_ray-from_face))<=max(1e-6,2e-6*extent),
        "V17B_INDEPENDENT_POSed_SUBPIXEL_GEOMETRIC_HIT_NOT_ON_SOURCE_FACE")


def evaluate(proposal_path,v17_public,v17_proposal,v17_court,source,
             v14_public,v16_public,v14_dir,v14_job,v16_manifest,
             v16_court,v16_pixels):
    given=json.loads(Path(proposal_path).read_text())
    # This recomputes each of the original native A+B signed 4x4 offsets,
    # not only checking expected JSON schema and numeric summary.
    expected=propose(v17_public,v17_proposal,v17_court,source,
                     v14_public,v16_public,v14_dir,v14_job,
                     v16_manifest,v16_court,v16_pixels)
    require(given==expected
        and given.get("schema")==PRODUCER
        and given.get("status")==SOURCE_MEASURED
        and given.get("sample_policy")==
            "4X4_FIXED_GEOMETRIC_PIXEL_CELL_OFFSETS__NO_RGBA_RESPONSE_TUNING"
        and given.get("subpixel_offsets_x")==list(X_OFFSETS)
        and given.get("subpixel_offsets_y")==list(Y_OFFSETS)
        and given.get("parent_original_v17_source_ray_unknowns_only") is True
        and given.get("observed_subpixel_coverage_is_not_original_painted_alpha") is True
        and given.get("blender_cycles_raster_filter_exact_match_authorized") is False
        and given.get("artist_drawable_same_pairs_minted")==0
        and given.get("artist_drawable_different_pairs_minted")==0
        and given.get("alpha_draw_order_authorized") is False
        and given.get("graphics_owner_authorized") is False
        and given.get("training_authorized") is False
        and given.get("materialization_authorized") is False,
        "V17B_INDEPENDENT_ORIGINAL_SUBPIXEL_REPLAY_TAMPER_OR_FALSE_AUTHORITY")
    (identity,manifest,job,faces,calibration,_,_,_,_,joint,
     cases,parents)=validate_and_open(
        v14_public,v16_public,source,v14_dir,v14_job,
        v16_manifest,v16_court,v16_pixels)
    cameras={int(x["view_index"]):x for x in calibration["cameras"]}
    extent=max(1e-12,float(np.linalg.norm(np.ptp(joint[0,0,0],axis=0))))
    valid_world_hits=0
    for row in given["private_subpixel_ray_rows"]:
        pi=row["source_pose_ordinal"]
        view=row["source_camera_view"]
        sign=row["native_joint_sign_pair"]
        px=row["geometric_pixel_x"];py=row["geometric_pixel_y"]
        require(pi in range(3) and view in VIEWS and sign in SIGNS
            and type(px) is int and type(py) is int
            and 0<=px<192 and 0<=py<192,
            "V17B_INVALID_ORIGINAL_SUBPIXEL_SOURCE_INDEX")
        xyz=joint[pi,VIEWS.index(view),SIGNS.index(sign)]
        cam=camera_from_mapping(dict(cameras[view],resolution=192))
        require(row["original_framebuffer_pixel_y"]==191-py
            and row["source_drawable_identity"]=="UNKNOWN"
            and len(row["subpixel_rays"])==N_SAMPLES,
            "V17B_WRONG_IMAGE_SUBPIXEL_FRAME_OR_DRAWABLE")
        for sample in row["subpixel_rays"]:
            if not sample["qualified_subpixel_geometric_hit"]:
                require(sample["first_source_face_ordinal_teacher_only"] is None
                    and sample["first_hit_barycentric_teacher_only"] is None
                    and sample["first_hit_world_depth"] is None,
                    "V17B_FALSE_SUBPIXEL_GEOMETRIC_UNKNOWN_FACE")
                continue
            validate_source_subpixel_world_point(
                xyz,faces,cam,px,py,
                sample["subpixel_offset_x"],sample["subpixel_offset_y"],
                sample["first_source_face_ordinal_teacher_only"],
                sample["first_hit_barycentric_teacher_only"],
                sample["first_hit_world_depth"],extent
            )
            valid_world_hits+=1
    s=given["status_counts"]
    require(valid_world_hits==s["subpixel_complete_untied_hits"]
        and s["subpixel_physical_rays_tested"]==
            N_SAMPLES*s["original_unqualified_rays_tested"]
        and s["original_unknown_center_ray_recovered_by_subpixel_coverage"]+
            s["original_unknown_center_ray_remains_unknown"]==
            s["original_unqualified_rays_tested"],
        "V17B_VERIFIED_WORLD_BARYCENTRIC_COVERAGE_COUNTS_DRIFT")
    return {
        "schema":SCHEMA,
        "status":PASS,
        "original_source_id":identity["subject_id"],
        "original_source_sha256":identity["source_sha256"],
        "source_name_private_teacher_only":source,
        "private_v17b_original_subpixel_proposal_sha256":sha(proposal_path),
        "private_v17a_original_pose_ray_court_sha256":given["original_v17_private_court_sha256"],
        "private_v16_original_world_vertices_rgba_sha256":
            given["original_v16_rgba_vertices_sha256"],
        "source_pixel_camera_sha256":given["original_camera_sha256"],
        "source_rest_geometry_sha256":given["rest_geometry_sha256"],
        "independently_recomputed_subpixel_world_point_hits":valid_world_hits,
        "recalculated_subpixel_geometric_coverage":s,
        "all_4x4_subpixel_offsets_and_barycentric_ray_hits_verified":True,
        "original_raster_alpha_compositing_qualified":False,
        "original_raster_filter_exact_qualified":False,
        "author_same_drawable_labels_qualified":0,
        "author_different_drawable_labels_qualified":0,
        "graphics_owner_authorized":False,
        "iris_training_authorized":False,
        "tessa_training_authorized":False,
        "materialization_authorized":False,
        "training_authorized":False
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True,choices=("CAT","TRICERATOPS","COWFISH"))
    for f in ("proposal","v17_public","v17_proposal","v17_court",
              "v14_public","v16_public","v14_dir","v14_job",
              "v16_manifest","v16_court","v16_pixels","output"):
        p.add_argument("--"+f.replace("_","-"),type=Path,required=True)
    a=p.parse_args()
    result=evaluate(a.proposal,a.v17_public,a.v17_proposal,
        a.v17_court,a.source,a.v14_public,a.v16_public,a.v14_dir,
        a.v14_job,a.v16_manifest,a.v16_court,a.v16_pixels)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":result["status"],"source":a.source,
        "counts":result["recalculated_subpixel_geometric_coverage"],
        "training_authorized":False},sort_keys=True),flush=True)


if __name__=="__main__":
    main()
