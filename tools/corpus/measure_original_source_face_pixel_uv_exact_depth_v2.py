#!/usr/bin/env python3
"""Original physical source-face/sample-pixel UV edit truth V2: exact local depth.

Improves conservative V1 without weakening any evidence condition. For every
candidate original face covering a GUARDED full pixel, compare each possible
other original triangle only on the exact polygon intersection of its source
screen projection and the guarded pixel. The depth difference is affine, so
checking all convex intersection vertices proves strict first-hit dominance
over the entire pixel, without reconstructing Cycles stochastic ray offsets.

Original GLB/teacher/face UV, source policy, original V1 court and original
256 Cycles FULL physical edit frames are SHA-locked, reused, NOT rerendered.
Retains source-only scope, per-face edit2 UNKNOWN outside admitted pixels,
no artist drawable grouping and NO student training or materialization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from shapely.geometry import box as polygon_box, Polygon
from shapely.strtree import STRtree

from tools.corpus import measure_original_source_face_pixel_uv_causality_v1 as v1

SCHEMA="RealSaS.OriginalSourceFacePixelUVInterventionExactLocalDepth.v2"
V1_RECEIPTS={
    "v1-producer.json":"ad26d6f23baa25cb8ebcf38b9d8256344b83c9bed7c342f7a1688cc659208a65",
    "v1-independent-court.json":"2b77b43ec09afe37fb1f277887c9e946133c034d3d736f4b8ed0f5d40f2f5bc0",
}
DEPTH_EPS=1e-5
PIXEL_GUARD=v1.PIXEL_GUARD
UV_GUARD=v1.UV_TEXEL_GUARD


def req(ok,why):
    if not ok:raise ValueError("ORIGINAL_FACE_PIXEL_EXACT_DEPTH_"+why)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def prepare_visible_source_faces(triangles):
    """Shapely STRtree for nondegenerate source projection only."""
    polys=[]
    to_source_face=[]
    degenerate=[]
    for idx,pts in enumerate(triangles):
        a=pts[1]-pts[0];b=pts[2]-pts[0]
        area=a[0]*b[1]-a[1]*b[0]
        if abs(area)<1e-10:
            degenerate.append(idx)
        else:
            polys.append(Polygon(pts))
            to_source_face.append(idx)
    return STRtree(polys),polys,np.asarray(to_source_face,np.int64),np.asarray(degenerate,np.int64)


def affine_source_depth(points,vertex_triangle,vertex_depths):
    """Source orthographic depth is affine in the 2D projected triangle."""
    pts=np.asarray(points,np.float64)
    tri=np.asarray(vertex_triangle,np.float64)
    z=np.asarray(vertex_depths,np.float64)
    matrix=np.column_stack((tri[1]-tri[0],tri[2]-tri[0]))
    if abs(np.linalg.det(matrix))<1e-10:
        raise ValueError("DEGENERATE_PROJECTED_ORIGINAL_SOURCE_FACE")
    w=np.linalg.solve(matrix,(pts-tri[0]).T).T
    return z[0]+w[:,0]*(z[1]-z[0])+w[:,1]*(z[2]-z[0])


def possible_source_occluder_intersection(
        current_face,screen_triangles,face_depths,tree,polys,index_to_face,
        degenerate_faces,*,pixel_x,pixel_y,strict_depth=DEPTH_EPS):
    """Exact convex intersection check with strict depth separation.

    Returns False when another source triangle could match/occlude nearest
    original face at ANY point inside entire guarded pixel. Degenerate
    projected triangles block conservatively if their AABB can intersect.
    """
    left=pixel_x-PIXEL_GUARD;right=pixel_x+1+PIXEL_GUARD
    top=pixel_y-PIXEL_GUARD;bottom=pixel_y+1+PIXEL_GUARD
    square=polygon_box(left,top,right,bottom)
    tri=screen_triangles[current_face]
    z=face_depths[current_face]
    # Degenerate source face projection may touch a ray on a measure-zero
    # screen set; for all-point mathematical proof we must not ignore it.
    for other in degenerate_faces:
        other_tri=screen_triangles[other]
        if (other_tri[:,0].min()<=right and other_tri[:,0].max()>=left and
                other_tri[:,1].min()<=bottom and other_tri[:,1].max()>=top and
                face_depths[other].min()<=
                  max(affine_source_depth(np.asarray(square.exterior.coords),
                                          tri,z))+strict_depth):
            return False
    for idx in tree.query(square):
        other=int(index_to_face[int(idx)])
        if other==current_face:
            continue
        piece=square.intersection(polys[int(idx)])
        if piece.is_empty:
            continue
        # Convex triangle∩rectangle should be one convex Polygon, LineString
        # or Point. Unrecognized geometry -> fail closed, never claim proof.
        if piece.geom_type=="Polygon":
            points=np.asarray(piece.exterior.coords,np.float64)
        elif piece.geom_type=="LineString":
            points=np.asarray(piece.coords,np.float64)
        elif piece.geom_type=="Point":
            points=np.asarray([[piece.x,piece.y]],np.float64)
        else:
            return False
        other_depth=affine_source_depth(points,screen_triangles[other],
                                          face_depths[other])
        candidate_depth=affine_source_depth(points,tri,z)
        if np.any(other_depth-candidate_depth<=strict_depth):
            return False
    return True


def exact_depth_original_face_uv_rows(vertices,faces,uv,camera,rectangles,
                                      *,resolution=v1.RES):
    xyz=np.asarray(vertices,np.float64)
    f=np.asarray(faces,np.int64)
    face_uv=np.asarray(uv,np.float64)
    req(face_uv.shape==(len(f),3,2) and np.isfinite(face_uv).all(),
        "INVALID_NATIVE_SOURCE_FACE_UV")
    projected,source_depth=v1.project_original(xyz,f,camera,resolution=resolution)
    tree,polys,index_to_face,degenerate=prepare_visible_source_faces(projected)
    result=[]
    stats={
        "full_pixel_face_inside_edited_source_uv_rect_candidates":0,
        "source_degenerate_projected_original_faces":len(degenerate),
        "exact_source_geometry_first_hit_witnesses":0,
    }
    for fi,(tri,z,uvtri) in enumerate(zip(projected,source_depth,face_uv)):
        if fi in degenerate:
            continue
        xmin,xmax,ymin,ymax=v1.box(tri,resolution)
        if xmin>=xmax or ymin>=ymax:
            continue
        xx,yy=np.meshgrid(np.arange(xmin,xmax),np.arange(ymin,ymax))
        corners=np.stack((
            np.stack((xx-PIXEL_GUARD,yy-PIXEL_GUARD),axis=-1),
            np.stack((xx+1+PIXEL_GUARD,yy-PIXEL_GUARD),axis=-1),
            np.stack((xx-PIXEL_GUARD,yy+1+PIXEL_GUARD),axis=-1),
            np.stack((xx+1+PIXEL_GUARD,yy+1+PIXEL_GUARD),axis=-1),
        ),axis=-2)
        bary=v1.triangle_corner_barycentric(tri,corners)
        if bary is None:continue
        covered=np.all(bary>=v1.BARY_MARGIN,axis=(-2,-1))
        if not np.any(covered):continue
        depth_four=np.sum(bary*z[None,None,None,:],axis=-1)
        covered &= (
            (np.min(depth_four,axis=-1)>camera["clip_start"]+DEPTH_EPS) &
            (np.max(depth_four,axis=-1)<camera["clip_end"]-DEPTH_EPS))
        if not np.any(covered):continue
        uv_four=np.sum(bary[...,None]*uvtri[None,None,None,:,:],axis=-2)
        for operation_index,rect in enumerate(rectangles):
            lo=np.asarray(rect[:2],np.float64)/512
            hi=np.asarray(rect[2:],np.float64)/512
            in_edit=(covered &
                np.all(uv_four>lo+UV_GUARD,axis=(-2,-1)) &
                np.all(uv_four<hi-UV_GUARD,axis=(-2,-1)))
            ys,xs=np.where(in_edit)
            stats["full_pixel_face_inside_edited_source_uv_rect_candidates"]+=len(xs)
            for y,x in zip(ys,xs):
                screen_y=ymin+int(y);screen_x=xmin+int(x)
                if not possible_source_occluder_intersection(
                        fi,projected,source_depth,tree,polys,index_to_face,
                        degenerate,pixel_x=screen_x,pixel_y=screen_y):
                    continue
                result.append({
                    "face_index":int(fi),
                    "pixel_x":screen_x,
                    "pixel_y_top_first":screen_y,
                    "source_uv_edit_operation_index":int(operation_index),
                })
    req(len(result)==len(set((r["pixel_y_top_first"],r["pixel_x"])
                             for r in result)),
        "MORE_THAN_ONE_SOURCE_FACE_FOR_ONE_PIXEL")
    stats["exact_source_geometry_first_hit_witnesses"]=len(result)
    result.sort(key=lambda x:(x["pixel_y_top_first"],x["pixel_x"]))
    return result,stats


def produce(source_root,output):
    root=Path(source_root)
    for filename,pin in {**v1.PINS,**V1_RECEIPTS}.items():
        req(sha(root/filename)==pin,"ORIGINAL_SOURCE_BYTE_PIN_DRIFT:"+filename)
    old=json.loads((root/"v1-producer.json").read_text())
    old_court=json.loads((root/"v1-independent-court.json").read_text())
    req(old["status"]==
        "ORIGINAL_SOURCE_FACE_TO_PIXEL_TO_UV_EDIT_CONSERVATIVE_WITNESSES__INDEPENDENT_COURT_REQUIRED"
        and old_court["status"]==
        "PASS_ORIGINAL_FACE_PIXEL_UV_EDIT_CAUSALITY_CONSERVATIVE_SUBSET__OWNER_UNKNOWN"
        and old_court["source_producer_witness_sha256"]==V1_RECEIPTS["v1-producer.json"]
        and old["counts"]["qualified_original_face_pixel_witness_count"]==608
        and old_court["source_original_face_pixel_court_count"]==608
        and old_court["iris_training_authorized"] is False,
        "V1_INDEPENDENT_SOURCE_RECOMPUTATION_NOT_QUALIFIED")
    plan=json.loads((root/"uv-plan.json").read_text())
    camera=json.loads((root/"camera.json").read_text())
    manifest=json.loads((root/"physical-manifest.json").read_text())
    policy=json.loads((root/"source-policy.json").read_text())
    req(plan["source_original_glb_sha256"]==v1.CAT
        and policy["material_mode"]=="unlit_preserve_alpha"
        and manifest["original_source_sha256"]==v1.CAT
        and manifest["source_original_sample_npz_sha256"]==v1.PINS["physical.npz"]
        and manifest["original_source_rest_only"] is True
        and manifest["camera_indices"]==list(v1.VIEWS)
        and manifest["original_camera_bundle_sha256"]==v1.PINS["camera.json"],
        "ORIGINAL_SOURCE_CAMERA_MATERIAL_OR_SAMPLE_DRIFT")
    with np.load(root/"teacher.npz",allow_pickle=False) as teacher, \
         np.load(root/"rest.npz",allow_pickle=False) as rest, \
         np.load(root/"face-uv.npz",allow_pickle=False) as u, \
         np.load(root/"physical.npz",allow_pickle=False) as samples:
        faces=np.asarray(teacher["faces"],np.int64)
        vertices=np.asarray(teacher["rest_vertices_source_world"],np.float64)
        uv=np.asarray(u["face_uv"],np.float64)
        render=np.asarray(samples["source_uv_edit_cycles_sample_rgba"],np.float32)
        req(np.array_equal(faces,rest["faces"]) and
            np.array_equal(vertices,rest["rest_vertices_source_world"]) and
            faces.shape==(684,3) and render.shape==(4,8,8,192,192,4),
            "ORIGINAL_SOURCE_FACE_ORDER_OR_CYCLES_PIXEL_LAYOUT_INVALID")
    rectangles=[tuple(r["source_rgba_png_pixel_rectangle_xyxy_exclusive"])
                for r in plan["selected_original_source_uv_edits"]]
    rows=[]
    source_stats=[]
    for ix,view in enumerate(v1.VIEWS):
        new,stats=exact_depth_original_face_uv_rows(
            vertices,faces,uv,camera["cameras"][view],rectangles)
        sample_truth=v1.physically_court_witness(new,render[ix],view_ordinal=view)
        rows.extend([{"original_camera_view":view,**r} for r in new])
        source_stats.append({**sample_truth,**stats})
    old_keys={
        (r["original_camera_view"],r["pixel_y_top_first"],r["pixel_x"],
         r["face_index"],r["source_uv_edit_operation_index"])
        for r in old["source_original_face_pixel_witness_rows_PRIVATE"]
    }
    new_keys={
        (r["original_camera_view"],r["pixel_y_top_first"],r["pixel_x"],
         r["face_index"],r["source_uv_edit_operation_index"]) for r in rows
    }
    req(old_keys <= new_keys and len(old_keys)==608,
        "EXACT_DEPTH_MUST_PRESERVE_EVERY_PREVIOUSLY_PROVEN_V1_FACE_PIXEL")
    counts={
        "qualified_original_source_face_pixels":len(rows),
        "qualified_original_unique_source_faces":len({r["face_index"] for r in rows}),
        "qualified_original_source_same_sample_positive_edit_effect_incidents":
            sum(row["independently_edited_sample_incidents"] for row in source_stats),
        "qualified_original_source_other_edit_absence_obligations":
            sum(row["absent_other_edit_effect_obligations"] for row in source_stats),
        "original_source_face_pixels_by_atlas_edit":[
            sum(row["source_uv_edit_pixel_counts"][i] for row in source_stats)
            for i in range(3)],
        "V1_qualified_source_face_pixels_preserved":len(old_keys),
    }
    req(counts["qualified_original_source_face_pixels"]>=608,
        "NO_NEW_REAL_SOURCE_FACE_CORRESPONDENCE")
    report={
        "schema":SCHEMA,
        "status":"EXACT_LOCAL_SOURCE_DEPTH_PIXEL_UV_EDIT_TRUTH_PRODUCED__INDEPENDENT_COURT_REQUIRED",
        "original_source_sha256":v1.CAT,
        "original_parent_sha256":{**v1.PINS,**V1_RECEIPTS},
        "source_court_expression":
            "GUARDED_PIXEL_FULL_TRIANGLE_COVERAGE__EXACT_PROJECTED_TRIANGLE_PIXEL_INTERSECTION__AFFINE_DEPTH_DOMINANCE__UV_INTERIOR",
        "original_source_physical_same_sample_render_count_reused":256,
        "new_original_source_renders":0,
        "strict_original_source_pixel_guard_px":PIXEL_GUARD,
        "strict_source_uv_interior_guard_texels":1,
        "independent_original_source_face_pixel_V1_subset_preserved":True,
        "source_counts":counts,
        "per_source_view":source_stats,
        "source_face_pixel_uv_edit_private_rows":rows,
        "source_artist_drawable_ownership":"UNKNOWN",
        "source_original_faces_without_positive_witness":"UNKNOWN__NOT_FALSE",
        "source_exact_cycles_sample_pixel_jitter_coordinates":"UNKNOWN__NO_RANDOM_JITTER_RECONSTRUCTION",
        "iris_student_training_target_authorized":False,
        "graphics_owner_authorized":False,
        "same_drawable_training_pairs_qualified":0,
        "different_drawable_training_pairs_qualified":0,
        "iris_training_authorized":False,
        "tessa_training_authorized":False,
        "training_authorized":False,
        "materialization_authorized":False,
    }
    dest=Path(output);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({
        "status":report["status"],"source_counts":counts,
        "original_source_artist_owner":"UNKNOWN","training_authorized":False,
    },sort_keys=True),flush=True)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    produce(a.source_root,a.output)
