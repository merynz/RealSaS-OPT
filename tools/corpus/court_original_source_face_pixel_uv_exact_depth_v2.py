#!/usr/bin/env python3
"""Independent original Cat exact pixel-face UV edit source physical V2 court.

Does NOT import producer V2, Shapely, or the earlier V1 face-pixel producer.
Uses custom Sutherland–Hodgman half-plane clipping, direct barycentric affine
depth algebra and original bytes. Independently re-verifies EVERY emitted
source face↔image-pixel↔UV edit witness and all eight physical Cycles samples.

Proves soundness of all emitted witnesses, not global completeness.
No original artistic drawing layer or IRIS student authority.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA="RealSaS.IndependentExactLocalOriginalFacePixelEditCourt.v2"
SOURCE="RealSaS.OriginalSourceFacePixelUVInterventionExactLocalDepth.v2"
PASS="PASS_ORIGINAL_FACE_PIXEL_UV_EDIT_CAUSALITY_EXACT_LOCAL_DEPTH__OWNER_UNKNOWN"
CAT="1e73aaf7d497dd7d2946badf92d189d2333f163e66dced91feb446bc9bdee850"
PINS={
    "source.glb":CAT,
    "teacher.npz":"c2792c8efb36c4ccdf9178c6b933351b50364bbd28707e9365a3c115db00d603",
    "rest.npz":"748609aca5ba7c509aa390ed65baeca7612b8879aa00409735da3a96758379cf",
    "face-uv.npz":"7f3afdc4f0f848de3aa06bda2220151427d352368a0f235ca09d5d96f7834787",
    "uv-binding.json":"b966a3793c682cb06ec70341e659aabbded9343ace3d645c08643de40f2799d7",
    "camera.json":"300b935a47379f6c44d4b4356554d1c32a86ca3b732a62405c73da9e0ec6adf0",
    "source-policy.json":"c8fe4e307ddbbaf5cc0892cffc4267bd08e394f2299f4e70155ffc6207190a4b",
    "uv-plan.json":"8be8fccde3d9cc97f7ba452fd50c1a9f4ebea2e34da590a22d30872b2b0f497b",
    "physical.npz":"d422b7d7d082fba5deac5dd1239b8c6aec078f4d635a7bd5b1f10e514100b114",
    "physical-manifest.json":"8a56e47a09f5c1216de2c97e6f13d5e04385f7fe7d2e580d0085b2ab69dc2da1",
    "physical-court.json":"90b85b9c1a963b23f7c3037ca443975bf2d5c5d4123568d8aa8e54557e433c08",
    "v1-producer.json":"ad26d6f23baa25cb8ebcf38b9d8256344b83c9bed7c342f7a1688cc659208a65",
    "v1-independent-court.json":"2b77b43ec09afe37fb1f277887c9e946133c034d3d736f4b8ed0f5d40f2f5bc0",
}
VIEWS=(1,3,5,7)
GUARD=.05
DEPTH_EPS=1e-5
BARY_EPS=1e-7
UV_EPS=1/512
TOL=.025
RES=192


def req(ok,code):
    if not ok:raise ValueError("INDEPENDENT_EXACT_SOURCE_"+code)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def direct_source_triangles(vertices,faces,camera):
    """Independent original projection, not the producer's camera helper."""
    p=np.asarray(vertices,np.float64)[np.asarray(faces,np.int64)]
    c=np.asarray(camera["origin"],np.float64)
    right=np.asarray(camera["right"],np.float64)
    up=np.asarray(camera["screen_up"],np.float64)
    forward=np.asarray(camera["forward"],np.float64)
    basis=np.stack((right,up,forward))
    req(np.max(abs(basis@basis.T-np.eye(3)))<3e-5,
        "SOURCE_CAMERA_BASIS_NOT_ORTHONORMAL")
    t=p-c
    scale=192/(2*float(camera["half_extent"]))
    projected=np.empty((len(p),3,2),np.float64)
    projected[:,:,0]=96+np.sum(t*right,axis=-1)*scale
    projected[:,:,1]=96-np.sum(t*up,axis=-1)*scale
    z=np.sum(t*forward,axis=-1)
    return projected,z


def affine_depth(tri,depth,points):
    """Direct determinant barycentric interpolation, not matrix solve."""
    p=np.asarray(tri,np.float64)
    z=np.asarray(depth,np.float64)
    q=np.asarray(points,np.float64)
    a=p[1]-p[0];b=p[2]-p[0]
    det=a[0]*b[1]-a[1]*b[0]
    if abs(det)<1e-10:
        raise ValueError("INDEPENDENT_EXACT_SOURCE_DEGENERATE_CERTIFIED_TRIANGLE")
    t=q-p[0]
    beta=(t[...,0]*b[1]-t[...,1]*b[0])/det
    gamma=(a[0]*t[...,1]-a[1]*t[...,0])/det
    return z[0]+beta*(z[1]-z[0])+gamma*(z[2]-z[0]), \
           np.stack((1-beta-gamma,beta,gamma),axis=-1)


def clip_triangle_against_guarded_pixel(tri,left,right,top,bottom):
    """Independent convex polygon Sutherland–Hodgman clip, no Shapely."""
    polygon=[np.asarray(point,np.float64) for point in tri]
    for axis,bound,at_least in (
            (0,left,True),(0,right,False),(1,top,True),(1,bottom,False)):
        if not polygon:
            return []
        inside=lambda p: p[axis]>=bound if at_least else p[axis]<=bound
        nxt=[]
        prev=polygon[-1];prev_inside=inside(prev)
        for current in polygon:
            current_inside=inside(current)
            if prev_inside!=current_inside:
                denominator=current[axis]-prev[axis]
                req(abs(denominator)>1e-14,
                    "CLIP_NUMERIC_INTERSECTION_NOT_QUALIFIED")
                fraction=(bound-prev[axis])/denominator
                nxt.append(prev+fraction*(current-prev))
            if current_inside:
                nxt.append(current)
            prev=current;prev_inside=current_inside
        polygon=nxt
    return polygon


def verify_one_row(row,face_uv,source_tri,source_depth,camera,rectangles):
    fi=int(row["face_index"])
    x=int(row["pixel_x"]);y=int(row["pixel_y_top_first"])
    edit=int(row["source_uv_edit_operation_index"])
    req(0<=fi<len(source_tri) and 0<=x<RES and 0<=y<RES
        and 0<=edit<3,"SOURCE_FACE_PIXEL_ROW_INDEX_INVALID")
    left=x-GUARD;right=x+1+GUARD;top=y-GUARD;bottom=y+1+GUARD
    corners=np.asarray([
        (left,top),(right,top),(left,bottom),(right,bottom)
    ],np.float64)
    cand=source_tri[fi]
    z,bary=affine_depth(cand,source_depth[fi],corners)
    req(np.all(bary>=BARY_EPS),
        "CERTIFIED_SOURCE_FACE_NOT_COVERING_ENTIRE_PIXEL")
    req(np.min(z)>float(camera["clip_start"])+DEPTH_EPS and
        np.max(z)<float(camera["clip_end"])-DEPTH_EPS,
        "SOURCE_CAMERA_CLIP_COURT_FAIL")
    u=bary@face_uv[fi]
    rect=np.asarray(rectangles[edit],np.float64)
    req(np.all(u>rect[:2]/512+UV_EPS) and
        np.all(u<rect[2:]/512-UV_EPS),
        "ORIGINAL_SOURCE_FACE_UV_ADDRESS_NOT_IN_EDIT_TILE")
    # Entire original source triangle set rechecked, not only chosen ray hits.
    other_boxes=(
        (np.min(source_tri[:,:,0],axis=1)<=right)&
        (np.max(source_tri[:,:,0],axis=1)>=left)&
        (np.min(source_tri[:,:,1],axis=1)<=bottom)&
        (np.max(source_tri[:,:,1],axis=1)>=top)&
        (np.min(source_depth,axis=1)<=np.max(z)+DEPTH_EPS)
    )
    other_boxes[fi]=False
    for g in np.flatnonzero(other_boxes):
        gt=source_tri[g]
        aa=gt[1]-gt[0];bb=gt[2]-gt[0]
        det=aa[0]*bb[1]-aa[1]*bb[0]
        if abs(det)<1e-10:
            # Even a line/point might meet the guarded pixel; conservative.
            raise ValueError("INDEPENDENT_EXACT_SOURCE_DEGENERATE_NEARER_SOURCE_OCCLUDER")
        poly=clip_triangle_against_guarded_pixel(gt,left,right,top,bottom)
        if not poly:
            continue
        points=np.asarray(poly,np.float64)
        z_other,_=affine_depth(gt,source_depth[g],points)
        z_first,_=affine_depth(cand,source_depth[fi],points)
        req(np.all(z_other-z_first>DEPTH_EPS),
            "ORIGINAL_SOURCE_FACE_DEPTH_DOMINANCE_FALSIFIED")


def court(root,producer_path,output):
    root=Path(root);producer_path=Path(producer_path)
    for file,pin in PINS.items():
        req(sha(root/file)==pin,"ORIGINAL_SOURCE_BYTES_SHA_PIN:"+file)
    source=json.loads(producer_path.read_text())
    v1=json.loads((root/"v1-producer.json").read_text())
    v1c=json.loads((root/"v1-independent-court.json").read_text())
    manifest=json.loads((root/"physical-manifest.json").read_text())
    plan=json.loads((root/"uv-plan.json").read_text())
    policy=json.loads((root/"source-policy.json").read_text())
    binding=json.loads((root/"uv-binding.json").read_text())
    cams=json.loads((root/"camera.json").read_text())
    req(source["schema"]==SOURCE
        and source["status"]==
        "EXACT_LOCAL_SOURCE_DEPTH_PIXEL_UV_EDIT_TRUTH_PRODUCED__INDEPENDENT_COURT_REQUIRED"
        and source["original_source_sha256"]==CAT
        and source["original_parent_sha256"]==PINS
        and source["strict_original_source_pixel_guard_px"]==GUARD
        and source["strict_source_uv_interior_guard_texels"]==1
        and source["new_original_source_renders"]==0
        and source["source_artist_drawable_ownership"]=="UNKNOWN"
        and source["source_original_faces_without_positive_witness"]=="UNKNOWN__NOT_FALSE"
        and all(source[key] is False for key in (
            "graphics_owner_authorized","iris_training_authorized",
            "tessa_training_authorized","training_authorized",
            "materialization_authorized"))
        and source["same_drawable_training_pairs_qualified"]==0
        and source["different_drawable_training_pairs_qualified"]==0
        and source["independent_original_source_face_pixel_V1_subset_preserved"] is True
        and v1c["status"]==
         "PASS_ORIGINAL_FACE_PIXEL_UV_EDIT_CAUSALITY_CONSERVATIVE_SUBSET__OWNER_UNKNOWN"
        and v1c["source_producer_witness_sha256"]==PINS["v1-producer.json"]
        and v1c["source_original_face_pixel_court_count"]==608,
        "UNVERIFIED_SOURCE_PRODUCER_OR_ARTIST_PROMOTION")
    req(binding["original_glb_sha256"]==CAT
        and binding["source_uv_comparison"]["matched"] is True
        and binding["original_teacher_sha256"]==PINS["teacher.npz"]
        and policy["source_sha256"]==CAT
        and policy["material_mode"]=="unlit_preserve_alpha"
        and manifest["source_original_sample_npz_sha256"]==PINS["physical.npz"]
        and manifest["original_source_rest_only"] is True
        and manifest["cycles_subset_indices"]==[0,4,8,12,16,20,24,28]
        and manifest["camera_indices"]==list(VIEWS)
        and plan["source_original_glb_sha256"]==CAT,
        "SOURCE_NATIVE_GLTF_AND_PHYSICAL_CYCLES_PARENT_SCOPE")
    with np.load(root/"teacher.npz",allow_pickle=False) as t, \
         np.load(root/"rest.npz",allow_pickle=False) as rest, \
         np.load(root/"face-uv.npz",allow_pickle=False) as u, \
         np.load(root/"physical.npz",allow_pickle=False) as samples:
        faces=np.asarray(t["faces"],np.int64)
        verts=np.asarray(t["rest_vertices_source_world"],np.float64)
        face_uv=np.asarray(u["face_uv"],np.float64)
        phys=np.asarray(samples["source_uv_edit_cycles_sample_rgba"],np.float32)
        req(np.array_equal(faces,rest["faces"])
            and np.array_equal(verts,rest["rest_vertices_source_world"])
            and faces.shape==(684,3) and face_uv.shape==(684,3,2)
            and phys.shape==(4,8,8,RES,RES,4),
            "SOURCE_INDEXED_FACES_UV_OR_PHYSICAL_SAMPLES_INVALID")
    rects=[r["source_rgba_png_pixel_rectangle_xyxy_exclusive"]
            for r in plan["selected_original_source_uv_edits"]]
    projected={view:direct_source_triangles(
        verts,faces,cams["cameras"][view]) for view in VIEWS}
    rows=source["source_face_pixel_uv_edit_private_rows"]
    req(isinstance(rows,list) and len(rows)>608,
        "EXPECTED_MORE_THAN_CONSERVATIVE_SOURCE_FACE_WITNESSES")
    census={v:Counter() for v in VIEWS}
    seen=set();faces_seen=set();total_positive=total_other=0
    for row in rows:
        view=int(row["original_camera_view"])
        req(view in VIEWS and set(row)=={
            "original_camera_view","face_index","pixel_x",
            "pixel_y_top_first","source_uv_edit_operation_index"},
            "SOURCE_WITNESS_LOCAL_ADDRESS_OR_VIEW_INVALID")
        addr=(view,row["pixel_y_top_first"],row["pixel_x"])
        req(addr not in seen,"ORIGINAL_SOURCE_PIXEL_AMBIGUOUS_FACE")
        seen.add(addr)
        faces_seen.add(row["face_index"])
        xy,z=projected[view]
        verify_one_row(row,face_uv,xy,z,cams["cameras"][view],rects)
        x=int(row["pixel_x"]);y=int(row["pixel_y_top_first"])
        edit=int(row["source_uv_edit_operation_index"])
        # Original Cycles samples same edit/controller/view sample index,
        # original EXR byte raster BOTTOM-FIRST. No model estimate.
        obs=phys[VIEWS.index(view),:,:,RES-1-y,x,:].astype(np.float64)
        obs[...,:3]*=obs[...,3:4]
        req(np.all(abs(obs[...,3]-1)<TOL),
            "ACTUAL_SOURCE_MATERIAL_NOT_OPAQUE_IN_WITNESS")
        delta=np.max(abs(obs[:,1:4]-obs[:,0:1]),axis=-1)
        req(np.all(delta[:,edit]>TOL)
            and np.all(delta[:,[i for i in range(3) if i!=edit]]<=TOL),
            "ACTUAL_SOURCE_INTERVENTION_NOT_ISOLATED_TO_FACE_UV_ADDRESS")
        census[view]["source_face_pixel_witnesses"]+=1
        census[view]["source_positive_same_sample_edits"]+=8
        census[view]["source_other_edits_excluded"]+=16
        census[view][f"original_source_edit_{edit}_pixels"]+=1
        total_positive+=8
        total_other+=16
    v1keys={
        (r["original_camera_view"],r["pixel_y_top_first"],r["pixel_x"],
         r["face_index"],r["source_uv_edit_operation_index"])
        for r in v1["source_original_face_pixel_witness_rows_PRIVATE"]
    }
    newkeys={
        (r["original_camera_view"],r["pixel_y_top_first"],r["pixel_x"],
         r["face_index"],r["source_uv_edit_operation_index"])
        for r in rows
    }
    req(len(v1keys)==608 and v1keys<=newkeys,
        "V1_608_PHYSICALLY_COURTED_ORIGINAL_SOURCE_ROWS_NOT_PRESERVED")
    counts=source["source_counts"]
    byedit=[sum(census[v][f"original_source_edit_{i}_pixels"] for v in VIEWS)
            for i in range(3)]
    req(len(rows)==counts["qualified_original_source_face_pixels"]
        and len(faces_seen)==counts["qualified_original_unique_source_faces"]
        and total_positive==counts[
          "qualified_original_source_same_sample_positive_edit_effect_incidents"]
        and total_other==counts[
          "qualified_original_source_other_edit_absence_obligations"]
        and byedit==counts["original_source_face_pixels_by_atlas_edit"]
        and counts["V1_qualified_source_face_pixels_preserved"]==608
        and all(census[v]["source_face_pixel_witnesses"]==
               source["per_source_view"][ix]["face_pixels"]
               for ix,v in enumerate(VIEWS)),
        "INDEPENDENT_SOURCE_EXACT_LOCAL_DEPTH_CENSUS_DRIFT")
    receipt={
        "schema":SCHEMA,
        "status":PASS,
        "original_source_sha256":CAT,
        "producer_source_witness_sha256":sha(producer_path),
        "original_source_parent_sha256":PINS,
        "source_original_camera_views":list(VIEWS),
        "original_source_distinct_faces_independently_courted":len(faces_seen),
        "original_source_pixel_face_uv_witnesses_independently_courted":len(rows),
        "original_source_physically_observable_edit_sample_positive":total_positive,
        "original_source_non_target_edit_sample_absence":total_other,
        "original_source_face_pixels_by_edit":byedit,
        "independent_original_pixel_view_count":len(VIEWS),
        "independent_v1_conservative_physical_witnesses_preserved":608,
        "independent_source_per_view_recount":{str(v):dict(census[v]) for v in VIEWS},
        "independent_geometry_court":
            "NO_V2_IMPORT__CUSTOM_SUTHERLAND_HODGMAN__ALL_684_SOURCE_TRIANGLES__AFFINE_DEPTH",
        "original_source_256_full_Cycles_renders_reused":True,
        "new_original_source_Cycles_renders":0,
        "artist_drawable_owner":"UNKNOWN",
        "unwitnessed_original_faces_and_pixels":"UNKNOWN__NOT_FALSE",
        "source_exact_Cycles_sample_positions":"UNKNOWN__NOT_INFERRED",
        "iris_student_training_authorized":False,
        "graphics_owner_authorized":False,
        "same_drawable_training_pairs_qualified":0,
        "different_drawable_training_pairs_qualified":0,
        "iris_training_authorized":False,
        "tessa_training_authorized":False,
        "training_authorized":False,
        "materialization_authorized":False,
    }
    dest=Path(output);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(receipt,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({
        "status":PASS,"face_count":len(faces_seen),"source_face_pixels":len(rows),
        "true_physical_edit_samples":total_positive,
        "excluded_other_edit_samples":total_other,
        "source_pixels_by_edit":byedit,
        "artist_owner":"UNKNOWN","training_authorized":False,
    },sort_keys=True),flush=True)
    return receipt


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source-root",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    court(a.source_root,a.report,a.output)
