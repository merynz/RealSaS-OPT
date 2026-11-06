#!/usr/bin/env python3
"""Original Cat authored-pose source face/UV causal transport truth V1.

Reuses SHA-sealed V3 physically rendered Cycles source edit samples at authored
teacher pose ordinals 0/14, even cameras V0/V2/V4/V6. Native source indexed
face and GLTF UV are held immutable. Source face first hit is proved over an
entire guarded pixel, with exact projected triangle ∩ pixel affine depth.
For a source barycentric anchor at pose0 pixel CENTER, project the SAME face
and barycentric source UV anchor through pose14, requiring the containing
pose14 pixel independently passes whole-pixel source face/UV/physical-edit
court. This is not exact stochastic Cycles ray correspondence.

NO new source renderer, artist layer partition, student truth or training.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib,json
from pathlib import Path
import numpy as np
from tools.corpus.measure_original_source_face_pixel_uv_exact_depth_v2 import (
    exact_depth_original_face_uv_rows,
)
from tools.corpus.measure_original_source_face_pixel_uv_causality_v1 import (
    project_original,triangle_corner_barycentric,
)

SCHEMA="RealSaS.OriginalCatPhysicalPoseSubfaceUVAnchorTransport.v1"
SOURCE="1e73aaf7d497dd7d2946badf92d189d2333f163e66dced91feb446bc9bdee850"
PINS={
 "teacher.npz":"c2792c8efb36c4ccdf9178c6b933351b50364bbd28707e9365a3c115db00d603",
 "face-uv.npz":"7f3afdc4f0f848de3aa06bda2220151427d352368a0f235ca09d5d96f7834787",
 "uv-binding.json":"b966a3793c682cb06ec70341e659aabbded9343ace3d645c08643de40f2799d7",
 "camera.json":"300b935a47379f6c44d4b4356554d1c32a86ca3b732a62405c73da9e0ec6adf0",
 "uv-plan.json":"8be8fccde3d9cc97f7ba452fd50c1a9f4ebea2e34da590a22d30872b2b0f497b",
 "v3-manifest.json":"89dcb4d516e5579e766b19bd2d25eb31cbfaaa64f27ce34632405d99ce6874bc",
 "v3-court.json":"0171e9ec836d7b3c44ec64abcc59f712ebd33a87c2ad4edf6188976dc08e272c",
 "v3-samples.npz":"b2bb4fa9fc0b777bb8418aebc729123616b774b55eeb215ba17651e7f3448e0d",
 "v18u-court.json":"6e864c460aa4b27acd243787f4cee992018e90a1c2e7cdc2d39c590ea0af564a",
}
POSES=(0,14)
VIEWS=(0,2,4,6)
OFFSETS=(0,16)
RES=192
TOL=.025
FLAGS=("graphics_owner_authorized","iris_training_authorized",
       "tessa_training_authorized","training_authorized","materialization_authorized")

def req(cond,code):
    if not cond:raise ValueError("POSE_SUBFACE_SOURCE_TRANSPORT_"+code)

def sha(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

def physical_source_witnesses(rows,frame,pose_ordinal,view):
    """The two ORIGINAL same-index full-scene Cycles frames, no estimators."""
    a=np.asarray(frame,np.float32)
    req(a.shape==(2,4,RES,RES,4) and np.isfinite(a).all(),
        "SOURCE_V3_CYCLES_SCOPE")
    if not rows:return {"source_face_pixel_count":0,
                        "source_positive_sample_count":0,
                        "source_other_edit_absence_count":0,
                        "source_edit_pixel_census":[0,0,0],
                        "pose_ordinal":pose_ordinal,"view":view}
    ys=np.array([r["pixel_y_top_first"] for r in rows],np.int64)
    xs=np.array([r["pixel_x"] for r in rows],np.int64)
    op=np.array([r["source_uv_edit_operation_index"] for r in rows],np.int64)
    req(np.all((ys>=0)&(ys<RES)&(xs>=0)&(xs<RES)) and
        np.all((op>=0)&(op<3)),
        "SOURCE_FACE_PIXEL_EDIT_INDEX_INVALID")
    samples=a[:,:,RES-1-ys,xs].astype(np.float64)  # original Blender EXR is bottom-first
    samples[...,:3]*=samples[...,3:4]
    req(np.all(np.abs(samples[...,3]-1)<TOL),
        "SOURCE_ORIGINAL_AUTHORED_POSE_ALPHA_NOT_OPAQUE")
    delta=np.max(np.abs(samples[:,1:4]-samples[:,0:1]),axis=-1)
    mask=np.arange(3)[None,:,None]==op[None,None,:]
    req(np.all(delta[mask.repeat(2,axis=0)]>TOL),
        "SOURCE_ORIGINAL_POSE_EDIT_NOT_PHYSICALLY_OBSERVED")
    req(np.all(delta[~mask.repeat(2,axis=0)]<=TOL),
        "SOURCE_ORIGINAL_POSE_NON_TARGET_EDIT_FALSE_POSITIVE")
    return {
        "source_face_pixel_count":len(rows),
        "source_positive_sample_count":len(rows)*2,
        "source_other_edit_absence_count":len(rows)*4,
        "source_edit_pixel_census":[int(np.count_nonzero(op==i)) for i in range(3)],
        "pose_ordinal":pose_ordinal,"view":view,
    }

def transport_anchor_rows(pose0_rows,pose14_rows,tri0,tri14,view):
    """Barycentric position is the source identity, NOT a tracked Cycles ray."""
    target={
        (r["face_index"],r["source_uv_edit_operation_index"],
         r["pixel_y_top_first"],r["pixel_x"])
        for r in pose14_rows
    }
    result=[]
    for row in pose0_rows:
        fi=row["face_index"];edit=row["source_uv_edit_operation_index"]
        x=row["pixel_x"];y=row["pixel_y_top_first"]
        center=np.asarray([[x+.5,y+.5]],np.float64)
        bary=triangle_corner_barycentric(tri0[fi],center)
        req(bary is not None and np.all(bary>=0),
            "ORIGINAL_POSE0_CERTIFIED_CENTER_OUTSIDE_TRIANGLE")
        q=(bary[0]@tri14[fi]).astype(np.float64)
        tx,ty=int(np.floor(q[0])),int(np.floor(q[1]))
        if (fi,edit,ty,tx) not in target:
            continue
        displacement=float(np.linalg.norm(q-center[0]))
        result.append({
            "original_face_id":int(fi),
            "original_uv_edit_index":int(edit),
            "original_camera_view":int(view),
            "pose0_pixel_xy":[int(x),int(y)],
            "pose14_qualified_pixel_xy":[tx,ty],
            "pose14_transported_barycentric_screen_xy":q.tolist(),
            "pose0_center_source_barycentric":bary[0].tolist(),
            "screen_displacement_px":displacement,
            "source_pose0_and_pose14_independent_physical_positive":True,
            "same_source_face_UV_anchor":True,
            "exact_Cycles_sample_ray_UV_correspondence":"UNKNOWN_NOT_ASSERTED",
            "original_artist_drawable_owner":"UNKNOWN",
        })
    return result

def run(root,output):
    root=Path(root)
    for f,d in PINS.items():
        req(sha(root/f)==d,"IMMUTABLE_SOURCE_BYTE_SHA:"+f)
    read=lambda f:json.loads((root/f).read_text())
    m=read("v3-manifest.json");court=read("v3-court.json")
    bind=read("uv-binding.json");cams=read("camera.json")
    plan=read("uv-plan.json");incidence=read("v18u-court.json")
    req(m["schema"]=="RealSaS.Corpus370OriginalCatAuthoredPoseUVPhysicalSamples.v3"
        and m["status"]=="PHYSICAL_SOURCE_NATIVE_POSE_UV_EDITS_RENDERED__INDEPENDENT_COURT_REQUIRED"
        and court["status"]=="PASS_ORIGINAL_CAT_POSE_UV_EDIT_PHYSICAL_SOURCE__FIXED_SCREEN_CLASS_VIOLATED"
        and court["source_original_blender_physical_render_manifest_sha256"]==PINS["v3-manifest.json"]
        and court["source_original_blender_physical_sample_sha256"]==PINS["v3-samples.npz"]
        and court["source_uv_surface_attached_edit_native_source_status"]==
            "SATISFIED_PHYSICAL_ORIGINAL_SOURCE_WITNESS_ONLY"
        and m["source_samples_npz_sha256"]==PINS["v3-samples.npz"]
        and m["original_source_sha256"]==SOURCE
        and m["original_native_teacher_sha256"]==PINS["teacher.npz"]
        and m["original_camera_sha256"]==PINS["camera.json"]
        and m["original_source_uv_plan_sha256"]==PINS["uv-plan.json"]
        and m["original_source_sample_count"]==64
        and m["teacher_pose_sample_ordinals"]==list(POSES)
        and m["original_calibrated_view_indices"]==list(VIEWS)
        and m["original_same_cycles_sample_offsets"]==list(OFFSETS)
        and m["original_source_uv_edit_arms"]==["BASE","SINGLE_0","SINGLE_1","SINGLE_2"]
        and m["original_source_pose_geometry_replay_receipts"][0]["source_geometry_max_abs_error"]==0
        and m["original_source_pose_geometry_replay_receipts"][1]["source_geometry_max_abs_error"]==0
        and m["source_pose_world_max_displacement"]>.2
        and bind["original_glb_sha256"]==SOURCE
        and bind["original_teacher_sha256"]==PINS["teacher.npz"]
        and bind["source_uv_comparison"]["matched"] is True
        and plan["source_original_glb_sha256"]==SOURCE
        and plan["source_original_camera_sha256"]==PINS["camera.json"]
        and incidence["status"]==
            "PASS_SOURCE_SUBFACE_EDIT_INCIDENCE_V18S_CROSS_SCOPE_RECOUNT__NO_ARTIST_OWNER"
        and incidence["original_source_sha256"]==SOURCE
        and all(v.get(f) is False for v in (m,court,incidence) for f in FLAGS),
        "SOURCE_V3_NATIVE_POSE_OR_V18U_AUTHORITY_INVALID")
    with np.load(root/"teacher.npz",allow_pickle=False) as teacher, \
         np.load(root/"face-uv.npz",allow_pickle=False) as ut, \
         np.load(root/"v3-samples.npz",allow_pickle=False) as rr:
        xyz=np.asarray(teacher["posed_vertices_source_world"],np.float64)
        faces=np.asarray(teacher["faces"],np.int64)
        uv=np.asarray(ut["face_uv"],np.float64)
        actual=np.asarray(rr["source_native_pose_uv_rgba"],np.float32)
    req(xyz.shape==(32,1204,3) and faces.shape==(684,3) and
        uv.shape==(684,3,2) and actual.shape==(2,4,2,4,RES,RES,4),
        "SOURCE_ORIGINAL_POSE_FACE_UV_AND_RENDER_SHAPES")
    rectangles=[e["source_rgba_png_pixel_rectangle_xyxy_exclusive"]
                for e in plan["selected_original_source_uv_edits"]]
    per_pose=[]
    rows_by_condition={}
    tris={}
    all_rows=[]
    for pi,pose in enumerate(POSES):
        for vi,view in enumerate(VIEWS):
            rows,geom=exact_depth_original_face_uv_rows(
                xyz[pose],faces,uv,cams["cameras"][view],rectangles)
            physical=physical_source_witnesses(rows,actual[pi,vi],pose,view)
            rows_by_condition[(pose,view)]=rows
            tris[(pose,view)],_=project_original(
                xyz[pose],faces,cams["cameras"][view])
            per_pose.append({**physical,**geom,
                             "source_distinct_original_faces":
                                 len({r["face_index"] for r in rows})})
            all_rows.extend([{"pose_ordinal":int(pose),
                              "original_camera_view":int(view),**r}
                             for r in rows])
            print(json.dumps({"pose":pose,"view":view,
                "physical_positive_face_pixels":len(rows)}),flush=True)
    anchors=[]
    for view in VIEWS:
        anchors.extend(transport_anchor_rows(
            rows_by_condition[(0,view)],rows_by_condition[(14,view)],
            tris[(0,view)],tris[(14,view)],view))
    req(bool(anchors),"NO_SAME_UV_ANCHOR_TRANSPORT_PHYSICAL_WITNESSES")
    moved=sum(r["screen_displacement_px"]>=1.0 for r in anchors)
    face_scope={p:len({r["face_index"] for v in VIEWS
          for r in rows_by_condition[(p,v)]}) for p in POSES}
    counts={
        "qualified_original_face_pixel_pose0_count":
          sum(len(rows_by_condition[(0,v)]) for v in VIEWS),
        "qualified_original_face_pixel_pose14_count":
          sum(len(rows_by_condition[(14,v)]) for v in VIEWS),
        "qualified_original_distinct_faces_pose0":face_scope[0],
        "qualified_original_distinct_faces_pose14":face_scope[14],
        "original_barycentric_UV_anchors_physically_supported_both_poses":
          len(anchors),
        "source_anchor_transport_pixel_displacement_ge_1":moved,
        "source_anchor_transport_distinct_original_faces":
          len({r["original_face_id"] for r in anchors}),
        "source_anchor_transport_distinct_face_camera_edit":
          len({(r["original_face_id"],r["original_camera_view"],r["original_uv_edit_index"]) for r in anchors}),
        "source_anchor_transport_by_edit":[sum(r["original_uv_edit_index"]==i
                                             for r in anchors) for i in range(3)],
        "source_anchor_transport_max_pixel_displacement":
          max(r["screen_displacement_px"] for r in anchors),
    }
    out={
      "schema":SCHEMA,
      "status":"SOURCE_NATIVE_POSE_SUBFACE_UV_ANCHOR_TRANSPORT_PRODUCED__INDEPENDENT_COURT_REQUIRED",
      "original_source_sha256":SOURCE,
      "original_parent_bytes_sha256":PINS,
      "original_pose_ordinals":list(POSES),
      "original_camera_views":list(VIEWS),
      "same_cycles_sample_offsets":list(OFFSETS),
      "original_source_physical_64_cycles_frames_reused":True,
      "new_original_blender_renders":0,
      "source_original_face_first_hit_rule":"EXACT_TRIANGLE_PIXEL_INTERSECTION_AFFINE_DEPTH__0p05PX_GUARD",
      "source_original_uv_rule":"GLTF_FACE_CORNER_UV__1_TEXEL_INTERIOR",
      "source_original_barycentric_anchor":"POSE0_PIXEL_CENTER_TO_SAME_POSE14_FACE_BARYCENTRIC_POINT",
      "source_original_camera_ray_sample_positions":"UNKNOWN__NOT_RECONSTRUCTED",
      "original_source_measured_pixel_witness_rows_PRIVATE":all_rows,
      "original_source_transported_uv_anchors_PRIVATE":anchors,
      "source_per_original_pose_camera":per_pose,
      "counts":counts,
      "source_appearance_transport_scope":"TWO_ORIGINAL_AUTHORED_POSES_UNLIT_OPAQUE_LINKED_TEXTURE",
      "source_appearance_at_unobserved_barycentric_points":"UNKNOWN",
      "source_hide_reveal_proven_by_missing_pixel_certificate":False,
      "source_original_artist_layer_ownership":"UNKNOWN",
      "same_drawable_training_pairs_qualified":0,
      "different_drawable_training_pairs_qualified":0,
      **{f:False for f in FLAGS},
    }
    p=Path(output);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(out,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":out["status"],"counts":counts,
            "artist_owner":"UNKNOWN","training_authorized":False},
            sort_keys=True),flush=True)
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    run(a.root,a.output)
if __name__=="__main__":main()
