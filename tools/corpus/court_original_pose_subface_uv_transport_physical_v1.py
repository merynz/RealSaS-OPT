#!/usr/bin/env python3
"""Independent original authored-pose physical face→UV anchor transport court.

No import of V1 producer or Shapely. Uses the earlier independent V2
Sutherland-Hodgman all-original-triangle court, original teacher geometry,
the original SHA-sealed independent V3 physical Cycles source edit oracle.
Rechecks every source pixel and EVERY transported original barycentric
anchor. No exact stochastic Cycles sample ray inference or artist owner.
"""
from __future__ import annotations
import argparse,hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
from tools.corpus.court_original_source_face_pixel_uv_exact_depth_v2 import (
    direct_source_triangles,verify_one_row,affine_depth
)

SCHEMA="RealSaS.IndependentCatPhysicalPoseSubfaceUVAnchorTransportCourt.v1"
PARENT="RealSaS.OriginalCatPhysicalPoseSubfaceUVAnchorTransport.v1"
PASS="PASS_ORIGINAL_CAT_POSE0_POSE14_SAME_FACE_UV_ANCHOR_PHYSICAL_TRANSPORT__OWNER_UNKNOWN"
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
VIEWS=(0,2,4,6);POSES=(0,14);RES=192;TOL=.025
AUTH=("graphics_owner_authorized","iris_training_authorized",
      "tessa_training_authorized","training_authorized","materialization_authorized")

def req(c,why):
    if not c:raise ValueError("INDEPENDENT_POSE_UV_TRANSPORT_"+why)

def sha(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

def court(root,producer,output):
    root=Path(root);producer=Path(producer)
    for name,pin in PINS.items():
        req(sha(root/name)==pin,"SEALED_PARENT_SOURCE_SHA:"+name)
    src=json.loads(producer.read_text())
    m=json.loads((root/"v3-manifest.json").read_text())
    v3=json.loads((root/"v3-court.json").read_text())
    cam=json.loads((root/"camera.json").read_text())
    plan=json.loads((root/"uv-plan.json").read_text())
    bind=json.loads((root/"uv-binding.json").read_text())
    u=json.loads((root/"v18u-court.json").read_text())
    req(src["schema"]==PARENT
        and src["status"]==
          "SOURCE_NATIVE_POSE_SUBFACE_UV_ANCHOR_TRANSPORT_PRODUCED__INDEPENDENT_COURT_REQUIRED"
        and src["original_source_sha256"]==
             m["original_source_sha256"]==u["original_source_sha256"]==
             bind["original_glb_sha256"]==
             "1e73aaf7d497dd7d2946badf92d189d2333f163e66dced91feb446bc9bdee850"
        and src["original_parent_bytes_sha256"]==PINS
        and src["original_pose_ordinals"]==list(POSES)
        and src["original_camera_views"]==list(VIEWS)
        and src["same_cycles_sample_offsets"]==[0,16]
        and src["new_original_blender_renders"]==0
        and src["original_source_physical_64_cycles_frames_reused"] is True
        and src["source_original_artist_layer_ownership"]=="UNKNOWN"
        and src["source_hide_reveal_proven_by_missing_pixel_certificate"] is False
        and src["source_original_camera_ray_sample_positions"]==
            "UNKNOWN__NOT_RECONSTRUCTED"
        and all(src[z] is False for z in AUTH)
        and src["same_drawable_training_pairs_qualified"]==0
        and src["different_drawable_training_pairs_qualified"]==0
        and v3["status"]==
            "PASS_ORIGINAL_CAT_POSE_UV_EDIT_PHYSICAL_SOURCE__FIXED_SCREEN_CLASS_VIOLATED"
        and v3["source_original_blender_physical_sample_sha256"]==PINS["v3-samples.npz"]
        and v3["source_original_blender_physical_render_manifest_sha256"]==PINS["v3-manifest.json"]
        and m["source_samples_npz_sha256"]==PINS["v3-samples.npz"]
        and m["original_native_teacher_sha256"]==PINS["teacher.npz"]
        and m["original_camera_sha256"]==PINS["camera.json"]
        and m["original_source_uv_plan_sha256"]==PINS["uv-plan.json"]
        and m["teacher_pose_sample_ordinals"]==[0,14]
        and m["original_calibrated_view_indices"]==list(VIEWS)
        and m["original_same_cycles_sample_offsets"]==[0,16]
        and len(m["original_source_pose_geometry_replay_receipts"])==2,
        "SEALED_PRODUCER_OR_ORIGINAL_PHYSICAL_SCOPE")
    # Explicit chained validation separate from the compound to avoid
    # accidental boolean-precedence bypass of any single source obligation.
    req(m["original_source_pose_geometry_replay_receipts"][0]["source_geometry_max_abs_error"]==0
        and m["original_source_pose_geometry_replay_receipts"][1]["source_geometry_max_abs_error"]==0
        and m["source_pose_world_max_displacement"]>.2
        and plan["source_original_glb_sha256"]==src["original_source_sha256"]
        and bind["original_teacher_sha256"]==PINS["teacher.npz"]
        and bind["source_uv_comparison"]["matched"] is True
        and u["status"]=="PASS_SOURCE_SUBFACE_EDIT_INCIDENCE_V18S_CROSS_SCOPE_RECOUNT__NO_ARTIST_OWNER"
        and all(t.get(k) is False for t in (m,v3,u) for k in AUTH),
        "SOURCE_V3_AUTHORITIES_OR_SOURCE_POSE_MOTION_DRIFT")
    with np.load(root/"teacher.npz",allow_pickle=False) as t, \
         np.load(root/"face-uv.npz",allow_pickle=False) as uu, \
         np.load(root/"v3-samples.npz",allow_pickle=False) as rr:
        faces=np.asarray(t["faces"],np.int64)
        pts=np.asarray(t["posed_vertices_source_world"],np.float64)
        uv=np.asarray(uu["face_uv"],np.float64)
        source=np.asarray(rr["source_native_pose_uv_rgba"],np.float32)
    req(pts.shape==(32,1204,3)
        and faces.shape==(684,3)
        and uv.shape==(684,3,2)
        and source.shape==(2,4,2,4,RES,RES,4)
        and np.isfinite(source).all(),"ORIGINAL_SOURCE_FACE_PIXEL_ARRAYS_INVALID")
    rect=[e["source_rgba_png_pixel_rectangle_xyxy_exclusive"]
          for e in plan["selected_original_source_uv_edits"]]
    projected={}
    for pose in POSES:
        for view in VIEWS:
            projected[(pose,view)]=direct_source_triangles(
                pts[pose],faces,cam["cameras"][view])
    rows=src["original_source_measured_pixel_witness_rows_PRIVATE"]
    req(isinstance(rows,list) and len(rows)>0,
        "NO_ORIGINAL_SOURCE_EDIT_PIXEL_WITNESSES")
    valid=set()
    face_sets={p:set() for p in POSES}
    view_counter=Counter(); edit_counter=Counter()
    for row in rows:
        pose=row["pose_ordinal"];view=row["original_camera_view"]
        req(pose in POSES and view in VIEWS and set(row)=={
            "pose_ordinal","original_camera_view","face_index",
            "pixel_x","pixel_y_top_first","source_uv_edit_operation_index"},
            "ORIGINAL_SOURCE_FACE_EDIT_PIXEL_SCHEMA")
        fi=row["face_index"];x=row["pixel_x"];y=row["pixel_y_top_first"]
        op=row["source_uv_edit_operation_index"]
        key=(pose,view,fi,op,y,x)
        req(key not in valid,"DUPLICATE_ORIGINAL_SOURCE_PIXEL_FACE_EDIT_CLAIM")
        valid.add(key)
        xyz,depth=projected[(pose,view)]
        verify_one_row(row,uv,xyz,depth,cam["cameras"][view],rect)
        pi=POSES.index(pose);vi=VIEWS.index(view)
        physical=source[pi,vi,:,:,RES-1-y,x,:].astype(np.float64)
        physical[...,:3]*=physical[...,3:4]
        req(np.all(np.abs(physical[...,3]-1)<TOL),
            "ORIGINAL_SOURCE_OPAQUE_PIXEL_CHANGED")
        effects=np.max(np.abs(physical[:,1:]-physical[:,0:1]),axis=-1)
        req(np.all(effects[:,op]>TOL)
            and np.all(effects[:,[j for j in range(3) if j!=op]]<=TOL),
            "ORIGINAL_POSE_PHYSICAL_SOURCE_UV_EDIT_DIFFERENCE")
        face_sets[pose].add(fi);view_counter[(pose,view)]+=1
        edit_counter[(pose,op)]+=1
    # Important: no artificially matched pixel coordinates across poses:
    # transport a FACE BARYCENTRIC CENTER from pose0 to pose14, then find an
    # INDEPENDENTLY VERIFIED pose14 physical edited pixel at the transported
    # screen position. Exactly matched stochastic Cycles ray UV is UNKNOWN.
    anchor=src["original_source_transported_uv_anchors_PRIVATE"]
    p0keys={(r["original_camera_view"],r["face_index"],
            r["source_uv_edit_operation_index"],r["pixel_y_top_first"],
            r["pixel_x"]) for r in rows if r["pose_ordinal"]==0}
    common=set();move_ge_1=0;displacements=[];byedit=Counter()
    for row in anchor:
        view=row["original_camera_view"];fi=row["original_face_id"]
        edit=row["original_uv_edit_index"]
        x,y=row["pose0_pixel_xy"];tx,ty=row["pose14_qualified_pixel_xy"]
        k0=(view,fi,edit,y,x)
        req(k0 in p0keys and (14,view,fi,edit,ty,tx) in valid,
            "UNPROVED_ORIGINAL_SOURCE_POSE0_OR_POSE14_PIXEL")
        req(row["source_pose0_and_pose14_independent_physical_positive"] is True
            and row["same_source_face_UV_anchor"] is True
            and row["original_artist_drawable_owner"]=="UNKNOWN"
            and row["exact_Cycles_sample_ray_UV_correspondence"]==
                "UNKNOWN_NOT_ASSERTED","UNQUALIFIED_TRANSPORT_SOURCE_SCOPE")
        tri0=projected[(0,view)][0][fi]
        tri14=projected[(14,view)][0][fi]
        center=np.asarray([[x+.5,y+.5]],np.float64)
        _,bary=affine_depth(tri0,np.zeros(3),center)
        req(np.all(bary>=0),"ORIGINAL_SOURCE_POSE0_CENTER_NOT_INSIDE_FACE")
        q=bary[0]@tri14
        transported=np.asarray(row["pose14_transported_barycentric_screen_xy"],np.float64)
        claimed_bary=np.asarray(row["pose0_center_source_barycentric"],np.float64)
        req(np.max(np.abs(q-transported))<1e-7
            and np.max(np.abs(bary[0]-claimed_bary))<1e-7
            and int(np.floor(q[0]))==tx
            and int(np.floor(q[1]))==ty,
            "SOURCE_POSE_FACE_BARYCENTRIC_UV_ANCHOR_CORRUPTED")
        d=float(np.linalg.norm(q-center[0]))
        req(abs(d-row["screen_displacement_px"])<1e-7,
            "SOURCE_TRANSPORT_DISPLACEMENT_MEASUREMENT_FALSE")
        ak=(view,fi,edit,y,x)
        req(ak not in common,"DUPLICATE_TRANSPORTED_SOURCE_UV_ANCHOR")
        common.add(ak)
        move_ge_1+=int(d>=1.0)
        displacements.append(d);byedit[edit]+=1
    counts=src["counts"]
    req(counts["qualified_original_face_pixel_pose0_count"]==
            sum(view_counter[(0,v)] for v in VIEWS)
        and counts["qualified_original_face_pixel_pose14_count"]==
            sum(view_counter[(14,v)] for v in VIEWS)
        and counts["qualified_original_distinct_faces_pose0"]==len(face_sets[0])
        and counts["qualified_original_distinct_faces_pose14"]==len(face_sets[14])
        and counts["original_barycentric_UV_anchors_physically_supported_both_poses"]==len(anchor)
        and counts["source_anchor_transport_pixel_displacement_ge_1"]==move_ge_1
        and counts["source_anchor_transport_by_edit"]==[byedit[i] for i in range(3)]
        and counts["source_anchor_transport_distinct_original_faces"]==
           len({r["original_face_id"] for r in anchor})
        and counts["source_anchor_transport_distinct_face_camera_edit"]==
           len({(r["original_face_id"],r["original_camera_view"],r["original_uv_edit_index"])
                 for r in anchor})
        and abs(counts["source_anchor_transport_max_pixel_displacement"]-max(displacements))<1e-7
        and len(anchor)>0 and move_ge_1>0,
        "INDEPENDENT_CROSS_POSE_RECOUNT_ORIGINAL_SOURCE_CENSUS_DRIFT")
    result={
      "schema":SCHEMA,"status":PASS,
      "original_source_sha256":src["original_source_sha256"],
      "producer_source_sha256":sha(producer),
      "immutable_original_parent_receipts_sha256":PINS,
      "qualified_source_face_pixels_pose0":view_counter[(0,0)]+view_counter[(0,2)]+view_counter[(0,4)]+view_counter[(0,6)],
      "qualified_source_face_pixels_pose14":view_counter[(14,0)]+view_counter[(14,2)]+view_counter[(14,4)]+view_counter[(14,6)],
      "original_source_distinct_faces_pose0":len(face_sets[0]),
      "original_source_distinct_faces_pose14":len(face_sets[14]),
      "original_same_barycentric_UV_anchor_physical_transport_witnesses":len(anchor),
      "original_transported_anchors_moving_at_least_1_pixel":move_ge_1,
      "original_transported_distinct_source_faces":
            counts["source_anchor_transport_distinct_original_faces"],
      "original_transported_anchors_by_edit":[byedit[i] for i in range(3)],
      "original_source_pixel_edit_2sample_positives":len(rows)*2,
      "original_source_pixel_other_edit_2sample_absence_obligations":len(rows)*4,
      "original_source_same_sample_edited_source_64_physical_frames_reused":True,
      "new_original_source_Cycles_renders":0,
      "independent_geometry_method":"ALL_684_SOURCE_FACES_CUSTOM_CONVEX_PIXEL_CLIP__NO_V1_POSE_PRODUCER_IMPORT",
      "source_pose0_and_pose14_physical_Cycles_sample_effects_qualified":True,
      "source_exact_same_stochastic_uv_sample_ray_across_poses":"UNKNOWN_NOT_ASSERTED",
      "source_hidden_vs_unobserved_under_pose_change":"UNKNOWN_NOT_NEGATIVE",
      "original_source_artist_drawable_owner":"UNKNOWN",
      "same_drawable_training_pairs_qualified":0,
      "different_drawable_training_pairs_qualified":0,
      **{z:False for z in AUTH},
    }
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({
        "status":PASS,"pose0_source_face_pixels":result["qualified_source_face_pixels_pose0"],
        "pose14_source_face_pixels":result["qualified_source_face_pixels_pose14"],
        "physically_supported_same_UV_anchor_count":len(anchor),
        "moved_at_least_1px":move_ge_1,"owner":"UNKNOWN","training":False},sort_keys=True),flush=True)
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--producer",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    court(a.root,a.producer,a.output)
