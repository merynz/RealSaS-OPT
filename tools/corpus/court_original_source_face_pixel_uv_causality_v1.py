#!/usr/bin/env python3
"""Independent conservative source-face pixel and original UV edit physical court.

Does NOT import producer, recomputes every emitted proof row from raw immutable
source positions/face order/UV/camera and original same-Cycles physical RGBA.
Uses a slower direct all-other-triangle bound for each candidate row. Checks
each row (soundness) but does not claim to enumerate ALL certifiable pixels.
Original artist layer, entire visible face/pixel screen and unseen inference
remain UNKNOWN.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np

SCHEMA="RealSaS.IndependentOriginalSourceFacePixelUVEditCausalityCourt.v1"
SOURCE="RealSaS.OriginalSourceFacePixelUVInterventionWitness.v1"
PASS="PASS_ORIGINAL_FACE_PIXEL_UV_EDIT_CAUSALITY_CONSERVATIVE_SUBSET__OWNER_UNKNOWN"
ORIGINAL_SOURCE="1e73aaf7d497dd7d2946badf92d189d2333f163e66dced91feb446bc9bdee850"
PINS={
    "teacher.npz":"c2792c8efb36c4ccdf9178c6b933351b50364bbd28707e9365a3c115db00d603",
    "rest.npz":"748609aca5ba7c509aa390ed65baeca7612b8879aa00409735da3a96758379cf",
    "face-uv.npz":"7f3afdc4f0f848de3aa06bda2220151427d352368a0f235ca09d5d96f7834787",
    "camera.json":"300b935a47379f6c44d4b4356554d1c32a86ca3b732a62405c73da9e0ec6adf0",
    "source-policy.json":"c8fe4e307ddbbaf5cc0892cffc4267bd08e394f2299f4e70155ffc6207190a4b",
    "uv-plan.json":"8be8fccde3d9cc97f7ba452fd50c1a9f4ebea2e34da590a22d30872b2b0f497b",
    "physical.npz":"d422b7d7d082fba5deac5dd1239b8c6aec078f4d635a7bd5b1f10e514100b114",
    "physical-manifest.json":"8a56e47a09f5c1216de2c97e6f13d5e04385f7fe7d2e580d0085b2ab69dc2da1",
    "physical-court.json":"90b85b9c1a963b23f7c3037ca443975bf2d5c5d4123568d8aa8e54557e433c08",
    "source.glb":ORIGINAL_SOURCE,
    "uv-binding.json":"b966a3793c682cb06ec70341e659aabbded9343ace3d645c08643de40f2799d7",
}
VIEWS=(1,3,5,7)
N=192
TOL=.025


def req(cond,code):
    if not cond:raise ValueError("INDEPENDENT_SOURCE_FACE_EDIT_"+code)


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def direct_row_geometry_check(face_xyz,face_uv,camera,rect,face_idx,top_y,x,
                              *,guard=.05,uv_guard=1/512):
    """All-other-original-faces independent witness test.

    For each other face whose projected AABB overlaps enlarged pixel Q,
    assert its smallest possible depth exceeds candidate's LARGEST depth
    over Q. No stochastic sample position inferred. UV convex interpolation
    must lie completely inside the stated original source atlas edit rect.
    """
    xyz=np.asarray(face_xyz,np.float64)
    uv=np.asarray(face_uv,np.float64)
    req(xyz.ndim==3 and xyz.shape[1:]==(3,3) and uv.shape==(len(xyz),3,2),
        "ORIGINAL_TRIANGLE_ADDRESS_SHAPE")
    o=np.asarray(camera["origin"],np.float64)
    r=np.asarray(camera["right"],np.float64)
    up=np.asarray(camera["screen_up"],np.float64)
    fw=np.asarray(camera["forward"],np.float64)
    scale=2*float(camera["half_extent"])
    rel=xyz-o
    projected=np.stack((
        192*(.5+np.einsum("fck,k->fc",rel,r)/scale),
        192*(.5-np.einsum("fck,k->fc",rel,up)/scale)),axis=-1)
    distances=np.einsum("fck,k->fc",rel,fw)
    q=np.asarray([
        (x-guard,top_y-guard),(x+1+guard,top_y-guard),
        (x-guard,top_y+1+guard),(x+1+guard,top_y+1+guard)
    ],np.float64)
    tri=projected[face_idx]
    e0=tri[1]-tri[0];e1=tri[2]-tri[0]
    area=np.linalg.det(np.stack((e0,e1),axis=-1))
    req(abs(area)>1e-10,"DEGENERATE_CERTIFIED_ORIGINAL_FACE")
    delta=q-tri[0]
    a=np.linalg.solve(np.stack((e0,e1),axis=-1),delta.T).T
    weights=np.column_stack((1-a[:,0]-a[:,1],a))
    req(np.all(weights>=1e-7),"CANDIDATE_DOES_NOT_COVER_ENLARGED_PIXEL")
    z=weights@distances[face_idx]
    req(z.min()>camera["clip_start"]+1e-5 and
        z.max()<camera["clip_end"]-1e-5,"SOURCE_ORIGINAL_CAMERA_CLIPPING")
    zmax=float(z.max())
    # Independent direct all-triangle occlusion check, no producer envelope.
    pxmax,pymax=q.max(axis=0)
    pxmin,pymin=q.min(axis=0)
    occlusion=(
        (projected[:,:,0].min(axis=1)<=pxmax)
        &(projected[:,:,0].max(axis=1)>=pxmin)
        &(projected[:,:,1].min(axis=1)<=pymax)
        &(projected[:,:,1].max(axis=1)>=pymin)
        &(distances.min(axis=1)<=zmax+1e-5)
    )
    occlusion[face_idx]=False
    req(not np.any(occlusion),
        "ANOTHER_SOURCE_FACE_MIGHT_OCCLUDE_ENLARGED_PIXEL")
    uv_on_corners=weights@uv[face_idx]
    source_lo=np.asarray(rect[:2],np.float64)/512
    source_hi=np.asarray(rect[2:],np.float64)/512
    req(np.all(uv_on_corners>source_lo+uv_guard)
        and np.all(uv_on_corners<source_hi-uv_guard),
        "SOURCE_FACE_UV_CROSSES_EDIT_BOUNDARY")
    return True


def independently_court(root,report_path,output):
    root=Path(root);report_path=Path(report_path)
    for name,pin in PINS.items():
        req(sha(root/name)==pin,"IMMUTABLE_ORIGINAL_SOURCE_PARENT_SHA:"+name)
    q=json.loads(report_path.read_text())
    plan=json.loads((root/"uv-plan.json").read_text())
    camera=json.loads((root/"camera.json").read_text())
    manifest=json.loads((root/"physical-manifest.json").read_text())
    policy=json.loads((root/"source-policy.json").read_text())
    court=json.loads((root/"physical-court.json").read_text())
    bind=json.loads((root/"uv-binding.json").read_text())
    req(q["schema"]==SOURCE
        and q["source_original_glb_sha256"]==ORIGINAL_SOURCE
        and q["original_parent_bytes_sha256"]==PINS
        and q["original_camera_views"]==list(VIEWS)
        and q["source_pixel_sample_offsets"]==[0,4,8,12,16,20,24,28]
        and q["whole_pixel_guard_px"]==.05
        and q["original_source_uv_interior_guard_texels"]==1
        and q["re_rendered_source_frames"]==0
        and q["first_visible_original_triangle"]==
            "SUFFICIENT_ENTIRE_PIXEL_PROOF_ONLY__NOT_COMPLETE"
        and q["source_artist_layer_or_drawable_owner"]=="UNKNOWN"
        and q["source_unwitnessed_faces_and_uv_edits"]=="UNKNOWN__NOT_FALSE"
        and q["iris_training_authorized"] is False
        and q["training_authorized"] is False
        and q["materialization_authorized"] is False
        and q["same_drawable_training_pairs_qualified"]==0
        and q["different_drawable_training_pairs_qualified"]==0,
        "SOURCE_PRODUCER_SCOPE_OR_OWNER_FRAUD")
    req(plan["source_original_glb_sha256"]==ORIGINAL_SOURCE
        and manifest["original_source_sha256"]==ORIGINAL_SOURCE
        and court["source_original_glb_sha256"]==ORIGINAL_SOURCE
        and bind["original_glb_sha256"]==ORIGINAL_SOURCE
        and bind["source_uv_comparison"]["matched"] is True
        and bind["original_teacher_sha256"]==PINS["teacher.npz"]
        and manifest["source_original_sample_npz_sha256"]==PINS["physical.npz"]
        and manifest["original_camera_bundle_sha256"]==PINS["camera.json"]
        and manifest["original_source_uv_edit_plan_sha256"]==PINS["uv-plan.json"]
        and manifest["original_source_rest_only"] is True
        and manifest["original_source_material_policy_sha256"]==PINS["source-policy.json"]
        and policy["material_mode"]=="unlit_preserve_alpha"
        and policy["source_sha256"]==ORIGINAL_SOURCE
        and court["status"]==
           "PASS_ORIGINAL_CAT_LINKED_UV_EDIT_CONSEQUENCE_SUPERPOSITION__OWNER_UNKNOWN"
        and court["failed_obligations"]==[],
        "ORIGINAL_SOURCE_PHYSICAL_EXPERIMENT_LINEAGE")
    with np.load(root/"teacher.npz",allow_pickle=False) as teacher, \
         np.load(root/"rest.npz",allow_pickle=False) as native_rest, \
         np.load(root/"face-uv.npz",allow_pickle=False) as u, \
         np.load(root/"physical.npz",allow_pickle=False) as p:
        f=np.asarray(teacher["faces"],np.int64)
        xyz=np.asarray(teacher["rest_vertices_source_world"],np.float64)
        face_uv=np.asarray(u["face_uv"],np.float64)
        source=np.asarray(p["source_uv_edit_cycles_sample_rgba"],np.float32)
        req(np.array_equal(f,native_rest["faces"])
            and np.array_equal(xyz,native_rest["rest_vertices_source_world"])
            and f.shape==(684,3) and len(xyz)==1204
            and face_uv.shape==(684,3,2)
            and source.shape==(4,8,8,N,N,4),
            "ORIGINAL_GEOMETRY_UV_RENDER_ARRAY_BINDING")
    face_xyz=xyz[f]
    rects=[r["source_rgba_png_pixel_rectangle_xyxy_exclusive"]
           for r in plan["selected_original_source_uv_edits"]]
    rows=q["source_original_face_pixel_witness_rows_PRIVATE"]
    req(isinstance(rows,list) and len(rows)>0
        and len(rows)==q["counts"]["qualified_original_face_pixel_witness_count"],
        "ORIGINAL_FACE_PIXEL_WITNESSES_ABSENT")
    per_view={v:Counter() for v in VIEWS}
    seen=set();original_faces=set()
    mismatched_wrong_edits=0
    positive=0
    negative=0
    for idx,row in enumerate(rows):
        fi=int(row["face_index"])
        x=int(row["pixel_x"])
        y=int(row["pixel_y_top_first"])
        view=int(row["original_camera_view"])
        edit=int(row["source_uv_edit_operation_index"])
        req(view in VIEWS and 0<=fi<len(f) and 0<=x<N and 0<=y<N
            and 0<=edit<3 and set(row)=={
                "original_camera_view","face_index","pixel_x",
                "pixel_y_top_first","source_uv_edit_operation_index"},
            "SOURCE_FACE_PIXEL_ADDRESS_INVALID")
        key=(view,y,x)
        req(key not in seen,"DUPLICATE_PIXEL_ASSERTING_TWO_FACE_IDS")
        seen.add(key);original_faces.add(fi)
        direct_row_geometry_check(face_xyz,face_uv,camera["cameras"][view],
                                  rects[edit],fi,y,x)
        # Direct observed ORIGINAL physical image samples; Blender storage
        # was bottom-first, whereas our geometry/camera addresses top-first.
        frame=source[VIEWS.index(view),:,:,N-1-y,x,:].astype(np.float64)
        frame[...,:3]*=frame[...,3:4]
        req(np.all(np.abs(frame[:,0,3]-1.0)<=TOL)
            and np.all(np.abs(frame[:,:,3]-1.0)<=TOL),
            "FIRST_ORIGINAL_FACE_NOT_OPAQUE_IN_PHYSICAL_SAMPLE")
        effects=np.max(np.abs(frame[:,1:4]-frame[:,0:1]),axis=-1)
        req(np.all(effects[:,edit]>TOL),
            "EDIT_EXPECTED_AT_SOURCE_FACE_NOT_PHYSICALLY_OBSERVABLE")
        other=[j for j in range(3) if j!=edit]
        req(np.all(effects[:,other]<=TOL),
            "INDEPENDENT_OTHER_EDIT_CHANGED_CERTIFIED_SOURCE_FACE")
        positive+=8
        negative+=16
        per_view[view]["qualified_face_pixels"]+=1
        per_view[view][f"source_edit_{edit}_face_pixels"]+=1
        per_view[view]["positive_original_samples"]+=8
        per_view[view]["other_edit_absence_obligations"]+=16
    require(len(original_faces)==q["counts"][
        "qualified_original_source_face_count"]
        and positive==q["counts"][
        "qualified_original_source_sample_face_edit_effect_count"]
        and negative==q["counts"][
        "qualified_original_source_other_edit_absence_obligations"]
        and [sum(per_view[v][f"source_edit_{i}_face_pixels"] for v in VIEWS)
             for i in range(3)] == q["counts"]["source_pixels_by_edit"]
        and all(per_view[v]["qualified_face_pixels"] ==
            q["per_source_camera"][i]["face_pixels"]
            for i,v in enumerate(VIEWS)),
        "INDEPENDENT_SOURCE_FACE_SAMPLE_ACCOUNTING_DRIFT")
    out={
        "schema":SCHEMA,
        "status":PASS,
        "source_original_glb_sha256":ORIGINAL_SOURCE,
        "source_producer_witness_sha256":sha(report_path),
        "original_source_parent_sha256":PINS,
        "source_original_face_pixel_court_count":len(rows),
        "source_original_distinct_face_count":len(original_faces),
        "source_original_8_sample_positive_face_edit_court":positive,
        "source_original_2_other_edits_absence_obligations":negative,
        "source_original_edit_qualified_pixel_by_camera":{
            str(v):dict(per_view[v]) for v in VIEWS},
        "source_original_edit_qualified_pixels_by_edit":[
            sum(per_view[v][f"source_edit_{i}_face_pixels"] for v in VIEWS)
            for i in range(3)],
        "face_to_sample_effect_witness_is_sufficient_not_complete":True,
        "source_unwitnessed_faces_and_edits":"UNKNOWN__NOT_FALSE",
        "source_original_artist_layer_ownership":"UNKNOWN",
        "source_material_identity_between_unwitnessed_faces":"UNKNOWN",
        "source_pose_transport_to_new_poses":"UNKNOWN",
        "source_exact_Cycles_jitter_rays":"UNOBSERVED__UNNEEDED_FOR_WHOLE_PIXEL_SUFFICIENT_PROOF",
        "new_source_physical_render_count":0,
        "graphics_owner_authorized":False,
        "same_drawable_training_pairs_qualified":0,
        "different_drawable_training_pairs_qualified":0,
        "iris_training_authorized":False,
        "tessa_training_authorized":False,
        "training_authorized":False,
        "materialization_authorized":False,
    }
    dest=Path(output);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(out,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({
        "status":PASS,
        "original_face_count":len(original_faces),
        "source_pixel_face_witnesses":len(rows),
        "physical_positive_original_sample_edit_effects":positive,
        "negative_other_edit_checks":negative,
        "source_original_owner":"UNKNOWN",
        "training_authorized":False,
    },sort_keys=True),flush=True)
    return out


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source-root",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    independently_court(a.source_root,a.report,a.output)
