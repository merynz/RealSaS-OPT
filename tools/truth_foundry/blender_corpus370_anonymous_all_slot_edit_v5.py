#!/usr/bin/env python3
"""Full-FOV, image-grid anonymous puppet layer editability research probe.

Unlike the old one-pixel source-locus test this renders the ENTIRE camera FOV
in all eight existing source-camera views, including original-source and full-source-triangle
replays, no-op parity, A/B whole anonymous group suppression and paired
suppression. Uses original exact source material alpha/UV (not segmentation
colors). This is **bounded rest-pose source appearance** only: no CAA or
production-ready skin/attachment/renderer authority can be minted.

The job is created only by the SHA-exact private hosted source consumer.
Input object/material names are privileged import addresses, never model labels.
Outputs contain private pixel buffers and are never uploaded as GitHub artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import bpy
import numpy as np
from mathutils import Vector

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from tools.observation import blender_controlled_8view_v2 as observations
from tools.iris_vnext import blender_authored_teacher_export_v1 as source_transport
from tools.iris_vnext import blender_authored_teacher_export_v3 as geometry_transport
from tools.truth_foundry import blender_source_appearance_policy_replay_v1 as replay
from tools.truth_foundry import blender_source_ray_appearance_measure_v1 as image_probe
from tools.truth_foundry.blender_teacher_scene_v1 import _appearance_and_uv

SCHEMA="RealSaS.Corpus370AnonymousAllSlotEditorRender.v5"
JOB_SCHEMA="RealSaS.Corpus370AnonymousAllSlotEditorJob.v5"
BASE_ARMS=("SOURCE_ORIGINAL","SOUP_FULL","SOUP_NOOP")
def build_arms(slot_ids):
    require(slot_ids==list(range(len(slot_ids))) and 2<=len(slot_ids)<=64,
            "ALL_SLOT_SOURCE_IDS_NONCONTIGUOUS")
    return BASE_ARMS+tuple(f"REMOVE_GROUP_{k:04d}" for k in slot_ids)+(
        "REMOVE_BOTH_ANCHOR_GROUPS",)


def require(ok,code):
    if not ok:raise RuntimeError(code)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def verify(ref,code):
    p=Path(ref["path"]).resolve()
    require(p.is_file() and sha(p)==ref["sha256"],code)
    return p


def _camera_full(camera,row,resolution):
    camera.location=Vector(tuple(float(x) for x in row["origin"]))
    camera.rotation_euler=Vector(tuple(float(x) for x in row["forward"])).to_track_quat("-Z","Y").to_euler()
    camera.data.type="ORTHO"
    camera.data.ortho_scale=2.*float(row["half_extent"])
    camera.data.clip_start=float(row["clip_start"])
    camera.data.clip_end=float(row["clip_end"])
    scene=bpy.context.scene
    scene.render.resolution_x=int(resolution)
    scene.render.resolution_y=int(resolution)
    scene.render.resolution_percentage=100
    bpy.context.view_layer.update()


def _frame(path,resolution):
    scene=bpy.context.scene
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.unlink(missing_ok=True)
    scene.render.filepath=str(path)
    bpy.ops.render.render(write_still=True)
    require(path.is_file() and path.stat().st_size>0,
            "ANON_ALL_SLOT_EXR_MISSING")
    pic=bpy.data.images.load(str(path),check_existing=False)
    try:
        require(tuple(map(int,pic.size))==(resolution,resolution),
                "ANON_ALL_SLOT_EXR_RESOLUTION")
        arr=np.empty(resolution*resolution*4,np.float32)
        pic.pixels.foreach_get(arr)
        frame=arr.reshape(resolution,resolution,4).copy()
    finally:
        bpy.data.images.remove(pic)
        path.unlink(missing_ok=True)
    require(np.isfinite(frame).all() and
            bool(np.all(frame[...,3]>=-1e-5))
            and bool(np.all(frame[...,3]<=1.+1e-5)),
            "ANON_ALL_SLOT_NONFINITE_OR_ALPHA")
    frame[...,3]=np.clip(frame[...,3],0.,1.)
    return frame


def _group_mode(face_map,group_labels,*,excluded):
    require(all(type(i) is int and i>=0 for i in excluded),
            "ANON_ALL_SLOT_GROUP_ID")
    touched=set()
    for face,(mesh,local,orig,clear) in face_map.items():
        material=clear if int(group_labels[face]) in excluded else orig
        mesh.polygons[local].material_index=material
        touched.add(mesh)
    for mesh in touched:mesh.update()
    bpy.context.view_layer.update()


def run(job_path,output_path):
    job=json.loads(Path(job_path).read_text())
    require(job.get("schema")==JOB_SCHEMA
            and job.get("teacher_only") is True
            and job.get("training_authorized") is False
            and job.get("graphics_owner_authorized") is False
            and job.get("full_frame_render_qualifies_model") is False
            and job.get("views")==list(range(8))
            and job.get("persistent_pair_anchor_view")==4
            and type(job.get("raster_resolution")) is int
            and 128<=job["raster_resolution"]<=512
            and type(job.get("cycles_samples")) is int
            and 8<=job["cycles_samples"]<=128,
            "ANON_ALL_SLOT_SOURCE_RESEARCH_SCOPE")
    source=verify(job["source"],"ANON_ALL_SLOT_SOURCE_SHA")
    admission=verify(job["admission"],"ANON_ALL_SLOT_ADMISSION_SHA")
    candidate_path=verify(job["candidate"],"ANON_ALL_SLOT_CANDIDATE_SHA")
    camera_path=verify(job["camera"],"ANON_ALL_SLOT_CAMERA_SHA")
    rest_path=verify(job["rest_geometry"],"ANON_ALL_SLOT_REST_SHA")
    layer_path=verify(job["layer_proposal"],"ANON_ALL_SLOT_LAYERS_SHA")
    layer_court_path=verify(job["layer_court"],"ANON_ALL_SLOT_LAYER_COURT_SHA")
    pairs_path=verify(job["source_graphics_pairs"],"ANON_ALL_SLOT_PAIR_SHA")
    candidate=json.loads(candidate_path.read_text())
    bundle=json.loads(camera_path.read_text())
    layer=json.loads(layer_path.read_text())
    court=json.loads(layer_court_path.read_text())
    pairs=json.loads(pairs_path.read_text())
    sid=job["subject_id"]
    require(job["source"]["sha256"]==pairs["parents"]["source_sha256"]
            and job["source"]["sha256"]==layer["source_sha256"]
            and candidate["source_sha256"]==job["source"]["sha256"]
            and layer["status"]=="ALL_8_INPUT_VIEW_GEOMETRY_LAYER_HYPOTHESES__RENDER_OUTCOME_HELDOUT"
            and court["schema"]=="RealSaS.Corpus370AllViewGeometryEditorProxyCourt.v3"
            and court["training_authorized"] is False
            and layer["training_authorized"] is False
            and layer["view4_geometry_seen_by_compiler"] is True
            and layer["view4_rendered_edit_consequences_used_by_compiler"] is False
            and court["not_view_holdout_generalization"] is True
            and court["render_intervention_view4_seen_for_compilation"] is False
            and len(pairs["rows"])==2
            and {x["view_index"] for x in pairs["rows"]}=={0,4}
            and bundle["schema"]=="RealSaS.CameraProjectionBundle.v1",
            "ANON_ALL_SLOT_COURTED_SOURCE_SCOPE")
    chosen=next((x for x in layer["candidates"]
                if x["candidate"]==court["selected_candidate_train_only"]),None)
    require(chosen is not None
            and chosen["slot_count"]==court["selected_group_count"]
            and chosen["training_authorized"] is False
            and len(chosen["face_slot_teacher_only"])==layer["source_face_count"],
            "ANON_ALL_SLOT_COURT_SELECTED_HYPOTHESIS")
    labels=np.asarray(chosen["face_slot_teacher_only"],np.int32)
    source_slot_ids=sorted(set(int(i) for i in labels))
    require(source_slot_ids==list(range(chosen["slot_count"]))
            and 2<=len(source_slot_ids)<=64
            and job.get("all_source_group_slots_requested") is True
            and job.get("expected_source_slot_count")==len(source_slot_ids),
            "ALL_SLOT_COMPLETE_SOURCE_FACE_GROUP_CENSUS")
    arms=build_arms(source_slot_ids)
    with np.load(rest_path,allow_pickle=False) as rest:
        rest_v=np.asarray(rest["rest_vertices_source_world"],np.float64)
        rest_f=np.asarray(rest["faces"],np.int64)
        rest_owner=np.asarray(rest["face_owner_privileged"],np.int32)
        require(len(rest_f)==len(labels),"ANON_ALL_SLOT_SELECTED_REST_FACE_CENSUS")
    out=Path(output_path).resolve()
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="c370_full_frame_") as temporary:
        temp=Path(temporary)
        observations.import_rest(source,temp)
        meshes=source_transport._mesh_objects()
        snapshot=geometry_transport._extract_snapshot(meshes,freeze_triangulation=True)
        actual_v=np.asarray(snapshot["vertices"],np.float64)
        require(np.array_equal(np.asarray(snapshot["faces"]),rest_f)
                and np.array_equal(np.asarray(snapshot["face_owner"]),rest_owner)
                and actual_v.shape==rest_v.shape
                and np.max(np.abs(actual_v-rest_v))<=
                    1e-6*max(1.,float(np.linalg.norm(np.ptp(rest_v,axis=0)))),
                "ANON_ALL_SLOT_REIMPORTED_ORIGINAL_SOURCE_GEOMETRY_DRIFT")
        material=replay._materials_from_render_meshes(meshes)
        replay._replay_transform(material,list(candidate.get("material_transform") or []))
        image_probe._configure_measurement(job["cycles_samples"])
        cameras={int(x["view_index"]):x for x in bundle["cameras"]}
        require(set(cameras)==set(range(8)),
                "ANON_ALL_SLOT_EIGHT_SOURCE_CAMERAS")
        camera=image_probe._make_camera()
        uv,uv_valid,_,_= _appearance_and_uv(meshes,snapshot)
        arrays={
            "rest_vertices_source_world":rest_v,
            "faces":rest_f,
            "face_owner_privileged":rest_owner,
            "face_material_privileged":np.asarray(snapshot["face_material"],np.int32),
            "triangle_uv":uv,"triangle_uv_valid":uv_valid,
        }
        clear=image_probe._transparent_material()
        soup,face_map=image_probe._build_triangle_soup(arrays,meshes,clear)
        require(len(face_map)==len(labels),"ANON_ALL_SLOT_FACE_MAP_COVERAGE")
        # The same source-addressed candidate layer IDs survive *every* camera.
        # Anchor face pair is independently measured at view4; no per-view
        # retargeting or dynamic source-face switching is allowed.
        row=next(x for x in pairs["rows"] if x["view_index"]==4)
        face_a=row["source_face_a_teacher_only"]
        face_b=row["source_face_b_teacher_only"]
        require(type(face_a) is int and type(face_b) is int
                and 0<=face_a<len(labels) and 0<=face_b<len(labels),
                "ANON_ALL_SLOT_PERSISTENT_SOURCE_FACES")
        a=int(labels[face_a]);b=int(labels[face_b])
        require(a!=b,"ALL_SLOT_ANCHOR_PAIR_MERGED")
        out_frames=[]
        private_view_details=[]
        for view in job["views"]:
            _camera_full(camera,cameras[view],job["raster_resolution"])
            for obj in meshes:obj.hide_render=False
            for obj in soup:obj.hide_render=True
            bpy.context.view_layer.update()
            raster=[_frame(temp/(f"original_{view}.exr"),
                           job["raster_resolution"])]
            for obj in meshes:obj.hide_render=True
            for obj in soup:obj.hide_render=False
            for arm in arms[1:]:
                exclude=(
                    {int(arm.split("_")[-1])} if arm.startswith("REMOVE_GROUP_") else
                    {a,b} if arm=="REMOVE_BOTH_ANCHOR_GROUPS" else set()
                )
                _group_mode(face_map,labels,excluded=exclude)
                raster.append(_frame(temp/(f"{arm}_{view}.exr"),
                                     job["raster_resolution"]))
            out_frames.append(np.stack(raster,axis=0))
            private_view_details.append({
                "view":view,
                "persistent_pair_anchor_view":4,
                "source_pixel_x":row["pixel_x"],
                "source_pixel_y":row["pixel_y"],
                "original_source_resolution":cameras[view]["resolution"],
                "selected_slot_a_teacher_only":a,
                "selected_slot_b_teacher_only":b,
                "source_face_a_teacher_only":face_a,
                "source_face_b_teacher_only":face_b,
                "group_a_and_b_separable":a!=b,
            })
        pixels=np.stack(out_frames,axis=0)
        require(pixels.shape==(8,len(arms),job["raster_resolution"],
                               job["raster_resolution"],4),
                "ANON_ALL_SLOT_REAL_MULTIARM_RASTER_SHAPE")
        arrays_path=out.with_suffix(".npz")
        np.savez_compressed(arrays_path,source_original_vs_anonymous_group_rgba=pixels)
        result={
            "schema":SCHEMA,
            "status":"SOURCE_FULL_FRAME_ALL_SLOT_REST_ALPHA_EDIT_CANDIDATE__INDEPENDENT_COURT_REQUIRED",
            "source_sha256":job["source"]["sha256"],
            "subject_id":sid,
            "original_geometry_reimport_parity_verified":True,
            "private_job_sha256":sha(job_path),
            "source_camera_bundle_sha256":sha(camera_path),
            "selected_layer_proposal_sha256":sha(layer_path),
            "selected_layer_court_sha256":sha(layer_court_path),
            "source_graphics_pair_sha256":sha(pairs_path),
            "sample_count":job["cycles_samples"],
            "full_frame_resolution":job["raster_resolution"],
            "original_source_resolution":[int(cameras[v]["resolution"]) for v in job["views"]],
            "views":job["views"],
            "persistent_pair_anchor_view":4,
            "persistent_selected_slot_ids_teacher_only":[a,b],
            "arms":list(arms),
            "source_slot_ids_teacher_only":source_slot_ids,
            "all_source_group_slots_rendered":True,
            "source_group_count":len(source_slot_ids),
            "array_path":str(arrays_path),"array_sha256":sha(arrays_path),
            "private_view_details":private_view_details,
            "rendered_full_field_of_view":True,
            "downsampled_relative_to_original":any(
                job["raster_resolution"]!=int(cameras[v]["resolution"]) for v in job["views"]),
            "teacher_only":True,
            "same_drawable_identity":"UNKNOWN",
            "graphics_owner_authorized":False,
            "model_training_authorized":False,
            "training_authorized":False,
        }
        out.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
        print(json.dumps({
            "status":result["status"],
            "view_count":len(job["views"]),
            "full_frame_resolution":result["full_frame_resolution"],
            "training_authorized":False,
        },sort_keys=True),flush=True)
        return result


def main():
    raw=sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
    p=argparse.ArgumentParser()
    p.add_argument("--job",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(raw)
    run(a.job,a.output)


if __name__=="__main__":main()
