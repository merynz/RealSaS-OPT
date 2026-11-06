#!/usr/bin/env python3
"""Original Cat source-authored pose UV edit transport physical Cycles oracle v3.

Source original animated glTF, source teacher geometry and source-linked atlas
edits. It never creates artist drawable labels or model training truth.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from tools.iris_vnext import blender_authored_teacher_export_v1 as base
from tools.iris_vnext import blender_authored_teacher_export_v2 as transport
from tools.iris_vnext import blender_authored_teacher_export_v4 as v4
from tools.iris_vnext import blender_authored_teacher_export_v4_actions as actions
from tools.truth_foundry import blender_source_appearance_policy_replay_v1 as replay
from tools.truth_foundry.blender_corpus370_original_cat_uv_texture_cycles_v2 import (
    frame,sha,req
)
from tools.observation.blender_controlled_8view_v2 import configure

SCHEMA="RealSaS.Corpus370OriginalCatAuthoredPoseUVPhysicalSamples.v3"
POSES=(0,14)
VIEWS=(0,2,4,6)
OFFSETS=(0,16)
ARMS=("BASE","SINGLE_0","SINGLE_1","SINGLE_2")
ORIGINAL_GLTF="1e73aaf7d497dd7d2946badf92d189d2333f163e66dced91feb446bc9bdee850"
ORIGINAL_TEXTURE="2f6d0032e51c2d49b13b8e689d59eb39aa3ee6c32f2a1851213e11fc7593177d"
ORIGINAL_TEACHER="c2792c8efb36c4ccdf9178c6b933351b50364bbd28707e9365a3c115db00d603"
ORIGINAL_TEACHER_RECEIPT="a0f224019deca20bf5d807ab4d27a7c341a65bc21930fc99ad8e88ca5b4bc023"
ORIGINAL_CAMERA="300b935a47379f6c44d4b4356554d1c32a86ca3b732a62405c73da9e0ec6adf0"
ORIGINAL_SHADER="c8fe4e307ddbbaf5cc0892cffc4267bd08e394f2299f4e70155ffc6207190a4b"
ORIGINAL_UVPLAN="8be8fccde3d9cc97f7ba452fd50c1a9f4ebea2e34da590a22d30872b2b0f497b"
ORIGINAL_EDIT_ASSETS="0e9a2fcf6a2c048eb6e99048793290b8865d802689f59b7c8c20bc6390cf479a"
FLAGS=("graphics_owner_authorized","iris_training_authorized",
       "tessa_training_authorized","materialization_authorized","training_authorized")

def read(p):return json.loads(Path(p).read_text())

def main():
    argv=sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("source","original-atlas","teacher-npz","teacher-receipt",
                 "camera-bundle","source-policy","uv-plan","atlas-assets",
                 "atlas-dir","output-dir"):
        parser.add_argument("--"+name,required=True,type=Path)
    x=parser.parse_args(argv)
    for p,expected in (
        (x.source,ORIGINAL_GLTF),(x.original_atlas,ORIGINAL_TEXTURE),
        (x.teacher_npz,ORIGINAL_TEACHER),(x.teacher_receipt,ORIGINAL_TEACHER_RECEIPT),
        (x.camera_bundle,ORIGINAL_CAMERA),(x.source_policy,ORIGINAL_SHADER),
        (x.uv_plan,ORIGINAL_UVPLAN),(x.atlas_assets,ORIGINAL_EDIT_ASSETS)):
        req(sha(p)==expected,"V3_PARENT_SHA:"+p.name)
    req(bpy.app.version>=(4,5,0),"V3_ORIGINAL_CYCLES45")
    teacher_doc=read(x.teacher_receipt)
    plan=read(x.uv_plan)
    variants=read(x.atlas_assets)
    candidate=read(x.source_policy)
    cameras=read(x.camera_bundle)
    req(teacher_doc.get("source",{}).get("sha256")==ORIGINAL_GLTF and
        teacher_doc.get("transport_revision")==4 and
        teacher_doc.get("sampled_frame_count")==32 and
        teacher_doc.get("teacher_only") is True and
        teacher_doc.get("teacher_rig_skin_motion_allowed_as_inference_input") is False
        and teacher_doc.get("face_count")==684 and
        teacher_doc.get("output",{}).get("sha256")==ORIGINAL_TEACHER and
        len(teacher_doc.get("sampled_frames",[]))==32 and
        len(teacher_doc.get("native_motion_inventory_expected",[]))==8,
        "V3_ORIGINAL_AUTHORED_TEACHER_PROVENANCE")
    req(plan.get("source_original_glb_sha256")==ORIGINAL_GLTF
        and plan.get("source_original_texture_sha256")==ORIGINAL_TEXTURE
        and plan.get("source_uv_coordinate_origin")==
            "GLTF_2_IMAGE_TOP_LEFT__UV_V_INCREASES_DOWN"
        and plan.get("source_uv_to_png_y_flip_applied") is False
        and variants.get("source_operation_plan_sha256")==ORIGINAL_UVPLAN
        and variants.get("original_texture_source_sha256")==ORIGINAL_TEXTURE
        and variants.get("original_source_glb_sha256")==ORIGINAL_GLTF
        and [z.get("arm") for z in variants.get("original_atlas_arm_rows",[])][:4]
            ==list(ARMS) and
        all(z.get(flag) is False for z in (plan,variants) for flag in FLAGS)
        and candidate.get("source_sha256")==ORIGINAL_GLTF
        and candidate.get("material_mode")=="unlit_preserve_alpha"
        and candidate.get("camera_bundle",{}).get("sha256")==ORIGINAL_CAMERA
        and len(cameras.get("cameras",[]))==8,
        "V3_ORIGINAL_SHADER_UV_AND_CAMERA_PROVENANCE")
    with np.load(x.teacher_npz,allow_pickle=False) as data:
        rest=np.asarray(data["rest_vertices_source_world"],dtype=np.float64).copy()
        faces=np.asarray(data["faces"],dtype=np.int64).copy()
        owner=np.asarray(data["face_owner"],dtype=np.int32).copy()
        poses=np.asarray(data["posed_vertices_source_world"],dtype=np.float64).copy()
    req(faces.shape==(684,3) and rest.ndim==2 and rest.shape[1]==3 and
        poses.shape==(32,*rest.shape) and np.isfinite(poses).all(),
        "V3_ORIGINAL_AUTHORED_GEOMETRY")
    extent=max(1.,float(np.linalg.norm(np.ptp(rest,axis=0))))
    tolerance=1e-6*extent
    displacement=float(np.max(np.linalg.norm(poses[POSES[1]]-poses[POSES[0]],axis=1)))
    req(displacement>tolerance,"V3_POSES_NOT_MECHANICALLY_DISTINCT")
    out=x.output_dir.resolve()
    req(not out.exists(),"V3_OUTPUT_EXISTS")
    out.mkdir(parents=True)
    base._import_animated(x.source)
    requested=teacher_doc["native_motion_inventory_expected"]
    resolved={name:transport._action_by_requested_take(name) for name in requested}
    bindings=actions.capture_imported_action_bindings(resolved)
    meshes=base._mesh_objects()
    base._neutralize_animation()
    snap=v4._extract_snapshot(meshes,freeze_triangulation=True)
    req(np.array_equal(np.asarray(snap["faces"]),faces) and
        np.array_equal(np.asarray(snap["face_owner"]),owner) and
        np.max(np.abs(np.asarray(snap["vertices"],np.float64)-rest))<=tolerance,
        "V3_ORIGINAL_SOURCE_REST_AND_FACE_PROOF")
    mat=replay._materials_from_render_meshes(meshes)
    req(len(mat)==1 and len(candidate.get("material_transform",[]))==1 and
        candidate["material_transform"][0].get("source_color_linked") is True,
        "V3_ORIGINAL_TEXTURE_SHADER_NOT_LINKED")
    replayed=replay._replay_transform(mat,candidate["material_transform"])
    req(len(replayed)==1 and replayed[0].get("replayed") is True
        and replayed[0].get("alpha_policy")==
            "IMPORTED_OPAQUE_MODE__BSDF_ALPHA_INACTIVE",
        "V3_SOURCE_ALPHA_REPLAY_MISMATCH")
    material=next(iter(mat.values()))
    texture_nodes=[n for n in material.node_tree.nodes if n.type=="TEX_IMAGE"
                   and n.image and n.outputs["Color"].is_linked]
    req(len(texture_nodes)==1,"V3_NATIVE_SOURCE_IMAGE_NODE_COUNT")
    tex=texture_nodes[0]
    imported_image=tex.image
    req(tuple(imported_image.size)==(512,512) and
        imported_image.colorspace_settings.name in ("sRGB","sRGB EOTF"),
        "V3_NATIVE_SOURCE_COLORSPACE")
    original_colorspace=imported_image.colorspace_settings.name
    original_alpha=imported_image.alpha_mode
    images={}
    for row in variants["original_atlas_arm_rows"][:4]:
        p=(x.atlas_dir/row["filename"]).resolve()
        req(p.is_relative_to(x.atlas_dir.resolve()) and p.is_file()
            and sha(p)==row["source_atlas_rgba_sha256"],
            "V3_ATLAS_ASSET_HASH")
        img=bpy.data.images.load(str(p),check_existing=False)
        req(tuple(img.size)==(512,512),"V3_ATLAS_IMAGE_RESOLUTION")
        img.colorspace_settings.name=original_colorspace
        img.alpha_mode=original_alpha
        images[row["arm"]]=img
    req(sha(x.atlas_dir/"BASE.png")==ORIGINAL_TEXTURE,
        "V3_ORIGINAL_BASE_TEXTURE_NOT_IDENTICAL")
    configure(192)
    scene=bpy.context.scene
    req(scene.render.engine=="CYCLES","V3_RENDERER_NOT_CYCLES")
    scene.cycles.samples=32
    scene.cycles.seed=0
    scene.cycles.use_adaptive_sampling=False
    scene.cycles.use_denoising=False
    scene.cycles.use_sample_subset=True
    scene.cycles.sample_subset_length=1
    scene.cycles.pixel_filter_type="BOX"
    scene.cycles.filter_width=.01
    scene.render.image_settings.file_format="OPEN_EXR"
    scene.render.image_settings.color_mode="RGBA"
    scene.render.image_settings.color_depth="32"
    scene.render.image_settings.exr_codec="ZIP"
    camera_data=bpy.data.cameras.new("OriginalCatV3_ProofCamera")
    camera_data.type="ORTHO"
    cam=bpy.data.objects.new(camera_data.name,camera_data)
    scene.collection.objects.link(cam)
    scene.camera=cam
    samples=np.empty((2,4,2,4,192,192,4),np.float32)
    receipts=[]
    scratch=out/"scratch"
    scratch.mkdir()
    started=time.monotonic()
    try:
        for pi,poseindex in enumerate(POSES):
            row=teacher_doc["sampled_frames"][poseindex]
            actions.activate_action(resolved[row["action"]],bindings)
            scene.frame_set(int(row["frame"]))
            bpy.context.view_layer.update()
            actual=v4._extract_snapshot(meshes,freeze_triangulation=False)
            observed=np.asarray(actual["vertices"],np.float64)
            err=float(np.max(np.abs(observed-poses[poseindex])))
            req(actual.get("polygon_signature")==snap.get("polygon_signature")
                and err<=tolerance,
                "V3_ANIMATED_ORIGINAL_SOURCE_POSE_NOT_TEACHER")
            receipts.append({
                "teacher_pose_sample_ordinal":poseindex,
                "source_teacher_action_name":"PRIVATE_TEACHER_ONLY",
                "source_native_frame":int(row["frame"]),
                "source_geometry_max_abs_error":err,
                "source_geometry_tolerance":tolerance,
                "source_teacher_original_face_order_matches":True,
                "source_pose_original_vertex_count":len(rest),
            })
            for vi,view in enumerate(VIEWS):
                camera_row=cameras["cameras"][view]
                req(camera_row["view_index"]==view and
                    camera_row["view_id"]==f"V{view}" and
                    camera_row["resolution"]==1024,
                    "V3_ORIGINAL_CAMERA_ORDER")
                cam.location=Vector(tuple(map(float,camera_row["origin"])))
                cam.rotation_euler=Vector(tuple(map(float,camera_row["forward"]))).to_track_quat(
                    "-Z","Y").to_euler()
                camera_data.ortho_scale=2*float(camera_row["half_extent"])
                camera_data.clip_start=float(camera_row["clip_start"])
                camera_data.clip_end=float(camera_row["clip_end"])
                bpy.context.view_layer.update()
                for si,offset in enumerate(OFFSETS):
                    scene.cycles.sample_offset=offset
                    for ai,arm in enumerate(ARMS):
                        tex.image=images[arm]
                        bpy.context.view_layer.update()
                        samples[pi,vi,si,ai]=frame(scene,scratch/"source.exr")
                    print(json.dumps({"pose":poseindex,"view":view,
                        "sample_offset":offset,
                        "arms_physically_rendered":len(ARMS),
                        "elapsed_sec":round(time.monotonic()-started,1)}),flush=True)
    finally:
        tex.image=imported_image
    req(np.isfinite(samples).all() and
        float(samples[...,3].min())>=-1e-5 and
        float(samples[...,3].max())<=1.00001,"V3_ORIGINAL_RGBA_DOMAIN")
    path=out/"PRIVATE_CAT_NATIVE_POSE_UV_EDIT_CYCLES_SAMPLES_V3.npz"
    np.savez_compressed(path,source_native_pose_uv_rgba=samples)
    manifest={
        "schema":SCHEMA,
        "status":"PHYSICAL_SOURCE_NATIVE_POSE_UV_EDITS_RENDERED__INDEPENDENT_COURT_REQUIRED",
        "original_source_sha256":ORIGINAL_GLTF,
        "original_texture_sha256":ORIGINAL_TEXTURE,
        "original_native_teacher_sha256":ORIGINAL_TEACHER,
        "original_teacher_receipt_sha256":ORIGINAL_TEACHER_RECEIPT,
        "original_camera_sha256":ORIGINAL_CAMERA,
        "original_material_shader_sha256":ORIGINAL_SHADER,
        "original_source_uv_plan_sha256":ORIGINAL_UVPLAN,
        "original_source_atlas_edits_manifest_sha256":ORIGINAL_EDIT_ASSETS,
        "source_atlas_arm_sha256":[r["source_atlas_rgba_sha256"]
            for r in variants["original_atlas_arm_rows"][:4]],
        "teacher_pose_sample_ordinals":list(POSES),
        "original_source_pose_geometry_replay_receipts":receipts,
        "source_pose_world_max_displacement":displacement,
        "source_pose_tolerance":tolerance,
        "original_calibrated_view_indices":list(VIEWS),
        "original_same_cycles_sample_offsets":list(OFFSETS),
        "source_sample_alpha_policy":"ORIGINAL_OPAQUE_TEXTURE__NO_NONBINARY_ALPHA",
        "original_source_uv_edit_arms":list(ARMS),
        "original_pose_dependent_uv_surface_texture_rendered":True,
        "original_source_sample_count":64,
        "source_samples_npz_shape":list(samples.shape),
        "source_samples_npz_sha256":sha(path),
        "source_drawable_owner":"UNKNOWN",
        "same_drawable_training_pairs_qualified":0,
        "different_drawable_training_pairs_qualified":0,
        **{k:False for k in FLAGS},
    }
    (out/"PRIVATE_CAT_NATIVE_POSE_UV_EDIT_CYCLES_V3.json").write_text(
        json.dumps(manifest,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({"status":manifest["status"],
        "new_real_original_cycles_source_renders":64,
        "source_max_pose_displacement":displacement,
        "source_original_artist_owner":"UNKNOWN",
        "training_authorized":False}),flush=True)

if __name__=="__main__":main()
