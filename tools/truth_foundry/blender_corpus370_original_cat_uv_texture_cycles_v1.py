#!/usr/bin/env python3
"""Real Cat source GLB texture-linked edit Cycles SAME sample capture.

Only original source atlas bytes are changed by exact source-UV tile actions.
This uses Blender's actual material image Texture node, not approximate shader
palette substitutions or fake segmentations. All multi-edit image results are
heldout independent original-source renderer targets.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import bpy
import numpy as np
from mathutils import Vector

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.observation.blender_controlled_8view_v2 import import_rest,configure
from tools.truth_foundry import blender_source_appearance_policy_replay_v1 as replay

SCHEMA="RealSaS.Corpus370OriginalCatSourceTextureSameCyclesEditEvidence.v1"
PLAN="RealSaS.Corpus370OriginalSourceUVAtlasEditPlan.v1"
ATLAS="RealSaS.Corpus370OriginalSourceUVAtlasEditImages.v1"
SOURCE_SHA="1e73aaf7d497dd7d2946badf92d189d2333f163e66dced91feb446bc9bdee850"
ATLAS_SHA="2f6d0032e51c2d49b13b8e689d59eb39aa3ee6c32f2a1851213e11fc7593177d"
VIEWS=(1,3,5,7)
SAMPLE_INDICES=(0,4,8,12,16,20,24,28)
ARMS=("BASE","SINGLE_0","SINGLE_1","SINGLE_2","PAIR_01","PAIR_02","PAIR_12","JOINT_012")
RES=192
FLAGS=("graphics_owner_authorized","iris_training_authorized",
       "tessa_training_authorized","materialization_authorized","training_authorized")

def req(ok,code):
    if not ok:raise ValueError("ORIGINAL_CAT_UV_CYCLES_"+code)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def frame(scene,output):
    scene.render.filepath=str(output)
    output.unlink(missing_ok=True)
    bpy.ops.render.render(write_still=True)
    req(output.is_file() and output.stat().st_size>0,"EXR_RENDER_MISSING")
    image=bpy.data.images.load(str(output),check_existing=False)
    try:
        req(tuple(map(int,image.size))==(RES,RES),"RENDER_RESOLUTION_DRIFT")
        array=np.empty(RES*RES*4,dtype=np.float32)
        image.pixels.foreach_get(array)
        result=array.reshape(RES,RES,4).copy()
    finally:
        bpy.data.images.remove(image)
        output.unlink(missing_ok=True)
    req(np.isfinite(result).all() and
        np.min(result[...,3])>=-1e-5 and
        np.max(result[...,3])<=1.00001,"RENDER_RGBA_NONFINITE")
    return result

def main():
    raw=sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
    a=argparse.ArgumentParser(description=__doc__)
    for key in ("source","original-atlas","source-plan","prepared-atlas-manifest",
                "prepared-atlas-dir","camera-bundle","source-policy","output-dir"):
        a.add_argument("--"+key,type=Path,required=True)
    x=a.parse_args(raw)
    req(bpy.app.version>=(4,5,0),"BLENDER45_CYCLES_SAMPLE_SUBSET_REQUIRED")
    req(sha(x.source)==SOURCE_SHA and sha(x.original_atlas)==ATLAS_SHA,
        "ORIGINAL_SOURCE_BYTES")
    policy=read(x.source_policy)
    cameras=read(x.camera_bundle)
    plan=read(x.source_plan)
    variants=read(x.prepared_atlas_manifest)
    req(plan.get("schema")==PLAN and variants.get("schema")==ATLAS and
        plan.get("source_original_glb_sha256")==SOURCE_SHA and
        plan.get("source_original_texture_sha256")==ATLAS_SHA and
        plan.get("source_original_camera_sha256")==sha(x.camera_bundle) and
        plan.get("source_original_policy_sha256")==sha(x.source_policy) and
        plan.get("source_glb_triangle_count")==684 and
        len(plan["selected_original_source_uv_edits"])==3 and
        variants.get("source_operation_plan_sha256")==sha(x.source_plan) and
        variants.get("original_texture_source_sha256")==ATLAS_SHA and
        variants.get("original_source_glb_sha256")==SOURCE_SHA and
        [r["arm"] for r in variants["original_atlas_arm_rows"]]==list(ARMS) and
        [tuple(r["selected_source_uv_operation_indices"]) for r in
            variants["original_atlas_arm_rows"]]==[
                (), (0,), (1,), (2,), (0,1), (0,2), (1,2), (0,1,2)],
        "SOURCE_PLAN_OR_VARIANT_PARENT_DRIFT")
    req(policy.get("source_sha256")==SOURCE_SHA and
        policy.get("material_mode")=="unlit_preserve_alpha" and
        policy.get("camera_bundle",{}).get("sha256")==sha(x.camera_bundle) and
        len(cameras.get("cameras",[]))==8 and
        all(z.get(k) is False for z in (plan,variants) for k in FLAGS),
        "SOURCE_CAMERA_SHADER_AUTHORITY")

    out=x.output_dir.resolve()
    req(not out.exists(),"OUTPUT_ALREADY_EXISTS")
    out.mkdir(parents=True)
    import_rest(x.source,out/"import_receipt")
    meshes=sorted((o for o in bpy.context.scene.objects
                   if o.type=="MESH" and not o.hide_render),key=lambda o:o.name)
    req(bool(meshes),"IMPORTED_SOURCE_MESHES_ABSENT")
    materials=replay._materials_from_render_meshes(meshes)
    req(len(materials)==1,"SOURCE_LINKED_TEXTURE_MATERIAL_UNEXPECTED")
    candidate=policy.get("material_transform") or []
    req(len(candidate)==1 and candidate[0]["source_color_linked"] is True,
        "SOURCE_SHADER_NOT_TEXTURE_LINKED")
    receipts=replay._replay_transform(materials,candidate)
    req(len(receipts)==1 and receipts[0].get("replayed") is True and
        receipts[0].get("alpha_policy")==
            "IMPORTED_OPAQUE_MODE__BSDF_ALPHA_INACTIVE",
        "ORIGINAL_CROSS_BLENDER_ALPHA_REPLAY")
    original=next(iter(materials.values()))
    image_nodes=[n for n in original.node_tree.nodes if n.type=="TEX_IMAGE" and
                 n.image is not None]
    req(len(image_nodes)==1 and
        image_nodes[0].outputs["Color"].is_linked,
        "SOURCE_NATIVE_IMAGE_TEXTURE_NODE_MISSING")
    tex=image_nodes[0]
    original_image=tex.image
    req(tuple(original_image.size)==(512,512) and
        original_image.colorspace_settings.name in ("sRGB","sRGB EOTF"),
        "SOURCE_ORIGINAL_IMAGE_COLORSPACE_NOT_SRGB")
    original_alpha_mode=str(original_image.alpha_mode)
    original_colorspace=original_image.colorspace_settings.name
    loaded={}
    for row in variants["original_atlas_arm_rows"]:
        name=row["arm"]
        path=(x.prepared_atlas_dir/row["filename"]).resolve()
        req(path.is_relative_to(x.prepared_atlas_dir.resolve()) and
            path.is_file() and sha(path)==row["source_atlas_rgba_sha256"],
            "PREPARED_TEXTURE_IMAGE_SHA")
        asset=bpy.data.images.load(str(path),check_existing=False)
        req(tuple(asset.size)==(512,512),"PREPARED_TEXTURE_SIZE")
        asset.colorspace_settings.name=original_colorspace
        asset.alpha_mode=original_alpha_mode
        loaded[name]=asset
    req(sha(x.prepared_atlas_dir/"BASE.png")==ATLAS_SHA,
        "SOURCE_TEXTURE_ORIGINAL_BASE_NOT_BITWISE_IDENTICAL")

    configure(RES)
    scene=bpy.context.scene
    req(scene.render.engine=="CYCLES","CYCLES_RENDER_MISSING")
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
    camdata=bpy.data.cameras.new("PRIVATE_CAT_ORIGINAL_SOURCE_UV_EDIT_CAMERA")
    camdata.type="ORTHO"
    camera=bpy.data.objects.new(camdata.name,camdata)
    scene.collection.objects.link(camera)
    scene.camera=camera

    outputs=np.empty((len(VIEWS),len(SAMPLE_INDICES),len(ARMS),RES,RES,4),
                     dtype=np.float32)
    scratch=out/"scratch"
    scratch.mkdir()
    started=time.monotonic()
    try:
        for vi,view in enumerate(VIEWS):
            row=cameras["cameras"][view]
            req(row["view_index"]==view and row["view_id"]==f"V{view}" and
                row["resolution"]==1024,"ORIGINAL_CAMERA_ORDER")
            camera.location=Vector(tuple(map(float,row["origin"])))
            camera.rotation_euler=Vector(tuple(map(float,row["forward"]))).to_track_quat(
                "-Z","Y").to_euler()
            camdata.ortho_scale=2*float(row["half_extent"])
            camdata.clip_start=float(row["clip_start"])
            camdata.clip_end=float(row["clip_end"])
            bpy.context.view_layer.update()
            for si,source_idx in enumerate(SAMPLE_INDICES):
                scene.cycles.sample_offset=source_idx
                for ai,arm in enumerate(ARMS):
                    tex.image=loaded[arm]
                    bpy.context.view_layer.update()
                    outputs[vi,si,ai]=frame(scene,scratch/"sample.exr")
                print(json.dumps({"source_view":view,
                    "sample_idx":source_idx,"source_edited_full_arms":len(ARMS),
                    "elapsed_sec":round(time.monotonic()-started,1)}),
                    flush=True)
    finally:tex.image=original_image
    req(np.isfinite(outputs).all(),"RENDERED_SOURCE_ARRAY_NONFINITE")
    archive=out/"PRIVATE_CAT_SOURCE_UV_SAME_CYCLES_SAMPLES_V1.npz"
    np.savez_compressed(archive,source_uv_edit_cycles_sample_rgba=outputs)
    manifest={
        "schema":SCHEMA,"status":"ORIGINAL_SOURCE_CAT_TEXTURE_UV_CYCLES_RENDERED__INDEPENDENT_COURT_REQUIRED",
        "original_source_sha256":SOURCE_SHA,
        "original_source_atlas_sha256":ATLAS_SHA,
        "original_camera_bundle_sha256":sha(x.camera_bundle),
        "original_source_material_policy_sha256":sha(x.source_policy),
        "original_source_uv_edit_plan_sha256":sha(x.source_plan),
        "original_source_prepared_atlas_manifest_sha256":sha(x.prepared_atlas_manifest),
        "original_source_uv_texture_arms":list(ARMS),
        "source_atlas_texture_arm_sha256":[z["source_atlas_rgba_sha256"]
                                          for z in variants["original_atlas_arm_rows"]],
        "camera_indices":list(VIEWS),"cycles_subset_indices":list(SAMPLE_INDICES),
        "cycles_configured_samples":32,"cycles_seed":0,
        "cycles_adaptive_sampling":False,"cycles_subset_length":1,
        "cycles_pixel_filter":"BOX","cycles_filter_width":.01,
        "cycles_resolution":RES,
        "original_source_texture_colorspace":original_colorspace,
        "original_source_texture_alpha_mode":original_alpha_mode,
        "original_source_rest_only":True,
        "source_original_pose2_edit_court_passed":False,
        "source_uv_atlas_edit_rule":"ATLAS_TILE_U8_RGB_INVERT__ALPHA_UNCHANGED",
        "source_uv_png_edit_supports_disjoint":True,
        "source_original_sample_npz_sha256":sha(archive),
        "source_original_sample_npz_shape":list(outputs.shape),
        "all_texture_edit_sample_frames_independent_full_source_renders":True,
        "artist_drawable_owner":"UNKNOWN","same_drawable_training_pairs_qualified":0,
        "different_drawable_training_pairs_qualified":0,
        **{flag:False for flag in FLAGS}}
    path=out/"PRIVATE_CAT_SOURCE_UV_SAME_CYCLES_SAMPLES_V1.json"
    path.write_text(json.dumps(manifest,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":manifest["status"],
        "total_sample_renders":len(VIEWS)*len(SAMPLE_INDICES)*len(ARMS),
        "source_shader_image_link_preserved":True,
        "training_authorized":False},sort_keys=True),flush=True)

if __name__=="__main__":main()
