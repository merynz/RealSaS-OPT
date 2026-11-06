#!/usr/bin/env python3
"""Original Triceratops 3-palette 1spp SAME Cycles index physical source renderer.

Source-opaque controlled appearance only; physically delete other source
triangles in isolated clone meshes. FULL / EDITED_FULL are heldout court targets.
No artist drawable names or training supervision minted.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import bpy
import numpy as np
from mathutils import Vector

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.observation.blender_controlled_8view_v2 import (
    import_rest, configure
)
from tools.truth_foundry import blender_source_appearance_policy_replay_v1 as replay
from tools.truth_foundry.blender_corpus370_v18m_cycles_source_depth import (
    source_depth_material, normalize_source_depth_for_camera,
)

SCHEMA="RealSaS.Corpus370OriginalThreePaletteCommonCyclesSampleEditEvidence.v1"
SOURCE_SHA="ad66b51cc5fc44c9a3429a5b1b98a61621af08d58d102d8a817499bd6a7f5b5a"
VIEWS=(1,3,5,7)
SAMPLES=(0,4,8,12,16,20,24,28)
RES=192
ARMS=("SOURCE_FULL","ISOLATE_0","ISOLATE_1","ISOLATE_2",
      "DEPTH_0","DEPTH_1","DEPTH_2",
      "EDITED_ISOLATE_0","EDITED_ISOLATE_1","EDITED_ISOLATE_2",
      "EDITED_SOURCE_FULL_0","EDITED_SOURCE_FULL_1","EDITED_SOURCE_FULL_2")
DENIED=("graphics_owner_authorized","iris_training_authorized",
        "tessa_training_authorized","materialization_authorized","training_authorized")

def req(v,what):
    if not v:raise ValueError("V18_SOURCE_K3_PHYSICAL_"+what)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def rgb_key(mat):
    req(mat is not None and mat.use_nodes and mat.node_tree,"UNQUALIFIED_SOURCE_MATERIAL")
    principled=[n for n in mat.node_tree.nodes if n.type=="BSDF_PRINCIPLED"]
    req(len(principled)==1,"SOURCE_PRINCIPLED_NOT_UNIQUE")
    base=principled[0].inputs.get("Base Color")
    alpha=principled[0].inputs.get("Alpha")
    req(base is not None and not base.is_linked
        and alpha is not None and not alpha.is_linked,
        "SOURCE_COLOR_OR_ALPHA_NOT_CONSTANT")
    return tuple(float(x).hex() for x in base.default_value)

def frame(scene,output):
    scene.render.filepath=str(output)
    output.unlink(missing_ok=True)
    bpy.ops.render.render(write_still=True)
    req(output.is_file() and output.stat().st_size>0,"ORIGINAL_SOURCE_EXR_MISSING")
    image=bpy.data.images.load(str(output),check_existing=False)
    try:
        req(tuple(int(v) for v in image.size)==(RES,RES),"SOURCE_FRAME_DIMENSION")
        arr=np.empty(RES*RES*4,dtype=np.float32)
        image.pixels.foreach_get(arr)
        img=arr.reshape(RES,RES,4).copy()
    finally:
        bpy.data.images.remove(image)
        output.unlink(missing_ok=True)
    req(np.isfinite(img).all() and np.min(img[...,3])>=-1e-5
        and np.max(img[...,3])<=1.00001,"SOURCE_RGBA_NAN_OR_ALPHA")
    return img

def build_source_isolates(meshes,colors,original_faces,expected_keys):
    """Reconstruct isolated meshes from EXACT evaluated original source tris.

    FBX polygons are not necessarily triangles. The *source truth address* is
    the canonical evaluated Blender loop_triangle, not raw object polygons.
    These physical meshes contain ONLY their own source palette triangles.
    No nonselected geometry exists, and no transparent-face simulation occurs.
    """
    source_verts=[]
    source_faces=[]
    by_color=[0,0,0]
    clones={k:[] for k in range(3)}
    deps=bpy.context.evaluated_depsgraph_get()
    for obj in meshes:
        evaluated=obj.evaluated_get(deps)
        mesh=evaluated.to_mesh()
        try:
            mesh.calc_loop_triangles()
            start=len(source_verts)
            world=[tuple(float(x) for x in (evaluated.matrix_world@v.co))
                   for v in mesh.vertices]
            source_verts.extend(world)
            selected={k:[] for k in range(3)}
            selected_slots={k:[] for k in range(3)}
            for tri in mesh.loop_triangles:
                global_face=[start+int(k) for k in tri.vertices]
                source_faces.append(global_face)
                slot_idx=int(tri.material_index)
                req(0<=slot_idx<len(obj.material_slots),
                    "SOURCE_TRIANGLE_MATERIAL_SLOT_INVALID")
                mat=obj.material_slots[slot_idx].material
                key=rgb_key(mat)
                req(key in colors,"SOURCE_ORIGINAL_COLOR_NOT_QUALIFIED")
                group=colors.index(key)
                selected[group].append(tuple(int(k) for k in tri.vertices))
                selected_slots[group].append(slot_idx)
                by_color[group]+=1
            for k in range(3):
                part=bpy.data.meshes.new("PRIVATE_SOURCE_EXACT_TRIANGLES_"+str(k)+"_"+obj.name)
                part.from_pydata(world,[],selected[k])
                part.update()
                for slot in obj.material_slots:
                    req(slot.material is not None,"SOURCE_MATERIAL_SLOT_NULL")
                    part.materials.append(slot.material)
                req(len(part.polygons)==len(selected_slots[k]),
                    "TRIANGLE_ONLY_PHYSICAL_ISOLATE_CENSUS")
                for poly,slot_idx in zip(part.polygons,selected_slots[k]):
                    poly.material_index=slot_idx
                isolated=bpy.data.objects.new(
                    "PRIVATE_SOURCE_PALETTE_ISOLATE_"+str(k)+"_"+obj.name,part)
                bpy.context.scene.collection.objects.link(isolated)
                isolated.hide_render=True
                clones[k].append(isolated)
        finally:evaluated.to_mesh_clear()
    req(np.array_equal(np.asarray(source_faces,dtype=np.int64),original_faces),
        "SOURCE_EXACT_TRIANGLE_CANONICAL_FACE_ORDER_DRIFT")
    req(len(source_faces)==len(expected_keys),"SOURCE_FACE_KEY_CENSUS")
    # Crucially: actual source color classification came from Blender original
    # per-face material slots and MUST independently match the SHA-sealed
    # previously qualified original face-color membership.
    flattened=[]
    for obj in meshes:
        evaluated=obj.evaluated_get(deps)
        mesh=evaluated.to_mesh()
        try:
            mesh.calc_loop_triangles()
            for tri in mesh.loop_triangles:
                mat=obj.material_slots[int(tri.material_index)].material
                flattened.append(colors.index(rgb_key(mat)))
        finally:evaluated.to_mesh_clear()
    req(flattened==expected_keys,"SOURCE_ORIGINAL_FACE_COLOR_ADDRESS_DRIFT")
    req(sum(by_color)==len(original_faces) and all(k>0 for k in by_color),
        "ORIGINAL_PHYSICAL_FACE_CENSUS")
    req(sum(len(ob.data.polygons) for grp in clones.values() for ob in grp)
        ==len(original_faces),"ISOLATED_TRIANGLES_EXACT_UNION")
    return clones,by_color,np.asarray(source_verts,dtype=np.float64)

def prepare_original(args,out):
    policy=read(args.source_policy)
    palette=read(args.palette_facts)
    cameras=read(args.camera_bundle)
    req(sha(args.source)==SOURCE_SHA==policy.get("source_sha256")
        ==palette.get("original_source_sha256"),"SOURCE_FILE_SHA")
    req(palette.get("schema")=="RealSaS.Corpus370OriginalShaderConstantPaletteFacts.v1"
        and palette.get("qualified_constant_face_count")==1334
        and palette.get("source_face_count")==1334
        and palette.get("unsupported_face_count")==0
        and palette.get("palette_operation_count")==3
        and sha(args.palette_facts)=="8ff5534e4e8d7549cb9d419e4aea58511e0e042f9113e7d5807a8652a1b76edc"
        and sha(args.rest_geometry)==palette.get("original_rest_npz_sha256")
        and sha(args.source_policy)==palette.get("original_material_policy_sha256")
        and sha(args.camera_bundle)=="e73917f650ee2dd925efe68a88e2fafdd5f99dbc5b7c410b7de0532ecf66bb0d",
        "SOURCE_PALETTE_PARENT_SHA")
    req(cameras.get("schema")=="RealSaS.CameraProjectionBundle.v1"
        and [c["view_index"] for c in cameras["cameras"]]==list(range(8))
        and policy.get("material_mode")=="unlit_preserve_alpha"
        and policy.get("camera_bundle",{}).get("sha256")==sha(args.camera_bundle)
        and all(palette.get(k) is False for k in DENIED),
        "SOURCE_PALETTE_CAMERA_AND_AUTHORITY")
    with np.load(args.rest_geometry,allow_pickle=False) as rest:
        faces=np.asarray(rest["faces"],dtype=np.int64)
        world=np.asarray(rest["rest_vertices_source_world"],dtype=np.float64)
    req(faces.shape==(1334,3) and np.isfinite(world).all()
        and hashlib.sha256(np.ascontiguousarray(faces,dtype="<i8").tobytes()).hexdigest()
             ==palette["source_face_order_sha256"],
        "ORIGINAL_SOURCE_REST_GEOMETRY")
    colors=[tuple(v) for v in palette["constant_palette_color_keys_hex_rgba"]]
    req(len(set(colors))==3,"PALETTE_COLORS_NOT_UNIQUE")
    import_rest(args.source,out/"source_import")
    meshes=sorted((obj for obj in bpy.context.scene.objects
                   if obj.type=="MESH" and not obj.hide_render),key=lambda o:o.name)
    req(bool(meshes),"ORIGINAL_SOURCE_NO_MESH")
    clones,counts,source_world=build_source_isolates(
        meshes,colors,faces,palette["face_palette_color_key_index_or_null"])
    req(source_world.shape==world.shape
        and np.allclose(source_world,world,atol=2e-6,rtol=0),
        "SOURCE_IMPORT_WORLD_REST_DRIFT")
    source_mats={}
    for obj in meshes:
        for slot in obj.material_slots:
            mat=slot.material
            if mat is None:continue
            key=rgb_key(mat)
            req(key in colors,"SOURCE_MATERIAL_NOT_IN_PALETTE")
            if mat.name in source_mats:
                req(source_mats[mat.name][0] is mat,"SOURCE_MATERIAL_NAME_COLLISION")
            source_mats[mat.name]=(mat,colors.index(key))
    # V18n's prequalified cross-Blender source shader adapter consumes the
    # sealed 4.0 observation policy, NOT the 4.5 importer blend_method.
    # It rejects any upstream Base Color/Alpha link/default drift.
    receipts=replay._replay_transform(
        replay._materials_from_render_meshes(meshes),
        list(policy["material_transform"]))
    req(len(receipts)==len(policy["material_transform"])
        and {r["material"] for r in receipts}
            =={r["material"] for r in policy["material_transform"]}
        and all(r.get("replayed") is True for r in receipts),
        "SOURCE_ORIGINAL_MATERIAL_REPLAY_DID_NOT_CLOSE")
    for current in receipts:
        declared=next(x for x in policy["material_transform"]
                      if x["material"]==current["material"])
        for field in ("operation","alpha_policy"):
            req(current.get(field)==declared.get(field),
                "SOURCE_SEALED_ALPHA_REPLAY_POLICY_DRIFT_"+field)
    emitted={}
    for name,(mat,k) in source_mats.items():
        emissive=[v for v in mat.node_tree.nodes if v.type=="EMISSION"]
        req(len(emissive)==1 and not emissive[0].inputs["Color"].is_linked,
            "SOURCE_MATERIAL_NOT_ONE_COLOR_EMISSION")
        emitted[name]=(emissive[0].inputs["Color"],k)
    return meshes,clones,counts,world,cameras,emitted

def set_source_visibility(meshes,clones,group=None):
    for ob in meshes:ob.hide_render=group is not None
    for k,rows in clones.items():
        for ob in rows:ob.hide_render=(k!=group)
    bpy.context.view_layer.update()

def change_palette(emitted,group,new):
    old=[]
    for name,(sock,k) in emitted.items():
        if k!=group:continue
        prior=tuple(float(x) for x in sock.default_value)
        replacement=tuple(.15 if x>=.5 else .85 for x in prior[:3])+(prior[3],)
        sock.default_value=replacement if new else prior
        old.append((sock,prior))
    req(bool(old),"SOURCE_PALETTE_EDIT_EMPTY")
    return old

def main():
    raw=sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
    p=argparse.ArgumentParser()
    for key in ("source","source-policy","palette-facts","camera-bundle",
                "rest-geometry","output-dir"):
        p.add_argument("--"+key,required=True,type=Path)
    a=p.parse_args(raw)
    out=a.output_dir.resolve()
    req(not out.exists(),"OUTPUT_ALREADY_EXISTS")
    out.mkdir(parents=True)
    req(bpy.app.version>=(4,5,0),"CYCLES_SAMPLE_SUBSET_BLENDER45_REQUIRED")
    meshes,clones,counts,world,cameras,emitted=prepare_original(a,out)
    configure(RES)
    scene=bpy.context.scene
    req(scene.render.engine=="CYCLES","CYCLES_SOURCE_RENDER_ENGINE")
    required=("use_sample_subset","sample_offset","sample_subset_length",
              "use_adaptive_sampling","samples","seed","pixel_filter_type","filter_width")
    req(all(hasattr(scene.cycles,k) for k in required),"CYCLES_SAMPLE_API_MISSING")
    scene.cycles.samples=32
    scene.cycles.seed=0
    scene.cycles.use_adaptive_sampling=False
    scene.cycles.use_sample_subset=True
    scene.cycles.sample_subset_length=1
    scene.cycles.pixel_filter_type="BOX"
    scene.cycles.filter_width=.01
    scene.render.image_settings.file_format="OPEN_EXR"
    scene.render.image_settings.color_mode="RGBA"
    scene.render.image_settings.color_depth="32"
    scene.render.image_settings.exr_codec="ZIP"
    # Only source-private camera depth spans; never source FULL pixel values.
    depth_mat,depth_mapping=source_depth_material("PRIVATE_SOURCE_K3_CAMERA_VIEW_Z")
    slots={}
    for rows in clones.values():
        for obj in rows:
            slots[obj.name]=[mat for mat in obj.data.materials]
    camdata=bpy.data.cameras.new("PRIVATE_SOURCE_K3_ORIGINAL_CAMERA")
    camdata.type="ORTHO"
    cam=bpy.data.objects.new(camdata.name,camdata)
    scene.collection.objects.link(cam)
    scene.camera=cam
    data=np.empty((len(VIEWS),len(SAMPLES),len(ARMS),RES,RES,4),dtype=np.float32)
    depth_checks=[]
    scratch=out/"scratch"
    scratch.mkdir()
    started=time.monotonic()
    for vi,view in enumerate(VIEWS):
        row=cameras["cameras"][view]
        req(row["view_index"]==view and row["view_id"]==f"V{view}"
            and row["resolution"]==1024,"SOURCE_CAMERA_VIEW_BINDING")
        cam.location=Vector(tuple(float(x) for x in row["origin"]))
        cam.rotation_euler=Vector(tuple(float(x) for x in row["forward"])).to_track_quat("-Z","Y").to_euler()
        camdata.ortho_scale=2.*float(row["half_extent"])
        camdata.clip_start=float(row["clip_start"])
        camdata.clip_end=float(row["clip_end"])
        depth_bounds=normalize_source_depth_for_camera(depth_mapping,row,world)
        depth_checks.append(depth_bounds)
        scene.render.resolution_x=RES
        scene.render.resolution_y=RES
        scene.render.resolution_percentage=100
        for si,source_index in enumerate(SAMPLES):
            scene.cycles.sample_offset=source_index
            for ai,arm in enumerate(ARMS):
                if arm=="SOURCE_FULL":
                    set_source_visibility(meshes,clones)
                    frame_rgba=frame(scene,scratch/"sample.exr")
                elif arm.startswith("ISOLATE_"):
                    k=int(arm.rsplit("_",1)[1])
                    set_source_visibility(meshes,clones,k)
                    frame_rgba=frame(scene,scratch/"sample.exr")
                elif arm.startswith("DEPTH_"):
                    k=int(arm.rsplit("_",1)[1])
                    set_source_visibility(meshes,clones,k)
                    for ob in clones[k]:
                        for slot in range(len(ob.data.materials)):
                            ob.data.materials[slot]=depth_mat
                    try:frame_rgba=frame(scene,scratch/"sample.exr")
                    finally:
                        for ob in clones[k]:
                            for idx,mat in enumerate(slots[ob.name]):
                                ob.data.materials[idx]=mat
                elif arm.startswith("EDITED_ISOLATE_") or arm.startswith("EDITED_SOURCE_FULL_"):
                    k=int(arm.rsplit("_",1)[1])
                    set_source_visibility(meshes,clones,
                                          k if arm.startswith("EDITED_ISOLATE_") else None)
                    replaced=change_palette(emitted,k,True)
                    try:frame_rgba=frame(scene,scratch/"sample.exr")
                    finally:
                        for sock,value in replaced:sock.default_value=value
                else:
                    raise ValueError("SOURCE_K3_UNSUPPORTED_RENDER_ARM")
                data[vi,si,ai]=frame_rgba
            print(json.dumps({"view":view,"source_sample_index":source_index,
                "completed_arms":len(ARMS),"elapsed_seconds":round(time.monotonic()-started,1)}),
                flush=True)
    req(np.isfinite(data).all(),"SAMPLE_IMAGES_NONFINITE")
    archive=out/"PRIVATE_ORIGINAL_SOURCE_K3_SAME_CYCLES_SAMPLES_V1.npz"
    np.savez_compressed(archive,source_original_palette_k3_sample_rgba=data)
    report={"schema":SCHEMA,"status":"SOURCE_ORIGINAL_K3_CYCLES_EDIT_SAMPLES_RENDERED__COURT_REQUIRED",
        "original_source_sha256":SOURCE_SHA,
        "source_original_material_palette_sha256":sha(a.palette_facts),
        "source_original_rest_geometry_sha256":sha(a.rest_geometry),
        "source_original_camera_bundle_sha256":sha(a.camera_bundle),
        "source_original_render_policy_sha256":sha(a.source_policy),
        "source_cycle_sample_indices":list(SAMPLES),
        "source_cycle_camera_view_indices":list(VIEWS),
        "source_cycle_arms":list(ARMS),
        "source_cycle_samples_max":32,"source_cycle_subset_length":1,
        "source_cycle_seed":0,"source_cycle_adaptive_sampling":False,
        "source_cycle_pixel_filter":"BOX","source_cycle_filter_width":.01,
        "source_cycle_resolution":RES,
        "source_original_artwork_heldout_pose_test":False,
        "source_original_rest_pose_only":True,
        "source_original_triangles_per_color":counts,
        "physical_nonselected_faces_removed":True,
        "source_depth_normalization_bounds_by_view":depth_checks,
        "source_color_edit_rule":"RGB_COMPONENT_GE_HALF_TO_0P15_OTHER_TO_0P85_ALPHA_UNCHANGED",
        "source_full_and_edited_full_are_court_targets_only":True,
        "source_original_edited_sample_npz_sha256":sha(archive),
        "source_original_edited_sample_npz_shape":list(data.shape),
        "original_artist_drawable_owner":"UNKNOWN",
        **{x:False for x in DENIED}}
    manifest=out/"PRIVATE_ORIGINAL_SOURCE_K3_SAME_CYCLES_SAMPLES_V1.json"
    manifest.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":report["status"],"sample_count":len(VIEWS)*len(SAMPLES),
        "render_count":len(VIEWS)*len(SAMPLES)*len(ARMS),
        "source_face_groups":counts,"training_authorized":False}),flush=True)

if __name__=="__main__":main()
