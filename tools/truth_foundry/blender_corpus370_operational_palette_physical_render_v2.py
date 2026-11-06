#!/usr/bin/env python3
"""Original Triceratops shader-palette physical edit producer V2 on disjoint camera directions.

Renders preregistered new V1/V3/V5/V7 at 256x256, Cycles CPU 16 fixed samples,
pinned original source material unlit-alpha replay policy. Each admitted source
color is edited exactly once, with all other original surface shaders retained.
No evidence from these renders is used to select color groups or thresholds.
No product/training/artist owner authority is minted.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.observation.blender_controlled_8view_v2 import import_rest,preserve_alpha_unlit,configure

SCHEMA="RealSaS.OriginalSourcePhysicalPaletteEditRenderCandidate.v2"
POLICY="ORIGINAL_SHADER_CONSTANT_PALETTE__CONTROLLED_UNLIT_REPLAY__CYCLES_256_16SPP__NEW_V1_V3_V5_V7"
VIEWS=(1,3,5,7)
RESOLUTION=256
DENIED=("graphics_owner_authorized","iris_training_authorized",
        "tessa_training_authorized","materialization_authorized","training_authorized")

def require(ok,why):
    if not ok:raise ValueError("PHYSICAL_PALETTE_PRODUCER_"+why)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))

def rgba(mat):
    if mat is None or not mat.use_nodes or mat.node_tree is None:return None
    n=[r for r in mat.node_tree.nodes if r.type=="BSDF_PRINCIPLED"]
    if len(n)!=1:return None
    col=n[0].inputs.get("Base Color")
    if col is None or col.is_linked:return None
    return tuple(float(x).hex() for x in col.default_value)

def verify_original(source,expected_sha,policy,camera,palette,v18a_npz):
    require(sha(source)==expected_sha==policy.get("source_sha256")
            ==palette.get("original_source_sha256"),"SOURCE_IDENTITY")
    require(policy.get("schema")=="RealSaS.Controlled8ViewSourceRestCandidate.v2"
            and policy.get("material_mode")=="unlit_preserve_alpha"
            and policy.get("training_admission_claimed") is False,
            "ORIGINAL_UNLIT_RENDER_POLICY")
    require(camera.get("schema")=="RealSaS.CameraProjectionBundle.v1"
            and [c["view_id"] for c in camera["cameras"]]==[f"V{i}" for i in range(8)]
            and len(camera["cameras"])==8,"SEALED_EIGHT_VIEW_CAMERAS")
    require(palette.get("schema")=="RealSaS.Corpus370OriginalShaderConstantPaletteFacts.v1"
            and palette.get("original_v18a_npz_sha256")==sha(v18a_npz)
            and palette.get("original_material_policy_sha256")==sha(policy["__path"])
            and palette.get("source_face_count")==1334
            and palette.get("qualified_constant_face_count")==1334
            and palette.get("unsupported_face_count")==0
            and palette.get("palette_operation_count")==3
            and palette.get("source_rendered_tint_consequence")=="NOT_EVALUATED",
            "PREDECLARED_TRICERATOPS_PALETTE_ONLY")
    require(sha(camera["__path"])==policy.get("camera_bundle",{}).get("sha256"),
            "CAMERA_SHA_BOUND_TO_ADMITTED_SOURCE")
    with np.load(v18a_npz,allow_pickle=False) as z:
        f=np.asarray(z["faces"],np.int64)
    require(f.shape==(1334,3)
            and hashlib.sha256(np.ascontiguousarray(f,dtype="<i8").tobytes()).hexdigest()
                ==palette["source_face_order_sha256"],
            "SOURCE_FACE_ADDRESS")
    require(all(palette.get(x) is False for x in DENIED),"AUTHORITY_INPUT_PROMOTED")
    return f

def material_index_map(meshes,faces,palette,policy):
    colors=tuple(tuple(x) for x in palette["constant_palette_color_keys_hex_rgba"])
    policy_by_name={x["material"]:x for x in policy["material_transform"]}
    materials={}
    observed_faces=[]
    observed_keys=[]
    deps=bpy.context.evaluated_depsgraph_get()
    offset=0
    for obj in meshes:
        e=obj.evaluated_get(deps)
        mesh=e.to_mesh()
        try:
            mesh.calc_loop_triangles()
            for t in mesh.loop_triangles:
                observed_faces.append([offset+int(k) for k in t.vertices])
                slot=obj.material_slots[int(t.material_index)]
                mat=slot.material
                require(mat is not None,"SOURCE_FACE_WITHOUT_MATERIAL")
                orig=rgba(mat)
                require(orig in colors,"SOURCE_ORIGINAL_COLOR_NOT_ADMITTED")
                rec=policy_by_name.get(mat.name)
                require(rec is not None
                        and rec.get("operation")=="BSDF_LIGHT_RESPONSE_TO_EMISSION__ALPHA_GRAPH_RETAINED"
                        and rec.get("alpha_policy")=="IMPORTED_OPAQUE_MODE__BSDF_ALPHA_INACTIVE"
                        and rec.get("source_color_linked") is False
                        and rec.get("source_alpha_linked") is False,
                        "UNVERIFIED_MATERIAL_POLICY")
                observed_keys.append(colors.index(orig))
                if mat.name in materials:require(materials[mat.name][0] is mat,"MATERIAL_NAME_COLLISION")
                materials[mat.name]=(mat,colors.index(orig))
            offset+=len(mesh.vertices)
        finally:e.to_mesh_clear()
    require(np.array_equal(np.asarray(observed_faces,dtype=np.int64),faces),
            "SOURCE_IMPORT_ORIGINAL_FACE_ORDER_DRIFT")
    require(observed_keys==palette["face_palette_color_key_index_or_null"],
            "SOURCE_ORIGINAL_SHADER_FACE_KEYS_DRIFT")
    return materials,colors

def render(path,camera,views):
    scene=bpy.context.scene
    cdata=bpy.data.cameras.new("RealSaS_OriginalPhysicalPaletteCamera")
    cdata.type="ORTHO"
    cam=bpy.data.objects.new(cdata.name,cdata)
    scene.collection.objects.link(cam);scene.camera=cam
    saved={}
    for vi in views:
        r=camera["cameras"][vi]
        origin=Vector(tuple(float(x) for x in r["origin"]))
        forward=Vector(tuple(float(x) for x in r["forward"]))
        require(r["view_index"]==vi and r["resolution"]==1024 and forward.length>0,
                "ORIGINAL_CAMERA_ADDRESS")
        cam.location=origin
        cam.rotation_euler=forward.to_track_quat("-Z","Y").to_euler()
        cdata.ortho_scale=2.0*float(r["half_extent"])
        cdata.clip_start=float(r["clip_start"])
        cdata.clip_end=float(r["clip_end"])
        bpy.context.view_layer.update()
        out=path/f"V{vi}.png"
        scene.render.filepath=str(out)
        bpy.ops.render.render(write_still=True)
        require(out.is_file() and out.stat().st_size>0,"RENDER_RASTER_MISSING")
        saved[str(vi)]={"file":out.name,"sha256":sha(out),"bytes":out.stat().st_size}
    return saved

def main():
    raw=sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("source","source-policy","camera-bundle","source-palette","v18a-npz","output-dir"):
        p.add_argument("--"+name,required=True,type=Path)
    p.add_argument("--source-sha256",required=True)
    a=p.parse_args(raw)
    out=a.output_dir.expanduser().resolve()
    require(not out.exists(),"NEW_OUTPUT_DIRECTORY_REQUIRED")
    out.mkdir(parents=True)
    policy=read(a.source_policy);policy["__path"]=a.source_policy
    camera=read(a.camera_bundle);camera["__path"]=a.camera_bundle
    palette=read(a.source_palette)
    faces=verify_original(a.source,a.source_sha256,policy,camera,palette,a.v18a_npz)
    import_rest(a.source,out/"import_rest")
    meshes=sorted((x for x in bpy.context.scene.objects if x.type=="MESH" and not x.hide_render),
                  key=lambda x:x.name)
    mats,colors=material_index_map(meshes,faces,palette,policy)
    receipts=preserve_alpha_unlit(meshes)
    require(len(receipts)==len(policy["material_transform"])
            and {r["material"] for r in receipts}
                =={r["material"] for r in policy["material_transform"]},
            "SOURCE_UNLIT_REPLAY_MATERIAL_SET_MISMATCH")
    for current in receipts:
        declared=next(x for x in policy["material_transform"]
                      if x["material"]==current["material"])
        for field in ("operation","alpha_policy","source_color_linked","source_alpha_linked"):
            require(current.get(field)==declared.get(field),
                    "SOURCE_UNLIT_REPLAY_POLICY_DRIFT_"+field)

    configure(RESOLUTION)
    baseline_dir=out/"BASELINE"
    baseline_dir.mkdir()
    baseline=render(baseline_dir,camera,VIEWS)
    edits=[]
    for k,color in enumerate(colors):
        changed=[]
        for name,(mat,idx) in mats.items():
            if idx!=k:continue
            emissions=[x for x in mat.node_tree.nodes if x.type=="EMISSION"]
            require(len(emissions)==1 and not emissions[0].inputs["Color"].is_linked,
                    "ORIGINAL_UNLIT_SHADER_EDIT_SITE_NOT_SINGLE")
            sock=emissions[0].inputs["Color"]
            old=tuple(float(x) for x in sock.default_value)
            new=tuple(0.15 if v>=0.5 else 0.85 for v in old[:3])+(old[3],)
            sock.default_value=new
            changed.append((sock,old,name,new))
        require(bool(changed),"EDIT_PALETTE_GROUP_EMPTY")
        dest=out/f"EDIT_{k}"
        dest.mkdir()
        observed=render(dest,camera,VIEWS)
        for sock,old,name,new in changed:sock.default_value=old
        edits.append({"color_key_index":k,"source_hex_rgba":list(color),
            "exact_source_material_count":len(changed),
            "physical_palette_edit_rule":"RGB_COMPONENT_TO_0P15_IF_GE_0P5_ELSE_0P85__ALPHA_RETAINED",
            "raster":observed})
    report={"schema":SCHEMA,"policy":POLICY,
        "original_source_sha256":a.source_sha256,
        "original_source_file_sha256":sha(a.source),
        "original_source_policy_sha256":sha(a.source_policy),
        "source_camera_bundle_sha256":sha(a.camera_bundle),
        "source_palette_facts_sha256":sha(a.source_palette),
        "original_v18a_npz_sha256":sha(a.v18a_npz),
        "face_order_sha256":palette["source_face_order_sha256"],
        "source_face_count":len(faces),
        "palette_color_count":len(colors),
        "rendered_view_indices":list(VIEWS),
        "prior_exploratory_views":[0,2,4,6],
        "new_disjoint_evaluation_views":[1,3,5,7],
        "render_resolution":RESOLUTION,
        "cycles_samples":16,"cycles_seed":0,
        "baseline":baseline,"edits":edits,
        "source_shader_replay":"ORIGINAL_SOURCE_UNLIT_ALPHA_POLICY__NOT_RAW_LIT_PARITY",
        "independent_ray_consequence_court":"NOT_EVALUATED",
        "original_artist_owner":"UNKNOWN",
        **{x:False for x in DENIED}}
    (out/"PRIVATE_ORIGINAL_PHYSICAL_PALETTE_RENDER_MANIFEST_V1.json").write_text(
        json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"status":"ORIGINAL_SOURCE_EDIT_RASTERS_PRODUCED__NO_OWNER",
        "edits":len(edits),"views":list(VIEWS),"training_authorized":False}),flush=True)

if __name__=="__main__":main()
