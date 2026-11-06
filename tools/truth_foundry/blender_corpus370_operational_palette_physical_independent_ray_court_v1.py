#!/usr/bin/env python3
"""Independent original Blender source-ray × observed edit-raster court V1.

Imports the ORIGINAL FBX anew, never imports the edit renderer. The court
checks SHA-bound source/camera/palette, per-view original physical alpha
invariance, and source-ray-qualified *interior* pixels that must change or
remain fixed. Edge/boundary pixels are explicitly out of the locality court.
This only courts original rest-pose color operations; it is NOT owner truth.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
import bpy
from mathutils import Vector

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.observation.blender_controlled_8view_v2 import import_rest
from tools.corpus.court_corpus370_operational_palette_pixel_consequence_v1 import judge_edit_rgba

SCHEMA="RealSaS.IndependentOriginalPhysicalPaletteSourceRayCourt.v1"
PRODUCER="RealSaS.OriginalSourcePhysicalPaletteEditRenderCandidate.v1"
VIEWS=(0,2,4,6)
STRIDE=4
RES=256
MIN_SELECTED=8
MIN_UNSELECTED=8

def require(cond,msg):
    if not cond:raise ValueError("PALETTE_PHYSICAL_SOURCE_COURT_"+msg)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()

def load(path):return json.loads(Path(path).read_text(encoding="utf-8"))

def read_rgba_png(path):
    image=bpy.data.images.load(str(path),check_existing=False)
    try:
        w,h=(int(x) for x in image.size)
        require(w==RES and h==RES,"ORIGINAL_RASTER_RESOLUTION")
        out=np.empty(w*h*4,dtype=np.float32)
        image.pixels.foreach_get(out)
        return out.reshape((h,w,4))[::-1,:,:].copy()
    finally:bpy.data.images.remove(image)

def material_key(material):
    if material is None or not material.use_nodes or material.node_tree is None:
        return None
    nodes=[x for x in material.node_tree.nodes if x.type=="BSDF_PRINCIPLED"]
    if len(nodes)!=1:return None
    base=nodes[0].inputs.get("Base Color")
    if base is None or base.is_linked:return None
    value=tuple(float(x) for x in base.default_value)
    if len(value)!=4 or not all(math.isfinite(x) for x in value):return None
    return tuple(v.hex() for v in value)

def ray_color(deps,origin,forward,materials):
    hit,location,normal,index,obj,matrix=bpy.context.scene.ray_cast(
        deps,origin,forward,distance=200.0)
    if not hit or obj is None or obj.type!="MESH":return None
    if index<0 or index>=len(obj.data.polygons):return None
    material_slot=int(obj.data.polygons[index].material_index)
    if not 0<=material_slot<len(obj.material_slots):return None
    mat=obj.material_slots[material_slot].material
    k=material_key(mat)
    return materials.get(k) if k is not None else None

def original_source_rays(camera,available_keys):
    """Independent per-pixel source-world ray intersections at safe interiors."""
    deps=bpy.context.evaluated_depsgraph_get()
    color_ids={tuple(row):index for index,row in enumerate(available_keys)}
    views={}
    for vi in VIEWS:
        c=camera["cameras"][vi]
        require(int(c["view_index"])==vi and c["view_id"]==f"V{vi}"
            and int(c["resolution"])==1024,"ORIGINAL_CAMERA_ADDRESS")
        origin=Vector(tuple(map(float,c["origin"])))
        right=Vector(tuple(map(float,c["right"])))
        screen_up=Vector(tuple(map(float,c["screen_up"])))
        forward=Vector(tuple(map(float,c["forward"])))
        extent=float(c["half_extent"])
        require(forward.length>0 and right.length>0 and screen_up.length>0
            and extent>0 and math.isfinite(extent),"ORIGINAL_CAMERA_INVALID")
        forward.normalize()
        selected=[]
        counts={"background_or_unsupported":0,"boundary_or_cross_palette":0,
                "qualified_interiors":0}
        def sample(x,y):
            px=(2.0*((x+0.5)/RES)-1.0)*extent
            py=(1.0-2.0*((y+0.5)/RES))*extent
            return ray_color(deps,origin+px*right+py*screen_up,forward,color_ids)
        for y in range(2,RES-2,STRIDE):
            for x in range(2,RES-2,STRIDE):
                center=sample(x,y)
                if center is None:
                    counts["background_or_unsupported"]+=1
                    continue
                # Cross-color, surface silhouettes and true subpixel edges
                # must not be allowed into a blanket "untouched" claim.
                neighbors=(sample(x-2,y),sample(x+2,y),
                           sample(x,y-2),sample(x,y+2))
                if any(k!=center for k in neighbors):
                    counts["boundary_or_cross_palette"]+=1
                    continue
                selected.append({"x":x,"y":y,"color_index":int(center)})
                counts["qualified_interiors"]+=1
        views[str(vi)]={"samples":selected,"coverage":counts}
    return views

def court(source,source_sha,source_policy_path,camera_path,palette_path,
          manifest_path,render_root,outdir):
    manifest=load(manifest_path)
    palette=load(palette_path)
    original_policy=load(source_policy_path)
    camera=load(camera_path)
    require(manifest.get("schema")==PRODUCER
        and manifest.get("policy")=="ORIGINAL_SHADER_CONSTANT_PALETTE__CONTROLLED_UNLIT_REPLAY__CYCLES_256_16SPP"
        and manifest.get("rendered_view_indices")==list(VIEWS)
        and manifest.get("fit_view_indices")==[0,2]
        and manifest.get("heldout_view_indices")==[4,6]
        and manifest.get("render_resolution")==RES
        and manifest.get("cycles_samples")==16
        and manifest.get("independent_ray_consequence_court")=="NOT_EVALUATED",
        "MANIFEST_POLICY_CHANGED")
    require(sha(source)==source_sha==manifest["original_source_sha256"]
        ==palette["original_source_sha256"]
        ==original_policy["source_sha256"],"ORIGINAL_SOURCE_SHA_DRIFT")
    require(manifest["source_camera_bundle_sha256"]==sha(camera_path)
        and manifest["source_palette_facts_sha256"]==sha(palette_path)
        and manifest["original_source_policy_sha256"]==sha(source_policy_path)
        and original_policy["camera_bundle"]["sha256"]==sha(camera_path),
        "SOURCE_PARENT_HASH_DRIFT")
    colors=palette["constant_palette_color_keys_hex_rgba"]
    require(manifest["palette_color_count"]==len(colors)==3
        and manifest["source_face_count"]==palette["source_face_count"]==1334
        and palette["qualified_constant_face_count"]==1334
        and palette["unsupported_face_count"]==0
        and manifest.get("original_artist_owner")=="UNKNOWN"
        and manifest.get("graphics_owner_authorized") is False
        and manifest.get("iris_training_authorized") is False
        and manifest.get("training_authorized") is False,
        "NOT_A_QUALIFIED_TRICERATOPS_PALETTE")
    require(len(manifest["edits"])==len(colors),"MISSING_PALETTE_EDIT")

    # The source ray court is independent of the producer's mesh/material
    # address inventory and does not trust its proposed visibility masks.
    import_rest(source,outdir/"independent_original_import")
    material_keys={}
    for obj in bpy.context.scene.objects:
        if obj.type!="MESH" or obj.hide_render:continue
        for slot in obj.material_slots:
            mat=slot.material
            if mat is None:continue
            k=material_key(mat)
            if k is not None:
                material_keys[k]=material_keys.get(k,0)+1
    expected_keys={tuple(x) for x in colors}
    require(set(material_keys)==expected_keys,"ORIGINAL_SOURCE_SHADER_RGB_CHANGED")
    source_rays=original_source_rays(camera,colors)
    receipts=[]
    sample_totals={k:{"selected":0,"unselected":0,"selected_unchanged":0,
                       "collateral":0,"max_alpha":0.0} for k in range(len(colors))}
    for vi in VIEWS:
        base_row=manifest["baseline"][str(vi)]
        base_file=render_root/"BASELINE"/base_row["file"]
        require(sha(base_file)==base_row["sha256"],"BASELINE_RASTER_HASH_DRIFT")
        base=read_rgba_png(base_file)
        for edit in manifest["edits"]:
            idx=edit["color_key_index"]
            require(type(idx) is int and idx in sample_totals
                and edit["source_hex_rgba"]==colors[idx]
                and edit["physical_palette_edit_rule"]
                    =="RGB_COMPONENT_TO_0P15_IF_GE_0P5_ELSE_0P85__ALPHA_RETAINED",
                "EDIT_COLOR_ADDRESS_DRIFT")
            r=edit["raster"][str(vi)]
            edited=render_root/f"EDIT_{idx}"/r["file"]
            require(sha(edited)==r["sha256"],"EDIT_RASTER_SHA_DRIFT")
            obj=read_rgba_png(edited)
            valid_samples=[s for s in source_rays[str(vi)]["samples"]
                           if min(base[s["y"],s["x"],3],
                                  obj[s["y"],s["x"],3])>=0.995]
            claim=judge_edit_rgba(base,obj,valid_samples,idx,
                                  min_selected_samples=0,min_unselected_samples=0)
            sums=sample_totals[idx]
            sums["selected"]+=claim["selected_count"]
            sums["unselected"]+=claim["unselected_count"]
            sums["selected_unchanged"]+=claim["selected_unchanged_count"]
            sums["collateral"]+=claim["collateral_count"]
            sums["max_alpha"]=max(sums["max_alpha"],claim["alpha_max_difference"])
            receipts.append({"view_index":vi,"edit_palette_color_index":idx,
                "sample_count":len(valid_samples),"sampled_selected":claim["selected_count"],
                "sampled_unselected":claim["unselected_count"],
                "sampled_selected_unchanged":claim["selected_unchanged_count"],
                "sampled_collateral":claim["collateral_count"],
                "alpha_max_difference":claim["alpha_max_difference"],
                "unqualified_boundary_sample_count":
                    source_rays[str(vi)]["coverage"]["boundary_or_cross_palette"],
                "source_ray_qualified_interior_count":
                    source_rays[str(vi)]["coverage"]["qualified_interiors"]})
    failures=[]
    for idx,v in sample_totals.items():
        if v["selected"]<MIN_SELECTED:
            failures.append(f"EDIT_{idx}_INSUFFICIENT_VISIBLE_SELECTED_INTERIORS")
        if v["unselected"]<MIN_UNSELECTED:
            failures.append(f"EDIT_{idx}_INSUFFICIENT_VISIBLE_COLLATERAL_INTERIORS")
        if v["selected_unchanged"]>0:
            failures.append(f"EDIT_{idx}_DID_NOT_CHANGE_SOURCE_SELECTED_INTERIORS")
        if v["collateral"]>0:
            failures.append(f"EDIT_{idx}_CHANGED_UNSELECTED_INTERIORS")
    result={"schema":SCHEMA,
        "status":"PASS_SOURCE_PHYSICAL_EDIT_INTERIOR_LOCALITY__NOT_DRAWABLE_OWNER"
            if not failures else "FAIL_SOURCE_PHYSICAL_EDIT_INTERIOR_LOCALITY",
        "original_source_sha256":source_sha,
        "original_raster_candidate_manifest_sha256":sha(manifest_path),
        "original_source_shader_palette_fact_sha256":sha(palette_path),
        "source_camera_bundle_sha256":sha(camera_path),
        "controlled_view_indices":list(VIEWS),
        "fit_view_indices":[0,2],"heldout_view_indices":[4,6],
        "pose_heldout_edit_test":"NOT_EVALUATED",
        "screen_pixel_classification":"ORIGINAL_REIMPORTED_SCENE_RAY_INTERIOR_ONLY__BOUNDARY_UNKNOWN",
        "sample_stride_px":STRIDE,"neighbor_margin_px":2,
        "min_visible_selected_interiors_per_color_across_views":MIN_SELECTED,
        "sampled_view_receipts":receipts,
        "original_source_ray_view_coverage":{k:v["coverage"] for k,v in source_rays.items()},
        "edit_color_totals":sample_totals,
        "failures":failures,
        "source_raw_lit_shading_parity":"NOT_EVALUATED",
        "all_pixels_physical_locality":"NOT_CLAIMED__ONLY_INTERIOR_SAMPLES_AND_GLOBAL_ALPHA",
        "original_art_drawable_ownership":"UNKNOWN",
        "graphics_owner_authorized":False,
        "iris_training_authorized":False,
        "tessa_training_authorized":False,
        "materialization_authorized":False,
        "training_authorized":False}
    outdir.mkdir(parents=True,exist_ok=True)
    dest=outdir/"PRIVATE_INDEPENDENT_ORIGINAL_PHYSICAL_PALETTE_RAY_COURT_V1.json"
    dest.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n",
                    encoding="utf-8")
    print(json.dumps({"status":result["status"],"source_ray_interior_counts":sample_totals,
        "failure_count":len(failures),"training_authorized":False},sort_keys=True),
        flush=True)
    require(not failures,"SOURCE_PIXELS_OR_RAY_SUPPORT_DID_NOT_PASS")

def main():
    raw=sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
    p=argparse.ArgumentParser(description=__doc__)
    for s in ("source","source-policy","camera-bundle","source-palette",
              "render-manifest","render-root","output-dir"):
        p.add_argument("--"+s,required=True,type=Path)
    p.add_argument("--source-sha256",required=True)
    a=p.parse_args(raw)
    court(a.source,a.source_sha256,a.source_policy,a.camera_bundle,a.source_palette,
          a.render_manifest,a.render_root,a.output_dir)

if __name__=="__main__":main()
