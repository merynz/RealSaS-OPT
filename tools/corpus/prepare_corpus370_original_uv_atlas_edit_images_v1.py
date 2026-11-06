#!/usr/bin/env python3
"""Make physical original Cat atlas counterfactuals from immutable UV operations.

Original image source bytes never edited. All variants are ordinary PNG assets
with no body/semantic identifiers and no inferred source layer ownership.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

SCHEMA="RealSaS.Corpus370OriginalSourceUVAtlasEditImages.v1"
ARMS={
    "BASE":(),
    "SINGLE_0":(0,),"SINGLE_1":(1,),"SINGLE_2":(2,),
    "PAIR_01":(0,1),"PAIR_02":(0,2),"PAIR_12":(1,2),
    "JOINT_012":(0,1,2),
}
FLAGS=("graphics_owner_authorized","iris_training_authorized",
       "tessa_training_authorized","materialization_authorized","training_authorized")

def req(ok,code):
    if not ok:raise ValueError("SOURCE_UV_PHYSICAL_IMAGES_"+code)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()

def build(original,plan,plan_sha,out):
    req(plan["schema"]=="RealSaS.Corpus370OriginalSourceUVAtlasEditPlan.v1"
        and plan["status"]=="ORIGINAL_SOURCE_UV_OPERATION_PLAN_FROZEN__NOT_RENDER_COURTED"
        and plan["source_original_texture_sha256"]==sha(original)
        and plan["original_source_texture_edit_rule"]==
            "ATLAS_TILE_U8_RGB_INVERT__ALPHA_UNCHANGED"
        and plan["original_source_uv_operation_count"]==3
        and plan["source_texture_size_px"]==[512,512]
        and all(plan.get(k) is False for k in FLAGS),
        "SOURCE_UV_PLAN_PROVENANCE")
    operations=plan["selected_original_source_uv_edits"]
    req([x["source_uv_edit_operation_index"] for x in operations]==[0,1,2],
        "OPERATION_INDEX_DRIFT")
    rectangles=[tuple(x["source_rgba_png_pixel_rectangle_xyxy_exclusive"])
                for x in operations]
    for (x0,y0,x1,y1),op in zip(rectangles,operations):
        req(x1-x0==y1-y0==64 and 0<=x0<x1<=512 and
            0<=y0<y1<=512 and (x0,y0,x1,y1)==
            (op["uv_tile_u"]*64,(7-op["uv_tile_v"])*64,
             (op["uv_tile_u"]+1)*64,(8-op["uv_tile_v"])*64),
            "SOURCE_UV_TILE_RGBA_RECT_NOT_64X64")
    masks=[]
    for x0,y0,x1,y1 in rectangles:
        m=np.zeros((512,512),dtype=np.bool_)
        m[y0:y1,x0:x1]=True
        masks.append(m)
    req(not np.any(masks[0]&masks[1]) and
        not np.any(masks[0]&masks[2]) and
        not np.any(masks[1]&masks[2]),
        "ATLAS_PATCHES_OVERLAP")
    with Image.open(original) as image:
        req(image.mode=="RGBA" and image.size==(512,512),
            "ORIGINAL_SOURCE_RGBA_FORMAT")
        rgba=np.asarray(image,dtype=np.uint8).copy()
    req(np.all(rgba[...,3]==255),"ORIGINAL_SOURCE_ALPHA_NOT_OPAQUE")
    out.mkdir(parents=True,exist_ok=False)
    rows=[]
    for arm,selected in ARMS.items():
        image=rgba.copy()
        for idx in selected:
            pix=masks[idx]
            image[...,0][pix]=255-rgba[...,0][pix]
            image[...,1][pix]=255-rgba[...,1][pix]
            image[...,2][pix]=255-rgba[...,2][pix]
        req(np.array_equal(image[...,3],rgba[...,3]),
            "EDIT_CORRUPTED_ORIGINAL_ALPHA")
        changed=np.any(image[...,:3]!=rgba[...,:3],axis=-1)
        mask=np.logical_or.reduce([masks[i] for i in selected]) if selected else np.zeros_like(masks[0])
        req(np.array_equal(changed,mask),"SOURCE_ACTUAL_RGB_CHANGE_SUPPORT_NOT_EXACT")
        target=out/(arm+".png")
        if arm=="BASE":
            # preserve ORIGINAL bytes and its color-profile metadata exactly
            target.write_bytes(Path(original).read_bytes())
        else:
            Image.fromarray(image,"RGBA").save(target,format="PNG",compress_level=9)
        rows.append({"arm":arm,"selected_source_uv_operation_indices":list(selected),
                     "original_atlas_edited_rgb_pixel_count":int(np.count_nonzero(mask)),
                     "source_atlas_rgba_sha256":sha(target),
                     "filename":target.name,
                     "atlas_pixel_edit_equals_original_declared_source_mask":True,
                     "source_texture_alpha_preserved_exactly":True})
    return {
        "schema":SCHEMA,
        "status":"PHYSICAL_ORIGINAL_SOURCE_UV_ATLAS_EDIT_IMAGES_PREPARED__CYCLES_COURT_REQUIRED",
        "original_source_glb_sha256":plan["source_original_glb_sha256"],
        "original_texture_source_sha256":sha(original),
        "source_operation_plan_sha256":plan_sha,
        "source_texture_rgba_resolution":[512,512],
        "source_atlas_edit_rule":plan["original_source_texture_edit_rule"],
        "original_atlas_arm_rows":rows,
        "original_artist_drawable_owner":"UNKNOWN",
        "source_original_atlas_operations_are_not_drawables":True,
        **{k:False for k in FLAGS},
    }

def main():
    a=argparse.ArgumentParser(description=__doc__)
    for key in ("original-atlas","source-plan","output-dir"):
        a.add_argument("--"+key,required=True,type=Path)
    opt=a.parse_args()
    plan=json.loads(opt.source_plan.read_text())
    report=build(opt.original_atlas,plan,sha(opt.source_plan),opt.output_dir)
    (opt.output_dir/"PRIVATE_SOURCE_UV_ATLAS_EDIT_IMAGES_V1.json").write_text(
        json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":report["status"],"atlas_arms":len(report["original_atlas_arm_rows"]),
        "source_art_drawable_owner":"UNKNOWN","training_authorized":False},sort_keys=True))

if __name__=="__main__":main()
