#!/usr/bin/env python3
"""Independent SHA + pixel QA of a Graphics2DArtOrbit *candidate*.

This numerical gate cannot grant 2D art *quality* or source truth.
It is deliberately diagnostic and fail closed on spoofed image sources,
truncated view sets, alpha mismatch and unreadable/cropped renders.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

from PIL import Image

SCHEMA = "RealSaS.Graphics2DArtOrbitCandidatePixelAudit.v1"
RECEIPT_SCHEMA = "RealSaS.Graphics2DArtOrbitCandidate.v1"
SHA = re.compile(r"^[0-9a-f]{64}$")
MODES = ("source_reference", "cel_candidate", "inked_2d_candidate")


def require(v, why):
    if not v:
        raise ValueError("GRAPHICS_2D_ORBIT_" + why)


def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(4<<20),b""):
            h.update(chunk)
    return h.hexdigest()


def safe_png(root, name, expected):
    require(isinstance(name,str) and name==Path(name).name
            and name.endswith(".png"),"IMAGE_PATH")
    require(isinstance(expected,str) and SHA.fullmatch(expected),"IMAGE_SHA")
    f=root/name
    require(f.is_file() and digest(f)==expected, "PIXEL_HASH")
    with Image.open(f) as im:
        im.load()
        require(im.mode in {"RGBA","RGB"}, "IMAGE_MODE")
        img=im.convert("RGBA")
    require(img.size[0]==img.size[1] and 256<=img.size[0]<=1024,
            "IMAGE_DIMENSIONS")
    return img


def stats(im):
    a=im.getchannel("A")
    alpha=a.point(lambda x:255 if x>20 else 0)
    bbox=alpha.getbbox()
    require(bbox is not None,"EMPTY_IMAGE")
    w,h=im.size
    pix=sum(1 for x in alpha.getdata() if x)
    return {
      "bbox_fraction_x":(bbox[2]-bbox[0])/w,
      "bbox_fraction_y":(bbox[3]-bbox[1])/h,
      "mask_occupancy":pix/(w*h),
      "distance_from_crop_x":min(bbox[0],w-bbox[2])/w,
      "distance_from_crop_y":min(bbox[1],h-bbox[3])/h,
      "alpha_pixels":pix,
    }


def alpha_iou(a,b):
    aa=a.getchannel("A").point(lambda x:255 if x>20 else 0)
    bb=b.getchannel("A").point(lambda x:255 if x>20 else 0)
    inter=union=0
    for x,y in zip(aa.getdata(), bb.getdata()):
        if x or y:union+=1
        if x and y:inter+=1
    require(union>0,"EMPTY_ALPHA_UNION")
    return inter/union


def audit(root, *, enforce=False):
    root=root.resolve()
    manifest=root/"GRAPHICS_2D_ART_ORBIT_CANDIDATE_V1.json"
    doc=json.loads(manifest.read_text())
    require(doc.get("schema")==RECEIPT_SCHEMA,"MANIFEST_SCHEMA")
    require(doc.get("status")=="EIGHT_SOURCE_SPACE_STYLIZED_VIEWS_DIAGNOSTIC_ONLY",
            "MANIFEST_STATUS")
    require(doc.get("training_authorized") is False
            and doc.get("graphics_style_2d_truth_qualified") is False
            and doc.get("original_source_rgb_authority_qualified") is False,
            "AUTHORITY_FORGERY")
    source_sha=doc.get("source_sha256")
    require(isinstance(source_sha,str) and SHA.fullmatch(source_sha),
            "SOURCE_SHA")
    require(doc.get("style_origin")=="DETERMINISTIC_NON_GENERATIVE_RENDER",
            "GENERATIVE_ORIGIN")
    rows=doc.get("views")
    require(doc.get("view_count")==8 and isinstance(rows,list)
            and len(rows)==8,"VIEW_COUNT")
    ids=set()
    results=[]
    for i,v in enumerate(rows):
        expected="SOURCE_ORBIT_%02d"%i
        require(v.get("view_id")==expected and expected not in ids,
                "VIEW_ID")
        ids.add(expected)
        require(v.get("source_space_orbit_degrees")==45*i,
                "CAMERA_ORBIT_ANGLE")
        m=v.get("camera_source_space_matrix")
        require(isinstance(m,list) and len(m)==4 and
                all(isinstance(row,list) and len(row)==4 and
                    all(isinstance(x,(int,float)) and math.isfinite(x) for x in row)
                    for row in m),"CAMERA_MATRIX")
        images={}
        for mode in MODES:
            images[mode]=safe_png(root,v.get(mode+"_png"),v.get(mode+"_sha256"))
        require(len({x.size for x in images.values()})==1,
                "PAIR_IMAGE_SIZE")
        a=images["source_reference"]
        b=images["cel_candidate"]
        c=images["inked_2d_candidate"]
        scalars={mode:stats(im) for mode,im in images.items()}
        alpha_src_style=alpha_iou(a,c)
        alpha_cel_ink=alpha_iou(b,c)
        silhouette_valid=(alpha_src_style>=.98 and alpha_cel_ink==1.0)
        crop_valid=all(
            z["bbox_fraction_x"]>=.42 and z["bbox_fraction_y"]>=.42
            and z["bbox_fraction_x"]<=.95 and z["bbox_fraction_y"]<=.95
            and z["mask_occupancy"]>=.07
            for z in scalars.values())
        results.append({
            "view_id":expected,
            "alpha_src_to_2d_jaccard":alpha_src_style,
            "alpha_cel_to_2d_jaccard":alpha_cel_ink,
            "scene_framing_pass":crop_valid,
            "render_silhouette_parity_pass":silhouette_valid,
            "observed_masks":scalars,
            "numeric_style_candidate_pass":crop_valid and silhouette_valid,
            "human_2d_art_quality_court":"UNKNOWN",
        })
    require(len({tuple(tuple(x) for x in v["camera_source_space_matrix"])
                 for v in rows})==8,"DUPLICATE_CAMERA")
    passed=all(x["numeric_style_candidate_pass"] for x in results)
    report={
      "schema":SCHEMA,"asset_id":doc["asset_id"],
      "source_sha256":source_sha,
      "candidate_manifest_sha256":digest(manifest),
      "view_count":8,
      "numerically_suitable_stylized_2d_candidate":passed,
      "per_view":results,
      "2d_art_human_visual_quality_status":"UNKNOWN",
      "native_source_color_and_texture_parity_status":"UNKNOWN",
      "anatomical_view_labels_status":"UNKNOWN",
      "source_face_to_pixel_mapping_status":"UNKNOWN",
      "model_graphics_target_status":"UNKNOWN",
      "style_training_authorized":False,
      "training_authorized":False,
    }
    if enforce:require(passed,"NUMERIC_CANDIDATE_FAILED")
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--enforce",action="store_true")
    a=p.parse_args()
    d=audit(a.root,enforce=a.enforce)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(d,indent=2,sort_keys=True)+"\n")
    print(json.dumps({
      "asset_id":d["asset_id"],"views":d["view_count"],
      "numerically_suitable":d["numerically_suitable_stylized_2d_candidate"],
      "training_authorized":False,
    },sort_keys=True))


if __name__=="__main__":
    main()
