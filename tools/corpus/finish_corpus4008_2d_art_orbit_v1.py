#!/usr/bin/env python3
"""Deterministic host-Pillow ink pass for exact source/cel 8-view Blender orbit.

Only writes derived 2D render CANDIDATES and a manifest checksum.
Never claims original art, native 2D rendering, independent visual PASS,
loss/trainer suitability or training admission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from tools.corpus.blender_corpus4008_2d_art_orbit_pilot_v1 import image_ink

SCHEMA = "RealSaS.Graphics2DArtOrbitCandidate.v1"


def require(x, code):
    if not x:
        raise ValueError("GRAPHICS_2D_INK_" + code)


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(4<<20),b""):
            h.update(b)
    return h.hexdigest()


def run(root: Path):
    root=root.resolve()
    path=root/"GRAPHICS_2D_ART_ORBIT_CANDIDATE_V1.json"
    doc=json.loads(path.read_text())
    require(doc.get("schema")==SCHEMA,"SCHEMA")
    require(doc.get("status")=="EIGHT_SOURCE_SPACE_SOURCE_AND_CEL_READY_AWAITING_INK",
            "EXPECTED_UNPROCESSED_MANIFEST")
    require(doc.get("training_authorized") is False
            and doc.get("graphics_style_2d_truth_qualified") is False,
            "NO_TRAINING_AUTHORITY")
    require(doc.get("view_count")==8 and len(doc.get("views",[]))==8,
            "VIEW_COUNT")
    for i,v in enumerate(doc["views"]):
        require(v["view_id"]=="SOURCE_ORBIT_%02d"%i,
                "VIEW_ID")
        for prefix in ("source_reference","cel_candidate"):
            filename=v[prefix+"_png"]
            expected=v[prefix+"_sha256"]
            require(isinstance(filename,str) and filename==Path(filename).name
                    and filename.endswith(".png"),
                    "SOURCE_PNG_PATH")
            require(sha(root/filename)==expected,"SOURCE_PNG_SHA")
        target=v["inked_2d_candidate_png"]
        require(target=="inked_2d_orbit_%02d.png"%i, "INK_PATH")
        require(v["inked_2d_candidate_sha256"] is None,
                "NO_OVERWRITE_VERIFIED_STYLE")
        info=image_ink(root/v["cel_candidate_png"],root/target)
        require(info["source_stylized_sha256"]==v["cel_candidate_sha256"],
                "CEL_SHA_CHANGED")
        v["inked_2d_candidate_sha256"]=info["inked_target_sha256"]
        v["inked_2d_alpha_identical_to_cel"]=info["style_alpha_identical_to_cel"]
        v["view_2d_art_truth_admitted"]=False
    doc["status"]="EIGHT_SOURCE_SPACE_STYLIZED_VIEWS_DIAGNOSTIC_ONLY"
    doc["ink_postprocess_completed"]=True
    doc["training_authorized"]=False
    doc["graphics_style_2d_truth_qualified"]=False
    dst=path.with_suffix(".json.tmp")
    dst.write_text(json.dumps(doc,indent=2,sort_keys=True,
                              allow_nan=False)+"\n")
    os.replace(dst,path)
    return {
        "schema":doc["schema"],"inked_png_count":8,
        "manifest_sha256":sha(path),
        "human_2d_art_review":"PENDING",
        "training_authorized":False,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(run(args.root),sort_keys=True))


if __name__=="__main__":
    main()
