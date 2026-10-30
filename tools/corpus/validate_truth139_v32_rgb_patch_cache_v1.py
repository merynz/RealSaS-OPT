#!/usr/bin/env python3
"""Rebind V3.2 patch RGB evidence to exact source PNG bytes."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

SCHEMA="RealSaS.Truth139V32RGBPatchEvidenceCourt.v1"
STATUS="PASS_V32_RGB37_PATCH_EVIDENCE_REBINDS_EXACT_SOURCE_PIXELS"
PARENT_STATUS="PASS_V32_RGB37_PATCH_EVIDENCE_MATERIALIZED"

def sha256(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for c in iter(lambda:f.read(8<<20),b""):h.update(c)
    return h.hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def validate(manifest):
    if manifest.get("status")!=PARENT_STATUS or manifest.get("schema")!="RealSaS.Truth139V32RGBPatchEvidence.v1":raise ValueError("V32_RGB_COURT_PARENT")
    out=Path(manifest["output"]["path"]).resolve()
    if not out.is_file() or sha256(out)!=manifest["output"]["sha256"]:raise ValueError("V32_RGB_COURT_OUTPUT_DRIFT")
    with np.load(out,allow_pickle=False) as z:patch=np.asarray(z["rgb_patch_588"],dtype=np.float32);mean=np.asarray(z["rgb_patch_mean"],dtype=np.float32)
    if patch.shape!=(8,37,37,588) or mean.shape!=(8,37,37,3):raise ValueError("V32_RGB_COURT_LAYOUT")
    checks=[]
    for row in manifest.get("source_views") or []:
        vi=int(row["view_index"]);p=Path(row["path"]).resolve()
        if not p.is_file() or sha256(p)!=row["sha256"]:raise ValueError("V32_RGB_COURT_SOURCE_DRIFT")
        with Image.open(p) as im:rgba=np.asarray(im.convert("RGBA"),dtype=np.uint8)
        x=torch.from_numpy(rgba[...,:3].copy()).permute(2,0,1).unsqueeze(0).to(torch.float32).div(255.0)
        resized=TF.resize(x,[518,518],interpolation=InterpolationMode.BICUBIC,antialias=True).clamp(0,1)[0]
        for py,px in ((0,0),(7,13),(18,18),(29,5),(36,36)):
            expected=resized[:,py*14:(py+1)*14,px*14:(px+1)*14].permute(1,2,0).reshape(-1).numpy();got=patch[vi,py,px]
            err=float(np.max(np.abs(expected-got)))
            if err>6e-4:raise ValueError(f"V32_RGB_COURT_PATCH_DRIFT:{vi}:{py}:{px}:{err}")
            expected_mean=expected.reshape(196,3).mean(0);mean_err=float(np.max(np.abs(expected_mean-mean[vi,py,px])))
            if mean_err>6e-4:raise ValueError("V32_RGB_COURT_MEAN_DRIFT")
            checks.append(err)
    r={"schema":SCHEMA,"status":STATUS,"subject_id":manifest["subject_id"],"rgb_evidence_identity_sha256":manifest["rgb_evidence_identity_sha256"],"validated_patch_count":len(checks),"maximum_abs_error":float(max(checks) if checks else 0),"training_performed":False,"qualification_minted":False};r["court_identity_sha256"]=digest(r);return r

def main():
    p=argparse.ArgumentParser();p.add_argument("--rgb-manifest",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();r=validate(json.loads(a.rgb_manifest.resolve().read_text()));out=a.output.resolve();out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix(out.suffix+".partial");tmp.write_text(json.dumps(r,sort_keys=True,indent=2)+"\n");tmp.replace(out);print(json.dumps({"status":r["status"],"court_identity_sha256":r["court_identity_sha256"]},sort_keys=True))
if __name__=="__main__":main()
