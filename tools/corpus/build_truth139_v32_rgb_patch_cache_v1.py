#!/usr/bin/env python3
"""Materialize V3.2 patch-aligned direct RGB evidence for one Truth139 packet.

The resize path exactly matches the sealed DINOv2 input geometry (whole 1024 ->
518 bicubic antialiased). Values are kept in direct [0,1] photometric space and
are not ImageNet-normalized. Each 14x14 patch contains 588 RGB scalars.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

SCHEMA="RealSaS.Truth139V32RGBPatchEvidence.v1"
STATUS="PASS_V32_RGB37_PATCH_EVIDENCE_MATERIALIZED"

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda:f.read(8<<20),b""):h.update(c)
    return h.hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def write_npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".partial.npz");np.savez_compressed(tmp,**arrays);tmp.replace(path)

def build(subject_id:str,packet_root:Path,output_dir:Path)->dict:
    packet_root=packet_root.resolve();output_dir=output_dir.resolve();views=[];source_refs=[]
    for i in range(8):
        p=packet_root/f"V{i}.png"
        if not p.is_file():raise FileNotFoundError(f"V32_RGB_VIEW_MISSING:{subject_id}:V{i}")
        with Image.open(p) as im:
            rgba=np.asarray(im.convert("RGBA"),dtype=np.uint8)
        if rgba.shape!=(1024,1024,4):raise ValueError(f"V32_RGB_VIEW_LAYOUT:{subject_id}:V{i}:{rgba.shape}")
        views.append(torch.from_numpy(rgba[...,:3].copy()).permute(2,0,1))
        source_refs.append({"view_index":i,"path":str(p.resolve()),"sha256":sha256(p)})
    x=torch.stack(views,dim=0).to(torch.float32).div(255.0)
    resized=TF.resize(x,[518,518],interpolation=InterpolationMode.BICUBIC,antialias=True).clamp(0.0,1.0)
    patches=resized.unfold(2,14,14).unfold(3,14,14)  # [8,3,37,37,14,14]
    patches=patches.permute(0,2,3,4,5,1).contiguous().reshape(8,37,37,588)
    means=patches.reshape(8,37,37,196,3).mean(dim=3)
    out=output_dir/"V32_RGB_PATCH_EVIDENCE.npz"
    write_npz(out,rgb_patch_588=patches.cpu().numpy().astype(np.float16),rgb_patch_mean=means.cpu().numpy().astype(np.float16))
    payload={"schema":SCHEMA,"status":STATUS,"subject_id":subject_id,"packet_root":str(packet_root),"source_views":source_refs,"preprocess":{"native_hw":[1024,1024],"resize_hw":[518,518],"interpolation":"TORCHVISION_BICUBIC_ANTIALIAS_TRUE","patch_size":14,"patch_grid":[37,37],"rgb_scalars_per_patch":588,"range":"DIRECT_SOURCE_RGB_0_1","imagenet_normalization_applied":False},"output":{"path":str(out),"sha256":sha256(out)},"training_performed":False,"qualification_minted":False}
    payload["rgb_evidence_identity_sha256"]=digest(payload)
    manifest=output_dir/"V32_RGB_PATCH_EVIDENCE_MANIFEST.json";manifest.parent.mkdir(parents=True,exist_ok=True);manifest.write_text(json.dumps(payload,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"status":STATUS,"subject_id":subject_id,"rgb_evidence_identity_sha256":payload["rgb_evidence_identity_sha256"]},sort_keys=True));return payload

def main():
    p=argparse.ArgumentParser();p.add_argument("--subject-id",required=True);p.add_argument("--packet-root",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);a=p.parse_args();build(a.subject_id,a.packet_root,a.output_dir)
if __name__=="__main__":main()
