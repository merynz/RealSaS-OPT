from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tempfile
import zipfile

import numpy as np


def _load(root: Path):
    norm=json.loads((root/"08_NORMALIZATION_DOMAIN_QUALIFIED"/"normalization_domain.json").read_text())
    cameras=json.loads((root/"05_CAMERA_CONTRACT_SOLVED"/"qualified_camera_set.json").read_text())["cameras"]
    half=float(norm["half_extent"])
    if not math.isfinite(half) or half<=0:
        raise ValueError("invalid normalization half extent")
    masks=[]
    for view in range(8):
        raw=np.fromfile(root/"observations"/f"V{view}.mask.bin",dtype=np.uint8)
        if raw.size!=1024*1024 or np.any((raw!=0)&(raw!=1)):
            raise ValueError(f"invalid V{view} source mask")
        masks.append(raw.reshape(1024,1024).astype(bool))
    return half,sorted(cameras,key=lambda row:int(row["view_index"])),masks


def derive(*, evidence_zip: Path, lattice_resolution: int=512, guard_px: float=math.sqrt(2.0)) -> dict:
    r=int(lattice_resolution)
    if r<2 or not math.isfinite(float(guard_px)) or float(guard_px)<0:
        raise ValueError("invalid shell parameters")
    spacing=2.0/float(r-1)
    with tempfile.TemporaryDirectory(prefix="realsas-shell-") as tmp:
        root=Path(tmp)
        with zipfile.ZipFile(evidence_zip,"r") as zf:
            zf.extractall(root)
        norm_half,cameras,masks=_load(root)
        if [int(c["view_index"]) for c in cameras]!=list(range(8)):
            raise ValueError("exact ordered eight views required")
        rows=[]
        global_shell=0.0
        min_frame_margin=None
        for camera,mask in zip(cameras,masks):
            view=int(camera["view_index"])
            width=int(camera["resolution"])
            height=width
            half_norm=float(camera["half_extent"])/norm_half
            right=np.asarray(camera["right"],dtype=np.float64); right/=np.linalg.norm(right)
            up=np.asarray(camera["screen_up"],dtype=np.float64); up/=np.linalg.norm(up)
            scale=float(width)/(2.0*half_norm)
            maximum=0.0
            argmax=None
            for ix in (-1,0,1):
                for iy in (-1,0,1):
                    for iz in (-1,0,1):
                        if ix==iy==iz==0:
                            continue
                        delta=spacing*np.asarray([ix,iy,iz],dtype=np.float64)
                        dx=scale*float(delta@right)
                        dy=scale*float(delta@up)
                        dist=math.hypot(dx,dy)
                        if dist>maximum:
                            maximum=dist
                            argmax=[ix,iy,iz]
            shell=float(guard_px)+maximum
            ys,xs=np.nonzero(mask)
            frame_margin=int(min(xs.min(),width-1-xs.max(),ys.min(),height-1-ys.max()))
            min_frame_margin=frame_margin if min_frame_margin is None else min(min_frame_margin,frame_margin)
            global_shell=max(global_shell,shell)
            rows.append({
                "view_index":view,
                "camera_half_extent_normalized":half_norm,
                "pixel_scale_per_normalized_unit":scale,
                "maximum_projected_cell_corner_pair_distance_px":maximum,
                "argmax_axis_delta_in_lattice_steps":argmax,
                "guard_px":float(guard_px),
                "in_frame_boundary_crossing_cell_shell_upper_bound_px":shell,
                "source_foreground_minimum_frame_margin_px":frame_margin,
            })
    return {
        "schema":"RealSaS.IRIS.V44CertificationShellDerivation.v1",
        "lattice_resolution":r,
        "lattice_spacing_normalized":spacing,
        "pixel_quantization_guard_px":float(guard_px),
        "per_view":rows,
        "maximum_in_frame_boundary_crossing_cell_shell_upper_bound_px":global_shell,
        "minimum_source_foreground_frame_margin_px":int(min_frame_margin),
        "frame_margin_dominates_shell":bool(float(min_frame_margin)>global_shell),
        "derivation":"guard_px + maximum exact projection of any R512 axis-aligned MC cell corner-pair displacement under each qualified source camera",
        "scope":"IN_FRAME_BOUNDARY_CROSSING_MC_CELL_LOCALIZATION_BOUND__NOT_STAGE13_THRESHOLD",
        "claim_boundary":[
            "This is a geometric upper bound induced by the frozen source-hull lookup guard and R512 cell geometry.",
            "It does not prove learned residuals will attain this bound.",
            "It is not a Stage13 relaxation, a product gate, or a TP64 capacity proof."
        ],
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--evidence-zip",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--lattice-resolution",type=int,default=512)
    p.add_argument("--guard-px",type=float,default=math.sqrt(2.0))
    a=p.parse_args()
    result=derive(evidence_zip=a.evidence_zip,lattice_resolution=a.lattice_resolution,guard_px=a.guard_px)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
