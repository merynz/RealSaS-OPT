from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from PIL import Image

from compiler.realsas_compiler_core.hashing import content_sha256
from tools.demo.frozen.render_knight_motion_preview_v1_7917be02 import _ctx
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import stage_output_payload

RUNS = [
    "SUBJECT2_KNIGHT_DEMO_V2_20260924",
    "SUBJECT2_KNIGHT_SOLVED_LINEAGE_V1_20260929",
    "SUBJECT2_KNIGHT_CORRECTED_MECHANICAL_CAA_HARMONIC_FIRST_IMMUTABLE_V1_70CBD671",
]
STAGES = [
    ("18_CANONICAL_MESH_ADDRESSING_BUILD","RealSaS.CanonicalMeshCandidateIR.v1"),
    ("28_SKELETON_QUALIFIED","RealSaS.QualifiedSkeletonIR.v1"),
    ("32_SKIN_QUALIFIED","RealSaS.QualifiedSkinIR.v1"),
    ("23_COMPLETE_APPEARANCE_ASSET_BAKED","RealSaS.CompleteAppearanceAssetIR.v2"),
]

def img_diff(a: Path, b: Path):
    ia=np.asarray(Image.open(a).convert("RGBA"),dtype=np.int16)
    ib=np.asarray(Image.open(b).convert("RGBA"),dtype=np.int16)
    if ia.shape != ib.shape:
        return {"shape_a":list(ia.shape),"shape_b":list(ib.shape),"same_shape":False}
    d=np.abs(ia-ib)
    return {
        "same_shape":True,
        "exact_equal":bool(np.array_equal(ia,ib)),
        "pixel_exact_fraction":float(np.mean(np.all(d==0,axis=2))),
        "channel_mean_abs_diff":float(np.mean(d)),
        "channel_p95_abs_diff":float(np.percentile(d,95)),
        "channel_max_abs_diff":int(np.max(d)),
    }

def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--authority-root",type=Path,required=True)
    p.add_argument("--new-preview-root",type=Path,required=True)
    a=p.parse_args()
    out={"schema":"RealSaS.KnightOldVsCorrectedLineageAudit.v1","runs":{},"preview_candidates":{}}
    for rid in RUNS:
        root=a.authority_root/"runs"/rid
        row={"exists":root.exists(),"stages":{}}
        if root.exists():
            ctx=_ctx(a.authority_root,rid)
            for sid,schema in STAGES:
                try:
                    payload=stage_output_payload(ctx,sid,schema)
                    row["stages"][sid]={"payload_hash":content_sha256(payload)}
                except Exception as e:
                    row["stages"][sid]={"error":f"{type(e).__name__}:{e}"}
            found=[]
            for pth in root.rglob("KNIGHT_*_DEMO_PREVIEW_V1.png"):
                found.append(str(pth))
            row["preview_pngs"]=found[:50]
        out["runs"][rid]=row

    newroot=a.new_preview_root
    for name in ("IDLE","RUN","SLASH"):
        newp=newroot/f"KNIGHT_{name}_DEMO_PREVIEW_V1.png"
        matches=[]
        for rid,row in out["runs"].items():
            for s in row.get("preview_pngs",[]):
                oldp=Path(s)
                if oldp.name==newp.name and newp.exists():
                    matches.append({"run_id":rid,"path":s,"diff":img_diff(oldp,newp)})
        out["preview_candidates"][name]=matches
    print("KNIGHT_OLD_VS_CORRECTED_AUDIT="+json.dumps(out,sort_keys=True),flush=True)

if __name__=="__main__":
    main()
