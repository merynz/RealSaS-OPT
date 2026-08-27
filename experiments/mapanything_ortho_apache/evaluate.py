from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from realsas_mapanything_ortho.config import OrthoConfig
from realsas_mapanything_ortho.dataset import OrthoFamilyDataset, collate_one_family
from realsas_mapanything_ortho.model import MapAnythingOrthoIRIS
from realsas_mapanything_ortho.losses import geometry_metrics


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--manifest",required=True);ap.add_argument("--split",required=True);ap.add_argument("--checkpoint",required=True);ap.add_argument("--output",required=True);ap.add_argument("--baseline-json");args=ap.parse_args();cfg=OrthoConfig()
    if not torch.cuda.is_available():raise RuntimeError("CUDA_REQUIRED")
    device=torch.device("cuda");ds=OrthoFamilyDataset(args.manifest,args.split,cfg.native_size,cfg.target_xy_reference_size);dl=DataLoader(ds,batch_size=1,shuffle=False,num_workers=1,pin_memory=True,collate_fn=collate_one_family)
    ck=torch.load(args.checkpoint,map_location="cpu",weights_only=False);model=MapAnythingOrthoIRIS(cfg).to(device);model.load_state_dict(ck["model"],strict=True);model.eval();rows=[]
    with torch.inference_mode():
        for b in dl:
            images=b["images"].to(device);target={k:(v.to(device) if torch.is_tensor(v) else v) for k,v in b["target"].items()}
            with torch.autocast("cuda",dtype=torch.bfloat16):out=model(images)
            m=geometry_metrics(out,target);m["family_id"]=int(target["family_id"][0]);rows.append(m);print(json.dumps(m),flush=True)
    keys=["P_mean","P50","P90","P95","N_deg_median","N_deg_p95"];agg={k:float(np.mean([r[k] for r in rows])) for k in keys};agg["family_P95_p90"]=float(np.quantile([r["P95"] for r in rows],.90));agg["family_P95_worst"]=float(max(r["P95"] for r in rows))
    report={"schema":"RealSaS.IRIS.MapAnythingOrtho.Eval.v1","split":args.split,"aggregate":agg,"families":rows,"decision":{"status":"NO_MATCHED_BASELINE_SUPPLIED"}}
    if args.baseline_json:
        base=json.loads(Path(args.baseline_json).read_text());b=float(base["aggregate"]["P95"]);n=agg["P95"];gain=(b-n)/b if b>0 else float("nan");base_by={int(r["family_id"]):r for r in base.get("families",[])};family_reg=[]
        for r in rows:
            br=base_by.get(int(r["family_id"]))
            if br and float(br["P95"])>0:family_reg.append((float(r["P95"])-float(br["P95"]))/float(br["P95"]))
        max_reg=max(family_reg) if family_reg else None;promote=(gain>=.20) and (max_reg is None or max_reg<=.10)
        report["decision"]={"status":"PROMOTE" if promote else "DO_NOT_PROMOTE","matched_P95_reduction_fraction":gain,"max_family_P95_regression_fraction":max_reg,"rule":"aggregate >=20% P95 reduction; no matched family >10% P95 regression"};report["baseline"]=base
    Path(args.output).write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n");print(json.dumps(report,indent=2,sort_keys=True,allow_nan=False),flush=True)

if __name__=="__main__":main()
