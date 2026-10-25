#!/usr/bin/env python3
"""Execute DINOV2_DUAL_LEVEL_VIEW_TOKEN_PREREG_V1 exactly."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from models.iris.v2.dinov2_foundation_v2 import load_exact_dinov2_s_v2
from experiments.iris_tessa_v3_apparatus_audit_20261025.dinov2_feature_access_probe_v1 import (
    WIDTHS,
    TRAIN_CANVASES,
    EVAL_CANVASES,
    MASTER_SEED,
    make_canvases,
    extract_all_layers,
)

SCHEMA = "RealSaS.DINOv2DualLevelViewTokenCourt.v1"
SEEDS = (731, 947, 1217, 1597, 2017)
STEPS = 1200
LR = 1e-3
EXPECTED_ADAPTER_PARAMS = 197120


class CurrentFinal512(nn.Module):
    def __init__(self):
        super().__init__(); self.adapter=nn.Linear(384,512)
    def forward(self,l2,final): del l2; return self.adapter(final)


class EarlyL2512(nn.Module):
    def __init__(self):
        super().__init__(); self.adapter=nn.Linear(384,512)
    def forward(self,l2,final): del final; return self.adapter(l2)


class DualMatched512(nn.Module):
    def __init__(self):
        super().__init__(); self.final_branch=nn.Linear(384,384); self.detail_branch=nn.Linear(384,128)
    def forward(self,l2,final): return torch.cat((self.final_branch(final),self.detail_branch(l2)),-1)


class CourtModel(nn.Module):
    def __init__(self, arm: nn.Module):
        super().__init__(); self.arm=arm; self.local=nn.Linear(512,4); self.recon=nn.Linear(512,384)
    def forward(self,l2,final):
        z=self.arm(l2,final); return self.local(z),self.recon(z)


def adapter_params(m): return sum(p.numel() for p in m.arm.parameters() if p.requires_grad)


def build_features(model):
    per_width={}; identity=0.0
    for w in WIDTHS:
        tri,tla=make_canvases(TRAIN_CANVASES,width=w,seed=MASTER_SEED+w*100+1)
        eim,ela=make_canvases(EVAL_CANVASES,width=w,seed=MASTER_SEED+w*100+2)
        tf,ea=extract_all_layers(model,tri); ef,eb=extract_all_layers(model,eim); identity=max(identity,ea,eb)
        per_width[w]={
            "train_l2":tf["L2"].reshape(-1,384).float(),
            "train_final":tf["FINAL_CONTRACT"].reshape(-1,384).float(),
            "train_y":tla.reshape(-1,4).float(),
            "eval_l2":ef["L2"].reshape(-1,384).float(),
            "eval_final":ef["FINAL_CONTRACT"].reshape(-1,384).float(),
            "eval_y":ela.reshape(-1,4).float(),
        }
    return per_width,identity


def train_eval(seed,arm_name,arm_template,data):
    torch.manual_seed(seed); m=CourtModel(copy.deepcopy(arm_template)); opt=torch.optim.AdamW(m.parameters(),lr=LR,betas=(.9,.999),weight_decay=0)
    l2=torch.cat([data[w]["train_l2"] for w in WIDTHS]); fin=torch.cat([data[w]["train_final"] for w in WIDTHS]); y=torch.cat([data[w]["train_y"] for w in WIDTHS])
    for _ in range(STEPS):
        opt.zero_grad(set_to_none=True); logits,pred=m(l2,fin); loss=F.binary_cross_entropy_with_logits(logits,y)+(1-F.cosine_similarity(pred,fin,dim=-1).mean()); loss.backward(); opt.step()
    out=[]; m.eval()
    with torch.no_grad():
        for w in WIDTHS:
            d=data[w]; logits,pred=m(d["eval_l2"],d["eval_final"]); acc=float(((logits>=0)==(d["eval_y"]>=.5)).float().mean()); cos=float(F.cosine_similarity(pred,d["eval_final"],dim=-1).mean())
            out.append({"seed":seed,"arm":arm_name,"width":w,"local_accuracy":acc,"final_cosine":cos})
    return out


def run(source:Path,weights:Path):
    torch.set_num_threads(1); foundation,seal=load_exact_dinov2_s_v2(source,weights,device="cpu"); data,identity=build_features(foundation)
    arms={"CURRENT_FINAL_512":CurrentFinal512(),"EARLY_L2_512":EarlyL2512(),"DUAL_L2_FINAL_MATCHED_512":DualMatched512()}
    counts={k:adapter_params(CourtModel(copy.deepcopy(v))) for k,v in arms.items()}
    rows=[]
    for seed in SEEDS:
        for name,arm in arms.items(): rows+=train_eval(seed,name,arm,data)
    def vals(a,key): return [r[key] for r in rows if r["arm"]==a]
    S={a:{"mean_local":sum(vals(a,"local_accuracy"))/len(vals(a,"local_accuracy")),"min_local":min(vals(a,"local_accuracy")),"mean_final_cosine":sum(vals(a,"final_cosine"))/len(vals(a,"final_cosine")),"min_final_cosine":min(vals(a,"final_cosine"))} for a in arms}
    current=S["CURRENT_FINAL_512"]; dual=S["DUAL_L2_FINAL_MATCHED_512"]
    seed_gain=[]
    for s in SEEDS:
        ca=sum(r["local_accuracy"] for r in rows if r["arm"]=="CURRENT_FINAL_512" and r["seed"]==s)/len(WIDTHS)
        da=sum(r["local_accuracy"] for r in rows if r["arm"]=="DUAL_L2_FINAL_MATCHED_512" and r["seed"]==s)/len(WIDTHS)
        seed_gain.append(da-ca)
    validity=identity<=1e-6 and all(v==EXPECTED_ADAPTER_PARAMS for v in counts.values()) and current["mean_final_cosine"]>=.98
    local_ok=dual["min_local"]>=.98 and dual["mean_local"]>=.995
    retention_ok=dual["min_final_cosine"]>=.98 and dual["mean_final_cosine"]>=current["mean_final_cosine"]-.01
    gain_ok=sum(g>=.05 for g in seed_gain)>=4
    if not validity: status="DUAL_LEVEL_VIEW_TOKEN_COURT_INVALID"
    elif local_ok and retention_ok and gain_ok: status="DUAL_LEVEL_VIEW_TOKEN_SUPPORTED_TINY_V1"
    elif retention_ok and not local_ok: status="DUAL_LEVEL_RETENTION_OK_LOCAL_GAIN_INCONCLUSIVE_TINY_V1"
    elif local_ok and not retention_ok: status="DUAL_LEVEL_LOCAL_GAIN_FINAL_RETENTION_FAIL_TINY_V1"
    else: status="DUAL_LEVEL_VIEW_TOKEN_NOT_SUPPORTED_TINY_V1"
    return {"schema":SCHEMA,"status":status,"authority":{"source_head":seal.source_head,"weight_sha256":seal.weight_sha256},"identity_error":identity,"adapter_parameter_counts":counts,"results":rows,"summary":S,"paired_dual_minus_current_local_gain_by_seed":seed_gain,"gates":{"validity":validity,"dual_local":local_ok,"dual_final_retention":retention_ok,"paired_gain":gain_ok},"execution":{"device":"cpu","gpu_required":False,"foundation_frozen":True},"claims":{"product_authority_changed":False,"truth139_training_authorized":False,"fit8_training_authorized":False,"dinov3_adoption_authorized":False}}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--dinov2-source",type=Path,required=True); p.add_argument("--dinov2-weights",type=Path,required=True); p.add_argument("--out",type=Path,required=True); a=p.parse_args(); r=run(a.dinov2_source,a.dinov2_weights); a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n"); print(json.dumps({"status":r["status"],"summary":r["summary"],"adapter_parameter_counts":r["adapter_parameter_counts"],"out":str(a.out)},sort_keys=True))

if __name__=="__main__": main()
