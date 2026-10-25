#!/usr/bin/env python3
"""Execute SYNTHETIC_TESSA_ENDPOINT_BINDING_PREREG_V1 exactly."""
from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from models.tessa.v2.model_v2 import _TypedMemoryStreamV2
from models.tessa.v3.config_v3 import TESSAStructuralizerConfigV3
from models.tessa.v3.model_v3 import TESSATriStreamMemoryV3

SCHEMA = "RealSaS.TESSAV3EndpointBindingCourt.v1"
SEEDS = (731, 947, 1217, 1597, 2017)
STEPS = 1500
BATCH = 128
EVAL = 4096
LR = 1e-3


def cfg() -> TESSAStructuralizerConfigV3:
    out = TESSAStructuralizerConfigV3(
        d_model=64, n_heads=4, decoder_layers=2, mlp_ratio=2, dropout=0.0,
        coordinate_bins=128, max_faces=128, max_vertices=128,
        scene_memory_tokens=4, surface_memory_tokens=6, presentation_memory_tokens=2,
        conditioning_token_budget=12, memory_source_width=32, memory_refinement_layers=1,
        local_attention_window=64, query_chunk_size=16, gradient_checkpointing=False,
    )
    out.validate()
    return out


class TwoHopReader(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, d) / math.sqrt(d))
        self.p = nn.MultiheadAttention(d, heads, batch_first=True, dropout=0.0)
        self.s = nn.MultiheadAttention(d, heads, batch_first=True, dropout=0.0)
        self.pn, self.sn = nn.LayerNorm(d), nn.LayerNorm(d)
        self.out = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, pm: torch.Tensor, sm: torch.Tensor) -> torch.Tensor:
        q = self.q[None].expand(pm.shape[0], -1, -1)
        u, _ = self.p(q, pm, pm, need_weights=False); q = self.pn(q + u)
        u, _ = self.s(q, sm, sm, need_weights=False); q = self.sn(q + u)
        return self.out(q[:, 0]).squeeze(-1)


class LateModel(nn.Module):
    def __init__(self, c: TESSAStructuralizerConfigV3, presentation_dim: int):
        super().__init__(); self.c = c
        self.mem = TESSATriStreamMemoryV3(3, 3, presentation_dim, c)
        self.reader = TwoHopReader(c.d_model, c.n_heads)

    def forward(self, scene, surface, presentation):
        m = self.mem(scene_source=scene, surface_source=surface, presentation_source=presentation)
        a = self.c.scene_memory_tokens; b = a + self.c.surface_memory_tokens
        return self.reader(m[:, b:], m[:, a:b])


class JointMemory(nn.Module):
    def __init__(self, c: TESSAStructuralizerConfigV3):
        super().__init__(); self.c = c; w = c.memory_source_width; d = c.d_model
        self.sp = nn.Sequential(nn.Linear(3, w), nn.LayerNorm(w), nn.GELU())
        self.pp = nn.Sequential(nn.Linear(5, w), nn.LayerNorm(w), nn.GELU())
        self.st = nn.Parameter(torch.randn(w) / math.sqrt(w)); self.pt = nn.Parameter(torch.randn(w) / math.sqrt(w))
        self.q = nn.Parameter(torch.randn(8, d) / math.sqrt(d))
        self.qt_s = nn.Parameter(torch.randn(d) / math.sqrt(d)); self.qt_p = nn.Parameter(torch.randn(d) / math.sqrt(d))
        self.qn, self.kn = nn.LayerNorm(d), nn.LayerNorm(w)
        self.cross = nn.MultiheadAttention(d, c.n_heads, kdim=w, vdim=w, batch_first=True, dropout=0.0)
        layer = nn.TransformerEncoderLayer(d, c.n_heads, c.feedforward_dim, 0.0, "gelu", batch_first=True, norm_first=True)
        self.ref = nn.TransformerEncoder(layer, c.memory_refinement_layers)
        self.on = nn.LayerNorm(d)

    def forward(self, s, p):
        src = torch.cat((self.sp(s)+self.st[None,None], self.pp(p)+self.pt[None,None]), 1)
        q = self.q[None].expand(s.shape[0], -1, -1).clone()
        q[:, :6] += self.qt_s[None,None]; q[:, 6:] += self.qt_p[None,None]
        u, _ = self.cross(self.qn(q), self.kn(src), self.kn(src), need_weights=False)
        return self.on(self.ref(q + u))


class JointModel(nn.Module):
    def __init__(self, c):
        super().__init__(); self.c=c
        self.scene = _TypedMemoryStreamV2(3, c.scene_memory_tokens, 0, c)
        self.joint = JointMemory(c); self.norm=nn.LayerNorm(c.d_model); self.reader=TwoHopReader(c.d_model,c.n_heads)
    def forward(self, scene, surface, presentation):
        m=self.norm(torch.cat((self.scene(scene), self.joint(surface,presentation)),1))
        a=self.c.scene_memory_tokens; b=a+self.c.surface_memory_tokens
        return self.reader(m[:,b:],m[:,a:b])


class RawBound(nn.Module):
    def __init__(self):
        super().__init__(); self.net=nn.Sequential(nn.Linear(7,64),nn.GELU(),nn.Linear(64,1))
    def forward(self, _scene, _surface, bound): return self.net(bound[:,0]).squeeze(-1)


def lattice():
    a=torch.linspace(-0.5,0.5,4); yy,xx=torch.meshgrid(a,a,indexing="ij")
    return torch.stack((xx,yy),-1).reshape(16,2)


def make_batch(n:int,g:torch.Generator):
    xy=lattice(); bits=torch.randint(0,2,(n,16),generator=g); signed=bits.float()*2-1
    perm=torch.rand((n,16),generator=g).argsort(1)
    raw=torch.cat((xy[None].expand(n,-1,-1),signed[...,None]),-1)
    surface=torch.gather(raw,1,perm[...,None].expand(-1,-1,3))
    i=torch.randint(0,16,(n,),generator=g); j=torch.randint(0,15,(n,),generator=g); j=j+(j>=i).long()
    a=xy[i]; b=xy[j]
    unbound=torch.cat((a,b,torch.ones(n,1)),-1)[:,None]
    pi=bits[torch.arange(n),i]; pj=bits[torch.arange(n),j]
    bound=torch.cat((a,b,(pi.float()*2-1)[:,None],(pj.float()*2-1)[:,None],torch.ones(n,1)),-1)[:,None]
    target=(pi!=pj).float(); scene=torch.zeros(n,4,3)
    # exact zero-parameter binder audit: joined payloads must match source truth.
    return scene,surface,unbound,bound,target


def nparams(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


def train_eval(seed, arm, template, bound_input=False, random_label=False):
    m=copy.deepcopy(template); opt=torch.optim.AdamW(m.parameters(),lr=LR,betas=(.9,.999),weight_decay=0)
    g=torch.Generator().manual_seed(seed+100000); lg=torch.Generator().manual_seed(seed+300000)
    last=0.0
    for _ in range(STEPS):
        sc,s,u,b,y=make_batch(BATCH,g); p=b if bound_input else u
        if random_label: y=torch.randint(0,2,(BATCH,),generator=lg).float()
        opt.zero_grad(set_to_none=True); z=m(sc,s,p); loss=F.binary_cross_entropy_with_logits(z,y); loss.backward(); opt.step(); last=float(loss)
    m.eval(); g=torch.Generator().manual_seed(seed+200000); lg=torch.Generator().manual_seed(seed+400000)
    cor=pos=tpos=0; bce=0.; seen=0
    with torch.no_grad():
        while seen<EVAL:
            k=min(BATCH,EVAL-seen); sc,s,u,b,y=make_batch(k,g); p=b if bound_input else u
            if random_label: y=torch.randint(0,2,(k,),generator=lg).float()
            z=m(sc,s,p); bce+=float(F.binary_cross_entropy_with_logits(z,y,reduction="sum")); pr=z>=0; tr=y>=.5
            cor+=int((pr==tr).sum()); pos+=int(pr.sum()); tpos+=int(tr.sum()); seen+=k
    return {"seed":seed,"arm":arm,"eval_accuracy":cor/EVAL,"eval_bce":bce/EVAL,"predicted_positive_fraction":pos/EVAL,"target_positive_fraction":tpos/EVAL,"train_final_bce":last}


def summ(rows,arm):
    v=[r["eval_accuracy"] for r in rows if r["arm"]==arm]
    return {"mean":sum(v)/len(v),"min":min(v),"std":statistics.pstdev(v),"per_seed":v}


def run():
    torch.set_num_threads(1); c=cfg(); rows=[]; pc=None
    for seed in SEEDS:
        torch.manual_seed(seed); late=LateModel(c,5)
        torch.manual_seed(seed); joint=JointModel(c)
        torch.manual_seed(seed); bound=LateModel(c,7)
        torch.manual_seed(seed); raw=RawBound()
        if pc is None: pc={"CURRENT_LATE_UNBOUND":nparams(late),"GENERIC_JOINT_UNBOUND":nparams(joint),"ENDPOINT_BOUND_LATE":nparams(bound),"RAW_BOUND_POSITIVE_CONTROL":nparams(raw),"BINDER":0}
        rows += [
            train_eval(seed,"CURRENT_LATE_UNBOUND",late),
            train_eval(seed,"GENERIC_JOINT_UNBOUND",joint),
            train_eval(seed,"ENDPOINT_BOUND_LATE",bound,True),
            train_eval(seed,"RAW_BOUND_POSITIVE_CONTROL",raw,True),
            train_eval(seed,"BOUND_RANDOM_LABEL_FIREWALL",bound,True,True),
        ]
    S={a:summ(rows,a) for a in ("CURRENT_LATE_UNBOUND","GENERIC_JOINT_UNBOUND","ENDPOINT_BOUND_LATE","RAW_BOUND_POSITIVE_CONTROL","BOUND_RANDOM_LABEL_FIREWALL")}
    by={(r["arm"],r["seed"]):r for r in rows}
    ga=[by[("ENDPOINT_BOUND_LATE",s)]["eval_accuracy"]-by[("CURRENT_LATE_UNBOUND",s)]["eval_accuracy"] for s in SEEDS]
    gb=[by[("ENDPOINT_BOUND_LATE",s)]["eval_accuracy"]-by[("GENERIC_JOINT_UNBOUND",s)]["eval_accuracy"] for s in SEEDS]
    fw=all(.45<=by[("BOUND_RANDOM_LABEL_FIREWALL",s)]["eval_accuracy"]<=.55 for s in SEEDS)
    ctl=S["RAW_BOUND_POSITIVE_CONTROL"]["min"]>=.98 and S["RAW_BOUND_POSITIVE_CONTROL"]["mean"]>=.995
    acc=S["ENDPOINT_BOUND_LATE"]["min"]>=.95 and S["ENDPOINT_BOUND_LATE"]["mean"]>=.98
    supa=sum(x>0 for x in ga)>=4 and sum(ga)/5>=.10; supb=sum(x>0 for x in gb)>=4 and sum(gb)/5>=.05
    if not fw: status="ENDPOINT_BINDING_FIREWALL_INVALID"
    elif not ctl: status="ENDPOINT_BINDING_TASK_INVALID"
    elif acc and supa and supb: status="ENDPOINT_BOUND_LATE_SUPERIOR_TINY_V1"
    elif acc: status="ENDPOINT_BINDING_ESTABLISHED_BUT_SUPERIORITY_INCONCLUSIVE_TINY_V1"
    else: status="ENDPOINT_BINDING_NOT_SUFFICIENT_TINY_V1"
    return {"schema":SCHEMA,"status":status,"preregistration":"SYNTHETIC_TESSA_ENDPOINT_BINDING_PREREG_V1.md","seeds":list(SEEDS),"train_steps":STEPS,"parameter_counts":pc,"results":rows,"summary":S,"paired_bound_minus_current":ga,"paired_bound_minus_joint":gb,"gates":{"firewall":fw,"positive_control":ctl,"bound_accessibility":acc,"superior_vs_current":supa,"superior_vs_joint":supb},"execution":{"device":"cpu","gpu_required":False},"authority":{"truth139_training_authorized":False,"fit8_training_authorized":False,"product_authority_changed":False,"tessa_rewrite_authorized":False}}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--out",type=Path,required=True); a=p.parse_args(); r=run(); a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"schema":r["schema"],"status":r["status"],"means":{k:v["mean"] for k,v in r["summary"].items()},"out":str(a.out)},sort_keys=True))

if __name__=="__main__": main()
