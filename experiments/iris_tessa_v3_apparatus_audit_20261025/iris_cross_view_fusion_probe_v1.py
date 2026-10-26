#!/usr/bin/env python3
"""Execute SYNTHETIC_IRIS_CROSS_VIEW_FUSION_PREREG_V1 exactly."""
from __future__ import annotations
import argparse, copy, json, math, statistics
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

SCHEMA="RealSaS.IRISCrossViewFusionCourt.v1"; SEEDS=(731,947,1217,1597,2017); STEPS=1200; BATCH=128; EVAL=4096; D=64; HEADS=4

def lattice():
    a=torch.linspace(-.75,.75,4); yy,xx=torch.meshgrid(a,a,indexing="ij"); return torch.stack((xx,yy),-1).reshape(16,2)

def rotate(xy,k):
    x,y=xy[...,0],xy[...,1]
    if k==0:return xy
    if k==1:return torch.stack((-y,x),-1)
    if k==2:return torch.stack((-x,-y),-1)
    return torch.stack((y,-x),-1)

def batch(n,g):
    xy=lattice(); bits=torch.randint(0,2,(n,16,4),generator=g); signed=bits.float()*2-1; toks=[]
    for v in range(4):
        p=rotate(xy,v); th=v*math.pi/2; vc=torch.tensor([math.cos(th),math.sin(th)]).float()[None,None].expand(n,16,-1)
        toks.append(torch.cat((p[None].expand(n,-1,-1),vc,signed[:,:,v,None]),-1))
    src=torch.cat(toks,1); perm=torch.rand((n,64),generator=g).argsort(1); src=torch.gather(src,1,perm[...,None].expand(-1,-1,5))
    return src,bits.float()

class Fusion(nn.Module):
    def __init__(self,early=False):
        super().__init__(); self.src=nn.Sequential(nn.Linear(5,D),nn.LayerNorm(D),nn.GELU()); self.q=nn.Parameter(torch.randn(16,D)/math.sqrt(D)); self.qpos=nn.Sequential(nn.Linear(2,32),nn.GELU(),nn.Linear(32,D))
        if early:
            el=nn.TransformerEncoderLayer(D,HEADS,D*4,0.,"gelu",batch_first=True,norm_first=True); self.enc=nn.TransformerEncoder(el,1); decn=2
        else:self.enc=nn.Identity(); decn=3
        dl=nn.TransformerDecoderLayer(D,HEADS,D*4,0.,batch_first=True,norm_first=True,activation="gelu"); self.dec=nn.TransformerDecoder(dl,decn,norm=nn.LayerNorm(D)); self.out=nn.Linear(D,4)
    def forward(self,s):
        m=self.enc(self.src(s)); q=self.q[None]+self.qpos(lattice())[None]; q=q.expand(s.shape[0],-1,-1); return self.out(self.dec(q,m))

class Raw(nn.Module):
    def __init__(self):super().__init__(); self.out=nn.Linear(4,4)
    def forward(self,s):
        # Deterministic inverse-rotation gather from source tokens.
        n=s.shape[0]; xy=lattice(); vals=torch.empty(n,16,4)
        for v in range(4):
            th=v*math.pi/2; c,sn=math.cos(th),math.sin(th); mask=(s[...,2]-c).abs()+(s[...,3]-sn).abs()<1e-5
            for j,p in enumerate(xy):
                pp=rotate(p[None],v)[0]; d=((s[...,:2]-pp)**2).sum(-1); d=torch.where(mask,d,torch.full_like(d,1e9)); idx=d.argmin(1); vals[:,j,v]=s[torch.arange(n),idx,4]
        return self.out(vals)

def np(m):return sum(p.numel() for p in m.parameters() if p.requires_grad)
def train(seed,name,tpl,random_label=False):
    m=copy.deepcopy(tpl); opt=torch.optim.AdamW(m.parameters(),lr=1e-3,betas=(.9,.999),weight_decay=0); g=torch.Generator().manual_seed(seed+100000); lg=torch.Generator().manual_seed(seed+300000)
    for _ in range(STEPS):
        x,y=batch(BATCH,g); y=torch.randint(0,2,y.shape,generator=lg).float() if random_label else y; opt.zero_grad(set_to_none=True); z=m(x); l=F.binary_cross_entropy_with_logits(z,y); l.backward(); opt.step()
    m.eval(); g=torch.Generator().manual_seed(seed+200000); lg=torch.Generator().manual_seed(seed+400000); cor=tot=0;bce=0.;seen=0
    with torch.no_grad():
        while seen<EVAL:
            k=min(BATCH,EVAL-seen); x,y=batch(k,g); y=torch.randint(0,2,y.shape,generator=lg).float() if random_label else y; z=m(x); bce+=float(F.binary_cross_entropy_with_logits(z,y,reduction="sum")); cor+=int(((z>=0)==(y>=.5)).sum()); tot+=y.numel(); seen+=k
    return {"seed":seed,"arm":name,"accuracy":cor/tot,"bce_per_bit":bce/tot}
def summ(rows,a):
    v=[r["accuracy"] for r in rows if r["arm"]==a]; return {"mean":sum(v)/len(v),"min":min(v),"std":statistics.pstdev(v),"per_seed":v}
def run():
    torch.set_num_threads(1); rows=[]; pc=None
    for s in SEEDS:
        torch.manual_seed(s);a=Fusion(False);torch.manual_seed(s);b=Fusion(True);torch.manual_seed(s);c=Raw()
        if pc is None:pc={"CURRENT_QUERY_MEDIATED":np(a),"SOURCE_SELF_ATTENTION_FIRST":np(b),"RAW_CORRESPONDENCE_POSITIVE_CONTROL":np(c)}
        rows += [train(s,"CURRENT_QUERY_MEDIATED",a),train(s,"SOURCE_SELF_ATTENTION_FIRST",b),train(s,"RAW_CORRESPONDENCE_POSITIVE_CONTROL",c),train(s,"RANDOM_LABEL_FIREWALL",a,True)]
    S={k:summ(rows,k) for k in ("CURRENT_QUERY_MEDIATED","SOURCE_SELF_ATTENTION_FIRST","RAW_CORRESPONDENCE_POSITIVE_CONTROL","RANDOM_LABEL_FIREWALL")}; by={(r["arm"],r["seed"]):r for r in rows}; gains=[by[("SOURCE_SELF_ATTENTION_FIRST",s)]["accuracy"]-by[("CURRENT_QUERY_MEDIATED",s)]["accuracy"] for s in SEEDS]
    fw=all(.48<=v<=.52 for v in S["RANDOM_LABEL_FIREWALL"]["per_seed"]); ctl=S["RAW_CORRESPONDENCE_POSITIVE_CONTROL"]["min"]>=.995; cur=S["CURRENT_QUERY_MEDIATED"]["min"]>=.95 and S["CURRENT_QUERY_MEDIATED"]["mean"]>=.98; early=S["SOURCE_SELF_ATTENTION_FIRST"]["min"]>=.95 and S["SOURCE_SELF_ATTENTION_FIRST"]["mean"]>=.98 and sum(g>0 for g in gains)>=4 and sum(gains)/5>=.10 and pc["SOURCE_SELF_ATTENTION_FIRST"]<=pc["CURRENT_QUERY_MEDIATED"]
    if not fw:st="CROSS_VIEW_FUSION_FIREWALL_INVALID"
    elif not ctl:st="CROSS_VIEW_TASK_INVALID"
    elif cur:st="CURRENT_QUERY_MEDIATED_CORRESPONDENCE_ESTABLISHED_TINY_V1"
    elif early:st="SOURCE_SELF_ATTENTION_SUPPORTED_TINY_V1"
    elif sum(gains)/5>0:st="CROSS_VIEW_FUSION_SIGNAL_INCONCLUSIVE_TINY_V1"
    else:st="CROSS_VIEW_CORRESPONDENCE_NOT_ESTABLISHED_TINY_V1"
    return {"schema":SCHEMA,"status":st,"parameter_counts":pc,"results":rows,"summary":S,"paired_early_minus_current":gains,"gates":{"firewall":fw,"control":ctl,"current_adequate":cur,"early_supported":early},"execution":{"device":"cpu","gpu_required":False},"authority":{"truth139_training_authorized":False,"fit8_training_authorized":False,"product_authority_changed":False}}
def main():
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True);a=p.parse_args();r=run();a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n");print(json.dumps({"status":r["status"],"summary":r["summary"],"parameter_counts":r["parameter_counts"],"out":str(a.out)},sort_keys=True))
if __name__=="__main__":main()
