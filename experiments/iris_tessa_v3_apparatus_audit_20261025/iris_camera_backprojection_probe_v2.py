#!/usr/bin/env python3
from __future__ import annotations
import argparse, copy, json, math, statistics
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

SCHEMA="RealSaS.IRISCameraBackprojectionCourt.v2"
SEEDS=(731,947,1217,1597,2017)
TRAIN_YAWS=tuple(float(x) for x in range(0,360,30))
EVAL_YAWS=tuple(float(x) for x in range(15,360,30))
TRAIN_HALFS=(0.70,0.90,1.10,1.30)
EVAL_HALFS=(0.80,1.00,1.20)
STEPS=800; BATCH=1024; EVAL=32768; LR=1e-3; D=32


def lattice():
    a=torch.linspace(-.45,.45,4); yy,xx=torch.meshgrid(a,a,indexing='ij'); return torch.stack((xx,yy),-1).reshape(16,2)

def basis(yaw):
    t=torch.deg2rad(yaw); return torch.stack((torch.cos(t),torch.sin(t)),-1),torch.stack((-torch.sin(t),torch.cos(t)),-1)

def batch(n,g,train=True,random_label=False):
    ys=TRAIN_YAWS if train else EVAL_YAWS; hs=TRAIN_HALFS if train else EVAL_HALFS
    label=torch.randint(0,16,(n,),generator=g); xy=lattice()[label]
    yi=torch.randint(0,len(ys),(n,),generator=g); yaw=torch.tensor(ys)[yi]
    hi=torch.randint(0,len(hs),(n,),generator=g); half=torch.tensor(hs)[hi]
    r,uvec=basis(yaw)
    pu=(xy*r).sum(-1)/half; pv=(xy*uvec).sum(-1)/half
    uv=torch.stack((pu,pv),-1); cam=torch.cat((r,uvec,half[:,None]),-1)
    back=pu[:,None]*half[:,None]*r + pv[:,None]*half[:,None]*uvec
    target=torch.randint(0,16,(n,),generator=g) if random_label else label
    return cam,uv,back,xy,target

class Head(nn.Module):
    def __init__(self): super().__init__(); self.head=nn.Sequential(nn.LayerNorm(D),nn.Linear(D,64),nn.GELU(),nn.Linear(64,16))

class Current(Head):
    def __init__(self):
        super().__init__(); self.cam=nn.Sequential(nn.Linear(5,D),nn.GELU(),nn.Linear(D,D)); self.uv=nn.Sequential(nn.Linear(2,D),nn.GELU(),nn.Linear(D,D))
    def forward(self,cam,uv,back,xy): del back,xy; return self.head(self.cam(cam)+self.uv(uv))

class Joint(Head):
    def __init__(self): super().__init__(); self.enc=nn.Sequential(nn.Linear(7,D),nn.GELU(),nn.Linear(D,D))
    def forward(self,cam,uv,back,xy): del back,xy; return self.head(self.enc(torch.cat((cam,uv),-1)))

class Coord(Head):
    def __init__(self,use_true=False): super().__init__(); self.use_true=use_true; self.enc=nn.Sequential(nn.Linear(2,D),nn.GELU(),nn.Linear(D,D))
    def forward(self,cam,uv,back,xy): del cam,uv; p=xy if self.use_true else back; return self.head(self.enc(p))

def np(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def train(seed,name,tpl,random_label=False):
    torch.manual_seed(seed); m=copy.deepcopy(tpl); opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=0.0); g=torch.Generator().manual_seed(seed+100000)
    for _ in range(STEPS):
        b=batch(BATCH,g,True,random_label); opt.zero_grad(set_to_none=True); z=m(*b[:-1]); loss=F.cross_entropy(z,b[-1]); loss.backward(); opt.step()
    m.eval(); g=torch.Generator().manual_seed(seed+200000); cor=tot=0; seen=0
    with torch.no_grad():
        while seen<EVAL:
            k=min(BATCH,EVAL-seen); b=batch(k,g,False,random_label); z=m(*b[:-1]); cor+=int((z.argmax(-1)==b[-1]).sum()); tot+=k; seen+=k
    return {'seed':seed,'arm':name,'accuracy':cor/tot}

def summ(rows,a):
    v=[r['accuracy'] for r in rows if r['arm']==a]; return {'mean':sum(v)/len(v),'min':min(v),'std':statistics.pstdev(v),'per_seed':v}

def run():
    torch.set_num_threads(1); torch.manual_seed(1); cur=Current(); torch.manual_seed(1); joint=Joint(); torch.manual_seed(1); det=Coord(False); torch.manual_seed(1); pos=Coord(True)
    counts={'CURRENT_FACTORIZED_ADD':np(cur),'JOINT_RAW_CAMERA_UV':np(joint),'DETERMINISTIC_BACKPROJECTED':np(det),'TRUE_CANONICAL_POSITIVE_CONTROL':np(pos)}
    rows=[]
    for s in SEEDS:
        rows += [train(s,'CURRENT_FACTORIZED_ADD',cur),train(s,'JOINT_RAW_CAMERA_UV',joint),train(s,'DETERMINISTIC_BACKPROJECTED',det),train(s,'TRUE_CANONICAL_POSITIVE_CONTROL',pos),train(s,'RANDOM_LABEL_FIREWALL',cur,True)]
    S={a:summ(rows,a) for a in ('CURRENT_FACTORIZED_ADD','JOINT_RAW_CAMERA_UV','DETERMINISTIC_BACKPROJECTED','TRUE_CANONICAL_POSITIVE_CONTROL','RANDOM_LABEL_FIREWALL')}
    # Exact deterministic geometry check on a disjoint deterministic batch.
    g=torch.Generator().manual_seed(888888); cam,uv,back,xy,_=batch(65536,g,False); recon=float((back-xy).abs().max())
    by={(r['arm'],r['seed']):r for r in rows}; gains=[by[('DETERMINISTIC_BACKPROJECTED',s)]['accuracy']-by[('CURRENT_FACTORIZED_ADD',s)]['accuracy'] for s in SEEDS]
    pos_ok=S['TRUE_CANONICAL_POSITIVE_CONTROL']['mean']>=.995 and S['TRUE_CANONICAL_POSITIVE_CONTROL']['min']>=.99
    fw_ok=S['RANDOM_LABEL_FIREWALL']['mean']<=.09
    recon_ok=recon<=1e-5
    cur_ok=S['CURRENT_FACTORIZED_ADD']['mean']>=.98 and S['CURRENT_FACTORIZED_ADD']['min']>=.95
    det_ok=S['DETERMINISTIC_BACKPROJECTED']['mean']>=.995 and S['DETERMINISTIC_BACKPROJECTED']['min']>=.99
    det_sel=det_ok and counts['DETERMINISTIC_BACKPROJECTED']<=counts['CURRENT_FACTORIZED_ADD'] and sum(gains)/5>=.05 and sum(x>0 for x in gains)>=4
    joint_ok=S['JOINT_RAW_CAMERA_UV']['mean']>=.98 and S['JOINT_RAW_CAMERA_UV']['min']>=.95
    if not(pos_ok and fw_ok and recon_ok): st='CAMERA_GEOMETRY_V2_INVALID'
    elif cur_ok: st='CURRENT_FACTORIZED_CAMERA_GEOMETRY_SUFFICIENT_TINY_V2'
    elif det_sel: st='DETERMINISTIC_BACKPROJECTION_SELECTED_TINY_V2'
    elif joint_ok and not cur_ok: st='ADDITIVE_FACTORIZATION_BOTTLENECK__DETERMINISTIC_NOT_SELECTED_V2'
    else: st='CAMERA_GEOMETRY_INCONCLUSIVE_TINY_V2'
    return {'schema':SCHEMA,'status':st,'parameter_counts':counts,'summary':S,'paired_deterministic_minus_current':gains,'max_backprojection_abs_error':recon,'gates':{'positive_control':pos_ok,'firewall':fw_ok,'exact_backprojection':recon_ok,'current_sufficient':cur_ok,'joint_raw_sufficient':joint_ok,'deterministic_sufficient':det_ok,'deterministic_selected':det_sel},'execution':{'steps':STEPS,'batch':BATCH,'eval_samples':EVAL,'lr':LR,'device':'cpu'},'authority':{'product_authority_changed':False,'truth139_training_authorized':False,'fit8_training_authorized':False}}

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=run();a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':r['status'],'summary':r['summary'],'gates':r['gates'],'out':str(a.out)},sort_keys=True))
if __name__=='__main__': main()
