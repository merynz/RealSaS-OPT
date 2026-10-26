#!/usr/bin/env python3
from __future__ import annotations
import argparse, copy, json, math, statistics
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

SCHEMA='RealSaS.SharedScenePremiseStageA.v1'; SEEDS=(731,947,1217,1597,2017); BATCH=512; JOINT_STEPS=1400; SINGLE_STEPS=900; EVAL=16384; LR=1e-3
ARMS=('S0_INDEPENDENT','S1_LATE_FUSION','S2_SHARED_TRUNK_PRIVATE','S3_FULL_SHARED_Z','S4_SHARED_Z_FAMILY_ADAPTERS')

def factors(n,g,shift=False):
    c=torch.empty(n,4).uniform_(-1,1,generator=g); gm=torch.empty(n,4).uniform_(-1,1,generator=g); gp=torch.empty(n,4).uniform_(-1,1,generator=g); noise=.18*torch.randn(n,4,generator=g)
    mix=torch.stack((c[:,0],gm[:,0]-gp[:,0],c[:,3]+.5*(gm[:,1]+gp[:,1]),c[:,1]*c[:,2]),-1)
    nuisance=(-mix if shift else mix)+noise if shift else mix+noise
    x=torch.cat((c,gm,gp,nuisance),-1)
    ygm=torch.stack((torch.sin(math.pi*c[:,0])+.5*gm[:,0],c[:,1]*c[:,2]+.5*torch.tanh(gm[:,1]),torch.tanh(c[:,3]+.7*gm[:,2]),.5*torch.cos(math.pi*c[:,0]*c[:,1])+.5*gm[:,3]),-1)
    ygp=torch.stack((torch.cos(math.pi*c[:,0])+.5*gp[:,0],c[:,1]-c[:,2]+.5*torch.tanh(gp[:,1]),torch.tanh(-c[:,3]+.7*gp[:,2]),.5*torch.sin(math.pi*c[:,0]*c[:,2])+.5*gp[:,3]),-1)
    return x,ygm,ygp

class Encoder(nn.Module):
    def __init__(self,inp,h,out): super().__init__(); self.net=nn.Sequential(nn.Linear(inp,h),nn.GELU(),nn.Linear(h,out),nn.GELU())
    def forward(self,x): return self.net(x)
class Head(nn.Module):
    def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Linear(16,32),nn.GELU(),nn.Linear(32,4))
    def forward(self,x): return self.net(x)

class S0(nn.Module):
    def __init__(self): super().__init__();self.g=Encoder(16,48,16);self.p=Encoder(16,48,16);self.hg=Head();self.hp=Head()
    def forward(self,x,mode=None):
        g,p=self.g(x),self.p(x)
        if mode=='zero_primary': g=torch.zeros_like(g);p=torch.zeros_like(p)
        if mode=='cross_shuffle': g=g.roll(1,0);p=p.roll(-1,0)
        return self.hg(g),self.hp(p)
    def shared_params(self): return []
class S1(nn.Module):
    def __init__(self): super().__init__();self.g=Encoder(16,53,8);self.p=Encoder(16,53,8);self.fg=nn.Linear(16,16);self.fp=nn.Linear(16,16);self.hg=Head();self.hp=Head()
    def forward(self,x,mode=None):
        g,p=self.g(x),self.p(x); pg=p.roll(1,0) if mode=='cross_shuffle' else p; gp=g.roll(-1,0) if mode=='cross_shuffle' else g
        zg=self.fg(torch.cat((g,pg),-1));zp=self.fp(torch.cat((gp,p),-1))
        if mode=='zero_primary': zg=torch.zeros_like(zg);zp=torch.zeros_like(zp)
        return self.hg(F.gelu(zg)),self.hp(F.gelu(zp))
    def shared_params(self): return []
class S2(nn.Module):
    def __init__(self): super().__init__();self.s=Encoder(16,48,8);self.g=Encoder(16,40,8);self.p=Encoder(16,40,8);self.hg=Head();self.hp=Head()
    def forward(self,x,mode=None):
        s,g,p=self.s(x),self.g(x),self.p(x)
        if mode=='zero_primary': s=torch.zeros_like(s)
        if mode=='cross_shuffle': g=g.roll(1,0);p=p.roll(-1,0)
        return self.hg(torch.cat((s,g),-1)),self.hp(torch.cat((s,p),-1))
    def shared_params(self): return list(self.s.parameters())
class S3(nn.Module):
    def __init__(self): super().__init__();self.s=Encoder(16,96,16);self.hg=Head();self.hp=Head()
    def forward(self,x,mode=None):
        s=self.s(x)
        if mode=='zero_primary': s=torch.zeros_like(s)
        if mode=='cross_shuffle': s=s.roll(1,0)
        return self.hg(s),self.hp(s)
    def shared_params(self): return list(self.s.parameters())
class S4(nn.Module):
    def __init__(self): super().__init__();self.s=Encoder(16,80,16);self.ag=nn.Linear(16,16);self.ap=nn.Linear(16,16);self.hg=Head();self.hp=Head()
    def forward(self,x,mode=None):
        s=self.s(x)
        if mode=='zero_primary': s=torch.zeros_like(s)
        rg=F.gelu(self.ag(s));rp=F.gelu(self.ap(s))
        if mode=='cross_shuffle': rg=rg.roll(1,0);rp=rp.roll(-1,0)
        return self.hg(s+rg),self.hp(s+rp)
    def shared_params(self): return list(self.s.parameters())

def make(a): return {'S0_INDEPENDENT':S0,'S1_LATE_FUSION':S1,'S2_SHARED_TRUNK_PRIVATE':S2,'S3_FULL_SHARED_Z':S3,'S4_SHARED_Z_FAMILY_ADAPTERS':S4}[a]()
def np(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
def scales(seed):
    x,g,p=factors(32768,torch.Generator().manual_seed(seed+9090));return g.var(0,unbiased=False).clamp_min(.05),p.var(0,unbiased=False).clamp_min(.05)
def lossfam(z,y,v): return (((z-y)**2)/v).mean()
def train(seed,arm,which='joint'):
    torch.manual_seed(seed);m=make(arm);opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=0);G=torch.Generator().manual_seed(seed+100000);vg,vp=scales(seed);steps=JOINT_STEPS if which=='joint' else SINGLE_STEPS
    for _ in range(steps):
        x,yg,yp=factors(BATCH,G);zg,zp=m(x);lg=lossfam(zg,yg,vg);lp=lossfam(zp,yp,vp);loss=lg+lp if which=='joint' else (lg if which=='gm' else lp);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    return m,vg,vp
def evalm(m,vg,vp,seed,shift=False,mode=None):
    G=torch.Generator().manual_seed(seed+(400000 if shift else 200000)); remain=EVAL;sg=sp=0.;n=0
    with torch.no_grad():
      while remain:
        k=min(BATCH,remain);x,yg,yp=factors(k,G,shift);zg,zp=m(x,mode);sg+=float((((zg-yg)**2)/vg).sum());sp+=float((((zp-yp)**2)/vp).sum());n+=k*4;remain-=k
    return sg/n,sp/n
def gradcos(m,vg,vp,seed):
    ps=m.shared_params()
    if not ps:return None
    x,yg,yp=factors(BATCH,torch.Generator().manual_seed(seed+700000));zg,zp=m(x);lg=lossfam(zg,yg,vg);lp=lossfam(zp,yp,vp);gg=torch.autograd.grad(lg,ps,retain_graph=True,allow_unused=True);gp=torch.autograd.grad(lp,ps,allow_unused=True);a=torch.cat([q.reshape(-1) for q in gg if q is not None]);b=torch.cat([q.reshape(-1) for q in gp if q is not None]);return float(F.cosine_similarity(a,b,dim=0))
def firewall():
    torch.manual_seed(999);m=S0();opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=0);G=torch.Generator().manual_seed(12345)
    for _ in range(700):
        x,_,_=factors(BATCH,G);yg=torch.randn(BATCH,4,generator=G);yp=torch.randn(BATCH,4,generator=G);zg,zp=m(x);loss=F.mse_loss(zg,yg)+F.mse_loss(zp,yp);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    G=torch.Generator().manual_seed(54321);x,_,_=factors(EVAL,G);yg=torch.randn(EVAL,4,generator=G);yp=torch.randn(EVAL,4,generator=G)
    with torch.no_grad():zg,zp=m(x);e=.5*(float(F.mse_loss(zg,yg))+float(F.mse_loss(zp,yp)))
    return {'mean_normalized_error':e,'valid':e>=.90}
def run():
    torch.set_num_threads(1);counts={a:np(make(a)) for a in ARMS};base=counts['S0_INDEPENDENT'];budget_ok=all(abs(v-base)/base<=.02 for v in counts.values());fw=firewall();rows=[]
    for a in ARMS:
      for s in SEEDS:
        joint,vg,vp=train(s,a,'joint');gm,_,_=train(s,a,'gm');gp,_,_=train(s,a,'gp');ji=evalm(joint,vg,vp,s);js=evalm(joint,vg,vp,s,True);gmo=evalm(gm,vg,vp,s)[0];gpo=evalm(gp,vg,vp,s)[1];zero=evalm(joint,vg,vp,s,False,'zero_primary');shuffle=evalm(joint,vg,vp,s,False,'cross_shuffle');rows.append({'arm':a,'seed':s,'joint_iid_gm':ji[0],'joint_iid_gp':ji[1],'joint_shift_gm':js[0],'joint_shift_gp':js[1],'single_gm':gmo,'single_gp':gpo,'interference_gm':(ji[0]-gmo)/max(gmo,1e-9),'interference_gp':(ji[1]-gpo)/max(gpo,1e-9),'gradient_cosine':gradcos(joint,vg,vp,s),'zero_primary_gm':zero[0],'zero_primary_gp':zero[1],'cross_shuffle_gm':shuffle[0],'cross_shuffle_gp':shuffle[1]})
    S={}
    for a in ARMS:
      q=[r for r in rows if r['arm']==a];mean=lambda k:sum(r[k] for r in q)/len(q);S[a]={'iid_gm':mean('joint_iid_gm'),'iid_gp':mean('joint_iid_gp'),'shift_gm':mean('joint_shift_gm'),'shift_gp':mean('joint_shift_gp'),'interference_gm':mean('interference_gm'),'interference_gp':mean('interference_gp'),'worst_interference':max(mean('interference_gm'),mean('interference_gp')),'gradient_cosine_mean':None if q[0]['gradient_cosine'] is None else mean('gradient_cosine'),'zero_primary_gm':mean('zero_primary_gm'),'zero_primary_gp':mean('zero_primary_gp'),'cross_shuffle_gm':mean('cross_shuffle_gm'),'cross_shuffle_gp':mean('cross_shuffle_gp')}
    s0=S['S0_INDEPENDENT'];valid=budget_ok and fw['valid'] and s0['iid_gm']<=.08 and s0['iid_gp']<=.08
    rejected={}
    for a in ARMS[1:]:
      x=S[a];shift_bad=max((x['shift_gm']-s0['shift_gm'])/max(s0['shift_gm'],1e-9),(x['shift_gp']-s0['shift_gp'])/max(s0['shift_gp'],1e-9))>.20; rejected[a]=x['worst_interference']>.15 or shift_bad
    preferred='S3_FULL_SHARED_Z'
    if valid:
      if rejected.get('S3_FULL_SHARED_Z',True):
        cand=[a for a in ('S2_SHARED_TRUNK_PRIVATE','S4_SHARED_Z_FAMILY_ADAPTERS') if not rejected.get(a,True)]
        if cand: preferred=min(cand,key=lambda a:max(S[a]['iid_gm'],S[a]['iid_gp'],S[a]['shift_gm'],S[a]['shift_gp']))
        elif not rejected.get('S1_LATE_FUSION',True):preferred='S1_LATE_FUSION'
        else:preferred='S0_INDEPENDENT'
      else:
        s3=S['S3_FULL_SHARED_Z'];better=[]
        for a in ('S2_SHARED_TRUNK_PRIVATE','S4_SHARED_Z_FAMILY_ADAPTERS'):
          if rejected.get(a,True):continue
          x=S[a];iid=(max(s3['iid_gm'],s3['iid_gp'])-max(x['iid_gm'],x['iid_gp']))/max(max(s3['iid_gm'],s3['iid_gp']),1e-9);sh=(max(s3['shift_gm'],s3['shift_gp'])-max(x['shift_gm'],x['shift_gp']))/max(max(s3['shift_gm'],s3['shift_gp']),1e-9)
          if iid>=.10 and sh>=.10:better.append(a)
        if better:preferred=min(better,key=lambda a:max(S[a]['iid_gm'],S[a]['iid_gp'],S[a]['shift_gm'],S[a]['shift_gp']))
    status='SHARED_SCENE_STAGE_A_VALID__'+preferred if valid else 'SHARED_SCENE_STAGE_A_INVALID'
    return {'schema':SCHEMA,'status':status,'valid':valid,'parameter_counts':counts,'parameter_budget_valid':budget_ok,'firewall':fw,'summary':S,'rejected_as_default':rejected,'provisional_pretruth_preference':preferred if valid else None,'rows':rows,'downstream':{'tessa':'NOT_AVAILABLE_PRE_TRUTH','compiler':'NOT_AVAILABLE_PRE_TRUTH'},'authority':{'final_shared_scene_selection_authorized':False,'truth_builder_stage_b_required':True,'fit8_training_authorized':False,'product_authority_changed':False}}
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=run();a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':r['status'],'valid':r['valid'],'parameter_counts':r['parameter_counts'],'summary':r['summary'],'rejected':r['rejected_as_default'],'preference':r['provisional_pretruth_preference'],'out':str(a.out)},sort_keys=True))
if __name__=='__main__':main()
