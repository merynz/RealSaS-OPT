#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, random
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

SCHEMA='RealSaS.OracleZSurfaceCalibrationCourt.v3'
SEEDS=(731,947,1217,1601,1999); CAL_SEEDS=(101,211,307); LRS=(3e-3,1e-3,3e-4)
H_REF=2.0/24.0; FACTORS=(4.0,2.0,1.0,0.5); OUT=4
CAL_STEPS=1400; STEPS=2200; TRAIN_POOL=4096; EVAL_POINTS=2048; BATCH=32; HIDDEN=96

def seed_all(s): random.seed(s); torch.manual_seed(s)
def gen(s): return torch.Generator().manual_seed(s)

def make_scene(seed,rep,family,radius):
    g=gen(seed); k=4
    centers=torch.empty(k,3).uniform_(-.42,.42,generator=g); attrs=torch.empty(k,3).uniform_(-.85,.85,generator=g)
    scales=torch.empty(k,3).uniform_(.16,.34,generator=g)
    if family=='SURFACE_STRUCTURED': scales[0].fill_(float(radius))
    elif family=='THIN_SURFACE':
        for i in range(k): scales[i,i%3]=float(radius)
    else: raise ValueError(family)
    scales.clamp_(min=.025,max=.36)
    return {'seed':seed,'rep':rep,'family':family,'radius':float(radius),'centers':centers,'attrs':attrs,'scales':scales}

def targets(scene,p):
    q=(p[:,None,:]-scene['centers'][None])/scene['scales'][None]
    d=torch.linalg.vector_norm(q,dim=-1)-1.0
    w=torch.exp(-12.0*d.square())
    y0=w.sum(1,keepdim=True)/4.0
    ya=torch.einsum('nk,kc->nc',w,scene['attrs'])/4.0
    return torch.cat((y0,ya),-1)

def sample(scene,n,g):
    nu=n//2; uni=torch.empty(nu,3).uniform_(-.95,.95,generator=g); ns=n-nu
    comp=torch.randint(0,4,(ns,),generator=g); direction=F.normalize(torch.randn(ns,3,generator=g),dim=-1); radial=1.0+.10*torch.randn(ns,1,generator=g)
    near=scene['centers'][comp]+direction*scene['scales'][comp]*radial; near=near.clamp(-.98,.98)
    return torch.cat((uni,near),0)

def scenes_for(seeds):
    out=[]
    for s in seeds:
        for j,f in enumerate(FACTORS):
            r=f*H_REF; out.append(make_scene(s*10000+100+j,s,'SURFACE_STRUCTURED',r)); out.append(make_scene(s*10000+200+j,s,'THIN_SURFACE',r))
    return out

def cal_scenes():
    out=[]; r=4*H_REF
    for s in CAL_SEEDS: out.append(make_scene(880000+s,s,'SURFACE_STRUCTURED',r)); out.append(make_scene(890000+s,s,'THIN_SURFACE',r))
    return out

def bank(scenes,tp=TRAIN_POOL,ep=EVAL_POINTS):
    ps=[];ys=[];eps=[];eys=[];masks=[]
    for i,s in enumerate(scenes):
        p=sample(s,tp,gen(700000+s['seed']*13+i)); y=targets(s,p); q=sample(s,ep,gen(900000+s['seed']*17+i)); ey=targets(s,q)
        ps.append(p);ys.append(y);eps.append(q);eys.append(ey);masks.append(ey[:,0]>=.05)
    P=torch.stack(ps);Y=torch.stack(ys);EP=torch.stack(eps);EY=torch.stack(eys); scale=Y.std(1).clamp_min(.05)
    return {'scenes':scenes,'p':P,'y':Y,'scale':scale,'ep':EP,'ey':EY,'mask':torch.stack(masks)}

def gs2(planes,p):
    x,y,z=p.unbind(-1); grids=(torch.stack((x,y),-1),torch.stack((x,z),-1),torch.stack((y,z),-1)); out=[]
    for i,g in enumerate(grids): out.append(F.grid_sample(planes[:,i],g.unsqueeze(2),mode='bilinear',padding_mode='border',align_corners=False).squeeze(-1).transpose(1,2))
    return torch.cat(out,-1)
def gs3(grid,p): return F.grid_sample(grid,p.unsqueeze(2).unsqueeze(2),mode='bilinear',padding_mode='border',align_corners=False).squeeze(-1).squeeze(-1).transpose(1,2)

class PerSceneMLP(nn.Module):
    def __init__(self,S,inp,h=HIDDEN):
        super().__init__(); self.w1=nn.Parameter(torch.empty(S,h,inp));self.b1=nn.Parameter(torch.zeros(S,h));self.w2=nn.Parameter(torch.empty(S,h,h));self.b2=nn.Parameter(torch.zeros(S,h));self.w3=nn.Parameter(torch.empty(S,OUT,h));self.b3=nn.Parameter(torch.zeros(S,OUT))
        for s in range(S): nn.init.kaiming_uniform_(self.w1[s],a=math.sqrt(5));nn.init.kaiming_uniform_(self.w2[s],a=math.sqrt(5));nn.init.kaiming_uniform_(self.w3[s],a=math.sqrt(5))
    def forward(self,x):
        h=F.silu(torch.einsum('sni,shi->snh',x,self.w1)+self.b1[:,None]);h=F.silu(torch.einsum('sni,shi->snh',h,self.w2)+self.b2[:,None]);return torch.einsum('snh,soh->sno',h,self.w3)+self.b3[:,None]

class Model(nn.Module):
    def __init__(self,arm,S):
        super().__init__();self.arm=arm
        if arm=='DIRECT_XYZ_PER_SCENE': self.q=PerSceneMLP(S,3)
        elif arm=='TRIPLANE_PER_SCENE': self.c=nn.Parameter(torch.randn(S,3,16,24,24)*.02);self.q=PerSceneMLP(S,48)
        elif arm=='DENSE3D_PER_SCENE': self.c=nn.Parameter(torch.randn(S,16,12,12,12)*.02);self.q=PerSceneMLP(S,16)
        else: raise ValueError(arm)
    def forward(self,p):
        if self.arm=='DIRECT_XYZ_PER_SCENE': f=p
        elif self.arm=='TRIPLANE_PER_SCENE': f=gs2(self.c,p)
        else:f=gs3(self.c,p)
        return self.q(f)

def sample_batch(B,g):
    S,N,_=B['p'].shape; idx=torch.randint(0,N,(S,BATCH),generator=g); return torch.gather(B['p'],1,idx[...,None].expand(-1,-1,3)),torch.gather(B['y'],1,idx[...,None].expand(-1,-1,OUT))
def train(arm,B,lr,steps,seed):
    seed_all(seed);m=Model(arm,len(B['scenes']));opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=0);g=gen(seed*7919+17);scale=B['scale'][:,None]
    for _ in range(steps):
        p,y=sample_batch(B,g);z=m(p);loss=(((z-y)/scale)**2).mean();opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    return m

def nrmse(pred,truth,scale): return (torch.sqrt(((pred-truth)**2).mean(1))/scale).mean(-1)
def summary(m,B):
    m.eval();rows=[]
    with torch.no_grad():
        z=m(B['ep']); vals=nrmse(z,B['ey'],B['scale'])
        for i,s in enumerate(B['scenes']):
            mask=B['mask'][i]; lv=float(nrmse(z[i][mask][None],B['ey'][i][mask][None],B['scale'][i][None])[0]) if int(mask.sum())>=16 else None
            rows.append({'seed':s['rep'],'family':s['family'],'radius':s['radius'],'nrmse':float(vals[i]),'local_nrmse':lv,'local_count':int(mask.sum())})
    groups={}
    for fam,r in sorted({(x['family'],x['radius']) for x in rows}):
        q=[x for x in rows if x['family']==fam and x['radius']==r]; k=f'{fam}:{r:g}'; groups[k]={'family':fam,'radius':r,'mean_nrmse':sum(x['nrmse'] for x in q)/len(q),'worst_nrmse':max(x['nrmse'] for x in q),'mean_local_nrmse':sum(x['local_nrmse'] for x in q)/len(q),'worst_local_nrmse':max(x['local_nrmse'] for x in q),'per_seed':[x['nrmse'] for x in q]}
    return {'rows':rows,'groups':groups}
def calibrate():
    B=bank(cal_scenes());rows=[]
    for lr in LRS:
        S=summary(train('DIRECT_XYZ_PER_SCENE',B,lr,CAL_STEPS,424242),B);v=[x['nrmse'] for x in S['rows']];rows.append({'lr':lr,'mean_nrmse':sum(v)/len(v),'worst_nrmse':max(v)})
    win=min(rows,key=lambda x:(round(x['mean_nrmse'],12),x['lr']));return {'rows':rows,'winner':win,'selected_lr':win['lr'],'valid':win['mean_nrmse']<=.10 and win['worst_nrmse']<=.14}
def firewall(lr):
    ss=[make_scene(990001,1,'SURFACE_STRUCTURED',4*H_REF),make_scene(990002,2,'THIN_SURFACE',4*H_REF)];B=bank(ss,1024,1024);g1=gen(123);g2=gen(456);B['y']=torch.randn(B['y'].shape,generator=g1);B['ey']=torch.randn(B['ey'].shape,generator=g2);B['scale']=B['y'].std(1).clamp_min(.05);m=train('DIRECT_XYZ_PER_SCENE',B,lr,1200,777);S=summary(m,B);v=[x['nrmse'] for x in S['rows']];mean=sum(v)/len(v);return {'mean_nrmse':mean,'worst_nrmse':max(v),'valid':mean>=.90}
def interpret(results):
    D=results['DIRECT_XYZ_PER_SCENE']['groups'];T=results['TRIPLANE_PER_SCENE']['groups'];G=results['DENSE3D_PER_SCENE']['groups'];points={};concerns=[]
    for fam in ('SURFACE_STRUCTURED','THIN_SURFACE'):
      ordered=[]
      for f in FACTORS:
        r=f*H_REF;k=f'{fam}:{r:g}'; d,t,g=D[k],T[k],G[k]
        valid=d['mean_nrmse']<=.12 and d['worst_nrmse']<=.18 and d['mean_local_nrmse']<=.18
        tp=valid and t['mean_nrmse']<=.12 and t['worst_nrmse']<=.18 and t['mean_local_nrmse']<=.18
        gp=valid and g['mean_nrmse']<=.12 and g['worst_nrmse']<=.18 and g['mean_local_nrmse']<=.18
        concern=valid and gp and not tp; improvement=(t['mean_nrmse']-g['mean_nrmse'])/max(t['mean_nrmse'],1e-9)
        points[k]={'direct_valid':valid,'triplane_supported':tp,'dense_supported':gp,'credible_triplane_concern':concern,'dense_relative_mean_gain':improvement};ordered.append((r,concern,improvement))
      for a,b in zip(ordered,ordered[1:]):
        if a[1] and b[1] and a[2]>=.25 and b[2]>=.25: concerns.append((fam,a[0],b[0]))
    return {'points':points,'dense_selected':bool(concerns),'adjacent_material_concerns':concerns,'provisional_carrier':'DENSE3D' if concerns else 'TRIPLANE'}
def run():
    torch.set_num_threads(1);cal=calibrate()
    if not cal['valid']: return {'schema':SCHEMA,'status':'APPARATUS_INCONCLUSIVE__SURFACE_CALIBRATION_FAIL','calibration':cal}
    fw=firewall(float(cal['selected_lr']))
    if not fw['valid']: return {'schema':SCHEMA,'status':'APPARATUS_INCONCLUSIVE__FIREWALL_FAIL','calibration':cal,'firewall':fw}
    B=bank(scenes_for(SEEDS));R={}
    for i,a in enumerate(('DIRECT_XYZ_PER_SCENE','TRIPLANE_PER_SCENE','DENSE3D_PER_SCENE')): R[a]=summary(train(a,B,float(cal['selected_lr']),STEPS,555000+i*101),B)
    I=interpret(R);st='DENSE3D_SELECTED_TINY_V3' if I['dense_selected'] else 'TRIPLANE_RETAINED_PROVISIONAL_TINY_V3'
    return {'schema':SCHEMA,'status':st,'calibration':cal,'firewall':fw,'interpretation':I,'results':R,'configuration':{'seeds':list(SEEDS),'factors':list(FACTORS),'h_ref':H_REF,'steps':STEPS,'train_pool':TRAIN_POOL,'eval_points':EVAL_POINTS,'batch':BATCH},'authority':{'product_authority_changed':False,'truth139_training_authorized':False,'fit8_training_authorized':False}}
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=run();a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':r['status'],'calibration':r.get('calibration'),'firewall':r.get('firewall'),'interpretation':r.get('interpretation'),'out':str(a.out)},sort_keys=True))
if __name__=='__main__':main()
