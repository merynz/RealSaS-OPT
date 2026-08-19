from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

import sys
sys.path.insert(0,'/mnt/data/sufficiency')
import current_h_probe as m
import current_h_breadth_fast as bf
sys.path.insert(0,str(m.SRC))
from realsas_iris_sees.targets import build_observation_target_from_sidecar

EPS=1e-8

def unit(x,eps=EPS):
    x=np.asarray(x,np.float64);return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),eps)

def consensus_np(view_values,V):
    w=np.asarray(V,np.float64)[...,None]
    return (np.asarray(view_values,np.float64)*w).sum(0)/np.maximum(w.sum(0),1.)

def per_view_topk_with_scores(desc_view,zA,mask,cache,coarse_k=8,final_k=16):
    coarse,zc=cache['coarse'],cache['zc']
    if len(coarse)==0:return np.empty((0,2),np.float32),np.empty(0,np.float32)
    score=zc@zA; seeds=coarse[np.argsort(-score,kind='stable')[:coarse_k]];refined=[]
    for x,y in seeds:
        for dy in (-4,-2,0,2,4):
            for dx in (-4,-2,0,2,4):
                xx=int(round(float(x+dx)));yy=int(round(float(y+dy)))
                if 0<=xx<mask.shape[1] and 0<=yy<mask.shape[0] and mask[yy,xx]:refined.append((xx,yy))
    if not refined:refined=[tuple(map(int,q)) for q in seeds]
    refined=np.array(sorted(set(refined),key=lambda q:(q[1],q[0])),np.float32)
    sr=m.normalize_rows(m.sample_desc_single_view(desc_view,refined))@zA;ids=np.argsort(-sr,kind='stable')[:final_k]
    return refined[ids],sr[ids].astype(np.float64)

def softmax_entropy(scores):
    s=np.asarray(scores,np.float64)
    if len(s)<=1:return 0.0
    x=s-s.max();p=np.exp(x);p/=p.sum();return float(-(p*np.log(np.maximum(p,1e-12))).sum()/np.log(len(p)))

def H_summary(H,PA):
    H=np.asarray(H,np.float64);PA=np.asarray(PA,np.float64)
    if len(H)==0:return np.zeros(15,np.float64)
    D=H-PA[None];cent=D.mean(0)
    if len(D)>1:
        cov=np.cov(D,rowvar=False);eig=np.sort(np.linalg.eigvalsh(cov))[::-1]
    else:eig=np.zeros(3)
    mag=np.linalg.norm(D,axis=1);q=np.quantile(mag,[.10,.25,.50,.75,.90])
    dirs=D/np.maximum(mag[:,None],EPS);resultant=dirs.mean(0)
    return np.concatenate([[math.log1p(len(H))],cent,eig,q,resultant])

def truth_ids(PA,t):
    pers=np.asarray(t.get('persistent_obs',np.ones(len(t['P_A']),np.uint8))).astype(bool);eligible=np.where(pers)[0]
    if len(eligible)<len(PA):eligible=np.arange(len(t['P_A']))
    rr,cc=linear_sum_assignment(cdist(PA,t['P_A'][eligible]));order=np.argsort(rr);return eligible[cc[order]]

def family_features(root:Path):
    A=sorted((root/'A').glob('*.png'));B=sorted((root/'B').glob('*.png'));assert len(A)==8 and len(B)==8
    x=m.load_model_tensor(A,B)
    with torch.no_grad():out=m.MODEL(x)
    PA,VA,_,_,_,_=m.exact_problem_a_frontdoor(A,B,n=64,res=96)
    xyA=np.stack([m.project_points(PA,v) for v in range(8)],0).astype(np.float32)
    descA=m.sample_field_np(out['descriptor'][0,0],xyA);zA=m.descriptor_consensus(descA,VA)
    NA=unit(consensus_np(m.sample_field_np(out['normal_A'][0,0],xyA),VA))
    cur=consensus_np(m.sample_field_np(out['delta_point_map_srcA'][0],xyA),VA)
    masks=[m.global_foreground_mask(p) for p in B];cache=[m.prepare_b_search(out['descriptor'][0,1,v],masks[v]) for v in range(8)]
    t=build_observation_target_from_sidecar(root/'B'/'observation_sidecar.npz');ids=truth_ids(PA,t)
    targetA=np.asarray(t['P_A'])[ids];targetB=np.asarray(t['P_B'])[ids];tdelta=targetB-targetA;tamp=np.linalg.norm(tdelta,axis=1);tactive=tamp>.005;tdir=tdelta/np.maximum(tamp[:,None],EPS)
    rows=[]
    for i in range(len(PA)):
        cands=[];score_sets=[]
        for v in range(8):
            if not bool(VA[v,i]):cands.append(np.empty((0,2),np.float32));score_sets.append(np.empty(0));continue
            q,s=per_view_topk_with_scores(out['descriptor'][0,1,v],zA[i],masks[v],cache[v]);cands.append(q);score_sets.append(s)
        H=bf.make_H(cands)
        vis_scores=[s for s in score_sets if len(s)]
        margins12=[];spread=[];ents=[]
        for s in vis_scores:
            margins12.append(float(s[0]-s[1]) if len(s)>=2 else 0.0);spread.append(float(s[0]-s[-1]) if len(s)>=2 else 0.0);ents.append(softmax_entropy(s))
        unc=[float(np.median(margins12)) if margins12 else 0.,float(np.median(spread)) if spread else 0.,float(np.median(ents)) if ents else 0.,float(VA[:,i].sum())]
        hs=H_summary(H,PA[i])
        current=cur[i];camp=float(np.linalg.norm(current))
        num=np.concatenate([PA[i].astype(np.float64),NA[i],VA[:,i].astype(np.float64),current,[math.log(EPS+camp)],hs,np.asarray(unc)])
        rows.append({'family':int(root.name),'carrier':i,'z':zA[i].astype(np.float64),'num':num,'target_active':bool(tactive[i]),'target_amp':float(tamp[i]),'target_dir':tdir[i].astype(np.float64),'target_delta':tdelta[i].astype(np.float64),'H_n':int(len(H))})
    return rows

def run(roots):
    rows=[]
    for r in roots:rows.extend(family_features(r))
    Z=np.stack([r['z'] for r in rows]);Z=unit(Z);X=np.stack([r['num'] for r in rows])
    med=np.median(X,axis=0);q25=np.quantile(X,.25,axis=0);q75=np.quantile(X,.75,axis=0);iqr=q75-q25;keep=iqr>1e-9;Xs=(X[:,keep]-med[keep])/iqr[keep]
    dO=np.sqrt(np.mean((Xs[:,None,:]-Xs[None,:,:])**2,axis=2));dZ=1-np.clip(Z@Z.T,-1,1);D=np.sqrt(dZ*dZ+dO*dO)
    fam=np.array([r['family'] for r in rows]);D[fam[:,None]==fam[None,:]]=np.inf
    nn=np.argmin(D,axis=1);nd=D[np.arange(len(rows)),nn]
    pair=[]
    for i,j in enumerate(nn):
        a,b=rows[i],rows[j];active_mismatch=a['target_active']!=b['target_active'];amp_ratio=1.0;dir_cos=None;amp_flag=False;dir_flag=False;logdiff=None
        if a['target_active'] and b['target_active']:
            lo=max(min(a['target_amp'],b['target_amp']),EPS);hi=max(a['target_amp'],b['target_amp']);amp_ratio=float(hi/lo);amp_flag=amp_ratio>3.;dir_cos=float(np.clip(np.dot(a['target_dir'],b['target_dir']),-1,1));dir_flag=dir_cos<.5;logdiff=float(abs(math.log(a['target_amp']+EPS)-math.log(b['target_amp']+EPS)))
        collision=bool(active_mismatch or amp_flag or dir_flag)
        pair.append({'i':i,'j':int(j),'family_i':a['family'],'family_j':b['family'],'carrier_i':a['carrier'],'carrier_j':b['carrier'],'d_repr':float(nd[i]),'active_mismatch':bool(active_mismatch),'amp_ratio':amp_ratio,'amp_flag':bool(amp_flag),'dir_cos':dir_cos,'dir_flag':bool(dir_flag),'log_amp_diff':logdiff,'collision':collision})
    out={'schema':'RealSaS.N1D.PostStageB.RepresentationSufficiency.TestB.CarrierCollision.v1','n_carriers':len(rows),'families':sorted(set(fam.tolist())),'numeric_dimensions_total':int(X.shape[1]),'numeric_dimensions_after_iqr_filter':int(keep.sum()),'pairs':pair,'tails':{}}
    for frac in (.05,.10):
        thr=float(np.quantile(nd,frac));ids=np.where(nd<=thr)[0];pp=[pair[i] for i in ids]
        unions=[p['collision'] for p in pp];mism=[p['active_mismatch'] for p in pp];af=[p['amp_flag'] for p in pp];df=[p['dir_flag'] for p in pp]
        dirs=[p['dir_cos'] for p in pp if p['dir_cos'] is not None];logs=[p['log_amp_diff'] for p in pp if p['log_amp_diff'] is not None]
        collpairs=sorted({tuple(sorted((p['family_i'],p['family_j']))) for p in pp if p['collision']});collfams=sorted({x for p in pp if p['collision'] for x in (p['family_i'],p['family_j'])})
        out['tails'][str(frac)]={'distance_threshold':thr,'count':len(pp),'union_collision_fraction':float(np.mean(unions)) if pp else 0.,'active_mismatch_fraction':float(np.mean(mism)) if pp else 0.,'amp_gt3x_fraction':float(np.mean(af)) if pp else 0.,'direction_cos_lt0_5_fraction':float(np.mean(df)) if pp else 0.,'median_direction_cos_both_active':float(np.median(dirs)) if dirs else None,'median_log_amp_diff_both_active':float(np.median(logs)) if logs else None,'distinct_collision_family_pairs':collpairs,'distinct_collision_families':collfams,'collision_examples':[p for p in pp if p['collision']]}
    t5=out['tails']['0.05'];uf=t5['union_collision_fraction'];paircount=len(t5['distinct_collision_family_pairs'])
    if uf<.10 and paircount==0:ver='GREEN'
    elif uf>=.20 and paircount>=2:ver='RED'
    else:ver='AMBER'
    out['preregistered_decision']=ver
    return out

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('roots',nargs='+',type=Path);ap.add_argument('--out',type=Path);a=ap.parse_args();r=run(a.roots);txt=json.dumps(r,indent=2,allow_nan=False);print(json.dumps({k:v for k,v in r.items() if k!='pairs'},indent=2));
    if a.out:a.out.write_text(txt)
