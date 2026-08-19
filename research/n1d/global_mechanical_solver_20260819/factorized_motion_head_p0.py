from __future__ import annotations
import json,hashlib,sys,math
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,balanced_accuracy_score,brier_score_loss
sys.path.insert(0,'/mnt/data/sufficiency')
import adaptive_h_v1 as a
m=a.m
F=a.FAMILIES;DIR=a.DIRNAME
EPS=1e-8

def stats_visible(x,V):
    x=np.asarray(x,float); V=np.asarray(V,bool); N=x.shape[1];out=[]
    for i in range(N):
        q=x[V[:,i],i]
        if len(q)==0:q=x[:,i]
        out.append(q)
    return out

def scalar_stats(arrs):
    return np.array([[float(np.mean(q)),float(np.std(q)),float(np.max(q))] for q in arrs],float)

def extract(fid):
    root=Path('/mnt/data/sufficiency/currentH')/DIR[fid];A=sorted((root/'A').glob('*.png'));B=sorted((root/'B').glob('*.png'))
    x=m.load_model_tensor(A,B)
    with torch.no_grad():out=m.MODEL(x)
    PA,VA,_,_,_,_=m.exact_problem_a_frontdoor(A,B,n=64,res=96)
    xy=np.stack([m.project_points(PA,v) for v in range(8)],0).astype(np.float32)
    cur=m.sample_field_np(out['delta_point_map_srcA'][0],xy)
    desc=m.sample_field_np(out['descriptor'][0,0],xy)
    dsig=m.sample_field_np(out['descriptor_log_sigma'][0,0],xy)
    lsig=m.sample_field_np(out['log_sigma_A'][0,0],xy)
    vlog=m.sample_field_np(out['visibility_logit_A'][0,0],xy)
    gate=m.sample_field_np(out['transport_gate_srcA'][0],xy)
    ent=m.sample_field_np(out['transport_entropy_srcA'][0],xy)
    peak=m.sample_field_np(out['transport_peak_srcA'][0],xy)
    edge=m.sample_field_np(out['transport_edge_mass_srcA'][0],xy)
    w=VA[...,None].astype(float); cons=(cur*w).sum(0)/np.maximum(w.sum(0),1)
    camp=np.linalg.norm(cons,axis=1); viewamp=np.linalg.norm(cur,axis=2)
    va=stats_visible(viewamp[...,None],VA); vast=np.array([[np.mean(q),np.std(q),np.max(q),np.median(q)] for q in va],float)
    agreement=camp/np.maximum(vast[:,0],EPS)
    feats=[cons,camp[:,None],vast,agreement[:,None]]
    for q in [gate,ent,peak,edge,dsig,lsig]: feats.append(scalar_stats(stats_visible(q,VA)))
    visprob=1/(1+np.exp(-vlog)); feats.append(scalar_stats(stats_visible(visprob,np.ones_like(VA))))
    feats.append(VA.sum(0)[:,None].astype(float))
    obs_scale=m.local_scale(PA);feats.append(obs_scale[:,None])
    B1=np.concatenate(feats,axis=1)
    Z=m.descriptor_consensus(desc,VA);B2=np.concatenate([B1,Z],axis=1)
    z=np.load(root/'B'/'observation_sidecar.npz');center=np.asarray(z['camera_center'],float);half=float(z['camera_half_extent'])
    TA=(np.asarray(z['surface_points_A'],float)-center)/(2*half);TB=(np.asarray(z['surface_points_B'],float)-center)/(2*half)
    sA=m.local_scale(TA);D=np.linalg.norm(PA[:,None,:]-TA[None,:,:],axis=2);j=D.argmin(1);maperr=D[np.arange(64),j];reliable=maperr<=2*sA[j]
    td=TB[j]-TA[j];tamp=np.linalg.norm(td,axis=1);active=tamp>.005;mtarget=np.log(EPS+tamp/np.maximum(obs_scale,EPS))
    return {'fid':fid,'reliable':reliable,'active':active,'tamp':tamp,'mtarget':mtarget,'B0':np.log(EPS+camp)[:,None],'B1':B1,'B2':B2}

def threshold_best(y,p):
    qs=np.unique(np.r_[0,1,np.quantile(p,np.linspace(0,1,201))]);best=(.5,-1)
    for t in qs:
        ba=balanced_accuracy_score(y,p>=t)
        if ba>best[1]+1e-12:best=(float(t),float(ba))
    return best[0]
def pairacc(ytrue,score,amp):
    ok=[]
    n=len(amp)
    for i in range(n):
      for j in range(i+1,n):
        if abs(amp[i]-amp[j])/max(amp[i],amp[j],EPS)<.10:continue
        ok.append((score[i]-score[j])*(amp[i]-amp[j])>0)
    return float(np.mean(ok)) if ok else None

data={f:extract(f) for f in F}
out={'schema':'RealSaS.N1D.FactorizedMotionHeadP0.v1','date':'2026-08-19','prereg_commit':'5ad75bbe5434af901f105f1654e1f8e2b074ff33','arms':{}}
for arm in ['B0','B1','B2']:
    pf={};ally=[];allp=[];allamp=[];alls=[];allm=[]
    for held in F:
        tr=[];te=[]
        for f,d in data.items():
            ids=np.where(d['reliable'])[0]
            target=te if f==held else tr
            for i in ids:target.append((d[arm][i],int(d['active'][i]),float(d['tamp'][i]),float(d['mtarget'][i])))
        Xtr=np.stack([r[0] for r in tr]);ytr=np.array([r[1] for r in tr]);Xte=np.stack([r[0] for r in te]);yte=np.array([r[1] for r in te]);amptr=np.array([r[2] for r in tr]);ampte=np.array([r[2] for r in te]);mtr=np.array([r[3] for r in tr]);mte=np.array([r[3] for r in te])
        clf=make_pipeline(StandardScaler(),LogisticRegression(C=.3,class_weight='balanced',max_iter=2000,random_state=0));clf.fit(Xtr,ytr);ptr=clf.predict_proba(Xtr)[:,1];pte=clf.predict_proba(Xte)[:,1];thr=threshold_best(ytr,ptr)
        pa={'n':len(yte),'active':int(yte.sum()),'auroc':float(roc_auc_score(yte,pte)),'balanced_accuracy':float(balanced_accuracy_score(yte,pte>=thr)),'brier':float(brier_score_loss(yte,pte)),'threshold_from_train':thr}
        acttr=ytr==1;actte=yte==1
        if arm=='B0':
            str_=Xte[actte,0]
            rg=Ridge(alpha=10).fit(Xtr[acttr],mtr[acttr]);predm=rg.predict(Xte[actte])
        else:
            rg=make_pipeline(StandardScaler(),Ridge(alpha=10));rg.fit(Xtr[acttr],mtr[acttr]);predm=rg.predict(Xte[actte]);str_=predm
        rho=float(spearmanr(ampte[actte],str_).statistic) if actte.sum()>=3 else None
        pacc=pairacc(mte[actte],str_,ampte[actte]);mae=float(np.median(np.abs(predm-mte[actte]))) if actte.sum() else None
        la={'n_active':int(actte.sum()),'spearman':rho,'pairwise_accuracy':pacc,'median_abs_error':mae}
        pf[str(held)]={'p_active':pa,'log_amp':la}
        ally.extend(yte.tolist());allp.extend(pte.tolist());allamp.extend(ampte[actte].tolist());alls.extend(str_.tolist());allm.extend(mte[actte].tolist())
    p_pool={'auroc':float(roc_auc_score(ally,allp)),'brier':float(brier_score_loss(ally,allp)),'balanced_accuracy_mean_family':float(np.mean([q['p_active']['balanced_accuracy'] for q in pf.values()])),'worst_family_auroc':float(min(q['p_active']['auroc'] for q in pf.values())),'worst_family_balanced_accuracy':float(min(q['p_active']['balanced_accuracy'] for q in pf.values()))}
    lsp=[q['log_amp']['spearman'] for q in pf.values() if q['log_amp']['spearman'] is not None];lp=[q['log_amp']['pairwise_accuracy'] for q in pf.values() if q['log_amp']['pairwise_accuracy'] is not None]
    l_pool={'pooled_spearman':float(spearmanr(allamp,alls).statistic),'median_family_spearman':float(np.median(lsp)),'worst_family_spearman':float(min(lsp)),'negative_family_count':int(sum(x<0 for x in lsp)),'pooled_pairwise_accuracy_proxy':float(np.mean(lp)),'worst_family_pairwise_accuracy':float(min(lp))}
    pg=p_pool['auroc']>=.93 and p_pool['worst_family_auroc']>=.88 and p_pool['worst_family_balanced_accuracy']>=.70 and p_pool['balanced_accuracy_mean_family']>=.82 and all(q['p_active']['auroc']>=.85 for q in pf.values())
    lg=l_pool['pooled_spearman']>=.75 and l_pool['median_family_spearman']>=.65 and l_pool['worst_family_spearman']>=.45 and l_pool['negative_family_count']==0 and l_pool['pooled_pairwise_accuracy_proxy']>=.72 and l_pool['worst_family_pairwise_accuracy']>=.60
    out['arms'][arm]={'per_family':pf,'p_active_pooled':p_pool,'log_amp_pooled':l_pool,'gates':{'p_active':bool(pg),'log_amp':bool(lg),'both':bool(pg and lg)}}
if out['arms']['B1']['gates']['both']:
    b1=out['arms']['B1'];b2=out['arms']['B2'];
    if b2['gates']['both'] and b2['p_active_pooled']['worst_family_auroc']>=b1['p_active_pooled']['worst_family_auroc']+.02 and b2['log_amp_pooled']['worst_family_spearman']>=b1['log_amp_pooled']['worst_family_spearman']+.05:dec='B2_TYPED_PLUS_Z'
    else:dec='B1_TYPED_SCALAR'
elif out['arms']['B2']['gates']['both']:dec='B2_TYPED_PLUS_Z'
elif out['arms']['B0']['gates']['both']:dec='B0_RAW_SCALAR_ONLY__RICHER_HEAD_NOT_AUTHORIZED'
else:dec='NO_ARM_PASSES__FACTOR_HEAD_NOT_AUTHORIZED'
out['decision']=dec
p=Path('/mnt/data/sufficiency/FACTORIZED_MOTION_HEAD_P0_RESULT.json');p.write_text(json.dumps(out,indent=2,allow_nan=False));print(json.dumps({'decision':dec,'arms':{k:{'p':v['p_active_pooled'],'l':v['log_amp_pooled'],'gates':v['gates'],'per_family':{f:{'p_auc':q['p_active']['auroc'],'p_ba':q['p_active']['balanced_accuracy'],'rho':q['log_amp']['spearman'],'pair':q['log_amp']['pairwise_accuracy']} for f,q in v['per_family'].items()}} for k,v in out['arms'].items()}},indent=2));print('sha',hashlib.sha256(p.read_bytes()).hexdigest())
