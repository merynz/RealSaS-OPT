from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import hashlib
import numpy as np
import torch
import torch.nn.functional as F

FINE_RES=128
ROW_TOL=0.030
COARSE_KEEP=16
P_KEEP=4
TOPK_MAX=8
RRF_K=60.0
TRUTH_TOL=0.018
AGREE_TOL=0.020
CYCLE_TOL=0.030
QUERY_CAP_PER_PAIR=64
UNCERTAIN_Q=.75
VERY_UNCERTAIN_Q=.90


def sample_field_view(field: torch.Tensor, view: int, xy: np.ndarray) -> np.ndarray:
    if len(xy)==0: return np.zeros((0,field.shape[2]),np.float32)
    g=torch.as_tensor(xy,dtype=field.dtype,device=field.device)[None,:,None,:]
    s=F.grid_sample(field[:,view],g,mode='bilinear',padding_mode='zeros',align_corners=False)
    return s[0,:,:,0].T.detach().float().cpu().numpy()
def normalize(x): return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8)
def fine_grid_coords(mask: np.ndarray) -> np.ndarray:
    yy,xx=np.nonzero(mask); gx=2.0*(xx.astype(np.float32)+.5)/FINE_RES-1.0; gy=2.0*(yy.astype(np.float32)+.5)/FINE_RES-1.0; return np.stack([gx,gy],-1).astype(np.float32)
def risk_percentiles(values):
    v=np.asarray(values,np.float64).reshape(-1)
    if len(v)<=1: return np.zeros(len(v),np.float32),np.sort(v)
    order=np.argsort(v,kind='stable'); r=np.empty(len(v),np.float64); r[order]=np.arange(len(v)); return (r/(len(v)-1)).astype(np.float32),np.sort(v)
def query_risk_percentile(value,sorted_values):
    if len(sorted_values)<=1:return 0.0
    return float(np.searchsorted(sorted_values,float(value),side='right')-1)/float(len(sorted_values)-1)
def rank_positions(order,n):
    r=np.empty(n,np.int32);r[order]=np.arange(1,n+1,dtype=np.int32);return r
def hit_coords(cands: np.ndarray,truth:np.ndarray,k:int,tol:float=TRUTH_TOL)->bool:
    x=cands[:k]
    if len(x)==0:return False
    m=np.isfinite(x).all(-1)
    if not np.any(m):return False
    return bool(np.any(np.linalg.norm(x[m]-truth[None],axis=1)<=tol))
def close(a,b,tol=AGREE_TOL):return bool(np.linalg.norm(np.asarray(a)-np.asarray(b))<=tol)

@dataclass
class DenseView:
    coords:np.ndarray;P:np.ndarray;U:np.ndarray;risk_q:np.ndarray;sorted_U:np.ndarray;Zc:np.ndarray;Zf:np.ndarray
@dataclass
class DenseMatch:
    top_coords:np.ndarray;top_scores:np.ndarray;coarse_top_coords:np.ndarray;fine_top_coords:np.ndarray;p_top_coords:np.ndarray;coarse_top1:np.ndarray;fine_top1:np.ndarray;p_top1:np.ndarray;top1_target_risk_q:np.ndarray;source_risk_q:np.ndarray

def build_dense_views(outputs:Dict[str,torch.Tensor],images:torch.Tensor)->List[DenseView]:
    out=[]
    for v in range(8):
        alpha=images[0,v,3:4][None];a=F.interpolate(alpha,size=(FINE_RES,FINE_RES),mode='bilinear',align_corners=False)[0,0];mask=(a>.20).detach().cpu().numpy();coords=fine_grid_coords(mask)
        if len(coords)==0:raise RuntimeError(f'empty alpha support view {v}')
        P=sample_field_view(outputs['P'],v,coords);U=sample_field_view(outputs['log_sigma'],v,coords)[:,0];Zc=normalize(sample_field_view(outputs['Z_coarse'],v,coords));Zf=normalize(sample_field_view(outputs['Z_fine'],v,coords));rq,su=risk_percentiles(U);out.append(DenseView(coords,P,U,rq,su,Zc,Zf))
    return out

def match_queries(outputs,views:List[DenseView],s:int,t:int,qxy:np.ndarray)->DenseMatch:
    qxy=np.asarray(qxy,np.float32);Q=len(qxy);tv=views[t];Ps=sample_field_view(outputs['P'],s,qxy);Us=sample_field_view(outputs['log_sigma'],s,qxy)[:,0];Zcs=normalize(sample_field_view(outputs['Z_coarse'],s,qxy));Zfs=normalize(sample_field_view(outputs['Z_fine'],s,qxy));coarse=Zcs@tv.Zc.T;fine=Zfs@tv.Zf.T;pd=np.linalg.norm(Ps[:,None,:]-tv.P[None,:,:],axis=-1);row_ok=np.abs(qxy[:,None,1]-tv.coords[None,:,1])<=ROW_TOL
    top=np.full((Q,TOPK_MAX,2),np.nan,np.float32);scores=np.full((Q,TOPK_MAX),-np.inf,np.float32);ct=np.full_like(top,np.nan);ft=np.full_like(top,np.nan);pt=np.full_like(top,np.nan);c1=np.full((Q,2),np.nan,np.float32);f1=np.full_like(c1,np.nan);p1=np.full_like(c1,np.nan);trisk=np.ones(Q,np.float32);srisk=np.asarray([query_risk_percentile(x,views[s].sorted_U) for x in Us],np.float32);all_t=np.arange(len(tv.coords),dtype=np.int32)
    for qi in range(Q):
        valid=np.flatnonzero(row_ok[qi]).astype(np.int32)
        if len(valid)==0:valid=all_t
        co=valid[np.argsort(-coarse[qi,valid],kind='stable')];fo=valid[np.argsort(-fine[qi,valid],kind='stable')];po=valid[np.argsort(pd[qi,valid],kind='stable')]
        for arr,ordr in ((ct,co),(ft,fo),(pt,po)):
            take=min(TOPK_MAX,len(ordr));arr[qi,:take]=tv.coords[ordr[:take]]
        c1[qi]=tv.coords[co[0]];f1[qi]=tv.coords[fo[0]];p1[qi]=tv.coords[po[0]];keep=np.unique(np.concatenate([co[:min(COARSE_KEEP,len(co))],po[:min(P_KEEP,len(po))]])).astype(np.int32);rc=rank_positions(np.argsort(-coarse[qi,keep],kind='stable'),len(keep));rf=rank_positions(np.argsort(-fine[qi,keep],kind='stable'),len(keep));rp=rank_positions(np.argsort(pd[qi,keep],kind='stable'),len(keep));rrf=1/(RRF_K+rc)+1/(RRF_K+rf)+1/(RRF_K+rp);oo=np.argsort(-rrf,kind='stable');take=min(TOPK_MAX,len(oo));ids=keep[oo[:take]];top[qi,:take]=tv.coords[ids];scores[qi,:take]=rrf[oo[:take]].astype(np.float32);trisk[qi]=tv.risk_q[ids[0]]
    return DenseMatch(top,scores,ct,ft,pt,c1,f1,p1,trisk,srisk)
def third_views(s,t):
    out=[]
    for d in (2,-2,3,-3,1,-1):
        u=(s+d)%8
        if u not in (s,t) and u not in out:out.append(u)
        if len(out)==2:break
    return out
def select_queries(asset_id,s,t,track_ids):
    ids=list(map(int,track_ids));ids=sorted(ids,key=lambda x:hashlib.sha256(f'{asset_id}|{s}|{t}|{x}'.encode()).hexdigest());return np.asarray(ids[:QUERY_CAP_PER_PAIR],np.int32)
def evaluate_dense_asset(asset_id,outputs,images,z):
    views=build_dense_views(outputs,images);vis=z['track_visible'].astype(bool);xy=z['track_xy'].astype(np.float32);C={k:0 for k in ['queries','row_truth','c1','c4','c8','f1','f4','f8','p1','p4','p8','m1','m4','m8','r1','r4','cycle','single','single_ok','set4','set8','out_ok']};C['set_sum']=0;hard=[];wrong=[]
    for s in range(8):
      for t in range(8):
        if s==t:continue
        common=np.flatnonzero(vis[:,s]&vis[:,t]);qids=select_queries(asset_id,s,t,common)
        if len(qids)==0:continue
        qxy=xy[qids,s];truth=xy[qids,t];M=match_queries(outputs,views,s,t,qxy);R=match_queries(outputs,views,t,s,M.top_coords[:,0]);directs={};indirects={}
        for u in third_views(s,t):directs[u]=match_queries(outputs,views,s,u,qxy);indirects[u]=match_queries(outputs,views,t,u,M.top_coords[:,0])
        for i,tr in enumerate(truth):
            C['queries']+=1;C['row_truth']+=int(abs(float(qxy[i,1]-tr[1]))<=ROW_TOL)
            for pref,arr in [('c',M.coarse_top_coords),('f',M.fine_top_coords),('p',M.p_top_coords),('m',M.top_coords)]:
                for k in (1,4,8):C[f'{pref}{k}']+=int(hit_coords(arr[i],tr,k))
            recip1=hit_coords(R.top_coords[i],qxy[i],1,AGREE_TOL);recip4=hit_coords(R.top_coords[i],qxy[i],4,AGREE_TOL);C['r1']+=int(recip1);C['r4']+=int(recip4);cyc=0
            for u in third_views(s,t):
                if close(directs[u].top_coords[i,0],indirects[u].top_coords[i,0],CYCLE_TOL):cyc+=1
            C['cycle']+=int(cyc>=1);votes=int(close(M.coarse_top1[i],M.top_coords[i,0]))+int(close(M.fine_top1[i],M.top_coords[i,0]))+int(close(M.p_top1[i],M.top_coords[i,0]));risk=max(float(M.source_risk_q[i]),float(M.top1_target_risk_q[i]));singleton=bool(recip1 and cyc>=1 and votes>=2 and risk<UNCERTAIN_Q)
            if singleton:
                size=1;C['single']+=1;ok=hit_coords(M.top_coords[i],tr,1);C['single_ok']+=int(ok)
                if not ok:wrong.append({'s':s,'t':t,'query':qxy[i].tolist(),'truth':tr.tolist(),'pred':M.top_coords[i,0].tolist(),'risk_q':risk,'votes':votes,'cycle':cyc})
            elif risk>=VERY_UNCERTAIN_Q or not recip4:size=8;C['set8']+=1
            else:size=4;C['set4']+=1
            C['set_sum']+=size;ok=hit_coords(M.top_coords[i],tr,size);C['out_ok']+=int(ok)
            if not hit_coords(M.top_coords[i],tr,8):hard.append({'s':s,'t':t,'query':qxy[i].tolist(),'truth':tr.tolist(),'top8':M.top_coords[i].tolist()})
    q=max(C['queries'],1);sc=max(C['single'],1)
    return {'queries':C['queries'],'truth_row_domain_recall':C['row_truth']/q,'raw_coarse':{f'top{k}':C[f"c{k}"]/q for k in(1,4,8)},'raw_fine':{f'top{k}':C[f"f{k}"]/q for k in(1,4,8)},'raw_P':{f'top{k}':C[f"p{k}"]/q for k in(1,4,8)},'composite':{f'top{k}':C[f"m{k}"]/q for k in(1,4,8)},'reciprocal_top1_rate':C['r1']/q,'reciprocal_top4_rate':C['r4']/q,'cycle_any_rate':C['cycle']/q,'qualification':{'singleton_coverage':C['single']/q,'singleton_precision':C['single_ok']/sc if C['single'] else None,'set4_rate':C['set4']/q,'set8_rate':C['set8']/q,'mean_set_size':C['set_sum']/q,'output_truth_coverage':C['out_ok']/q},'hard_tail_count':len(hard),'singleton_wrong_examples':wrong[:20],'hard_tail_examples':hard[:20]}
def aggregate(rows):
    if not rows:return {'assets':0,'queries':0}
    weights=np.asarray([max(1,r['queries']) for r in rows],np.float64)
    def w(path):
        vals=[]
        for r in rows:
            x=r
            for k in path:x=x[k]
            vals.append(np.nan if x is None else float(x))
        a=np.asarray(vals);m=np.isfinite(a);return None if not np.any(m) else float(np.sum(a[m]*weights[m])/np.sum(weights[m]))
    o={'assets':len(rows),'queries':int(weights.sum()),'truth_row_domain_recall':w(['truth_row_domain_recall'])}
    for sec in('raw_coarse','raw_fine','raw_P','composite'):o[sec]={k:w([sec,k]) for k in('top1','top4','top8')}
    o['reciprocal_top1_rate']=w(['reciprocal_top1_rate']);o['reciprocal_top4_rate']=w(['reciprocal_top4_rate']);o['cycle_any_rate']=w(['cycle_any_rate']);o['qualification']={k:w(['qualification',k]) for k in('singleton_coverage','singleton_precision','set4_rate','set8_rate','mean_set_size','output_truth_coverage')};o['family_tail']={'composite_top4_p10':float(np.percentile([r['composite']['top4'] for r in rows],10)),'composite_top8_p10':float(np.percentile([r['composite']['top8'] for r in rows],10)),'output_truth_coverage_p10':float(np.percentile([r['qualification']['output_truth_coverage'] for r in rows],10))};o['hard_tail_count']=int(sum(r['hard_tail_count'] for r in rows));return o
