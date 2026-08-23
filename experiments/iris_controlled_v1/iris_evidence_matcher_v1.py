from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F


def _norm(x, axis=-1, eps=1e-8):
    x=np.asarray(x,np.float32); return x/np.maximum(np.linalg.norm(x,axis=axis,keepdims=True),eps)

def _rank01(cost):
    cost=np.asarray(cost,np.float32); n=len(cost)
    if n<=1: return np.zeros(n,np.float32)
    order=np.argsort(cost,kind='stable'); r=np.empty(n,np.float32); r[order]=np.arange(n,dtype=np.float32)
    return r/float(n-1)

def _sim_cost(a,b):
    return (1.0-np.clip(np.sum(a*b,axis=-1),-1.0,1.0))*0.5

def sample_field_at_points(field: torch.Tensor, view: int, xy: np.ndarray) -> np.ndarray:
    """field [1,V,C,H,W], xy [K,2] normalized grid -> [K,C]."""
    if len(xy)==0: return np.zeros((0,field.shape[2]),np.float32)
    g=torch.as_tensor(xy,dtype=field.dtype,device=field.device)[None,:,None,:]
    f=field[:,view]
    y=F.grid_sample(f,g,mode='bilinear',padding_mode='zeros',align_corners=False)
    return y[0,:,:,0].T.detach().float().cpu().numpy()

@dataclass
class MatcherConfig:
    row_tol: float = 0.0625
    coarse_k: int = 8
    output_k: int = 8
    reverse_k: int = 4
    cycle_k: int = 4
    cycle_tol: float = 0.05
    w_coarse: float = 0.25
    w_fine: float = 0.45
    w_p: float = 0.30
    singleton_margin: float = 0.10
    singleton_require_reciprocal: bool = True
    singleton_min_cycle: int = 1

class ViewEvidence:
    def __init__(self, xy, P, N, U, Zc, Zf, track_ids=None):
        self.xy=np.asarray(xy,np.float32); self.P=np.asarray(P,np.float32); self.N=_norm(N); self.U=np.asarray(U,np.float32).reshape(-1)
        self.Zc=_norm(Zc); self.Zf=_norm(Zf); self.track_ids=None if track_ids is None else np.asarray(track_ids,np.int64)
    def __len__(self): return len(self.xy)

class IRISEvidenceMatcherV1:
    """Deterministic evidence composition. Ground-truth track ids are never used by matching logic."""
    def __init__(self, views: Dict[int,ViewEvidence], cfg: MatcherConfig): self.views=views; self.cfg=cfg; self._base_cache={}

    def _domain(self, sv:int, si:int, tv:int):
        s=self.views[sv]; t=self.views[tv]
        if len(t)==0: return np.empty(0,np.int64)
        dy=np.abs(t.xy[:,1]-s.xy[si,1]); ids=np.flatnonzero(dy<=self.cfg.row_tol)
        if len(ids)==0: ids=np.arange(len(t),dtype=np.int64)
        return ids

    def base_rank(self, sv:int, si:int, tv:int, limit:int|None=None):
        key=(int(sv),int(si),int(tv))
        if key in self._base_cache:
            full=self._base_cache[key]
            return full if limit is None else full[:limit]
        s=self.views[sv]; t=self.views[tv]; ids=self._domain(sv,si,tv)
        if len(ids)==0: return []
        zc_cost=_sim_cost(np.repeat(s.Zc[si:si+1],len(ids),0),t.Zc[ids])
        c_order=np.argsort(zc_cost,kind='stable')[:min(self.cfg.coarse_k,len(ids))]
        ids=ids[c_order]; zc_cost=zc_cost[c_order]
        zf_cost=_sim_cost(np.repeat(s.Zf[si:si+1],len(ids),0),t.Zf[ids])
        pd=np.linalg.norm(t.P[ids]-s.P[si],axis=1)
        sigma=np.exp(np.clip(t.U[ids],-6,3))+float(np.exp(np.clip(s.U[si],-6,3)))+1e-3
        p_cost=pd/sigma
        fused=(self.cfg.w_coarse*_rank01(zc_cost)+self.cfg.w_fine*_rank01(zf_cost)+self.cfg.w_p*_rank01(p_cost))
        order=np.argsort(fused,kind='stable')
        full=[{'idx':int(ids[j]),'cost':float(fused[j]),'zc_cost':float(zc_cost[j]),'zf_cost':float(zf_cost[j]),'p_dist':float(pd[j])} for j in order[:self.cfg.output_k]]
        self._base_cache[key]=full
        return full if limit is None else full[:limit]

    def reciprocal(self,sv,si,tv,ti):
        back=self.base_rank(tv,ti,sv,self.cfg.reverse_k)
        return any(x['idx']==si for x in back)

    def cycle_support(self,sv,si,tv,ti):
        support=0; checked=0
        for cv,c in self.views.items():
            if cv in (sv,tv) or len(c)==0: continue
            a=self.base_rank(sv,si,cv,self.cfg.cycle_k); b=self.base_rank(tv,ti,cv,self.cfg.cycle_k)
            if not a or not b: continue
            checked+=1; axy=c.xy[[x['idx'] for x in a]]; bxy=c.xy[[x['idx'] for x in b]]
            d=np.linalg.norm(axy[:,None,:]-bxy[None,:,:],axis=-1)
            if float(d.min())<=self.cfg.cycle_tol: support+=1
        return support,checked

    def match(self,sv:int,si:int,tv:int):
        base=self.base_rank(sv,si,tv,self.cfg.output_k)
        if not base: return {'ranked':[],'set_size':0,'confident_singleton':False}
        for r in base:
            r['reciprocal']=self.reciprocal(sv,si,tv,r['idx'])
            r['cycle_support'],r['cycle_checked']=self.cycle_support(sv,si,tv,r['idx'])
        ranked=sorted(base,key=lambda r:(-int(r['reciprocal']),-r['cycle_support'],r['cost']))
        margin=(ranked[1]['cost']-ranked[0]['cost']) if len(ranked)>1 else 1.0
        top=ranked[0]
        confident=(margin>=self.cfg.singleton_margin and ((not self.cfg.singleton_require_reciprocal) or top['reciprocal']) and top['cycle_support']>=self.cfg.singleton_min_cycle)
        if confident: set_size=1
        elif top['reciprocal'] or top['cycle_support']>=1: set_size=min(4,len(ranked))
        else: set_size=min(8,len(ranked))
        return {'ranked':ranked,'margin':float(margin),'confident_singleton':bool(confident),'set_size':int(set_size)}


def build_view_evidence(outputs:Dict[str,torch.Tensor], cache:Dict[str,np.ndarray]):
    vis=cache['track_visible'].astype(bool); xy=cache['track_xy'].astype(np.float32)
    views={}
    for v in range(vis.shape[1]):
        tids=np.flatnonzero(vis[:,v]); pts=xy[tids,v]
        views[v]=ViewEvidence(pts,
            sample_field_at_points(outputs['P'],v,pts),
            sample_field_at_points(outputs['N'],v,pts),
            sample_field_at_points(outputs['log_sigma'],v,pts)[:,0],
            sample_field_at_points(outputs['Z_coarse'],v,pts),
            sample_field_at_points(outputs['Z_fine'],v,pts),tids)
    return views


def calibrate_config(base_cfg:MatcherConfig, asset_views:List[Dict[int,ViewEvidence]], max_queries=3000):
    grids=[]
    for rt in (0.03125,0.0625,0.125):
      for wc,wf,wp in ((.2,.5,.3),(.25,.45,.3),(.2,.4,.4),(.3,.5,.2)):
        grids.append((rt,wc,wf,wp))
    best=None
    for rt,wc,wf,wp in grids:
        cfg=MatcherConfig(**{**base_cfg.__dict__,'row_tol':rt,'w_coarse':wc,'w_fine':wf,'w_p':wp,'singleton_margin':0.0})
        hit1=hit4=n=0
        for views in asset_views:
            m=IRISEvidenceMatcherV1(views,cfg)
            for sv,s in views.items():
                for si,tid in enumerate(s.track_ids):
                    for tv,t in views.items():
                        if tv==sv: continue
                        truth=np.flatnonzero(t.track_ids==tid)
                        if not len(truth): continue
                        r=m.base_rank(sv,si,tv,8); ids=[x['idx'] for x in r]; n+=1
                        hit1+=bool(ids and ids[0]==int(truth[0])); hit4+=int(truth[0]) in ids[:4]
                        if n>=max_queries: break
                    if n>=max_queries: break
                if n>=max_queries: break
            if n>=max_queries: break
        key=(hit1/max(n,1),hit4/max(n,1),-rt)
        if best is None or key>best[0]: best=(key,cfg)
    return best[1]


def calibrate_singleton_margin(cfg:MatcherConfig, asset_views:List[Dict[int,ViewEvidence]], target_precision=.99,max_queries=3000):
    rows=[]; n=0
    probe=MatcherConfig(**{**cfg.__dict__,'singleton_margin':0.0})
    for views in asset_views:
        m=IRISEvidenceMatcherV1(views,probe)
        for sv,s in views.items():
            for si,tid in enumerate(s.track_ids):
                for tv,t in views.items():
                    if tv==sv: continue
                    truth=np.flatnonzero(t.track_ids==tid)
                    if not len(truth): continue
                    rr=m.match(sv,si,tv); ranked=rr['ranked']
                    if ranked:
                        top=ranked[0]; correct=(top['idx']==int(truth[0]))
                        eligible=top['reciprocal'] and top['cycle_support']>=cfg.singleton_min_cycle
                        rows.append((rr['margin'],correct,eligible)); n+=1
                    if n>=max_queries: break
                if n>=max_queries: break
            if n>=max_queries: break
        if n>=max_queries: break
    margins=sorted({x[0] for x in rows if x[2]})
    chosen=max(margins) if margins else 1.0; bestcov=-1
    for th in margins:
        sel=[x for x in rows if x[2] and x[0]>=th]
        if not sel: continue
        prec=sum(x[1] for x in sel)/len(sel); cov=len(sel)/max(len(rows),1)
        if prec>=target_precision and cov>bestcov: chosen=th; bestcov=cov
    return MatcherConfig(**{**cfg.__dict__,'singleton_margin':float(chosen)})
