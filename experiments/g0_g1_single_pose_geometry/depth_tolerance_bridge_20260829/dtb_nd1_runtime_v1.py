#!/usr/bin/env python3
"""Frozen DTB-ND1 robust-local-plane persistence runtime.

Only differential-normal derivation differs from DTB-S1/DTB-S1R. Depth corruption,
source rows, matcher gates, reciprocal D2 admission and X36 contract remain unchanged.
"""
from __future__ import annotations
import numpy as np

from bridge_persistence_v1 import CorruptedPersistenceCarrier, _x36_from_carrier
from depth_corruption_v1 import DepthCorruptionSpec, depth_delta_for_view, corrupt_points_along_ray, corruption_diagnostics
from e0_geometry import camera_for_view, project_grid, grid_to_nearest_pixel, raster_lookup_near
from surface_builder_e0_v1 import ObservableView
from robust_local_plane_v1 import robust_local_plane_normals_for_rows

MUTUAL_P003=0.003
VIEWS=8
class NormalCache:
    def __init__(self, obs: ObservableView):
        self.obs=obs
        n=len(obs.P)
        self.N=np.zeros((n,3),np.float32); self.valid=np.zeros(n,bool); self.done=np.zeros(n,bool)
        self.count=np.zeros(n,np.int16); self.requested=set()
    def get(self, rows):
        rows=np.asarray(rows,np.int64).reshape(-1)
        if len(rows)==0: return np.zeros((0,3),np.float32),np.zeros(0,bool)
        u=np.unique(rows[rows>=0]); self.requested.update(map(int,u))
        missing=u[~self.done[u]]
        if len(missing):
            n,v,c=robust_local_plane_normals_for_rows(self.obs.pixel_linear_index,self.obs.P,self.obs.resolution,missing)
            self.N[missing]=n; self.valid[missing]=v; self.count[missing]=c; self.done[missing]=True
        return self.N[rows],self.valid[rows]


def corrupt_p_only(asset_id, clean_obs, spec):
    out=[]; reports=[]
    for obs in clean_obs:
        cam=camera_for_view(obs.view,obs.half_extent)
        dd=depth_delta_for_view(asset_id,obs.view,obs.pixel_linear_index,resolution=obs.resolution,spec=spec)
        ph=corrupt_points_along_ray(obs.P,cam['forward'],dd)
        out.append(ObservableView(obs.view,obs.resolution,obs.pixel_linear_index.copy(),ph,np.zeros_like(ph),np.zeros(len(ph),bool),obs.half_extent,obs.camera_recovery))
        reports.append({'view':int(obs.view),**corruption_diagnostics(obs.P,ph,cam['forward'],dd)})
    return tuple(out),tuple(reports)


def robust_batch_rows(anchor_P,anchor_N,anchor_valid,anchor_grid,source_view,source_half_extent,target:ObservableView,target_cache:NormalCache,*,radius_px=4,max_common_frame_error=0.006,min_abs_normal_cos=np.cos(np.deg2rad(50.0)),max_reciprocal_error_px=3.0):
    P=np.asarray(anchor_P,np.float32); N=np.asarray(anchor_N,np.float32); Gd=np.asarray(anchor_grid,np.float32); av=np.asarray(anchor_valid,bool)
    K=len(P)
    if K==0: return np.zeros(0,np.int64)
    tg=project_grid(P,camera_for_view(target.view,target.half_extent)); inframe=np.all(np.abs(tg)<=1.0,axis=1)
    x,y=grid_to_nearest_pixel(tg,target.resolution); cand=raster_lookup_near(target.pixel_linear_index,x,y,target.resolution,radius_px)
    cvalid=cand>=0; cc=np.clip(cand,0,max(0,len(target.P)-1))
    tP=target.P[cc]; pd=np.linalg.norm(tP-P[:,None,:],axis=2)
    flat=cc.reshape(-1); tNflat,tVflat=target_cache.get(flat); tN=tNflat.reshape(cc.shape+(3,)); tV=tVflat.reshape(cc.shape)
    src_norm=np.linalg.norm(N,axis=1); tgt_norm=np.linalg.norm(tN,axis=2)
    nvalid=cvalid & av[:,None] & (src_norm[:,None]>0.5) & tV & (tgt_norm>0.5)
    ndot=np.zeros_like(pd,dtype=np.float32); numer=np.abs(np.sum(tN*N[:,None,:],axis=2)); denom=tgt_norm*src_norm[:,None]+1e-8; ndot[nvalid]=(numer/denom)[nvalid]
    back=project_grid(tP.reshape(-1,3),camera_for_view(source_view,source_half_extent)).reshape(K,-1,2)
    recip=np.linalg.norm((back-Gd[:,None,:])*(target.resolution/2.0),axis=2)
    valid=cvalid&inframe[:,None]&(pd<=max_common_frame_error)&nvalid&(ndot>=min_abs_normal_cos)&(recip<=max_reciprocal_error_px)
    score=(pd/max_common_frame_error)+(1.0-ndot)+(recip/max_reciprocal_error_px); score[~valid]=np.inf
    j=np.argmin(score,axis=1); best=cand[np.arange(K),j]; ok=np.isfinite(score[np.arange(K),j])
    return np.where(ok,best,-1).astype(np.int64)


def build_robust(prepared,spec,clean_caches):
    obs,reports=corrupt_p_only(prepared.asset_id,prepared.clean_obs,spec); caches=[NormalCache(o) for o in obs]
    sv=np.asarray(prepared.source_view,np.int64); sr=np.asarray(prepared.source_row,np.int64); K=len(sv)
    anchor_P=np.zeros((K,3),np.float32); source_grid=np.zeros((K,2),np.float32); source_N=np.zeros((K,3),np.float32); source_V=np.zeros(K,bool)
    matched=np.full((K,VIEWS),-1,np.int64); support=np.zeros((K,VIEWS),np.uint8); raster=np.zeros((K,VIEWS,2),np.float32); grids=[o.grid for o in obs]
    for sview in range(VIEWS):
        ids=np.flatnonzero(sv==sview)
        if not len(ids): continue
        rows=sr[ids]; anchor_P[ids]=obs[sview].P[rows]; source_grid[ids]=grids[sview][rows]
        nn,vv=caches[sview].get(rows); source_N[ids]=nn; source_V[ids]=vv
        matched[ids,sview]=rows; support[ids,sview]=1; raster[ids,sview]=source_grid[ids]
        for tv in range(VIEWS):
            if tv==sview: continue
            rr=robust_batch_rows(anchor_P[ids],source_N[ids],source_V[ids],source_grid[ids],sview,obs[sview].half_extent,obs[tv],caches[tv])
            matched[ids,tv]=rr; ok=rr>=0
            if np.any(ok):
                good=ids[ok]; support[good,tv]=1; raster[good,tv]=grids[tv][rr[ok]]
    base=support.astype(bool); d2=np.zeros_like(base); d2[np.arange(K),sv]=True
    for sview in range(VIEWS):
      for tv in range(VIEWS):
        if tv==sview: continue
        ids=np.flatnonzero((sv==sview)&base[:,tv])
        if not len(ids): continue
        br=matched[ids,tv]; n,v=caches[tv].get(br)
        rr=robust_batch_rows(obs[tv].P[br],n,v,grids[tv][br],tv,obs[tv].half_extent,obs[sview],caches[sview])
        ok=rr>=0; cyc=np.full(len(ids),np.inf,np.float32)
        if np.any(ok): cyc[ok]=np.linalg.norm(obs[sview].P[rr[ok]]-anchor_P[ids[ok]],axis=1)
        d2[ids,tv]=ok&(cyc<=MUTUAL_P003)
    admitted=matched.copy(); admitted[~d2]=-1
    dgrid=np.zeros((K,VIEWS,2),np.float32)
    for v in range(VIEWS):
        ids=np.flatnonzero(d2[:,v]&(admitted[:,v]>=0))
        if len(ids): dgrid[ids,v]=grids[v][admitted[ids,v]]
    X=np.concatenate([anchor_P,d2.astype(np.float32),dgrid.reshape(K,16),(d2.sum(axis=1,keepdims=True)/8.0).astype(np.float32),np.stack([anchor_P@np.asarray(camera_for_view(v,obs[v].half_extent)['forward'],np.float32) for v in range(8)],axis=1)],axis=1).astype(np.float32)
    di=[]
    for v in range(8):
        rows=np.asarray(sorted(caches[v].requested),np.int64)
        if len(rows)==0: continue
        cn,cv=clean_caches[v].get(rows); pn,pv=caches[v].get(rows); both=cv&pv
        ang=np.array([],np.float32)
        if np.any(both):
            dot=np.clip(np.abs(np.sum(cn[both]*pn[both],axis=1)),-1,1); ang=np.degrees(np.arccos(dot)).astype(np.float32)
        di.append({'view':v,'requested_rows':int(len(rows)),'valid_fraction':float(np.mean(pv)),'both_valid_fraction':float(np.mean(both)),'theta_p50_deg':float(np.quantile(ang,.5)) if len(ang) else None,'theta_p95_deg':float(np.quantile(ang,.95)) if len(ang) else None})
    carrier=CorruptedPersistenceCarrier(prepared.asset_id,anchor_P,support,raster,np.asarray(prepared.source_view,np.int16),np.asarray(prepared.source_row,np.int64),matched,_x36_from_carrier(anchor_P,support,raster),reports)
    return carrier,d2,admitted,dgrid,X,di
