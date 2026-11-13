from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Iterable

import numpy as np


def _hash(payload: object) -> str:
    def norm(v):
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, dict): return {str(k): norm(v[k]) for k in sorted(v, key=str)}
        if isinstance(v, (tuple, list)): return [norm(x) for x in v]
        if isinstance(v, (np.floating, np.integer)): return v.item()
        if isinstance(v, float) and not math.isfinite(v): raise ValueError("non-finite conditioning payload")
        return v
    return sha256(json.dumps(norm(payload), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class GeometryNormalizationV2:
    center: tuple[float,float,float]
    scale: float
    policy: str="BBOX_CENTER_MAX_EXTENT_V2"
    def normalize(self,xyz)->np.ndarray: return (np.asarray(xyz,np.float32)-np.asarray(self.center,np.float32))/float(self.scale)
    def denormalize(self,xyz)->np.ndarray: return np.asarray(xyz,np.float32)*float(self.scale)+np.asarray(self.center,np.float32)


@dataclass(frozen=True)
class GeppettoConditioningBatchV2:
    surface_ids: tuple[tuple[str,...],...]
    features: np.ndarray
    positions_normalized: np.ndarray
    valid_mask: np.ndarray
    normalizations: tuple[GeometryNormalizationV2,...]
    source_surface_hashes: tuple[str,...]
    local_geometry_operator_hashes: tuple[str,...]
    conditioning_hashes: tuple[str,...]
    feature_contract: tuple[str,...]
    schema_version: str="RealSaS.GeppettoConditioningBatch.v2"


FEATURE_CONTRACT_V2=("P_norm_x","P_norm_y","P_norm_z","local_cov_eig_0","local_cov_eig_1","local_cov_eig_2","radius","N_x","N_y","N_z","N_valid","support_v0","support_v1","support_v2","support_v3","support_v4","support_v5","support_v6","support_v7","support_fraction","raster_mean_x","raster_mean_y","raster_std_x","raster_std_y")
assert len(FEATURE_CONTRACT_V2)==24


def _normalization(points:np.ndarray)->GeometryNormalizationV2:
    p=np.asarray(points,np.float64)
    if p.ndim!=2 or p.shape[1]!=3 or len(p)==0 or not np.isfinite(p).all(): raise ValueError("surface points must be finite non-empty [N,3]")
    lo,hi=p.min(0),p.max(0); center=(lo+hi)*0.5; scale=float(np.max(hi-lo)); scale=1.0 if scale<=1e-12 else scale; return GeometryNormalizationV2(tuple(map(float,center)),scale)


def _covariance_spectrum(points_norm:np.ndarray,k:int=16)->np.ndarray:
    p=np.asarray(points_norm,np.float64); n=len(p); out=np.zeros((n,3),np.float32)
    if n<3: return out
    d2=np.sum((p[:,None,:]-p[None,:,:])**2,axis=-1); np.fill_diagonal(d2,np.inf); kk=min(max(2,int(k)),n-1); nn=np.argpartition(d2,kth=kk-1,axis=1)[:,:kk]
    for i in range(n):
        x=p[nn[i]]-p[i]; C=(x.T@x)/float(max(len(x),1)); ev=np.maximum(np.linalg.eigvalsh(C),0.0)[::-1]; s=float(ev.sum()); ev=ev/s if s>1e-12 else ev; out[i]=ev.astype(np.float32)
    return out


def _normalized_raster_bindings(surface,node)->np.ndarray:
    if not node.raster_bindings: return np.zeros((0,2),np.float32)
    xy=np.asarray([pair[1] for pair in node.raster_bindings],np.float64); mode=str(getattr(surface,"metadata",{}).get("raster_coordinate_system",""))
    if mode in {"GRID_XY","GRID_SAMPLE_XY","NORMALIZED_GRID_XY"}: g=xy
    elif mode=="PIXEL_CENTER_XY":
        resolution=int(getattr(surface,"metadata",{}).get("resolution",0))
        if resolution<=0: raise ValueError("PIXEL_CENTER_XY surface requires metadata.resolution")
        g=np.empty_like(xy); g[:,0]=2.0*(xy[:,0]+0.5)/float(resolution)-1.0; g[:,1]=2.0*(xy[:,1]+0.5)/float(resolution)-1.0
    else: raise ValueError("Geppetto V2 requires explicit raster_coordinate_system metadata")
    if not np.isfinite(g).all(): raise ValueError("non-finite raster binding")
    return g.astype(np.float32)


def _surface_features(surface,norm:GeometryNormalizationV2):
    nodes=tuple(sorted(surface.surface_nodes,key=lambda n:n.surface_id))
    if not nodes: raise ValueError("RiggingSurfaceIR contains no surface nodes")
    ids=tuple(n.surface_id for n in nodes)
    if len(ids)!=len(set(ids)): raise ValueError("duplicate surface_id")
    P=np.asarray([n.P for n in nodes],np.float32); pn=norm.normalize(P); cov=_covariance_spectrum(pn); rows=[]
    for i,n in enumerate(nodes):
        normal=np.zeros(3,np.float32); nvalid=0.0
        if n.derived_normal is not None:
            raw=np.asarray(n.derived_normal,np.float32); ln=float(np.linalg.norm(raw))
            if np.isfinite(raw).all() and ln>1e-8: normal=raw/ln; nvalid=1.0
        support=[1.0 if v in set(map(int,n.support_views)) else 0.0 for v in range(8)]; raster=_normalized_raster_bindings(surface,n); rstats=[float(raster[:,0].mean()),float(raster[:,1].mean()),float(raster[:,0].std()),float(raster[:,1].std())] if len(raster) else [0.0,0.0,0.0,0.0]
        row=[*map(float,pn[i]),*map(float,cov[i]),float(np.linalg.norm(pn[i])),*map(float,normal),nvalid,*support,float(sum(support)/8.0),*rstats]
        if len(row)!=24: raise AssertionError(f"Geppetto V2 feature width drift:{len(row)}")
        rows.append(row)
    feat=np.asarray(rows,np.float32)
    if not np.isfinite(feat).all(): raise ValueError("non-finite Geppetto V2 feature")
    return ids,feat,pn.astype(np.float32)


def _pad(items:list[np.ndarray],width:int):
    m=max(len(x) for x in items); out=np.zeros((len(items),m,width),np.float32); mask=np.zeros((len(items),m),bool)
    for b,x in enumerate(items):
        if x.ndim!=2 or x.shape[1]!=width: raise ValueError("conditioning width mismatch")
        out[b,:len(x)]=x; mask[b,:len(x)]=True
    return out,mask


class GeppettoConditioningAdapterV2:
    feature_dim=24
    def __call__(self,surfaces:Iterable)->GeppettoConditioningBatchV2:
        surfaces=tuple(surfaces)
        if not surfaces: raise ValueError("at least one surface required")
        ids_all=[]; feat_all=[]; pos_all=[]; norms=[]; hashes=[]; op_hashes=[]; cond=[]
        for surface in surfaces:
            source_hash=str(surface.geometry_lineage_hash)
            if not source_hash: raise ValueError("surface geometry_lineage_hash required")
            op_hash=str(getattr(surface,"metadata",{}).get("Nd_operator_sha256",""))
            if not op_hash: raise ValueError("Geppetto V2 requires surface.metadata.Nd_operator_sha256")
            norm=_normalization(np.asarray([n.P for n in surface.surface_nodes],np.float32)); ids,feat,pos=_surface_features(surface,norm); ids_all.append(ids); feat_all.append(feat); pos_all.append(pos); norms.append(norm); hashes.append(source_hash); op_hashes.append(op_hash); cond.append(_hash({"surface":source_hash,"Nd_operator_sha256":op_hash,"feature_contract":FEATURE_CONTRACT_V2,"ids":ids,"features":feat,"normalization":norm.__dict__}))
        features,valid=_pad(feat_all,24); positions,_=_pad(pos_all,3); return GeppettoConditioningBatchV2(tuple(ids_all),features,positions,valid,tuple(norms),tuple(hashes),tuple(op_hashes),tuple(cond),FEATURE_CONTRACT_V2)
