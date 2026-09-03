from __future__ import annotations

from dataclasses import replace
import math
import numpy as np

from .hashing import content_sha256
from .types import RiggingSurfaceIR, QualificationError

DTB_ND1_HISTORICAL_OPERATOR_BLOB_SHA = "4e612978c3f70a537dcf3189948dc19a38b6e582"
DTB_ND1_OPERATOR_ID = "RealSaS.DTB-ND1.RobustLocalPlane.v1"

_OFFSETS=[]
for _dy in (-6,-4,-2,0,2,4,6):
    for _dx in (-6,-4,-2,0,2,4,6):
        if _dx or _dy: _OFFSETS.append((_dx,_dy))
OFFSETS=np.asarray(_OFFSETS,np.int64)
SPATIAL_W=np.exp(-(OFFSETS[:,0]**2+OFFSETS[:,1]**2)/(2.0*3.0**2)).astype(np.float32)


def dtb_nd1_operator_identity()->dict:
    return {"operator_id":DTB_ND1_OPERATOR_ID,"historical_blob_sha":DTB_ND1_HISTORICAL_OPERATOR_BLOB_SHA,"offsets":OFFSETS.tolist(),"spatial_sigma_px":3.0,"minimum_retained_neighbors":6,"robust_gate":"dist<=median+3*max(1.4826*MAD,0.25*median,1e-6)"}


def dtb_nd1_operator_hash()->str: return content_sha256(dtb_nd1_operator_identity())


def _lookup_rows(pixel_linear_index,x,y,resolution:int):
    pix=np.asarray(pixel_linear_index,np.int64)
    if pix.ndim!=1 or (len(pix) and np.any(pix[1:]<pix[:-1])): raise ValueError("pixel_linear_index must be sorted rank-1")
    valid=(x>=0)&(x<resolution)&(y>=0)&(y<resolution); q=y*resolution+x; pos=np.searchsorted(pix,q); rows=np.full(q.shape,-1,np.int64)
    if len(pix):
        pc=np.minimum(pos,len(pix)-1); hit=valid&(pos<len(pix))&(pix[pc]==q); rows[hit]=pos[hit]
    return rows,valid


def robust_local_plane_normals_for_rows(pixel_linear_index,P,resolution:int,rows,*,chunk:int=8192):
    pix=np.asarray(pixel_linear_index,np.int64); points=np.asarray(P,np.float32); rows=np.asarray(rows,np.int64).reshape(-1)
    if resolution<=0: raise ValueError("resolution must be positive")
    if points.ndim!=2 or points.shape[1]!=3 or len(pix)!=len(points): raise ValueError("pixel index / P shape mismatch")
    if np.any(rows<0) or np.any(rows>=len(points)): raise ValueError("row out of range")
    if not np.all(np.isfinite(points)): raise ValueError("P must be finite")
    out=np.zeros((len(rows),3),np.float32); good_out=np.zeros(len(rows),bool); cnt_out=np.zeros(len(rows),np.int16)
    for st in range(0,len(rows),int(chunk)):
        rr=rows[st:st+int(chunk)]; cp=points[rr]; pids=pix[rr]; cy=pids//resolution; cx=pids%resolution
        nx=cx[:,None]+OFFSETS[None,:,0]; ny=cy[:,None]+OFFSETS[None,:,1]; nrows,_=_lookup_rows(pix,nx,ny,resolution); valid=nrows>=0
        clip=np.clip(nrows,0,max(len(points)-1,0)); neigh=points[clip]; diff=neigh-cp[:,None,:]; dist=np.linalg.norm(diff,axis=2); nd=np.where(valid,dist,np.nan)
        with np.errstate(all="ignore"):
            med=np.nanmedian(nd,axis=1); mad=np.nanmedian(np.abs(nd-med[:,None]),axis=1)
        med=np.where(np.isfinite(med),med,0.0); mad=np.where(np.isfinite(mad),mad,0.0); sigma=np.maximum.reduce([1.4826*mad,0.25*med,np.full_like(med,1e-6)])
        gate=valid&(dist<=med[:,None]+3.0*sigma[:,None]); count=gate.sum(axis=1); w=gate.astype(np.float32)*SPATIAL_W[None,:]; sw=1.0+w.sum(axis=1)
        mu=(cp+np.sum(w[:,:,None]*neigh,axis=1))/sw[:,None]; dc=cp-mu; dn=neigh-mu[:,None,:]; C=np.empty((len(rr),3,3),np.float32)
        C[:,0,0]=(dc[:,0]*dc[:,0]+np.sum(w*dn[:,:,0]*dn[:,:,0],axis=1))/sw; C[:,0,1]=(dc[:,0]*dc[:,1]+np.sum(w*dn[:,:,0]*dn[:,:,1],axis=1))/sw; C[:,0,2]=(dc[:,0]*dc[:,2]+np.sum(w*dn[:,:,0]*dn[:,:,2],axis=1))/sw
        C[:,1,1]=(dc[:,1]*dc[:,1]+np.sum(w*dn[:,:,1]*dn[:,:,1],axis=1))/sw; C[:,1,2]=(dc[:,1]*dc[:,2]+np.sum(w*dn[:,:,1]*dn[:,:,2],axis=1))/sw; C[:,2,2]=(dc[:,2]*dc[:,2]+np.sum(w*dn[:,:,2]*dn[:,:,2],axis=1))/sw
        C[:,1,0]=C[:,0,1]; C[:,2,0]=C[:,0,2]; C[:,2,1]=C[:,1,2]
        evals,evecs=np.linalg.eigh(C.astype(np.float64)); n=evecs[:,:,0].astype(np.float32); finite=np.all(np.isfinite(n),axis=1)&np.all(np.isfinite(evals),axis=1); ok=(count>=6)&finite&(evals[:,1]>1e-14); n[~ok]=0.0
        out[st:st+len(rr)]=n; good_out[st:st+len(rr)]=ok; cnt_out[st:st+len(rr)]=count.astype(np.int16)
    return out,good_out,cnt_out


def robust_local_plane_normals(pixel_linear_index,P,resolution:int):
    n,v,_=robust_local_plane_normals_for_rows(pixel_linear_index,P,resolution,np.arange(len(P),dtype=np.int64)); return n,v


def _unit(v):
    a=np.asarray(v,np.float32); n=float(np.linalg.norm(a))
    if not math.isfinite(n) or n<=1e-12: raise QualificationError("invalid normal/ray vector")
    return (a/n).astype(np.float32)


def orient_normal_against_ray(normal,ray_forward):
    n=_unit(normal); f=_unit(ray_forward); return -n if float(np.dot(n,f))>0 else n


def attach_dtb_nd1_normals(surface:RiggingSurfaceIR,normals_by_surface_id:dict[str,tuple[float,float,float]|np.ndarray],*,diagnostics_by_surface_id:dict[str,dict]|None=None)->RiggingSurfaceIR:
    known={n.surface_id for n in surface.surface_nodes}; extra=set(normals_by_surface_id)-known
    if extra: raise QualificationError(f"local geometry references unknown surface ids:{sorted(extra)[:3]}")
    diagnostics_by_surface_id=diagnostics_by_surface_id or {}; nodes=[]
    for node in surface.surface_nodes:
        normal=normals_by_surface_id.get(node.surface_id)
        if normal is None: nodes.append(node); continue
        arr=_unit(normal); metadata=dict(node.metadata); metadata.update({"derived_normal_operator_id":DTB_ND1_OPERATOR_ID,"derived_normal_operator_hash":dtb_nd1_operator_hash(),"derived_normal_historical_blob_sha":DTB_ND1_HISTORICAL_OPERATOR_BLOB_SHA})
        if node.surface_id in diagnostics_by_surface_id: metadata["derived_normal_diagnostics"]=diagnostics_by_surface_id[node.surface_id]
        nodes.append(replace(node,derived_normal=tuple(map(float,arr)),metadata=metadata))
    metadata=dict(surface.metadata); metadata.update({"derived_local_geometry_operator":DTB_ND1_OPERATOR_ID,"Nd_operator_sha256":dtb_nd1_operator_hash(),"Nd_historical_blob_sha":DTB_ND1_HISTORICAL_OPERATOR_BLOB_SHA})
    lineage=content_sha256({"base_geometry_lineage_hash":surface.geometry_lineage_hash,"operator":dtb_nd1_operator_identity(),"nodes":[n.to_dict() for n in nodes]})
    return replace(surface,surface_nodes=tuple(nodes),geometry_lineage_hash=lineage,metadata=metadata)
