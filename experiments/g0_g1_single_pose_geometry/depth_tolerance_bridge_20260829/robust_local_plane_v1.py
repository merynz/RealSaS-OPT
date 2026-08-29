from __future__ import annotations
import numpy as np

_OFFSETS=[]
for dy in (-6,-4,-2,0,2,4,6):
    for dx in (-6,-4,-2,0,2,4,6):
        if dx==0 and dy==0:
            continue
        _OFFSETS.append((dx,dy))
OFFSETS=np.asarray(_OFFSETS,np.int64)
SPATIAL_W=np.exp(-(OFFSETS[:,0]**2+OFFSETS[:,1]**2)/(2.0*3.0**2)).astype(np.float32)


def _lookup_rows(pixel_linear_index: np.ndarray, x: np.ndarray, y: np.ndarray, resolution: int) -> tuple[np.ndarray,np.ndarray]:
    pix=np.asarray(pixel_linear_index,np.int64)
    valid=(x>=0)&(x<resolution)&(y>=0)&(y<resolution)
    q=y*resolution+x
    pos=np.searchsorted(pix,q)
    rows=np.full(q.shape,-1,np.int64)
    if len(pix):
        pc=np.minimum(pos,len(pix)-1)
        hit=valid&(pos<len(pix))&(pix[pc]==q)
        rows[hit]=pos[hit]
    return rows,valid


def robust_local_plane_normals_for_rows(pixel_linear_index, P, resolution: int, rows, *, chunk: int=8192):
    """Frozen DTB-ND1 robust local-plane operator for selected raster rows.

    Returns (N, valid, retained_neighbor_count), preserving input row order.
    """
    pix=np.asarray(pixel_linear_index,np.int64)
    P=np.asarray(P,np.float32)
    rows=np.asarray(rows,np.int64).reshape(-1)
    if np.any(rows<0) or np.any(rows>=len(P)):
        raise ValueError('row out of range')
    out=np.zeros((len(rows),3),np.float32)
    good_out=np.zeros(len(rows),bool)
    cnt_out=np.zeros(len(rows),np.int16)
    for st in range(0,len(rows),chunk):
        rr=rows[st:st+chunk]
        cp=P[rr]
        pids=pix[rr]
        cy=pids//resolution; cx=pids%resolution
        nx=cx[:,None]+OFFSETS[None,:,0]
        ny=cy[:,None]+OFFSETS[None,:,1]
        nrows,_=_lookup_rows(pix,nx,ny,resolution)
        valid=nrows>=0
        clip=np.clip(nrows,0,max(len(P)-1,0))
        neigh=P[clip]
        diff=neigh-cp[:,None,:]
        dist=np.linalg.norm(diff,axis=2)
        nd=np.where(valid,dist,np.nan)
        with np.errstate(all='ignore'):
            med=np.nanmedian(nd,axis=1)
            mad=np.nanmedian(np.abs(nd-med[:,None]),axis=1)
        med=np.where(np.isfinite(med),med,0.0)
        mad=np.where(np.isfinite(mad),mad,0.0)
        sigma=np.maximum.reduce([1.4826*mad,0.25*med,np.full_like(med,1e-6)])
        gate=valid&(dist <= med[:,None] + 3.0*sigma[:,None])
        count=gate.sum(axis=1)
        w=gate.astype(np.float32)*SPATIAL_W[None,:]
        sw=1.0+w.sum(axis=1)
        mu=(cp + np.sum(w[:,:,None]*neigh,axis=1))/sw[:,None]
        dc=cp-mu
        dn=neigh-mu[:,None,:]
        C=np.empty((len(rr),3,3),np.float32)
        C[:,0,0]=(dc[:,0]*dc[:,0] + np.sum(w*dn[:,:,0]*dn[:,:,0],axis=1))/sw
        C[:,0,1]=(dc[:,0]*dc[:,1] + np.sum(w*dn[:,:,0]*dn[:,:,1],axis=1))/sw
        C[:,0,2]=(dc[:,0]*dc[:,2] + np.sum(w*dn[:,:,0]*dn[:,:,2],axis=1))/sw
        C[:,1,1]=(dc[:,1]*dc[:,1] + np.sum(w*dn[:,:,1]*dn[:,:,1],axis=1))/sw
        C[:,1,2]=(dc[:,1]*dc[:,2] + np.sum(w*dn[:,:,1]*dn[:,:,2],axis=1))/sw
        C[:,2,2]=(dc[:,2]*dc[:,2] + np.sum(w*dn[:,:,2]*dn[:,:,2],axis=1))/sw
        C[:,1,0]=C[:,0,1]; C[:,2,0]=C[:,0,2]; C[:,2,1]=C[:,1,2]
        evals,evecs=np.linalg.eigh(C.astype(np.float64))
        n=evecs[:,:,0].astype(np.float32)
        finite=np.all(np.isfinite(n),axis=1)&np.all(np.isfinite(evals),axis=1)
        ok=(count>=6)&finite&(evals[:,1]>1e-14)
        n[~ok]=0.0
        out[st:st+len(rr)]=n
        good_out[st:st+len(rr)]=ok
        cnt_out[st:st+len(rr)]=count.astype(np.int16)
    return out,good_out,cnt_out


def robust_local_plane_normals(pixel_linear_index, P, resolution: int):
    rows=np.arange(len(P),dtype=np.int64)
    n,v,_=robust_local_plane_normals_for_rows(pixel_linear_index,P,resolution,rows)
    return n,v
