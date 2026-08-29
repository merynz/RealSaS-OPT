from __future__ import annotations
from dataclasses import dataclass
import hashlib, math
import numpy as np
from scipy import ndimage

@dataclass(frozen=True)
class ObservableForVolume:
    view:int
    half_extent:float
    resolution:int
    pixel_linear_index:np.ndarray
    P:np.ndarray

@dataclass(frozen=True)
class ClosedRiggingVolumeV0:
    asset_id:str; resolution:int
    bounds_min:tuple[float,float,float]; bounds_max:tuple[float,float,float]
    possible_interior:np.ndarray; strict_hull:np.ndarray; certain_outside:np.ndarray
    supported_surface:np.ndarray; high_confidence_interior:np.ndarray; unknown_concavity:np.ndarray
    voxel_size:float; silhouette_expansion_px:int; lineage_sha256:str

@dataclass(frozen=True)
class InteriorRiggingSubstrateV0:
    asset_id:str; distance_to_possible_boundary:np.ndarray; distance_to_strict_boundary:np.ndarray
    medialness:np.ndarray; local_thickness:np.ndarray; branch_likelihood:np.ndarray
    lineage_sha256:str

def _camera(view:int):
    th=math.radians(view*45.0); c,s=math.cos(th),math.sin(th)
    return np.array([c,-s,0],np.float32),np.array([0,0,1],np.float32)

def _mask_from_pixels(pix,R):
    m=np.zeros((R,R),bool); pix=np.asarray(pix,np.int64); y=pix//R; x=pix%R; m[y,x]=True; return m

def _project_pixels(P,view,he,R):
    right,up=_camera(view); gx=(P@right)/he; gy=-(P@up)/he
    x=np.floor((gx+1)*.5*R).astype(np.int64); y=np.floor((gy+1)*.5*R).astype(np.int64)
    return x,y

def _lookup(mask,x,y):
    R=mask.shape[0]; ok=(x>=0)&(x<R)&(y>=0)&(y<R); out=np.zeros(len(x),bool); out[ok]=mask[y[ok],x[ok]]; return out

def _voxelize_points(P,b0,b1,R):
    q=np.rint((np.asarray(P)-b0)/(b1-b0)*(R-1)).astype(np.int64); ok=np.all((q>=0)&(q<R),axis=1)
    m=np.zeros((R,R,R),bool); q=q[ok]; m[q[:,0],q[:,1],q[:,2]]=True; return m

def build_closed_volume_v0(asset_id:str, observations:list[ObservableForVolume], resolution:int=64):
    if sorted(o.view for o in observations)!=list(range(8)): raise ValueError('need exactly V0..V7')
    hs=np.array([o.half_extent for o in sorted(observations,key=lambda o:o.view)],float)
    h=float(hs.max())*1.02; b0=np.array([-h,-h,-h],np.float32); b1=-b0
    axes=[np.linspace(b0[i],b1[i],resolution,dtype=np.float32) for i in range(3)]
    X,Y,Z=np.meshgrid(*axes,indexing='ij'); Q=np.stack([X.ravel(),Y.ravel(),Z.ravel()],1)
    vox=float((b1[0]-b0[0])/(resolution-1))
    # Half projected voxel diagonal converted to native pixels; conservative cell-footprint support.
    max_expand=0
    strict=np.ones(len(Q),bool); possible=np.ones(len(Q),bool)
    for o in sorted(observations,key=lambda o:o.view):
        if o.resolution<=0: raise ValueError('bad resolution')
        m=_mask_from_pixels(o.pixel_linear_index,o.resolution)
        projected_half_diag=math.sqrt(2.0)*(vox/2.0)/float(o.half_extent)*(o.resolution/2.0)
        rad=max(1,int(math.ceil(projected_half_diag))); max_expand=max(max_expand,rad)
        md=ndimage.binary_dilation(m,iterations=rad)
        x,y=_project_pixels(Q,o.view,o.half_extent,o.resolution)
        strict &= _lookup(m,x,y); possible &= _lookup(md,x,y)
    strict=strict.reshape((resolution,)*3); possible=possible.reshape((resolution,)*3)
    # Observed P is a typed surface witness; it is the only source of SUPPORTED_SURFACE.
    surf=np.zeros_like(possible)
    for o in observations: surf |= _voxelize_points(o.P,b0,b1,resolution)
    surf=ndimage.binary_dilation(surf,iterations=1)&possible
    dstrict=ndimage.distance_transform_edt(strict)
    high=strict & (dstrict>=2.0)
    outside=~possible
    unknown=possible & ~surf
    hsh=hashlib.sha256();
    for a in (possible,strict,surf,high): hsh.update(np.packbits(a.reshape(-1)).tobytes())
    hsh.update(np.asarray([*b0,*b1,vox,max_expand],np.float32).tobytes())
    return ClosedRiggingVolumeV0(asset_id,resolution,tuple(map(float,b0)),tuple(map(float,b1)),possible,strict,outside,surf,high,unknown,vox,max_expand,hsh.hexdigest())

def build_interior_substrate_v0(vol:ClosedRiggingVolumeV0):
    dp=ndimage.distance_transform_edt(vol.possible_interior).astype(np.float32)*vol.voxel_size
    ds=ndimage.distance_transform_edt(vol.strict_hull).astype(np.float32)*vol.voxel_size
    # Conservative interior radius: strict where available, otherwise possible hull radius with uncertainty implicit.
    base=np.where(vol.strict_hull,ds,0.5*dp).astype(np.float32)
    mx=ndimage.maximum_filter(base,size=3,mode='constant'); ridge=(base>0)&(base>=mx-1e-7)
    rb=ndimage.binary_dilation(ridge,iterations=1)&vol.possible_interior
    med=np.zeros_like(base); med[rb]=base[rb]/max(float(base.max()),1e-8)
    thick=2*base
    neigh=ndimage.convolve(ridge.astype(np.int16),np.ones((3,3,3),np.int16),mode='constant')-ridge.astype(np.int16)
    branch=np.zeros_like(base); branch[ridge]=np.clip((neigh[ridge]-2)/4,0,1)
    h=hashlib.sha256(); h.update(vol.lineage_sha256.encode());
    for a in (dp,ds,med,thick,branch): h.update(a.tobytes())
    return InteriorRiggingSubstrateV0(vol.asset_id,dp,ds,med,thick,branch,h.hexdigest())
