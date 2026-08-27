#!/usr/bin/env python3
"""Shared exact geometry utilities for RealSaS E0.

Semantics follow the frozen P-V5/R256 FIT scale-ladder apparatus: native 1024 raster authority,
align_corners=False pixel centers, yaw 0..315 in 45-degree steps, +Z screen-up, half-extent 0.54.
"""
from __future__ import annotations
import math
import numpy as np

VIEWS=8
NATIVE_RESOLUTION=1024
RENDER_HALF_EXTENT=0.54

def pixel_center_to_grid(pixel_xy,resolution:int):
    p=np.asarray(pixel_xy,np.float32)
    return (2.0*(p+0.5)/float(resolution)-1.0).astype(np.float32)

def grid_to_pixel_center(grid_xy,resolution:int):
    g=np.asarray(grid_xy,np.float32)
    return ((g+1.0)*(float(resolution)/2.0)-0.5).astype(np.float32)

def grid_distance_in_pixels(a,b,resolution:int):
    return np.linalg.norm(grid_to_pixel_center(a,resolution)-grid_to_pixel_center(b,resolution),axis=-1)

def camera_for_view(view:int,half_extent:float=RENDER_HALF_EXTENT)->dict:
    if not 0<=int(view)<VIEWS: raise ValueError(view)
    t=math.radians(float(view)*45.0); c,s=math.cos(t),math.sin(t)
    return {'view':int(view),'yaw_deg':float(view)*45.0,'right':np.asarray([c,-s,0.0],np.float32),'forward':np.asarray([s,c,0.0],np.float32),'up':np.asarray([0.0,0.0,1.0],np.float32),'half_extent':float(half_extent)}

def project_grid(points,camera):
    p=np.asarray(points,np.float32); right=np.asarray(camera['right'],np.float32); up=np.asarray(camera['up'],np.float32); he=float(camera['half_extent'])
    return np.stack([(p@right)/he,-(p@up)/he],axis=-1).astype(np.float32)

def grid_to_nearest_pixel(grid,resolution:int):
    g=np.asarray(grid,np.float32)
    x=np.floor((g[...,0]+1.0)*0.5*resolution).astype(np.int64); y=np.floor((g[...,1]+1.0)*0.5*resolution).astype(np.int64)
    return x,y

def geometric_vertex_normals(vertices,faces):
    v=np.asarray(vertices,np.float32); f=np.asarray(faces,np.int64); tri=v[f]
    fn=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]).astype(np.float32); out=np.zeros_like(v,dtype=np.float32)
    for j in range(3): np.add.at(out,f[:,j],fn)
    out/=np.maximum(np.linalg.norm(out,axis=1,keepdims=True),1e-8); return out

def bary_weights(uv):
    uv=np.asarray(uv,np.float32); return np.stack([uv[...,0],uv[...,1],1.0-uv[...,0]-uv[...,1]],axis=-1)

def reconstruct_surface(vertices,faces,vertex_normals,triangle_id,barycentric_uv):
    tri=np.asarray(triangle_id,np.int64); w=bary_weights(barycentric_uv)
    p=(vertices[faces[tri]]*w[...,None]).sum(axis=-2).astype(np.float32)
    n=(vertex_normals[faces[tri]]*w[...,None]).sum(axis=-2).astype(np.float32); n/=np.maximum(np.linalg.norm(n,axis=-1,keepdims=True),1e-8)
    return p,n

def raster_lookup_near(pixel_linear_index,query_x,query_y,resolution:int,radius:int):
    pix=np.asarray(pixel_linear_index,np.int64); qx=np.asarray(query_x,np.int64).reshape(-1); qy=np.asarray(query_y,np.int64).reshape(-1)
    if len(pix) and np.any(pix[1:]<pix[:-1]): raise ValueError('pixel_linear_index must be sorted')
    out=np.full((len(qx),(2*radius+1)**2),-1,np.int64); col=0
    for dy in range(-radius,radius+1):
        for dx in range(-radius,radius+1):
            xx=qx+dx; yy=qy+dy; valid=(xx>=0)&(xx<resolution)&(yy>=0)&(yy<resolution); target=yy*resolution+xx; pos=np.searchsorted(pix,target)
            if len(pix):
                pc=np.minimum(pos,len(pix)-1); hit=valid&(pos<len(pix))&(pix[pc]==target); out[hit,col]=pos[hit]
            col+=1
    return out

def _row_lookup(pix,targets):
    pos=np.searchsorted(pix,targets); out=np.full(targets.shape,-1,np.int64)
    if len(pix):
        pc=np.minimum(pos,len(pix)-1); hit=(pos<len(pix))&(pix[pc]==targets); out[hit]=pos[hit]
    return out

def derive_view_local_normals(pixel_linear_index,P,resolution:int,stride_px:int=2):
    """Normal from visible P only. Sign is non-authoritative; matching uses abs(dot)."""
    pix=np.asarray(pixel_linear_index,np.int64); P=np.asarray(P,np.float32)
    if len(pix)!=len(P): raise ValueError('pix/P length mismatch')
    if len(pix) and np.any(pix[1:]<pix[:-1]): raise ValueError('pixel ids must be sorted')
    y=pix//resolution; x=pix%resolution; cur=np.arange(len(pix),dtype=np.int64)
    def lookup(dx,dy):
        xx,yy=x+dx,y+dy; valid=(xx>=0)&(xx<resolution)&(yy>=0)&(yy<resolution); r=_row_lookup(pix,yy*resolution+xx); r[~valid]=-1; return r
    l=lookup(-stride_px,0); r=lookup(stride_px,0); u=lookup(0,-stride_px); d=lookup(0,stride_px)
    tx=np.zeros_like(P); ty=np.zeros_like(P); vx=np.zeros(len(P),bool); vy=np.zeros(len(P),bool)
    both=(l>=0)&(r>=0); tx[both]=P[r[both]]-P[l[both]]; vx|=both
    only=(~both)&(r>=0); tx[only]=P[r[only]]-P[cur[only]]; vx|=only
    only=(~both)&(l>=0); tx[only]=P[cur[only]]-P[l[only]]; vx|=only
    both=(u>=0)&(d>=0); ty[both]=P[d[both]]-P[u[both]]; vy|=both
    only=(~both)&(d>=0); ty[only]=P[d[only]]-P[cur[only]]; vy|=only
    only=(~both)&(u>=0); ty[only]=P[cur[only]]-P[u[only]]; vy|=only
    n=np.cross(tx,ty).astype(np.float32); nn=np.linalg.norm(n,axis=1); good=vx&vy&(nn>1e-9); n[good]/=nn[good,None]; n[~good]=0.0
    return n,good
