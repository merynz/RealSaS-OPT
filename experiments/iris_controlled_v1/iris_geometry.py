from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Dict, Tuple
import numpy as np

def sha256_file(path: str | Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''): h.update(block)
    return h.hexdigest()

def geometric_vertex_normals(vertices, faces):
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

def pixel_to_grid(pixel_linear_index,resolution):
    pix=np.asarray(pixel_linear_index,np.int64); y=pix//resolution; x=pix%resolution
    gx=2.0*(x.astype(np.float32)+.5)/float(resolution)-1.0; gy=2.0*(y.astype(np.float32)+.5)/float(resolution)-1.0
    return np.stack([gx,gy],axis=-1).astype(np.float32)

def project_grid(points,camera:Dict):
    p=np.asarray(points,np.float32); right=np.asarray(camera['right'],np.float32); up=np.asarray(camera['up'],np.float32); he=float(camera['half_extent'])
    return np.stack([(p@right)/he,-(p@up)/he],axis=-1).astype(np.float32)

def grid_to_nearest_pixel(grid,resolution):
    g=np.asarray(grid,np.float32); return np.floor((g[...,0]+1)*.5*resolution).astype(np.int64),np.floor((g[...,1]+1)*.5*resolution).astype(np.int64)

def raster_lookup_near(pixel_linear_index,query_x,query_y,resolution,radius=2):
    pix=np.asarray(pixel_linear_index,np.int64); qx=np.asarray(query_x,np.int64).reshape(-1); qy=np.asarray(query_y,np.int64).reshape(-1); n=len(qx); width=(2*radius+1)**2
    out=np.full((n,width),-1,np.int64); d2=np.full((n,width),np.inf,np.float32); col=0
    for dy in range(-radius,radius+1):
        for dx in range(-radius,radius+1):
            xx=qx+dx; yy=qy+dy; valid=(xx>=0)&(xx<resolution)&(yy>=0)&(yy<resolution); target=yy*resolution+xx; pos=np.searchsorted(pix,target); pos_clip=np.minimum(pos,max(len(pix)-1,0)); hit=valid&(pos<len(pix))
            if len(pix): hit &= pix[pos_clip]==target
            out[hit,col]=pos[hit]; d2[hit,col]=float(dx*dx+dy*dy); col+=1
    return out,d2

def choose_visible_correspondence(points,target_camera,target_ra,vertices,faces,vertex_normals,radius_px=2,max_surface_error=.0025):
    points=np.asarray(points,np.float32); res=int(np.asarray(target_ra['resolution']).reshape(-1)[0]); grid=project_grid(points,target_camera); x,y=grid_to_nearest_pixel(grid,res); in_frame=(np.abs(grid[:,0])<=1)&(np.abs(grid[:,1])<=1); cand,_=raster_lookup_near(target_ra['pixel_linear_index'],x,y,res,radius_px)
    best_row=np.full(len(points),-1,np.int64); best_err=np.full(len(points),np.inf,np.float32)
    for j in range(cand.shape[1]):
        row=cand[:,j]; m=in_frame&(row>=0)
        if not np.any(m): continue
        pp,_=reconstruct_surface(vertices,faces,vertex_normals,target_ra['triangle_id'][row[m]],target_ra['barycentric_uv'][row[m]]); err=np.linalg.norm(pp-points[m],axis=1); ids=np.flatnonzero(m); better=err<best_err[ids]; best_err[ids[better]]=err[better]; best_row[ids[better]]=row[m][better]
    ok=best_err<=float(max_surface_error); target_grid=np.zeros((len(points),2),np.float32)
    if np.any(ok): target_grid[ok]=pixel_to_grid(target_ra['pixel_linear_index'][best_row[ok]],res)
    return ok,target_grid,best_err,best_row
