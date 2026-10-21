from __future__ import annotations

"""Exact triangle geometry helpers for H-GEO-P1."""

from dataclasses import dataclass
import hashlib
import heapq
from typing import Iterable
import numpy as np

SUBJECT_IDS = ("Snake", "Bat", "T-Rex", "Horse", "Spider", "GreenBlob", "Shark", "George")
SUBJECT_SOURCE_IDS = {"Snake":"fit8_snake_v1","Bat":"fit8_bat_v1","T-Rex":"fit8_trex_v1","Horse":"fit8_horse_v1","Spider":"fit8_spider_v1","GreenBlob":"fit8_greenblob_v1","Shark":"fit8_shark_v1","George":"fit8_george_v1"}


def deterministic_seed_v1(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16) % (2**63 - 1)


def triangle_geometry_v1(vertices: np.ndarray, faces: np.ndarray):
    vertices=np.asarray(vertices,dtype=np.float64); faces=np.asarray(faces,dtype=np.int64)
    if vertices.ndim!=2 or vertices.shape[1]!=3 or faces.ndim!=2 or faces.shape[1]!=3: raise ValueError("H_GEO_P1_TRIANGLE_SHAPE_INVALID")
    tri=vertices[faces]; e1=tri[:,1]-tri[:,0]; e2=tri[:,2]-tri[:,0]; cross=np.cross(e1,e2); double_area=np.linalg.norm(cross,axis=1)
    if not np.isfinite(tri).all() or np.any(double_area<=1e-12): raise ValueError("H_GEO_P1_DEGENERATE_OR_NONFINITE_TRIANGLE")
    return tri,0.5*double_area,cross/double_area[:,None]


def sample_surface_exact_replay_v1(vertices: np.ndarray, faces: np.ndarray, count: int, *, seed_text: str):
    tri,area,normals=triangle_geometry_v1(vertices,faces); rng=np.random.default_rng(deterministic_seed_v1(seed_text)); fi=rng.choice(len(faces),size=int(count),replace=True,p=area/area.sum()); r1=rng.random(int(count)); r2=rng.random(int(count)); s=np.sqrt(r1); bary=np.stack((1-s,s*(1-r2),s*r2),axis=1); points=np.einsum("ni,nij->nj",bary,tri[fi]); return points.astype(np.float64),fi.astype(np.int64),bary,normals


@dataclass(frozen=True)
class RecoveredNormalizationV1:
    scale: float
    translation: tuple[float,float,float]
    center_source_world: tuple[float,float,float]
    half_extent_source_world: float
    maximum_replay_error_iris: float
    def apply(self,value:np.ndarray)->np.ndarray: return np.asarray(value,dtype=np.float64)*float(self.scale)+np.asarray(self.translation,dtype=np.float64)


def recover_prepared_normalization_v1(*,subject_id:str,rest_vertices_source_world:np.ndarray,faces:np.ndarray,stored_surface_points_iris:np.ndarray,maximum_error_iris:float=1e-6)->RecoveredNormalizationV1:
    stored=np.asarray(stored_surface_points_iris,dtype=np.float64)
    if stored.shape!=(4096,3): raise ValueError("H_GEO_P1_STORED_SURFACE_REPLAY_SHAPE_INVALID:"+subject_id)
    source,_,_,_=sample_surface_exact_replay_v1(rest_vertices_source_world,faces,len(stored),seed_text=subject_id); xb=source.mean(0); yb=stored.mean(0); cx=source-xb; cy=stored-yb; denom=float(np.sum(cx*cx))
    if not np.isfinite(denom) or denom<=1e-16: raise ValueError("H_GEO_P1_NORMALIZATION_REPLAY_DEGENERATE:"+subject_id)
    scale=float(np.sum(cx*cy)/denom); translation=yb-scale*xb
    if not np.isfinite(scale) or scale<=0 or not np.isfinite(translation).all(): raise ValueError("H_GEO_P1_NORMALIZATION_REPLAY_INVALID:"+subject_id)
    error=float(np.max(np.abs(source*scale+translation-stored)))
    if error>float(maximum_error_iris): raise ValueError(f"H_GEO_P1_NORMALIZATION_REPLAY_ERROR:{subject_id}:{error}")
    half=1.0/scale; center=-translation/scale; normalized=np.asarray(rest_vertices_source_world,dtype=np.float64)*scale+translation
    if float(np.max(np.abs(normalized)))>1.00001: raise ValueError("H_GEO_P1_NORMALIZED_TEACHER_OUTSIDE_IRIS_DOMAIN:"+subject_id)
    return RecoveredNormalizationV1(scale,tuple(float(x) for x in translation),tuple(float(x) for x in center),float(half),error)


def _closest_points_to_triangles(p:np.ndarray,tri:np.ndarray)->np.ndarray:
    p=np.asarray(p,dtype=np.float64).reshape(3); tri=np.asarray(tri,dtype=np.float64); a,b,c=tri[:,0],tri[:,1],tri[:,2]; ab,ac=b-a,c-a; ap=p[None]-a; d1=np.einsum("ij,ij->i",ab,ap); d2=np.einsum("ij,ij->i",ac,ap); bp=p[None]-b; d3=np.einsum("ij,ij->i",ab,bp); d4=np.einsum("ij,ij->i",ac,bp); cp=p[None]-c; d5=np.einsum("ij,ij->i",ab,cp); d6=np.einsum("ij,ij->i",ac,cp); q=np.empty_like(a); assigned=np.zeros(len(tri),dtype=bool)
    m=(d1<=0)&(d2<=0); q[m]=a[m]; assigned|=m
    m=(~assigned)&(d3>=0)&(d4<=d3); q[m]=b[m]; assigned|=m
    vc=d1*d4-d3*d2; m=(~assigned)&(vc<=0)&(d1>=0)&(d3<=0); den=d1-d3; v=np.zeros_like(d1); v[m]=d1[m]/den[m]; q[m]=a[m]+v[m,None]*ab[m]; assigned|=m
    m=(~assigned)&(d6>=0)&(d5<=d6); q[m]=c[m]; assigned|=m
    vb=d5*d2-d1*d6; m=(~assigned)&(vb<=0)&(d2>=0)&(d6<=0); den=d2-d6; w=np.zeros_like(d2); w[m]=d2[m]/den[m]; q[m]=a[m]+w[m,None]*ac[m]; assigned|=m
    va=d3*d6-d5*d4; m=(~assigned)&(va<=0)&((d4-d3)>=0)&((d5-d6)>=0); den=(d4-d3)+(d5-d6); wbc=np.zeros_like(d3); wbc[m]=(d4[m]-d3[m])/den[m]; q[m]=b[m]+wbc[m,None]*(c[m]-b[m]); assigned|=m
    m=~assigned; den=va+vb+vc; inv=np.zeros_like(den); inv[m]=1.0/den[m]; vf=vb*inv; wf=vc*inv; q[m]=a[m]+ab[m]*vf[m,None]+ac[m]*wf[m,None]
    if not np.isfinite(q).all(): raise RuntimeError("H_GEO_P1_CLOSEST_POINT_NONFINITE")
    return q


def _bbox_distance_sq(point,bmin,bmax):
    d=np.maximum(np.maximum(bmin-point,0.0),point-bmax); return float(np.dot(d,d))


class ExactTriangleBVHV1:
    def __init__(self,vertices,faces,leaf_size:int=16):
        self.vertices=np.asarray(vertices,dtype=np.float64); self.faces=np.asarray(faces,dtype=np.int64); self.tri,self.area,self.normals=triangle_geometry_v1(self.vertices,self.faces); self.centroids=self.tri.mean(1); self.tri_min=self.tri.min(1); self.tri_max=self.tri.max(1); self.leaf_size=int(leaf_size); self.nodes=[]; self.root=self._build(np.arange(len(self.faces),dtype=np.int64))
    def _build(self,indices):
        bmin=self.tri_min[indices].min(0); bmax=self.tri_max[indices].max(0); ni=len(self.nodes); node={"bmin":bmin,"bmax":bmax,"left":-1,"right":-1,"indices":None}; self.nodes.append(node)
        if len(indices)<=self.leaf_size: node["indices"]=indices; return ni
        extent=self.centroids[indices].max(0)-self.centroids[indices].min(0); axis=int(np.argmax(extent)); order=indices[np.argsort(self.centroids[indices,axis],kind="mergesort")]; middle=len(order)//2
        if middle<=0 or middle>=len(order): node["indices"]=order; return ni
        node["left"]=self._build(order[:middle]); node["right"]=self._build(order[middle:]); return ni
    def nearest_one(self,point):
        point=np.asarray(point,dtype=np.float64).reshape(3); best=float("inf"); best_point=None; best_face=-1; queue=[(0.0,self.root)]
        while queue:
            lower,ni=heapq.heappop(queue)
            if lower>=best: break
            node=self.nodes[ni]; indices=node["indices"]
            if indices is not None:
                q=_closest_points_to_triangles(point,self.tri[indices]); diff=q-point[None]; dsq=np.einsum("ij,ij->i",diff,diff); j=int(np.argmin(dsq)); value=float(dsq[j])
                if value<best: best=value; best_point=q[j]; best_face=int(indices[j])
                continue
            for child in (int(node["left"]),int(node["right"])):
                c=self.nodes[child]; d=_bbox_distance_sq(point,c["bmin"],c["bmax"])
                if d<best: heapq.heappush(queue,(d,child))
        if best_point is None: raise RuntimeError("H_GEO_P1_BVH_NEAREST_FAILED")
        return best_point,best_face,best
    def nearest_many(self,points,*,progress_every:int=4096):
        points=np.asarray(points,dtype=np.float64); closest=np.empty((len(points),3)); face=np.empty(len(points),dtype=np.int64); distance=np.empty(len(points))
        for i,p in enumerate(points):
            q,fi,dsq=self.nearest_one(p); closest[i]=q; face[i]=fi; distance[i]=np.sqrt(max(0.0,dsq))
            if progress_every and (i+1)%int(progress_every)==0: print(f"H_GEO_P1_EXACT_TRIANGLE_QUERY {i+1}/{len(points)}",flush=True)
        return closest,face,distance


def make_query_split_v1(*,vertices_iris,faces,seed:int,near_count:int,uniform_count:int,offsets_iris:Iterable[float],near_threshold_iris:float=0.03,normal_valid_distance_iris:float=0.08):
    tri,area,normals=triangle_geometry_v1(vertices_iris,faces); rng=np.random.default_rng(int(seed)); offsets=np.asarray(tuple(float(x) for x in offsets_iris),dtype=np.float64); queries=[]
    while sum(len(x) for x in queries)<int(near_count):
        remaining=int(near_count)-sum(len(x) for x in queries); draw=max(remaining*2,1024); fi=rng.choice(len(faces),size=draw,replace=True,p=area/area.sum()); r1=rng.random(draw); r2=rng.random(draw); s=np.sqrt(r1); bary=np.stack((1-s,s*(1-r2),s*r2),1); surface=np.einsum("ni,nij->nj",bary,tri[fi]); off=offsets[rng.integers(0,len(offsets),size=draw)]; candidate=surface+normals[fi]*off[:,None]; valid=np.max(np.abs(candidate),1)<=1.0
        if bool(valid.any()): queries.append(candidate[valid][:remaining])
    near=np.concatenate(queries,0)[:int(near_count)]; uniform=rng.uniform(-1,1,size=(int(uniform_count),3)); query=np.concatenate((near,uniform),0); query=query[rng.permutation(len(query))]; bvh=ExactTriangleBVHV1(vertices_iris,faces); closest,face,distance=bvh.nearest_many(query); offset=closest-query; normal=bvh.normals[face]
    return {"xyz_iris":query.astype(np.float32),"closest_surface_iris":closest.astype(np.float32),"distance_iris":distance.astype(np.float32),"offset_to_surface_iris":offset.astype(np.float32),"normal_unit":normal.astype(np.float32),"normal_valid":(distance<=float(normal_valid_distance_iris)).astype(np.uint8),"near_label":(distance<=float(near_threshold_iris)).astype(np.uint8),"closest_face_index":face.astype(np.int32)}
