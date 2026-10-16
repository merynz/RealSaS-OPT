from __future__ import annotations
from dataclasses import asdict, dataclass, field
import math
from typing import Any
import numpy as np
Json=dict[str,Any]

@dataclass(frozen=True)
class MechanicalDemandTruthPolicyV1:
    rigid_edge_strain_tolerance:float=0.02; rigid_area_change_tolerance:float=0.03; rigid_bend_radians_tolerance:float=math.radians(3.0); epsilon:float=1e-10
    def validate(self):
        if self.epsilon<=0 or min(self.rigid_edge_strain_tolerance,self.rigid_area_change_tolerance,self.rigid_bend_radians_tolerance)<=0: raise ValueError("MECHANICAL_TRUTH_POLICY_INVALID")

@dataclass(frozen=True)
class MechanicalDemandTruthV1:
    stretch_major:tuple[float,...]; stretch_minor:tuple[float,...]; tangent_direction_2theta:tuple[tuple[float,float],...]; bend_demand:tuple[float,...]; rigidity_probability:tuple[float,...]; area_change_p95:tuple[float,...]; condition_number_p95:tuple[float,...]; edge_strain_p95:tuple[float,...]; contact_persistence:tuple[float,...]; contact_persistence_valid:bool; face_count:int; frame_count:int; schema_version:str="RealSaS.MechanicalDemandTruth.v1"; metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)

def _validate(rest_vertices,faces,posed_vertices):
    r=np.asarray(rest_vertices,dtype=np.float64); f=np.asarray(faces,dtype=np.int64); p=np.asarray(posed_vertices,dtype=np.float64)
    if r.ndim!=2 or r.shape[1]!=3 or not np.isfinite(r).all(): raise ValueError("MECHANICAL_TRUTH_REST_INVALID")
    if f.ndim!=2 or f.shape[1]!=3 or len(f)==0 or np.any(f<0) or np.any(f>=len(r)): raise ValueError("MECHANICAL_TRUTH_FACES_INVALID")
    if p.ndim!=3 or p.shape[1:]!=r.shape or p.shape[0]<1 or not np.isfinite(p).all(): raise ValueError("MECHANICAL_TRUTH_POSES_INVALID")
    return r,f,p

def _basis(vertices,faces,eps):
    a,b,c=vertices[faces[:,0]],vertices[faces[:,1]],vertices[faces[:,2]]; e1,e2=b-a,c-a; l1=np.linalg.norm(e1,axis=1); n=np.cross(e1,e2); nn=np.linalg.norm(n,axis=1)
    if np.any(l1<=eps) or np.any(nn<=eps): raise ValueError("MECHANICAL_TRUTH_DEGENERATE_FACE")
    u=e1/l1[:,None]; normal=n/nn[:,None]; v=np.cross(normal,u); x2=np.sum(e2*u,axis=1); y2=np.sum(e2*v,axis=1)
    if np.any(np.abs(y2)<=eps): raise ValueError("MECHANICAL_TRUTH_SINGULAR_BASIS")
    dm=np.zeros((len(faces),2,2)); dm[:,0,0]=l1; dm[:,0,1]=x2; dm[:,1,1]=y2
    return np.linalg.inv(dm),0.5*nn,normal

def _frame(posed,faces,inv_dm,rest_area,eps):
    a,b,c=posed[faces[:,0]],posed[faces[:,1]],posed[faces[:,2]]; e1,e2=b-a,c-a; l1=np.linalg.norm(e1,axis=1); n=np.cross(e1,e2); nn=np.linalg.norm(n,axis=1)
    if np.any(l1<=eps) or np.any(nn<=eps): raise ValueError("MECHANICAL_TRUTH_DEGENERATE_POSE")
    u=e1/l1[:,None]; normal=n/nn[:,None]; v=np.cross(normal,u); ds=np.zeros((len(faces),2,2)); ds[:,0,0]=l1; ds[:,0,1]=np.sum(e2*u,axis=1); ds[:,1,1]=np.sum(e2*v,axis=1); f2=np.einsum("fij,fjk->fik",ds,inv_dm); _,s,vh=np.linalg.svd(f2); major,minor=s[:,0],s[:,1]; d=vh[:,0,:]; theta=np.arctan2(d[:,1],d[:,0]); dir2=np.stack((np.cos(2*theta),np.sin(2*theta)),axis=1); return major,minor,dir2,(0.5*nn)/rest_area,major/np.maximum(minor,eps),normal

def _adjacency(faces):
    e={}
    for fi,(a,b,c) in enumerate(faces.tolist()):
        for u,v in ((a,b),(b,c),(c,a)): e.setdefault(tuple(sorted((int(u),int(v)))),[]).append(fi)
    return tuple(tuple(v) for v in e.values() if len(v)==2)

def _bend(rest_normals,posed_normals,faces):
    out=np.zeros(len(faces))
    for a,b in _adjacency(faces):
        rd=float(np.clip(np.dot(rest_normals[a],rest_normals[b]),-1,1)); pd=float(np.clip(np.dot(posed_normals[a],posed_normals[b]),-1,1)); d=abs(math.acos(pd)-math.acos(rd)); out[a]=max(out[a],d); out[b]=max(out[b],d)
    return out

def _edge_strain(rest,posed,faces,eps):
    rf,pf=rest[faces],posed[faces]; vals=[]
    for a,b in ((0,1),(1,2),(2,0)):
        r=np.linalg.norm(rf[:,b]-rf[:,a],axis=1); p=np.linalg.norm(pf[:,b]-pf[:,a],axis=1)
        if np.any(r<=eps): raise ValueError("MECHANICAL_TRUTH_DEGENERATE_EDGE")
        vals.append(np.abs(p/r-1))
    return np.max(np.stack(vals,axis=1),axis=1)

def build_mechanical_demand_truth_v1(rest_vertices,faces,posed_vertices,*,contact_face_mask=None,policy=MechanicalDemandTruthPolicyV1()):
    """Training-only labels from authored animation; teacher G/W/motion never become inference inputs."""
    policy.validate(); rest,tri,posed=_validate(rest_vertices,faces,posed_vertices); inv_dm,rest_area,rest_normals=_basis(rest,tri,policy.epsilon); maj=[]; minr=[]; dirs=[]; areas=[]; cond=[]; bends=[]; edges=[]
    for frame in posed:
        a,b,d,ar,c,n=_frame(frame,tri,inv_dm,rest_area,policy.epsilon); maj.append(np.abs(a-1)); minr.append(np.abs(b-1)); dirs.append(d); areas.append(np.abs(ar-1)); cond.append(c); bends.append(_bend(rest_normals,n,tri)); edges.append(_edge_strain(rest,frame,tri,policy.epsilon))
    maj,minr,dirs,areas,cond,bends,edges=map(lambda x:np.stack(x,axis=0),(maj,minr,dirs,areas,cond,bends,edges)); w=maj+minr+1e-8; d=np.sum(dirs*w[...,None],axis=0); d=d/np.maximum(np.linalg.norm(d,axis=1,keepdims=True),policy.epsilon); rigid=((edges<=policy.rigid_edge_strain_tolerance)&(areas<=policy.rigid_area_change_tolerance)&(bends<=policy.rigid_bend_radians_tolerance)).mean(axis=0)
    if contact_face_mask is None: contact=np.zeros(len(tri)); valid=False
    else:
        cm=np.asarray(contact_face_mask,dtype=bool)
        if cm.shape!=(len(posed),len(tri)): raise ValueError("MECHANICAL_TRUTH_CONTACT_SHAPE_INVALID")
        contact=cm.mean(axis=0); valid=True
    q=lambda x:tuple(map(float,np.quantile(x,0.95,axis=0)))
    return MechanicalDemandTruthV1(q(maj),q(minr),tuple((float(x),float(y)) for x,y in d),q(bends),tuple(map(float,rigid)),q(areas),q(cond),q(edges),tuple(map(float,contact)),valid,len(tri),len(posed),metadata={"teacher_only":True,"rig_or_skin_inference_input":False,"principal_direction_representation":"COS2THETA_SIN2THETA","aggregation":"P95_DEFORMATION_DEMAND_OVER_AUTHORED_SEQUENCE","policy":asdict(policy)})
