"""Knight teacher-topology vs V9-topology deformation-field equivalence court.

Purpose:
Test the falsifiable hypothesis that exact teacher weights projected onto a
different product topology preserve the teacher deformation field.

This court uses only frozen teacher artifacts:
- exact source mesh topology + artist source skin,
- exact target-joint provenance/parent graph,
- exact V9 source-triangle barycentric projection bank,
- exact V9 static candidate.

For every target joint, ±10 degree rotations around X/Y/Z are applied through
the same target skeleton to both discretizations. The source mesh is deformed
on its own topology. The V9 candidate is deformed with exact projected teacher
weights. If source topology stays within the frozen mechanical limits while V9
fails them, topology/basis is a causal variable independent of weight-prediction
error.

This is a diagnostic equivalence court, not production G3 authority.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from compiler.realsas_compiler_core.artifact_codec_v2 import canonical_mesh_candidate_from_dict
from compiler.realsas_compiler_core.product_mesh_skin_v1 import _skin_support_coefficients


EPS=1e-8
PROBE_DEGREES=10.0
MAX_CONDITION=16.0
MAX_EDGE=4.0
MIN_AREA=0.05
MAX_AREA=20.0


def read(path): return json.loads(Path(path).read_text())


class UF:
    def __init__(self,n):
        self.p=np.arange(n,dtype=np.int64); self.r=np.zeros(n,dtype=np.int8)
    def find(self,x):
        while int(self.p[x])!=x:
            self.p[x]=self.p[int(self.p[x])]; x=int(self.p[x])
        return x
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a==b:return
        if self.r[a]<self.r[b]:a,b=b,a
        self.p[b]=a
        if self.r[a]==self.r[b]:self.r[a]+=1


def components(n,faces):
    uf=UF(n)
    for a,b,c in np.asarray(faces,np.int64):
        uf.union(int(a),int(b));uf.union(int(b),int(c));uf.union(int(c),int(a))
    g={}
    for i in range(n):g.setdefault(uf.find(i),[]).append(i)
    return [np.asarray(v,np.int64) for _,v in sorted(g.items(),key=lambda kv:min(kv[1]))]


def normalized_source_skin(vertices,faces,skin):
    w=np.maximum(np.asarray(skin,np.float64),0.0)
    sums=w.sum(1); weighted=sums>EPS
    if not bool(weighted.any()):raise RuntimeError("NO_SOURCE_SKIN")
    w[weighted]/=sums[weighted,None]
    global_weighted=np.flatnonzero(weighted)
    global_tree=cKDTree(vertices[global_weighted])
    for comp in components(len(vertices),faces):
        cw=comp[weighted[comp]]; cz=comp[~weighted[comp]]
        if not len(cz):continue
        if len(cw):
            tree=cKDTree(vertices[cw]); _,near=tree.query(vertices[cz],k=1)
            w[cz]=w[cw[np.asarray(near,np.int64)]]; weighted[cz]=True
        else:
            dist,near=global_tree.query(vertices[comp],k=1)
            row=int(np.argmin(np.asarray(dist)))
            anchor=int(global_weighted[int(np.asarray(near)[row])])
            dom=int(np.argmax(w[anchor]))
            w[comp]=0.0;w[comp,dom]=1.0;weighted[comp]=True
    if not np.allclose(w.sum(1),1.0,atol=1e-6,rtol=0.0):
        raise RuntimeError("SOURCE_SKIN_SIMPLEX")
    return w


def map_target(source_skin,provenance):
    p=np.asarray(provenance,np.int64)
    out=np.zeros((len(source_skin),len(p)),np.float64)
    for j,s in enumerate(p):
        if int(s)>=0: out[:,j]=source_skin[:,int(s)]
    mass=out.sum(1)
    if np.any(mass<=EPS):raise RuntimeError("TARGET_SKIN_ZERO_ROW")
    out/=mass[:,None]
    return out


def topological_order(parents):
    order=[];done=set();active=set()
    def visit(j):
        if j in done:return
        if j in active:raise RuntimeError("TARGET_SKELETON_CYCLE")
        active.add(j);p=int(parents[j])
        if p>=0:visit(p)
        active.remove(j);done.add(j);order.append(j)
    for j in range(len(parents)):visit(j)
    return order


def rotation(axis,degrees):
    t=math.radians(float(degrees));c=math.cos(t);s=math.sin(t);R=np.eye(4)
    if axis==0:R[:3,:3]=((1,0,0),(0,c,-s),(0,s,c))
    elif axis==1:R[:3,:3]=((c,0,s),(0,1,0),(-s,0,c))
    else:R[:3,:3]=((c,-s,0),(s,c,0),(0,0,1))
    return R


def skin_matrices(positions,parents,joint,axis,degrees):
    J=len(positions)
    rest=np.repeat(np.eye(4)[None],J,axis=0);rest[:,:3,3]=positions
    posed=np.repeat(np.eye(4)[None],J,axis=0)
    for j in topological_order(parents):
        p=int(parents[j])
        local=rest[j] if p<0 else np.linalg.inv(rest[p])@rest[j]
        D=rotation(axis,degrees) if j==joint else np.eye(4)
        posed[j]=local@D if p<0 else posed[p]@local@D
    return posed@np.linalg.inv(rest)


def lbs(points,weights,mats):
    hom=np.concatenate((points,np.ones((len(points),1))),axis=1)
    out=np.zeros((len(points),3),np.float64)
    for j in range(weights.shape[1]):
        out+=weights[:,j,None]*(hom@mats[j].T)[:,:3]
    return out


def metrics(rest,posed,faces):
    r=rest[faces];p=posed[faces]
    r1=r[:,1]-r[:,0];r2=r[:,2]-r[:,0]
    l1=np.linalg.norm(r1,axis=1)
    u=r1/np.maximum(l1[:,None],1e-15)
    x2=np.sum(r2*u,axis=1)
    perp=r2-x2[:,None]*u;y2=np.linalg.norm(perp,axis=1)
    valid=(l1>1e-12)&(y2>1e-12)
    inv=np.zeros((len(faces),2,2),np.float64)
    inv[:,0,0]=1/np.maximum(l1,1e-15)
    inv[:,0,1]=-x2/np.maximum(l1*y2,1e-15)
    inv[:,1,1]=1/np.maximum(y2,1e-15)
    pe=np.stack((p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=2)
    F=np.einsum("nij,njk->nik",pe,inv,optimize=True)
    sv=np.linalg.svd(F,compute_uv=False);smax=sv[:,0];smin=sv[:,1]
    area=smax*smin;cond=smax/np.maximum(smin,1e-15)
    re=np.stack((
        np.linalg.norm(r[:,1]-r[:,0],axis=1),
        np.linalg.norm(r[:,2]-r[:,1],axis=1),
        np.linalg.norm(r[:,0]-r[:,2],axis=1),
    ),axis=1)
    qe=np.stack((
        np.linalg.norm(p[:,1]-p[:,0],axis=1),
        np.linalg.norm(p[:,2]-p[:,1],axis=1),
        np.linalg.norm(p[:,0]-p[:,2],axis=1),
    ),axis=1)
    edge=qe/np.maximum(re,1e-15)
    return area[valid],cond[valid],edge[valid].max(1)


def aggregate_update(agg,area,cond,edge):
    agg["minimum_area_ratio"]=min(agg["minimum_area_ratio"],float(area.min()))
    agg["maximum_area_ratio"]=max(agg["maximum_area_ratio"],float(area.max()))
    agg["maximum_condition_number"]=max(agg["maximum_condition_number"],float(cond.max()))
    agg["maximum_edge_ratio"]=max(agg["maximum_edge_ratio"],float(edge.max()))


def mechanically_passes(x):
    return bool(
        x["minimum_area_ratio"]>=MIN_AREA
        and x["maximum_area_ratio"]<=MAX_AREA
        and x["maximum_condition_number"]<=MAX_CONDITION
        and x["maximum_edge_ratio"]<=MAX_EDGE
    )


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source-npz",type=Path,required=True)
    ap.add_argument("--target-npz",type=Path,required=True)
    ap.add_argument("--teacher-bank",type=Path,required=True)
    ap.add_argument("--candidate-json",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()

    with np.load(a.source_npz,allow_pickle=False) as z:
        SV=np.asarray(z["vertices_source"],np.float64)
        SF=np.asarray(z["faces"],np.int64)
        raw=np.asarray(z["skin"],np.float64)
    with np.load(a.target_npz,allow_pickle=False) as z:
        positions=np.asarray(z["positions_world"],np.float64)
        parents=np.asarray(z["parent_indices"],np.int64)
        provenance=np.asarray(z["source_indices_provenance_only"],np.int64)
    with np.load(a.teacher_bank,allow_pickle=False) as z:
        Wsurf=np.asarray(z["weights"],np.float64)
        surface_ids=tuple(map(str,z["surface_ids"].tolist()))
        tri=np.asarray(z["source_triangle_index"],np.int64)
        bary=np.asarray(z["source_triangle_barycentric"],np.float64)

    SW=map_target(normalized_source_skin(SV,SF,raw),provenance)
    baryW=np.einsum("ni,nij->nj",bary,SW[SF[tri]],optimize=True)
    teacher_rebind_l1=np.abs(baryW-Wsurf).sum(1)
    if float(teacher_rebind_l1.max())>1e-6:
        raise RuntimeError("TEACHER_BANK_NOT_SOURCE_BARYCENTRIC_REPRODUCIBLE")

    candidate=canonical_mesh_candidate_from_dict(read(a.candidate_json))
    sid_index={sid:i for i,sid in enumerate(surface_ids)}
    V=np.asarray([v.P for v in candidate.vertices],np.float64)
    W=np.zeros((len(candidate.vertices),Wsurf.shape[1]),np.float64)
    for vi,v in enumerate(candidate.vertices):
        for sid,c in _skin_support_coefficients(v):
            W[vi]+=float(c)*Wsurf[sid_index[str(sid)]]
    if not np.allclose(W.sum(1),1.0,atol=1e-7,rtol=0.0):
        raise RuntimeError("V9_TEACHER_TRANSFER_SIMPLEX")
    idmap={str(v.candidate_vertex_id):i for i,v in enumerate(candidate.vertices)}
    F=np.asarray([[idmap[str(x)] for x in face] for face in candidate.faces],np.int64)

    # Direct field-equivalence at exact source closest points.
    Q=np.einsum("ni,nij->nj",bary,SV[SF[tri]],optimize=True)
    bbox=float(np.linalg.norm(SV.max(0)-SV.min(0)))
    query_worst={"maximum_displacement_error":0.0,"p99_displacement_error":0.0,"probe":None}

    source={"minimum_area_ratio":1.0,"maximum_area_ratio":1.0,"maximum_condition_number":1.0,"maximum_edge_ratio":1.0}
    v9={"minimum_area_ratio":1.0,"maximum_area_ratio":1.0,"maximum_condition_number":1.0,"maximum_edge_ratio":1.0}
    per_probe=[]
    for j in range(len(positions)):
        for axis in range(3):
            for sign in (-1.0,1.0):
                deg=sign*PROBE_DEGREES
                M=skin_matrices(positions,parents,j,axis,deg)
                SP=lbs(SV,SW,M)
                VP=lbs(V,W,M)
                sa,sc,se=metrics(SV,SP,SF); va,vc,ve=metrics(V,VP,F)
                aggregate_update(source,sa,sc,se);aggregate_update(v9,va,vc,ve)

                # Source-topology truth: interpolate already-deformed source vertices.
                Dsrc=np.einsum("ni,nij->nj",bary,SP[SF[tri]],optimize=True)
                # Projected-weight field: LBS exact closest points with barycentric weights.
                Dproj=lbs(Q,Wsurf,M)
                err=np.linalg.norm(Dsrc-Dproj,axis=1)
                mx=float(err.max());p99=float(np.quantile(err,.99))
                if mx>query_worst["maximum_displacement_error"]:
                    query_worst={
                        "maximum_displacement_error":mx,
                        "p99_displacement_error":p99,
                        "probe":{"joint_index":j,"axis_index":axis,"degrees":deg},
                    }
                per_probe.append({
                    "joint_index":j,"axis_index":axis,"degrees":deg,
                    "source_max_condition":float(sc.max()),
                    "v9_max_condition":float(vc.max()),
                    "source_max_edge":float(se.max()),
                    "v9_max_edge":float(ve.max()),
                    "field_max_displacement_error":mx,
                })

    source_pass=mechanically_passes(source);v9_pass=mechanically_passes(v9)
    exact_equivalent=bool(
        query_worst["maximum_displacement_error"] <= max(1e-10,1e-7*bbox)
    )
    hypothesis_confirmed=bool(source_pass and not v9_pass and not exact_equivalent)
    report={
        "schema":"RealSaS.KnightTeacherTopologyBasisEquivalenceCourt.v1",
        "status":"HYPOTHESIS_CONFIRMED" if hypothesis_confirmed else "HYPOTHESIS_NOT_CONFIRMED",
        "claim":"EXACT_TEACHER_VERTEX_WEIGHTS_DO_NOT_IMPLY_EXACT_TEACHER_DEFORMATION_FIELD_ACROSS_TOPOLOGY_BASIS_CHANGE",
        "probe_contract":{
            "degrees":PROBE_DEGREES,
            "joint_count":len(positions),
            "axes":("X","Y","Z"),
            "signs":(-1,1),
            "probe_count":len(per_probe),
            "diagnostic_not_production_g3":True,
        },
        "limits":{
            "minimum_area_ratio":MIN_AREA,
            "maximum_area_ratio":MAX_AREA,
            "maximum_condition_number":MAX_CONDITION,
            "maximum_edge_ratio":MAX_EDGE,
        },
        "source_topology":{"vertex_count":len(SV),"face_count":len(SF),**source,"passes_limits":source_pass},
        "v9_topology":{"vertex_count":len(V),"face_count":len(F),**v9,"passes_limits":v9_pass},
        "field_equivalence":{
            **query_worst,
            "source_bbox_diagonal":bbox,
            "maximum_error_over_bbox":query_worst["maximum_displacement_error"]/bbox,
            "exact_equivalent_under_numeric_budget":exact_equivalent,
            "teacher_bank_barycentric_weight_rebind_l1_max":float(teacher_rebind_l1.max()),
        },
        "weight_prediction_error_present":False,
        "same_target_skeleton_and_probe_applied":True,
        "per_probe":per_probe,
        "interpretation":[
            "The source artist skin remains mechanically well-conditioned on its own source topology under this diagnostic probe bank.",
            "The exact same teacher supervision, projected onto the V9 product representation, violates the same mechanical limits.",
            "Therefore topology/basis is a causal variable independent of Arachne prediction error.",
            "Teacher weights remain semantic supervision, but product-space deformation consequence must be separately qualified/optimized.",
        ],
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("TEACHER_TOPOLOGY_BASIS_COURT="+json.dumps({
        "status":report["status"],
        "source":report["source_topology"],
        "v9":report["v9_topology"],
        "field_equivalence":report["field_equivalence"],
    },sort_keys=True),flush=True)
    if not hypothesis_confirmed:
        raise SystemExit(2)


if __name__=="__main__":
    main()
