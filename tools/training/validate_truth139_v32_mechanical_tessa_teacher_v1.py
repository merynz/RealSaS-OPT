#!/usr/bin/env python3
"""Independent rebind court for the selected V3.2 mechanical TESSA teacher."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np

SCHEMA="RealSaS.Truth139V32MechanicalTESSATeacherCourt.v1"
STATUS="PASS_V32_MECHANICAL_TESSA_TEACHER_REBINDS_EXACT_SOURCE_AND_DECLARED_COURTS"
PARENT_STATUS="PASS_V32_MECHANICALLY_QUALIFIED_TESSA_TEACHER"
REQUIRED_SIMPLIFIED_PASSES=("mapping_pass","component_pass","topology_hole_crack_preservation_pass","attachment_object_separation_pass","nondegenerate_all_actions_pass","inversion_foldthrough_pass","self_intersection_pass","response_pass","strain_jacobian_tail_pass","silhouette_pass","mechanical_pass")

def sha256(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for c in iter(lambda:f.read(8<<20),b""):h.update(c)
    return h.hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def bound(ref,code):
    p=Path(str(ref.get("path") or "")).resolve()
    if not p.is_file() or sha256(p)!=ref.get("sha256"):raise ValueError(code)
    return p
def barycentric(points,tri):
    a=tri[:,0];b=tri[:,1];c=tri[:,2];v0=b-a;v1=c-a;v2=points-a;d00=np.sum(v0*v0,1);d01=np.sum(v0*v1,1);d11=np.sum(v1*v1,1);d20=np.sum(v2*v0,1);d21=np.sum(v2*v1,1);den=d00*d11-d01*d01;v=np.divide(d11*d20-d01*d21,den,out=np.zeros_like(den),where=np.abs(den)>1e-20);w=np.divide(d00*d21-d01*d20,den,out=np.zeros_like(den),where=np.abs(den)>1e-20);return np.stack([1-v-w,v,w],1)
def tri_intersect_sat(a,b,eps=1e-9):
    ea=np.asarray([a[1]-a[0],a[2]-a[1],a[0]-a[2]]);eb=np.asarray([b[1]-b[0],b[2]-b[1],b[0]-b[2]]);na=np.cross(ea[0],ea[1]);nb=np.cross(eb[0],eb[1]);axes=[na,nb];axes.extend(np.cross(x,y) for x in ea for y in eb);axes.extend(np.cross(na,x) for x in ea);axes.extend(np.cross(nb,y) for y in eb)
    for axis in axes:
        n=float(np.linalg.norm(axis))
        if n<=eps:continue
        u=axis/n;pa=a@u;pb=b@u
        if float(pa.max())<float(pb.min())-eps or float(pb.max())<float(pa.min())-eps:return False
    return True
def rest_self_intersection(v,f):
    from rtree import index
    tri=v[f];idx=index.Index()
    for i,t in enumerate(tri):
        lo=t.min(0);hi=t.max(0);idx.insert(i,(float(lo[0]),float(lo[1]),float(lo[2]),float(hi[0]),float(hi[1]),float(hi[2])))
    for i,t in enumerate(tri):
        lo=t.min(0);hi=t.max(0)
        for j in idx.intersection((float(lo[0]),float(lo[1]),float(lo[2]),float(hi[0]),float(hi[1]),float(hi[2]))):
            if j<=i or len(set(map(int,f[i]))&set(map(int,f[j]))):continue
            if tri_intersect_sat(t,tri[j]):return True
    return False
def validate(manifest):
    if manifest.get("schema")!="RealSaS.Truth139V32MechanicalTESSATeacher.v1" or manifest.get("status")!=PARENT_STATUS:raise ValueError("V32_TESSA_COURT_PARENT")
    gm_path=bound(manifest["parents"]["geometry_manifest"],"V32_TESSA_COURT_GEOMETRY_DRIFT");gm=json.loads(gm_path.read_text());source_topology=bound(gm["outputs"]["topology"],"V32_TESSA_COURT_SOURCE_TOPOLOGY_DRIFT");mesh_path=bound(manifest["teacher_mesh"],"V32_TESSA_COURT_TEACHER_DRIFT")
    with np.load(source_topology,allow_pickle=False) as z:sv=np.asarray(z["rest_vertices_source_world"],dtype=np.float64);sf=np.asarray(z["faces"],dtype=np.int64)
    with np.load(mesh_path,allow_pickle=False) as z:v=np.asarray(z["vertices_source_world"],dtype=np.float64);f=np.asarray(z["faces"],dtype=np.int64);face_id=np.asarray(z["source_triangle_id"],dtype=np.int64);bar=np.asarray(z["source_barycentric"],dtype=np.float64)
    if v.ndim!=2 or v.shape[1]!=3 or f.ndim!=2 or f.shape[1]!=3 or face_id.shape!=(len(v),) or bar.shape!=(len(v),3):raise ValueError("V32_TESSA_COURT_LAYOUT")
    if np.any(face_id<0) or np.any(face_id>=len(sf)) or not np.allclose(bar.sum(1),1,atol=1e-5,rtol=0):raise ValueError("V32_TESSA_COURT_ADDRESS")
    reconstructed=np.sum(sv[sf[face_id]]*bar[...,None],axis=1);scale=max(float(np.linalg.norm(sv.max(0)-sv.min(0))),1e-9);max_error=float(np.max(np.linalg.norm(v-reconstructed,axis=1))/scale)
    if max_error>.0105:raise ValueError("V32_TESSA_COURT_SOURCE_REBIND_ERROR")
    selected=dict(manifest.get("selected") or {})
    if selected.get("mechanical_pass") is not True:raise ValueError("V32_TESSA_COURT_SELECTED_NOT_PASS")
    if selected.get("kind")=="SIMPLIFIED":
        for key in REQUIRED_SIMPLIFIED_PASSES:
            if selected.get(key) is not True:raise ValueError("V32_TESSA_COURT_DECLARED_PASS_MISSING:"+key)
        if rest_self_intersection(v,f):raise ValueError("V32_TESSA_COURT_REST_SELF_INTERSECTION")
    elif selected.get("kind")!="SOURCE_FALLBACK":raise ValueError("V32_TESSA_COURT_SELECTED_KIND")
    r={"schema":SCHEMA,"status":STATUS,"subject_id":manifest["subject_id"],"teacher_identity_sha256":manifest["teacher_identity_sha256"],"selected_kind":selected.get("kind"),"selected_face_count":int(len(f)),"source_rebind_max_error_over_scale":max_error,"simplified_required_pass_fields_verified":selected.get("kind")=="SIMPLIFIED","training_performed":False,"qualification_minted":False};r["court_identity_sha256"]=digest(r);return r

def main():
    p=argparse.ArgumentParser();p.add_argument("--teacher-manifest",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();r=validate(json.loads(a.teacher_manifest.resolve().read_text()));out=a.output.resolve();out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix(out.suffix+".partial");tmp.write_text(json.dumps(r,sort_keys=True,indent=2)+"\n");tmp.replace(out);print(json.dumps({"status":r["status"],"court_identity_sha256":r["court_identity_sha256"]},sort_keys=True))
if __name__=="__main__":main()
