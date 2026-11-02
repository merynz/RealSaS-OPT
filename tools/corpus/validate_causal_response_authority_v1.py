#!/usr/bin/env python3
from __future__ import annotations

"""Independent numerical court for CausalResponseAuthority.v1.

This court does not import the producer or presentation-response helper. It derives
its own control-balanced action measure, triangle deformation metric, centroid
response energy, edge adjacency, response coherence and response jump directly from
the hash-bound Stage47 intervention tensor and the courted mechanical topology.
"""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

PARENT_SCHEMA="RealSaS.CausalResponseAuthority.v1"
PARENT_STATUS="PASS_CAUSAL_RESPONSE_AUTHORITY_CANDIDATE__INDEPENDENT_COURT_REQUIRED"
SCHEMA="RealSaS.CausalResponseAuthorityCourt.v1"
STATUS="PASS_CAUSAL_RESPONSE_INDEPENDENT_COURT__FINAL_ADMISSION_SEPARATE"


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda:f.read(8<<20),b""): h.update(c)
    return h.hexdigest()


def bound(row,code):
    row=dict(row or {}); p=Path(str(row.get("path") or "")).resolve()
    if not p.is_file() or sha256(p)!=str(row.get("sha256") or ""): raise ValueError(code)
    return p


def control_weights(actions):
    by={}
    for i,row in enumerate(actions):
        control=row.get("control_index_privileged"); amp=row.get("amplitude"); sign=row.get("sign")
        if type(control) is not int or control<0 or not isinstance(amp,(int,float)) or float(amp)<=0 or sign not in (-1,1): raise ValueError("CAUSAL_COURT_ACTION_CONTRACT")
        by.setdefault(control,[]).append(i)
    keys=sorted(by)
    if not keys or keys!=list(range(len(keys))): raise ValueError("CAUSAL_COURT_CONTROL_SET")
    w=np.zeros(len(actions),np.float64)
    for control in keys:
        for i in by[control]: w[i]=1.0/len(keys)/len(by[control])
    if not np.isclose(w.sum(),1.0,atol=1e-12,rtol=0): raise ValueError("CAUSAL_COURT_ACTION_MASS")
    return w


def collapse(raw,mapping,count,scale):
    out=np.zeros((len(raw),count,3),np.float64); seen=np.zeros(count,bool); tol=max(scale*1e-9,1e-11)
    for rid,mid in enumerate(mapping):
        mid=int(mid)
        if mid<0: continue
        x=raw[:,rid,:]
        if not seen[mid]: out[:,mid,:]=x; seen[mid]=True
        elif float(np.max(np.linalg.norm(x-out[:,mid,:],axis=-1),initial=0.0))>tol: raise ValueError("CAUSAL_COURT_TRAJECTORY_DRIFT")
    if not np.all(seen): raise ValueError("CAUSAL_COURT_UNBOUND_MECHANICAL_VERTEX")
    return out


def deformation_measure(rest,faces,posed,weights,scale):
    nface=len(faces); na=len(posed)
    valid=np.ones((na,nface),bool); major=np.zeros((na,nface)); minor=np.zeros((na,nface)); area=np.zeros((na,nface)); aniso=np.zeros((na,nface)); tangent=np.zeros((na,nface,2))
    tol=1e-10*scale*scale
    for fi,tri in enumerate(faces):
        r=rest[tri]; e1=r[1]-r[0]; e2=r[2]-r[0]; l1=float(np.linalg.norm(e1));
        if l1<=0: valid[:,fi]=False; continue
        u=e1/l1; x2=float(np.dot(e2,u)); y2sq=float(np.dot(e2,e2)-x2*x2)
        if y2sq<=tol: valid[:,fi]=False; continue
        y2=float(np.sqrt(max(y2sq,0.0))); R=np.asarray([[l1,x2],[0.0,y2]],np.float64); invR=np.linalg.inv(R)
        # Batch only independent actions of this face. The rest basis, operation
        # sequence, validity thresholds and reduction order below stay fixed.
        # Working memory is O(action_count), never O(actions * all faces).
        p=posed[:,tri,:]; pe1=p[:,1]-p[:,0]; pe2=p[:,2]-p[:,0]
        P=np.stack((pe1,pe2),axis=2); F=P@invR; C=np.swapaxes(F,-1,-2)@F
        eigval,eigvec=np.linalg.eigh(C)
        action_valid=np.isfinite(eigval).all(axis=1)&(eigval[:,0]>1e-14)
        valid[:,fi]=action_valid
        selected=np.flatnonzero(action_valid)
        if not len(selected): continue
        s=np.sqrt(np.maximum(eigval[selected],0.0))
        minor[selected,fi]=s[:,0]; major[selected,fi]=s[:,1]
        area[selected,fi]=np.sqrt(np.maximum(np.linalg.det(C[selected]),0.0))
        aniso[selected,fi]=np.abs(s[:,1]-s[:,0])
        axis=eigvec[selected,:,1]; theta=np.arctan2(axis[:,1],axis[:,0])
        tangent[selected,fi]=np.stack((np.cos(2*theta),np.sin(2*theta)),axis=1)
    mass=np.sum(weights[:,None]*valid,axis=0); face_valid=mass>0; safe=np.where(face_valid,mass,1.0)
    wm=weights[:,None]
    major_mean=np.sum(wm*np.where(valid,major,0),axis=0)/safe; minor_mean=np.sum(wm*np.where(valid,minor,0),axis=0)/safe; area_mean=np.sum(wm*np.where(valid,area,0),axis=0)/safe; aniso_mean=np.sum(wm*np.where(valid,aniso,0),axis=0)/safe
    vec=np.sum((wm*valid*aniso)[...,None]*tangent,axis=0); norm=np.linalg.norm(vec,axis=1); tangent_valid=face_valid&(aniso_mean>1e-5)&(norm>1e-8); tangent_mean=np.zeros((nface,2)); tangent_mean[tangent_valid]=vec[tangent_valid]/norm[tangent_valid,None]
    delta=np.stack((major-1.0,minor-1.0),axis=-1); nonrigid=np.sqrt(np.sum(weights[:,None,None]*np.where(valid[...,None],delta*delta,0.0),axis=(0,2))/np.maximum(mass,1e-12))
    return face_valid,major_mean,minor_mean,area_mean,aniso_mean,tangent_valid,tangent_mean,nonrigid


def adjacency(faces):
    table={}
    for fi,tri in enumerate(faces):
        for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
            a,b=int(a),int(b); table.setdefault((min(a,b),max(a,b)),[]).append(fi)
    pairs=[]
    for ids in table.values():
        if len(ids)==2: pairs.append(tuple(sorted(map(int,ids))))
    return np.asarray(sorted(set(pairs)),np.int64).reshape(-1,2)


def validate(manifest_path:Path)->dict:
    manifest_path=manifest_path.resolve(); m=json.loads(manifest_path.read_text())
    if m.get("schema")!=PARENT_SCHEMA or m.get("status")!=PARENT_STATUS: raise ValueError("CAUSAL_COURT_PARENT")
    p=dict(m.get("parents") or {}); mm_path=bound(p.get("mechanical_surface_manifest"),"CAUSAL_COURT_MECH_MANIFEST_DRIFT"); mc_path=bound(p.get("mechanical_surface_court"),"CAUSAL_COURT_MECH_COURT_DRIFT"); scene_path=bound(p.get("teacher_scene"),"CAUSAL_COURT_SCENE_DRIFT"); stage47=bound(p.get("stage47_geometry_response"),"CAUSAL_COURT_STAGE47_DRIFT"); out=bound(m.get("output"),"CAUSAL_COURT_OUTPUT_DRIFT")
    mm=json.loads(mm_path.read_text()); mc=json.loads(mc_path.read_text()); scene=json.loads(scene_path.read_text())
    if mc.get("status")!="PASS_MECHANICAL_SURFACE_TOPOLOGY_INDEPENDENT_COURT__FINAL_ADMISSION_SEPARATE" or not bool((mc.get("authority") or {}).get("independent_court_passed")): raise ValueError("CAUSAL_COURT_MECHANICAL_NOT_QUALIFIED")
    mech=bound(mm.get("output"),"CAUSAL_COURT_MECH_ARRAY_DRIFT")
    if bound((mm.get("source") or {}).get("geometry_and_response"),"CAUSAL_COURT_MECH_SOURCE_DRIFT")!=stage47: raise ValueError("CAUSAL_COURT_SOURCE_BINDING")
    with np.load(mech,allow_pickle=False) as z:
        mapping=np.asarray(z["raw_to_mechanical_vertex"],np.int64); rest=np.asarray(z["mechanical_vertices_source_world"],np.float64); faces=np.asarray(z["mechanical_faces"],np.int64); owner=np.asarray(z["face_owner"],np.int32); geo_valid=np.asarray(z["face_valid"],bool); component=np.asarray(z["surface_component_id"],np.int32)
    with np.load(stage47,allow_pickle=False) as z:
        raw_rest=np.asarray(z["rest_vertices_source_world"],np.float64); raw_faces=np.asarray(z["faces"],np.int64); raw_owner=np.asarray(z["face_owner_privileged"],np.int32); raw_posed=np.asarray(z["isolated_position"],np.float64)
    if not np.array_equal(mapping[raw_faces],faces) or not np.array_equal(owner,raw_owner): raise ValueError("CAUSAL_COURT_FACE_BINDING")
    scale=max(float(np.linalg.norm(raw_rest.max(0)-raw_rest.min(0))),1e-9); posed=collapse(raw_posed,mapping,len(rest),scale); actions=list((scene.get("calibration") or {}).get("actions") or []); weights=control_weights(actions)
    if len(posed)!=len(weights): raise ValueError("CAUSAL_COURT_ACTION_CARDINALITY")
    face_valid,major,minor,area,aniso,tangent_valid,tangent,nonrigid=deformation_measure(rest,faces,posed,weights,scale); face_valid &= geo_valid
    cent_rest=rest[faces].mean(axis=1); cent_pose=posed[:,faces,:].mean(axis=2); response=cent_pose-cent_rest[None,:,:]; energy=np.sqrt(np.sum(weights[:,None,None]*response*response,axis=(0,2)))
    pairs=adjacency(faces); pair_valid=np.zeros(len(pairs),bool); coherence=np.zeros(len(pairs)); jump=np.zeros(len(pairs)); weighted=response*np.sqrt(weights)[:,None,None]; flat=np.transpose(weighted,(1,0,2)).reshape(len(faces),-1); norm=np.linalg.norm(flat,axis=1)
    for i,(a,b) in enumerate(pairs):
        denom=float(norm[a]*norm[b])
        if denom>1e-12 and face_valid[a] and face_valid[b]:
            c=float(np.clip(np.dot(flat[a],flat[b])/denom,-1,1)); coherence[i]=c; jump[i]=.5*(1-c); pair_valid[i]=True
    expected_face=np.flatnonzero(face_valid).astype(np.int32); expected_tangent=np.flatnonzero(tangent_valid&face_valid).astype(np.int32); expected_pair=np.flatnonzero(pair_valid).astype(np.int32)
    with np.load(out,allow_pickle=False) as z:
        required={"action_weight","face_index","stretch_major","stretch_minor","area_ratio","anisotropy","nonrigid_response","response_energy","tangent_face_index","tangent_direction_2theta","pair_face","pair_coherence","pair_jump","face_owner","surface_component_id"}
        if not required.issubset(z.files): raise ValueError("CAUSAL_COURT_OUTPUT_ARRAYS")
        actual={name:np.asarray(z[name]) for name in required}
    if not np.allclose(actual["action_weight"],weights,atol=1e-7,rtol=0): raise ValueError("CAUSAL_COURT_ACTION_WEIGHT")
    if not np.array_equal(actual["face_index"],expected_face): raise ValueError("CAUSAL_COURT_FACE_INDEX")
    checks=(("stretch_major",major[expected_face]),("stretch_minor",minor[expected_face]),("area_ratio",area[expected_face]),("anisotropy",aniso[expected_face]),("nonrigid_response",nonrigid[expected_face]),("response_energy",energy[expected_face]))
    for name,expected in checks:
        if not np.allclose(actual[name],expected,atol=2e-5,rtol=2e-5): raise ValueError("CAUSAL_COURT_NUMERIC:"+name)
    if not np.array_equal(actual["tangent_face_index"],expected_tangent) or not np.allclose(actual["tangent_direction_2theta"],tangent[expected_tangent],atol=2e-5,rtol=2e-5): raise ValueError("CAUSAL_COURT_TANGENT")
    if not np.array_equal(actual["pair_face"],pairs[expected_pair]): raise ValueError("CAUSAL_COURT_PAIR_FACE")
    if not np.allclose(actual["pair_coherence"],coherence[expected_pair],atol=2e-5,rtol=2e-5) or not np.allclose(actual["pair_jump"],jump[expected_pair],atol=2e-5,rtol=2e-5): raise ValueError("CAUSAL_COURT_PAIR_VALUES")
    if not np.array_equal(actual["face_owner"],owner) or not np.array_equal(actual["surface_component_id"],component): raise ValueError("CAUSAL_COURT_TOPOLOGY_METADATA")
    cross=int(np.count_nonzero(owner[pairs[expected_pair,0]]!=owner[pairs[expected_pair,1]])) if len(expected_pair) else 0
    counts=dict(m.get("counts") or {})
    expected_counts=(int(len(expected_face)),int(len(expected_tangent)),int(len(expected_pair)),cross)
    if (counts.get("qualified_face_count"),counts.get("qualified_tangent_face_count"),counts.get("qualified_pair_count"),counts.get("qualified_cross_owner_pair_count"))!=expected_counts: raise ValueError("CAUSAL_COURT_COUNTS")
    disposition="QUALIFIED_CORE_CANDIDATE" if len(expected_face)==len(faces) else "QUALIFIED_MASKED_CANDIDATE"
    result={"schema":SCHEMA,"status":STATUS,"subject_id":m.get("subject_id"),"counts":{"face_count":int(len(faces)),"qualified_face_count":int(len(expected_face)),"qualified_pair_count":int(len(expected_pair)),"qualified_cross_owner_pair_count":cross},"family_candidate_outcomes":{"CAUSAL_RESPONSE":disposition},"authority":{"independent_court_passed":True,"final_family_authority_minted":False,"training_authorized":False},"parent_manifest":{"path":str(manifest_path),"sha256":sha256(manifest_path)}}
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args(); r=validate(a.manifest); a.output.resolve().parent.mkdir(parents=True,exist_ok=True); a.output.resolve().write_text(json.dumps(r,sort_keys=True,indent=2)+"\n"); print(json.dumps({"status":r["status"],"subject_id":r["subject_id"],"counts":r["counts"]},sort_keys=True))


if __name__=="__main__": main()
