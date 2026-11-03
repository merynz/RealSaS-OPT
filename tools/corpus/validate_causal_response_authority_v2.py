#!/usr/bin/env python3
from __future__ import annotations

"""Independent numerical court for CausalResponseAuthority.v2.

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

PARENT_SCHEMA="RealSaS.CausalResponseAuthority.v2"
PARENT_STATUS="PASS_CAUSAL_RESPONSE_AUTHORITY_CANDIDATE__INDEPENDENT_COURT_REQUIRED"
SCHEMA="RealSaS.CausalResponseAuthorityCourt.v2"
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
    # Deliberately independent of the producer: per-face coordinates and pulled
    # edge Gram metrics rather than its batched world Jacobian implementation.
    na,nf=len(posed),len(faces)
    valid=np.zeros((na,nf),bool); major=np.zeros((na,nf)); minor=np.zeros((na,nf))
    area=np.zeros((na,nf)); aniso=np.zeros((na,nf)); tangent=np.zeros((na,nf,2))
    threshold=1e-10*scale*scale
    for fi,ids in enumerate(faces):
        triangle=rest[ids]; a,b=triangle[1]-triangle[0],triangle[2]-triangle[0]
        oriented=np.cross(a,b); twice_area=float(np.linalg.norm(oriented))
        longest=max(float(np.linalg.norm(a)),float(np.linalg.norm(b)),float(np.linalg.norm(b-a)))
        if longest<=0 or .5*twice_area<=threshold or (twice_area/longest)**2<=threshold:
            continue
        n=oriented/twice_area
        choice=min(range(3),key=lambda i:(abs(float(n[i])),i))
        horizontal=np.zeros(3); horizontal[choice]=1.; horizontal-=n[choice]*n
        horizontal/=np.linalg.norm(horizontal); vertical=np.cross(n,horizontal)
        R=np.array([[np.dot(horizontal,a),np.dot(horizontal,b)],
                    [np.dot(vertical,a),np.dot(vertical,b)]])
        inverse=np.linalg.solve(R,np.eye(2))
        q=posed[:,ids,:]; da,db=q[:,1]-q[:,0],q[:,2]-q[:,0]
        gram=np.empty((na,2,2)); gram[:,0,0]=np.einsum('ij,ij->i',da,da)
        gram[:,1,1]=np.einsum('ij,ij->i',db,db)
        gram[:,0,1]=gram[:,1,0]=np.einsum('ij,ij->i',da,db)
        metric=inverse.T @ gram @ inverse
        eigen,axes=np.linalg.eigh(metric)
        ok=np.isfinite(eigen).all(axis=1)&(eigen[:,0]>1e-14)&(.5*np.linalg.norm(np.cross(da,db),axis=1)>threshold)
        valid[:,fi]=ok; selected=np.flatnonzero(ok)
        if not len(selected): continue
        stretches=np.sqrt(eigen[selected]); minor[selected,fi]=stretches[:,0]; major[selected,fi]=stretches[:,1]
        area[selected,fi]=np.sqrt(np.maximum(np.linalg.det(metric[selected]),0.))
        aniso[selected,fi]=stretches[:,1]-stretches[:,0]
        angle=np.arctan2(axes[selected,1,1],axes[selected,0,1])
        tangent[selected,fi]=np.column_stack([np.cos(2*angle),np.sin(2*angle)])
    wm=weights[:,None]; mass=np.sum(wm*valid,axis=0); face_valid=mass>0; safe=np.where(face_valid,mass,1.)
    average=lambda x:np.sum(wm*np.where(valid,x,0.),axis=0)/safe
    vector=np.sum((wm*valid*aniso)[...,None]*tangent,axis=0); length=np.linalg.norm(vector,axis=1)
    tv=face_valid&(average(aniso)>1e-5)&(length>1e-8); direction=np.zeros((nf,2)); direction[tv]=vector[tv]/length[tv,None]
    nonrigid=np.sqrt(np.sum(wm*np.where(valid,(major-1.)**2+(minor-1.)**2,0.),axis=0)/np.maximum(mass,1e-12))
    return face_valid,average(major),average(minor),average(area),average(aniso),tv,direction,nonrigid


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
    policy=m.get("measurement_policy") or {}
    if policy.get("tangent_frame")!="SOURCE_WORLD_NORMAL_PROJECTED_LEAST_ALIGNED_AXIS_V1" or policy.get("face_validity")!="ALL_REST_ALTITUDES_REST_POSED_AREA_POSITIVE_METRIC_V1":
        raise ValueError("CAUSAL_V2_COURT_MEASUREMENT_POLICY")
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
    for name,value in actual.items():
        if not np.isfinite(value).all(): raise ValueError("CAUSAL_V2_COURT_NONFINITE:"+name)
    shapes={"action_weight":weights.shape,"face_index":expected_face.shape,"tangent_face_index":expected_tangent.shape,
            "tangent_direction_2theta":(len(expected_tangent),2),"pair_face":(len(expected_pair),2),
            "pair_coherence":expected_pair.shape,"pair_jump":expected_pair.shape,"face_owner":owner.shape,"surface_component_id":component.shape}
    for name in ("stretch_major","stretch_minor","area_ratio","anisotropy","nonrigid_response","response_energy"): shapes[name]=expected_face.shape
    for name,shape in shapes.items():
        if actual[name].shape!=shape: raise ValueError("CAUSAL_V2_COURT_LAYOUT:"+name)
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
    disposition=("UNAVAILABLE_CANDIDATE" if not len(expected_face) else "QUALIFIED_CORE_CANDIDATE" if len(expected_face)==len(faces) else "QUALIFIED_MASKED_CANDIDATE")
    result={"schema":SCHEMA,"status":STATUS,"subject_id":m.get("subject_id"),"counts":{"face_count":int(len(faces)),"qualified_face_count":int(len(expected_face)),"qualified_pair_count":int(len(expected_pair)),"qualified_cross_owner_pair_count":cross},"family_candidate_outcomes":{"CAUSAL_RESPONSE":disposition},"authority":{"independent_court_passed":True,"final_family_authority_minted":False,"training_authorized":False},"parent_manifest":{"path":str(manifest_path),"sha256":sha256(manifest_path)}}
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args(); r=validate(a.manifest); a.output.resolve().parent.mkdir(parents=True,exist_ok=True); a.output.resolve().write_text(json.dumps(r,sort_keys=True,indent=2)+"\n"); print(json.dumps({"status":r["status"],"subject_id":r["subject_id"],"counts":r["counts"]},sort_keys=True))


if __name__=="__main__": main()
