#!/usr/bin/env python3
"""Independent court for response-evidence V2 geometric adjacency."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np

from tools.corpus.validate_truth139_response_evidence_v1 import validate as validate_face_v1

PARENT_SCHEMA="RealSaS.Truth139IrisResponseEvidence.v2"
PARENT_STATUS="PASS_IRIS_RESPONSE_EVIDENCE_V2__INDEPENDENT_COURT_REQUIRED"
SCHEMA="RealSaS.Truth139IrisResponseEvidenceCourt.v2"
PASS_STATUS="PASS_IRIS_RESPONSE_EVIDENCE_V2_INDEPENDENT_NUMERIC_COURT"


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda:f.read(8<<20),b""): h.update(c)
    return h.hexdigest()


def digest(v):
    return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def bound(ref,code):
    p=Path(str((ref or {}).get("path") or "")).resolve()
    if not p.is_file() or sha256(p)!=str((ref or {}).get("sha256") or ""): raise ValueError(code)
    return p


def independent_adjacency(vertices,faces,face_owner):
    v=np.asarray(vertices,np.float64); f=np.asarray(faces,np.int64); owner=np.asarray(face_owner,np.int32)
    raw_owner=np.full(len(v),-1,np.int32)
    for fi,tri in enumerate(f):
        for raw in tri:
            raw=int(raw); oid=int(owner[fi])
            if raw_owner[raw] == -1: raw_owner[raw]=oid
            elif raw_owner[raw] != oid: raise ValueError("TRUTH139_RESPONSE_V2_COURT_VERTEX_OWNER_CROSS")
    lookup={}; remap=np.full(len(v),-1,np.int32)
    for raw in np.unique(f):
        raw=int(raw); p=v[raw].copy(); p[p==0.0]=0.0; key=(int(raw_owner[raw]),p.tobytes())
        if key not in lookup: lookup[key]=len(lookup)
        remap[raw]=lookup[key]
    qf=remap[f]
    edges={}
    for fi,tri in enumerate(qf):
        for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
            edges.setdefault(tuple(sorted((int(a),int(b)))),[]).append(fi)
    pairs=set()
    for rows in edges.values():
        if len(rows)==2:
            a,b=sorted(rows); pairs.add((a,b))
    return np.asarray(sorted(pairs),np.int64).reshape(-1,2)


def validate(manifest):
    if manifest.get("schema")!=PARENT_SCHEMA or manifest.get("status")!=PARENT_STATUS: raise ValueError("TRUTH139_RESPONSE_V2_COURT_PARENT")
    scene_path=bound((manifest.get("source") or {}).get("teacher_scene"),"TRUTH139_RESPONSE_V2_COURT_SCENE")
    geometry_path=bound((manifest.get("source") or {}).get("geometry_and_response"),"TRUTH139_RESPONSE_V2_COURT_GEOMETRY")
    output_path=bound(manifest.get("output"),"TRUTH139_RESPONSE_V2_COURT_OUTPUT")
    face_manifest=dict(manifest)
    face_manifest["schema"]="RealSaS.Truth139IrisResponseEvidence.v1"
    face_manifest["status"]="PASS_IRIS_RESPONSE_EVIDENCE_CANDIDATE__INDEPENDENT_COURT_REQUIRED"
    face_manifest["source"]={"teacher_scene":manifest["source"]["teacher_scene"],"geometry_and_response":manifest["source"]["geometry_and_response"]}
    face_manifest["response_evidence_identity_sha256"]="V2_PAIR_REPAIR"
    face_court=validate_face_v1(face_manifest)
    scene=json.loads(scene_path.read_text()); actions=list((scene.get("calibration") or {}).get("actions") or [])
    with np.load(geometry_path,allow_pickle=False) as z:
        rest=np.asarray(z["rest_vertices_source_world"],np.float64); faces=np.asarray(z["faces"],np.int64); owner=np.asarray(z["face_owner_privileged"],np.int32); isolated=np.asarray(z["isolated_position"],np.float64)
    with np.load(output_path,allow_pickle=False) as z:
        pair=np.asarray(z["response_pair_face"],np.int64); coherence=np.asarray(z["response_pair_coherence"],np.float64); jump=np.asarray(z["response_pair_jump"],np.float64); valid=np.asarray(z["response_pair_valid"],bool); weights=np.asarray(z["action_weight"],np.float64)
    expected_pair=independent_adjacency(rest,faces,owner)
    if not np.array_equal(pair,expected_pair): raise ValueError("TRUTH139_RESPONSE_V2_COURT_ADJACENCY_MISMATCH")
    if len(weights)!=len(actions) or len(weights)!=len(isolated): raise ValueError("TRUTH139_RESPONSE_V2_COURT_ACTION_LAYOUT")
    rest_centroid=rest[faces].mean(1); posed_centroid=isolated[:,faces,:].mean(2); response=posed_centroid-rest_centroid[None,:,:]
    flat=np.transpose(response*np.sqrt(weights)[:,None,None],(1,0,2)).reshape(len(faces),-1); norm=np.linalg.norm(flat,axis=1)
    ec=np.zeros(len(pair)); ej=np.zeros(len(pair)); ev=np.zeros(len(pair),bool)
    for i,(a,b) in enumerate(pair):
        denom=float(norm[a]*norm[b])
        if denom>1e-12:
            value=float(np.clip(np.dot(flat[a],flat[b])/denom,-1,1)); ec[i]=value; ej[i]=.5*(1-value); ev[i]=True
    if not np.array_equal(valid,ev) or np.max(np.abs(coherence-ec),initial=0)>1e-12 or np.max(np.abs(jump-ej),initial=0)>1e-12:
        raise ValueError("TRUTH139_RESPONSE_V2_COURT_PAIR_VALUE_MISMATCH")
    payload={"schema":SCHEMA,"status":PASS_STATUS,"subject_id":manifest["subject_id"],"response_evidence_identity_sha256":manifest["response_evidence_identity_sha256"],"face_metric_court_status":face_court["status"],"pair_adjacency_method":"INDEPENDENT_OWNER_PLUS_EXACT_POSITION_QUOTIENT","response_pair_count":int(len(pair)),"valid_response_pair_count":int(valid.sum()),"training_performed":False,"qualification_minted":False}
    payload["court_identity_sha256"]=digest(payload); return payload


def main():
    p=argparse.ArgumentParser(); p.add_argument("--response-manifest",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    m=json.loads(a.response_manifest.resolve().read_text()); out=validate(m); a.output.resolve().parent.mkdir(parents=True,exist_ok=True); a.output.resolve().write_text(json.dumps(out,sort_keys=True,indent=2,allow_nan=False)+"\n"); print(json.dumps({"status":out["status"],"subject_id":out["subject_id"],"response_pair_count":out["response_pair_count"]},sort_keys=True))


if __name__=="__main__": main()
