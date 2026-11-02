#!/usr/bin/env python3
"""Independent court for V3.2 exact-source geometry targets."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np

from tools.corpus.build_truth139_v32_exact_geometry_v1 import (
    SCHEMA as PARENT_SCHEMA, STATUS as PARENT_STATUS, sha256, matrix4,
    load_geometry, transform_points, face_geometry, topology, point_triangle_distance_sq, ray_parity,
)

SCHEMA = "RealSaS.Truth139V32ExactGeometryCourt.v1"
STATUS = "PASS_V32_EXACT_GEOMETRY_REBINDS_SOURCE_AND_INDEPENDENT_SAMPLES"

def digest(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def bound(ref, code):
    p=Path(str(ref.get("path") or "")).resolve(); h=str(ref.get("sha256") or "")
    if not p.is_file() or sha256(p)!=h: raise ValueError(code)
    return p

def close(code,a,b,atol):
    if a.shape!=b.shape or not np.allclose(a,b,atol=atol,rtol=0,equal_nan=False): raise ValueError(code)

def validate(manifest: dict) -> dict:
    if manifest.get("schema")!=PARENT_SCHEMA or manifest.get("status")!=PARENT_STATUS: raise ValueError("V32_GEOMETRY_COURT_PARENT")
    subject=str(manifest.get("subject_id") or ""); parents=dict(manifest["parents"]); outputs=dict(manifest["outputs"])
    gp=bound(parents["geometry_npz"],"V32_GEOMETRY_COURT_GEOMETRY_DRIFT"); sp=bound(parents["source_scene_snapshot"],"V32_GEOMETRY_COURT_SNAPSHOT_DRIFT")
    tp=bound(outputs["topology"],"V32_GEOMETRY_COURT_TOPOLOGY_DRIFT"); sfp=bound(outputs["surface"],"V32_GEOMETRY_COURT_SURFACE_DRIFT"); qp=bound(outputs["distance_sign"],"V32_GEOMETRY_COURT_QUERY_DRIFT")
    snapshot=json.loads(sp.read_text()); m=matrix4(snapshot["source_world_to_canonical"]); vsw,faces=load_geometry(gp); v=transform_points(vsw,m)
    area,normals,valid=face_geometry(vsw,faces); edges,inc,boundary,nonmanifold,orient,vc,fc=topology(len(v),faces)
    with np.load(tp,allow_pickle=False) as z:
        if not np.array_equal(np.asarray(z["faces"],dtype=np.int64),faces): raise ValueError("V32_GEOMETRY_COURT_FACE_DRIFT")
        close("V32_GEOMETRY_COURT_VERTEX_DRIFT",np.asarray(z["rest_vertices_canonical"],dtype=np.float64),v,2e-5)
        if not np.array_equal(np.asarray(z["unique_edge_index"],dtype=np.int32),edges): raise ValueError("V32_GEOMETRY_COURT_EDGE_DRIFT")
        if not np.array_equal(np.asarray(z["edge_face_incidence_count"],dtype=np.int32),inc): raise ValueError("V32_GEOMETRY_COURT_INCIDENCE_DRIFT")
        if not np.array_equal(np.asarray(z["boundary_edge"],dtype=np.bool_),boundary): raise ValueError("V32_GEOMETRY_COURT_BOUNDARY_DRIFT")
        if not np.array_equal(np.asarray(z["nonmanifold_edge"],dtype=np.bool_),nonmanifold): raise ValueError("V32_GEOMETRY_COURT_NONMANIFOLD_DRIFT")
    with np.load(sfp,allow_pickle=False) as z:
        tri=np.asarray(z["triangle_id"],dtype=np.int64); bary=np.asarray(z["barycentric"],dtype=np.float64); xyz=np.asarray(z["surface_xyz_canonical"],dtype=np.float64)
    if len(tri)<1 or bary.shape!=(len(tri),3): raise ValueError("V32_GEOMETRY_COURT_SURFACE_LAYOUT")
    idx=np.linspace(0,len(tri)-1,min(128,len(tri)),dtype=np.int64); recon=np.sum(v[faces[tri[idx]]] * bary[idx,...,None],axis=1); close("V32_GEOMETRY_COURT_SURFACE_ADDRESS",xyz[idx],recon,4e-5)
    with np.load(qp,allow_pickle=False) as z:
        q=np.asarray(z["query_xyz_canonical"],dtype=np.float64); d=np.asarray(z["unsigned_distance"],dtype=np.float64); sign_valid=np.asarray(z["sign_valid"],dtype=np.bool_); occ=np.asarray(z["occupancy"],dtype=np.bool_)
    qidx=np.linspace(0,len(q)-1,min(48,len(q)),dtype=np.int64); triarr=v[faces]
    for start in range(0,len(qidx),8):
        ids=qidx[start:start+8]; d2=point_triangle_distance_sq(q[ids],triarr); exact=np.sqrt(np.min(d2,axis=1)); close("V32_GEOMETRY_COURT_DISTANCE",d[ids],exact,5e-5)
    valid_ids=qidx[sign_valid[qidx]]
    if len(valid_ids):
        independent=ray_parity(q[valid_ids],v,faces,np.asarray([0.431,0.719,1.0],dtype=np.float64))
        if not np.array_equal(independent,occ[valid_ids]): raise ValueError("V32_GEOMETRY_COURT_SIGN_PARITY_DRIFT")
    result={"schema":SCHEMA,"status":STATUS,"subject_id":subject,"geometry_identity_sha256":manifest["geometry_identity_sha256"],"validated":{"surface_address_rows":int(len(idx)),"distance_rows":int(len(qidx)),"independent_sign_rows":int(len(valid_ids))},"training_performed":False,"qualification_minted":False}
    result["court_identity_sha256"]=digest(result); return result

def main():
    p=argparse.ArgumentParser();p.add_argument("--geometry-manifest",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    r=validate(json.loads(a.geometry_manifest.resolve().read_text())); out=a.output.resolve();out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix(out.suffix+".partial");tmp.write_text(json.dumps(r,sort_keys=True,indent=2)+"\n");tmp.replace(out);print(json.dumps({"status":r["status"],"court_identity_sha256":r["court_identity_sha256"]},sort_keys=True))
if __name__=="__main__":main()
