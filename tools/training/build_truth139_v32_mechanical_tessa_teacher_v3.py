#!/usr/bin/env python3
"""Build mechanically-qualified TESSA teacher on geometric topology V2.

V1 inherited render-vertex seam splits from geometry V1. V3 keeps the prior 3D
self-intersection fix, collapses Stage47 rest/posed tensors through the exact
geometric quotient, proves duplicate trajectories coincide, then runs the
existing mechanical simplification courts on the repaired mesh.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

from tools.training import build_truth139_v32_mechanical_tessa_teacher_v1 as impl
from tools.training.truth139_v32_spatial_index_v1 import new_rtree3

SCHEMA="RealSaS.Truth139V32MechanicalTESSATeacher.v3"
STATUS="PASS_V32_GEOMETRIC_TOPOLOGY_MECHANICALLY_QUALIFIED_TESSA_TEACHER"


def has_self_intersection_3d(vertices, faces):
    v=np.asarray(vertices,dtype=np.float64); f=np.asarray(faces,dtype=np.int64); tri=v[f]; spatial=new_rtree3()
    for i,t in enumerate(tri):
        lo=t.min(0); hi=t.max(0); spatial.insert(i,(float(lo[0]),float(lo[1]),float(lo[2]),float(hi[0]),float(hi[1]),float(hi[2])))
    for i,t in enumerate(tri):
        lo=t.min(0); hi=t.max(0); bounds=(float(lo[0]),float(lo[1]),float(lo[2]),float(hi[0]),float(hi[1]),float(hi[2]))
        for j in spatial.intersection(bounds):
            if j<=i or len(set(map(int,f[i]))&set(map(int,f[j]))): continue
            if impl.tri_intersect_sat(t,tri[j]): return True
    return False


impl.has_self_intersection=has_self_intersection_3d


def collapse_positions(raw_positions, raw_to_topology, topology_vertex_count, tolerance, code):
    raw=np.asarray(raw_positions,dtype=np.float64); mapping=np.asarray(raw_to_topology,dtype=np.int64)
    if raw.ndim==2: raw=raw[None,...]; squeeze=True
    elif raw.ndim==3: squeeze=False
    else: raise ValueError(code+"_LAYOUT")
    if raw.shape[1]!=len(mapping): raise ValueError(code+"_CARDINALITY")
    out=np.zeros((raw.shape[0],topology_vertex_count,3),dtype=np.float64); seen=np.zeros(topology_vertex_count,dtype=np.bool_); max_drift=0.0
    for raw_index,top in enumerate(mapping):
        top=int(top)
        if top<0: continue
        value=raw[:,raw_index,:]
        if not seen[top]: out[:,top,:]=value; seen[top]=True
        else:
            drift=float(np.max(np.linalg.norm(value-out[:,top,:],axis=-1),initial=0.0)); max_drift=max(max_drift,drift)
            if drift>tolerance: raise ValueError(code+"_DUPLICATE_TRAJECTORY_DRIFT")
    if not np.all(seen): raise ValueError(code+"_TOPOLOGY_VERTEX_UNBOUND")
    return (out[0] if squeeze else out),max_drift


def build(subject_id,geometry_manifest_path,response_chain_path,camera_bundle,output_dir):
    gm=json.loads(geometry_manifest_path.resolve().read_text()); rc=json.loads(response_chain_path.resolve().read_text()); camera=json.loads(camera_bundle.resolve().read_text())
    if gm.get("schema")!="RealSaS.Truth139V32ExactGeometry.v2" or gm.get("status")!="PASS_V32_GEOMETRIC_TOPOLOGY_SDF_V2__INDEPENDENT_COURT_REQUIRED": raise ValueError("V32_TESSA_V3_GEOMETRY_NOT_V2_CANDIDATE")
    topology_path=impl.load_ref(gm["outputs"]["topology"],"V32_TESSA_V3_TOPOLOGY_DRIFT")
    scene_path=impl.load_ref(rc["outputs"]["teacher_scene"],"V32_TESSA_V3_RESPONSE_SCENE_DRIFT"); scene=json.loads(scene_path.read_text())
    stage47=impl.load_ref(next(r for r in scene["geometry_payloads"] if Path(r["path"]).name=="geometry_and_response.npz"),"V32_TESSA_V3_STAGE47_GEOMETRY_DRIFT")
    replay=json.loads(impl.load_ref(rc["inputs"]["admitted_observation_import"],"V32_TESSA_V3_REPLAY_DRIFT").read_text())
    with np.load(topology_path,allow_pickle=False) as z:
        raw_to_q=np.asarray(z["raw_to_topology_vertex"],np.int64); source_f=np.asarray(z["topology_faces"],np.int64); source_fc=np.asarray(z["face_component_id"],np.int32); qowner=np.asarray(z["topology_vertex_owner"],np.int32); qv_declared=np.asarray(z["topology_vertices_source_world"],np.float64)
    with np.load(stage47,allow_pickle=False) as z:
        raw_rest=np.asarray(z["rest_vertices_source_world"],np.float64); raw_faces=np.asarray(z["faces"],np.int64); raw_owner=np.asarray(z["face_owner_privileged"],np.int32); raw_isolated=np.asarray(z["isolated_position"],np.float64)
    if len(raw_to_q)!=len(raw_rest) or len(raw_faces)!=len(source_f): raise ValueError("V32_TESSA_V3_RAW_TOPOLOGY_BINDING")
    scale=max(float(np.linalg.norm(raw_rest.max(0)-raw_rest.min(0))),1e-9); tolerance=max(scale*1e-9,1e-11)
    source_v,rest_duplicate_drift=collapse_positions(raw_rest,raw_to_q,len(qv_declared),tolerance,"V32_TESSA_V3_REST")
    isolated,posed_duplicate_drift=collapse_positions(raw_isolated,raw_to_q,len(qv_declared),tolerance,"V32_TESSA_V3_POSED")
    if np.max(np.abs(source_v-qv_declared),initial=0.0)>max(scale*2e-6,2e-7): raise ValueError("V32_TESSA_V3_QUOTIENT_REST_DRIFT")
    if np.any(qowner[source_f]!=qowner[source_f][:,:1]): raise ValueError("V32_TESSA_V3_FACE_OWNER_CROSS")
    source_face_object=qowner[source_f[:,0]]; attachment_ids={i for i,obj in enumerate(replay.get("objects") or []) if obj.get("parent_bone")}
    actions=list((scene.get("calibration") or {}).get("actions") or []); cameras=list(camera.get("cameras") or [])
    if len(actions)!=len(isolated) or len(cameras)!=8: raise ValueError("V32_TESSA_V3_ACTION_CAMERA_BINDING")
    source_topology=impl.topology_signature(len(source_v),source_f,source_fc); ladder=impl.simplify_by_component(source_v,source_f,[.35,.50,.65,.80,1.0],source_fc); rows=[]; qualified=[]
    for ratio,v,f,kind in ladder:
        metrics,face_id,bar=impl.evaluate_candidate(v,f,source_v,source_f,source_fc,source_topology,source_face_object,attachment_ids,isolated,actions,cameras,ratio,kind); rows.append(metrics)
        if metrics["mechanical_pass"]: qualified.append((metrics["face_count"],metrics["vertex_count"],ratio,v,f,face_id,bar,metrics))
    if not qualified: raise RuntimeError("V32_TESSA_V3_NO_MECHANICALLY_ADMISSIBLE_FALLBACK")
    qualified.sort(key=lambda x:(x[0],x[1],x[2])); _,_,ratio,v,f,face_id,bar,selected=qualified[0]
    out=output_dir.resolve(); out.mkdir(parents=True,exist_ok=True); mesh=out/"V32_MECHANICAL_TESSA_TEACHER_V3.npz"
    impl.write_npz(mesh,vertices_source_world=v.astype(np.float32),faces=f.astype(np.int32),source_triangle_id=face_id.astype(np.int32),source_barycentric=bar.astype(np.float32))
    manifest={"schema":SCHEMA,"status":STATUS,"subject_id":subject_id,"parents":{"geometry_manifest":impl.ref(geometry_manifest_path),"response_chain":impl.ref(response_chain_path),"camera_bundle":impl.ref(camera_bundle)},"geometric_topology":{"source_raw_vertex_count":int(len(raw_rest)),"source_topology_vertex_count":int(len(source_v)),"source_face_count":int(len(source_f)),"component_count":int(len(np.unique(source_fc))),"maximum_rest_duplicate_trajectory_drift":float(rest_duplicate_drift),"maximum_posed_duplicate_trajectory_drift":float(posed_duplicate_drift),"render_vertex_identity_used_as_topology":False},"policy":{"mechanical_admissibility_precedes_compactness":True,"simplification_ladder_ratios":[.35,.5,.65,.8,1.0],"source_topology_is_fallback_member":True,"teacher_topology_is_student_conditioning":False,"selection":"MIN_FACE_COUNT_THEN_VERTEX_COUNT_AMONG_PASS"},"candidates":rows,"selected":selected,"teacher_mesh":impl.ref(mesh),"training_performed":False,"qualification_minted":False}
    manifest["teacher_identity_sha256"]=impl.digest(manifest); mp=out/"V32_MECHANICAL_TESSA_TEACHER_MANIFEST_V3.json"; mp.write_text(json.dumps(manifest,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":STATUS,"subject_id":subject_id,"selected_face_count":selected["face_count"],"selected_kind":selected["kind"],"source_topology_vertex_count":len(source_v),"teacher_identity_sha256":manifest["teacher_identity_sha256"]},sort_keys=True)); return manifest


def main():
    p=argparse.ArgumentParser(); p.add_argument("--subject-id",required=True); p.add_argument("--geometry-manifest",type=Path,required=True); p.add_argument("--response-chain",type=Path,required=True); p.add_argument("--camera-bundle",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); a=p.parse_args(); build(a.subject_id,a.geometry_manifest,a.response_chain,a.camera_bundle,a.output_dir)


if __name__=="__main__": main()
