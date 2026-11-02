#!/usr/bin/env python3
from __future__ import annotations

"""Build TESSA mechanical teacher from qualified mechanical surface topology.

V5 removes the remaining dependency on the owner-preserving SDF/sign topology.
Cross-owner seam identification is consumed only when it has passed the dedicated
mechanical-surface producer and independent court. Appearance/render addressing is
not consulted or rewritten here.
"""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from tools.training import build_corpus_mechanical_tessa_teacher_v4 as v4

SCHEMA="RealSaS.CorpusMechanicalTESSATeacher.v5"
STATUS="PASS_CORPUS_MECHANICAL_TESSA_TEACHER_V5_CANDIDATE__INDEPENDENT_COURT_REQUIRED"
MECH_SCHEMA="RealSaS.MechanicalSurfaceTopologyAuthority.v1"
MECH_STATUS="PASS_MECHANICAL_SURFACE_TOPOLOGY_CANDIDATE__INDEPENDENT_COURT_REQUIRED"
COURT_SCHEMA="RealSaS.MechanicalSurfaceTopologyAuthorityCourt.v1"
COURT_STATUS="PASS_MECHANICAL_SURFACE_TOPOLOGY_INDEPENDENT_COURT__FINAL_ADMISSION_SEPARATE"


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda:f.read(8<<20),b""): h.update(c)
    return h.hexdigest()


def ref(path: Path) -> dict:
    path=Path(path).resolve()
    return {"path":str(path),"sha256":sha256(path)}


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def bound(row,code):
    row=dict(row or {}); p=Path(str(row.get("path") or "")).resolve()
    if not p.is_file() or sha256(p)!=str(row.get("sha256") or ""): raise ValueError(code)
    return p


def build(*,subject_id,mechanical_manifest_path,mechanical_court_path,response_chain_path,camera_bundle_path,output_dir):
    mechanical_manifest_path=Path(mechanical_manifest_path).resolve(); mechanical_court_path=Path(mechanical_court_path).resolve()
    mm=json.loads(mechanical_manifest_path.read_text()); mc=json.loads(mechanical_court_path.read_text())
    if mm.get("schema")!=MECH_SCHEMA or mm.get("status")!=MECH_STATUS or mm.get("subject_id")!=subject_id: raise ValueError("TESSA_V5_MECHANICAL_AUTHORITY")
    if mc.get("schema")!=COURT_SCHEMA or mc.get("status")!=COURT_STATUS or mc.get("subject_id")!=subject_id: raise ValueError("TESSA_V5_MECHANICAL_COURT")
    if not bool((mc.get("authority") or {}).get("independent_court_passed")): raise ValueError("TESSA_V5_MECHANICAL_COURT_PASS")
    if (mc.get("family_candidate_outcomes") or {}).get("SURFACE_TOPOLOGY")!="QUALIFIED_CORE_CANDIDATE": raise ValueError("TESSA_V5_SURFACE_TOPOLOGY_NOT_QUALIFIED")
    topology_path=bound(mm.get("output"),"TESSA_V5_MECHANICAL_TOPOLOGY_DRIFT")
    with np.load(topology_path,allow_pickle=False) as z:
        required={"raw_to_mechanical_vertex","mechanical_vertices_source_world","mechanical_faces","face_owner","face_valid","surface_component_id"}
        if not required.issubset(z.files): raise ValueError("TESSA_V5_MECHANICAL_ARRAYS")
        raw_to_m=np.asarray(z["raw_to_mechanical_vertex"],np.int64)
        source_v_declared=np.asarray(z["mechanical_vertices_source_world"],np.float64)
        source_f=np.asarray(z["mechanical_faces"],np.int64)
        source_owner=np.asarray(z["face_owner"],np.int32)
        face_valid=np.asarray(z["face_valid"],bool)
        source_fc=np.asarray(z["surface_component_id"],np.int32)
    if not np.all(face_valid): raise ValueError("TESSA_V5_INVALID_SOURCE_FACE_REPAIR_REQUIRED")

    response_chain_path=Path(response_chain_path).resolve(); response_chain=json.loads(response_chain_path.read_text())
    scene_path,scene=v4._teacher_scene_from_response_chain(response_chain,subject_id)
    stage47=v4._stage47_geometry(scene)
    source_stage47=bound((mm.get("source") or {}).get("geometry_and_response"),"TESSA_V5_MECHANICAL_SOURCE_DRIFT")
    if stage47!=source_stage47 or sha256(stage47)!=sha256(source_stage47): raise ValueError("TESSA_V5_STAGE47_MECHANICAL_SOURCE_BINDING")
    camera_bundle_path=Path(camera_bundle_path).resolve(); camera=json.loads(camera_bundle_path.read_text())
    replay_path=bound((response_chain.get("inputs") or {}).get("admitted_observation_import"),"TESSA_V5_REPLAY_DRIFT"); replay=json.loads(replay_path.read_text())
    with np.load(stage47,allow_pickle=False) as z:
        raw_rest=np.asarray(z["rest_vertices_source_world"],np.float64); raw_faces=np.asarray(z["faces"],np.int64); raw_owner=np.asarray(z["face_owner_privileged"],np.int32); raw_isolated=np.asarray(z["isolated_position"],np.float64)
    if raw_to_m.shape!=(len(raw_rest),) or source_f.shape!=raw_faces.shape or not np.array_equal(raw_to_m[raw_faces],source_f): raise ValueError("TESSA_V5_STAGE47_FACE_REMAP")
    if not np.array_equal(raw_owner,source_owner): raise ValueError("TESSA_V5_FACE_OWNER_BINDING")
    scale=max(float(np.linalg.norm(raw_rest.max(0)-raw_rest.min(0))),1e-9); tol=max(scale*1e-9,1e-11)
    source_v,rest_drift=v4.collapse_positions(raw_rest,raw_to_m,len(source_v_declared),tol,"TESSA_V5_REST")
    isolated,posed_drift=v4.collapse_positions(raw_isolated,raw_to_m,len(source_v_declared),tol,"TESSA_V5_POSED")
    declared_drift=float(np.max(np.linalg.norm(source_v-source_v_declared,axis=1),initial=0.0))
    if declared_drift>max(scale*2e-6,2e-7): raise ValueError("TESSA_V5_DECLARED_REST_DRIFT")

    actions=list((scene.get("calibration") or {}).get("actions") or []); cameras=list(camera.get("cameras") or [])
    if len(actions)!=len(isolated) or len(cameras)!=8: raise ValueError("TESSA_V5_ACTION_CAMERA_BINDING")
    attachment_ids={i for i,obj in enumerate(replay.get("objects") or []) if obj.get("parent_bone")}
    source_topology=v4.impl.topology_signature(len(source_v),source_f,source_fc)
    ladder=v4.impl.simplify_by_component(source_v,source_f,[.35,.50,.65,.80,1.0],source_fc)
    rows=[]; qualified=[]
    source_context=v4.impl.source_evaluation_context(source_v,source_f,isolated,actions,cameras)
    for ratio,vertices,faces,kind in ladder:
        metrics,face_id,bar=v4.impl.evaluate_candidate(vertices,faces,source_v,source_f,source_fc,source_topology,source_owner,attachment_ids,isolated,actions,cameras,ratio,kind,source_context=source_context)
        rows.append(metrics)
        if metrics["mechanical_pass"]: qualified.append((metrics["face_count"],metrics["vertex_count"],ratio,vertices,faces,face_id,bar,metrics))
    diagnostic=v4.impl.write_candidate_diagnostics(subject_id=subject_id,output_dir=output_dir,candidates=rows,
        parents={"mechanical_manifest":ref(mechanical_manifest_path),"mechanical_court":ref(mechanical_court_path),"response_chain":ref(response_chain_path),"camera_bundle":ref(camera_bundle_path)},producer_files=[Path(__file__),Path(v4.__file__),Path(v4.impl.__file__)])
    if not qualified: raise RuntimeError("TESSA_V5_NO_MECHANICALLY_ADMISSIBLE_FALLBACK:"+str(diagnostic))
    qualified.sort(key=lambda x:(x[0],x[1],x[2])); _,_,ratio,vertices,faces,face_id,bar,selected=qualified[0]
    output_dir=Path(output_dir).resolve(); output_dir.mkdir(parents=True,exist_ok=True); mesh_path=output_dir/"CORPUS_MECHANICAL_TESSA_TEACHER_V5.npz"
    v4.impl.write_npz(mesh_path,vertices_source_world=vertices.astype(np.float32),faces=faces.astype(np.int32),source_triangle_id=face_id.astype(np.int32),source_barycentric=bar.astype(np.float32))
    manifest={
        "schema":SCHEMA,"status":STATUS,"subject_id":subject_id,
        "parents":{"mechanical_surface_manifest":ref(mechanical_manifest_path),"mechanical_surface_court":ref(mechanical_court_path),"response_chain":ref(response_chain_path),"teacher_scene":ref(scene_path),"stage47_geometry_response":ref(stage47),"camera_bundle":ref(camera_bundle_path),"admitted_observation_import":ref(replay_path)},
        "source_topology":{"raw_render_vertex_count":int(len(raw_rest)),"mechanical_topology_vertex_count":int(len(source_v)),"face_count":int(len(source_f)),"surface_component_count":int(len(np.unique(source_fc))),"qualified_cross_owner_seam_edge_count":int((mm.get("counts") or {}).get("qualified_cross_owner_seam_edge_count",0)),"component_relation":"QUALIFIED_MECHANICAL_EDGE_CONNECTED_SURFACE","sdf_sign_topology_consumed":False,"appearance_addressing_consumed":False,"render_vertex_identity_used_as_topology":False,"maximum_rest_duplicate_trajectory_drift":float(rest_drift),"maximum_posed_duplicate_trajectory_drift":float(posed_drift),"maximum_declared_rest_drift_over_scale":float(declared_drift/scale)},
        "policy":{"mechanical_admissibility_precedes_compactness":True,"simplification_ladder_ratios":[.35,.5,.65,.8,1.0],"source_topology_is_fallback_member":True,"selection":"MIN_FACE_COUNT_THEN_VERTEX_COUNT_AMONG_PASS","mechanical_topology_authority":"MECHANICAL_SURFACE_TOPOLOGY_INDEPENDENT_COURT","sdf_sign_authority":"NOT_CONSUMED","appearance_addressing":"NOT_CONSUMED"},
        "candidates":rows,"selected":selected,"teacher_mesh":ref(mesh_path),
        "family_candidate_outcomes":{"SURFACE_TOPOLOGY":"QUALIFIED_CONSUMER_CANDIDATE","CAUSAL_RESPONSE":"EXISTING_EVIDENCE_CONSUMED__NEW_FAMILY_COURT_STILL_REQUIRED"},
        "authority":{"mechanical_surface_independent_court_passed":True,"tessa_teacher_candidate_only":True,"final_family_authority_minted":False,"training_authorized":False},
    }
    manifest["teacher_identity_sha256"]=digest(manifest); mp=output_dir/"CORPUS_MECHANICAL_TESSA_TEACHER_MANIFEST_V5.json"; mp.write_text(json.dumps(manifest,sort_keys=True,indent=2,allow_nan=False)+"\n")
    return manifest


def main():
    p=argparse.ArgumentParser(); p.add_argument("--subject-id",required=True); p.add_argument("--mechanical-manifest",type=Path,required=True); p.add_argument("--mechanical-court",type=Path,required=True); p.add_argument("--response-chain",type=Path,required=True); p.add_argument("--camera-bundle",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); a=p.parse_args()
    result=build(subject_id=a.subject_id,mechanical_manifest_path=a.mechanical_manifest,mechanical_court_path=a.mechanical_court,response_chain_path=a.response_chain,camera_bundle_path=a.camera_bundle,output_dir=a.output_dir); print(json.dumps({"status":result["status"],"subject_id":result["subject_id"],"selected":result["selected"]},sort_keys=True))


if __name__=="__main__": main()
