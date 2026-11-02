#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from tools.training.validate_corpus_mechanical_tessa_teacher_v4 import (
    _edge_surface_components,
    _partition,
    _rest_self_intersection,
    REQUIRED,
)

PARENT_SCHEMA="RealSaS.CorpusMechanicalTESSATeacher.v5"
PARENT_STATUS="PASS_CORPUS_MECHANICAL_TESSA_TEACHER_V5_CANDIDATE__INDEPENDENT_COURT_REQUIRED"
SCHEMA="RealSaS.CorpusMechanicalTESSATeacherCourt.v5"
STATUS="PASS_CORPUS_MECHANICAL_TESSA_V5_CONSUMER_COURT__CAUSAL_RESPONSE_AUTHORITY_SEPARATE"


def sha256(path: Path)->str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda:f.read(8<<20),b""): h.update(c)
    return h.hexdigest()


def bound(row,code):
    row=dict(row or {}); p=Path(str(row.get("path") or "")).resolve()
    if not p.is_file() or sha256(p)!=str(row.get("sha256") or ""): raise ValueError(code)
    return p


def validate(manifest:dict)->dict:
    if manifest.get("schema")!=PARENT_SCHEMA or manifest.get("status")!=PARENT_STATUS: raise ValueError("TESSA_V5_COURT_PARENT")
    parents=dict(manifest.get("parents") or {})
    mm_path=bound(parents.get("mechanical_surface_manifest"),"TESSA_V5_COURT_MECHANICAL_MANIFEST_DRIFT")
    mc_path=bound(parents.get("mechanical_surface_court"),"TESSA_V5_COURT_MECHANICAL_COURT_DRIFT")
    mm=json.loads(mm_path.read_text()); mc=json.loads(mc_path.read_text())
    if mm.get("schema")!="RealSaS.MechanicalSurfaceTopologyAuthority.v1": raise ValueError("TESSA_V5_COURT_MECHANICAL_SCHEMA")
    if mc.get("schema")!="RealSaS.MechanicalSurfaceTopologyAuthorityCourt.v1" or mc.get("status")!="PASS_MECHANICAL_SURFACE_TOPOLOGY_INDEPENDENT_COURT__FINAL_ADMISSION_SEPARATE": raise ValueError("TESSA_V5_COURT_MECHANICAL_COURT_STATUS")
    if not bool((mc.get("authority") or {}).get("independent_court_passed")): raise ValueError("TESSA_V5_COURT_MECHANICAL_NOT_QUALIFIED")
    topology_path=bound(mm.get("output"),"TESSA_V5_COURT_TOPOLOGY_DRIFT")
    with np.load(topology_path,allow_pickle=False) as z:
        source_v=np.asarray(z["mechanical_vertices_source_world"],np.float64); source_f=np.asarray(z["mechanical_faces"],np.int64); valid=np.asarray(z["face_valid"],bool); component=np.asarray(z["surface_component_id"],np.int32)
    if not np.all(valid): raise ValueError("TESSA_V5_COURT_INVALID_SOURCE_FACE")
    independent=_edge_surface_components(source_f)
    if _partition(independent)!=_partition(component): raise ValueError("TESSA_V5_COURT_COMPONENT_PARTITION")
    st=dict(manifest.get("source_topology") or {})
    if st.get("component_relation")!="QUALIFIED_MECHANICAL_EDGE_CONNECTED_SURFACE": raise ValueError("TESSA_V5_COURT_COMPONENT_RELATION")
    if st.get("sdf_sign_topology_consumed") is not False or st.get("appearance_addressing_consumed") is not False or st.get("render_vertex_identity_used_as_topology") is not False: raise ValueError("TESSA_V5_COURT_AUTHORITY_CROSS_CONTAMINATION")
    if int(st.get("qualified_cross_owner_seam_edge_count",-1))!=int((mm.get("counts") or {}).get("qualified_cross_owner_seam_edge_count",-2)): raise ValueError("TESSA_V5_COURT_SEAM_COUNT")
    mesh_path=bound(manifest.get("teacher_mesh"),"TESSA_V5_COURT_TEACHER_DRIFT")
    with np.load(mesh_path,allow_pickle=False) as z:
        vertices=np.asarray(z["vertices_source_world"],np.float64); faces=np.asarray(z["faces"],np.int64); face_id=np.asarray(z["source_triangle_id"],np.int64); bar=np.asarray(z["source_barycentric"],np.float64)
    if face_id.shape!=(len(vertices),) or bar.shape!=(len(vertices),3) or np.any(face_id<0) or np.any(face_id>=len(source_f)): raise ValueError("TESSA_V5_COURT_ADDRESS_LAYOUT")
    if not np.allclose(bar.sum(1),1,atol=1e-5,rtol=0): raise ValueError("TESSA_V5_COURT_BARYCENTRIC")
    reconstructed=np.sum(source_v[source_f[face_id]]*bar[...,None],axis=1); scale=max(float(np.linalg.norm(source_v.max(0)-source_v.min(0))),1e-9); max_error=float(np.max(np.linalg.norm(vertices-reconstructed,axis=1),initial=0.0)/scale)
    if max_error>.0105: raise ValueError("TESSA_V5_COURT_SOURCE_REBIND")
    selected=dict(manifest.get("selected") or {}); passed=[dict(x) for x in manifest.get("candidates") or [] if x.get("mechanical_pass") is True]
    if selected.get("mechanical_pass") is not True or not passed: raise ValueError("TESSA_V5_COURT_SELECTION")
    expected=min(passed,key=lambda x:(int(x.get("face_count",1<<60)),int(x.get("vertex_count",1<<60)),float(x.get("ratio",1e30))))
    for key in ("face_count","vertex_count","ratio","kind"):
        if selected.get(key)!=expected.get(key): raise ValueError("TESSA_V5_COURT_SELECTION_DRIFT:"+key)
    if selected.get("candidate_evaluation_contract") != "MEASURED_ALL_KINDS_NO_FALLBACK_OVERRIDE_V1":
        raise ValueError("TESSA_V5_COURT_LEGACY_CANDIDATE_REEVALUATION_REQUIRED")
    if selected.get("kind") not in {"SIMPLIFIED", "SOURCE_FALLBACK"}:
        raise ValueError("TESSA_V5_COURT_SELECTED_KIND")
    for key in REQUIRED:
        if selected.get(key) is not True:
            raise ValueError("TESSA_V5_COURT_DECLARED_PASS_MISSING:" + key)
    if _rest_self_intersection(vertices, faces):
        raise ValueError("TESSA_V5_COURT_REST_SELF_INTERSECTION")
    if (manifest.get("family_candidate_outcomes") or {}).get("CAUSAL_RESPONSE")!="EXISTING_EVIDENCE_CONSUMED__NEW_FAMILY_COURT_STILL_REQUIRED": raise ValueError("TESSA_V5_COURT_RESPONSE_OVERCLAIM")
    return {"schema":SCHEMA,"status":STATUS,"subject_id":manifest.get("subject_id"),"surface_component_count":int(len(np.unique(independent))),"qualified_cross_owner_seam_edge_count":int(st.get("qualified_cross_owner_seam_edge_count",0)),"source_rebind_max_error_over_scale":max_error,"family_candidate_outcomes":{"SURFACE_TOPOLOGY":"QUALIFIED_CONSUMER_CANDIDATE","CAUSAL_RESPONSE":"PENDING_NEW_FAMILY_COURT"},"authority":{"tessa_mechanical_surface_consumer_court_passed":True,"final_family_authority_minted":False,"training_authorized":False}}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--teacher-manifest",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args(); result=validate(json.loads(a.teacher_manifest.resolve().read_text())); a.output.resolve().parent.mkdir(parents=True,exist_ok=True); a.output.resolve().write_text(json.dumps(result,sort_keys=True,indent=2)+"\n"); print(json.dumps({"status":result["status"],"subject_id":result["subject_id"]},sort_keys=True))


if __name__=="__main__": main()
