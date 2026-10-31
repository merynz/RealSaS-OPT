#!/usr/bin/env python3
from __future__ import annotations

"""Independent court for SourceObservationAuthority.v1.

The court reopens every raw/control/view file and independently verifies source
immutability, bind-rest lineage, normalization/camera agreement, 8-view hash
bindings and admitted observation tolerances. It never imports the producer.
"""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

PARENT_SCHEMA="RealSaS.SourceObservationAuthority.v1"
PARENT_STATUS="PASS_SOURCE_OBSERVATION_AUTHORITY_CANDIDATE__INDEPENDENT_COURT_REQUIRED"
SCHEMA="RealSaS.SourceObservationAuthorityCourt.v1"
STATUS="PASS_SOURCE_OBSERVATION_INDEPENDENT_COURT__FINAL_ADMISSION_SEPARATE"


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda:f.read(8<<20),b""): h.update(c)
    return h.hexdigest()


def load_bound(row,code):
    row=dict(row or {}); p=Path(str(row.get("path") or "")).resolve()
    if not p.is_file() or sha256(p)!=str(row.get("sha256") or ""): raise ValueError(code)
    return p,json.loads(p.read_text())


def matrix(value,code):
    m=np.asarray(value,np.float64)
    if m.shape!=(4,4) or not np.isfinite(m).all() or not np.allclose(m[3],[0,0,0,1],atol=1e-10,rtol=0): raise ValueError(code)
    return m


def validate(manifest_path:Path)->dict:
    manifest_path=manifest_path.resolve(); m=json.loads(manifest_path.read_text())
    if m.get("schema")!=PARENT_SCHEMA or m.get("status")!=PARENT_STATUS: raise ValueError("SOURCE_OBSERVATION_COURT_PARENT")
    subject=str(m.get("subject_id") or ""); source=Path(str((m.get("source") or {}).get("path") or "")).resolve()
    if not subject or not source.is_file() or sha256(source)!=str((m.get("source") or {}).get("sha256") or ""): raise ValueError("SOURCE_OBSERVATION_COURT_SOURCE")
    controls=dict(m.get("controls") or {}); candidate_path,candidate=load_bound(controls.get("candidate"),"SOURCE_OBSERVATION_COURT_CANDIDATE"); snapshot_path,snapshot=load_bound(controls.get("snapshot"),"SOURCE_OBSERVATION_COURT_SNAPSHOT"); proof_path,proof=load_bound(controls.get("proof"),"SOURCE_OBSERVATION_COURT_PROOF"); camera_path,camera=load_bound(controls.get("camera"),"SOURCE_OBSERVATION_COURT_CAMERA")
    source_sha=sha256(source)
    if candidate.get("schema")!="RealSaS.Controlled8ViewSourceRestCandidate.v2" or snapshot.get("schema")!="RealSaS.ImportedSourceSceneSnapshot.v2" or proof.get("schema")!="RealSaS.IrisBindObservationAdmission.v1" or camera.get("schema")!="RealSaS.CameraProjectionBundle.v1": raise ValueError("SOURCE_OBSERVATION_COURT_SCHEMA")
    if str(candidate.get("source_sha256"))!=source_sha or str(snapshot.get("source_sha256"))!=source_sha or str(proof.get("source_sha256"))!=source_sha: raise ValueError("SOURCE_OBSERVATION_COURT_SOURCE_BINDING")
    if str(candidate.get("subject_id"))!=subject or str(proof.get("subject_id"))!=subject: raise ValueError("SOURCE_OBSERVATION_COURT_SUBJECT")
    if str(proof.get("candidate_sha256"))!=sha256(candidate_path) or str(proof.get("camera_sha256"))!=sha256(camera_path) or str((candidate.get("source_scene_snapshot") or {}).get("sha256"))!=sha256(snapshot_path) or str((candidate.get("camera_bundle") or {}).get("sha256"))!=sha256(camera_path): raise ValueError("SOURCE_OBSERVATION_COURT_CONTROL_BINDING")
    if candidate.get("raw_source_mutated",(candidate.get("import") or {}).get("raw_source_mutated")) is not False or candidate.get("pose_address")!="SOURCE_BIND_REST" or candidate.get("native_frame0_is_rest_assumed") is not False: raise ValueError("SOURCE_OBSERVATION_COURT_REST_CONTRACT")
    if candidate.get("training_admission_claimed") is not False or candidate.get("product_pass_claimed") is not False: raise ValueError("SOURCE_OBSERVATION_COURT_AUTHORITY_ESCALATION")
    consistency=dict(candidate.get("internal_consistency") or {})
    if consistency.get("animation_free_rest") is not True or consistency.get("normalization_uniform") is not True: raise ValueError("SOURCE_OBSERVATION_COURT_INTERNAL_CONSISTENCY")
    norm=matrix(snapshot.get("source_world_to_canonical"),"SOURCE_OBSERVATION_COURT_NORMALIZATION"); inverse=matrix(snapshot.get("canonical_to_source_world"),"SOURCE_OBSERVATION_COURT_NORMALIZATION_INV")
    if not np.allclose(norm@inverse,np.eye(4),atol=1e-6,rtol=0): raise ValueError("SOURCE_OBSERVATION_COURT_NORMALIZATION_INV")
    gram=norm[:3,:3].T@norm[:3,:3]
    if not np.allclose(gram/np.trace(gram),np.eye(3)/3,atol=1e-6,rtol=0) or np.linalg.slogdet(norm[:3,:3])[0]<=0: raise ValueError("SOURCE_OBSERVATION_COURT_UNIFORM_ORIENTATION")
    nodes=list(snapshot.get("nodes") or []); index={str(x.get("node_id")):x for x in nodes}
    if not nodes or len(index)!=len(nodes) or "" in index: raise ValueError("SOURCE_OBSERVATION_COURT_NODE_SET")
    for row in nodes:
        node_id=str(row["node_id"]); seen={node_id}; parent=row.get("parent_node_id")
        while parent is not None:
            if parent not in index or parent in seen: raise ValueError("SOURCE_OBSERVATION_COURT_PARENT_GRAPH")
            seen.add(parent); parent=index[parent].get("parent_node_id")
        world=matrix(row.get("node_to_source_world"),"SOURCE_OBSERVATION_COURT_WORLD"); local=matrix(row.get("effective_node_to_parent"),"SOURCE_OBSERVATION_COURT_LOCAL"); parent_id=row.get("parent_node_id"); pw=np.eye(4) if parent_id is None else matrix(index[parent_id].get("node_to_source_world"),"SOURCE_OBSERVATION_COURT_PARENT_WORLD")
        if not np.allclose(world,pw@local,atol=1e-6,rtol=1e-6): raise ValueError("SOURCE_OBSERVATION_COURT_PARENT_WORLD_DRIFT")
        canonical=matrix(row.get("node_to_canonical"),"SOURCE_OBSERVATION_COURT_CANONICAL")
        if not np.allclose(canonical,norm@world,atol=1e-6,rtol=1e-6): raise ValueError("SOURCE_OBSERVATION_COURT_CANONICAL_BINDING")
        if any(int(row.get(k,0))!=0 for k in ("active_action_count","active_nla_track_count","driver_count")): raise ValueError("SOURCE_OBSERVATION_COURT_LIVE_REST")
    if camera.get("per_view_recenter") is not False or camera.get("per_view_pose_change") is not False or len(camera.get("cameras") or [])!=8: raise ValueError("SOURCE_OBSERVATION_COURT_CAMERA_POLICY")
    half=float(camera.get("common_half_extent",0)); center=np.asarray(camera.get("common_center_xyz"),np.float64)
    if half<=0 or not np.isfinite(half) or center.shape!=(3,) or not np.isfinite(center).all(): raise ValueError("SOURCE_OBSERVATION_COURT_CAMERA_EXTENT")
    expected=np.eye(4); expected[:3,:3]/=half; expected[:3,3]=-center/half
    if not np.allclose(norm,expected,atol=1e-6,rtol=1e-6): raise ValueError("SOURCE_OBSERVATION_COURT_CAMERA_NORMALIZATION")
    if proof.get("status")!="PASS_IRIS_BIND_OBSERVATIONS" or proof.get("scope")!="IMAGE_DERIVED_FSTAR_BIND_OBSERVATIONS__NOT_FULL_MECHANICAL_TRUTH": raise ValueError("SOURCE_OBSERVATION_COURT_PROOF")
    policy=dict(proof.get("policy") or {})
    if int(policy.get("resolution",0))!=1024 or int(policy.get("view_count",0))!=8 or int(policy.get("minimum_border_margin_px",-1))!=8 or float(policy.get("maximum_relative_vertex_error",float("inf")))!=1e-5 or float(policy.get("maximum_mask_gap_px",float("inf")))!=2.0 or policy.get("reference_coverage")!="TRIANGLE_UNIT_PIXEL_SQUARE_INTERSECTION" or policy.get("pose")!="SOURCE_BIND_REST" or policy.get("native_frame0_is_bind_assumed") is not False: raise ValueError("SOURCE_OBSERVATION_COURT_PROOF_POLICY")
    if float(proof.get("relative_vertex_error",float("inf")))>1e-5: raise ValueError("SOURCE_OBSERVATION_COURT_VERTEX_PARITY")
    cv={int(x.get("view_index",-1)):x for x in candidate.get("views") or []}; pv={int(x.get("view_index",-1)):x for x in proof.get("views") or []}; manifest_views={int(x.get("view_index",-1)):x for x in m.get("views") or []}
    if set(cv)!=set(range(8)) or set(pv)!=set(range(8)) or set(manifest_views)!=set(range(8)): raise ValueError("SOURCE_OBSERVATION_COURT_VIEW_SET")
    min_margin=1<<30; max_gap=0.0
    for i in range(8):
        mr=manifest_views[i]; raster=Path(str((mr.get("raster") or {}).get("path") or "")).resolve(); mask=Path(str((mr.get("mask") or {}).get("path") or "")).resolve()
        if not raster.is_file() or not mask.is_file(): raise ValueError("SOURCE_OBSERVATION_COURT_VIEW_FILE")
        rsha,msha=sha256(raster),sha256(mask)
        expected_r=[str((cv[i].get("raster") or {}).get("sha256")),str(pv[i].get("raster_sha256")),str((mr.get("raster") or {}).get("sha256"))]
        expected_m=[str((cv[i].get("foreground_mask") or {}).get("sha256")),str(pv[i].get("mask_sha256")),str((mr.get("mask") or {}).get("sha256"))]
        if any(x!=rsha for x in expected_r) or any(x!=msha for x in expected_m): raise ValueError("SOURCE_OBSERVATION_COURT_VIEW_HASH")
        with Image.open(raster) as image: rgba=np.asarray(image.convert("RGBA"),np.uint8)
        raw_mask=np.frombuffer(mask.read_bytes(),np.uint8)
        if rgba.shape!=(1024,1024,4) or raw_mask.size!=1024*1024 or np.any((raw_mask!=0)&(raw_mask!=1)): raise ValueError("SOURCE_OBSERVATION_COURT_VIEW_LAYOUT")
        if hashlib.sha256(rgba.tobytes()).hexdigest()!=str((mr.get("raster") or {}).get("decoded_rgba_sha256")): raise ValueError("SOURCE_OBSERVATION_COURT_DECODED_RGBA")
        if int(np.count_nonzero(raw_mask))!=int((mr.get("mask") or {}).get("foreground_pixel_count",-1)): raise ValueError("SOURCE_OBSERVATION_COURT_MASK_COUNT")
        gap=float(pv[i].get("maximum_mask_gap_px",float("inf"))); margin=int(pv[i].get("border_margin_px",-1)); max_gap=max(max_gap,gap); min_margin=min(min_margin,margin)
    if max_gap>2.0 or min_margin<8: raise ValueError("SOURCE_OBSERVATION_COURT_VIEW_TOLERANCE")
    if (m.get("family_candidate_outcomes") or {})!={"SOURCE_IMPORT":"QUALIFIED_CORE_CANDIDATE","CAMERA_OBSERVATION":"QUALIFIED_CORE_CANDIDATE"}: raise ValueError("SOURCE_OBSERVATION_COURT_OUTCOME_OVERCLAIM")
    return {"schema":SCHEMA,"status":STATUS,"subject_id":subject,"source_sha256":source_sha,"view_count":8,"maximum_mask_gap_px":max_gap,"minimum_border_margin_px":min_margin,"family_candidate_outcomes":{"SOURCE_IMPORT":"QUALIFIED_CORE_CANDIDATE","CAMERA_OBSERVATION":"QUALIFIED_CORE_CANDIDATE"},"authority":{"independent_court_passed":True,"final_family_authority_minted":False,"training_authorized":False},"parent_manifest":{"path":str(manifest_path),"sha256":sha256(manifest_path)}}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args(); r=validate(a.manifest); a.output.resolve().parent.mkdir(parents=True,exist_ok=True); a.output.resolve().write_text(json.dumps(r,sort_keys=True,indent=2)+"\n"); print(json.dumps({"status":r["status"],"subject_id":r["subject_id"]},sort_keys=True))


if __name__=="__main__": main()
