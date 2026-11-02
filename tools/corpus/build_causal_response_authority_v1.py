#!/usr/bin/env python3
from __future__ import annotations

"""Build compact causal-response truth on qualified mechanical surface topology.

The Stage47 authored intervention tensor is the measured source consequence. This
producer does not infer semantic part classes or physical stiffness. It derives
continuous deformation/response evidence under an explicit control-balanced action
measure and stores only qualified face/tangent/pair rows. Unknown rows are absent,
not encoded as numeric zero targets.

Pair coherence/jump is computed on the independently-courted mechanical surface
connectivity, so qualified cross-owner modular seams participate in adjacency while
render/material/appearance ownership remains untouched.
"""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from tools.training.bounded_surface_response_v1 import build_surface_response_arrays_v1

SCHEMA="RealSaS.CausalResponseAuthority.v1"
STATUS="PASS_CAUSAL_RESPONSE_AUTHORITY_CANDIDATE__INDEPENDENT_COURT_REQUIRED"
MECH_SCHEMA="RealSaS.MechanicalSurfaceTopologyAuthority.v1"
MECH_STATUS="PASS_MECHANICAL_SURFACE_TOPOLOGY_CANDIDATE__INDEPENDENT_COURT_REQUIRED"
MECH_COURT_SCHEMA="RealSaS.MechanicalSurfaceTopologyAuthorityCourt.v1"
MECH_COURT_STATUS="PASS_MECHANICAL_SURFACE_TOPOLOGY_INDEPENDENT_COURT__FINAL_ADMISSION_SEPARATE"


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda:f.read(8<<20),b""): h.update(c)
    return h.hexdigest()


def digest(value)->str:
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def bound(row,code):
    row=dict(row or {}); p=Path(str(row.get("path") or "")).resolve()
    if not p.is_file() or sha256(p)!=str(row.get("sha256") or ""): raise ValueError(code)
    return p


def ref(path:Path)->dict:
    path=Path(path).resolve(); return {"path":str(path),"sha256":sha256(path)}


def write_npz(path:Path,**arrays):
    for name,value in arrays.items():
        a=np.asarray(value)
        if a.dtype==object or (np.issubdtype(a.dtype,np.floating) and not np.isfinite(a).all()): raise ValueError("CAUSAL_RESPONSE_INVALID_ARRAY:"+name)
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".partial.npz"); np.savez_compressed(tmp,**arrays); tmp.replace(path)


def collapse_trajectory(raw,mapping,count,scale):
    raw=np.asarray(raw,np.float64); mapping=np.asarray(mapping,np.int64)
    if raw.ndim!=3 or raw.shape[1] != len(mapping) or raw.shape[2]!=3: raise ValueError("CAUSAL_RESPONSE_TRAJECTORY_LAYOUT")
    out=np.zeros((len(raw),count,3),np.float64); seen=np.zeros(count,bool); max_drift=0.0
    tol=max(float(scale)*1e-9,1e-11)
    for rid,mid in enumerate(mapping):
        mid=int(mid)
        if mid<0: continue
        values=raw[:,rid,:]
        if not seen[mid]: out[:,mid,:]=values; seen[mid]=True
        else:
            drift=float(np.max(np.linalg.norm(values-out[:,mid,:],axis=-1),initial=0.0)); max_drift=max(max_drift,drift)
            if drift>tol: raise ValueError("CAUSAL_RESPONSE_MECHANICAL_TRAJECTORY_DRIFT")
    if not np.all(seen): raise ValueError("CAUSAL_RESPONSE_MECHANICAL_VERTEX_UNBOUND")
    return out,max_drift


def teacher_scene_geometry(scene):
    rows=[dict(x) for x in scene.get("geometry_payloads") or [] if Path(str(x.get("path") or "")).name=="geometry_and_response.npz"]
    if len(rows)!=1: raise ValueError("CAUSAL_RESPONSE_STAGE47_CARDINALITY")
    return rows[0]


def build(subject_id:str,mechanical_manifest_path:Path,mechanical_court_path:Path,teacher_scene_path:Path,output_dir:Path)->dict:
    mechanical_manifest_path=mechanical_manifest_path.resolve(); mechanical_court_path=mechanical_court_path.resolve(); teacher_scene_path=teacher_scene_path.resolve(); output_dir=output_dir.resolve()
    mm=json.loads(mechanical_manifest_path.read_text()); mc=json.loads(mechanical_court_path.read_text()); scene=json.loads(teacher_scene_path.read_text())
    if mm.get("schema")!=MECH_SCHEMA or mm.get("status")!=MECH_STATUS or mm.get("subject_id")!=subject_id: raise ValueError("CAUSAL_RESPONSE_MECHANICAL_MANIFEST")
    if mc.get("schema")!=MECH_COURT_SCHEMA or mc.get("status")!=MECH_COURT_STATUS or mc.get("subject_id")!=subject_id: raise ValueError("CAUSAL_RESPONSE_MECHANICAL_COURT")
    if not bool((mc.get("authority") or {}).get("independent_court_passed")): raise ValueError("CAUSAL_RESPONSE_MECHANICAL_COURT_PASS")
    if (mc.get("family_candidate_outcomes") or {}).get("SURFACE_TOPOLOGY")!="QUALIFIED_CORE_CANDIDATE": raise ValueError("CAUSAL_RESPONSE_SURFACE_TOPOLOGY_NOT_QUALIFIED")
    if scene.get("schema")!="RealSaS.TeacherSceneIR.v2" or scene.get("subject_key")!=subject_id: raise ValueError("CAUSAL_RESPONSE_TEACHER_SCENE")
    mech_path=bound(mm.get("output"),"CAUSAL_RESPONSE_MECHANICAL_TOPOLOGY_DRIFT")
    stage47_path=bound(teacher_scene_geometry(scene),"CAUSAL_RESPONSE_STAGE47_DRIFT")
    source_path=bound((mm.get("source") or {}).get("geometry_and_response"),"CAUSAL_RESPONSE_MECHANICAL_SOURCE_DRIFT")
    if stage47_path!=source_path or sha256(stage47_path)!=sha256(source_path): raise ValueError("CAUSAL_RESPONSE_SOURCE_BINDING")
    with np.load(mech_path,allow_pickle=False) as z:
        required={"raw_to_mechanical_vertex","mechanical_vertices_source_world","mechanical_faces","face_owner","face_valid","surface_component_id"}
        if not required.issubset(z.files): raise ValueError("CAUSAL_RESPONSE_MECHANICAL_ARRAYS")
        mapping=np.asarray(z["raw_to_mechanical_vertex"],np.int64); mv=np.asarray(z["mechanical_vertices_source_world"],np.float64); mf=np.asarray(z["mechanical_faces"],np.int64); face_owner=np.asarray(z["face_owner"],np.int32); geometry_face_valid=np.asarray(z["face_valid"],bool); component=np.asarray(z["surface_component_id"],np.int32)
    with np.load(stage47_path,allow_pickle=False) as z:
        required={"rest_vertices_source_world","faces","face_owner_privileged","isolated_position"}
        if not required.issubset(z.files): raise ValueError("CAUSAL_RESPONSE_STAGE47_ARRAYS")
        raw_rest=np.asarray(z["rest_vertices_source_world"],np.float64); raw_faces=np.asarray(z["faces"],np.int64); raw_owner=np.asarray(z["face_owner_privileged"],np.int32); raw_posed=np.asarray(z["isolated_position"],np.float64)
    if mapping.shape!=(len(raw_rest),) or not np.array_equal(mapping[raw_faces],mf): raise ValueError("CAUSAL_RESPONSE_FACE_REMAP")
    if not np.array_equal(raw_owner,face_owner): raise ValueError("CAUSAL_RESPONSE_FACE_OWNER")
    if geometry_face_valid.shape!=(len(mf),): raise ValueError("CAUSAL_RESPONSE_FACE_VALIDITY_LAYOUT")
    scale=max(float(np.linalg.norm(raw_rest.max(0)-raw_rest.min(0))),1e-9)
    posed,max_drift=collapse_trajectory(raw_posed,mapping,len(mv),scale)
    actions=list((scene.get("calibration") or {}).get("actions") or [])
    if not actions or len(actions)!=len(posed): raise ValueError("CAUSAL_RESPONSE_ACTION_BINDING")
    arrays=build_surface_response_arrays_v1(mv,mf,posed,actions)
    response_valid=np.asarray(arrays["surface_face_valid"],bool)&geometry_face_valid
    tangent_valid=np.asarray(arrays["surface_tangent_valid"],bool)&response_valid
    pair=np.asarray(arrays["response_pair_face"],np.int64)
    pair_valid=np.asarray(arrays["response_pair_valid"],bool)
    if pair.ndim!=2 or pair.shape[1:]!=(2,) or pair_valid.shape!=(len(pair),): raise ValueError("CAUSAL_RESPONSE_PAIR_LAYOUT")
    if len(pair): pair_valid &= response_valid[pair[:,0]] & response_valid[pair[:,1]]
    face_index=np.flatnonzero(response_valid).astype(np.int32); tangent_index=np.flatnonzero(tangent_valid).astype(np.int32); pair_index=np.flatnonzero(pair_valid).astype(np.int32)
    output_dir.mkdir(parents=True,exist_ok=True); arrays_path=output_dir/"CAUSAL_RESPONSE_AUTHORITY_V1.npz"
    write_npz(
        arrays_path,
        action_weight=np.asarray(arrays["action_weight"],np.float32),
        face_index=face_index,
        stretch_major=np.asarray(arrays["surface_stretch_major_mean"],np.float32)[face_index],
        stretch_minor=np.asarray(arrays["surface_stretch_minor_mean"],np.float32)[face_index],
        area_ratio=np.asarray(arrays["surface_area_ratio_mean"],np.float32)[face_index],
        anisotropy=np.asarray(arrays["surface_anisotropy_mean"],np.float32)[face_index],
        nonrigid_response=np.asarray(arrays["surface_nonrigid_response"],np.float32)[face_index],
        response_energy=np.asarray(arrays["surface_response_energy"],np.float32)[face_index],
        tangent_face_index=tangent_index,
        tangent_direction_2theta=np.asarray(arrays["surface_tangent_direction_2theta"],np.float32)[tangent_index],
        pair_face=pair[pair_index].astype(np.int32),
        pair_coherence=np.asarray(arrays["response_pair_coherence"],np.float32)[pair_index],
        pair_jump=np.asarray(arrays["response_pair_jump"],np.float32)[pair_index],
        face_owner=face_owner.astype(np.int32),
        surface_component_id=component.astype(np.int32),
    )
    cross_owner_pairs=int(np.count_nonzero(face_owner[pair[pair_index,0]]!=face_owner[pair[pair_index,1]])) if len(pair_index) else 0
    manifest={
        "schema":SCHEMA,"status":STATUS,"subject_id":subject_id,
        "parents":{"mechanical_surface_manifest":ref(mechanical_manifest_path),"mechanical_surface_court":ref(mechanical_court_path),"teacher_scene":ref(teacher_scene_path),"stage47_geometry_response":ref(stage47_path)},
        "output":{"path":str(arrays_path),"sha256":sha256(arrays_path)},
        "counts":{"mechanical_face_count":int(len(mf)),"qualified_face_count":int(len(face_index)),"qualified_tangent_face_count":int(len(tangent_index)),"qualified_pair_count":int(len(pair_index)),"qualified_cross_owner_pair_count":cross_owner_pairs,"action_count":int(len(actions)),"mechanical_surface_component_count":int(len(np.unique(component)))},
        "measurement_policy":{"action_measure":"CONTROL_BALANCED__EQUAL_MASS_PER_CONTROL_THEN_EQUAL_MASS_WITHIN_CONTROL","face_response":"AUTHORED_INTERVENTION_KINEMATIC_CONSEQUENCE__NOT_PHYSICAL_STIFFNESS","pair_adjacency":"QUALIFIED_MECHANICAL_SURFACE_FULL_EDGE_ADJACENCY","invalid_face_or_pair_target":"ABSENT_NOT_ZERO","semantic_part_labels_used":False,"appearance_addressing_used":False,"sdf_sign_topology_used":False},
        "maximum_mechanical_duplicate_trajectory_drift_source_units":float(max_drift),
        "family_candidate_outcomes":{"CAUSAL_RESPONSE":"QUALIFIED_MASKED_OR_CORE_CANDIDATE__INDEPENDENT_COURT_REQUIRED"},
        "authority":{"producer_candidate_only":True,"independent_court_required":True,"final_family_authority_minted":False,"training_authorized":False},
    }
    manifest["response_identity_sha256"]=digest(manifest); mp=output_dir/"CAUSAL_RESPONSE_AUTHORITY_MANIFEST_V1.json"; mp.write_text(json.dumps(manifest,sort_keys=True,indent=2,allow_nan=False)+"\n")
    return manifest


def main():
    p=argparse.ArgumentParser(); p.add_argument("--subject-id",required=True); p.add_argument("--mechanical-manifest",type=Path,required=True); p.add_argument("--mechanical-court",type=Path,required=True); p.add_argument("--teacher-scene",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); a=p.parse_args(); result=build(a.subject_id,a.mechanical_manifest,a.mechanical_court,a.teacher_scene,a.output_dir); print(json.dumps({"status":result["status"],"subject_id":result["subject_id"],"counts":result["counts"]},sort_keys=True))


if __name__=="__main__": main()
