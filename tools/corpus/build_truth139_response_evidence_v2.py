#!/usr/bin/env python3
"""Re-materialize Truth139 IRIS response evidence with geometric adjacency V2.

Stage47 authored intervention tensors are reused byte-for-byte. Only derived
response evidence is rebuilt; Blender/teacher evaluation is not rerun.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np

from models.iris.truth.presentation_response_v2 import build_surface_response_arrays_v2

SCHEMA = "RealSaS.Truth139IrisResponseEvidence.v2"
STATUS = "PASS_IRIS_RESPONSE_EVIDENCE_V2__INDEPENDENT_COURT_REQUIRED"


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda:f.read(8<<20),b""): h.update(c)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def bound_ref(ref, code):
    p=Path(str((ref or {}).get("path") or "")).resolve()
    if not p.is_file() or sha256(p)!=str((ref or {}).get("sha256") or ""):
        raise ValueError(code)
    return p


def geometry_ref(scene):
    rows=[dict(row) for row in scene.get("geometry_payloads") or () if Path(str(row.get("path") or "")).name=="geometry_and_response.npz"]
    if len(rows)!=1: raise ValueError("TRUTH139_RESPONSE_V2_STAGE47_GEOMETRY_CARDINALITY")
    return rows[0]


def save_npz(path, arrays):
    clean={}
    for name,value in arrays.items():
        a=np.asarray(value)
        if a.dtype==object or (np.issubdtype(a.dtype,np.floating) and not np.isfinite(a).all()):
            raise ValueError("TRUTH139_RESPONSE_V2_INVALID_ARRAY:"+name)
        clean[name]=a
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".partial.npz")
    np.savez_compressed(tmp,**clean); tmp.replace(path)


def build(*, subject_id, teacher_scene, output_dir):
    teacher_scene=teacher_scene.resolve(); output_dir=output_dir.resolve()
    scene=json.loads(teacher_scene.read_text())
    if scene.get("schema")!="RealSaS.TeacherSceneIR.v2" or scene.get("subject_key")!=subject_id:
        raise ValueError("TRUTH139_RESPONSE_V2_SCENE_BINDING")
    gpath=bound_ref(geometry_ref(scene),"TRUTH139_RESPONSE_V2_STAGE47_GEOMETRY_DRIFT")
    with np.load(gpath,allow_pickle=False) as z:
        required={"rest_vertices_source_world","faces","face_owner_privileged","isolated_position"}
        if not required.issubset(z.files): raise ValueError("TRUTH139_RESPONSE_V2_ARRAYS_MISSING")
        rest=np.asarray(z["rest_vertices_source_world"],np.float64)
        faces=np.asarray(z["faces"],np.int64)
        owner=np.asarray(z["face_owner_privileged"],np.int32)
        isolated=np.asarray(z["isolated_position"],np.float64)
    actions=list((scene.get("calibration") or {}).get("actions") or [])
    if len(actions)!=len(isolated) or not actions: raise ValueError("TRUTH139_RESPONSE_V2_ACTION_BINDING")
    arrays=build_surface_response_arrays_v2(rest,faces,owner,isolated,actions)
    output_dir.mkdir(parents=True,exist_ok=True)
    ap=output_dir/"IRIS_RESPONSE_EVIDENCE_V2.npz"; save_npz(ap,arrays)
    face_valid=np.asarray(arrays["surface_face_valid"],bool)
    pair_valid=np.asarray(arrays["response_pair_valid"],bool)
    payload={
        "schema":SCHEMA,"status":STATUS,"subject_id":subject_id,
        "source":{"teacher_scene":{"path":str(teacher_scene),"sha256":sha256(teacher_scene)},"geometry_and_response":{"path":str(gpath),"sha256":sha256(gpath)}},
        "output":{"path":str(ap),"sha256":sha256(ap)},
        "measurement_policy":{"action_measure":"CONTROL_BALANCED__EQUAL_MASS_PER_CONTROL_THEN_EQUAL_MASS_WITHIN_CONTROL","pair_adjacency":"OWNER_PLUS_EXACT_SOURCE_POSITION_GEOMETRIC_TOPOLOGY","render_vertex_identity_used_as_adjacency":False,"stage47_recomputed":False,"semantic_part_labels_used":False},
        "counts":{"face_count":int(len(faces)),"valid_face_count":int(face_valid.sum()),"action_count":int(len(actions)),"response_pair_count":int(len(arrays["response_pair_face"])),"valid_response_pair_count":int(pair_valid.sum())},
        "training_performed":False,"qualification_minted":False,
    }
    payload["response_evidence_identity_sha256"]=digest(payload)
    mp=output_dir/"IRIS_RESPONSE_EVIDENCE_MANIFEST_V2.json"
    mp.write_text(json.dumps(payload,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":STATUS,"subject_id":subject_id,"response_pair_count":payload["counts"]["response_pair_count"],"identity":payload["response_evidence_identity_sha256"]},sort_keys=True))
    return payload


def main():
    p=argparse.ArgumentParser(); p.add_argument("--subject-id",required=True); p.add_argument("--teacher-scene",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True)
    a=p.parse_args(); build(subject_id=a.subject_id,teacher_scene=a.teacher_scene,output_dir=a.output_dir)


if __name__=="__main__": main()
