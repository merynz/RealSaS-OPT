#!/usr/bin/env python3
"""Execute bounded source-locked Corpus370 authored temporal teachers via Blender.

This is a REAL mesh/pose data producer, not a readiness report. Authored clips
are observations of motion, NOT independently identified causal interventions
or constitutive stiffness. Every teacher remains privileged and cannot enter
ordinary IRIS student-visible inputs. Results are subject-courted/candidate
only; do not grant TESSA, IRIS response or training authority here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np

from tools.corpus.audit_corpus370_v5_source_rest_courts_v1 import row_input

SCHEMA="RealSaS.Corpus370V5AuthoredTemporalTeacherPilot.v1"
V5_SHA="8ab2ebf526bbf91d78117580b68ed3784e53bcbafdbd326f012e143c924c0ac4"
ROSTER_SHA="925dac90ecbf949375aa3ef5d2ed2f40788ae7dcf5669a02b708ada8d9435355"
TEACHER_SCHEMA="RealSaS.IRISVNextAuthoredTeacherExport.v1"
PASS="PASS_SOURCE_AUTHORED_TEMPORAL_GEOMETRY_CANDIDATE__CAUSAL_RESPONSE_SEPARATE"


def require(ok,code):
    if not ok:raise ValueError(code)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def seal(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+".partial")
    temp.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n")
    temp.replace(path)


def choose_subjects(v5,roster,max_subjects):
    """Fixed lexicographic eligibility; never select based on measured outcome."""
    by_id={row["subject_id"]:row for row in roster["subjects"]}
    rows=[]
    for row in sorted(v5["subjects"],key=lambda x:x["subject_id"]):
        sid=row["subject_id"];source=by_id[sid]
        require(row["source_sha256"]==source["source_sha256"],
                "TEMPORAL_PILOT_SOURCE_ROSTER_DRIFT:"+sid)
        bindings=row_input(row)
        admission=json.loads(Path(bindings["admission"]["path"]).read_text())
        require(admission["subject_id"]==sid and admission["source_sha256"]==source["source_sha256"],
                "TEMPORAL_PILOT_ADMISSION_DRIFT:"+sid)
        if not (admission.get("native_motion_inventory") or []):
            continue
        rows.append((row,source,bindings,admission))
    require(len(rows)>=max_subjects,"TEMPORAL_PILOT_INSUFFICIENT_AUTHORED_ACTIONS")
    chosen=[];used=set()
    for row,source,bindings,admission in rows:
        kind=source.get("source_id") or source.get("origin") or source.get("source_format")
        if kind in used:continue
        chosen.append((row,source,bindings,admission))
        used.add(kind)
        if len(chosen)==max_subjects:break
    if len(chosen)<max_subjects:
        for item in rows:
            if item not in chosen:chosen.append(item)
            if len(chosen)==max_subjects:break
    require(len(chosen)==max_subjects,"TEMPORAL_PILOT_SELECTED_SUBJECT_COUNT")
    return chosen


def court_teacher(row,source,bindings,output,npz_path):
    sid=row["subject_id"]
    receipt_path=npz_path.with_suffix(npz_path.suffix+".json")
    require(npz_path.is_file() and receipt_path.is_file(),"TEMPORAL_PILOT_EXPORT_MISSING")
    receipt=json.loads(receipt_path.read_text())
    require(receipt.get("schema")==TEACHER_SCHEMA
            and receipt.get("transport_revision")==3
            and receipt.get("subject_id")==sid
            and receipt.get("source",{}).get("sha256")==source["source_sha256"]
            and receipt.get("observation_admission_sha256")==bindings["admission"]["sha256"]
            and receipt.get("teacher_only") is True
            and receipt.get("canonical_M_G_W_minted") is False
            and receipt.get("teacher_rig_skin_motion_allowed_as_inference_input") is False
            and receipt.get("output",{}).get("sha256")==sha(npz_path),
            "TEMPORAL_PILOT_TEACHER_SOURCE_BINDING")
    with np.load(npz_path,allow_pickle=False) as data:
        require(set(data.files)=={"rest_vertices_source_world","faces","face_owner",
                                  "face_material","posed_vertices_source_world"},
                "TEMPORAL_PILOT_UNEXPECTED_TEACHER_ARRAYS")
        rest=data["rest_vertices_source_world"];posed=data["posed_vertices_source_world"]
        faces=data["faces"];owners=data["face_owner"];material=data["face_material"]
        require(rest.ndim==2 and rest.shape[1]==3 and
                posed.shape==(receipt["sampled_frame_count"],len(rest),3)
                and faces.ndim==2 and faces.shape[1]==3 and
                owners.shape==(len(faces),) and material.shape==(len(faces),)
                and np.isfinite(rest).all() and np.isfinite(posed).all()
                and faces.min()>=0 and faces.max()<len(rest),
                "TEMPORAL_PILOT_GEOMETRY_ARRAYS")
        movement=np.linalg.norm(posed.astype(np.float64)-rest.astype(np.float64)[None],axis=2)
        max_displacement=float(movement.max())
        min_displacement=float(np.median(movement.max(axis=1)))
        return {
            "schema":"RealSaS.Corpus370V5AuthoredTemporalTeacherCourt.v1",
            "status":PASS if max_displacement>1e-8 else "MASKED_STATIC_AUTHORED_CLIP",
            "subject_id":sid,"source_sha256":source["source_sha256"],
            "teacher_manifest":{"path":str(receipt_path),"sha256":sha(receipt_path)},
            "teacher_arrays":{"path":str(npz_path),"sha256":sha(npz_path)},
            "counts":{"vertex_count":len(rest),"face_count":len(faces),
                      "authored_clip_count":len(receipt["native_motion_inventory_exported"]),
                      "sampled_frame_count":len(posed)},
            "measured_max_vertex_displacement_source_units":max_displacement,
            "measured_median_frame_max_displacement_source_units":min_displacement,
            "source_to_v5_import_equivalence_court":"PENDING_SEPARATE",
            "causal_response_interventions_executed":False,
            "student_teacher_id_leakage_allowed":False,
            "family_authority_minted":False,"training_authorized":False,
        }


def run(v5_path,roster_path,root,blender,output,max_subjects=3,max_frames=4):
    v5_path=Path(v5_path).resolve();roster_path=Path(roster_path).resolve()
    root=Path(root).resolve();blender=Path(blender).resolve();output=Path(output).resolve()
    require(sha(v5_path)==V5_SHA and sha(roster_path)==ROSTER_SHA,
            "TEMPORAL_PILOT_EXACT_SOURCE_COHORT_REQUIRED")
    require(blender.is_file() and root.is_dir(),"TEMPORAL_PILOT_RUNTIME_SOURCE_MISSING")
    require(1<=max_subjects<=8 and 2<=max_frames<=16,"TEMPORAL_PILOT_BOUNDS")
    v5=json.loads(v5_path.read_text());roster=json.loads(roster_path.read_text())
    require((v5.get("prepared_subject_count"),v5.get("failure_count"))==(352,18),
            "TEMPORAL_PILOT_V5_SCOPE")
    selected=choose_subjects(v5,roster,max_subjects)
    output.mkdir(parents=True,exist_ok=True)
    source_code=Path(__file__).resolve()
    state={
        "schema":SCHEMA,"status":"PILOT_EXECUTION_INCOMPLETE",
        "input_v5_sha256":V5_SHA,"roster_sha256":ROSTER_SHA,
        "orchestrator_sha256":sha(source_code),
        "teacher_exporter":"tools/iris_vnext/blender_authored_teacher_export_v3.py",
        "teacher_source_revision":os.environ.get("GITHUB_SHA"),
        "selected_subject_ids":[row["subject_id"] for row,_,_,_ in selected],
        "results":[],"max_frames_per_clip":max_frames,
        "no_model_training":True,"training_authorized":False,
    }
    checkpoint=output/"CORPUS370_V5_AUTHORED_TEMPORAL_PILOT_V1.json"
    for index,(row,source,bindings,admission) in enumerate(selected,1):
        sid=row["subject_id"];folder=output/"subjects"/sid;folder.mkdir(parents=True,exist_ok=True)
        source_path=(root/source["source_relative_path"]).resolve()
        require(source_path.is_relative_to(root) and sha(source_path)==source["source_sha256"],
                "TEMPORAL_PILOT_SOURCE_HASH_DRIFT:"+sid)
        teacher=folder/"teacher.npz";log_path=folder/"BLENDER_LOG.txt"
        command=[
            str(blender),"-b","--factory-startup","--python-exit-code","3",
            "--python","tools/iris_vnext/blender_authored_teacher_export_v3.py","--",
            "--source",str(source_path),"--expected-source-sha256",source["source_sha256"],
            "--subject-id",sid,"--observation-admission",bindings["admission"]["path"],
            "--output",str(teacher),"--max-frames-per-clip",str(max_frames),
        ]
        started=time.monotonic()
        try:
            completed=subprocess.run(command,text=True,stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT,timeout=420,check=False)
            log_path.write_text(completed.stdout[-250000:])
            require(completed.returncode==0,"TEMPORAL_PILOT_BLENDER_EXIT:"+str(completed.returncode))
            result=court_teacher(row,source,bindings,folder,teacher)
        except (KeyboardInterrupt,MemoryError):raise
        except Exception as exc:
            result={"schema":"RealSaS.Corpus370V5AuthoredTemporalTeacherCourt.v1",
                    "status":"FAIL_SOURCE_AUTHORED_TEMPORAL_EXPORT",
                    "subject_id":sid,"source_sha256":source["source_sha256"],
                    "error_type":type(exc).__name__,"reason":str(exc),
                    "traceback":traceback.format_exc(),
                    "family_authority_minted":False,"training_authorized":False}
        result["elapsed_seconds"]=round(time.monotonic()-started,3)
        seal(folder/"TEMPORAL_TEACHER_COURT_V1.json",result)
        state["results"].append(result)
        seal(checkpoint,state)
        print(json.dumps({"event":"CORPUS370_V5_TEMPORAL_PILOT_SUBJECT",
                          "done":index,"total":len(selected),
                          "subject_id":sid,"status":result["status"],
                          "seconds":result["elapsed_seconds"]},sort_keys=True),flush=True)
    counts=Counter(item["status"] for item in state["results"])
    state["status"]="PILOT_ALL_SELECTED_ACCOUNTED__CAUSAL_AND_MODEL_ADMISSION_SEPARATE"
    state["rollup"]=dict(sorted(counts.items()))
    seal(checkpoint,state)
    print(json.dumps({"event":"CORPUS370_V5_TEMPORAL_PILOT_SUMMARY",
                      "rollup":state["rollup"],"training_authorized":False},
                     sort_keys=True),flush=True)
    return state


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--v5-index",type=Path,required=True)
    p.add_argument("--exact-source-roster",type=Path,required=True)
    p.add_argument("--extracted-source-root",type=Path,required=True)
    p.add_argument("--blender",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--max-subjects",type=int,default=3)
    p.add_argument("--max-frames",type=int,default=4)
    a=p.parse_args()
    run(a.v5_index,a.exact_source_roster,a.extracted_source_root,a.blender,
        a.output,a.max_subjects,a.max_frames)

if __name__=="__main__":main()
