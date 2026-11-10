#!/usr/bin/env python3
"""Full V5 authored-motion teacher capture: real Blender v4, SHA-locked, resumable.

Read every V5 observation PASS, classify missing authored animations explicitly,
and run exact source Action->owner (armature/node/shape-key) mesh replay for the
complete native-motion cohort. Failure receipts are retained, never silently
dropped. This is true authored temporal observation, NOT causal intervention,
physical stiffness, TESSA teacher admission, IRIS student labels, or training.
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

from tools.corpus.audit_corpus370_v5_source_rest_courts_v1 import row_input
from tools.corpus.run_corpus370_v5_native_teacher_temporal_pilot_v1 import court_teacher

SCHEMA="RealSaS.Corpus370V5AllAuthoredTemporalTeacherSweep.v1"
V5_SHA="8ab2ebf526bbf91d78117580b68ed3784e53bcbafdbd326f012e143c924c0ac4"
ROSTER_SHA="925dac90ecbf949375aa3ef5d2ed2f40788ae7dcf5669a02b708ada8d9435355"


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


def roster(v5,roster):
    """Return all source-authorised native-motion rows and typed no-motion rows."""
    require(v5.get("schema")=="RealSaS.Corpus370PreparedObservationPackets.v5"
            and (v5.get("subject_count"),v5.get("prepared_subject_count"),v5.get("failure_count"))==(370,352,18)
            and v5.get("training_authorized") is False,
            "ALL_NATIVE_V5_SOURCE_SCOPE")
    require(roster.get("schema")=="RealSaS.Corpus370ExactSourceRoster.v1"
            and roster.get("subject_count")==370 and roster.get("training_authorized") is False,
            "ALL_NATIVE_ROSTER_SOURCE_SCOPE")
    by_source={r["subject_id"]:r for r in roster["subjects"]}
    prepared={r["subject_id"]:r for r in v5["subjects"]}
    failed={r["subject_id"]:r for r in v5["failures"]}
    require(len(by_source)==370 and len(prepared)==352 and len(failed)==18
            and not set(prepared).intersection(failed)
            and set(prepared)|set(failed)==set(by_source),
            "ALL_NATIVE_COMPLETE_SOURCE_PARTITION")
    selected=[];no_native=[]
    for sid,row in sorted(prepared.items()):
        source=by_source[sid]
        require(source["source_sha256"]==row["source_sha256"],
                "ALL_NATIVE_SOURCE_IDENTITY:"+sid)
        refs=row_input(row)
        admission=json.loads(Path(refs["admission"]["path"]).read_text())
        require(admission.get("subject_id")==sid
                and admission.get("source_sha256")==source["source_sha256"]
                and admission.get("status")=="PASS_IRIS_BIND_OBSERVATIONS",
                "ALL_NATIVE_SOURCE_ADMISSION:"+sid)
        actions=list(admission.get("native_motion_inventory") or [])
        if actions:
            require(len(actions)==len(set(actions)),"ALL_NATIVE_DUPLICATE_ACTION:"+sid)
            selected.append({"subject_id":sid,"observation":row,"source":source,
                             "admission":refs["admission"],
                             "authored_action_count":len(actions)})
        else:
            no_native.append({"subject_id":sid,"source_sha256":row["source_sha256"],
                              "status":"MASKED_NO_AUTHORED_NATIVE_ANIMATION",
                              "temporal_target_valid":False})
    return selected,no_native,sorted(failed)


def source_identity(blender):
    repo=Path(__file__).resolve().parents[2]
    paths=[
        Path(__file__).resolve(),
        repo/"tools/corpus/run_corpus370_v5_native_teacher_temporal_pilot_v1.py",
        repo/"tools/iris_vnext/blender_authored_teacher_export_v4.py",
        repo/"tools/iris_vnext/blender_authored_teacher_export_v4_actions.py",
        repo/"tools/iris_vnext/blender_authored_teacher_export_v3.py",
    ]
    return {
        "v5_index_sha256":V5_SHA,"exact_source_roster_sha256":ROSTER_SHA,
        "source_code_sha256":{str(p.relative_to(repo)):sha(p) for p in paths},
        "blender_binary_sha256":sha(blender),
        "max_frames_per_clip":16,
    }


def _stable_receipt(row):
    return {key:row.get(key) for key in (
        "status","subject_id","source_sha256",
        "teacher_manifest","teacher_arrays","counts",
        "measured_max_vertex_displacement_source_units",
        "measured_median_frame_max_displacement_source_units",
        "causal_response_interventions_executed",
        "student_teacher_id_leakage_allowed",
        "family_authority_minted","training_authorized"
    )}


def measure(row,root,blender,output,max_frames=16,*,resume=None):
    sid=row["subject_id"]
    source_path=(root/row["source"]["source_relative_path"]).resolve()
    require(source_path.is_relative_to(root) and source_path.is_file()
            and sha(source_path)==row["source"]["source_sha256"],
            "ALL_NATIVE_SOURCE_BYTES:"+sid)
    admission=Path(row["admission"]["path"]).resolve()
    require(admission.is_file() and sha(admission)==row["admission"]["sha256"],
            "ALL_NATIVE_ADMISSION_BYTES:"+sid)
    folder=output/"subjects"/sid
    folder.mkdir(parents=True,exist_ok=True)
    teacher=folder/"teacher.npz"
    court_path=folder/"TEMPORAL_TEACHER_COURT_V1.json"
    if resume is not None and resume.get("status") in (
        "PASS_SOURCE_AUTHORED_TEMPORAL_GEOMETRY_CANDIDATE__CAUSAL_RESPONSE_SEPARATE",
        "MASKED_STATIC_AUTHORED_CLIP",
    ):
        recourt=court_teacher(
            {"subject_id":sid},row["source"],{"admission":row["admission"]},folder,teacher
        )
        require(_stable_receipt(resume)==_stable_receipt(recourt),
                "ALL_NATIVE_RESUME_COURT_DRIFT:"+sid)
        require(court_path.is_file() and sha(court_path)==resume["court_sha256"],
                "ALL_NATIVE_RESUME_RECEIPT_DRIFT:"+sid)
        return resume
    command=[
        str(blender),"-b","--factory-startup","--python-exit-code","3",
        "--python","tools/iris_vnext/blender_authored_teacher_export_v4.py","--",
        "--source",str(source_path),
        "--expected-source-sha256",row["source"]["source_sha256"],
        "--subject-id",sid,"--observation-admission",str(admission),
        "--output",str(teacher),"--max-frames-per-clip",str(max_frames),
    ]
    started=time.monotonic()
    try:
        result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                              text=True,timeout=420,check=False)
        (folder/"BLENDER_LOG.txt").write_text(result.stdout[-250000:])
        require(result.returncode==0,"ALL_NATIVE_BLENDER_EXIT:"+str(result.returncode))
        actual=court_teacher({"subject_id":sid},row["source"],
                             {"admission":row["admission"]},folder,teacher)
    except (KeyboardInterrupt,MemoryError):
        raise
    except Exception as exc:
        actual={
            "schema":"RealSaS.Corpus370V5AuthoredTemporalTeacherCourt.v1",
            "status":"FAIL_SOURCE_AUTHORED_TEMPORAL_EXPORT",
            "subject_id":sid,"source_sha256":row["source"]["source_sha256"],
            "reason":str(exc),"error_type":type(exc).__name__,
            "traceback":traceback.format_exc(),
            "family_authority_minted":False,"training_authorized":False,
        }
    actual["elapsed_seconds"]=round(time.monotonic()-started,3)
    seal(court_path,actual)
    actual["court_sha256"]=sha(court_path)
    return actual


def sweep(v5_path,roster_path,root,blender,output,*,max_frames=16):
    v5_path=Path(v5_path).resolve();roster_path=Path(roster_path).resolve()
    root=Path(root).resolve();blender=Path(blender).resolve();output=Path(output).resolve()
    require(sha(v5_path)==V5_SHA and sha(roster_path)==ROSTER_SHA,
            "ALL_NATIVE_PINNED_COHORT")
    require(blender.is_file() and root.is_dir() and max_frames==16,
            "ALL_NATIVE_RUNTIME_OR_SAMPLE_DRIFT")
    v5=json.loads(v5_path.read_text());sources=json.loads(roster_path.read_text())
    subjects,no_native,excluded=roster(v5,sources)
    identity=source_identity(blender)
    identity["selected_native_subject_ids"]=[x["subject_id"] for x in subjects]
    identity["masked_no_native_subject_ids"]=[x["subject_id"] for x in no_native]
    identity["excluded_observation_ids"]=[x for x in excluded]
    checkpoint=output/"CORPUS370_V5_ALL_AUTHORED_TEMPORAL_TEACHER_V1.json"
    output.mkdir(parents=True,exist_ok=True)
    if checkpoint.is_file():
        state=json.loads(checkpoint.read_text())
        require(state.get("identity")==identity,"ALL_NATIVE_RESUME_IDENTITY_DRIFT")
    else:
        state={
            "schema":SCHEMA,"status":"PARTIAL_NATIVE_TEACHER_SWEEP__NO_TRAINING",
            "identity":identity,"candidate_universe_count":370,
            "observation_pass_count":352,"observation_fail_count":18,
            "native_candidate_count":len(subjects),
            "no_authored_native_animation_subjects":no_native,
            "results":{},"training_authorized":False,
            "causal_response_authority_minted":False,
            "tessa_teacher_authority_minted":False,
            "student_teacher_id_leakage_allowed":False,
        }
    for i,row in enumerate(subjects,1):
        sid=row["subject_id"]
        prior=state["results"].get(sid)
        actual=measure(row,root,blender,output,max_frames=max_frames,resume=prior)
        state["results"][sid]=actual
        seal(checkpoint,state)
        print(json.dumps({"event":"CORPUS370_ALL_NATIVE_TEACHER_PROGRESS",
                         "done":i,"total":len(subjects),"subject_id":sid,
                         "status":actual["status"],
                         "seconds":actual.get("elapsed_seconds")},
                        sort_keys=True),flush=True)
    require(len(state["results"])==len(subjects)
            and 18+len(no_native)+len(subjects)==370,
            "ALL_NATIVE_COHORT_ACCOUNTING_DRIFT")
    rollup=Counter(r["status"] for r in state["results"].values())
    state["status"]="ALL_NATIVE_ACTION_SUBJECTS_ACCOUNTED__CAUSAL_RESPONSE_AND_TRAINING_SEPARATE"
    state["rollup"]=dict(sorted(rollup.items()))
    state["failure_count"]=sum(r["status"].startswith("FAIL_") for r in state["results"].values())
    state["masked_no_native_subject_count"]=len(no_native)
    seal(checkpoint,state)
    print(json.dumps({
        "event":"CORPUS370_ALL_NATIVE_TEACHER_SUMMARY",
        "observation_pass_count":352,"native_candidate_count":len(subjects),
        "no_native_count":len(no_native),"observation_fail_count":18,
        "rollup":state["rollup"],"failure_count":state["failure_count"],
        "training_authorized":False},sort_keys=True),flush=True)
    return state


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--v5-index",type=Path,required=True)
    p.add_argument("--exact-source-roster",type=Path,required=True)
    p.add_argument("--extracted-source-root",type=Path,required=True)
    p.add_argument("--blender",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--max-frames",type=int,default=16)
    a=p.parse_args()
    sweep(a.v5_index,a.exact_source_roster,a.extracted_source_root,
          a.blender,a.output,max_frames=a.max_frames)


if __name__=="__main__":main()
