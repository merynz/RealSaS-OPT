#!/usr/bin/env python3
"""Hosted source-locked Corpus370 V8 authored-pose group diagnostic.

One original subject per GitHub-hosted worker. Hydrated asset bytes are
private; source-face labels and group-level measurements stay on private Drive.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback
import types

from tools.corpus.run_corpus370_native_pose_partition_private_v8 import (
    run as court_run,
)
from tools.corpus.corpus370_partition_edit_plan_v7 import build_plan

SCHEMA="RealSaS.Corpus370NativePosePartitionPrivateBatch.v8"
V5_SHA="8ab2ebf526bbf91d78117580b68ed3784e53bcbafdbd326f012e143c924c0ac4"
ROSTER_SHA="925dac90ecbf949375aa3ef5d2ed2f40788ae7dcf5669a02b708ada8d9435355"
RAYS_SHA="0eeb24089322d0802567ba9736ada1de03e234bbf133bc91164132a80bec3c70"
JOIN_SHA="e72f3380c2f34cbf3e088218ebfcd4be0f4aa71c9138d4b3d1865a4f4074439a"
GROUP_SHA="1d8b8e2e8018d2cd3b94188f2683424659a8b0735f43c43eb381b6b252a3b8a1"
NATIVE_SHA="5aa3c4d9cc97ab669d2a7a951a4c9d18218ef5ea4d6f58f5b1ff1ac5b89cc272"


def require(ok,code):
    if not ok:raise ValueError(code)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def read(path,digest,code):
    path=Path(path)
    require(path.is_file() and sha(path)==digest,code)
    return json.loads(path.read_text())


def seal(path,doc):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+".partial")
    temp.write_text(json.dumps(doc,sort_keys=True,indent=2,allow_nan=False)+"\n")
    temp.replace(path)


def run(args):
    start=time.monotonic()
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    summary={"schema":SCHEMA,"status":"FAIL_V8_SOURCE_POSE_COURT_NOT_COMPLETED",
             "subject_ordinal":args.subject_ordinal,
             "graphics_owner_authorized":False,
             "training_authorized":False}
    try:
        v5=read(args.v5_index,V5_SHA,"V8_V5_PARENT")
        roster=read(args.exact_roster,ROSTER_SHA,"V8_ROSTER_PARENT")
        rays=read(args.geometric_ray_index,RAYS_SHA,"V8_RAYS_PARENT")
        join=read(Path(args.original_join_root)/
                  "CORPUS370_NATIVE_VISIBILITY_PAIR_JOIN_V1.json",
                  JOIN_SHA,"V8_JOIN_PARENT")
        groups=read(Path(args.allview_groups_root)/
                    "CORPUS370_ALLVIEW_GEOMETRY_GROUPS_PRIVATE_BATCH_V3.json",
                    GROUP_SHA,"V8_GROUP_PARENT")
        native=read(Path(args.native_teacher_root)/
                    "CORPUS370_V5_AUTHORED_TEMPORAL_PILOT_V1.json",
                    NATIVE_SHA,"V8_NATIVE_PILOT_PARENT")
        ids=sorted(join["subjects"])
        require(len(ids)==3 and type(args.subject_ordinal) is int
                and 0<=args.subject_ordinal<3
                and v5.get("prepared_subject_count")==352
                and v5.get("failure_count")==18
                and native.get("training_authorized") is False
                and groups.get("model_training_authorized") is False
                and set(ids)==set(groups["subjects"])
                and set(ids)==set(native["selected_subject_ids"]),
                "V8_SOURCE_EXACT_COHORT_AND_AUTHORITY")
        sid=ids[args.subject_ordinal]
        row={x["subject_id"]:x for x in roster["subjects"]}[sid]
        v5row={x["subject_id"]:x for x in v5["subjects"]}[sid]
        native_row={x["subject_id"]:x for x in native["results"]}[sid]
        source_sha=row["source_sha256"]
        require(source_sha==v5row["source_sha256"]==
                groups["subjects"][sid]["source_sha256"]==
                join["subjects"][sid]["source_sha256"]==
                native_row["source_sha256"],
                "V8_ORIGINAL_SOURCE_SHA_LINEAGE")
        group_dir=Path(args.allview_groups_root)/"subjects"/sid
        proposal=group_dir/"ALLVIEW_ANONYMOUS_GROUP_CANDIDATES_V3.json"
        proposal_sha=groups["subjects"][sid]["proposal_sha256"]
        layer=read(proposal,proposal_sha,"V8_SOURCE_GROUP_PROPOSAL")
        pairref=join["subjects"][sid]["pair_diagnostic_receipt"]
        pairs=read(pairref["path"],pairref["sha256"],"V8_NATIVE_SOURCE_PAIRS")
        plan=build_plan(layer,pairs)
        require(plan["source_sha256"]==source_sha,
                "V8_NATIVE_GROUP_SOURCE_DOMAIN")
        rayref=rays["subject_results"][sid]["receipt"]
        ray=read(rayref["path"],rayref["sha256"],"V8_SOURCE_RAY")
        restref=ray["source_rest_geometry"]
        require(restref["sha256"]==pairs["parents"]["rest_geometry_sha256"],
                "V8_SOURCE_REST_PAIR_LINK")
        native_dir=Path(args.native_teacher_root)/"subjects"/sid
        native_court_path=native_dir/"TEMPORAL_TEACHER_COURT_V1.json"
        court_sha=sha(native_court_path)
        court=read(native_court_path,court_sha,"V8_EXISTING_TEMPORAL_COURT")
        require(court["teacher_arrays"]["sha256"]==
                native_row["teacher_arrays"]["sha256"]
                and court["teacher_manifest"]["sha256"]==
                    native_row["teacher_manifest"]["sha256"],
                "V8_NATIVE_PILOT_INDEPENDENT_ARRAY_AND_RECEIPT_BINDING")
        local_args=types.SimpleNamespace(
            subject_id=sid,
            proposal=proposal,proposal_sha256=proposal_sha,
            source_pairs=pairref["path"],
            source_pairs_sha256=pairref["sha256"],
            original_rest=restref["path"],
            original_rest_sha256=restref["sha256"],
            authored_teacher=court["teacher_arrays"]["path"],
            authored_teacher_sha256=court["teacher_arrays"]["sha256"],
            teacher_receipt=court["teacher_manifest"]["path"],
            teacher_receipt_sha256=court["teacher_manifest"]["sha256"],
            teacher_court=native_court_path,teacher_court_sha256=court_sha,
            output=output/"PRIVATE_NATIVE_POSE_PARTITION_COURT_V8.json",
        )
        decision=court_run(local_args)
        require(decision["source_authored_rest_exact_face_order"] is True
                and decision["source_face_identity_in_pose_domain_verified"] is True
                and decision["same_drawable_positive_pairs"]==0
                and decision["different_drawable_negative_pairs"]==0
                and decision["camera_visibility_animated_courted"] is False
                and decision["training_authorized"] is False,
                "V8_NO_SHORTCUT_GRAPHICS_OR_MECHANICAL_AUTHORITY")
        summary.update({
            "status":"PASS_REAL_NATIVE_POSE_SOURCE_FACE_GROUP_DIAGNOSTIC__NO_OWNER_AUTHORITY",
            "subject_id":sid,
            "source_sha256":source_sha,
            "source_native_teacher_sha256":court["teacher_arrays"]["sha256"],
            "independent_pose_group_court_sha256":sha(local_args.output),
            "real_authored_pose_count":decision["authored_pose_count"],
            "real_authored_action_count":decision["authored_action_count"],
            "source_face_count":decision["source_rest_face_count"],
            "group_sizes":decision["available_candidate_sizes"],
            "unavailable_sizes":decision["unavailable_requested_sizes"],
            "whole_source_collapsed_face_pose_instances":
                decision["whole_source_collapsed_face_pose_instances"],
            "whole_source_max_vertex_motion_over_extent":
                decision["whole_source_max_vertex_motion_over_extent"],
            "source_face_identity_in_pose_domain_verified":True,
            "animated_render_caa_authorized":False,
            "graphics_owner_authorized":False,
            "training_authorized":False,
        })
    except (Exception,KeyboardInterrupt) as exc:
        summary["failure_type"]=type(exc).__name__
        summary["failure_reason"]=str(exc)
        seal(output/"PRIVATE_NATIVE_POSE_FAILURE_V8.json",{
            "error":type(exc).__name__,"reason":str(exc),
            "traceback":traceback.format_exc(),
            "training_authorized":False,
        })
    summary["elapsed_seconds"]=round(time.monotonic()-start,3)
    seal(output/"PRIVATE_NATIVE_POSE_BATCH_RESULT_V8.json",summary)
    print(json.dumps({k:v for k,v in summary.items()
                      if k not in ("source_sha256","failure_reason")},
                     sort_keys=True),flush=True)
    require(summary["status"].startswith("PASS_"),
            "V8_SOURCE_NATIVE_POSE_GROUP_EVIDENCE_REPAIR_REQUIRED")
    return summary


def main():
    p=argparse.ArgumentParser()
    for flag in ("v5_index","exact_roster","geometric_ray_index",
                 "original_join_root","allview_groups_root",
                 "native_teacher_root","output"):
        p.add_argument("--"+flag.replace("_","-"),type=Path,required=True)
    p.add_argument("--subject-ordinal",type=int,required=True)
    run(p.parse_args())


if __name__=="__main__":main()
