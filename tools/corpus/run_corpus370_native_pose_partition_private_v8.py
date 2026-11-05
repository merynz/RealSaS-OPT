#!/usr/bin/env python3
"""Run independently sealed V8 native-pose partition court on private teacher bytes.

Single real source subject per invocation. Private source geometry and group
addresses are never published; this script verifies every parent SHA.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from tools.corpus.court_corpus370_native_pose_partition_v8 import evaluate, require

SCHEMA="RealSaS.Corpus370NativePosePartitionPrivateRunner.v8"


def sha(p):
    with Path(p).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def checked(path,digest):
    p=Path(path)
    require(p.is_file() and sha(p)==digest,
            "V8_REJECT_NONEXACT_SHA_OR_UNMOUNTED_PRIVATE_SOURCE")
    return p


def run(args):
    paths={
        "proposal":(args.proposal,args.proposal_sha256),
        "source_pairs":(args.source_pairs,args.source_pairs_sha256),
        "original_rest":(args.original_rest,args.original_rest_sha256),
        "authored_teacher":(args.authored_teacher,args.authored_teacher_sha256),
        "teacher_receipt":(args.teacher_receipt,args.teacher_receipt_sha256),
        "teacher_court":(args.teacher_court,args.teacher_court_sha256),
    }
    checked_paths={k:checked(path,digest) for k,(path,digest) in paths.items()}
    proposal=json.loads(checked_paths["proposal"].read_text())
    pairs=json.loads(checked_paths["source_pairs"].read_text())
    receipt=json.loads(checked_paths["teacher_receipt"].read_text())
    court=json.loads(checked_paths["teacher_court"].read_text())
    require(court.get("teacher_manifest",{}).get("sha256")==args.teacher_receipt_sha256
            and receipt.get("subject_id")==court.get("subject_id")
            and args.subject_id==receipt.get("subject_id")
            and proposal.get("source_sha256")==court.get("source_sha256"),
            "V8_INDEPENDENT_COURT_SUBJECT_OR_PARENT_BINDING")
    with np.load(checked_paths["original_rest"],allow_pickle=False) as orig:
        require({"rest_vertices_source_world","faces"}<=set(orig.files)
                and set(orig.files)<={
                    "rest_vertices_source_world","faces","face_owner_privileged",
                    "face_material_privileged","triangle_uv","triangle_uv_valid"},
                "V8_ORIGINAL_REST_ARRAY_CONTRACT")
        original={k:np.array(orig[k]) for k in
                  ("rest_vertices_source_world","faces")}
    with np.load(checked_paths["authored_teacher"],allow_pickle=False) as native:
        require({"rest_vertices_source_world","faces","posed_vertices_source_world"}
                <=set(native.files), "V8_AUTHORED_SAMPLED_POSES_MISSING")
        authored={k:np.array(native[k]) for k in
                  ("rest_vertices_source_world","faces","posed_vertices_source_world")}
    result=evaluate(original,authored,proposal,pairs,receipt,court,
        rest_sha256=args.original_rest_sha256,
        teacher_sha256=args.authored_teacher_sha256,
        teacher_receipt_sha256=args.teacher_receipt_sha256,
        source_sha256=proposal["source_sha256"])
    result["subject_id"]=args.subject_id
    result["parents"]={
        "source_sha256":result["source_sha256"],
        "group_proposal_sha256":args.proposal_sha256,
        "original_source_rest_sha256":args.original_rest_sha256,
        "native_teacher_npz_sha256":args.authored_teacher_sha256,
        "native_teacher_manifest_sha256":args.teacher_receipt_sha256,
        "native_teacher_independent_court_sha256":args.teacher_court_sha256,
        "original_source_hit_pairs_sha256":args.source_pairs_sha256,
    }
    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({
        "status":result["status"],
        "subject":args.subject_id,
        "frames":result["authored_pose_count"],
        "actions":result["authored_action_count"],
        "candidates":result["available_candidate_sizes"],
        "training_authorized":False,
    },sort_keys=True),flush=True)
    return result


def main():
    q=argparse.ArgumentParser()
    for name in ("subject_id","proposal","proposal_sha256","source_pairs",
                 "source_pairs_sha256","original_rest","original_rest_sha256",
                 "authored_teacher","authored_teacher_sha256","teacher_receipt",
                 "teacher_receipt_sha256","teacher_court","teacher_court_sha256",
                 "output"):
        q.add_argument("--"+name.replace("_","-"),
                       required=True,type=str if name.endswith(("sha256","subject_id")) else Path)
    run(q.parse_args())


if __name__=="__main__":main()
