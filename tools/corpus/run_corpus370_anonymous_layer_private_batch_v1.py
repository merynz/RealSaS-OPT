#!/usr/bin/env python3
"""Real 3-source generic layer candidates from SHA-sealed native/Graphics join.

Consumes the PRIOR 41916724594-1 private results, not the original art.
Each source has canonical rest/posed triangles and independently court-replayed
source-hit pair addresses from 0/4 views. Does not inspect identity labels,
materials, author objects or split test view during candidate generation.
All per-face layers stay private in Drive. Prints aggregate counts only.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np

from tools.corpus.compile_corpus370_anonymous_layer_proposals_v1 import (
    compile_candidates,
)
from tools.corpus.court_corpus370_anonymous_layer_edit_proxy_v1 import court

SCHEMA="RealSaS.Corpus370AnonymousLayerPrivateBatch.v1"
PREVIOUS_JOIN_SHA="e72f3380c2f34cbf3e088218ebfcd4be0f4aa71c9138d4b3d1865a4f4074439a"
PARENT_SCHEMA="RealSaS.Corpus370HostedNativeVisibilityPairJoin.v1"
CANONICAL_SCHEMA="RealSaS.Corpus370V5SourceNativeCanonicalAnimationCandidate.v1"
PAIRS_SCHEMA="RealSaS.Corpus370GenericPairKinematicDiagnosticBatch.v1"


def require(ok,code):
    if not ok:
        raise ValueError(code)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def read_ref(ref,code):
    p=Path(ref["path"]).resolve()
    require(p.is_file() and sha(p)==ref["sha256"],code)
    return p,json.loads(p.read_text())


def write(path,payload):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".partial")
    tmp.write_text(json.dumps(payload,sort_keys=True,indent=2,
                              allow_nan=False)+"\n")
    tmp.replace(path)


def run(join_root,output_root):
    join_root=Path(join_root).resolve()
    output_root=Path(output_root).resolve()
    parent_path=join_root/"CORPUS370_NATIVE_VISIBILITY_PAIR_JOIN_V1.json"
    require(parent_path.is_file() and sha(parent_path)==PREVIOUS_JOIN_SHA,
            "ANON_BATCH_PRIVATE_REAL_JOIN_SHA")
    summary=json.loads(parent_path.read_text())
    require(summary.get("schema")==PARENT_SCHEMA
            and summary.get("status")==
                "ALL_THREE_NATIVE_VISIBILITY_PAIRS_ACCOUNTED__OWNER_AUTHORITY_SEPARATE"
            and summary.get("expected_subject_count")==3
            and summary.get("qualified_subject_count")==3
            and summary.get("failure_count")==0
            and summary.get("measured_kinematic_pair_count")==6
            and summary.get("graphic_owner_identity_minted") is False
            and summary.get("tessa_teacher_authorized") is False
            and summary.get("training_authorized") is False,
            "ANON_BATCH_NOT_SOURCE_BOUND")
    ids=sorted(summary.get("subjects") or {})
    require(len(ids)==3,"ANON_BATCH_SUBJECT_CENSUS")
    result={
        "schema":SCHEMA,
        "status":"PARTIAL_PRIVATE_ANON_LAYER_COMPILATION__NO_OWNER_AUTHORITY",
        "parent_join_sha256":PREVIOUS_JOIN_SHA,
        "expected_subject_count":3,
        "subjects":{},
        "training_authorized":False,
        "graphics_owner_truth_minted":False,
        "tessa_teacher_authorized":False,
    }
    report=output_root/"CORPUS370_ANON_LAYER_PRIVATE_BATCH_V1.json"
    for sid in ids:
        target=output_root/"subjects"/sid
        try:
            subject=summary["subjects"][sid]
            source_sha=subject["source_sha256"]
            require(subject["status"]==
                    "PASS_PRIVATE_SOURCE_REST_AUTHORED_AND_VISIBILITY_JOIN__OWNER_UNKNOWN"
                    and subject["graphics_owner_minted"] is False
                    and subject["training_authorized"] is False
                    and subject["kinematic_pair_count"]==2,
                    "ANON_BATCH_SOURCE_SUBJECT_AUTHORITY")
            canon_path,canonical=read_ref(subject["canonical_candidate_receipt"],
                                          "ANON_BATCH_CANONICAL_MANIFEST_SHA")
            pair_path,pairs=read_ref(subject["pair_diagnostic_receipt"],
                                     "ANON_BATCH_PAIR_MANIFEST_SHA")
            require(canonical["schema"]==CANONICAL_SCHEMA
                    and canonical["subject_id"]==sid
                    and canonical["source_sha256"]==source_sha
                    and canonical["authority"]["training_authorized"] is False
                    and canonical["authority"]["independent_native_action_replay_court_passed"] is False
                    and pairs["schema"]==PAIRS_SCHEMA
                    and pairs["parents"]["source_sha256"]==source_sha
                    and pairs["pair_count"]==2
                    and pairs["training_authorized"] is False
                    and pairs["qualified_same_drawable_positives"]==0
                    and pairs["qualified_different_drawable_negatives"]==0,
                    "ANON_BATCH_REAL_COURT_SCOPE")
            arrays=canonical["canonical_arrays"]
            npz=Path(arrays["path"]).resolve()
            require(npz.is_file() and sha(npz)==arrays["sha256"],
                    "ANON_BATCH_CANONICAL_ARRAY_SHA")
            with np.load(npz,allow_pickle=False) as data:
                require({"rest_vertices_canonical",
                         "posed_vertices_canonical","faces"}<=set(data.files),
                        "ANON_BATCH_CANONICAL_ARRAY_LAYOUT")
                proposals=compile_candidates(
                    data["rest_vertices_canonical"],data["faces"],
                    data["posed_vertices_canonical"],pairs["rows"],
                    parents={
                        "source_sha256":source_sha,
                        "canonical_arrays_sha256":sha(npz),
                        "source_pair_receipt_sha256":sha(pair_path),
                    },
                )
            proposal_path=target/"ANON_LAYER_CANDIDATES_V1.json"
            write(proposal_path,proposals)
            consumer=court(proposals,pairs)
            consumer["parents"]["proposal_sha256"]=sha(proposal_path)
            consumer_path=target/"ANON_LAYER_EDIT_PROXY_COURT_V1.json"
            write(consumer_path,consumer)
            result["subjects"][sid]={
                "status":"PASS_ANONYMOUS_LAYER_CANDIDATES_AND_BOUNDED_EDIT_PROXY",
                "source_sha256":source_sha,
                "candidate_count":proposals["candidate_count"],
                "source_face_count":proposals["face_count"],
                "authored_pose_count":proposals["authored_pose_count"],
                "train_view":0,"heldout_view":4,
                "train_only_selected_candidate":consumer["selected_candidate_train_only"],
                "selected_slot_count":consumer["selected_slot_count"],
                "selected_heldout_reveal_fraction":consumer[
                    "selected_heldout_reveal_fraction_diagnostic_only"],
                "proposal_sha256":sha(proposal_path),
                "consumer_sha256":sha(consumer_path),
                "graphics_owner_minted":False,
                "training_authorized":False,
            }
        except (MemoryError,KeyboardInterrupt):
            raise
        except Exception as ex:
            result["subjects"][sid]={
                "status":"FAIL_SOURCE_BOUND_OPERATIONAL_LAYER_CANDIDATES",
                "error_type":type(ex).__name__,"reason":str(ex),
                "graphics_owner_minted":False,
                "training_authorized":False,
            }
            write(target/"ANON_LAYER_FAILURE_V1.json",{
                "status":"FAIL_SOURCE_BOUND_OPERATIONAL_LAYER_CANDIDATES",
                "source_subject_id":sid,
                "error_type":type(ex).__name__,"reason":str(ex),
                "traceback":traceback.format_exc(),
                "training_authorized":False,
            })
        finally:
            write(report,result)
    tally=Counter(x["status"] for x in result["subjects"].values())
    result["status"]="ALL_THREE_ANON_LAYER_RESEARCH_CANDIDATES_ACCOUNTED__NO_OWNER_AUTHORITY"
    result["qualified_candidate_subjects"]=tally[
        "PASS_ANONYMOUS_LAYER_CANDIDATES_AND_BOUNDED_EDIT_PROXY"]
    result["failure_count"]=tally["FAIL_SOURCE_BOUND_OPERATIONAL_LAYER_CANDIDATES"]
    result["selected_heldout_full_reveal_count"]=sum(
        x.get("selected_heldout_reveal_fraction")==1
        for x in result["subjects"].values())
    result["source_face_count_total"]=sum(x.get("source_face_count",0)
                                          for x in result["subjects"].values())
    result["candidate_count_total"]=sum(x.get("candidate_count",0)
                                        for x in result["subjects"].values())
    write(report,result)
    print(json.dumps({
        "status":result["status"],
        "subject_count":len(result["subjects"]),
        "candidate_count_total":result["candidate_count_total"],
        "qualified_candidate_subjects":result["qualified_candidate_subjects"],
        "failure_count":result["failure_count"],
        "selected_heldout_full_reveal_count":result["selected_heldout_full_reveal_count"],
        "graphics_owner_minted":False,"training_authorized":False,
    },sort_keys=True),flush=True)
    return result


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("--private-native-visibility-join-root",type=Path,required=True)
    cli.add_argument("--output-root",type=Path,required=True)
    a=cli.parse_args()
    result=run(a.private_native_visibility_join_root,a.output_root)
    require(result["qualified_candidate_subjects"]==3
            and result["failure_count"]==0,
            "ANON_BATCH_PRIVATE_SOURCE_FAILURES_REQUIRE_REPAIR")


if __name__=="__main__":
    main()
