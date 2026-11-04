#!/usr/bin/env python3
"""Real 3-subject hierarchical graphics group candidates on existing sealed art.

No artist files need re-importing: identical source-bound canonical vertices,
authored poses, pre-courted anonymous source patches and two-view ray effects
are read from immutable prior private Drive receipts. Source faces/slots remain
TEACHER ONLY; public Actions summary contains aggregate counts only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np

from tools.corpus.compile_corpus370_hierarchical_layer_groups_v1 import build
from tools.corpus.court_corpus370_hierarchical_layer_groups_v1 import verify

SCHEMA="RealSaS.Corpus370HierarchicalLayerPrivateBatch.v1"
JOIN_SHA="e72f3380c2f34cbf3e088218ebfcd4be0f4aa71c9138d4b3d1865a4f4074439a"
PREVIOUS_ANONYMOUS_SHA="47b3f3beb7a3625504ccf8488d6fe0cd8fbe0014421c5e4c77748f829c9f01ca"


def require(ok,code):
    if not ok:raise ValueError(code)


def sha(path):
    with Path(path).open("rb") as s:
        return hashlib.file_digest(s,"sha256").hexdigest()


def read(path,expected,code):
    file=Path(path).resolve()
    require(file.is_file() and sha(file)==expected,code)
    return json.loads(file.read_text())


def save(path,obj):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    stage=path.with_suffix(path.suffix+".partial")
    stage.write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+"\n")
    stage.replace(path)


def run(join_root,anon_root,output):
    join_root=Path(join_root).resolve()
    anon_root=Path(anon_root).resolve()
    output=Path(output).resolve()
    earlier_join=read(join_root/"CORPUS370_NATIVE_VISIBILITY_PAIR_JOIN_V1.json",
                     JOIN_SHA,"HIER_BATCH_SOURCE_JOIN_SHA_DRIFT")
    earlier_anon=read(anon_root/"CORPUS370_ANON_LAYER_PRIVATE_BATCH_V1.json",
                     PREVIOUS_ANONYMOUS_SHA,"HIER_BATCH_PARENT_ANON_SHA_DRIFT")
    ids=set(earlier_anon.get("subjects") or {})
    require(earlier_join.get("schema")==
            "RealSaS.Corpus370HostedNativeVisibilityPairJoin.v1"
            and earlier_join.get("training_authorized") is False
            and earlier_join.get("graphic_owner_identity_minted") is False
            and earlier_join.get("qualified_subject_count")==3
            and earlier_anon.get("schema")=="RealSaS.Corpus370AnonymousLayerPrivateBatch.v1"
            and earlier_anon.get("qualified_candidate_subjects")==3
            and earlier_anon.get("failure_count")==0
            and earlier_anon.get("graphics_owner_truth_minted") is False
            and earlier_anon.get("training_authorized") is False
            and earlier_anon.get("parent_join_sha256")==JOIN_SHA
            and ids==set(earlier_join.get("subjects") or {})
            and len(ids)==3,
            "HIER_BATCH_SUBJECT_SCOPE")
    report={
        "schema":SCHEMA,
        "status":"PARTIAL_HIERARCHICAL_SOURCE_BOUND_GROUPING__NO_PRODUCT_AUTHORITY",
        "parents":{
            "prior_source_native_visibility_join_sha256":JOIN_SHA,
            "prior_anon_layer_private_batch_sha256":PREVIOUS_ANONYMOUS_SHA,
        },
        "expected_subject_count":3,
        "subjects":{},
        "source_semantics_in_student":False,
        "graphics_owner_authorized":False,
        "model_training_authorized":False,
        "product_compilation_authorized":False,
    }
    result_file=output/"CORPUS370_HIERARCHICAL_LAYER_PRIVATE_BATCH_V1.json"
    for sid in sorted(ids):
        target=output/"subjects"/sid
        old=earlier_anon["subjects"][sid]
        try:
            ground=earlier_join["subjects"][sid]
            require(old["status"]==
                    "PASS_ANONYMOUS_LAYER_CANDIDATES_AND_BOUNDED_EDIT_PROXY"
                    and ground["status"]==
                    "PASS_PRIVATE_SOURCE_REST_AUTHORED_AND_VISIBILITY_JOIN__OWNER_UNKNOWN"
                    and ground["source_sha256"]==old["source_sha256"]
                    and old["graphics_owner_minted"] is False
                    and old["training_authorized"] is False
                    and ground["graphics_owner_minted"] is False,
                    "HIER_BATCH_REAL_SOURCE_IDENTITY")
            old_subject_root=anon_root/"subjects"/sid
            candidate=read(old_subject_root/"ANON_LAYER_CANDIDATES_V1.json",
                           old["proposal_sha256"],"HIER_BATCH_PRIOR_ANON_CANDIDATES_SHA")
            base_consumer=read(old_subject_root/"ANON_LAYER_EDIT_PROXY_COURT_V1.json",
                               old["consumer_sha256"],"HIER_BATCH_PRIOR_ANON_COURT_SHA")
            canonical_manifest=read(ground["canonical_candidate_receipt"]["path"],
                                    ground["canonical_candidate_receipt"]["sha256"],
                                    "HIER_BATCH_CANONICAL_MANIFEST_SHA")
            pair_doc=read(ground["pair_diagnostic_receipt"]["path"],
                          ground["pair_diagnostic_receipt"]["sha256"],
                          "HIER_BATCH_PAIR_DOC_SHA")
            require(pair_doc["parents"]["source_sha256"]==old["source_sha256"]
                    and pair_doc["training_authorized"] is False
                    and pair_doc["pair_count"]==2
                    and base_consumer["parents"]["source_pair_receipt_sha256"]==
                        ground["pair_diagnostic_receipt"]["sha256"]
                    and candidate["parents"]["source_pair_receipt_sha256"]==
                        ground["pair_diagnostic_receipt"]["sha256"]
                    and candidate["parents"]["canonical_arrays_sha256"]==
                        canonical_manifest["canonical_arrays"]["sha256"]
                    and base_consumer["selected_slot_count"]==
                        old["selected_slot_count"],
                    "HIER_BATCH_COURT_PARENT_SOURCE_PAIR_BINDING")
            canon_npz=Path(canonical_manifest["canonical_arrays"]["path"]).resolve()
            require(canon_npz.is_file()
                    and sha(canon_npz)==canonical_manifest["canonical_arrays"]["sha256"],
                    "HIER_BATCH_SOURCE_POSE_ARRAYS_SHA")
            with np.load(canon_npz,allow_pickle=False) as arrays:
                train=[r for r in pair_doc["rows"] if r["view_index"]==0]
                h=build(arrays["rest_vertices_canonical"],
                        arrays["faces"],arrays["posed_vertices_canonical"],
                        candidate,base_consumer,train)
            receipt=target/"HIERARCHICAL_ANON_LAYER_GROUP_PROPOSALS_V1.json"
            save(receipt,h)
            decision=verify(h,pair_doc,candidate,base_consumer)
            decision["parents"]["hierarchical_proposal_sha256"]=sha(receipt)
            court_file=target/"HIERARCHICAL_ANON_LAYER_GROUP_COURT_V1.json"
            save(court_file,decision)
            report["subjects"][sid]={
                "status":"PASS_SOURCE_EXACT_HIERARCHY_AND_BOUNDED_EDIT_PROXY_ONLY",
                "source_sha256":old["source_sha256"],
                "source_face_count":candidate["face_count"],
                "parent_atomic_patch_count":decision["base_patch_count"],
                "selected_group_count":decision["selected_slot_count"],
                "selected_group_compression_ratio":decision["group_compression_ratio"],
                "heldout_reveal_fraction_diagnostic_only":decision[
                    "selected_heldout_edit_expressibility"],
                "measured_train_ray_count":decision["train_ray_count"],
                "measured_heldout_ray_count":decision["heldout_ray_count"],
                "hierarchy_candidate_count":h["actual_candidate_count"],
                "proposal_sha256":sha(receipt),
                "court_sha256":sha(court_file),
                "source_face_coverage_fraction":decision[
                    "independent_ray_face_coverage_fraction"],
                "full_image_alpha_depth_replay_executed":False,
                "product_compilation_authorized":False,
                "training_authorized":False,
            }
        except (MemoryError,KeyboardInterrupt):
            raise
        except Exception as exc:
            report["subjects"][sid]={
                "status":"FAIL_SOURCE_EXACT_HIERARCHICAL_LAYER_COURT",
                "error_type":type(exc).__name__,"reason":str(exc),
                "training_authorized":False,
            }
            save(target/"HIERARCHICAL_GROUP_FAILURE_V1.json",{
                "subject_id":sid,"status":"FAIL_SOURCE_EXACT_HIERARCHICAL_LAYER_COURT",
                "exception_type":type(exc).__name__,
                "reason":str(exc),
                "traceback":traceback.format_exc(),
                "training_authorized":False,
            })
        finally:
            save(result_file,report)
    counts=Counter(x["status"] for x in report["subjects"].values())
    report["status"]="THREE_HIERARCHICAL_GROUP_CANDIDATES_ACCOUNTED__FULL_RENDER_REQUIRED"
    report["candidate_qualified_subject_count"]=counts[
        "PASS_SOURCE_EXACT_HIERARCHY_AND_BOUNDED_EDIT_PROXY_ONLY"]
    report["failure_count"]=counts["FAIL_SOURCE_EXACT_HIERARCHICAL_LAYER_COURT"]
    report["candidate_group_counts"]={
        sid:v["selected_group_count"] for sid,v in report["subjects"].items()
        if v["status"].startswith("PASS_")
    }
    report["heldout_proxy_reveal_count"]=sum(
        int(v.get("heldout_reveal_fraction_diagnostic_only")==1.0)
        for v in report["subjects"].values())
    save(result_file,report)
    print(json.dumps({
        "subject_count":len(report["subjects"]),
        "candidate_qualified_subject_count":report["candidate_qualified_subject_count"],
        "failure_count":report["failure_count"],
        "selected_group_counts":sorted(report["candidate_group_counts"].values()),
        "heldout_proxy_reveal_count":report["heldout_proxy_reveal_count"],
        "product_compilation_authorized":False,
        "training_authorized":False,
    },sort_keys=True))
    require(report["candidate_qualified_subject_count"]==3
            and report["failure_count"]==0,
            "HIER_BATCH_TRUE_PRIVATE_SUBJECT_FAILURES")
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--native-visibility-join-root",type=Path,required=True)
    parser.add_argument("--prior-anonymous-layer-root",type=Path,required=True)
    parser.add_argument("--output-root",type=Path,required=True)
    a=parser.parse_args()
    run(a.native_visibility_join_root,a.prior_anonymous_layer_root,a.output_root)


if __name__=="__main__":
    main()
