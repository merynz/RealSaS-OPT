#!/usr/bin/env python3
"""Repair only previous Corpus370 observation FAILs; rebind original PASS bytes.

Historical V3 successes are re-audited, never rendered, copied, renamed PASS_V4,
or silently excluded.  Every newly passing V4 observation carries its distinct
producer code/runtime binding.  Original V3 failure receipts remain immutable.
No family authority, split, materialization or training is minted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import traceback

from tools.corpus.audit_corpus370_historical_observation_reuse_v1 import inspect
from tools.corpus.prepare_corpus370_observation_packets_v4 import (
    _packet_complete, prepare as prepare_v4,
)

PARENT_SCHEMA = "RealSaS.Corpus370PreparedObservationPackets.v3"
SCHEMA = "RealSaS.Corpus370PreparedObservationPackets.v5"


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def require(condition, code):
    if not condition:
        raise ValueError(code)


def rescue(*, parent_index_path, expected_parent_sha256, roster_path,
           expected_roster_sha256, extracted_source_root, output_root,
           blender, probe, policy_path, runtime_request_path,
           selected_subject_ids=None):
    parent_index_path, roster_path, output_root = (
        Path(parent_index_path).resolve(), Path(roster_path).resolve(),
        Path(output_root).resolve(),
    )
    require(sha(parent_index_path) == expected_parent_sha256,
            "CORPUS370_DELTA_V5_PARENT_SHA_DRIFT")
    require(sha(roster_path) == expected_roster_sha256,
            "CORPUS370_DELTA_V5_ROSTER_SHA_DRIFT")
    parent=json.loads(parent_index_path.read_text())
    roster=json.loads(roster_path.read_text())
    policy=json.loads(Path(policy_path).read_text())
    require(parent.get("schema")==PARENT_SCHEMA and parent.get("training_authorized") is False
            and parent.get("subject_count")==370, "CORPUS370_DELTA_V5_PARENT_SCHEMA")
    require(parent.get("parent",{}).get("sha256")
            == "129b173b5d789cf4dc020f883ee8b16db8156380dd0cae939319ad3c279c139a",
            "CORPUS370_DELTA_V5_ORIGINAL_V2_PARENT_DRIFT")
    require(roster.get("schema")=="RealSaS.Corpus370ExactSourceRoster.v1"
            and roster.get("subject_count")==370 and roster.get("training_authorized") is False,
            "CORPUS370_DELTA_V5_ROSTER")
    require(policy.get("schema")=="RealSaS.IrisObservationPreparationPolicy.v1"
            and policy.get("maximum_mask_gap_px")==2.0
            and policy.get("training_authorized") is False,
            "CORPUS370_DELTA_V5_UNCHANGED_SOURCE_POLICY")
    roster_rows={r["subject_id"]:r for r in roster["subjects"]}
    existing={r["subject_id"]:dict(r) for r in parent["subjects"]}
    original_failures={r["subject_id"]:dict(r) for r in parent["failures"]}
    require(len(roster_rows)==370 and len(existing)==294 and len(original_failures)==76
            and not set(existing).intersection(original_failures)
            and set(existing).union(original_failures)==set(roster_rows),
            "CORPUS370_DELTA_V5_PARENT_PARTITION")
    for sid,row in {**existing,**original_failures}.items():
        require(row["source_sha256"]==roster_rows[sid]["source_sha256"],
                "CORPUS370_DELTA_V5_SOURCE_IDENTITY:"+sid)
    preserved=[]
    for sid,row in sorted(existing.items()):
        checked=inspect(row,policy)
        require(checked["status"]=="EXACT_HISTORICAL_V3_PASS_BYTES_REUSABLE",
                "CORPUS370_DELTA_V5_UNPROVEN_OLD_PASS:"+sid+":"+checked.get("reason",""))
        packet=Path(row["packet_path"]).resolve()
        import_path=packet.parent.parent/"imported"/sid/"imported.json"
        assert sha(import_path)==checked["original_import_sha256"]
        preserved.append({
            **row,
            "evidence_producer":"HISTORICAL_V3_PASS_BYTES_RECHECKED",
            "admitted_import_geometry":{
                "path":str(import_path), "sha256":checked["original_import_sha256"],
            },
            "legacy_admission_sha256":checked["legacy_admission_sha256"],
            "new_v4_admission_claimed":False,
            "training_authorized":False,
        })
    selected=set(selected_subject_ids) if selected_subject_ids else set(original_failures)
    require(selected.issubset(original_failures)
            and len(selected)==len(selected_subject_ids or selected),
            "CORPUS370_DELTA_V5_REPAIR_SCOPE_NOT_PARENT_FAILURE")
    output_root.mkdir(parents=True,exist_ok=True)
    failure_dir=output_root/"delta_failures_v5"
    failure_dir.mkdir(parents=True,exist_ok=True)
    newly_prepared=[]
    new_failures=[]
    v4_binding=None
    for i,sid in enumerate(sorted(selected),1):
        old=original_failures[sid]
        try:
            response=prepare_v4(
                roster_path,Path(extracted_source_root),output_root/"v4_new",
                Path(blender),Path(probe),Path(policy_path),
                Path(runtime_request_path),sid,
            )
            require(response.get("schema")=="RealSaS.Corpus370PreparedObservationPackets.v4"
                    and len(response.get("subjects") or [])==1,
                    "CORPUS370_DELTA_V5_NEW_V4_ROW_SCOPE")
            row=dict(response["subjects"][0])
            require(row["subject_id"]==sid and row["source_sha256"]==old["source_sha256"],
                    "CORPUS370_DELTA_V5_NEW_V4_SOURCE_DRIFT")
            bindings=response["binding"]
            if v4_binding is None:
                v4_binding=bindings
            else:
                require(v4_binding==bindings, "CORPUS370_DELTA_V5_REPAIR_CODE_DRIFT")
            require(_packet_complete(Path(row["packet_path"]),sid,old["source_sha256"],
                                     policy,bindings["blender_binary_sha256"]),
                    "CORPUS370_DELTA_V5_V4_PACKET_INCOMPLETE")
            newly_prepared.append({
                **row,
                "evidence_producer":"NEW_V4_REPAIR_FROM_ORIGINAL_FAIL",
                "parent_failure_receipt":old["receipt"],
                "parent_failure_reason":old.get("reason"),
                "training_authorized":False,
            })
            print(json.dumps({"event":"CORPUS370_DELTA_V5_PASS",
                "done":i,"selected":len(selected),"subject_id":sid}),flush=True)
        except (KeyboardInterrupt,MemoryError):
            raise
        except Exception as exc:
            failure={
                "subject_id":sid,"source_sha256":old["source_sha256"],
                "source_format":old.get("source_format"),
                "source_id":old.get("source_id"),
                "error_type":type(exc).__name__,"reason":str(exc),
                "traceback":traceback.format_exc(),
                "parent_failure":old,"original_parent_failure_unchanged":True,
                "new_v4_admission_minted":False,"training_authorized":False,
            }
            p=failure_dir/(sid+".json")
            p.write_text(json.dumps(failure,sort_keys=True,indent=2)+"\n")
            failure["receipt"]={"path":str(p),"sha256":sha(p)}
            new_failures.append(failure)
            print(json.dumps({"event":"CORPUS370_DELTA_V5_REPAIR_FAIL",
                "done":i,"selected":len(selected),"subject_id":sid,
                "error_type":type(exc).__name__,"reason":str(exc)}),flush=True)
    remaining_failures=[r for sid,r in sorted(original_failures.items()) if sid not in selected]
    subjects=sorted([*preserved,*newly_prepared],key=lambda r:r["subject_id"])
    failures=sorted([*remaining_failures,*new_failures],key=lambda r:r["subject_id"])
    require(len(subjects)+len(failures)==370
            and len({r["subject_id"] for r in subjects+failures})==370,
            "CORPUS370_DELTA_V5_FINAL_PARTITION")
    result={
        "schema":SCHEMA,
        "status":"CORPUS370_DELTA_OBSERVATION_PREPARATION__FAMILY_COURTS_PENDING",
        "subject_count":370,
        "prepared_subject_count":len(subjects),"failure_count":len(failures),
        "historical_v3_pass_preserved_count":len(preserved),
        "new_v4_repaired_count":len(newly_prepared),
        "new_v4_repair_failure_count":len(new_failures),
        "original_v3_failures_selected_count":len(selected),
        "unselected_original_failure_count":len(remaining_failures),
        "parent":{
            "schema":PARENT_SCHEMA,"path":str(parent_index_path),
            "sha256":expected_parent_sha256,
            "prepared_subject_count":294,"failure_count":76,
        },
        "exact_source_roster":{
            "path":str(roster_path),"sha256":expected_roster_sha256,
        },
        "binding":{
            "policy":{"path":str(Path(policy_path).resolve()),"sha256":sha(Path(policy_path))},
            "blender_binary_sha256":sha(Path(blender)),
            "new_v4_worker_binding":v4_binding,
            "historical_v3_worker_binding":parent.get("rescue",{}).get("worker_binding"),
        },
        "subjects":subjects,"failures":failures,
        "authority":{
            "original_v3_pass_bytes_rechecked":True,
            "old_pass_status_upgraded_to_v4":False,
            "failed_observation_subject_kept_in_candidate_universe":True,
            "family_authority_minted":False,"training_authorized":False,
            "platform_attempt_created":False,"split_minted":False,
        },
        "training_authorized":False,
    }
    path=output_root/"CORPUS370_PREPARED_OBSERVATION_PACKETS_V5.json"
    partial=path.with_suffix(".partial")
    partial.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    partial.replace(path)
    print(json.dumps({k:result[k] for k in
        ("prepared_subject_count","failure_count","historical_v3_pass_preserved_count",
        "new_v4_repaired_count","new_v4_repair_failure_count",
        "unselected_original_failure_count")},sort_keys=True),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-index",type=Path,required=True)
    p.add_argument("--expected-parent-sha256",required=True)
    p.add_argument("--exact-source-roster",type=Path,required=True)
    p.add_argument("--expected-roster-sha256",required=True)
    p.add_argument("--extracted-source-root",type=Path,required=True)
    p.add_argument("--output-root",type=Path,required=True)
    p.add_argument("--blender",type=Path,required=True)
    p.add_argument("--probe",type=Path,required=True)
    p.add_argument("--policy",type=Path,required=True)
    p.add_argument("--runtime-request",type=Path,required=True)
    p.add_argument("--subject-id",action="append",dest="subject_ids")
    args=p.parse_args()
    rescue(parent_index_path=args.parent_index,
           expected_parent_sha256=args.expected_parent_sha256,
           roster_path=args.exact_source_roster,
           expected_roster_sha256=args.expected_roster_sha256,
           extracted_source_root=args.extracted_source_root,
           output_root=args.output_root,blender=args.blender,probe=args.probe,
           policy_path=args.policy,runtime_request_path=args.runtime_request,
           selected_subject_ids=args.subject_ids)


if __name__=="__main__":
    main()
