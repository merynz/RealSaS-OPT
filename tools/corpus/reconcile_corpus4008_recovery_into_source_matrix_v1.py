#!/usr/bin/env python3
"""Join all 32 *private* immutable full-recovery shard reports to Master3993.

Source byte availability is NOT character class, commercial license, semantic,
render, deform, independent source scene or training admission. Old 659
Drive *path* matches are only marked LEGACY_PATH_MATCH_UNREHASHED.
NO source bytes or private GitHub blob URLs are exported in matrix output.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

SCHEMA="RealSaS.Corpus4008SourceAvailabilityMatrix.v1"
SELECT_SHA="af2436d2a25a6f715e2d81af14b731837c7b02b609f1a4fc206acb591beb61c9"
PLAN_SHA="3c1088fb26d2171ccde8b8dfaac3406a26ba12e16eb3b6506c11c8ccc775426f"
SHARD_SCHEMA="RealSaS.Corpus4008PinnedGithubOriginalShard.v1"
PLAN_SCHEMA="RealSaS.Corpus4008MissingObjaverseRecoveryPlan.v1"
SHA=re.compile(r"^[a-f0-9]{64}$")
SUCCESSES={
    "EXACT_ORIGINAL_BYTES_STAGED__COURTS_PENDING",
    "EXACT_SOURCE_CACHED_AND_REVERIFIED",
}


def require(x,msg):
    if not x:
        raise ValueError("C4008_SOURCE_AVAILABILITY_"+msg)


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(4<<20),b""):
            h.update(b)
    return h.hexdigest()


def run(selection:Path,plan:Path,shards:Path,output:Path,*,
        selection_count:int=3993,
        expected_missing:int=3349,
        shard_count:int=32,
        expected_plan_sha:str=PLAN_SHA,
        expected_selection_sha:str=SELECT_SHA):
    require(shard_count>=1,"SHARD_COUNT")
    require(sha(selection)==expected_selection_sha,"MASTER_SHA")
    require(sha(plan)==expected_plan_sha,"PRIVATE_PLAN_SHA")
    master=json.loads(selection.read_text())
    raw=json.loads(plan.read_text())
    require(isinstance(master,dict) and len(master)==selection_count,
            "MASTER_COUNT")
    require(raw.get("schema")==PLAN_SCHEMA,"PLAN_SCHEMA")
    missing=raw.get("missing_originals_private")
    require(isinstance(missing,list) and len(missing)==expected_missing,
            "PLAN_MISSING_COUNT")
    require(raw["summary"].get("missing_original_count")==expected_missing,
            "PLAN_SUMMARY_COUNT")
    indexed={}
    for row in missing:
        aid=row["canonical_asset_id"]
        require(aid in master and aid not in indexed,"PLAN_ASSET_DUP_OR_UNKNOWN")
        require(row["candidate_id"]==master[aid]["candidate_id"],
                "PLAN_CANDIDATE_ID")
        require(row["historical_split"]==master[aid]["split"],
                "PLAN_SPLIT")
        full=row["source_expected_sha256"]
        require(isinstance(full,str) and SHA.fullmatch(full),
                "PLAN_SOURCE_SHA")
        token=master[aid]["candidate_id"].split(":",1)[1]
        require(len(token) in (24,64) and full.startswith(token),
                "PLAN_SOURCE_TOKEN")
        indexed[aid]=row
    seen={}
    shard_totals=Counter()
    for i in range(shard_count):
        f=shards/("shard-%d"%i)/"SHARD_PRIVATE.json"
        require(f.is_file(),"SHARD_REPORT_MISSING_%d"%i)
        j=json.loads(f.read_text())
        summary=j.get("summary",{})
        require(summary.get("schema")==SHARD_SCHEMA,"SHARD_SCHEMA")
        require(summary.get("shard_count")==shard_count
                and summary.get("shard_index")==i,"SHARD_INDEX")
        if expected_plan_sha==PLAN_SHA:
            require(summary.get("source_plan_sha256")==PLAN_SHA,
                    "SHARD_PLAN_LINEAGE")
        require(summary.get("private_quarantine_only") is True
                and summary.get("training_authorized") is False,
                "SHARD_NO_ADMISSION")
        require(summary.get("deferred_count")==0,"SHARD_HAS_DEFERRED")
        rows=j.get("candidates_private",[])
        require(len(rows)==summary.get("selected_count")==
                summary.get("eligible_in_shard_count"),
                "SHARD_SELECTED_COUNT")
        for o in rows:
            aid=o["canonical_asset_id"]
            require(aid in indexed and aid not in seen,"SHARD_FOREIGN_OR_DUP_ASSET")
            p=indexed[aid]
            require(p["historical_split"]!="EXTERNAL_HOLDOUT",
                    "HOLDOUT_IN_RECOVERY")
            require(o["candidate_id"]==p["candidate_id"]
                    and o["source_expected_sha256"]==p["source_expected_sha256"]
                    and o["historical_split"]==p["historical_split"],
                    "SHARD_SOURCE_IDENTITY")
            success=o.get("source_bytes_recovered") is True
            if success:
                require(o.get("status") in SUCCESSES and
                        o.get("retrieved_source_sha256")==p["source_expected_sha256"]
                        and isinstance(o.get("private_staged_filename"),str)
                        and o["private_staged_filename"].startswith(
                            p["source_expected_sha256"]+"."),
                        "RECOVERY_SUCCESS_WITHOUT_RECEIPT")
            else:
                require(o.get("status") not in SUCCESSES
                        and not o.get("retrieved_source_sha256")==p["source_expected_sha256"],
                        "FAILURE_MISREPRESENTED_AS_SOURCE")
            require(o.get("training_authorized") is False
                    and o.get("current_license_revalidated") is False,
                    "RECOVERY_PRIVILEGE_ESCALATION")
            seen[aid]=o
            shard_totals[o["status"]]+=1
    # Recovered: all eligible nonholdouts in the frozen source plan.
    nonhold=[x for x in indexed.values()
             if x["historical_split"]!="EXTERNAL_HOLDOUT"]
    require(len(seen)==len(nonhold),"ELIGIBLE_RECONCILIATION")
    holdouts=[x for x in indexed.values()
              if x["historical_split"]=="EXTERNAL_HOLDOUT"]
    require(all(x["canonical_asset_id"] not in seen for x in holdouts),
            "HOLDOUT_RECOVERED_UNDER_TRAIN")
    counts=Counter()
    rows=[]
    for aid,orig in sorted(master.items()):
        require(orig["canonical_asset_id"]==aid,"MASTER_ID")
        split=orig["split"]
        if aid in seen:
            o=seen[aid]
            status=("SHA_VERIFIED_RECOVERY_RECEIPT" if
                    o["source_bytes_recovered"] else
                    "RECOVERY_EXCEPTION")
            failure=None if o["source_bytes_recovered"] else o["status"]
            receipt=o.get("retrieved_source_sha256")
            src_sha=o["source_expected_sha256"]
            bytes_verified=o["byte_count"] if o["source_bytes_recovered"] else 0
        elif aid in indexed:
            status="PROTECTED_HOLDOUT_ORIGINAL_NOT_TOUCHED"
            require(split=="EXTERNAL_HOLDOUT","UNRECOVERED_ELIGIBLE")
            failure=None;receipt=None;src_sha=indexed[aid]["source_expected_sha256"]
            bytes_verified=0
        else:
            # Original 659 prior-path matches included 15 QA-only, so 644
            # Master-selected preexisting paths, not independently rehashed.
            status="LEGACY_PATH_MATCH_UNREHASHED"
            failure=None;receipt=None;src_sha=None;bytes_verified=0
        counts[status]+=1
        rows.append({
            "canonical_asset_id":aid,
            "provider":orig["source_registry_id"],
            "candidate_id":orig["candidate_id"],
            "historical_split":split,
            "holdout_protected":split=="EXTERNAL_HOLDOUT",
            "historical_decision_not_training":orig.get("decision"),
            "source_availability_status":status,
            "source_expected_sha256":src_sha,
            "sha_verified_in_immutable_recovery_receipt":receipt,
            "recovery_exception":failure,
            "reported_verified_byte_count":bytes_verified,
            "original_source_license_rights":"UNKNOWN",
            "source_character_vs_prop":"UNKNOWN",
            "source_morphology":"UNKNOWN",
            "anatomical_front_up_camera":"UNKNOWN",
            "imported_mesh_and_texture_court":"UNKNOWN",
            "native_control_ik_fk_court":"UNKNOWN",
            "source_skin_and_deformation_court":"UNKNOWN",
            "semantic_region_court":"UNKNOWN",
            "original_source_rgba_court":"UNKNOWN",
            "graphics_stylized_2d_art_court":"UNKNOWN",
            "graphics_image_target_domain":"STYLIZED_2D_GAME_ART_REQUIRED",
            "all_model_heads_admitted":False,
            "training_authorized":False,
        })
    require(len(rows)==selection_count,"OUTPUT_COUNT")
    report={
        "schema":SCHEMA,
        "selection_sha256":sha(selection),
        "recovery_plan_sha256":sha(plan),
        "shard_count":shard_count,
        "historical_master_source_selected":selection_count,
        "qa_only_outside_selected":15 if selection_count==3993 else None,
        "historical_missing_path_universe":expected_missing,
        "shard_recovery_status_counts":dict(sorted(shard_totals.items())),
        "source_stage_counts":dict(sorted(counts.items())),
        "reported_recovered_sha_receipt_count":sum(
            bool(r["sha_verified_in_immutable_recovery_receipt"]) for r in rows),
        "reported_verified_source_bytes":sum(
            r["reported_verified_byte_count"] for r in rows),
        "source_bytes_not_rehashed_again_in_this_join":True,
        "source_schema_and_identity_reconciled":True,
        "all_per_head_truth_admission_counts_zero":True,
        "all_original_source_rights_revalidation_counts_zero":True,
        "graphics_2d_art_target_required":True,
        "training_authorized":False,
    }
    output.mkdir(parents=True,exist_ok=True)
    with (output/"SOURCE_AVAILABILITY_MATRIX_V1.csv").open(
            "w",newline="",encoding="utf-8") as fp:
        writer=csv.DictWriter(fp,list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output/"SOURCE_AVAILABILITY_SUMMARY_V1.json").write_text(
        json.dumps(report,sort_keys=True,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--selection",type=Path,required=True)
    p.add_argument("--plan",type=Path,required=True)
    p.add_argument("--shards",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(run(a.selection,a.plan,a.shards,a.output),sort_keys=True))


if __name__=="__main__":
    main()
