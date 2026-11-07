#!/usr/bin/env python3
"""Derive a fail-closed per-target quarantine overlay, never a training PASS.

Source routes and cross-corpus identity evidence must be immutable and
SHA-pinned. This overlay never changes the original 161,986 route records.
"""
from __future__ import annotations
import argparse,csv,hashlib,json
from collections import defaultdict
from pathlib import Path

def req(ok,code):
    if not ok: raise ValueError("IDENTITY_QUARANTINE_"+code)

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def csvrows(path):
    with path.open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))

def run(pairs_path,identity_summary_path,route_path,out,*,expected_pairs=53,expected_per_source=37,expected_route_rows=161986):
    manifest=json.loads(identity_summary_path.read_text())
    req(manifest["schema"]=="RealSaS.CrossCorpusSourceIdentityCourt.v1"
        and manifest["training_authorized"] is False
        and manifest["matched_sha_prefix_pairs"]==expected_pairs,
        "IDENTITY_MANIFEST")
    # Accept the upstream identity court's SHA name even when local export naming differs.
    expected_pair_sha=manifest.get("pair_file_sha256",manifest.get("pairs_sha256"))
    req(sha(pairs_path)==expected_pair_sha,"PAIR_COURT_SHA")
    pairs=csvrows(pairs_path)
    routes=csvrows(route_path)
    req(len(pairs)==expected_pairs and len(routes)==expected_route_rows,
        "ROW_CENSUS")
    grouped=defaultdict(list)
    for item in routes:
        req(item["training_authorized"]=="False"
            and item["loss_mask_valid"]=="False"
            and item["negative_label_assigned"]=="False",
            "ALREADY_PROMOTED_ROUTE")
        grouped[item["global_record_key"]].append(item)
    output=[];seen=set();high_risk=0
    for pair in pairs:
        key_a="MASTER3993:"+pair["master_asset_id"]
        key_b="CORE370:"+pair["core_subject_id"]
        priority=pair.get("priority",pair.get("leakage_priority",""))
        evidence=pair["identity_evidence"]
        req(priority in ("P0_EVALUATION_HOLDOUT_SPLIT_RISK",
                         "P1_CROSS_CORPUS_SPLIT_UNRECONCILED",
                         "P0_PROTECTED_SPLIT",
                         "P1_CROSS_CORPUS_SPLIT_UNSET"),"UNKNOWN_PRIORITY")
        for key in (key_a,key_b):
            req(key not in seen,"DUPLICATE_IDENTITY_MEMBER")
            seen.add(key)
            source_routes=grouped.get(key,[])
            req(len(source_routes)==expected_per_source
                and len({(x["head"],x["target"]) for x in source_routes})
                    ==expected_per_source,"MISSING_OR_DUPLICATE_TARGET")
            for r in source_routes:
                if priority.startswith("P0_"):high_risk+=1
                output.append(dict(r,
                    identity_duplicate_candidate=True,
                    identity_master_asset_id=pair["master_asset_id"],
                    identity_core_subject_id=pair["core_subject_id"],
                    identity_evidence=evidence,
                    split_leakage_priority=priority,
                    disposition="QUARANTINED_PENDING_SOURCE_IDENTITY_SPLIT_COURT",
                    training_authorized=False,loss_mask_valid=False))
    req(len(seen)==expected_pairs*2 and
        len(output)==expected_pairs*2*expected_per_source,
        "QUARANTINE_COUNT")
    out.mkdir(parents=True,exist_ok=True)
    output_file=out/"CROSS_CORPUS_IDENTITY_BLOCKED_HEAD_TARGET_ROUTES_V1.csv"
    with output_file.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,list(output[0]))
        writer.writeheader();writer.writerows(output)
    report={
       "schema":"RealSaS.CrossCorpusIdentityHeadTargetQuarantine.v1",
       "source_records_quarantined":len(seen),
       "identity_pairs_quarantined":len(pairs),
       "head_target_routes_quarantined":len(output),
       "protected_split_risk_target_routes":high_risk,
       "source_identity_pairs_sha256":sha(pairs_path),
       "global_head_targets_sha256":sha(route_path),
       "blocked_target_sha256":sha(output_file),
       "all_loss_masks_false":True,
       "training_authorized":False,
       "unique_character_count_certified":False,
       "no_corpus_mutation":True,
    }
    (out/"CROSS_CORPUS_IDENTITY_QUARANTINE_SUMMARY_V1.json").write_text(
        json.dumps(report,sort_keys=True,indent=2)+"\n")
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("pairs","identity-summary","routes","output"):
        p.add_argument("--"+k,required=True,type=Path)
    x=p.parse_args()
    print(json.dumps(run(x.pairs,x.identity_summary,x.routes,x.output),sort_keys=True))
