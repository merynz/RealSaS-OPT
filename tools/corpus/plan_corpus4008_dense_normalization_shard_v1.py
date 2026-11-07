#!/usr/bin/env python3
"""Use audited source readback receipt to select up to 32 same-shard originals.

No new sampling, no source/clip duplicates, no holdout, no per-model PASS.
Only trust a pinned SHA-verified shard CSV and its summary.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

SCHEMA="RealSaS.Corpus4008DenseNormalizationShardPlan.v1"
SOURCE_SCHEMA="RealSaS.Corpus4008ParallelShardReadback.v1"
SUCCESS="IMPORTED_SOURCE_OBSERVATIONS_ONLY"


def require(v,code):
    if not v: raise ValueError("C4008_DENSE_PLAN_"+code)


def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda:f.read(4<<20),b""):
            h.update(part)
    return h.hexdigest()


def plan(summary_file:Path, csv_file:Path, shard:int):
    require(type(shard) is int and 0<=shard<32,"SHARD")
    require(summary_file.is_file() and csv_file.is_file(),"SOURCE_RECEIPT_MISSING")
    s=json.loads(summary_file.read_text())
    require(s.get("schema")==SOURCE_SCHEMA
            and s.get("shard")==shard
            and s.get("full_import_receipts_sha_verified") is True
            and s.get("training_authorized") is False
            and s.get("per_head_training_admitted_count")==0
            and s.get("typed_observations_csv_sha256")==digest(csv_file),
            "SOURCE_READBACK_PROOF")
    with csv_file.open(newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    require(len(rows)==s["total_unique_source_count"]==32,
            "EXPECTED_32_DISTINCT_SOURCE_ROWS")
    candidates=[]
    seen=set()
    for r in rows:
        sha=r["source_sha256"]
        aid=r["canonical_asset_id"]
        require(aid not in seen,"DUPLICATE_SOURCE")
        seen.add(aid)
        require(len(sha)==64 and all(c in "0123456789abcdef" for c in sha)
                and r["candidate_id"].endswith(sha)
                and r["source_format"] in (".fbx",".glb",".gltf")
                and r["source_shard"]==str(shard)
                and r["split"]=="FIT"
                and r["source_import_observation_status"]==SUCCESS
                and r["head_admission_count"]=="0"
                and r["training_authorized"]=="False"
                and r["functional_native_ik_fk"]=="UNKNOWN",
                "SOURCE_IDENTITY_OR_STATUS")
        candidates.append({
            "canonical_asset_id":aid,
            "candidate_id":r["candidate_id"],
            "original_source_sha256":sha,
            "original_source_extension":r["source_format"],
            "source_recovery_shard":shard,
            "split":"FIT",
            "source_character_vs_prop":"UNKNOWN",
            "source_import_success_observation_only":True,
            "source_rights_courted":False,
            "training_authorized":False,
        })
    require(len(candidates)==len({x["original_source_sha256"] for x in candidates}),
            "REPEATED_SHA")
    return {
        "schema":SCHEMA,
        "shard":shard,
        "source_readback_summary_sha256":digest(summary_file),
        "source_readback_csv_sha256":digest(csv_file),
        "source_count":len(candidates),
        "source_candidates":sorted(candidates,
                                  key=lambda x:x["original_source_sha256"]),
        "scope":"SOURCE_IMPORTED_VERTEX_BONE_SPARSE_WEIGHT_MATH_ONLY",
        "source_truth_courted":False,
        "training_authorized":False,
        "materialization_authorized":False,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--summary",type=Path,required=True)
    p.add_argument("--observations",type=Path,required=True)
    p.add_argument("--shard",type=int,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    x=plan(a.summary,a.observations,a.shard)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(x,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"shard":a.shard,"unique_original_sources":x["source_count"],
                      "materialization_authorized":False},sort_keys=True))


if __name__=="__main__":
    main()
