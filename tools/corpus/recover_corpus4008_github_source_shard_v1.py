"""Deterministic resumable source recovery shard, *private quarantine only*.

A source candidate belongs to exactly one of 32 static shards. A bounded first
pilot processes at most N non-holdout candidates in that shard. This is NOT a
truth builder and does not modify original Master Corpus authority.
"""
from __future__ import annotations
import argparse,hashlib,json
from collections import Counter
from pathlib import Path
import requests

from tools.corpus.recover_corpus4008_github_original_pilot_v1 import (
    fetch_one,pinned_raw_url,require,SCHEMA as PILOT_SCHEMA,
)
SCHEMA="RealSaS.Corpus4008PinnedGithubOriginalShard.v1"
PLAN_SHA="3c1088fb26d2171ccde8b8dfaac3406a26ba12e16eb3b6506c11c8ccc775426f"


def owner_for_sha(sha,shards):
    return int(hashlib.sha256(sha.encode()).hexdigest(),16)%shards


def run(plan,output,*,shard_count,shard_index,max_objects=16,max_bytes=64*1024*1024):
    require(shard_count == 32, "FROZEN_SHARD_COUNT")
    require(0<=shard_index<shard_count,"SHARD_INDEX")
    require(1<=max_objects<=128,"MAX_OBJECTS_PER_RUN")
    require(hashlib.sha256(plan.read_bytes()).hexdigest()==PLAN_SHA,
            "PRIVATE_PLAN_SHA_CHANGED")
    d=json.loads(plan.read_text())
    require(d.get("schema")=="RealSaS.Corpus4008MissingObjaverseRecoveryPlan.v1"
            and d["summary"]["missing_original_count"]==3349,"PRIVATE_RECOVERY_PLAN")
    rows=d["missing_originals_private"]
    eligible=[]
    for r in rows:
        if r["historical_split"]=="EXTERNAL_HOLDOUT":
            continue
        if r["source_candidate_metadata_count"]!=1 or r["historical_legal_decision"]!="PASS":
            continue
        try:
            pinned_raw_url(r["source_url_private"])
        except ValueError:
            continue
        if owner_for_sha(r["source_expected_sha256"],shard_count)==shard_index:
            eligible.append(r)
    eligible.sort(key=lambda x:x["source_expected_sha256"])
    output.mkdir(parents=True,exist_ok=True)
    cache=output/"verified_quarantine_objects";cache.mkdir(exist_ok=True)
    selected=eligible[:max_objects]
    with requests.Session() as session:
        outcomes=[fetch_one(r,cache,max_bytes=max_bytes,session=session) for r in selected]
    counts=Counter(x["status"] for x in outcomes)
    summary={
        "schema":SCHEMA,
        "status":"BOUNDED_SHA_PINNED_SOURCE_RECOVERY__NO_MODEL_ADMISSION",
        "source_plan_sha256":PLAN_SHA,
        "source_universe_missing":3349,
        "shard_count":shard_count,
        "shard_index":shard_index,
        "eligible_in_shard_count":len(eligible),
        "selected_count":len(selected),
        "deferred_count":len(eligible)-len(selected),
        "result_statuses":dict(sorted(counts.items())),
        "exact_source_bytes_recovered":sum(x["source_bytes_recovered"] for x in outcomes),
        "exact_source_verified_byte_total":sum(x["byte_count"] for x in outcomes
                                               if x["source_bytes_recovered"]),
        "current_license_revalidated":False,
        "source_character_class_courted":False,
        "source_anatomical_orientation_courted":False,
        "training_authorized":False,
        "private_quarantine_only":True,
    }
    (output/"SHARD_PRIVATE.json").write_text(json.dumps({
        "summary":summary,"candidates_private":outcomes},indent=2,sort_keys=True)+"\n")
    (output/"SHARD_PUBLIC.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    return summary


def main():
    a=argparse.ArgumentParser()
    a.add_argument("--plan",type=Path,required=True)
    a.add_argument("--output",type=Path,required=True)
    a.add_argument("--shard-count",type=int,default=32)
    a.add_argument("--shard-index",type=int,required=True)
    a.add_argument("--max-objects",type=int,default=16)
    a.add_argument("--max-original-mib",type=int,default=64)
    args=a.parse_args()
    print(json.dumps(run(args.plan,args.output,shard_count=args.shard_count,
                         shard_index=args.shard_index,max_objects=args.max_objects,
                         max_bytes=args.max_original_mib*1024*1024),sort_keys=True))


if __name__=="__main__":
    main()
