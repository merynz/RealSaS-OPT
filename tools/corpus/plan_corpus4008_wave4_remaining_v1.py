#!/usr/bin/env python3
"""Expand source IMPORT OBSERVATION beyond first 1024 without reimport overlap.

Exact shard-private source SHA is shared with immutable recovery plan. The
Wave1/2 first 8 and Wave3 next 24 sources are reconstructed to exclude.
Wave4 chooses <=64 further SHA-distinct FIT originals, still not training.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from tools.corpus.plan_corpus4008_wave1_import_v1 import (
    FORMATS,SHA,SOURCE_STATUSES,quantile_pick,plan as wave1,require,sha,
)
from tools.corpus.plan_corpus4008_wave3_disjoint_v1 import next_plan as wave3

SCHEMA="RealSaS.Corpus4008Wave4DisjointOriginalSourcePlan.v1"


def plan(path:Path,*,shard:int,limit:int=64):
    require(type(shard) is int and 0<=shard<32,"WAVE4_SHARD")
    require(1<=limit<=80,"WAVE4_SOURCE_LIMIT")
    first=wave1(path,shard_index=shard,limit=8)
    third=wave3(path,shard_index=shard,limit=24)
    excluded=set(x["source_sha256"] for x in first["entries"])
    excluded.update(x["source_sha256"] for x in third["entries"])
    require(len(excluded)==len(first["entries"])+len(third["entries"])
            and len(excluded)==32,
            "WAVE4_FIRST1024_NOT_DISJOINT")
    j=json.loads(path.read_text())
    rows=j.get("candidates_private",[])
    eligible={}
    for x in rows:
        if x.get("historical_split")!="FIT":continue
        if x.get("source_bytes_recovered") is not True:continue
        if x.get("status") not in SOURCE_STATUSES:continue
        original=x.get("source_expected_sha256")
        ext=x.get("file_type")
        if not isinstance(original,str) or not SHA.fullmatch(original):continue
        if ext not in FORMATS:continue
        if x.get("retrieved_source_sha256")!=original:continue
        if x.get("private_staged_filename")!=original+ext:continue
        n=x.get("byte_count")
        if type(n) is not int or not 0<n<=32*1024*1024:continue
        require(x.get("training_authorized") is False
                and x.get("current_license_revalidated") is False,
                "WAVE4_UNVERIFIED_RIGHTS_CANNOT_ADMIT_TRAINING")
        require(original not in eligible,"WAVE4_REPEATED_SOURCE_IN_SHARD")
        eligible[original]=x
    require(excluded.issubset(eligible),"WAVE4_PREVIOUS_PLAN_LINEAGE_CHANGED")
    remain={sha:x for sha,x in eligible.items() if sha not in excluded}
    if not remain:
        selected=[]
    else:
        budget=min(limit,len(remain))
        pools={
            f:sorted((x for x in remain.values() if x["file_type"]==f),
                     key=lambda x:x["source_expected_sha256"])
            for f in FORMATS
        }
        chosen=[]
        for ext in sorted(FORMATS,key=lambda x:(-len(pools[x]),x)):
            if pools[ext] and len(chosen)<budget:
                chosen.extend(quantile_pick(pools[ext],1))
        used={x["source_expected_sha256"] for x in chosen}
        rest=sorted((x for h,x in remain.items() if h not in used),
                    key=lambda x:x["source_expected_sha256"])
        selected=chosen+quantile_pick(rest,budget-len(chosen))
    require(len(selected)==len({x["source_expected_sha256"] for x in selected})
            and not excluded.intersection(
                x["source_expected_sha256"] for x in selected),
            "WAVE4_SOURCE_REIMPORT")
    out={
        "schema":SCHEMA,
        "shard":shard,
        "recovery_shard_report_sha256":sha(path),
        "previous_wave_shard_8_sha256":sorted(x["source_sha256"]
                                             for x in first["entries"]),
        "previous_wave_shard_24_sha256":sorted(x["source_sha256"]
                                              for x in third["entries"]),
        "previous_wave_shard_reserved_count":32,
        "eligible_fit_original_count":len(eligible),
        "new_selected_count":len(selected),
        "post_wave4_remaining_eligible_fit":len(remain)-len(selected),
        "selection_method":"DISJOINT_32_EXCLUDED_PLUS_FORMAT_COVERAGE_AND_SHA_QUANTILES",
        "entries":[{
            "canonical_asset_id":x["canonical_asset_id"],
            "candidate_id":x["candidate_id"],
            "source_sha256":x["source_expected_sha256"],
            "source_ext":x["file_type"],
            "source_byte_count":x["byte_count"],
            "source_quarantine_basename":x["private_staged_filename"],
            "source_shard":shard,
            "split":"FIT",
            "original_rights_courted":False,
            "source_morphology_courted":False,
            "training_authorized":False,
        } for x in selected],
        "training_authorized":False,
        "materialization_authorized":False,
    }
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--shard-report",type=Path,required=True)
    p.add_argument("--shard-index",type=int,required=True)
    p.add_argument("--limit",type=int,default=64)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    result=plan(a.shard_report,shard=a.shard_index,limit=a.limit)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"shard":a.shard_index,
                      "previous_imported_source_count":32,
                      "new_source_count":result["new_selected_count"],
                      "fit_remaining_after_wave4":result["post_wave4_remaining_eligible_fit"],
                      "training_authorized":False},sort_keys=True))


if __name__=="__main__":
    main()
