#!/usr/bin/env python3
"""Select next bounded FIT original-source tranche disjoint from first 256.

For every recovered source shard, the first 8 source identities are selected
by the canonical Wave1/2 source planner, not by this tool. Wave3 excludes
exactly those first 8 and chooses <=24 further candidate identities. No
source licenses, character class, control behavior, or Graphics 2D art minted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.corpus.plan_corpus4008_wave1_import_v1 import (
    FORMATS, SHA, SOURCE_SCHEMA, SOURCE_STATUSES, quantile_pick,
    plan as first_wave, require, sha,
)

SCHEMA="RealSaS.Corpus4008Wave1SourceImportPlan.v1"


def next_plan(shard_report:Path, *, shard_index:int,
              limit:int=24, max_bytes:int=32*1024*1024):
    require(1<=limit<=40, "WAVE3_LIMIT")
    require(0<=shard_index<32, "WAVE3_SHARD")
    prior=first_wave(shard_report,shard_index=shard_index,
                     limit=8,max_original_bytes=max_bytes)
    doc=json.loads(shard_report.read_text())
    summary=doc.get("summary",{})
    require(summary.get("schema")==SOURCE_SCHEMA and
            summary.get("shard_index")==shard_index and
            summary.get("shard_count")==32 and
            summary.get("private_quarantine_only") is True and
            summary.get("training_authorized") is False,
            "WAVE3_SHARD_LINEAGE")
    excluded={x["source_sha256"] for x in prior["entries"]}
    eligible={}
    for x in doc.get("candidates_private",[]):
        if (x.get("historical_split")!="FIT"
            or x.get("source_bytes_recovered") is not True
            or x.get("status") not in SOURCE_STATUSES):
            continue
        val=x.get("source_expected_sha256")
        if not isinstance(val,str) or not SHA.fullmatch(val):continue
        ext=x.get("file_type")
        if ext not in FORMATS:continue
        if (x.get("retrieved_source_sha256")!=val
                or x.get("private_staged_filename")!=val+ext):
            continue
        if (not isinstance(x.get("byte_count"),int)
                or not 0<x["byte_count"]<=max_bytes):
            continue
        require(x.get("training_authorized") is False and
                x.get("current_license_revalidated") is False,
                "WAVE3_SOURCE_ADMISSION")
        require(val not in eligible,"WAVE3_DUPLICATE_SHA")
        eligible[val]=x
    require(excluded.issubset(eligible),"WAVE3_EXCLUDED_SET_DRIFT")
    remain=[x for h,x in eligible.items() if h not in excluded]
    require(remain,"WAVE3_NO_REMAINING_FIT")
    limit=min(limit,len(remain))
    groups={fmt:sorted(
        (x for x in remain if x["file_type"]==fmt),
        key=lambda x:x["source_expected_sha256"]) for fmt in FORMATS}
    order=sorted(FORMATS,key=lambda e:(-len(groups[e]),e))
    selected=[]
    for fmt in order:
        if len(selected)<limit and groups[fmt]:
            selected.extend(quantile_pick(groups[fmt],1))
    selected_sha={x["source_expected_sha256"] for x in selected}
    rest=sorted((x for x in remain
                  if x["source_expected_sha256"] not in selected_sha),
                key=lambda x:x["source_expected_sha256"])
    selected.extend(quantile_pick(rest,limit-len(selected)))
    require(len(selected)==limit and
            len({x["source_expected_sha256"] for x in selected})==limit and
            not excluded.intersection(
                {x["source_expected_sha256"] for x in selected}),
            "WAVE3_SOURCE_OVERLAP")
    return {
        "schema":SCHEMA,
        "shard_index":shard_index,
        "source_shard_report_sha256":sha(shard_report),
        "input_verified_fit_candidate_count":len(eligible),
        "prior_wave_reserved_source_sha256":sorted(excluded),
        "prior_wave_reserved_count":len(excluded),
        "sample_method":"DISJOINT_AFTER_FIRST8_PER_SHARD__BALANCED_FORMAT_PLUS_SHA_QUANTILES",
        "sample_size":limit,
        "entries":[{
          "canonical_asset_id":x["canonical_asset_id"],
          "candidate_id":x["candidate_id"],
          "source_sha256":x["source_expected_sha256"],
          "source_ext":x["file_type"],
          "source_byte_count":x["byte_count"],
          "source_quarantine_basename":x["private_staged_filename"],
          "source_shard_index":shard_index,
          "split":"FIT",
          "original_rights_courted":False,
          "source_character_courted":False,
          "training_authorized":False,
        } for x in selected],
        "model_training_authorized":False,
        "not_a_statistically_representative_training_cohort":True,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--shard-report",type=Path,required=True)
    p.add_argument("--shard-index",type=int,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--limit",type=int,default=24)
    a=p.parse_args()
    out=next_plan(a.shard_report,shard_index=a.shard_index,limit=a.limit)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps({
       "shard":a.shard_index,"new_source_count":out["sample_size"],
       "prior_wave_excluded":out["prior_wave_reserved_count"],
       "total_unique_planned_this_shard":(
           out["sample_size"]+out["prior_wave_reserved_count"]),
       "training_authorized":False,
    },sort_keys=True))


if __name__=="__main__":
    main()
