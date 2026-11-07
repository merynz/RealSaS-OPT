#!/usr/bin/env python3
"""Deterministic diverse wave-1 source-scene observation from PRIVATE SHA receipts.

This samples exact recovered FIT sources for importer-risk and field-coverage
measurement. Not a representative model-training cohort. It cannot admit
source identity, commercial license, character morphology, semantics or head.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

SCHEMA="RealSaS.Corpus4008Wave1SourceImportPlan.v1"
SOURCE_SCHEMA="RealSaS.Corpus4008PinnedGithubOriginalShard.v1"
SOURCE_STATUSES={
    "EXACT_ORIGINAL_BYTES_STAGED__COURTS_PENDING",
    "EXACT_SOURCE_CACHED_AND_REVERIFIED",
}
SHA=re.compile(r"^[a-f0-9]{64}$")
FORMATS=(".fbx",".glb",".gltf")


def require(x,code):
    if not x:raise ValueError("C4008_WAVE1_"+code)


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(4<<20),b""):
            h.update(chunk)
    return h.hexdigest()


def quantile_pick(rows, count):
    if len(rows)<=count:return list(rows)
    indices=sorted(set(min(len(rows)-1,
        ((2*i+1)*len(rows))//(2*count)) for i in range(count)))
    require(len(indices)==count,"QUANTILE_UNIQUENESS")
    return [rows[i] for i in indices]


def plan(shard_file:Path,*,shard_index:int,limit:int=8,
         max_original_bytes:int=32*1024*1024):
    require(1<=limit<=16,"LIMIT")
    require(0<=shard_index<32,"SHARD_INDEX")
    j=json.loads(shard_file.read_text())
    s=j.get("summary",{})
    require(s.get("schema")==SOURCE_SCHEMA
            and s.get("shard_index")==shard_index
            and s.get("shard_count")==32
            and s.get("deferred_count")==0
            and s.get("private_quarantine_only") is True
            and s.get("training_authorized") is False,
            "SOURCE_SHARD_CERTIFICATE")
    valid=[]
    for x in j.get("candidates_private",[]):
        if x.get("historical_split")!="FIT": continue
        if x.get("status") not in SOURCE_STATUSES: continue
        if x.get("source_bytes_recovered") is not True:continue
        token=x.get("source_expected_sha256")
        if not isinstance(token,str) or not SHA.fullmatch(token):continue
        if x.get("retrieved_source_sha256")!=token:continue
        ext=x.get("file_type")
        if ext not in FORMATS:continue
        if x.get("private_staged_filename")!=token+ext:continue
        if not 0<x.get("byte_count",0)<=max_original_bytes:continue
        require(x.get("training_authorized") is False
                and x.get("current_license_revalidated") is False,
                "SOURCE_CANNOT_ADMIT_TRAINING")
        valid.append(x)
    require(valid,"NO_SAFE_FIT_CANDIDATES")
    # Balance source formats that actually exist; avoid selecting only
    # the very smallest or earliest-SHA objects in each shard.
    pools={ext:sorted((x for x in valid if x["file_type"]==ext),
                      key=lambda x:x["source_expected_sha256"])
           for ext in FORMATS}
    order=sorted(FORMATS,key=lambda e:(-len(pools[e]),e))
    chosen=[]
    # First select one example of each nonempty file type.
    for ext in order:
        if pools[ext] and len(chosen)<limit:
            chosen.extend(quantile_pick(pools[ext],1))
    used={x["canonical_asset_id"] for x in chosen}
    remain=sorted((x for x in valid if x["canonical_asset_id"] not in used),
                  key=lambda x:x["source_expected_sha256"])
    chosen.extend(quantile_pick(remain,max(0,limit-len(chosen))))
    require(len({x["canonical_asset_id"] for x in chosen})==len(chosen),
            "REPEATED_SOURCE_IDENTITY")
    return {
        "schema":SCHEMA,
        "shard_index":shard_index,
        "source_shard_report_sha256":sha(shard_file),
        "input_verified_fit_candidate_count":len(valid),
        "sample_method":"EACH_REPRESENTED_FORMAT_PLUS_DETERMINISTIC_SHA_QUANTILES",
        "sample_size":len(chosen),
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
        } for x in chosen],
        "model_training_authorized":False,
        "not_a_statistically_representative_training_cohort":True,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--shard-report",type=Path,required=True)
    p.add_argument("--shard-index",type=int,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--limit",type=int,default=8)
    a=p.parse_args()
    data=plan(a.shard_report,shard_index=a.shard_index,limit=a.limit)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(data,indent=2,sort_keys=True)+"\n")
    print(json.dumps({
        "shard":a.shard_index,"selected_count":data["sample_size"],
        "input_valid_fit":data["input_verified_fit_candidate_count"],
        "source_rights_qualified":0,"training_authorized":False,
    },sort_keys=True))


if __name__=="__main__":
    main()
