"""Fail-closed merge of full-universe sharded Corpus4008 original-file preflights.

Never merges training authority; all families remain UNKNOWN until independent
source/camera/semantic courts. Requires *all* shards, exact provenance, one
assigned source owner per asset, identical holdout and QA metadata everywhere.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

SCHEMA = "RealSaS.Corpus4008SourcePreflightCensus.v1"
MERGED = "RealSaS.Corpus4008MergedSourcePreflight.v1"


def require(condition, reason):
    if not condition:
        raise ValueError("CORPUS4008_MERGE_" + reason)


def load(path):
    doc = json.loads(path.read_text())
    require(doc["summary"]["schema"] == SCHEMA, "SHARD_SCHEMA")
    require(doc["summary"]["universe_count"] == 4008 and len(doc["subjects"]) == 4008,
            "FULL_4008_REQUIRED")
    require(doc["summary"]["training_authorized"] is False, "TRAINING_AUTHORITY")
    return doc


def merge(paths):
    require(bool(paths), "NO_INPUT")
    docs = [load(path) for path in paths]
    sharded = docs[0]["summary"]["shard_count"]
    require(type(sharded) is int and sharded >= 1 and len(docs) == sharded,
            "MISSING_OR_EXTRA_SHARDS")
    expected = list(range(sharded))
    actual = [d["summary"]["shard_index"] for d in docs]
    require(sorted(actual) == expected, "SHARD_INDEX_COVERAGE")
    master = docs[0]["summary"]["source_master_sha256"]
    selection = docs[0]["summary"]["selection_sha256"]
    for d in docs:
        info = d["summary"]
        require(info["source_master_sha256"] == master and
                info["selection_sha256"] == selection and
                info["shard_count"] == sharded, "PARENT_SOURCE_DRIFT")
        require(info["qa_only_count"] == 15 and info["old_holdout_count"] == 169,
                "SPLIT_OR_QA_DRIFT")
    source_maps = {
        d["summary"]["shard_index"]: {r["canonical_asset_id"]:r for r in d["subjects"]}
        for d in docs
    }
    base = source_maps[0]
    require(len(base) == 4008, "DUPLICATE_ASSET_ID")
    output = []
    for aid in sorted(base):
        specimens = []
        for index in expected:
            rows = source_maps[index]
            require(set(rows) == set(base), "SHARD_ROSTER_DRIFT")
            specimens.append(rows[aid])
        expected_owner = int(hashlib.sha256(base[aid]["candidate_id"].encode("utf-8")).hexdigest(),16)%sharded
        for i,r in enumerate(specimens):
            require(r["candidate_id"] == base[aid]["candidate_id"] and
                    r["legacy_split"] == base[aid]["legacy_split"] and
                    r["source_provider"] == base[aid]["source_provider"] and
                    r["source_shard_owner"] == expected_owner, "PER_ASSET_PROVENANCE_DRIFT")
            require(r["training_authorized"] is False and
                    r["semantic_training_authorized"] is False and
                    r["graphics_training_authorized"] is False, "FORGED_TARGET_AUTHORITY")
            if i != expected_owner:
                require(r["source_probe_status"] == "OTHER_SHARD_NOT_RUN",
                        "NON_OWNER_FILE_PROBE")
        owner = specimens[expected_owner].copy()
        owner["owning_shard_index"]=expected_owner
        output.append(owner)
    counts=Counter(r["source_probe_status"] for r in output)
    report={
        "schema":MERGED,
        "status":"ALL_4008_ORIGINAL_SOURCE_CENSUS_MERGED__NOT_TRAINING",
        "source_master_sha256":master,
        "selection_sha256":selection,
        "shard_count":sharded,
        "source_census_count":4008,
        "statuses":dict(sorted(counts.items())),
        "qa_only_count":sum(r["qa_only"] for r in output),
        "historical_holdout_count":sum(r["holdout_locked"] for r in output),
        "character_classified_count":0,
        "anatomical_camera_courted_count":0,
        "semantic_truth_admitted_count":0,
        "training_authorized":False,
        "subjects":output,
    }
    require(report["qa_only_count"] == 15 and report["historical_holdout_count"] == 169,
            "POSTMERGE_SPLIT_DRIFT")
    return report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--inputs",type=Path,nargs="+",required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    report=merge(args.inputs)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k != "subjects"},sort_keys=True))


if __name__=="__main__":
    main()
