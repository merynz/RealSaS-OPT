"""Whole-universe source-first Master Corpus worklist (no admission/training).

Run after mounting verified Master Corpus V3 raw source tree. NO Drive token
in this script. Records every one of 4008 historical identities and the
actual status of source files; do not equate legacy ADMIT with model readiness.

Resume safely via content-addressed source receipts; exact SHA must match.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from tools.corpus.probe_corpus4008_source_scene_v1 import build, sha256

SOURCE_EXTS = {".glb", ".obj", ".fbx", ".blend", ".gltf"}
SUPPORTED = {".glb", ".obj", ".fbx", ".blend"}
SCHEMA = "RealSaS.Corpus4008SourcePreflightCensus.v1"


def require(condition, reason):
    if not condition:
        raise ValueError("CORPUS4008_CENSUS_" + reason)


def load_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def index_sources(root):
    require(root.is_dir(), "SOURCE_ROOT_NOT_A_DIRECTORY")
    buckets = defaultdict(list)
    # Use source-provenance provider as part of the lookup key, not loose 24
    # character substrings across every folder on Drive.
    for provider in root.iterdir():
        if not provider.is_dir():
            continue
        for path in provider.rglob("*"):
            if path.is_symlink() or not path.is_file() or path.suffix.lower() not in SOURCE_EXTS:
                continue
            token = path.name.split("_",1)[0]
            if re.fullmatch(r"[0-9a-f]{24}|[0-9a-f]{64}",token):
                buckets[(provider.name,token)].append(path)
    return buckets


def build_census(*, master_path, selection_path, sources_root, output_dir, limit=0):
    rows = load_jsonl(master_path)
    selected = json.loads(selection_path.read_text())
    require(len(rows) == 4008, "MASTER_EXPECTS_4008")
    require(len(selected) == 3993, "MASTER_SELECTED_EXPECTS_3993")
    require(len({r["canonical_asset_id"] for r in rows}) == 4008, "DUPLICATE_ASSET_ID")
    require(len({r["candidate_id"] for r in rows}) == 4008, "DUPLICATE_SOURCE_CANDIDATE")
    require(len(set(selected)) == 3993, "SELECTED_DUPLICATE")
    sources = index_sources(sources_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    receipts_dir = output_dir / "source_receipts"
    receipts_dir.mkdir(exist_ok=True)
    report, scanned = [], 0
    for row in sorted(rows,key=lambda x:x["canonical_asset_id"]):
        candidate = row["candidate_id"]
        require(candidate.count(":")==1, "CANDIDATE_ID_FORMAT")
        provider, token = candidate.split(":")
        require(provider == row["source_registry_id"], "PROVIDER_DRIFT")
        require(re.fullmatch(r"[0-9a-f]{24}|[0-9a-f]{64}",token) is not None, "TOKEN_FORMAT")
        is_qa = provider == "khronos_gltf_qa"
        require((row["canonical_asset_id"] in selected) != is_qa, "LEGACY_SELECTED_QA_LEAKAGE")
        paths = sources.get((provider,token), [])
        record = {
            "canonical_asset_id":row["canonical_asset_id"],
            "candidate_id":candidate,
            "source_provider":provider,
            "source_token":token,
            "legacy_decision":row["decision"],
            "legacy_split":row["split"],
            "holdout_locked":row["split"]=="EXTERNAL_HOLDOUT",
            "qa_only":is_qa,
            "source_file_count":len(paths),
            "source_file_candidates":sorted(str(p.relative_to(sources_root)) for p in paths),
            "source_probe_status":"UNKNOWN",
            "source_bytes_verified":False,
            "semantic_training_authorized":False,
            "graphics_training_authorized":False,
            "training_authorized":False,
        }
        if not paths:
            record["source_probe_status"]="SOURCE_RAW_NOT_FOUND"
        elif len(paths)>1:
            record["source_probe_status"]="AMBIGUOUS_SOURCE_VARIANTS__REVIEW"
        elif limit and scanned >= limit:
            record["source_probe_status"]="SOURCE_PRESENT__PENDING_PROBE_BATCH_LIMIT"
        else:
            source = paths[0]
            scanned+=1
            if source.suffix.lower() not in SUPPORTED:
                record["source_probe_status"]="PINNED_BLENDER_GLTF_EXTERNAL_RESOURCE_PROBE_REQUIRED"
                record["source_bytes_verified"] = sha256(source).startswith(token)
            else:
                try:
                    receipt = build(source,source_format=source.suffix.lower().lstrip("."),
                                    expected_sha256=sha256(source),
                                    canonical_asset_id=row["canonical_asset_id"],candidate_id=candidate)
                    # build() also requires token->full SHA identity (24 or 64).
                    path = receipts_dir/(row["canonical_asset_id"] + ".json")
                    path.write_text(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+"\n")
                    record["source_probe_status"]=receipt["source_scene_status"]
                    record["source_bytes_verified"]=receipt["source_bytes_verified"]
                    record["source_receipt_path"]=str(path.relative_to(output_dir))
                except (ValueError,UnicodeError,KeyError,IndexError) as exc:
                    record["source_probe_status"]="SOURCE_PROBE_FAILED__QUARANTINE"
                    record["source_failure_code"]=str(exc)[:250]
        report.append(record)
    summary={
      "schema":SCHEMA,
      "source_master_sha256":sha256(master_path),
      "selection_sha256":sha256(selection_path),
      "universe_count":4008,
      "census_entries":len(report),
      "raw_source_probe_attempt_count":scanned,
      "statuses":dict(sorted(Counter(x["source_probe_status"] for x in report).items())),
      "old_holdout_count":sum(x["holdout_locked"] for x in report),
      "qa_only_count":sum(x["qa_only"] for x in report),
      "character_reviewed_count":0,
      "source_camera_qualified_count":0,
      "source_semantic_courted_count":0,
      "training_authorized":False,
      "status":"MASTER_CORPUS_SOURCE_PREFLIGHT_ONLY__NOT_TRUTH_CLOSURE",
    }
    (output_dir/"CORPUS4008_SOURCE_CENSUS_V1.json").write_text(json.dumps(
        {"summary":summary,"subjects":report},indent=2,sort_keys=True,allow_nan=False)+"\n")
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--master",type=Path,required=True)
    parser.add_argument("--selection",type=Path,required=True)
    parser.add_argument("--sources-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--limit",type=int,default=0)
    args=parser.parse_args()
    require(args.limit>=0, "NEGATIVE_BATCH_LIMIT")
    result=build_census(master_path=args.master,selection_path=args.selection,
                        sources_root=args.sources_root,output_dir=args.output,limit=args.limit)
    print(json.dumps(result,sort_keys=True))


if __name__=="__main__":
    main()
