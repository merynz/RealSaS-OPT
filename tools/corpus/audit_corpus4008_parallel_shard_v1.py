#!/usr/bin/env python3
"""Readback one source-shard in isolation; combine Wave1/2 + disjoint Wave3.

Inputs are private GitHub Actions receipts, NOT source authoring truth.
Every successful import is already hash-checked by per-wave reconcile().
The same source cannot enter both waves, be a holdout, gain a new head
supervision PASS, or have a missing per-asset detailed receipt silently ignored.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from tools.corpus.reconcile_corpus4008_wave1_source_rig_matrix_v1 import reconcile

SCHEMA = "RealSaS.Corpus4008ParallelShardReadback.v1"
FIRST_WAVE_SHARDS = frozenset(range(0, 32, 4))
SHA = "0123456789abcdef"


def require(test, why):
    if not test:
        raise ValueError("C4008_PARALLEL_SHARD_" + why)


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as fd:
        for chunk in iter(lambda: fd.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def audit(shard: int, prior_root: Path, next_root: Path, output: Path):
    require(type(shard) is int and 0 <= shard < 32, "INVALID_INDEX")
    prior_wave = 1 if shard in FIRST_WAVE_SHARDS else 2
    output.mkdir(parents=True, exist_ok=True)
    prior_summary = reconcile(
        prior_root, output / "first_revalidated", expected_shards=(shard,))
    next_summary = reconcile(
        next_root, output / "next_revalidated", expected_shards=(shard,))
    prior_base = prior_root / f"shard-{shard}"
    next_base = next_root / f"shard-{shard}"
    a = json.loads((prior_base / "PLAN.json").read_text())
    b = json.loads((next_base / "PLAN.json").read_text())
    require(a["shard_index"] == shard
            and a["sample_size"] == len(a["entries"]) == 8,
            "FIRST_WAVE_EXACT_EIGHT")
    require(b["shard_index"] == shard
            and b["sample_method"] ==
            "DISJOINT_AFTER_FIRST8_PER_SHARD__BALANCED_FORMAT_PLUS_SHA_QUANTILES",
            "SECOND_WAVE_SELECTION_METHOD")
    original = {x["source_sha256"] for x in a["entries"]}
    reserved = set(b["prior_wave_reserved_source_sha256"])
    later = {x["source_sha256"] for x in b["entries"]}
    require(reserved == original
            and b["prior_wave_reserved_count"] == len(original) == 8,
            "SECOND_WAVE_RESERVATION_MISMATCH")
    require(not original.intersection(later), "SAME_ORIGINAL_IMPORTED_TWICE")
    require(len(later) == len(b["entries"]) == b["sample_size"]
            and 1 <= len(later) <= 24,
            "SECOND_WAVE_SAMPLE_SET")
    prior_csv = (
        output / "first_revalidated"
        / "WAVE1_SOURCE_RIG_SCENE_TYPED_MATRIX_V1.csv")
    next_csv = (
        output / "next_revalidated"
        / "WAVE1_SOURCE_RIG_SCENE_TYPED_MATRIX_V1.csv")
    prior_rows = read_csv(prior_csv)
    next_rows = read_csv(next_csv)
    require(len(prior_rows) == 8
            and len(next_rows) == len(later), "RECEIPT_ROW_COUNT")
    rows = []
    ids = set()
    source_ids = set()
    status = Counter()
    for wave, part in ((prior_wave, prior_rows), (3, next_rows)):
        for row in part:
            aid = row["canonical_asset_id"]
            original_sha = row["source_sha256"]
            require(aid not in ids and original_sha not in source_ids,
                    "DUPLICATE_ASSET_OR_ORIGINAL")
            ids.add(aid)
            source_ids.add(original_sha)
            require(row["source_shard"] == str(shard)
                    and row["split"] == "FIT"
                    and row["training_authorized"] == "False"
                    and row["head_admission_count"] == "0"
                    and row["source_character_vs_prop"] == "UNKNOWN"
                    and row["functional_native_ik_fk"] == "UNKNOWN"
                    and row["graphics_source_bound_2d_art_qualified"] == "UNKNOWN",
                    "IMPORT_OBSERVATION_CANNOT_PROMOTE_COURT")
            row["import_wave"] = wave
            status[row["source_import_observation_status"]] += 1
            rows.append(row)
    require(source_ids == original.union(later),
            "ORIGINAL_SOURCE_IDENTITY_DRIFT")
    dest = output / "SHARD_TYPED_OBSERVATIONS.csv"
    with dest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r:r["canonical_asset_id"]))
    report = {
        "schema": SCHEMA,
        "shard": shard,
        "prior_wave": prior_wave,
        "prior_count": len(prior_rows),
        "additional_wave_count": len(next_rows),
        "total_unique_source_count": len(rows),
        "original_sha256_list": sorted(source_ids),
        "asset_id_list": sorted(ids),
        "source_import_status_counts": dict(sorted(status.items())),
        "prior_plan_sha256": file_hash(prior_base / "PLAN.json"),
        "additional_plan_sha256": file_hash(next_base / "PLAN.json"),
        "prior_readback_summary_sha256": file_hash(
            output / "first_revalidated" / "WAVE1_SOURCE_RIG_SCENE_SUMMARY_V1.json"),
        "additional_readback_summary_sha256": file_hash(
            output / "next_revalidated" / "WAVE1_SOURCE_RIG_SCENE_SUMMARY_V1.json"),
        "typed_observations_csv_sha256": file_hash(dest),
        "source_original_bytes_refetched": False,
        "full_import_receipts_sha_verified": True,
        "source_rights_qualified": False,
        "native_IK_FK_functionality_qualified": False,
        "graphics_2d_art_qualified": False,
        "per_head_training_admitted_count": 0,
        "training_authorized": False,
    }
    (output / "SHARD_READBACK_SUMMARY.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--prior", type=Path, required=True)
    ap.add_argument("--next", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    result = audit(args.shard, args.prior, args.next, args.output)
    print(json.dumps({
        "shard": result["shard"],
        "source_count": result["total_unique_source_count"],
        "status": result["source_import_status_counts"],
        "training_authorized": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
