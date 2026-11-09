#!/usr/bin/env python3
"""V14: resumable, source-locked, sharded recovery of V13 normalized derivatives.

This tool never rehashes creator-native original assets or admits training truth.
Only 726 historical SHA-recovery *metadata* candidates are eligible.
Protected holdouts, path-only candidates and unknown sources are never fetched.
"""
from __future__ import annotations
import argparse
import collections
import csv
import gzip
import hashlib
import json
import pathlib
import re
import shutil

import numpy as np

EXPECTED_QUEUE = 1527
EXPECTED_HOLDOUTS = 135
EXPECTED_ELIGIBLE = 726
FILES = ("normalized.npz", "source_structure.json")
ASSET = re.compile(r"[a-z0-9_]+:[a-f0-9]{24,64}\Z")
SOURCE_RECOVERED = "SHA_VERIFIED_RECOVERY_RECEIPT"
SOURCE_HOLDOUT = "PROTECTED_HOLDOUT_ORIGINAL_NOT_TOUCHED"
OBSERVED = "NUMERIC_DERIVATIVE_OBSERVATION_ONLY"
MISSING = "MISSING_HISTORICAL_NORMALIZED_DERIVATIVE"
BAD = "QUARANTINED_BAD_HISTORICAL_NPZ"
PER_ASSET = "MASTER726_NORMALIZED_DERIVATIVE_PER_ASSET_V13.jsonl.gz"
JOINTS = "MASTER726_NORMALIZED_DERIVATIVE_JOINTS_V13.csv.gz"
SUMMARY = "SUMMARY_NORMALIZED_V13.json"
SHARD_REPORT = "INDEPENDENT_SHARD_V14.json"

def sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def read_json(path):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))

def write_json(path, value):
    pathlib.Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def all_rows(queue_path, baseline_inventory):
    with pathlib.Path(queue_path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = list(reader.fieldnames or [])
        queue = list(reader)
    assert len(queue) == EXPECTED_QUEUE
    assert len({row["source_key"] for row in queue}) == EXPECTED_QUEUE
    statuses = {}
    with gzip.open(baseline_inventory, "rt", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["pool"] == "MASTER3993":
                statuses[row["source_key"]] = row["master_v11_source_recovery_status"]
    assert all(row["source_key"] in statuses for row in queue)
    holdouts = [row for row in queue if statuses[row["source_key"]] == SOURCE_HOLDOUT]
    eligible = [row for row in queue if statuses[row["source_key"]] == SOURCE_RECOVERED]
    assert len(holdouts) == EXPECTED_HOLDOUTS
    assert len(eligible) == EXPECTED_ELIGIBLE
    for row in eligible:
        assert ASSET.fullmatch(row["candidate_id"]), "BAD_CANDIDATE_ID"
    return cols, queue, eligible, holdouts

def shard_of(source_key: str, shard_count: int) -> int:
    assert 1 <= shard_count <= 64
    return int(hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:16], 16) % shard_count

def assigned(eligible, shard_index: int, shard_count: int):
    assert 0 <= shard_index < shard_count
    return [row for row in eligible if shard_of(row["source_key"], shard_count) == shard_index]

def plan(args):
    root = pathlib.Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    cols, queue, eligible, holdouts = all_rows(args.queue, args.baseline_inventory)
    rows = assigned(eligible, args.shard_index, args.shard_count)
    assert rows
    with (root / "queue_shard.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    paths = []
    for row in rows:
        prefix = row["candidate_id"].replace(":", "_", 1)
        paths.extend(f"{prefix}/{name}" for name in FILES)
    (root / "paths.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
    receipt = {
        "schema": "RealSaS.TruthFoundryV14ShardPlan.v1",
        "shard_index": args.shard_index, "shard_count": args.shard_count,
        "cohort_attempts": EXPECTED_ELIGIBLE, "protected_holdouts_excluded": EXPECTED_HOLDOUTS,
        "assigned_source_count": len(rows),
        "source_key_sha256_sorted": hashlib.sha256("\n".join(sorted(r["source_key"] for r in rows)).encode()).hexdigest(),
        "queue_input_sha256": sha(pathlib.Path(args.queue)),
        "inventory_input_sha256": sha(pathlib.Path(args.baseline_inventory)),
        "subset_queue_sha256": sha(root / "queue_shard.csv"),
        "file_list_sha256": sha(root / "paths.txt"),
        "training_admissions": 0, "creator_original_bytes_rehashed": 0,
    }
    write_json(root / "SHARD_PLAN_V14.json", receipt)
    print(json.dumps({"shard_index": args.shard_index, "shard_count": args.shard_count,
                      "assigned_sources": len(rows), "historical_files_requested": len(paths),
                      "protected_holdouts": EXPECTED_HOLDOUTS}, sort_keys=True))

def verify(args):
    plan_dir = pathlib.Path(args.plan)
    variant = pathlib.Path(args.variant_dir)
    out = pathlib.Path(args.out)
    p = read_json(plan_dir / "SHARD_PLAN_V14.json")
    _, _, eligible, _ = all_rows(args.queue, args.baseline_inventory)
    assert p["queue_input_sha256"] == sha(pathlib.Path(args.queue))
    assert p["inventory_input_sha256"] == sha(pathlib.Path(args.baseline_inventory))
    assert p["subset_queue_sha256"] == sha(plan_dir / "queue_shard.csv")
    assert p["file_list_sha256"] == sha(plan_dir / "paths.txt")
    subset = assigned(eligible, p["shard_index"], p["shard_count"])
    assert len(subset) == p["assigned_source_count"]
    with (plan_dir / "queue_shard.csv").open(newline="", encoding="utf-8") as f:
        q = list(csv.DictReader(f))
    assert [r["source_key"] for r in q] == [r["source_key"] for r in subset]
    rows = []
    with gzip.open(out / PER_ASSET, "rt", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    assert len(rows) == len(subset)
    assert {r["source_key"] for r in rows} == {r["source_key"] for r in subset}
    joint_count = collections.Counter()
    with gzip.open(out / JOINTS, "rt", newline="", encoding="utf-8") as f:
        for j in csv.DictReader(f):
            assert j["source_key"] in {r["source_key"] for r in subset}
            assert int(j["parent_index"]) >= -1
            assert int(j["influenced_vertices_count"]) >= 0
            joint_count[j["source_key"]] += 1
    counts = collections.Counter()
    rehashed = 0
    expected = {r["source_key"]: r for r in subset}
    for record in rows:
        key, state = record["source_key"], record["court"]
        assert state in {OBSERVED, MISSING, BAD}
        counts[state] += 1
        if state != OBSERVED:
            assert joint_count[key] == 0
            continue
        assert record["candidate_id"] == expected[key]["candidate_id"]
        assert record["teacher_training_admission"] is False
        assert record["global_sdf_sign_admission"] is False
        assert record["2d_art_approval"] is False
        path = variant / record["candidate_id"].replace(":", "_", 1) / "normalized.npz"
        assert path.is_file() and sha(path) == record["normalized_npz_sha256"]
        with np.load(path, allow_pickle=False) as z:
            v, f, w, parents = (z[name] for name in ("vertices", "faces", "skin", "parents"))
            assert v.shape == (record["vertex_count"], 3)
            assert f.shape == (record["triangle_count"], 3)
            assert w.shape == (record["vertex_count"], record["joint_count"])
            assert len(parents) == record["joint_count"] == joint_count[key]
            assert (f >= 0).all() and (f < len(v)).all()
            assert int(np.count_nonzero(w.astype(np.float64).sum(axis=1) <= 1e-8)) == record["skin_zero_row_count"]
        rehashed += 1
    summary = read_json(out / SUMMARY)
    assert summary["candidate_attempts"] == len(subset)
    assert summary["results"] == dict(counts)
    assert summary["per_joint_rows"] == sum(joint_count.values())
    for key in ("original_source_byte_rehash_admission", "global_sdf_sign_admission",
                "mechanically_qualified_tessa_teacher_admission", "rights_admission",
                "training_authorized", "protected_holdout_folders_visited"):
        assert summary[key] == 0, "FALSE_AUTHORITY:" + key
    assert rehashed > 0, "NO_NUMERIC_DERIVATIVES_RETRIEVED"
    report = {
        "schema": "RealSaS.TruthFoundryV14IndependentShardCourt.v1",
        "status": "PASS_HISTORICAL_DERIVATIVE_ONLY",
        "plan": p, "results": dict(counts),
        "source_attempts": len(rows), "independently_rehashed_npz": rehashed,
        "per_joint_rows": sum(joint_count.values()),
        "payload_sha256": {name: sha(out / name) for name in (PER_ASSET, JOINTS, SUMMARY)},
        "source_original_bytes_rehashed": 0, "head_training_admissions": 0,
        "protected_holdout_folders_visited": 0,
    }
    write_json(out / SHARD_REPORT, report)
    print(json.dumps({"shard": p["shard_index"], "attempted": len(rows), "observed": rehashed,
                      "results": dict(counts), "status": report["status"]}, sort_keys=True))

def merge(args):
    root = pathlib.Path(args.shards)
    dest = pathlib.Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    _, _, eligible, _ = all_rows(args.queue, args.baseline_inventory)
    all_expected = {r["source_key"] for r in eligible}
    all_assets = {}
    joint_rows = []
    combined = collections.Counter()
    bound_queue = sha(pathlib.Path(args.queue))
    bound_inventory = sha(pathlib.Path(args.baseline_inventory))
    digest_rows = {}
    observed_total = 0
    for shard_index in range(args.shard_count):
        d = root / f"shard-{shard_index:02d}"
        report = read_json(d / SHARD_REPORT)
        assert report["status"] == "PASS_HISTORICAL_DERIVATIVE_ONLY"
        p = report["plan"]
        assert p["shard_index"] == shard_index and p["shard_count"] == args.shard_count
        assert p["queue_input_sha256"] == bound_queue
        assert p["inventory_input_sha256"] == bound_inventory
        for name, hashed in report["payload_sha256"].items():
            assert sha(d / name) == hashed, "SHARD_PAYLOAD_DRIFT:" + name
        expected_shard = {r["source_key"] for r in assigned(eligible, shard_index, args.shard_count)}
        assets = []
        with gzip.open(d / PER_ASSET, "rt", encoding="utf-8") as f:
            assets = [json.loads(line) for line in f]
        assert len(assets) == len(expected_shard)
        assert {r["source_key"] for r in assets} == expected_shard
        counts = collections.Counter(r["court"] for r in assets)
        assert dict(counts) == report["results"]
        assert counts[OBSERVED] == report["independently_rehashed_npz"]
        observed_total += counts[OBSERVED]
        combined.update(counts)
        for r in assets:
            assert r["source_key"] not in all_assets
            all_assets[r["source_key"]] = r
        with gzip.open(d / JOINTS, "rt", newline="", encoding="utf-8") as f:
            j = list(csv.DictReader(f))
        assert len(j) == report["per_joint_rows"]
        assert all(r["source_key"] in expected_shard for r in j)
        joint_rows.extend(j)
        digest_rows[f"shard-{shard_index:02d}"] = sha(d / SHARD_REPORT)
    assert set(all_assets) == all_expected, "MISSING_OR_DUPLICATE_RECOVERY_COHORT"
    assert len(all_assets) == EXPECTED_ELIGIBLE
    order = [r["source_key"] for r in eligible]
    with gzip.open(dest / PER_ASSET, "wt", encoding="utf-8") as f:
        for key in order:
            f.write(json.dumps(all_assets[key], sort_keys=True, separators=(",", ":")) + "\n")
    fieldnames = ["source_key", "joint_index", "parent_index", "deform_mask",
                  "influenced_vertices_count", "summed_skin_weight_mass", "head_tail_length"]
    with gzip.open(dest / JOINTS, "wt", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(sorted(joint_rows, key=lambda r:(order.index(r["source_key"]),int(r["joint_index"]))))
    summary = {
        "schema": "RealSaS.HistoricalNormalizedDerivativeV14Reconciled",
        "status": "PASS_ALL_SHARDS_INDEPENDENTLY_REHASHED_DERIVATIVE_ONLY",
        "cohort_source_attempts": EXPECTED_ELIGIBLE,
        "observed_normalized_npz": observed_total,
        "results": dict(combined), "per_joint_rows": len(joint_rows),
        "shard_count": args.shard_count, "source_original_bytes_rehashed": 0,
        "training_admissions": 0, "rights_admissions": 0,
        "global_sdf_sign_admissions": 0, "tessa_animation_ready_admissions": 0,
        "protected_holdout_folders_visited": 0,
        "parent_shard_report_sha256": digest_rows,
        "sha256": {name:sha(dest / name) for name in (PER_ASSET, JOINTS)},
    }
    write_json(dest / "SUMMARY_RECONCILED_V14.json", summary)
    print(json.dumps({"status":summary["status"],"observed":observed_total,"attempts":len(all_assets),"results":dict(combined)},sort_keys=True))

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="action", required=True)
    a = sub.add_parser("plan")
    a.add_argument("--out", required=True)
    a.add_argument("--queue", required=True)
    a.add_argument("--baseline-inventory", required=True)
    a.add_argument("--shard-index", required=True, type=int)
    a.add_argument("--shard-count", required=True, type=int)
    b = sub.add_parser("verify")
    b.add_argument("--plan", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--variant-dir", required=True)
    b.add_argument("--queue", required=True)
    b.add_argument("--baseline-inventory", required=True)
    c = sub.add_parser("merge")
    c.add_argument("--shards", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--queue", required=True)
    c.add_argument("--baseline-inventory", required=True)
    c.add_argument("--shard-count", required=True, type=int)
    args = p.parse_args()
    {"plan":plan,"verify":verify,"merge":merge}[args.action](args)

if __name__ == "__main__":
    main()
