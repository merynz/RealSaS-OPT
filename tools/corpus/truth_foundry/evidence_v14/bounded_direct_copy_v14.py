#!/usr/bin/env python3
"""Direct-path, bounded, source-locked V14 historical derivative transfer.

Avoids bulk --files-from directory traversal. Copies only allowlisted V14 plan
paths via one-file rclone copyto, with bounded concurrency, per-file timeouts,
safe error-class counters and source-bound SHA receipts. No creator originals.
"""
from __future__ import annotations
import argparse
import collections
import concurrent.futures
import hashlib
import json
import pathlib
import re
import subprocess

from sharded_normalized_recovery_v14 import read_json, sha

ALLOWED = re.compile(r"[a-z0-9_]+_[0-9a-f]{24,64}/(?:normalized\.npz|source_structure\.json)\Z")
ERRORS = {
    # Google Drive rate-limit errors often use HTTP 403, not 429.
    "RATE_OR_QUOTA": r"\b429\b|userRateLimitExceeded|rateLimitExceeded|dailyLimitExceeded|downloadQuotaExceeded|storageQuotaExceeded|quota|rate.limit|too.many.requests",
    "NOT_FOUND": r"\b404\b|not found|directory not found|does not exist",
    "EXPLICIT_ACCESS_DENIED": r"insufficientFilePermissions|access.denied|permission.denied|not.authorized|invalid.grant|unauthoriz",
    "HTTP_401": r"\b401\b",
    "HTTP_403_UNSPECIFIED": r"\b403\b|forbidden",
    "NETWORK": r"connection.reset|i/o timeout|tls|network|connection refused",
    "CLI": r"unknown flag|unknown command|invalid argument|flag provided but not defined",
}
MAX_NPZ_BYTES = 400 * 1024 * 1024


def classify(stderr: bytes) -> str:
    text = stderr.decode("utf-8", "replace")[-20000:]
    for category, rx in ERRORS.items():
        if re.search(rx, text, flags=re.I):
            return category
    return "UNCLASSIFIED"


def fetch(item: str, remote: str, root: pathlib.Path, seconds: int, attempts: int) -> dict:
    if not ALLOWED.fullmatch(item):
        raise ValueError("V14_FORBIDDEN_PATH")
    dst = root / item
    if not dst.resolve().is_relative_to(root.resolve()):
        raise ValueError("V14_PATH_ESCAPE")
    dst.parent.mkdir(parents=True, exist_ok=True)
    kind = "NPZ" if item.endswith("/normalized.npz") else "STRUCTURE"
    result = {"kind": kind, "status": "UNKNOWN", "bytes": 0,
              "sha256": None, "path_id_sha256": hashlib.sha256(item.encode()).hexdigest()}
    if dst.exists() and dst.is_file() and dst.stat().st_size:
        result.update(status="CACHED_REHASHED", bytes=dst.stat().st_size, sha256=sha(dst))
        return result
    for attempt in range(1, attempts + 1):
        try:
            p = subprocess.run(
                ["rclone", "copyto", remote.rstrip("/") + "/" + item, str(dst),
                 "--retries", "1", "--low-level-retries", "1",
                 "--contimeout", "12s", "--timeout", "30s"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            result["status"] = "TIMEOUT"
        else:
            if p.returncode == 0 and dst.is_file() and dst.stat().st_size > 0:
                size = dst.stat().st_size
                if kind == "NPZ" and size > MAX_NPZ_BYTES:
                    dst.unlink()
                    result["status"] = "NPZ_RESOURCE_LIMIT"
                    break
                result.update(status="COPIED", bytes=size, sha256=sha(dst))
                break
            result["status"] = classify(p.stderr)
        if dst.is_file():
            dst.unlink()
        if result["status"] in ("NOT_FOUND", "EXPLICIT_ACCESS_DENIED", "HTTP_401", "CLI", "NPZ_RESOURCE_LIMIT"):
            break
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True)
    p.add_argument("--remote-base", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--variant-dir", required=True)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--timeout-seconds", type=int, default=110)
    p.add_argument("--attempts", type=int, default=2)
    p.add_argument("--pilot-files", type=int, default=0)
    a = p.parse_args()
    assert 1 <= a.workers <= 4
    assert 10 <= a.timeout_seconds <= 150
    assert 1 <= a.attempts <= 2
    assert a.pilot_files == 0 or 1 <= a.pilot_files <= 16
    plan = pathlib.Path(a.plan)
    receipt = read_json(plan / "SHARD_PLAN_V14.json")
    items = (plan / "paths.txt").read_text(encoding="utf-8").splitlines()
    assert sha(plan / "paths.txt") == receipt["file_list_sha256"]
    assert len(items) == 2 * receipt["assigned_source_count"]
    assert len(set(items)) == len(items)
    assert all(ALLOWED.fullmatch(item) for item in items)
    if a.pilot_files:
        # Round-robin source pairs; a transfer-only probe, NOT a V13 numeric court.
        items = items[:a.pilot_files]
    dest = pathlib.Path(a.out)
    dest.mkdir(parents=True, exist_ok=True)
    root = pathlib.Path(a.variant_dir)
    root.mkdir(parents=True, exist_ok=True)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:
        futures = [ex.submit(fetch, item, a.remote_base, root,
                             a.timeout_seconds, a.attempts) for item in items]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            results.append(result)
            if index % 10 == 0 or index == len(items):
                c = collections.Counter(r["status"] for r in results)
                print(json.dumps({"completed": index, "total": len(items),
                                  "states": dict(c)}, sort_keys=True), flush=True)
    summary = {
        "schema": "RealSaS.TruthFoundryV14DirectCopyto.v1",
        "mode": "PILOT_TRANSPORT_ONLY" if a.pilot_files else "FULL_SHARD_TRANSFER",
        "source_plan_sha256": sha(plan / "SHARD_PLAN_V14.json"),
        "plan_filelist_sha256": receipt["file_list_sha256"],
        "shard": receipt["shard_index"],
        "assigned_source_count": receipt["assigned_source_count"],
        "attempted_files": len(items),
        "actual_npz_copied": sum(x["kind"] == "NPZ" and x["status"] in ("COPIED", "CACHED_REHASHED") for x in results),
        "states": dict(collections.Counter(x["status"] for x in results)),
        "records": sorted(results, key=lambda x:x["path_id_sha256"]),
        "original_creator_source_rehash": 0,
        "training_admissions": 0,
    }
    (dest / "DIRECT_COPY_TRANSPORT_V14.json").write_text(json.dumps(summary,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"shard":summary["shard"],"mode":summary["mode"],
                      "npz":summary["actual_npz_copied"],"states":summary["states"]},sort_keys=True))
    return 0 if summary["actual_npz_copied"] > 0 else 4


if __name__ == "__main__":
    raise SystemExit(main())
