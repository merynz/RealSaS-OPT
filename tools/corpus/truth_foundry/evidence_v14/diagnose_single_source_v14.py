#!/usr/bin/env python3
"""Bounded, read-only Drive path diagnosis for historical normalized derivatives.

Only three source-locked recovered candidates (at most one/provider). Never
traverse or download original creator assets; never print remote/private paths.
Outputs transport diagnostics, NOT geometry, rights or training admission.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import pathlib
import re
import subprocess
import tempfile

from sharded_normalized_recovery_v14 import all_rows, sha

ERROR_MARKERS = {
    "ACCESS_DENIED_401_403": r"(?:\b401\b|\b403\b|permission|access.denied|insufficient.permi|forbidden|unauthoriz)",
    "RATE_OR_QUOTA_429": r"(?:\b429\b|ratelimit|rate.limit|quota|too.many.requests|userRateLimitExceeded)",
    "SOURCE_NOT_FOUND_404": r"(?:\b404\b|not.found|directory.not.found|file.not.found|does.not.exist)",
    "AUTH_EXPIRED": r"(?:invalid.grant|token.expir|oauth|invalid.credentials)",
    "NETWORK_OR_TLS": r"(?:i/o timeout|connection.reset|tls|transport|connection.refused|network.is.unreachable)",
    "UNSUPPORTED_CLI_FLAG": r"(?:unknown flag|unknown shorthand flag|flag provided but not defined|unknown command|unsupported flag|invalid argument)",
}
MAX_BYTES = 450 * 1024 * 1024


def run_command(argv, timeout_seconds: int) -> dict:
    try:
        proc = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=timeout_seconds, check=False)
        logs = (proc.stdout + proc.stderr).decode("utf-8", "replace")[:65536]
        flags = sorted(name for name, pattern in ERROR_MARKERS.items()
                       if re.search(pattern, logs, re.IGNORECASE))
        return {"exit_code": proc.returncode, "transport": "EXIT",
                "error_flags": flags or (["OTHER_ERROR"] if proc.returncode else []),
                "stdout_bytes": len(proc.stdout), "stderr_bytes": len(proc.stderr)}
    except subprocess.TimeoutExpired:
        return {"exit_code": None, "transport": "TIMEOUT",
                "error_flags": ["TIMEOUT"], "stdout_bytes": 0, "stderr_bytes": 0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--queue", required=True)
    p.add_argument("--baseline-inventory", required=True)
    p.add_argument("--remote-base", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    _, _, eligible, holdouts = all_rows(a.queue, a.baseline_inventory)
    provider_pick = {}
    for row in eligible:
        provider = row["candidate_id"].split(":", 1)[0]
        provider_pick.setdefault(provider, row)
    picks = [provider_pick[k] for k in sorted(provider_pick)[:3]]
    assert picks and len(holdouts) == 135
    summaries = []
    with tempfile.TemporaryDirectory() as temp:
        root = pathlib.Path(temp)
        for index, row in enumerate(picks):
            folder = row["candidate_id"].replace(":", "_", 1)
            target = root / f"probe-{index}.npz"
            remote_dir = a.remote_base.rstrip("/") + "/" + folder
            status = run_command(
                ["rclone", "lsf", remote_dir, "--files-only", "--max-depth", "1",
                 ],
                timeout_seconds=70,
            )
            download = {"transport": "SKIPPED", "error_flags": ["DIRECTORY_NOT_LISTED"]}
            copied = False
            content_sha256 = None
            nbytes = 0
            if status["transport"] == "EXIT" and status["exit_code"] == 0:
                download = run_command(
                    ["rclone", "copyto", remote_dir + "/normalized.npz", str(target),
                     "--retries", "1", "--low-level-retries", "1",
                     "--contimeout", "15s", "--timeout", "40s"],
                    timeout_seconds=200,
                )
                if download["transport"] == "EXIT" and download["exit_code"] == 0 and target.is_file():
                    nbytes = target.stat().st_size
                    if 0 < nbytes <= MAX_BYTES:
                        content_sha256 = sha(target)
                        copied = True
            summaries.append({
                "probe_index": index,
                "provider_hash_prefix": hashlib.sha256(row["candidate_id"].split(":", 1)[0].encode()).hexdigest()[:10],
                "source_key_hash_prefix": hashlib.sha256(row["source_key"].encode()).hexdigest()[:10],
                "folder_list": status,
                "file_copy": download,
                "copied": copied, "copied_bytes": nbytes,
                "derived_npz_sha256": content_sha256,
                "authority": "HISTORICAL_DERIVATIVE_TRANSPORT_ONLY__NO_ORIGINAL_SOURCE_OR_TRAINING_ADMISSION",
            })
            print(json.dumps({"probe": index, "listed": status["exit_code"] == 0, "list_exit_code": status["exit_code"], "list_flags": status["error_flags"],
                              "copy": download["transport"], "flags": download["error_flags"],
                              "copied": copied, "copied_bytes": nbytes}, sort_keys=True))
    summary = {
        "schema": "RealSaS.TruthFoundryV14SingleSourceReadProbe.v1",
        "cohort": 726, "holdouts_excluded": 135,
        "queue_sha256": sha(pathlib.Path(a.queue)),
        "inventory_sha256": sha(pathlib.Path(a.baseline_inventory)),
        "probes": summaries, "real_source_truth_admitted": 0,
    }
    (out / "TRANSPORT_DIAGNOSTICS_V14.json").write_text(json.dumps(summary, sort_keys=True, indent=2)+"\n")
    print("TRANSPORT_DIAGNOSTIC_COMPLETE source-original=0 training-admissions=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
