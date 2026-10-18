#!/usr/bin/env python3
from __future__ import annotations

"""Typed self-hosted V9 pre-training artifact builder.

This operator is deliberately read-only with respect to external storage. It may
hydrate pinned inputs and the quarantined snapshot, but it never writes to Drive
and never invokes a trainer or optimizer. Candidate bytes are produced locally
for GitHub Actions artifact transport; a separate authority step may persist and
round-trip them later.
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

REQUEST_SCHEMA = "RealSaS.IRISVNextV9PretrainingArtifactRequest.v1"
RESULT_SCHEMA = "RealSaS.IRISVNextV9PretrainingArtifactBuild.v1"
ALLOWED_BRANCH = "research/iris-vnext-post-v8-quality-plan-20261018"
QUARANTINED_REMOTE = "gdrive:RealSaS_Fit8_IRIS_vNext_20261016"
SHA40 = re.compile(r"^[0-9a-f]{40}$")


def run(args: list[str], *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, check=check, text=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("VNEXT_V9_OPERATOR_JSON_OBJECT_REQUIRED")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--authority-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    request = load(args.request.resolve())
    if request.get("schema") != REQUEST_SCHEMA:
        raise RuntimeError("VNEXT_V9_OPERATOR_REQUEST_SCHEMA_INVALID")
    if request.get("target_branch") != ALLOWED_BRANCH:
        raise RuntimeError("VNEXT_V9_OPERATOR_TARGET_BRANCH_INVALID")
    target_sha = str(request.get("target_source_sha") or "").lower()
    if not SHA40.fullmatch(target_sha):
        raise RuntimeError("VNEXT_V9_OPERATOR_TARGET_SHA_INVALID")
    if request.get("training_allowed") is not False:
        raise RuntimeError("VNEXT_V9_OPERATOR_TRAINING_MUST_BE_FALSE")
    if request.get("external_storage_write_allowed") is not False:
        raise RuntimeError("VNEXT_V9_OPERATOR_EXTERNAL_WRITE_MUST_BE_FALSE")

    repo = args.repo.resolve()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    audit_root = out / "quarantine_snapshot"
    for rel in ("teachers", "prepared", "receipts"):
        (audit_root / rel).mkdir(parents=True, exist_ok=True)

    # Same-repository exact-ref check; no arbitrary URL/ref execution.
    run(["git", "fetch", "--no-tags", "origin", f"refs/heads/{ALLOWED_BRANCH}:refs/remotes/origin/{ALLOWED_BRANCH}"], cwd=repo)
    actual = subprocess.check_output(["git", "rev-parse", f"refs/remotes/origin/{ALLOWED_BRANCH}"], cwd=repo, text=True).strip()
    if actual != target_sha:
        raise RuntimeError(f"VNEXT_V9_OPERATOR_SOURCE_DRIFT:{actual}:{target_sha}")

    # Read-only quarantine copy. Missing paths are part of the forensic evidence.
    run(["rclone", "copy", f"{QUARANTINED_REMOTE}/teachers", str(audit_root / "teachers"), "--checksum"], check=False)
    run(["rclone", "copy", f"{QUARANTINED_REMOTE}/prepared", str(audit_root / "prepared"), "--checksum"], check=False)
    run(["rclone", "copy", f"{QUARANTINED_REMOTE}/receipts/HYDRATION_RECEIPT.json", str(audit_root / "receipts"), "--checksum"], check=False)

    result = {
        "schema": RESULT_SCHEMA,
        "status": "AUDIT_INPUTS_HYDRATED__MATERIALIZATION_PENDING",
        "target_source_sha": target_sha,
        "target_branch": ALLOWED_BRANCH,
        "quarantined_remote": QUARANTINED_REMOTE,
        "external_storage_written": False,
        "training_steps_executed": 0,
        "scientific_fit_started": False,
    }
    (out / "OPERATOR_RESULT.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
