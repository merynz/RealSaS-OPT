#!/usr/bin/env python3
from __future__ import annotations

"""Typed, fail-closed bridge from canonical main to exact research-source operations.

This is intentionally *not* a generic command runner. Operation names and entrypoints
are hard-coded here. V1 permits only pre-training materialization and explicitly
forbids any optimizer/training capability.
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

SCHEMA = "RealSaS.ResearchOperationRequest.v1"
RECEIPT_SCHEMA = "RealSaS.ResearchOperationReceipt.v1"
OPERATION = "IRIS_VNEXT_V9_PRETRAINING_MATERIALIZE"
TARGET_BRANCH = "research/iris-vnext-post-v8-quality-plan-20261018"
ENTRYPOINT = "tools/iris_vnext/v9_pretraining_materialize_v1.py"
SHA40 = re.compile(r"^[0-9a-f]{40}$")


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("RESEARCH_OPERATOR_REQUEST_OBJECT_REQUIRED")
    return value


def run(*args: str, cwd: Path | None = None, capture: bool = False) -> str:
    result = subprocess.run(
        list(args), cwd=cwd, check=True, text=True,
        capture_output=capture,
    )
    return result.stdout.strip() if capture else ""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def execute(request_path: Path, repo: Path, authority_root: Path, output: Path) -> dict:
    request = load_json(request_path)
    if request.get("schema") != SCHEMA:
        raise RuntimeError("RESEARCH_OPERATOR_SCHEMA_INVALID")
    if request.get("operation") != OPERATION:
        raise RuntimeError("RESEARCH_OPERATOR_OPERATION_NOT_ALLOWLISTED")
    if request.get("target_branch") != TARGET_BRANCH:
        raise RuntimeError("RESEARCH_OPERATOR_TARGET_BRANCH_INVALID")
    target_sha = str(request.get("target_source_sha") or "").lower()
    if not SHA40.fullmatch(target_sha):
        raise RuntimeError("RESEARCH_OPERATOR_TARGET_SHA_INVALID")
    capabilities = request.get("capabilities") or {}
    if capabilities.get("training_allowed") is not False:
        raise RuntimeError("RESEARCH_OPERATOR_TRAINING_MUST_BE_FALSE")
    if capabilities.get("quarantine_read_only") is not True:
        raise RuntimeError("RESEARCH_OPERATOR_QUARANTINE_READ_ONLY_REQUIRED")
    if capabilities.get("candidate_sha_scoped_write") is not True:
        raise RuntimeError("RESEARCH_OPERATOR_SHA_SCOPED_WRITE_REQUIRED")

    repo = repo.resolve()
    authority_root = authority_root.resolve()
    request_sha = sha256(request_path)
    run_root = authority_root / "research_operations" / request_sha
    worktree = run_root / "source"
    result_path = run_root / "V9_PRETRAINING_OPERATION_RESULT.json"
    run_root.mkdir(parents=True, exist_ok=True)

    # Resolve the exact same-repository research branch. No arbitrary ref/URL input.
    run("git", "fetch", "--no-tags", "origin", f"refs/heads/{TARGET_BRANCH}:refs/remotes/origin/{TARGET_BRANCH}", cwd=repo)
    remote_head = run("git", "rev-parse", f"refs/remotes/origin/{TARGET_BRANCH}", cwd=repo, capture=True)
    if remote_head != target_sha:
        raise RuntimeError(f"RESEARCH_OPERATOR_BRANCH_SHA_DRIFT:{remote_head}:{target_sha}")

    if worktree.exists():
        shutil.rmtree(worktree)
    run("git", "worktree", "add", "--detach", str(worktree), target_sha, cwd=repo)
    try:
        entrypoint = worktree / ENTRYPOINT
        if not entrypoint.is_file():
            raise RuntimeError("RESEARCH_OPERATOR_ENTRYPOINT_MISSING")
        run(
            "python", str(entrypoint),
            "--expected-source-sha", target_sha,
            "--authority-root", str(authority_root),
            "--output", str(result_path),
            cwd=worktree,
        )
    finally:
        run("git", "worktree", "remove", "--force", str(worktree), cwd=repo)

    result = load_json(result_path)
    if result.get("training_steps_executed") != 0 or result.get("scientific_fit_started") is not False:
        raise RuntimeError("RESEARCH_OPERATOR_RESULT_CROSSED_TRAINING_BOUNDARY")
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "PASS",
        "operation": OPERATION,
        "request_sha256": request_sha,
        "target_branch": TARGET_BRANCH,
        "target_source_sha": target_sha,
        "result_path": str(result_path),
        "result_sha256": sha256(result_path),
        "training_allowed": False,
        "training_steps_executed": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--authority-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    execute(args.request.resolve(), args.repo.resolve(), args.authority_root.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
