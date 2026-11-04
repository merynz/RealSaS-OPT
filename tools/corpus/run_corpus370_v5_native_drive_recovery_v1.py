#!/usr/bin/env python3
"""Crash-safe transport wrapper for unchanged source-locked V5 native truth.

Important: the source-producing worker and Blender export scripts remain byte-
identical to their original run. This *separate* supervisor resumes the original
fixed local run directory rather than starting a new directory per GitHub SHA.

Remote ordering is subject files FIRST, then an atomic local checkpoint index
uploaded LAST, so no remote receipt points at data not yet checkpointed.
Every local resumed teacher is independently re-courted by the original worker.
No authority is granted and no failed authored actions are silently dropped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

INDEX = "CORPUS370_V5_ALL_AUTHORED_TEMPORAL_TEACHER_V1.json"
SCHEMA = "RealSaS.Corpus370V5AllAuthoredTemporalTeacherSweep.v1"
REMOTE_PREFIX = "gdrive:corpus370_truth_v5_native_temporal_v4/"
PASS_STATES = {
    "PASS_SOURCE_AUTHORED_TEMPORAL_GEOMETRY_CANDIDATE__CAUSAL_RESPONSE_SEPARATE",
    "MASKED_STATIC_AUTHORED_CLIP",
}
FAIL_STATE = "FAIL_SOURCE_AUTHORED_TEMPORAL_EXPORT"


def require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def sha(path: Path) -> str:
    with path.open("rb") as reader:
        return hashlib.file_digest(reader, "sha256").hexdigest()


def checked_result_files(root: Path, state: dict) -> list[str]:
    """Validate completed index and refs without importing the producer itself."""
    require(state.get("schema") == SCHEMA
            and state.get("training_authorized") is False
            and state.get("causal_response_authority_minted") is False
            and state.get("tessa_teacher_authority_minted") is False,
            "NATIVE_RECOVERY_NONAUTHORITATIVE_SCOPE")
    results = state.get("results")
    require(isinstance(results, dict), "NATIVE_RECOVERY_RESULTS_OBJECT")
    uploaded = []
    for sid, result in sorted(results.items()):
        require(isinstance(sid, str) and sid and "/" not in sid and "\\" not in sid
                and isinstance(result, dict) and result.get("subject_id") == sid,
                "NATIVE_RECOVERY_SUBJECT_ID")
        require(result.get("status") in PASS_STATES | {FAIL_STATE}
                and result.get("training_authorized") is False,
                "NATIVE_RECOVERY_UNEXPECTED_RESULT")
        folder = root / "subjects" / sid
        court = folder / "TEMPORAL_TEACHER_COURT_V1.json"
        require(court.is_file() and sha(court) == result.get("court_sha256"),
                "NATIVE_RECOVERY_COURT_SHA:" + sid)
        actual = json.loads(court.read_text())
        for key in ("status", "subject_id", "source_sha256", "training_authorized"):
            require(result.get(key) == actual.get(key),
                    "NATIVE_RECOVERY_COURT_INDEX_DRIFT:" + sid + ":" + key)
        if result["status"] in PASS_STATES:
            for key in ("teacher_arrays", "teacher_manifest"):
                ref = result.get(key) or {}
                path = Path(str(ref.get("path") or "")).resolve()
                require(path.is_file() and path.is_relative_to(folder.resolve())
                        and sha(path) == ref.get("sha256"),
                        "NATIVE_RECOVERY_TEACHER_HASH:" + sid + ":" + key)
        uploaded.append(sid)
    return uploaded


def rclone(command: list[str], *, timeout: int = 1800) -> None:
    proc = subprocess.run(["rclone", *command], check=False, timeout=timeout)
    require(proc.returncode == 0, "NATIVE_RECOVERY_REMOTE_COPY_FAILED:" + str(proc.returncode))


def snapshot(root: Path) -> tuple[dict | None, str | None]:
    index = root / INDEX
    if not index.is_file():
        return None, None
    try:
        raw = index.read_bytes()
        state = json.loads(raw)
        checked_result_files(root, state)
        return state, hashlib.sha256(raw).hexdigest()
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
        raise ValueError("NATIVE_RECOVERY_LOCAL_CHECKPOINT_INVALID:" + str(e)) from e


def recover_remote(root: Path, remote: str) -> None:
    """Restore trusted bytes before the original worker checks identity itself."""
    require(remote.startswith(REMOTE_PREFIX) and ".." not in remote and
            not remote.endswith("/"), "NATIVE_RECOVERY_REMOTE_IDENTITY")
    root.mkdir(parents=True, exist_ok=True)
    # An empty new remote is legitimate. Never assume remote metadata alone
    # confers truth authority; the original sweep verifies its exact source_id.
    exists = subprocess.run(["rclone", "lsf", remote, "--max-depth", "1"],
                            check=False, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, timeout=90).returncode == 0
    if exists:
        rclone(["copy", remote, str(root), "--checksum", "--transfers", "3"])
    snapshot(root)


def sync_progress(root: Path, remote: str, previous: set[str], previous_hash: str | None
                  ) -> tuple[set[str], str | None]:
    state, digest = snapshot(root)
    if state is None or digest == previous_hash:
        return previous, previous_hash
    subjects = set(state["results"])
    require(subjects.issuperset(previous), "NATIVE_RECOVERY_CHECKPOINT_REGRESSED")
    # Upload files for newly finished subjects first; remote index LAST.
    for sid in sorted(subjects - previous):
        rclone(["copy", str(root / "subjects" / sid),
                remote + "/subjects/" + sid, "--checksum", "--transfers", "2"])
    # A checkpoint may change its final status/rollup without a new subject.
    rclone(["copyto", str(root / INDEX), remote + "/" + INDEX, "--checksum"])
    print(json.dumps({
        "event": "NATIVE_V5_REMOTE_CHECKPOINT_VERIFIED",
        "measured_result_count": len(subjects),
        "newly_uploaded_subject_count": len(subjects - previous),
        "checkpoint_sha256": digest,
        "training_authorized": False,
    }, sort_keys=True), flush=True)
    return subjects, digest


def worker_command(args) -> list[str]:
    return [
        str(args.python), "-u", "tools/corpus/run_corpus370_v5_native_teacher_all_v1.py",
        "--v5-index", str(args.v5_index),
        "--exact-source-roster", str(args.exact_source_roster),
        "--extracted-source-root", str(args.extracted_source_root),
        "--blender", str(args.blender),
        "--output", str(args.output),
        "--max-frames", "16",
    ]


def run(args) -> None:
    root = args.output.resolve()
    remote = args.remote
    require(30 <= args.interval_seconds <= 300, "NATIVE_RECOVERY_INTERVAL")
    recover_remote(root, remote)
    previous, previous_hash = set(), None
    # Upload any locally finished subjects after checking their hash, even if
    # the previous attempt lost the runner during a remote copy.
    previous, previous_hash = sync_progress(root, remote, previous, previous_hash)
    proc = subprocess.Popen(worker_command(args), stdout=None, stderr=None)
    remote_errors: list[str] = []
    try:
        while proc.poll() is None:
            time.sleep(args.interval_seconds)
            try:
                previous, previous_hash = sync_progress(
                    root, remote, previous, previous_hash)
            except Exception as e:
                remote_errors.append(str(e))
                print(json.dumps({"event": "NATIVE_V5_REMOTE_RETRY_NEEDED",
                                  "error": str(e), "training_authorized": False}),
                      flush=True)
        ret = proc.wait()
        # Always try once more on process exit, including failed workers.
        previous, previous_hash = sync_progress(root, remote, previous, previous_hash)
        if ret != 0:
            raise RuntimeError("NATIVE_RECOVERY_SOURCE_WORKER_EXIT:" + str(ret))
        if remote_errors:
            print(json.dumps({"event": "NATIVE_V5_REMOTE_TRANSIENT_ERRORS",
                              "count": len(remote_errors),
                              "eventually_recovered": True}), flush=True)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--python", type=Path, default=Path(sys.executable))
    p.add_argument("--v5-index", type=Path, required=True)
    p.add_argument("--exact-source-roster", type=Path, required=True)
    p.add_argument("--extracted-source-root", type=Path, required=True)
    p.add_argument("--blender", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--remote", type=str, required=True)
    p.add_argument("--interval-seconds", type=int, default=45)
    args=p.parse_args()
    run(args)


if __name__=="__main__":
    main()
