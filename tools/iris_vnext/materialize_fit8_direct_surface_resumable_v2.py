#!/usr/bin/env python3
from __future__ import annotations

"""Resumable process-isolated FIT8 direct-surface materializer.

Scientific semantics are exactly the V1 direct-surface contract: exact completed
V8, identity learned scalar field, and SparseRegularTetraPolicyV5 T512->T1024.
V2 changes orchestration only. Each subject runs in a child Python process so a
SIGKILL/resource death is classified by the surviving parent, completed subjects
are hash-validated and reused, and partial progress is preserved.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any

SUBJECT_IDS = ("Snake", "Bat", "T-Rex", "Horse", "Spider", "GreenBlob", "Shark", "George")
DECODER_ID = "RealSaS.ZeroSurfaceDecoder.SparseRegularT512T1024IndexedMT.v5"
SUBJECT_SCHEMA = "RealSaS.Fit8DirectIRISSurfaceSubjectExecution.v2"
FAILURE_SCHEMA = "RealSaS.Fit8DirectIRISSurfaceParentFailure.v2"
BATCH_SCHEMA = "RealSaS.Fit8DirectIRISSurfaceMaterialization.v1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def value_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("DIRECT_SURFACE_V2_JSON_OBJECT_REQUIRED:" + str(path))
    return value


def _atomic_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=path.name + ".", suffix=".staged", delete=False,
    ) as handle:
        staged = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    try:
        os.replace(staged, path)
    finally:
        staged.unlink(missing_ok=True)


def git_head(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip().lower()


def classify_returncode(returncode: int) -> dict[str, Any]:
    rc = int(returncode)
    if rc >= 0:
        return {"returncode": rc, "terminated_by_signal": False, "signal_number": None, "signal_name": None}
    number = -rc
    try:
        name = signal.Signals(number).name
    except ValueError:
        name = "UNKNOWN_SIGNAL"
    return {"returncode": rc, "terminated_by_signal": True, "signal_number": number, "signal_name": name}


def _valid_completed_subject(
    subject_root: Path,
    *,
    subject_id: str,
    expected_source_sha: str,
    prepared_manifest_sha: str,
    prepared_shard_sha: str,
    v8_checkpoint_sha: str,
) -> dict[str, Any] | None:
    execution_path = subject_root / "DIRECT_IRIS_SURFACE_SUBJECT_EXECUTION_V2.json"
    direct_path = subject_root / "DIRECT_IRIS_SURFACE_RECEIPT.json"
    stage_path = subject_root / "DIRECT_IRIS_SURFACE.stage12.npz"
    if not (execution_path.is_file() and direct_path.is_file() and stage_path.is_file()):
        return None
    try:
        row = load_json(execution_path)
        direct = load_json(direct_path)
    except Exception:
        return None
    exact = {
        "schema": SUBJECT_SCHEMA,
        "status": "PASS__EXACT_T512_T1024_SUBJECT_MATERIALIZED__NO_TRAINING",
        "subject_id": subject_id,
        "execution_source_sha": expected_source_sha,
        "prepared_manifest_sha256": prepared_manifest_sha,
        "prepared_shard_sha256": prepared_shard_sha,
        "v8_checkpoint_sha256": v8_checkpoint_sha,
        "decoder_id": DECODER_ID,
        "base_cells": 512,
        "fine_cells": 1024,
    }
    if any(row.get(key) != expected for key, expected in exact.items()):
        return None
    if row.get("teacher_truth_used") is not False or row.get("tessa_invoked") is not False:
        return None
    if int(row.get("optimizer_steps_executed", -1)) != 0 or int(row.get("backward_steps_executed", -1)) != 0:
        return None
    if sha256_file(stage_path) != row.get("stage12_npz_sha256"):
        return None
    if sha256_file(direct_path) != row.get("receipt_sha256"):
        return None
    if direct.get("source_commit") != expected_source_sha or direct.get("decoder_id") != DECODER_ID:
        return None
    if int(direct.get("base_cells", -1)) != 512 or int(direct.get("fine_cells", -1)) != 1024:
        return None
    return row


def _batch_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "subject_id": row["subject_id"],
        "prepared_shard_file": row["prepared_shard_file"],
        "prepared_shard_sha256": row["prepared_shard_sha256"],
        "scene_binding_sha256": row["scene_binding_sha256"],
        "scene_planes_sha256": row["scene_planes_sha256"],
        "candidate_binding_hash": row["candidate_binding_hash"],
        "stage12_npz_sha256": row["stage12_npz_sha256"],
        "receipt_sha256": row["receipt_sha256"],
        "vertex_count": int(row["vertex_count"]),
        "face_count": int(row["face_count"]),
        "field_query_api": row.get("field_query_api"),
        "compiler_qualified": False,
    }


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    expected_source_sha = str(args.expected_source_sha).strip().lower()
    if git_head(repo_root) != expected_source_sha:
        raise RuntimeError("DIRECT_SURFACE_V2_CHECKOUT_SHA_DRIFT")

    prepared_root = Path(args.prepared_root).resolve()
    manifest_path = Path(args.prepared_manifest).resolve()
    manifest = load_json(manifest_path)
    if manifest.get("schema") != "RealSaS.IRISVNextFit8Manifest.v4":
        raise ValueError("DIRECT_SURFACE_V2_PREPARED_SCHEMA_INVALID")
    rows = list(manifest.get("subjects") or ())
    if tuple(row.get("subject_id") for row in rows) != SUBJECT_IDS:
        raise ValueError("DIRECT_SURFACE_V2_PREPARED_SUBJECT_ORDER_DRIFT")
    prepared_manifest_sha = sha256_file(manifest_path)

    checkpoint_path = Path(args.v8_checkpoint).resolve()
    v8_checkpoint_sha = sha256_file(checkpoint_path)
    if v8_checkpoint_sha != "0be15df5614a05343cf3850aff747ba4d31f00a322f058819cb5a54d5a5d2fc5":
        raise ValueError("DIRECT_SURFACE_V2_V8_CHECKPOINT_SHA_DRIFT")

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    completed: list[dict[str, Any]] = []

    for manifest_row in rows:
        subject_id = str(manifest_row["subject_id"])
        shard_sha = str(manifest_row["shard_sha256"]).lower()
        shard_path = (prepared_root / str(manifest_row["shard_file"])).resolve()
        if not shard_path.is_relative_to(prepared_root) or sha256_file(shard_path) != shard_sha:
            raise ValueError("DIRECT_SURFACE_V2_PREPARED_SHARD_DRIFT:" + subject_id)

        subject_root = output_root / "subjects" / subject_id
        prior = _valid_completed_subject(
            subject_root,
            subject_id=subject_id,
            expected_source_sha=expected_source_sha,
            prepared_manifest_sha=prepared_manifest_sha,
            prepared_shard_sha=shard_sha,
            v8_checkpoint_sha=v8_checkpoint_sha,
        )
        if prior is not None:
            print(f"DIRECT_SURFACE_SUBJECT_REUSE subject={subject_id} status=PASS", flush=True)
            completed.append(prior)
            continue

        shutil.rmtree(subject_root, ignore_errors=True)
        subject_root.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "tools.iris_vnext.materialize_fit8_direct_surface_subject_v2",
            "--repo-root", str(repo_root),
            "--expected-source-sha", expected_source_sha,
            "--subject-id", subject_id,
            "--prepared-root", str(prepared_root),
            "--prepared-manifest", str(manifest_path),
            "--v8-checkpoint", str(checkpoint_path),
            "--output-root", str(output_root),
            "--device", str(args.device),
            "--model-query-chunk", str(int(args.model_query_chunk)),
        ]
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        print("DIRECT_SURFACE_SUBJECT_START", " ".join(cmd), flush=True)
        started = time.time()
        proc = subprocess.run(cmd, cwd=repo_root, env=env)
        classification = classify_returncode(proc.returncode)
        print(
            f"DIRECT_SURFACE_SUBJECT_EXIT subject={subject_id} rc={proc.returncode} "
            f"signal={classification['signal_name']} elapsed={time.time() - started:.3f}s",
            flush=True,
        )
        if proc.returncode != 0:
            progress_path = subject_root / "DIRECT_IRIS_SURFACE_PROGRESS_V2.json"
            child_failure_path = subject_root / "DIRECT_IRIS_SURFACE_FAILURE_V2.json"
            failure = {
                "schema": FAILURE_SCHEMA,
                "status": "FAIL__ISOLATED_SUBJECT_PROCESS__NO_TRAINING",
                "subject_id": subject_id,
                "execution_source_sha": expected_source_sha,
                **classification,
                "elapsed_seconds": float(time.time() - started),
                "progress_receipt_present": progress_path.is_file(),
                "progress": load_json(progress_path) if progress_path.is_file() else None,
                "child_failure_receipt_present": child_failure_path.is_file(),
                "child_failure": load_json(child_failure_path) if child_failure_path.is_file() else None,
                "completed_subjects_preserved": [row["subject_id"] for row in completed],
                "decoder_id": DECODER_ID,
                "base_cells": 512,
                "fine_cells": 1024,
                "optimizer_steps_executed": 0,
                "backward_steps_executed": 0,
                "scientific_pass_claimed": False,
            }
            failure_path = output_root / "DIRECT_IRIS_SURFACE_PARENT_FAILURE_V2.json"
            _atomic_json(failure, failure_path)
            signal_text = classification["signal_name"] or "NONE"
            raise RuntimeError(
                f"DIRECT_SURFACE_V2_SUBJECT_PROCESS_FAILED:{subject_id}:rc={proc.returncode}:signal={signal_text}:receipt={failure_path}"
            )

        row = _valid_completed_subject(
            subject_root,
            subject_id=subject_id,
            expected_source_sha=expected_source_sha,
            prepared_manifest_sha=prepared_manifest_sha,
            prepared_shard_sha=shard_sha,
            v8_checkpoint_sha=v8_checkpoint_sha,
        )
        if row is None:
            raise RuntimeError("DIRECT_SURFACE_V2_SUBJECT_CHILD_EXITED_ZERO_WITHOUT_VALID_RECEIPT:" + subject_id)
        completed.append(row)
        # A previous failure receipt is stale only after a new subject PASS.
        (output_root / "DIRECT_IRIS_SURFACE_PARENT_FAILURE_V2.json").unlink(missing_ok=True)

    if tuple(row["subject_id"] for row in completed) != SUBJECT_IDS:
        raise RuntimeError("DIRECT_SURFACE_V2_SUBJECT_MATERIALIZATION_INCOMPLETE")

    batch = {
        "schema": BATCH_SCHEMA,
        "status": "MATERIALIZED_MODEL_SURFACES__COMPILER_COURT_PENDING",
        "execution_source_sha": expected_source_sha,
        "prepared_manifest_sha256": prepared_manifest_sha,
        "v8_authority": {
            "source_commit": "5083c9915533ec8ec6530c7d229c635887abfea7",
            "checkpoint_sha256": v8_checkpoint_sha,
            "binding_sha256": "735b37f450e947eaad5928f39fca2fb92c61b4642f71044d5b19da5877a53ec2",
            "global_step": 12000,
            "scene_core": "SEALED",
        },
        "numeric_policy": "FP32_NO_AUTOCAST",
        "sparse_surface_policy": {
            "base_cells": 512,
            "fine_cells": 1024,
            "policy": "EXACT_SELECTED_T512_TO_SPARSE_T1024_V5",
            "model_query_chunk": int(args.model_query_chunk),
        },
        "runtime_hardening": {
            "schema": "RealSaS.Fit8DirectIRISSurfaceRuntimeHardening.v2",
            "orchestration_only": True,
            "subject_process_isolation": True,
            "completed_subject_resume": True,
            "returncode_and_signal_classification": True,
            "decoder_semantics_changed": False,
            "field_semantics_changed": False,
            "surface_policy_changed": False,
        },
        "subject_order": list(SUBJECT_IDS),
        "subjects": [_batch_row(row) for row in completed],
        "tessa_invoked": False,
        "teacher_truth_used": False,
        "optimizer_steps_executed": 0,
        "compiler_court_executed": False,
        "training_authorized": False,
        "scientific_pass_claimed": False,
        "next_action": "RUN_INDEPENDENT_SOURCE_FIDELITY_AND_COMPILER_COURTS",
    }
    batch["binding_sha256"] = value_sha256(batch)
    batch_path = output_root / "FIT8_DIRECT_IRIS_SURFACE_BATCH_RECEIPT.json"
    _atomic_json(batch, batch_path)
    (output_root / "DIRECT_IRIS_SURFACE_PARENT_FAILURE_V2.json").unlink(missing_ok=True)
    return batch


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", required=True)
    p.add_argument("--expected-source-sha", required=True)
    p.add_argument("--prepared-root", required=True)
    p.add_argument("--prepared-manifest", required=True)
    p.add_argument("--v8-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--model-query-chunk", type=int, default=32768)
    return p


def main() -> int:
    result = materialize(parser().parse_args())
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
