#!/usr/bin/env python3
"""Seal an exact-hash execution receipt for one court-qualified Truth139 materialization job.

This is an execution/provenance receipt, not a scientific-admission or training authority.  It binds
one subject/job from the frozen materialization work order to the exact producer/court source bytes,
raw source identity, producer output manifest, independent court result and runtime environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_SCHEMA = "RealSaS.Truth139MaterializationWorkOrders.v1"
OUTPUT_SCHEMA = "RealSaS.Truth139MaterializationExecutionReceipt.v1"
PASS_STATUS = "PASS_TRUTH139_MATERIALIZATION_EXECUTION_BOUND__SCIENTIFIC_ADMISSION_SEPARATE"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _runtime() -> dict:
    torch_info: dict[str, Any] = {
        "available": False,
        "version": None,
        "cuda_runtime": None,
        "cuda_available": False,
        "gpu0": None,
    }
    try:
        import torch

        torch_info.update(
            available=True,
            version=torch.__version__,
            cuda_runtime=torch.version.cuda,
            cuda_available=bool(torch.cuda.is_available()),
        )
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            torch_info["gpu0"] = {
                "name": props.name,
                "total_memory_bytes": int(props.total_memory),
                "compute_capability": [int(props.major), int(props.minor)],
            }
    except Exception:
        pass
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch_info,
        "sys_executable": sys.executable,
    }


def _require_hex_sha(value: object, code: str, length: int = 64) -> str:
    text = str(value or "")
    if len(text) != length or any(ch not in "0123456789abcdefABCDEF" for ch in text):
        raise ValueError(code)
    return text.lower()


def checked_out_git_sha(repo_root: Path) -> str:
    repo_root = Path(repo_root).resolve()
    try:
        value = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("TRUTH139_RECEIPT_GIT_CHECKOUT_REQUIRED") from exc
    return _require_hex_sha(value, "TRUTH139_RECEIPT_GIT_HEAD_INVALID", length=40)


def _load_json(path: Path, code: str) -> dict:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(code + ":" + str(path))
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(code + "_NOT_OBJECT")
    return value


def _subject_job(work: dict, subject_id: str, job_name: str) -> tuple[dict, dict, dict]:
    rows = [row for row in work.get("subjects") or () if str(row.get("subject_id") or "") == subject_id]
    if len(rows) != 1:
        raise ValueError("TRUTH139_RECEIPT_SUBJECT_BINDING")
    subject = rows[0]
    jobs = dict(subject.get("jobs") or {})
    if job_name not in jobs:
        raise ValueError("TRUTH139_RECEIPT_JOB_BINDING")
    job = dict(jobs[job_name])
    if not bool(job.get("enabled")):
        raise ValueError("TRUTH139_RECEIPT_JOB_DISABLED")
    implementation = dict(job.get("implementation") or {})
    canonical = dict((work.get("canonical_implementations") or {}).get(job_name) or {})
    if not implementation or implementation != canonical:
        raise ValueError("TRUTH139_RECEIPT_IMPLEMENTATION_NOT_CANONICAL")
    return subject, job, implementation


def _source_file_binding(repo_root: Path, relative: str, code: str) -> dict:
    path = (repo_root / relative).resolve()
    if not path.is_relative_to(repo_root.resolve()) or not path.is_file():
        raise FileNotFoundError(code + ":" + relative)
    return {"path": relative, "sha256": file_sha256(path)}


def _manifest_hash_bindings(value: object, *, prefix: str = "") -> dict[str, str]:
    """Collect declared path+sha256 input/output bindings without interpreting their semantics."""
    out: dict[str, str] = {}
    if isinstance(value, dict):
        path = value.get("path")
        sha = value.get("sha256")
        if isinstance(path, str) and isinstance(sha, str) and len(sha) == 64:
            out[prefix or "root"] = sha.lower()
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.update(_manifest_hash_bindings(child, prefix=child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_prefix = f"{prefix}[{index}]"
            out.update(_manifest_hash_bindings(child, prefix=child_prefix))
    return out


def seal(
    *,
    work_orders_path: Path,
    subject_id: str,
    job_name: str,
    producer_manifest_path: Path,
    court_result_path: Path,
    code_commit_sha: str,
    execution_kind: str,
    wall_seconds: float | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict:
    repo_root = Path(repo_root).resolve()
    work_orders_path = Path(work_orders_path).resolve()
    producer_manifest_path = Path(producer_manifest_path).resolve()
    court_result_path = Path(court_result_path).resolve()
    work = _load_json(work_orders_path, "TRUTH139_RECEIPT_WORK_ORDERS")
    if work.get("schema") != INPUT_SCHEMA:
        raise ValueError("TRUTH139_RECEIPT_WORK_ORDER_SCHEMA")
    work_identity = _require_hex_sha(work.get("work_order_identity_sha256"), "TRUTH139_RECEIPT_WORK_ORDER_IDENTITY")
    # Recompute the work-order identity exactly as the builder does: the identity field itself is
    # excluded from its own digest.
    work_without_identity = dict(work)
    work_without_identity.pop("work_order_identity_sha256", None)
    if canonical_hash(work_without_identity) != work_identity:
        raise ValueError("TRUTH139_RECEIPT_WORK_ORDER_IDENTITY_DRIFT")

    subject, job, implementation = _subject_job(work, subject_id, job_name)
    source = dict(subject.get("source") or {})
    raw_source_sha = _require_hex_sha(source.get("sha256"), "TRUTH139_RECEIPT_RAW_SOURCE_SHA")
    producer_source = _source_file_binding(repo_root, str(implementation["producer"]), "TRUTH139_RECEIPT_PRODUCER_SOURCE")
    court_source = _source_file_binding(repo_root, str(implementation["independent_court"]), "TRUTH139_RECEIPT_COURT_SOURCE")

    producer = _load_json(producer_manifest_path, "TRUTH139_RECEIPT_PRODUCER_MANIFEST")
    if str(producer.get("subject_id") or "") != subject_id:
        raise ValueError("TRUTH139_RECEIPT_PRODUCER_SUBJECT")
    expected_producer_status = str(implementation.get("producer_pending_status") or "")
    if producer.get("status") != expected_producer_status:
        raise ValueError("TRUTH139_RECEIPT_PRODUCER_STATUS")
    if implementation.get("producer_revision") is not None and producer.get("producer_revision") != implementation["producer_revision"]:
        raise ValueError("TRUTH139_RECEIPT_PRODUCER_REVISION")

    court = _load_json(court_result_path, "TRUTH139_RECEIPT_COURT_RESULT")
    if str(court.get("subject_id") or "") != subject_id:
        raise ValueError("TRUTH139_RECEIPT_COURT_SUBJECT")
    if court.get("status") != implementation.get("required_court_status"):
        raise ValueError("TRUTH139_RECEIPT_COURT_STATUS")

    commit_sha = _require_hex_sha(code_commit_sha, "TRUTH139_RECEIPT_CODE_COMMIT_SHA", length=40)
    checkout_sha = checked_out_git_sha(repo_root)
    if commit_sha != checkout_sha:
        raise ValueError(f"TRUTH139_RECEIPT_CODE_COMMIT_DRIFT:{commit_sha}!={checkout_sha}")
    if not execution_kind:
        raise ValueError("TRUTH139_RECEIPT_EXECUTION_KIND")
    if wall_seconds is not None and (not np.isfinite(wall_seconds) or wall_seconds < 0):
        raise ValueError("TRUTH139_RECEIPT_WALL_SECONDS")

    producer_manifest_sha = file_sha256(producer_manifest_path)
    court_result_sha = file_sha256(court_result_path)
    receipt = {
        "schema": OUTPUT_SCHEMA,
        "date": "2026-10-29",
        "status": PASS_STATUS,
        "subject_id": subject_id,
        "job_name": job_name,
        "execution_kind": execution_kind,
        "code_commit_sha": commit_sha,
        "checked_out_git_sha": checkout_sha,
        "work_order": {
            "path": str(work_orders_path),
            "sha256": file_sha256(work_orders_path),
            "work_order_identity_sha256": work_identity,
            "parent_corpus_manifest_identity_sha256": _require_hex_sha(
                work.get("parent_corpus_manifest_identity_sha256"),
                "TRUTH139_RECEIPT_PARENT_CORPUS_IDENTITY",
            ),
        },
        "subject_source": {
            "corpus_relative_path": str(source.get("corpus_relative_path") or ""),
            "sha256": raw_source_sha,
            "dependency_count": int(source.get("dependency_count", len(source.get("dependencies") or ()))),
            "source_page_snapshot_sha256": _require_hex_sha(
                (source.get("source_page_snapshot") or {}).get("sha256"),
                "TRUTH139_RECEIPT_SOURCE_PAGE_SNAPSHOT_SHA",
            ),
        },
        "implementation": {
            "declared": implementation,
            "producer_source": producer_source,
            "independent_court_source": court_source,
        },
        "producer_output": {
            "manifest_path": str(producer_manifest_path),
            "manifest_sha256": producer_manifest_sha,
            "status": producer["status"],
            "producer_revision": producer.get("producer_revision"),
            "declared_path_sha_bindings": _manifest_hash_bindings(producer),
        },
        "independent_court": {
            "result_path": str(court_result_path),
            "result_sha256": court_result_sha,
            "status": court["status"],
            "court_identity_sha256": court.get("court_identity_sha256"),
        },
        "runtime_environment": _runtime(),
        "wall_seconds": None if wall_seconds is None else float(wall_seconds),
        "derived_truth_execution_completed": True,
        "independent_court_passed": True,
        "derived_truth_admitted": False,
        "training_performed": False,
        "split_created": False,
        "qualification_minted": False,
        "product_pass_claimed": False,
    }
    receipt["execution_binding_sha256"] = canonical_hash(receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-orders", required=True, type=Path)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--producer-manifest", required=True, type=Path)
    parser.add_argument("--court-result", required=True, type=Path)
    parser.add_argument("--code-commit-sha", required=True)
    parser.add_argument("--execution-kind", default="EXTERNAL_GPU_WORKER_PLUS_CPU_COURT")
    parser.add_argument("--wall-seconds", type=float)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = seal(
        work_orders_path=args.work_orders,
        subject_id=args.subject_id,
        job_name=args.job_name,
        producer_manifest_path=args.producer_manifest,
        court_result_path=args.court_result,
        code_commit_sha=args.code_commit_sha,
        execution_kind=args.execution_kind,
        wall_seconds=args.wall_seconds,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = output.with_suffix(output.suffix + ".partial")
    staged.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    staged.replace(output)
    print(json.dumps({
        "status": result["status"],
        "subject_id": result["subject_id"],
        "job_name": result["job_name"],
        "execution_binding_sha256": result["execution_binding_sha256"],
        "receipt_sha256": file_sha256(output),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
