#!/usr/bin/env python3
"""Materialize Truth139 IRIS response evidence across the frozen 139-subject work order.

This batch runner resolves each prepared source-observation packet, invokes the
single-subject Stage47 response chain, reuses already court-passed chains, and
seals a cohort receipt.  It does not run Stage48 renderer interventions, create a
split, train a model, or mint qualification/product authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.corpus.build_truth139_response_evidence_v1 import sha256

CHAIN_PASS = "PASS_TRUTH139_IRIS_RESPONSE_EVIDENCE_CHAIN__DERIVED_TRUTH_ADMISSION_SEPARATE"
BATCH_PASS = "PASS_FULL139_IRIS_RESPONSE_EVIDENCE_COMPLETE__DERIVED_TRUTH_ADMISSION_SEPARATE"
CONTRACT = ROOT / "canonical/TRUTH139_IRIS_RESPONSE_EVIDENCE_TEACHER_V1.json"
WORK_ORDER_SCHEMA = "RealSaS.Truth139MaterializationWorkOrders.v1"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def load(path: Path, code: str) -> dict:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(code + ":" + str(path))
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(code + "_NOT_OBJECT")
    return value


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def packet_root(prepared_root: Path, subject_id: str) -> Path:
    roots = sorted(prepared_root.glob(f"{subject_id}/objects/*/packet/{subject_id}"))
    if len(roots) != 1 or not roots[0].is_dir():
        raise RuntimeError("TRUTH139_RESPONSE_PACKET_CARDINALITY:" + subject_id + ":" + str(len(roots)))
    return roots[0].resolve()


def chain_complete(output_root: Path, subject_id: str) -> bool:
    path = output_root / subject_id / "IRIS_RESPONSE_EVIDENCE_CHAIN.json"
    try:
        value = load(path, "TRUTH139_RESPONSE_CHAIN")
        return value.get("status") == CHAIN_PASS and value.get("subject_id") == subject_id
    except Exception:
        return False


def run_subject(
    *,
    subject_id: str,
    source: Path,
    packet: Path,
    output: Path,
    blender: Path,
    maximum_seconds: float,
) -> None:
    command = [
        sys.executable,
        str(ROOT / "tools/corpus/run_truth139_response_evidence_subject_v1.py"),
        "--subject-id", subject_id,
        "--source", str(source),
        "--packet-root", str(packet),
        "--output-root", str(output),
        "--blender", str(blender),
        "--maximum-seconds", str(maximum_seconds),
    ]
    print("[RUN]", " ".join(command), flush=True)
    env = {**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    process = subprocess.run(command, cwd=str(ROOT), env=env, text=True, capture_output=True)
    log = output / "subject.runner.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text((process.stdout or "") + ("\n[STDERR]\n" + process.stderr if process.stderr else ""))
    if process.stdout:
        print(process.stdout, end="", flush=True)
    if process.returncode != 0:
        tail = log.read_text(errors="replace")[-20000:]
        raise RuntimeError("TRUTH139_RESPONSE_SUBJECT_FAILED\n" + tail)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-orders", required=True, type=Path)
    parser.add_argument("--prepared-root", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--blender", required=True, type=Path)
    parser.add_argument("--maximum-seconds-per-subject", type=float, default=1800.0)
    parser.add_argument("--continue-on-failure", action="store_true")
    args = parser.parse_args()

    work_path = args.work_orders.resolve()
    prepared_root = args.prepared_root.resolve()
    raw_root = args.raw_root.resolve()
    output_root = args.output_root.resolve()
    blender = args.blender.resolve()
    if not prepared_root.is_dir() or not raw_root.is_dir() or not blender.is_file():
        raise FileNotFoundError("TRUTH139_RESPONSE_BATCH_REQUIRED_ROOT_MISSING")
    output_root.mkdir(parents=True, exist_ok=True)

    work = load(work_path, "TRUTH139_RESPONSE_WORK_ORDERS")
    if work.get("schema") != WORK_ORDER_SCHEMA:
        raise ValueError("TRUTH139_RESPONSE_WORK_ORDER_SCHEMA")
    rows = list(work.get("subjects") or ())
    subject_ids_declared = [str(row.get("subject_id") or "") for row in rows]
    if (
        int(work.get("subject_count", -1)) != 139
        or len(rows) != 139
        or any(not subject_id for subject_id in subject_ids_declared)
        or len(set(subject_ids_declared)) != 139
    ):
        raise ValueError("TRUTH139_RESPONSE_WORK_ORDER_DENOMINATOR")
    counts = dict(work.get("job_candidate_counts") or {})
    if int(counts.get("IRIS_RESPONSE_EVIDENCE", -1)) != 139:
        raise ValueError("TRUTH139_RESPONSE_WORK_ORDER_CANDIDATE_COUNT")

    code_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()
    started = utc()
    failures: dict[str, str] = {}
    executed, reused = [], []
    subject_ids = []
    output_bindings = {}

    for index, row in enumerate(rows, start=1):
        subject_id = str(row.get("subject_id") or "")
        subject_ids.append(subject_id)
        jobs = dict(row.get("jobs") or {})
        response = dict(jobs.get("IRIS_RESPONSE_EVIDENCE") or {})
        if response.get("enabled") is not True:
            raise ValueError("TRUTH139_RESPONSE_JOB_NOT_ENABLED:" + subject_id)
        source_row = dict(row.get("source") or {})
        relative = str(source_row.get("corpus_relative_path") or "")
        source = (raw_root / relative).resolve()
        if not source.is_file() or sha256(source) != source_row.get("sha256"):
            raise ValueError("TRUTH139_RESPONSE_RAW_SOURCE_DRIFT:" + subject_id)
        packet = packet_root(prepared_root, subject_id)
        out = output_root / "subjects" / subject_id
        print(f"\n[RESPONSE {index:03d}/139] {subject_id}", flush=True)
        if chain_complete(output_root / "subjects", subject_id):
            print("[REUSE-RESPONSE]", subject_id, flush=True)
            reused.append(subject_id)
        else:
            try:
                run_subject(
                    subject_id=subject_id,
                    source=source,
                    packet=packet,
                    output=out,
                    blender=blender,
                    maximum_seconds=float(args.maximum_seconds_per_subject),
                )
                if not chain_complete(output_root / "subjects", subject_id):
                    raise RuntimeError("TRUTH139_RESPONSE_CHAIN_DID_NOT_CLOSE")
                executed.append(subject_id)
                print("[PASS-RESPONSE]", subject_id, flush=True)
            except Exception as exc:
                failures[subject_id] = repr(exc)
                print("[FAIL-RESPONSE]", subject_id, repr(exc), flush=True)
                if not args.continue_on_failure:
                    break
        if chain_complete(output_root / "subjects", subject_id):
            path = out / "IRIS_RESPONSE_EVIDENCE_CHAIN.json"
            value = load(path, "TRUTH139_RESPONSE_OUTPUT_CHAIN")
            output_bindings[subject_id] = {
                "chain_path": str(path),
                "chain_sha256": sha256(path),
                "chain_identity_sha256": value.get("chain_identity_sha256"),
            }
        progress = {
            "schema": "RealSaS.Truth139IrisResponseEvidenceProgress.v1",
            "subject_count": 139,
            "completed_subject_count": len(output_bindings),
            "failed_subject_count": len(failures),
            "failed_jobs_by_subject": failures,
            "last_subject_id": subject_id,
            "updated_at_utc": utc(),
            "training_performed": False,
            "split_created": False,
        }
        atomic_json(output_root / "FULL139_IRIS_RESPONSE_EVIDENCE_PROGRESS.json", progress)

    # Re-scan all chains because a prior partial run may have completed entries after a stale progress file.
    output_bindings = {}
    for subject_id in subject_ids:
        if chain_complete(output_root / "subjects", subject_id):
            path = output_root / "subjects" / subject_id / "IRIS_RESPONSE_EVIDENCE_CHAIN.json"
            value = load(path, "TRUTH139_RESPONSE_OUTPUT_CHAIN")
            output_bindings[subject_id] = {
                "chain_path": str(path),
                "chain_sha256": sha256(path),
                "chain_identity_sha256": value.get("chain_identity_sha256"),
            }
    complete = [subject_id for subject_id in subject_ids if subject_id in output_bindings]
    status = BATCH_PASS if len(complete) == 139 and not failures else "FULL139_IRIS_RESPONSE_EVIDENCE_COMPLETED_WITH_FAILURES__DERIVED_TRUTH_ADMISSION_SEPARATE"
    receipt = {
        "schema": "RealSaS.Truth139IrisResponseEvidenceBatchReceipt.v1",
        "status": status,
        "cohort_scope": "FULL_TRUTH139__139_OF_139_SUBJECT_DENOMINATOR",
        "code_commit_sha": code_sha,
        "contract": {"path": str(CONTRACT), "sha256": sha256(CONTRACT)},
        "work_orders": {"path": str(work_path), "sha256": sha256(work_path), "identity_sha256": work.get("work_order_identity_sha256")},
        "subject_count": 139,
        "subject_ids": subject_ids,
        "completed_subject_count": len(complete),
        "completed_subject_ids": complete,
        "executed_subject_ids": executed,
        "reused_subject_ids": reused,
        "failed_jobs_by_subject": failures,
        "subject_output_bindings": output_bindings,
        "stage48_renderer_witness_performed": False,
        "started_at_utc": started,
        "completed_at_utc": utc(),
        "training_performed": False,
        "split_created": False,
        "qualification_minted": False,
    }
    receipt["receipt_identity_sha256"] = digest(receipt)
    receipt_path = output_root / "FULL139_IRIS_RESPONSE_EVIDENCE_RECEIPT.json"
    atomic_json(receipt_path, receipt)
    print("\n" + json.dumps({
        "status": status,
        "completed_subject_count": len(complete),
        "failed_subject_count": len(failures),
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256(receipt_path),
        "receipt_identity_sha256": receipt["receipt_identity_sha256"],
    }, indent=2), flush=True)
    if status != BATCH_PASS:
        raise RuntimeError(f"FULL139_IRIS_RESPONSE_EVIDENCE_INCOMPLETE:{len(complete)}/139")


if __name__ == "__main__":
    main()
