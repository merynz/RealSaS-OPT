#!/usr/bin/env python3
"""Execute one Truth139 subject's canonical Graphics materialization chain.

The external GPU worker is deliberately thin: all scientific logic remains in versioned repository
producers/courts.  This entrypoint validates the frozen work-order/source/checkout bindings, runs
Dense Graphics -> independent geometry court -> receipt -> Visibility/Occlusion V2 -> independent
CPU ray court -> receipt, then seals a chain manifest.  It does not admit truth or train a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from tools.corpus.seal_truth139_materialization_receipt_v1 import (
    PASS_STATUS as RECEIPT_PASS_STATUS,
    checked_out_git_sha,
    file_sha256,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
WORK_SCHEMA = "RealSaS.Truth139MaterializationWorkOrders.v1"
CHAIN_SCHEMA = "RealSaS.Truth139GraphicsMaterializationChain.v1"
CHAIN_STATUS = "PASS_TRUTH139_GRAPHICS_MATERIALIZATION_CHAIN_EXECUTED__DERIVED_TRUTH_ADMISSION_SEPARATE"
DENSE_JOB = "DENSE_GRAPHICS_FRONT_SURFACE"
VISIBILITY_JOB = "VISIBILITY_OCCLUSION_CROSS_VIEW"


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _json(path: Path, code: str) -> dict:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(code + ":" + str(path))
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(code + "_NOT_OBJECT")
    return value


def _subject(work: dict, subject_id: str) -> dict:
    matches = [row for row in work.get("subjects") or () if str(row.get("subject_id") or "") == subject_id]
    if len(matches) != 1:
        raise ValueError("TRUTH139_GRAPHICS_WORKER_SUBJECT_BINDING")
    return matches[0]


def _implementation(work: dict, subject: dict, job_name: str) -> dict:
    job = dict((subject.get("jobs") or {}).get(job_name) or {})
    canonical = dict((work.get("canonical_implementations") or {}).get(job_name) or {})
    if not bool(job.get("enabled")):
        raise ValueError("TRUTH139_GRAPHICS_WORKER_JOB_DISABLED:" + job_name)
    if not canonical or dict(job.get("implementation") or {}) != canonical:
        raise ValueError("TRUTH139_GRAPHICS_WORKER_IMPLEMENTATION_DRIFT:" + job_name)
    for key in ("producer", "producer_pending_status", "independent_court", "required_court_status"):
        if not canonical.get(key):
            raise ValueError("TRUTH139_GRAPHICS_WORKER_IMPLEMENTATION_FIELD:" + job_name + ":" + key)
    return canonical


def validate_request(
    *,
    work_orders_path: Path,
    subject_id: str,
    raw_source_path: Path,
    geometry_npz: Path,
    source_scene_snapshot: Path,
    camera_bundle: Path,
    observation_dir: Path,
    output_root: Path,
    code_commit_sha: str,
    repo_root: Path = REPO_ROOT,
) -> dict:
    repo_root = Path(repo_root).resolve()
    work_orders_path = Path(work_orders_path).resolve()
    raw_source_path = Path(raw_source_path).resolve()
    geometry_npz = Path(geometry_npz).resolve()
    source_scene_snapshot = Path(source_scene_snapshot).resolve()
    camera_bundle = Path(camera_bundle).resolve()
    observation_dir = Path(observation_dir).resolve()
    output_root = Path(output_root).resolve()

    work = _json(work_orders_path, "TRUTH139_GRAPHICS_WORKER_WORK_ORDERS")
    if work.get("schema") != WORK_SCHEMA:
        raise ValueError("TRUTH139_GRAPHICS_WORKER_WORK_SCHEMA")
    subject = _subject(work, subject_id)
    dense_impl = _implementation(work, subject, DENSE_JOB)
    visibility_impl = _implementation(work, subject, VISIBILITY_JOB)

    if not raw_source_path.is_file():
        raise FileNotFoundError("TRUTH139_GRAPHICS_WORKER_RAW_SOURCE")
    expected_source_sha = str((subject.get("source") or {}).get("sha256") or "").lower()
    actual_source_sha = file_sha256(raw_source_path)
    if len(expected_source_sha) != 64 or actual_source_sha != expected_source_sha:
        raise ValueError(f"TRUTH139_GRAPHICS_WORKER_RAW_SOURCE_SHA:{actual_source_sha}!={expected_source_sha}")
    for path, code in (
        (geometry_npz, "GEOMETRY"),
        (source_scene_snapshot, "SCENE_SNAPSHOT"),
        (camera_bundle, "CAMERA_BUNDLE"),
    ):
        if not path.is_file():
            raise FileNotFoundError("TRUTH139_GRAPHICS_WORKER_" + code)
    if not observation_dir.is_dir():
        raise FileNotFoundError("TRUTH139_GRAPHICS_WORKER_OBSERVATION_DIR")

    checkout = checked_out_git_sha(repo_root)
    if str(code_commit_sha).lower() != checkout:
        raise ValueError(f"TRUTH139_GRAPHICS_WORKER_CODE_SHA:{code_commit_sha}!={checkout}")

    for implementation, job_name in ((dense_impl, DENSE_JOB), (visibility_impl, VISIBILITY_JOB)):
        for field in ("producer", "independent_court"):
            source_path = (repo_root / str(implementation[field])).resolve()
            if not source_path.is_relative_to(repo_root) or not source_path.is_file():
                raise FileNotFoundError("TRUTH139_GRAPHICS_WORKER_CODE_PATH:" + job_name + ":" + field)

    return {
        "work": work,
        "subject": subject,
        "dense_impl": dense_impl,
        "visibility_impl": visibility_impl,
        "paths": {
            "work_orders": work_orders_path,
            "raw_source": raw_source_path,
            "geometry_npz": geometry_npz,
            "source_scene_snapshot": source_scene_snapshot,
            "camera_bundle": camera_bundle,
            "observation_dir": observation_dir,
            "output_root": output_root,
        },
        "bindings": {
            "code_commit_sha": checkout,
            "work_order_identity_sha256": str(work.get("work_order_identity_sha256") or ""),
            "raw_source_sha256": actual_source_sha,
            "geometry_npz_sha256": file_sha256(geometry_npz),
            "source_scene_snapshot_sha256": file_sha256(source_scene_snapshot),
            "camera_bundle_sha256": file_sha256(camera_bundle),
        },
    }


def build_command_plan(request: dict, *, subject_id: str, device: str) -> list[dict]:
    p = request["paths"]
    repo_root = REPO_ROOT
    output_root = p["output_root"]
    dense_dir = output_root / "dense_graphics"
    visibility_dir = output_root / "visibility_occlusion"
    receipts_dir = output_root / "receipts"
    dense_manifest = dense_dir / "DENSE_GRAPHICS_GBUFFER_MANIFEST.json"
    dense_court = dense_dir / "DENSE_GRAPHICS_GEOMETRY_COURT.json"
    visibility_manifest = visibility_dir / "VISIBILITY_OCCLUSION_TEACHER_MANIFEST.json"
    visibility_court = visibility_dir / "VISIBILITY_OCCLUSION_COURT.json"
    dense_receipt = receipts_dir / "DENSE_GRAPHICS_FRONT_SURFACE_RECEIPT.json"
    visibility_receipt = receipts_dir / "VISIBILITY_OCCLUSION_CROSS_VIEW_RECEIPT.json"
    code_sha = request["bindings"]["code_commit_sha"]

    def py(relative: str, *args: object) -> list[str]:
        return [sys.executable, str((repo_root / relative).resolve()), *map(str, args)]

    return [
        {
            "name": "DENSE_GRAPHICS_FRONT_SURFACE",
            "command": py(
                request["dense_impl"]["producer"],
                "--subject-id", subject_id,
                "--geometry-npz", p["geometry_npz"],
                "--source-scene-snapshot", p["source_scene_snapshot"],
                "--camera-bundle", p["camera_bundle"],
                "--observation-dir", p["observation_dir"],
                "--output-dir", dense_dir,
                "--device", device,
                "--observation-provenance", "SOURCE_OBSERVATION",
            ),
            "expected_output": dense_manifest,
            "expected_status": request["dense_impl"]["producer_pending_status"],
        },
        {
            "name": "DENSE_GRAPHICS_GEOMETRY_COURT",
            "command": py(
                request["dense_impl"]["independent_court"],
                "--dense-graphics-manifest", dense_manifest,
                "--output", dense_court,
            ),
            "expected_output": dense_court,
            "expected_status": request["dense_impl"]["required_court_status"],
        },
        {
            "name": "DENSE_GRAPHICS_EXECUTION_RECEIPT",
            "command": py(
                "tools/corpus/seal_truth139_materialization_receipt_v1.py",
                "--work-orders", p["work_orders"],
                "--subject-id", subject_id,
                "--job-name", DENSE_JOB,
                "--producer-manifest", dense_manifest,
                "--court-result", dense_court,
                "--code-commit-sha", code_sha,
                "--execution-kind", "EXTERNAL_GPU_WORKER_PLUS_CPU_COURT",
                "--output", dense_receipt,
            ),
            "expected_output": dense_receipt,
            "expected_status": RECEIPT_PASS_STATUS,
        },
        {
            "name": "VISIBILITY_OCCLUSION_CROSS_VIEW",
            "command": py(
                request["visibility_impl"]["producer"],
                "--subject-id", subject_id,
                "--graphics-manifest", dense_manifest,
                "--camera-bundle", p["camera_bundle"],
                "--output-dir", visibility_dir,
            ),
            "expected_output": visibility_manifest,
            "expected_status": request["visibility_impl"]["producer_pending_status"],
        },
        {
            "name": "VISIBILITY_OCCLUSION_CPU_COURT",
            "command": py(
                request["visibility_impl"]["independent_court"],
                "--teacher-manifest", visibility_manifest,
                "--output", visibility_court,
            ),
            "expected_output": visibility_court,
            "expected_status": request["visibility_impl"]["required_court_status"],
        },
        {
            "name": "VISIBILITY_OCCLUSION_EXECUTION_RECEIPT",
            "command": py(
                "tools/corpus/seal_truth139_materialization_receipt_v1.py",
                "--work-orders", p["work_orders"],
                "--subject-id", subject_id,
                "--job-name", VISIBILITY_JOB,
                "--producer-manifest", visibility_manifest,
                "--court-result", visibility_court,
                "--code-commit-sha", code_sha,
                "--execution-kind", "EXTERNAL_GPU_WORKER_PLUS_CPU_COURT",
                "--output", visibility_receipt,
            ),
            "expected_output": visibility_receipt,
            "expected_status": RECEIPT_PASS_STATUS,
        },
    ]


def execute_plan(plan: list[dict], *, cwd: Path = REPO_ROOT) -> list[dict]:
    results = []
    for step in plan:
        output = Path(step["expected_output"])
        output.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        proc = subprocess.run(step["command"], cwd=str(cwd), text=True, capture_output=True)
        wall = time.monotonic() - started
        if proc.returncode != 0:
            raise RuntimeError(
                "TRUTH139_GRAPHICS_WORKER_STEP_FAILED:"
                + step["name"]
                + "\nSTDOUT\n"
                + proc.stdout[-12000:]
                + "\nSTDERR\n"
                + proc.stderr[-12000:]
            )
        payload = _json(output, "TRUTH139_GRAPHICS_WORKER_STEP_OUTPUT")
        if payload.get("status") != step["expected_status"]:
            raise ValueError(
                f"TRUTH139_GRAPHICS_WORKER_STEP_STATUS:{step['name']}:"
                f"{payload.get('status')}!={step['expected_status']}"
            )
        results.append(
            {
                "name": step["name"],
                "command": step["command"],
                "output": str(output),
                "output_sha256": file_sha256(output),
                "status": payload["status"],
                "wall_seconds": wall,
                "stdout_tail": proc.stdout[-4000:],
            }
        )
    return results


def seal_chain(*, request: dict, subject_id: str, steps: list[dict], output: Path) -> dict:
    if request["bindings"]["code_commit_sha"] != checked_out_git_sha(REPO_ROOT):
        raise ValueError("TRUTH139_GRAPHICS_WORKER_CHECKOUT_CHANGED_DURING_EXECUTION")
    payload = {
        "schema": CHAIN_SCHEMA,
        "date": "2026-10-29",
        "status": CHAIN_STATUS,
        "subject_id": subject_id,
        "bindings": request["bindings"],
        "steps": steps,
        "derived_truth_execution_completed": True,
        "independent_courts_passed": True,
        "derived_truth_admitted": False,
        "training_performed": False,
        "split_created": False,
        "qualification_minted": False,
    }
    payload["chain_identity_sha256"] = _digest(payload)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = output.with_suffix(output.suffix + ".partial")
    staged.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    staged.replace(output)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-orders", required=True, type=Path)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--raw-source", required=True, type=Path)
    parser.add_argument("--geometry-npz", required=True, type=Path)
    parser.add_argument("--source-scene-snapshot", required=True, type=Path)
    parser.add_argument("--camera-bundle", required=True, type=Path)
    parser.add_argument("--observation-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--code-commit-sha", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    request = validate_request(
        work_orders_path=args.work_orders,
        subject_id=args.subject_id,
        raw_source_path=args.raw_source,
        geometry_npz=args.geometry_npz,
        source_scene_snapshot=args.source_scene_snapshot,
        camera_bundle=args.camera_bundle,
        observation_dir=args.observation_dir,
        output_root=args.output_root,
        code_commit_sha=args.code_commit_sha,
    )
    plan = build_command_plan(request, subject_id=args.subject_id, device=args.device)
    if args.plan_only:
        print(json.dumps({
            "status": "PASS_TRUTH139_GRAPHICS_MATERIALIZATION_PLAN_ONLY",
            "subject_id": args.subject_id,
            "bindings": request["bindings"],
            "steps": [{**step, "expected_output": str(step["expected_output"])} for step in plan],
        }, sort_keys=True, default=str), flush=True)
        return

    steps = execute_plan(plan)
    chain_path = args.output_root.resolve() / "GRAPHICS_MATERIALIZATION_CHAIN.json"
    chain = seal_chain(request=request, subject_id=args.subject_id, steps=steps, output=chain_path)
    print(json.dumps({
        "status": chain["status"],
        "subject_id": args.subject_id,
        "chain_path": str(chain_path),
        "chain_sha256": file_sha256(chain_path),
        "chain_identity_sha256": chain["chain_identity_sha256"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
