#!/usr/bin/env python3
from __future__ import annotations

"""Isolated exact-V8 direct-surface materialization for one FIT8 subject.

This tool changes no scientific surface semantics.  It executes the same sealed
V8 scalar-field -> SparseRegularTetraPolicyV5 T512/T1024 decoder used by
``materialize_fit8_direct_surface_v1``.  The only purpose of the V2 subject lane
is process isolation and durable progress/failure observability so an external
process kill cannot erase which subject/stage was active.

No teacher truth, TESSA, optimizer or backward pass is accepted here.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import traceback
from typing import Any

import torch

from models.iris.v5.indexed_sparse_tetra_decoder_v5 import SparseRegularTetraPolicyV5
from models.iris.vnext.cohort_v1 import SUBJECT_IDS
from models.iris.vnext.cohort_v9 import V8_CHECKPOINT_SHA256
from models.iris.vnext.sparse_surface_v3 import (
    VNextSparseSurfaceQueryPolicyV3,
    decode_vnext_sparse_surface_v3,
    write_direct_iris_surface_bundle_v1,
)
from tools.iris_vnext.materialize_fit8_direct_surface_v1 import (
    assert_full_sparse_policy,
    git_head,
    load_exact_v8_model,
    load_json,
    load_subject_source,
    scene_binding_sha256,
    sha256_file,
    tensor_sha256,
)

EXECUTION_SCHEMA = "RealSaS.Fit8DirectIRISSurfaceSubjectExecution.v2"
PROGRESS_SCHEMA = "RealSaS.Fit8DirectIRISSurfaceSubjectProgress.v2"
FAILURE_SCHEMA = "RealSaS.Fit8DirectIRISSurfaceSubjectFailure.v2"


def _atomic_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".staged",
        delete=False,
    ) as handle:
        staged = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    try:
        os.replace(staged, path)
    finally:
        staged.unlink(missing_ok=True)


def _progress(
    subject_root: Path,
    *,
    subject_id: str,
    execution_source_sha: str,
    stage: str,
    extra: dict[str, Any] | None = None,
) -> None:
    value: dict[str, Any] = {
        "schema": PROGRESS_SCHEMA,
        "subject_id": subject_id,
        "execution_source_sha": execution_source_sha,
        "stage": stage,
        "pid": int(os.getpid()),
        "unix_time": float(time.time()),
        "decoder_id": "RealSaS.ZeroSurfaceDecoder.SparseRegularT512T1024IndexedMT.v5",
        "base_cells": 512,
        "fine_cells": 1024,
        "optimizer_steps_executed": 0,
        "backward_steps_executed": 0,
        "teacher_truth_used": False,
        "tessa_invoked": False,
        "scientific_pass_claimed": False,
    }
    if extra:
        value.update(extra)
    _atomic_json(value, subject_root / "DIRECT_IRIS_SURFACE_PROGRESS_V2.json")


def materialize_subject(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    expected_source_sha = str(args.expected_source_sha).strip().lower()
    if git_head(repo_root) != expected_source_sha:
        raise RuntimeError("DIRECT_SURFACE_V2_SUBJECT_CHECKOUT_SHA_DRIFT")
    if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
        raise RuntimeError("DIRECT_SURFACE_V2_SUBJECT_CUDA_REQUIRED")

    subject_id = str(args.subject_id)
    if subject_id not in SUBJECT_IDS:
        raise ValueError("DIRECT_SURFACE_V2_SUBJECT_ID_INVALID:" + subject_id)

    sparse_policy = SparseRegularTetraPolicyV5()
    assert_full_sparse_policy(sparse_policy)
    query_policy = VNextSparseSurfaceQueryPolicyV3(
        model_query_chunk=int(args.model_query_chunk)
    )
    query_policy.validate()

    prepared_root = Path(args.prepared_root).resolve()
    manifest_path = Path(args.prepared_manifest).resolve()
    manifest = load_json(manifest_path)
    if manifest.get("schema") != "RealSaS.IRISVNextFit8Manifest.v4":
        raise ValueError("DIRECT_SURFACE_V2_SUBJECT_PREPARED_SCHEMA_INVALID")
    rows = list(manifest.get("subjects") or ())
    if tuple(row.get("subject_id") for row in rows) != SUBJECT_IDS:
        raise ValueError("DIRECT_SURFACE_V2_SUBJECT_PREPARED_ORDER_DRIFT")
    row = next(r for r in rows if str(r["subject_id"]) == subject_id)
    shard_path = (prepared_root / str(row["shard_file"])).resolve()
    if not shard_path.is_relative_to(prepared_root):
        raise ValueError("DIRECT_SURFACE_V2_SUBJECT_SHARD_PATH_ESCAPE")
    shard_sha = str(row["shard_sha256"]).lower()
    if sha256_file(shard_path) != shard_sha:
        raise ValueError("DIRECT_SURFACE_V2_SUBJECT_SHARD_SHA_DRIFT")

    checkpoint_path = Path(args.v8_checkpoint).resolve()
    if sha256_file(checkpoint_path) != V8_CHECKPOINT_SHA256:
        raise ValueError("DIRECT_SURFACE_V2_SUBJECT_V8_CHECKPOINT_SHA_DRIFT")

    output_root = Path(args.output_root).resolve()
    subject_root = output_root / "subjects" / subject_id
    subject_root.mkdir(parents=True, exist_ok=True)
    work_root = subject_root / ".work"
    shutil.rmtree(work_root, ignore_errors=True)
    work_root.mkdir(parents=True, exist_ok=True)

    _progress(
        subject_root,
        subject_id=subject_id,
        execution_source_sha=expected_source_sha,
        stage="LOAD_EXACT_V8_AND_SOURCE",
        extra={
            "prepared_shard_sha256": shard_sha,
            "v8_checkpoint_sha256": V8_CHECKPOINT_SHA256,
        },
    )

    device = torch.device(args.device)
    model = load_exact_v8_model(checkpoint_path, device)
    source = load_subject_source(shard_path, shard_sha, device)

    _progress(
        subject_root,
        subject_id=subject_id,
        execution_source_sha=expected_source_sha,
        stage="ENCODE_SCENE",
    )
    with torch.inference_mode():
        scene_planes = model.encode_scene(
            source["image_tokens"], source["token_xy"], source["camera_features"]
        ).to(dtype=torch.float32).detach()
    if scene_planes.shape[0] != 1 or scene_planes.shape[1] != 3:
        raise ValueError("DIRECT_SURFACE_V2_SUBJECT_SCENE_SHAPE_INVALID")
    if not bool(torch.isfinite(scene_planes).all().item()):
        raise ValueError("DIRECT_SURFACE_V2_SUBJECT_SCENE_NONFINITE")

    scene_hash = scene_binding_sha256(
        subject_id=subject_id,
        prepared_shard_sha256=shard_sha,
        scene_planes=scene_planes,
        execution_source_sha=expected_source_sha,
    )
    scene_planes_sha = tensor_sha256(scene_planes)

    _progress(
        subject_root,
        subject_id=subject_id,
        execution_source_sha=expected_source_sha,
        stage="DECODE_T512_T1024_STARTED",
        extra={
            "scene_binding_sha256": scene_hash,
            "scene_planes_sha256": scene_planes_sha,
        },
    )
    started = time.time()
    mesh = decode_vnext_sparse_surface_v3(
        model,
        scene_planes,
        sparse_policy=sparse_policy,
        query_policy=query_policy,
        work_dir=work_root,
    )
    decode_seconds = float(time.time() - started)

    _progress(
        subject_root,
        subject_id=subject_id,
        execution_source_sha=expected_source_sha,
        stage="PERSIST_EXACT_SURFACE",
        extra={
            "decode_seconds": decode_seconds,
            "decoder_diagnostics": dict(mesh.diagnostics),
        },
    )
    receipt = write_direct_iris_surface_bundle_v1(
        mesh,
        subject_root,
        subject_id=subject_id,
        source_commit=expected_source_sha,
        checkpoint_sha256=V8_CHECKPOINT_SHA256,
        scene_binding_sha256=scene_hash,
        sparse_policy=sparse_policy,
    )
    receipt_path = subject_root / "DIRECT_IRIS_SURFACE_RECEIPT.json"
    stage_path = subject_root / "DIRECT_IRIS_SURFACE.stage12.npz"

    result = {
        "schema": EXECUTION_SCHEMA,
        "status": "PASS__EXACT_T512_T1024_SUBJECT_MATERIALIZED__NO_TRAINING",
        "subject_id": subject_id,
        "execution_source_sha": expected_source_sha,
        "prepared_manifest_sha256": sha256_file(manifest_path),
        "prepared_shard_file": str(row["shard_file"]),
        "prepared_shard_sha256": shard_sha,
        "v8_checkpoint_sha256": V8_CHECKPOINT_SHA256,
        "scene_binding_sha256": scene_hash,
        "scene_planes_sha256": scene_planes_sha,
        "candidate_binding_hash": receipt.candidate_binding_hash,
        "stage12_npz_sha256": sha256_file(stage_path),
        "receipt_sha256": sha256_file(receipt_path),
        "vertex_count": int(receipt.vertex_count),
        "face_count": int(receipt.face_count),
        "field_query_api": mesh.diagnostics.get("vnext_field_query_api"),
        "decoder_id": str(receipt.decoder_id),
        "base_cells": int(receipt.base_cells),
        "fine_cells": int(receipt.fine_cells),
        "decode_seconds": decode_seconds,
        "decoder_diagnostics": dict(mesh.diagnostics),
        "teacher_truth_used": False,
        "tessa_invoked": False,
        "optimizer_steps_executed": 0,
        "backward_steps_executed": 0,
        "compiler_qualified": False,
        "scientific_pass_claimed": False,
    }
    _atomic_json(result, subject_root / "DIRECT_IRIS_SURFACE_SUBJECT_EXECUTION_V2.json")
    shutil.rmtree(work_root, ignore_errors=True)
    _progress(
        subject_root,
        subject_id=subject_id,
        execution_source_sha=expected_source_sha,
        stage="PASS__SUBJECT_COMPLETE",
        extra={
            "stage12_npz_sha256": result["stage12_npz_sha256"],
            "execution_receipt_sha256": sha256_file(
                subject_root / "DIRECT_IRIS_SURFACE_SUBJECT_EXECUTION_V2.json"
            ),
        },
    )
    del mesh, source, scene_planes, model
    torch.cuda.empty_cache()
    return result


def _write_failure(args: argparse.Namespace, exc: BaseException) -> Path:
    subject_root = Path(args.output_root).resolve() / "subjects" / str(args.subject_id)
    subject_root.mkdir(parents=True, exist_ok=True)
    value = {
        "schema": FAILURE_SCHEMA,
        "status": "FAIL__CATCHABLE_SUBJECT_EXCEPTION__NO_TRAINING",
        "subject_id": str(args.subject_id),
        "execution_source_sha": str(args.expected_source_sha).strip().lower(),
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "traceback": traceback.format_exc(),
        "pid": int(os.getpid()),
        "unix_time": float(time.time()),
        "optimizer_steps_executed": 0,
        "backward_steps_executed": 0,
        "scientific_pass_claimed": False,
    }
    path = subject_root / "DIRECT_IRIS_SURFACE_FAILURE_V2.json"
    _atomic_json(value, path)
    return path


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", required=True)
    p.add_argument("--expected-source-sha", required=True)
    p.add_argument("--subject-id", required=True, choices=SUBJECT_IDS)
    p.add_argument("--prepared-root", required=True)
    p.add_argument("--prepared-manifest", required=True)
    p.add_argument("--v8-checkpoint", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--model-query-chunk", type=int, default=32768)
    return p


def main() -> int:
    args = parser().parse_args()
    try:
        result = materialize_subject(args)
    except Exception as exc:
        path = _write_failure(args, exc)
        print(f"DIRECT_SURFACE_SUBJECT_FAILURE receipt={path}", flush=True)
        traceback.print_exc()
        return 3
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
