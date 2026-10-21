#!/usr/bin/env python3
from __future__ import annotations

"""Materialize the TESSA-free Fit8 surface baseline from the exact completed V8.

This is an inference/evidence lane only. It performs zero optimizer steps, never
uses teacher topology, never invokes TESSA, and cannot qualify a mesh for product
use. Every subject is decoded with the selected full T512 -> sparse T1024 policy.
The resulting Stage12-like mesh candidates are hash-bound for later independent
Compiler/source-fidelity courts.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import numpy as np
import torch

from models.iris.v5.indexed_sparse_tetra_decoder_v5 import SparseRegularTetraPolicyV5
from models.iris.vnext.cohort_v1 import SUBJECT_IDS
from models.iris.vnext.cohort_v9 import (
    V8_BINDING_SHA256,
    V8_CHECKPOINT_SHA256,
    V8_GLOBAL_STEP,
    V8_SOURCE_COMMIT,
)
from models.iris.vnext.contracts_v1 import IRISVNextConfigV1
from models.iris.vnext.model_v1 import IRISVNextV1
from models.iris.vnext.sparse_surface_v3 import (
    VNextSparseSurfaceQueryPolicyV3,
    decode_vnext_sparse_surface_v3,
    write_direct_iris_surface_bundle_v1,
)


BATCH_SCHEMA = "RealSaS.Fit8DirectIRISSurfaceMaterialization.v1"
EXPECTED_PREPARED_SCHEMA = "RealSaS.IRISVNextFit8Manifest.v4"
SOURCE_KEYS = ("image_tokens", "token_xy", "camera_features")
EXPECTED_SOURCE_RANKS = {"image_tokens": 4, "token_xy": 4, "camera_features": 3}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    h = hashlib.sha256()
    h.update(str(cpu.dtype).encode("utf-8"))
    h.update(json.dumps(list(cpu.shape), separators=(",", ":")).encode("utf-8"))
    h.update(cpu.numpy().tobytes(order="C"))
    return h.hexdigest()


def value_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"DIRECT_SURFACE_JSON_OBJECT_REQUIRED:{path}")
    return value


def git_head(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip().lower()


def assert_full_sparse_policy(policy: SparseRegularTetraPolicyV5) -> None:
    policy.validate()
    reference = SparseRegularTetraPolicyV5()
    if int(policy.base_cells) != 512 or int(policy.fine_cells) != 1024:
        raise ValueError("DIRECT_SURFACE_FULL_T512_T1024_POLICY_REQUIRED")
    if policy != reference:
        raise ValueError("DIRECT_SURFACE_REFERENCE_SPARSE_POLICY_DRIFT")


def load_exact_v8_model(checkpoint_path: Path, device: torch.device) -> IRISVNextV1:
    checkpoint_path = checkpoint_path.resolve()
    if sha256_file(checkpoint_path) != V8_CHECKPOINT_SHA256:
        raise ValueError("DIRECT_SURFACE_V8_CHECKPOINT_SHA_DRIFT")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if int(payload.get("step", -1)) != V8_GLOBAL_STEP:
        raise ValueError("DIRECT_SURFACE_V8_STEP_DRIFT")
    if payload.get("binding_sha256") != V8_BINDING_SHA256:
        raise ValueError("DIRECT_SURFACE_V8_BINDING_SHA_DRIFT")
    binding = payload.get("binding") or {}
    if binding.get("schema") != "RealSaS.IRISVNextFit8Binding.v10":
        raise ValueError("DIRECT_SURFACE_V8_BINDING_SCHEMA_INVALID")
    scene_authority = str((binding.get("post_m1_parameter_authority") or {}).get("scene_core", ""))
    if not scene_authority.startswith("SEALED_LR_ZERO"):
        raise ValueError("DIRECT_SURFACE_V8_SCENE_CORE_NOT_SEALED")

    model = IRISVNextV1(IRISVNextConfigV1())
    model.load_state_dict(payload["model"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    model.to(device=device, dtype=torch.float32)
    return model


def load_subject_source(shard_path: Path, expected_sha256: str, device: torch.device) -> dict[str, torch.Tensor]:
    if sha256_file(shard_path) != str(expected_sha256).lower():
        raise ValueError("DIRECT_SURFACE_PREPARED_SHARD_SHA_DRIFT:" + shard_path.name)
    payload = torch.load(shard_path, map_location="cpu", weights_only=True)
    source: dict[str, torch.Tensor] = {}
    for key in SOURCE_KEYS:
        tensor = payload.get(key)
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("DIRECT_SURFACE_SOURCE_FIELD_MISSING:" + key)
        expected_rank = EXPECTED_SOURCE_RANKS[key]
        if tensor.ndim == expected_rank - 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim != expected_rank or tensor.shape[0] != 1:
            raise ValueError("DIRECT_SURFACE_SOURCE_SHAPE_INVALID:" + key)
        if not torch.is_floating_point(tensor) or not bool(torch.isfinite(tensor).all().item()):
            raise ValueError("DIRECT_SURFACE_SOURCE_TENSOR_INVALID:" + key)
        source[key] = tensor.to(device=device, dtype=torch.float32).contiguous()
    if source["image_tokens"].shape[:2] != (1, 8):
        raise ValueError("DIRECT_SURFACE_VIEW_COUNT_INVALID:image_tokens")
    if source["token_xy"].shape[:2] != (1, 8):
        raise ValueError("DIRECT_SURFACE_VIEW_COUNT_INVALID:token_xy")
    if source["camera_features"].shape[:2] != (1, 8):
        raise ValueError("DIRECT_SURFACE_VIEW_COUNT_INVALID:camera_features")
    return source


def scene_binding_sha256(
    *,
    subject_id: str,
    prepared_shard_sha256: str,
    scene_planes: torch.Tensor,
    execution_source_sha: str,
) -> str:
    binding = {
        "schema": "RealSaS.DirectIRISSceneBinding.v1",
        "subject_id": str(subject_id),
        "execution_source_sha": str(execution_source_sha),
        "v8_source_commit": V8_SOURCE_COMMIT,
        "v8_checkpoint_sha256": V8_CHECKPOINT_SHA256,
        "v8_binding_sha256": V8_BINDING_SHA256,
        "v8_global_step": V8_GLOBAL_STEP,
        "prepared_shard_sha256": str(prepared_shard_sha256).lower(),
        "scene_planes_sha256": tensor_sha256(scene_planes),
        "scene_numeric_policy": "FP32_NO_AUTOCAST",
    }
    return value_sha256(binding)


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    expected_source_sha = str(args.expected_source_sha).strip().lower()
    if len(expected_source_sha) != 40 or any(c not in "0123456789abcdef" for c in expected_source_sha):
        raise ValueError("DIRECT_SURFACE_EXECUTION_SOURCE_SHA_INVALID")
    actual_source_sha = git_head(repo_root)
    if actual_source_sha != expected_source_sha:
        raise RuntimeError(f"DIRECT_SURFACE_CHECKOUT_SHA_DRIFT:{actual_source_sha}:{expected_source_sha}")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("DIRECT_SURFACE_FULL_RESOLUTION_REQUIRES_CUDA")

    sparse_policy = SparseRegularTetraPolicyV5()
    assert_full_sparse_policy(sparse_policy)
    query_policy = VNextSparseSurfaceQueryPolicyV3(model_query_chunk=int(args.model_query_chunk))
    query_policy.validate()

    prepared_root = Path(args.prepared_root).resolve()
    manifest_path = Path(args.prepared_manifest).resolve()
    manifest = load_json(manifest_path)
    if manifest.get("schema") != EXPECTED_PREPARED_SCHEMA:
        raise ValueError("DIRECT_SURFACE_PREPARED_MANIFEST_SCHEMA_INVALID")
    rows = list(manifest.get("subjects") or ())
    if tuple(row.get("subject_id") for row in rows) != SUBJECT_IDS:
        raise ValueError("DIRECT_SURFACE_SUBJECT_ORDER_DRIFT")

    checkpoint_path = Path(args.v8_checkpoint).resolve()
    model = load_exact_v8_model(checkpoint_path, device)

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("DIRECT_SURFACE_OUTPUT_DIR_MUST_BE_EMPTY")
    output_dir.mkdir(parents=True, exist_ok=True)
    work_root = output_dir / ".work"
    work_root.mkdir(parents=True, exist_ok=True)

    subject_results: list[dict[str, Any]] = []
    try:
        for row in rows:
            subject_id = str(row["subject_id"])
            shard_path = (prepared_root / str(row["shard_file"])).resolve()
            if not shard_path.is_relative_to(prepared_root):
                raise ValueError("DIRECT_SURFACE_PREPARED_SHARD_PATH_ESCAPE")
            shard_sha = str(row["shard_sha256"]).lower()
            source = load_subject_source(shard_path, shard_sha, device)

            with torch.inference_mode():
                scene_planes = model.encode_scene(
                    source["image_tokens"], source["token_xy"], source["camera_features"]
                ).to(dtype=torch.float32).detach()
            if scene_planes.shape[0] != 1 or scene_planes.shape[1] != 3:
                raise ValueError("DIRECT_SURFACE_SCENE_PLANES_SHAPE_INVALID")
            if not bool(torch.isfinite(scene_planes).all().item()):
                raise ValueError("DIRECT_SURFACE_SCENE_PLANES_NONFINITE")

            scene_hash = scene_binding_sha256(
                subject_id=subject_id,
                prepared_shard_sha256=shard_sha,
                scene_planes=scene_planes,
                execution_source_sha=expected_source_sha,
            )
            subject_work = work_root / subject_id
            if subject_work.exists():
                shutil.rmtree(subject_work)
            mesh = decode_vnext_sparse_surface_v3(
                model,
                scene_planes,
                sparse_policy=sparse_policy,
                query_policy=query_policy,
                work_dir=subject_work,
            )
            subject_out = output_dir / "subjects" / subject_id
            receipt = write_direct_iris_surface_bundle_v1(
                mesh,
                subject_out,
                subject_id=subject_id,
                source_commit=expected_source_sha,
                checkpoint_sha256=V8_CHECKPOINT_SHA256,
                scene_binding_sha256=scene_hash,
                sparse_policy=sparse_policy,
            )
            receipt_path = subject_out / "DIRECT_IRIS_SURFACE_RECEIPT.json"
            subject_results.append(
                {
                    "subject_id": subject_id,
                    "prepared_shard_file": str(row["shard_file"]),
                    "prepared_shard_sha256": shard_sha,
                    "scene_binding_sha256": scene_hash,
                    "scene_planes_sha256": tensor_sha256(scene_planes),
                    "candidate_binding_hash": receipt.candidate_binding_hash,
                    "stage12_npz_sha256": receipt.stage12_npz_sha256,
                    "receipt_sha256": sha256_file(receipt_path),
                    "vertex_count": receipt.vertex_count,
                    "face_count": receipt.face_count,
                    "field_query_api": mesh.diagnostics.get("vnext_field_query_api"),
                    "compiler_qualified": False,
                }
            )
            shutil.rmtree(subject_work, ignore_errors=True)
            del source, scene_planes, mesh
            torch.cuda.empty_cache()
    finally:
        shutil.rmtree(work_root, ignore_errors=True)

    batch = {
        "schema": BATCH_SCHEMA,
        "status": "MATERIALIZED_MODEL_SURFACES__COMPILER_COURT_PENDING",
        "execution_source_sha": expected_source_sha,
        "prepared_manifest_sha256": sha256_file(manifest_path),
        "v8_authority": {
            "source_commit": V8_SOURCE_COMMIT,
            "checkpoint_sha256": V8_CHECKPOINT_SHA256,
            "binding_sha256": V8_BINDING_SHA256,
            "global_step": V8_GLOBAL_STEP,
            "scene_core": "SEALED",
        },
        "numeric_policy": "FP32_NO_AUTOCAST",
        "sparse_surface_policy": {
            "base_cells": sparse_policy.base_cells,
            "fine_cells": sparse_policy.fine_cells,
            "policy": "EXACT_SELECTED_T512_TO_SPARSE_T1024_V5",
            "model_query_chunk": query_policy.model_query_chunk,
        },
        "subject_order": list(SUBJECT_IDS),
        "subjects": subject_results,
        "tessa_invoked": False,
        "teacher_truth_used": False,
        "optimizer_steps_executed": 0,
        "compiler_court_executed": False,
        "training_authorized": False,
        "scientific_pass_claimed": False,
        "next_action": "RUN_INDEPENDENT_SOURCE_FIDELITY_AND_COMPILER_COURTS",
    }
    if len(subject_results) != len(SUBJECT_IDS):
        raise RuntimeError("DIRECT_SURFACE_SUBJECT_MATERIALIZATION_INCOMPLETE")
    batch["binding_sha256"] = value_sha256(batch)
    batch_path = output_dir / "FIT8_DIRECT_IRIS_SURFACE_BATCH_RECEIPT.json"
    batch_path.write_text(json.dumps(batch, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
    args = parser().parse_args()
    result = materialize(args)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
