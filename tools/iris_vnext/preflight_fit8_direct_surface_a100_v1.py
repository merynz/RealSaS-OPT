#!/usr/bin/env python3
from __future__ import annotations

"""Engineering-only preflight for exact-V8 full-resolution surface materialization.

This probe never extracts a reduced mesh and never produces a quality verdict. It
loads the exact V8 checkpoint, encodes one real sealed Fit8 subject in FP32, then
benchmarks the exact field-query chunk used by the full T512 -> sparse T1024 lane.
Admission is about accelerator/disk/runtime feasibility only.
"""

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

import numpy as np
import torch

from models.iris.v5.indexed_sparse_tetra_decoder_v5 import SparseRegularTetraPolicyV5
from models.iris.vnext.sparse_surface_v3 import VNextScalarFieldQueryV3, VNextSparseSurfaceQueryPolicyV3
from tools.iris_vnext.materialize_fit8_direct_surface_v1 import (
    assert_full_sparse_policy,
    git_head,
    load_exact_v8_model,
    load_json,
    load_subject_source,
    sha256_file,
)
from tools.iris_vnext.hydrate_fit8_direct_surface_inputs_v1 import load_json as load_binding_json, validate_binding


SCHEMA = "RealSaS.IRISVNextDirectSurfaceA100Preflight.v1"
MINIMUM_CUDA_MEMORY_BYTES = 39 * 1024**3
MINIMUM_FREE_DISK_BYTES = 12 * 1024**3
MINIMUM_HOST_MEMORY_BYTES = 24 * 1024**3
PROBE_SUBJECT = "Snake"
PROBE_REPETITIONS = 3


def host_memory_bytes() -> int:
    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        return pages * page_size
    except (ValueError, OSError, AttributeError):
        return 0


def engineering_admission(
    *,
    cuda_memory_bytes: int,
    host_memory: int,
    free_disk_bytes: int,
    compute_capability_major: int,
) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    if int(cuda_memory_bytes) < MINIMUM_CUDA_MEMORY_BYTES:
        blockers.append("CUDA_MEMORY_BELOW_39_GIB")
    if int(host_memory) < MINIMUM_HOST_MEMORY_BYTES:
        blockers.append("HOST_MEMORY_BELOW_24_GIB")
    if int(free_disk_bytes) < MINIMUM_FREE_DISK_BYTES:
        blockers.append("FREE_DISK_BELOW_12_GIB")
    if int(compute_capability_major) < 8:
        blockers.append("CUDA_COMPUTE_CAPABILITY_BELOW_8")
    return not blockers, blockers


def _deterministic_probe_points(count: int) -> np.ndarray:
    # Engineering workload only. These points are not a scientific surface court.
    rng = np.random.default_rng(26091991)
    points = rng.uniform(-1.0, 1.0, size=(int(count), 3)).astype(np.float32)
    points[0] = 0.0
    points[1] = np.asarray([-1.0, -1.0, -1.0], dtype=np.float32)
    points[2] = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
    return points


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    expected_source_sha = str(args.expected_source_sha).lower()
    if git_head(repo_root) != expected_source_sha:
        raise RuntimeError("DIRECT_SURFACE_PREFLIGHT_CHECKOUT_SHA_DRIFT")

    if not torch.cuda.is_available():
        raise RuntimeError("DIRECT_SURFACE_PREFLIGHT_CUDA_REQUIRED")
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    cc = torch.cuda.get_device_capability(device)

    hydration_root = Path(args.hydration_root).resolve()
    binding_path = Path(args.binding).resolve()
    binding = load_binding_json(binding_path)
    validate_binding(binding)
    hydration_receipt = load_json(hydration_root / "DIRECT_SURFACE_HYDRATION_RECEIPT.json")
    if hydration_receipt.get("status") != "PASS__EXACT_INPUT_BYTES_HYDRATED__NO_TRAINING__NO_REMOTE_WRITE":
        raise RuntimeError("DIRECT_SURFACE_PREFLIGHT_HYDRATION_NOT_PASS")
    if hydration_receipt.get("input_binding_sha256") != sha256_file(binding_path):
        raise RuntimeError("DIRECT_SURFACE_PREFLIGHT_INPUT_BINDING_DRIFT")

    sparse_policy = SparseRegularTetraPolicyV5()
    assert_full_sparse_policy(sparse_policy)
    model_query_chunk = int(args.model_query_chunk)
    query_policy = VNextSparseSurfaceQueryPolicyV3(model_query_chunk=model_query_chunk)
    query_policy.validate()

    prepared = binding["prepared"]
    snake_spec = next(row for row in prepared["subjects"] if row["subject_id"] == PROBE_SUBJECT)
    snake_path = hydration_root / "prepared" / str(snake_spec["file"])
    checkpoint_path = hydration_root / "v8" / str(binding["v8"]["checkpoint"]["file"])

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = load_exact_v8_model(checkpoint_path, device)
    source = load_subject_source(snake_path, str(snake_spec["sha256"]), device)
    with torch.inference_mode():
        encode_start = time.perf_counter()
        scene_planes = model.encode_scene(
            source["image_tokens"], source["token_xy"], source["camera_features"]
        ).float().detach()
        torch.cuda.synchronize()
        encode_seconds = time.perf_counter() - encode_start

    adapter = VNextScalarFieldQueryV3(model, scene_planes, policy=query_policy)
    probe = _deterministic_probe_points(model_query_chunk)
    _ = adapter(probe)  # warmup
    torch.cuda.synchronize()
    elapsed: list[float] = []
    extrema: list[tuple[float, float]] = []
    for _ in range(PROBE_REPETITIONS):
        started = time.perf_counter()
        values = adapter(probe)
        torch.cuda.synchronize()
        elapsed.append(time.perf_counter() - started)
        extrema.append((float(values.min()), float(values.max())))
    seconds_per_chunk = float(np.median(np.asarray(elapsed, dtype=np.float64)))
    points_per_second = float(model_query_chunk / seconds_per_chunk)
    coarse_points = int((sparse_policy.base_cells + 1) ** 3)
    estimated_coarse_query_seconds = float(coarse_points / points_per_second)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(output_dir)
    host_mem = host_memory_bytes()
    admitted, blockers = engineering_admission(
        cuda_memory_bytes=int(props.total_memory),
        host_memory=host_mem,
        free_disk_bytes=int(disk.free),
        compute_capability_major=int(cc[0]),
    )

    result = {
        "schema": SCHEMA,
        "status": "PASS__ENGINEERING_ONLY__FULL_SURFACE_NOT_MATERIALIZED" if admitted else "BLOCKED__ENGINEERING_RESOURCES",
        "execution_source_sha": expected_source_sha,
        "input_binding_sha256": sha256_file(binding_path),
        "hydration_receipt_sha256": sha256_file(hydration_root / "DIRECT_SURFACE_HYDRATION_RECEIPT.json"),
        "probe_subject": PROBE_SUBJECT,
        "probe_semantics": "REAL_V8_SCENE__FIELD_QUERY_CHUNK_ONLY__NO_REDUCED_MESH",
        "accelerator": {
            "name": props.name,
            "cuda_total_memory_bytes": int(props.total_memory),
            "compute_capability": [int(cc[0]), int(cc[1])],
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
        "host": {
            "memory_bytes": host_mem,
            "free_disk_bytes": int(disk.free),
        },
        "timing": {
            "scene_encode_seconds": encode_seconds,
            "model_query_chunk": model_query_chunk,
            "chunk_repetitions": PROBE_REPETITIONS,
            "chunk_seconds_median": seconds_per_chunk,
            "points_per_second": points_per_second,
            "full_t512_coarse_query_points": coarse_points,
            "estimated_coarse_query_seconds_engineering_only": estimated_coarse_query_seconds,
        },
        "field_probe_extrema": [{"min": lo, "max": hi} for lo, hi in extrema],
        "full_surface_policy": {
            "base_cells": sparse_policy.base_cells,
            "fine_cells": sparse_policy.fine_cells,
            "decoder": "SparseRegularTetraPolicyV5",
        },
        "blockers": blockers,
        "optimizer_steps_executed": 0,
        "reduced_resolution_mesh_executed": False,
        "scientific_quality_evaluated": False,
        "scientific_pass_claimed": False,
        "training_authorized": False,
        "next_action": "RUN_FULL_T512_T1024_MATERIALIZER" if admitted else "RESOLVE_ENGINEERING_BLOCKERS",
    }
    output = output_dir / "DIRECT_SURFACE_A100_PREFLIGHT.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", required=True)
    p.add_argument("--expected-source-sha", required=True)
    p.add_argument("--binding", default="canonical/VNEXT_DIRECT_SURFACE_INPUT_BINDING_V1.json")
    p.add_argument("--hydration-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-query-chunk", type=int, default=32768)
    return p


def main() -> int:
    args = parser().parse_args()
    result = preflight(args)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    if result["status"].startswith("BLOCKED"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
