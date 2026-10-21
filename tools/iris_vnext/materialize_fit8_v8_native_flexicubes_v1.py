#!/usr/bin/env python3
from __future__ import annotations

"""Materialize the exact V8-trained native FlexiCubes witness at R24.

This is a measurement control, not the post-V8 high-resolution A-arm.  It uses
exactly the surface path that received V8 G2 extraction supervision:

    field + learned extraction_delta + learned 21-D cube parameters -> FlexiCubes

The output can distinguish a raw-field sparse-transition failure from a failure of
the V8-trained extraction witness itself.  No optimizer step, teacher topology,
TESSA invocation, Compiler qualification or high-resolution capability claim is
permitted here.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.iris.vnext.cohort_v1 import SUBJECT_IDS
from models.iris.vnext.cohort_v9 import (
    V8_BINDING_SHA256,
    V8_CHECKPOINT_SHA256,
    V8_GLOBAL_STEP,
    V8_SOURCE_COMMIT,
)
from models.iris.vnext.flexicubes_v1 import (
    FLEXICUBES_COMMIT,
    PinnedFlexiCubesV1,
    vertex_normals_v1,
)
from tools.iris_vnext.materialize_fit8_direct_surface_v1 import (
    EXPECTED_PREPARED_SCHEMA,
    git_head,
    load_exact_v8_model,
    load_json,
    load_subject_source,
    scene_binding_sha256,
    sha256_file,
    tensor_sha256,
    value_sha256,
)

SCHEMA = "RealSaS.Fit8V8NativeFlexiCubesMaterialization.v1"
SUBJECT_SCHEMA = "RealSaS.V8NativeFlexiCubesWitness.v1"
MATERIALIZER_ID = "RealSaS.Fit8V8NativeFlexiCubesMaterializer.v1"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_witness(path: Path, vertices: torch.Tensor, faces: torch.Tensor) -> None:
    normals = vertex_normals_v1(vertices, faces)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        vertices_normalized=vertices.detach().cpu().float().numpy(),
        faces=faces.detach().cpu().to(dtype=torch.int32).numpy(),
        vertex_normals=normals.detach().cpu().float().numpy(),
    )


def _raw_cube_parameters(
    model: torch.nn.Module,
    h: torch.Tensor,
    cubes: torch.Tensor,
    *,
    chunk: int,
) -> torch.Tensor:
    if int(chunk) <= 0:
        raise ValueError("NATIVE_V8_FLEXICUBES_CUBE_CHUNK_INVALID")
    rows = []
    for start in range(0, int(cubes.shape[0]), int(chunk)):
        cube = cubes[start : start + int(chunk)]
        rows.append(model.scene_core.cube_extraction(h[:, cube]))
    raw = torch.cat(rows, dim=1)
    if raw.shape != (1, int(cubes.shape[0]), 21):
        raise RuntimeError("NATIVE_V8_FLEXICUBES_PARAMETER_SHAPE_DRIFT")
    return raw


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    expected_source_sha = str(args.expected_source_sha).strip().lower()
    if git_head(repo_root) != expected_source_sha:
        raise RuntimeError("NATIVE_V8_FLEXICUBES_CHECKOUT_SHA_DRIFT")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("NATIVE_V8_FLEXICUBES_REQUIRES_CUDA")

    manifest_path = Path(args.prepared_manifest).resolve()
    prepared_root = Path(args.prepared_root).resolve()
    manifest = load_json(manifest_path)
    if manifest.get("schema") != EXPECTED_PREPARED_SCHEMA:
        raise ValueError("NATIVE_V8_FLEXICUBES_PREPARED_SCHEMA_INVALID")
    rows = list(manifest.get("subjects") or ())
    if tuple(row.get("subject_id") for row in rows) != SUBJECT_IDS:
        raise ValueError("NATIVE_V8_FLEXICUBES_SUBJECT_ORDER_DRIFT")

    model = load_exact_v8_model(Path(args.v8_checkpoint), device)
    resolution = int(model.config.extraction_training_resolution)
    if resolution != 24:
        raise ValueError("NATIVE_V8_FLEXICUBES_EXPECTED_R24")
    extractor = PinnedFlexiCubesV1(
        Path(args.flexicubes_root),
        device=device,
        weight_scale=model.config.flexicubes_weight_scale,
    )
    if extractor.identity.get("commit") != FLEXICUBES_COMMIT:
        raise RuntimeError("NATIVE_V8_FLEXICUBES_PIN_DRIFT")

    output_root = Path(args.output_dir).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError("NATIVE_V8_FLEXICUBES_OUTPUT_DIR_MUST_BE_EMPTY")
    output_root.mkdir(parents=True, exist_ok=True)

    base_vertices, cubes = extractor.construct_iris_grid(resolution)
    base_vertices = base_vertices.to(device=device, dtype=torch.float32)
    cubes = cubes.to(device=device)
    grid = base_vertices.unsqueeze(0)

    subject_rows: list[dict[str, Any]] = []
    for row in rows:
        subject_id = str(row["subject_id"])
        shard_path = (prepared_root / str(row["shard_file"])).resolve()
        if not shard_path.is_relative_to(prepared_root):
            raise ValueError("NATIVE_V8_FLEXICUBES_SHARD_PATH_ESCAPE")
        shard_sha = str(row["shard_sha256"]).lower()
        source = load_subject_source(shard_path, shard_sha, device)

        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            planes = model.encode_scene(
                source["image_tokens"], source["token_xy"], source["camera_features"]
            ).float()
            evidence = model.query_evidence(planes, grid)
            field = evidence["field"].float()
            delta = evidence["extraction_delta"].float()
            h = evidence["H"].float()
            deformed = grid + delta
            raw = _raw_cube_parameters(model, h, cubes, chunk=int(args.cube_chunk)).float()
            vertices, faces, l_dev = extractor.extract(
                vertices_iris=deformed[0],
                field=field[0],
                cubes=cubes,
                raw_parameters=raw[0],
                resolution=resolution,
                training=False,
            )

        if not bool(torch.isfinite(vertices).all().item()):
            raise RuntimeError("NATIVE_V8_FLEXICUBES_VERTEX_NONFINITE")
        if not bool(torch.isfinite(field).all().item()):
            raise RuntimeError("NATIVE_V8_FLEXICUBES_FIELD_NONFINITE")

        scene_hash = scene_binding_sha256(
            subject_id=subject_id,
            prepared_shard_sha256=shard_sha,
            scene_planes=planes,
            execution_source_sha=expected_source_sha,
        )
        subject_root = output_root / "subjects" / subject_id
        witness_path = subject_root / "V8_NATIVE_FLEXICUBES_R24.npz"
        _write_witness(witness_path, vertices, faces)

        receipt = {
            "schema": SUBJECT_SCHEMA,
            "materializer_id": MATERIALIZER_ID,
            "subject_id": subject_id,
            "execution_source_sha": expected_source_sha,
            "prepared_shard_sha256": shard_sha,
            "scene_binding_sha256": scene_hash,
            "scene_planes_sha256": tensor_sha256(planes),
            "v8_authority": {
                "source_commit": V8_SOURCE_COMMIT,
                "checkpoint_sha256": V8_CHECKPOINT_SHA256,
                "binding_sha256": V8_BINDING_SHA256,
                "global_step": V8_GLOBAL_STEP,
            },
            "surface_semantics": "EXACT_V8_TRAINED_FLEXICUBES_PATH_AT_NATIVE_TRAINING_RESOLUTION",
            "resolution": resolution,
            "flexicubes": extractor.identity,
            "field_min": float(field.min().item()),
            "field_max": float(field.max().item()),
            "field_zero_bracket": bool(field.min().item() < 0.0 < field.max().item()),
            "extraction_delta_abs_mean": float(delta.abs().mean().item()),
            "extraction_delta_abs_max": float(delta.abs().max().item()),
            "extraction_delta_limit": float(model.config.maximum_extraction_delta),
            "raw_cube_parameter_abs_mean": float(raw.abs().mean().item()),
            "l_dev_mean": float(l_dev.mean().item()) if l_dev.numel() else 0.0,
            "vertex_count": int(vertices.shape[0]),
            "face_count": int(faces.shape[0]),
            "witness_file": witness_path.name,
            "witness_sha256": sha256_file(witness_path),
            "numeric_policy": "FP32_NO_AUTOCAST",
            "teacher_truth_used_in_inference_forward": False,
            "tessa_invoked": False,
            "optimizer_steps_executed": 0,
            "high_resolution_capability_claimed": False,
            "compiler_qualified": False,
            "scientific_pass_claimed": False,
        }
        receipt["binding_sha256"] = value_sha256(receipt)
        receipt_path = subject_root / "V8_NATIVE_FLEXICUBES_R24_RECEIPT.json"
        _write_json(receipt_path, receipt)
        subject_rows.append(
            {
                "subject_id": subject_id,
                "scene_binding_sha256": scene_hash,
                "witness_sha256": receipt["witness_sha256"],
                "receipt_sha256": sha256_file(receipt_path),
                "vertex_count": receipt["vertex_count"],
                "face_count": receipt["face_count"],
            }
        )
        del source, planes, evidence, field, delta, h, deformed, raw, vertices, faces, l_dev
        torch.cuda.empty_cache()

    result = {
        "schema": SCHEMA,
        "status": "MATERIALIZED_NATIVE_V8_FLEXICUBES_WITNESSES__MEASUREMENT_ONLY",
        "materializer_id": MATERIALIZER_ID,
        "execution_source_sha": expected_source_sha,
        "prepared_manifest_sha256": sha256_file(manifest_path),
        "subject_order": list(SUBJECT_IDS),
        "subjects": subject_rows,
        "surface_semantics": "EXACT_V8_TRAINED_FLEXICUBES_PATH_AT_NATIVE_R24",
        "high_resolution_capability_claimed": False,
        "tessa_invoked": False,
        "optimizer_steps_executed": 0,
        "compiler_court_executed": False,
        "training_authorized": False,
        "scientific_pass_claimed": False,
        "next_action": "RUN_NATIVE_V8_GEOMETRY_AND_EXACT_STAGE01_MASK_MEASUREMENT_CONTROL",
    }
    result["binding_sha256"] = value_sha256(result)
    _write_json(output_root / "FIT8_V8_NATIVE_FLEXICUBES_BATCH_RECEIPT.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", required=True)
    p.add_argument("--expected-source-sha", required=True)
    p.add_argument("--prepared-root", required=True)
    p.add_argument("--prepared-manifest", required=True)
    p.add_argument("--v8-checkpoint", required=True)
    p.add_argument("--flexicubes-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--cube-chunk", type=int, default=2048)
    return p


def main() -> int:
    result = materialize(parser().parse_args())
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
