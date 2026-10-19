#!/usr/bin/env python3
from __future__ import annotations

"""Measurement-only forensic audit for the failed V8 raw-field sparse A-arm.

This tool consumes the preserved T512 coarse-field memmap from the staged
materializer. It does not load IRIS, does not query CUDA, does not change the
sparse decoder policy, and cannot turn a failed materialization into a PASS.

The historical sparse selector uses a field-magnitude band expressed in spatial
units. That is safe for a distance-like field only when field variation is
compatible with spatial grid steps. This audit therefore records sign-crossing
support, refinement-boundary structure and finite-difference roughness indicators.
The indicators are diagnostics, not an Eikonal/Lipschitz proof.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from models.iris.v5.indexed_sparse_tetra_decoder_v5 import (
    SparseRegularTetraPolicyV5,
    boundary_parent_cells_v5,
    select_refined_cells_v5,
)

SCHEMA = "RealSaS.V8RawFieldSparseCompatibilityAudit.v1"
EXPECTED_FAILURE_SCHEMA = "RealSaS.DirectIRISSurfaceMaterializationFailure.v2"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("RAW_FIELD_AUDIT_JSON_OBJECT_REQUIRED")
    return value


def _cell_corner_extrema(field: np.ndarray, *, slab: int = 8):
    base = field.shape[0] - 1
    sign_parts: list[np.ndarray] = []
    band_parts: list[np.ndarray] = []
    policy = SparseRegularTetraPolicyV5(base_cells=base, fine_cells=2 * base)
    band = float(policy.refine_band)
    for x0 in range(0, base, slab):
        x1 = min(base, x0 + slab)
        corners = (
            field[x0:x1, :-1, :-1], field[x0:x1, :-1, 1:],
            field[x0:x1, 1:, :-1], field[x0:x1, 1:, 1:],
            field[x0 + 1:x1 + 1, :-1, :-1], field[x0 + 1:x1 + 1, :-1, 1:],
            field[x0 + 1:x1 + 1, 1:, :-1], field[x0 + 1:x1 + 1, 1:, 1:],
        )
        minimum = np.asarray(corners[0]).copy()
        maximum = np.asarray(corners[0]).copy()
        minimum_abs = np.abs(corners[0]).copy()
        for corner in corners[1:]:
            np.minimum(minimum, corner, out=minimum)
            np.maximum(maximum, corner, out=maximum)
            np.minimum(minimum_abs, np.abs(corner), out=minimum_abs)
        sign = (minimum <= 0.0) & (maximum >= 0.0)
        band_only = (~sign) & (minimum_abs <= band)
        lx, yy, zz = np.nonzero(sign)
        if len(lx):
            sign_parts.append(np.stack((lx + x0, yy, zz), axis=1).astype(np.int32))
        lx, yy, zz = np.nonzero(band_only)
        if len(lx):
            band_parts.append(np.stack((lx + x0, yy, zz), axis=1).astype(np.int32))
    empty = np.empty((0, 3), dtype=np.int32)
    return (
        np.concatenate(sign_parts, axis=0) if sign_parts else empty,
        np.concatenate(band_parts, axis=0) if band_parts else empty,
    )


def _edge_roughness(
    field: np.ndarray,
    spatial_step: float,
    *,
    slab: int = 8,
) -> dict[str, Any]:
    """Stream grid-edge diagnostics without allocating a full 513^3 delta array."""
    if int(slab) <= 0:
        raise ValueError("RAW_FIELD_AUDIT_EDGE_SLAB_INVALID")
    thresholds = (1.0, 2.0, 4.0, 8.0)
    total_edges = 0
    sign_flip_edges = 0
    exceed = {factor: 0 for factor in thresholds}
    maximum_ratio = 0.0
    mean_abs_accum = 0.0
    n = int(field.shape[0])
    if field.shape != (n, n, n) or n < 2:
        raise ValueError("RAW_FIELD_AUDIT_FIELD_SHAPE_INVALID")

    def consume(a: np.ndarray, b: np.ndarray) -> None:
        nonlocal total_edges, sign_flip_edges, maximum_ratio, mean_abs_accum
        aa = np.asarray(a, dtype=np.float32)
        bb = np.asarray(b, dtype=np.float32)
        delta = np.abs(bb - aa)
        ratio = delta / float(spatial_step)
        count = int(delta.size)
        total_edges += count
        sign_flip_edges += int(np.count_nonzero((aa < 0.0) != (bb < 0.0)))
        mean_abs_accum += float(np.sum(delta, dtype=np.float64))
        if count:
            maximum_ratio = max(maximum_ratio, float(np.max(ratio)))
        for factor in thresholds:
            exceed[factor] += int(np.count_nonzero(ratio > factor))

    base = n - 1
    for x0 in range(0, base, int(slab)):
        x1 = min(base, x0 + int(slab))
        consume(field[x0:x1, :, :], field[x0 + 1:x1 + 1, :, :])
    for x0 in range(0, n, int(slab)):
        x1 = min(n, x0 + int(slab))
        consume(field[x0:x1, :-1, :], field[x0:x1, 1:, :])
        consume(field[x0:x1, :, :-1], field[x0:x1, :, 1:])

    return {
        "total_grid_edges": int(total_edges),
        "sign_flip_edge_count": int(sign_flip_edges),
        "sign_flip_edge_fraction": float(sign_flip_edges / total_edges),
        "mean_abs_field_difference": float(mean_abs_accum / total_edges),
        "max_abs_field_difference_over_spatial_step": float(maximum_ratio),
        "edge_fraction_abs_difference_over_spatial_step": {
            f"gt_{int(factor)}x": float(exceed[factor] / total_edges)
            for factor in thresholds
        },
        "interpretation": "DISTANCE_LIKE_COMPATIBILITY_INDICATORS_ONLY__NOT_A_LIPSCHITZ_PROOF",
    }


def _finite_memmap(field: np.ndarray, *, slab: int = 8) -> bool:
    for x0 in range(0, int(field.shape[0]), int(slab)):
        x1 = min(int(field.shape[0]), x0 + int(slab))
        if not bool(np.isfinite(field[x0:x1]).all()):
            return False
    return True


def audit(*, failure_receipt: Path, output: Path) -> dict[str, Any]:
    failure = load_json(failure_receipt.resolve())
    if failure.get("schema") != EXPECTED_FAILURE_SCHEMA:
        raise ValueError("RAW_FIELD_AUDIT_FAILURE_SCHEMA_INVALID")
    if failure.get("exception_type") != "SparseSurfaceTruncationError":
        raise ValueError("RAW_FIELD_AUDIT_EXPECTS_SPARSE_TRUNCATION_FAILURE")
    diagnostics = dict(failure.get("stage_diagnostics") or {})
    if diagnostics.get("field_query_api") != "HISTORICAL_SCENE_CORE_QUERY_V1":
        raise ValueError("RAW_FIELD_AUDIT_FIELD_QUERY_API_DRIFT")

    work_dir = Path(str(failure.get("work_dir") or "")).resolve()
    coarse_path = work_dir / "v5_coarse_field.f32"
    if not coarse_path.is_file():
        raise FileNotFoundError("RAW_FIELD_AUDIT_COARSE_MEMMAP_MISSING")

    policy = SparseRegularTetraPolicyV5()
    expected_bytes = (policy.base_cells + 1) ** 3 * np.dtype(np.float32).itemsize
    if coarse_path.stat().st_size != expected_bytes:
        raise ValueError("RAW_FIELD_AUDIT_COARSE_MEMMAP_SIZE_DRIFT")
    coarse = np.memmap(
        coarse_path,
        mode="r",
        dtype=np.float32,
        shape=(policy.base_cells + 1,) * 3,
    )
    if not _finite_memmap(coarse, slab=policy.refine_slab_cells):
        raise ValueError("RAW_FIELD_AUDIT_COARSE_NONFINITE")

    refined = select_refined_cells_v5(coarse, policy=policy)
    boundary = boundary_parent_cells_v5(refined, policy=policy)
    sign_cells, band_only_cells = _cell_corner_extrema(
        coarse,
        slab=policy.refine_slab_cells,
    )

    base = int(policy.base_cells)

    def keys(cells: np.ndarray) -> np.ndarray:
        if len(cells) == 0:
            return np.empty(0, dtype=np.int64)
        c = cells.astype(np.int64, copy=False)
        return ((c[:, 0] * base + c[:, 1]) * base + c[:, 2])

    refined_keys = keys(refined)
    sign_keys = keys(sign_cells)
    sign_pos = np.searchsorted(refined_keys, sign_keys)
    if len(sign_keys) and (
        np.any(sign_pos >= len(refined_keys))
        or np.any(refined_keys[sign_pos] != sign_keys)
    ):
        raise RuntimeError("RAW_FIELD_AUDIT_SIGN_CELL_NOT_REFINED")
    sign_boundary_count = int(boundary[sign_pos].sum()) if len(sign_pos) else 0
    domain_boundary_sign = (
        int(
            np.count_nonzero(
                (sign_cells == 0).any(axis=1)
                | (sign_cells == (base - 1)).any(axis=1)
            )
        )
        if len(sign_cells)
        else 0
    )

    total_cells = base ** 3
    result = {
        "schema": SCHEMA,
        "status": "MEASURED__RAW_FIELD_SPARSE_A_ARM_REMAINS_FAILED",
        "subject_id": str(failure.get("subject_id")),
        "execution_source_sha": str(failure.get("execution_source_sha")),
        "failure_exception": str(failure.get("exception_message")),
        "coarse_field": {
            "base_cells": base,
            "query_points": int(coarse.size),
            "minimum": float(np.min(coarse)),
            "maximum": float(np.max(coarse)),
            "zero_bracket": bool(float(np.min(coarse)) < 0.0 < float(np.max(coarse))),
            "base_step": float(policy.base_step),
            "refine_band": float(policy.refine_band),
        },
        "refinement": {
            "total_coarse_cells": int(total_cells),
            "refined_cell_count": int(len(refined)),
            "refined_cell_fraction": float(len(refined) / total_cells),
            "coarse_sign_crossing_cell_count": int(len(sign_cells)),
            "coarse_sign_crossing_cell_fraction": float(len(sign_cells) / total_cells),
            "band_only_cell_count": int(len(band_only_cells)),
            "band_only_fraction_of_refined": float(len(band_only_cells) / len(refined)),
            "boundary_parent_cell_count": int(boundary.sum()),
            "boundary_parent_fraction_of_refined": float(boundary.mean()),
            "sign_crossing_boundary_parent_count": sign_boundary_count,
            "sign_crossing_boundary_parent_fraction": float(sign_boundary_count / len(sign_cells)) if len(sign_cells) else 0.0,
            "domain_boundary_sign_crossing_cell_count": domain_boundary_sign,
        },
        "roughness": _edge_roughness(
            coarse,
            policy.base_step,
            slab=policy.refine_slab_cells,
        ),
        "authority": {
            "input": "PRESERVED_FAILED_RAW_V8_FIELD_T512_MEMMAP",
            "teacher_truth_used": False,
            "model_requeried": False,
            "sparse_policy_changed": False,
            "failure_overridden": False,
            "scientific_pass_claimed": False,
        },
        "interpretation": {
            "this_can_falsify": "COMPATIBILITY_OF_RAW_V8_FIELD_WITH_HISTORICAL_DISTANCE_LIKE_SPARSE_SELECTOR",
            "this_does_not_falsify": [
                "EXACT_V8_NATIVE_FLEXICUBES_SURFACE_WITNESS",
                "SEALED_V8_SHARED_SCENE_CARRIER_H",
                "TESSA_V2_STRUCTURALIZATION",
            ],
            "next_control": "MATERIALIZE_EXACT_V8_NATIVE_R24_FLEXICUBES_WITNESS_AND_RUN_GEOMETRY_PLUS_STAGE01_MASK_MEASUREMENTS",
        },
    }
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--failure-receipt", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    result = audit(
        failure_receipt=Path(args.failure_receipt),
        output=Path(args.output),
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
