#!/usr/bin/env python3
"""Read-only real FIT8 morphology-scale census for V3.1 extraction policy.

Consumes only hash-bound admitted 8-view source masks and camera bundles already
present on the authority host. It measures projected foreground run widths and
internal silhouette gaps in normalized TESSA units, then reports how those real
scales compare with the already-established analytic 24/32/48/64 observability
table. It deliberately does not select a production extraction resolution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import runpy

import numpy as np


SCHEMA = "RealSaS.FIT8RealMorphologyScaleCensus.v1"
REQUEST_SCHEMA = "RealSaS.NativeTeacherReplayRequest.v1"
EXPECTED_SUBJECT_COUNT = 8
EXPECTED_VIEW_COUNT = 8
EXPECTED_RESOLUTION = 1024


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_authority_path(authority_root: Path, relative: str) -> Path:
    root = authority_root.resolve()
    path = (root / str(relative)).resolve()
    if not path.is_relative_to(root):
        raise ValueError("FIT8_MORPHOLOGY_AUTHORITY_PATH_ESCAPE")
    return path


def _runs_1d(values: np.ndarray, *, foreground: bool, internal_only: bool) -> list[int]:
    row = np.asarray(values, dtype=bool).reshape(-1)
    target = row if foreground else ~row
    if internal_only:
        support = np.flatnonzero(row)
        if len(support) < 2:
            return []
        target = target.copy()
        target[: int(support[0]) + 1] = False
        target[int(support[-1]) :] = False
    padded = np.concatenate((np.asarray([False]), target, np.asarray([False])))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    if len(edges) % 2:
        raise RuntimeError("FIT8_MORPHOLOGY_RUN_ACCOUNTING_DRIFT")
    return [int(b - a) for a, b in zip(edges[0::2], edges[1::2]) if b > a]


def mask_run_lengths(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(mask, dtype=bool)
    if value.ndim != 2 or min(value.shape) <= 0 or not bool(value.any()):
        raise ValueError("FIT8_MORPHOLOGY_MASK_INVALID")
    foreground: list[int] = []
    gaps: list[int] = []
    for row in value:
        foreground.extend(_runs_1d(row, foreground=True, internal_only=False))
        gaps.extend(_runs_1d(row, foreground=False, internal_only=True))
    for column in value.T:
        foreground.extend(_runs_1d(column, foreground=True, internal_only=False))
        gaps.extend(_runs_1d(column, foreground=False, internal_only=True))
    if not foreground:
        raise RuntimeError("FIT8_MORPHOLOGY_FOREGROUND_RUNS_EMPTY")
    return np.asarray(foreground, dtype=np.int64), np.asarray(gaps, dtype=np.int64)


def nearest_rank_quantiles(values: np.ndarray) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(arr):
        return {key: None for key in ("min", "p01", "p05", "p50", "p95", "max")}
    if not np.isfinite(arr).all() or (arr <= 0).any():
        raise ValueError("FIT8_MORPHOLOGY_SCALE_VALUES_INVALID")
    ordered = np.sort(arr)

    def q(probability: float) -> float:
        if probability <= 0:
            return float(ordered[0])
        if probability >= 1:
            return float(ordered[-1])
        index = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
        return float(ordered[index])

    return {
        "min": q(0.0),
        "p01": q(0.01),
        "p05": q(0.05),
        "p50": q(0.50),
        "p95": q(0.95),
        "max": q(1.0),
    }


def tessa_units_per_pixel(camera_half_extent: float, common_half_extent: float, resolution: int) -> float:
    camera_half = float(camera_half_extent)
    common_half = float(common_half_extent)
    if resolution <= 0 or not math.isfinite(camera_half) or not math.isfinite(common_half):
        raise ValueError("FIT8_MORPHOLOGY_CAMERA_SCALE_INVALID")
    if camera_half <= 0 or common_half <= 0:
        raise ValueError("FIT8_MORPHOLOGY_CAMERA_SCALE_NONPOSITIVE")
    # IRIS normalized screen width is 2*camera_half/common_half; TESSA is exactly
    # half the IRIS coordinate domain, so one raster pixel is this ratio / N.
    return camera_half / common_half / float(resolution)


def _fraction_at_least(values: np.ndarray, threshold: float) -> float | None:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(arr):
        return None
    return float(np.mean(arr >= float(threshold)))


def _analytic_observability(repo_root: Path) -> dict:
    path = repo_root / "experiments/iris_tessa_v3_apparatus_audit_20261025/extraction_topology_observability_probe_v1.py"
    module = runpy.run_path(str(path))
    result = module["run"]()
    if result.get("status") != "EXTRACTION_OBSERVABILITY_TABLE_ESTABLISHED_V1":
        raise RuntimeError("FIT8_MORPHOLOGY_ANALYTIC_OBSERVABILITY_INVALID")
    return result


def run(authority_root: Path, expected_main_sha: str, repo_root: Path) -> dict:
    authority_root = authority_root.resolve()
    request_path = repo_root / "canonical/NATIVE_TEACHER_REPLAY_REQUEST_V1.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request.get("schema") != REQUEST_SCHEMA or request.get("training_allowed") is not False:
        raise RuntimeError("FIT8_MORPHOLOGY_REQUEST_AUTHORITY_DRIFT")
    subjects = list(request.get("subjects") or ())
    if len(subjects) != EXPECTED_SUBJECT_COUNT:
        raise RuntimeError("FIT8_MORPHOLOGY_SUBJECT_COUNT_DRIFT")

    all_foreground: list[np.ndarray] = []
    all_gaps: list[np.ndarray] = []
    subject_rows = []
    total_views = 0

    for subject in subjects:
        subject_id = str(subject["subject_id"])
        admission_ref = subject["observation_admission"]
        proof_path = _safe_authority_path(authority_root, admission_ref["relative"])
        if not proof_path.is_file() or file_sha256(proof_path) != admission_ref["sha256"]:
            raise RuntimeError(f"FIT8_MORPHOLOGY_ADMISSION_BYTES_DRIFT:{subject_id}")
        proof = json.loads(proof_path.read_text(encoding="utf-8"))
        if proof.get("status") != "PASS_IRIS_BIND_OBSERVATIONS" or proof.get("subject_id") != subject_id:
            raise RuntimeError(f"FIT8_MORPHOLOGY_ADMISSION_RECEIPT_DRIFT:{subject_id}")
        if proof.get("fit8_pass_claimed") is not False:
            raise RuntimeError(f"FIT8_MORPHOLOGY_ADMISSION_AUTHORITY_VIOLATION:{subject_id}")
        folder = proof_path.parent
        camera_path = folder / "camera_bundle.json"
        if not camera_path.is_file() or file_sha256(camera_path) != proof.get("camera_sha256"):
            raise RuntimeError(f"FIT8_MORPHOLOGY_CAMERA_BYTES_DRIFT:{subject_id}")
        camera_bundle = json.loads(camera_path.read_text(encoding="utf-8"))
        common_half = float(camera_bundle.get("common_half_extent", 0.0))
        cameras = list(camera_bundle.get("cameras") or ())
        proof_views = {int(row["view_index"]): row for row in proof.get("views") or ()}
        if len(cameras) != EXPECTED_VIEW_COUNT or len(proof_views) != EXPECTED_VIEW_COUNT:
            raise RuntimeError(f"FIT8_MORPHOLOGY_VIEW_COUNT_DRIFT:{subject_id}")

        subject_foreground: list[np.ndarray] = []
        subject_gaps: list[np.ndarray] = []
        views = []
        for index, camera in enumerate(cameras):
            receipt = proof_views.get(index)
            if receipt is None:
                raise RuntimeError(f"FIT8_MORPHOLOGY_VIEW_INDEX_DRIFT:{subject_id}:{index}")
            mask_path = folder / f"V{index}.mask.bin"
            if not mask_path.is_file() or file_sha256(mask_path) != receipt.get("mask_sha256"):
                raise RuntimeError(f"FIT8_MORPHOLOGY_MASK_BYTES_DRIFT:{subject_id}:{index}")
            raw = mask_path.read_bytes()
            if len(raw) != EXPECTED_RESOLUTION * EXPECTED_RESOLUTION:
                raise RuntimeError(f"FIT8_MORPHOLOGY_MASK_SIZE_DRIFT:{subject_id}:{index}")
            mask = np.frombuffer(raw, dtype=np.uint8).reshape(EXPECTED_RESOLUTION, EXPECTED_RESOLUTION) > 0
            foreground_px, gap_px = mask_run_lengths(mask)
            scale = tessa_units_per_pixel(camera["half_extent"], common_half, EXPECTED_RESOLUTION)
            foreground = foreground_px.astype(np.float64) * scale
            gaps = gap_px.astype(np.float64) * scale
            subject_foreground.append(foreground)
            if len(gaps):
                subject_gaps.append(gaps)
            all_foreground.append(foreground)
            if len(gaps):
                all_gaps.append(gaps)
            views.append({
                "view_index": index,
                "mask_sha256": receipt["mask_sha256"],
                "tessa_units_per_pixel": scale,
                "mask_area_fraction": float(mask.mean()),
                "foreground_run_count": int(len(foreground)),
                "internal_gap_run_count": int(len(gaps)),
                "foreground_run_tessa": nearest_rank_quantiles(foreground),
                "internal_gap_tessa": nearest_rank_quantiles(gaps),
            })
            total_views += 1

        subject_fg = np.concatenate(subject_foreground)
        subject_gap = np.concatenate(subject_gaps) if subject_gaps else np.asarray([], dtype=np.float64)
        subject_rows.append({
            "subject_id": subject_id,
            "observation_admission_sha256": admission_ref["sha256"],
            "camera_bundle_sha256": proof["camera_sha256"],
            "view_count": len(views),
            "foreground_run_tessa": nearest_rank_quantiles(subject_fg),
            "internal_gap_tessa": nearest_rank_quantiles(subject_gap),
            "foreground_run_count": int(len(subject_fg)),
            "internal_gap_run_count": int(len(subject_gap)),
            "views": views,
        })

    foreground = np.concatenate(all_foreground) if all_foreground else np.asarray([], dtype=np.float64)
    gaps = np.concatenate(all_gaps) if all_gaps else np.asarray([], dtype=np.float64)
    if total_views != EXPECTED_SUBJECT_COUNT * EXPECTED_VIEW_COUNT or not len(foreground):
        raise RuntimeError("FIT8_MORPHOLOGY_CENSUS_INCOMPLETE")

    analytic = _analytic_observability(repo_root)
    comparison = {}
    for resolution in analytic["resolutions"]:
        table = analytic["table"][str(resolution)]
        cylinder = table["cylinder_diameter"]
        gap = table["parallel_sheet_gap"]
        comparison[str(resolution)] = {
            "grid_spacing_tessa": float(cylinder["grid_spacing_tessa"]),
            "foreground_vs_cylinder_p95_observable_fraction": _fraction_at_least(foreground, cylinder["p95_tessa"]),
            "foreground_vs_cylinder_analytic_guaranteed_fraction": _fraction_at_least(foreground, cylinder["analytic_guaranteed_tessa"]),
            "internal_gap_vs_parallel_sheet_p95_observable_fraction": _fraction_at_least(gaps, gap["p95_tessa"]),
            "internal_gap_vs_parallel_sheet_analytic_guaranteed_fraction": _fraction_at_least(gaps, gap["analytic_guaranteed_tessa"]),
            "cylinder_p95_threshold_tessa": float(cylinder["p95_tessa"]),
            "cylinder_analytic_guaranteed_threshold_tessa": float(cylinder["analytic_guaranteed_tessa"]),
            "gap_p95_threshold_tessa": float(gap["p95_tessa"]),
            "gap_analytic_guaranteed_threshold_tessa": float(gap["analytic_guaranteed_tessa"]),
        }

    return {
        "schema": SCHEMA,
        "status": "FIT8_REAL_MORPHOLOGY_SCALE_CENSUS_COMPLETE_V1",
        "source_code_sha": str(expected_main_sha).lower(),
        "authority_root": str(authority_root),
        "request_sha256": file_sha256(request_path),
        "subject_count": len(subject_rows),
        "view_count": total_views,
        "measurement_semantics": {
            "source": "HASH_BOUND_ADMITTED_BINARY_SILHOUETTE_MASKS",
            "foreground_scale": "ROW_AND_COLUMN_FOREGROUND_RUN_WIDTH",
            "gap_scale": "ROW_AND_COLUMN_INTERNAL_BACKGROUND_RUN_WIDTH_BETWEEN_FOREGROUND",
            "normalization": "TESSA_UNITS_FROM_EXACT_ORTHOGRAPHIC_CAMERA_HALF_EXTENT",
            "authored_triangle_edge_length_used": False,
            "three_dimensional_thickness_claimed": False,
        },
        "aggregate": {
            "foreground_run_count": int(len(foreground)),
            "internal_gap_run_count": int(len(gaps)),
            "foreground_run_tessa": nearest_rank_quantiles(foreground),
            "internal_gap_tessa": nearest_rank_quantiles(gaps),
        },
        "analytic_observability_schema": analytic["schema"],
        "resolution_comparison": comparison,
        "subjects": subject_rows,
        "selection": {
            "production_resolution_selected": False,
            "adaptive_policy_selected": False,
            "reason": "REAL_CENSUS_MEASURED__RESOLUTION_SELECTION_RULE_NOT_PREREGISTERED",
            "next_action": "PREREGISTER_REAL_MORPHOLOGY_TO_EXTRACTION_RESOLUTION_SELECTION_RULE",
        },
        "host_scan_is_execution_authority": False,
        "training_executed": False,
        "model_release_minted": False,
        "model_promotion_performed": False,
        "product_shape_activated": False,
        "shape_proposal_minted": False,
        "canonical_mesh_minted": False,
        "product_revision_minted": False,
        "fit8_pass_claimed": False,
        "genericity_claimed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority-root", type=Path, required=True)
    parser.add_argument("--expected-main-sha", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    result = run(args.authority_root, args.expected_main_sha, repo_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "subject_count": result["subject_count"],
        "view_count": result["view_count"],
        "aggregate": result["aggregate"],
        "resolution_comparison": result["resolution_comparison"],
        "next_action": result["selection"]["next_action"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
