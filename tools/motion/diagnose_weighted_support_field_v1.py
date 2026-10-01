from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _normalize_source(report: dict):
    rows = tuple(report.get("bone_roles") or ())
    roots = [row for row in rows if row.get("parent_source_joint_id") is None]
    if len(roots) != 1:
        raise RuntimeError("MOTION_SUPPORT_SOURCE_ROOT_CARDINALITY")
    root = roots[0]
    root_pos = np.asarray(root["rest_position"], dtype=np.float64)
    scale = float(report.get("body_scale") or 0.0)
    if not math.isfinite(scale) or scale <= 1e-9:
        raise RuntimeError("MOTION_SUPPORT_SOURCE_SCALE_INVALID")
    out = []
    for row in rows:
        centroid = row.get("mesh_influence_centroid")
        if centroid is None:
            continue
        p = (np.asarray(centroid, dtype=np.float64) - root_pos) / scale
        spread = np.asarray(
            row.get("mesh_influence_spread_eigenvalues") or [0.0, 0.0, 0.0],
            dtype=np.float64,
        ) / (scale * scale)
        out.append(
            {
                "id": str(row["source_joint_id"]),
                "centroid": p,
                "spread": np.maximum(spread, 0.0),
                "weight_sum": float(row.get("weight_sum") or 0.0),
                "weighted_vertex_count": int(
                    row.get("weighted_vertex_count") or 0
                ),
            }
        )
    if not out:
        raise RuntimeError("MOTION_SUPPORT_SOURCE_DESCRIPTOR_EMPTY")
    return out, root_pos, scale


def _normalize_target(report: dict):
    rows = tuple(report.get("joint_supports") or ())
    roots = [row for row in rows if row.get("parent_canonical_id") is None]
    if len(roots) != 1:
        raise RuntimeError("MOTION_SUPPORT_TARGET_ROOT_CARDINALITY")
    root_pos = np.asarray(
        roots[0]["joint_rest_position"], dtype=np.float64
    )
    joint_positions = np.asarray(
        [row["joint_rest_position"] for row in rows],
        dtype=np.float64,
    )
    scale = float(
        np.max(np.linalg.norm(joint_positions - root_pos[None, :], axis=1))
    )
    if not math.isfinite(scale) or scale <= 1e-9:
        raise RuntimeError("MOTION_SUPPORT_TARGET_SCALE_INVALID")
    out = []
    for row in rows:
        centroid = row.get("mesh_influence_centroid")
        if centroid is None:
            continue
        p = (np.asarray(centroid, dtype=np.float64) - root_pos) / scale
        spread = np.asarray(
            row.get("mesh_influence_spread_eigenvalues")
            or [0.0, 0.0, 0.0],
            dtype=np.float64,
        ) / (scale * scale)
        out.append(
            {
                "id": str(row["canonical_joint_id"]),
                "centroid": p,
                "spread": np.maximum(spread, 0.0),
                "weight_sum": float(row.get("skin_weight_sum") or 0.0),
                "weighted_surface_count": int(
                    row.get("weighted_surface_count") or 0
                ),
            }
        )
    if not out:
        raise RuntimeError("MOTION_SUPPORT_TARGET_DESCRIPTOR_EMPTY")
    return out, root_pos, scale


def _spread_distance(a: np.ndarray, b: np.ndarray) -> float:
    eps = 1.0e-8
    return float(
        np.linalg.norm(
            np.log(np.maximum(a, eps))
            - np.log(np.maximum(b, eps))
        )
    )


def diagnose(
    *,
    source_report: Path,
    target_report: Path,
    support_count: int,
) -> dict:
    source_raw = _load(source_report)
    target_raw = _load(target_report)
    source, source_root, source_scale = _normalize_source(source_raw)
    target, target_root, target_scale = _normalize_target(target_raw)
    k = int(support_count)
    if k <= 0:
        raise RuntimeError("MOTION_SUPPORT_COUNT_INVALID")

    bindings = []
    centroid_errors = []
    compactness = []
    for target_row in target:
        tc = target_row["centroid"]
        ts = target_row["spread"]
        candidates = []
        for source_row in source:
            sc = source_row["centroid"]
            side_penalty = 0.0
            if abs(float(tc[0])) > 0.04 and abs(float(sc[0])) > 0.04:
                if math.copysign(1.0, float(tc[0])) != math.copysign(
                    1.0, float(sc[0])
                ):
                    side_penalty = 4.0
            centroid_distance = float(np.linalg.norm(tc - sc))
            spread_distance = _spread_distance(
                ts, source_row["spread"]
            )
            cost = centroid_distance + 0.08 * min(spread_distance, 8.0)
            cost += side_penalty
            candidates.append(
                {
                    "source_joint_id": source_row["id"],
                    "cost": float(cost),
                    "centroid_distance": centroid_distance,
                    "spread_distance": spread_distance,
                    "side_penalty": side_penalty,
                    "centroid": sc,
                }
            )
        candidates.sort(
            key=lambda row: (row["cost"], row["source_joint_id"])
        )
        selected = candidates[: min(k, len(candidates))]
        costs = np.asarray([row["cost"] for row in selected], dtype=np.float64)
        if len(costs) == 1:
            weights = np.ones(1, dtype=np.float64)
        else:
            shifted = costs - float(np.min(costs))
            logits = np.exp(-4.0 * shifted)
            weights = logits / float(np.sum(logits))
        reconstructed = np.zeros(3, dtype=np.float64)
        for weight, row in zip(weights, selected):
            reconstructed += float(weight) * row["centroid"]
        error = float(np.linalg.norm(reconstructed - tc))
        centroid_errors.append(error)
        entropy = -float(
            np.sum(
                weights
                * np.log(np.maximum(weights, 1.0e-12))
            )
        )
        effective_support = float(math.exp(entropy))
        compactness.append(effective_support)
        bindings.append(
            {
                "target_joint_id": target_row["id"],
                "target_normalized_influence_centroid": [
                    float(x) for x in tc
                ],
                "support_reconstruction_error": error,
                "effective_support_count": effective_support,
                "supports": [
                    {
                        "source_joint_id": row["source_joint_id"],
                        "weight": float(weight),
                        "cost": float(row["cost"]),
                        "centroid_distance": float(
                            row["centroid_distance"]
                        ),
                        "spread_distance": float(
                            row["spread_distance"]
                        ),
                        "side_penalty": float(row["side_penalty"]),
                    }
                    for weight, row in zip(weights, selected)
                ],
            }
        )

    errors = np.asarray(centroid_errors, dtype=np.float64)
    effective = np.asarray(compactness, dtype=np.float64)
    return {
        "schema": "RealSaS.MotionWeightedSupportFieldDiagnostic.v1",
        "status": "MEASURED",
        "representation": (
            "NONINJECTIVE_WEIGHTED_SOURCE_MOTION_SUPPORT_FIELD"
        ),
        "source_descriptor": (
            "SOURCE_FBX_SKIN_INFLUENCE_CENTROID_SPREAD"
        ),
        "target_descriptor": (
            "STAGE15_SURFACE_X_STAGE32_SKIN_INFLUENCE_CENTROID_SPREAD"
        ),
        "coordinate_normalization": (
            "ROOT_CENTERED__BODY_SCALE_NORMALIZED__CANONICAL_OBJECT_FRAME"
        ),
        "source_scale": float(source_scale),
        "target_scale": float(target_scale),
        "source_supported_joint_count": len(source),
        "target_supported_joint_count": len(target),
        "maximum_support_count": k,
        "support_reconstruction_rms": float(
            math.sqrt(float(np.mean(errors * errors)))
        ),
        "support_reconstruction_p95": float(
            np.quantile(errors, 0.95)
        ),
        "support_reconstruction_max": float(np.max(errors)),
        "mean_effective_support_count": float(np.mean(effective)),
        "p95_effective_support_count": float(
            np.quantile(effective, 0.95)
        ),
        "bindings": bindings,
        "interpretation": {
            "diagnostic_only": True,
            "manual_names_used_for_cost": False,
            "source_parent_tree_used_for_cost": False,
            "source_mesh_is_motion_evidence_only": True,
            "target_skeleton_remains_mechanical_authority": True,
            "low_reconstruction_error_is_necessary_not_sufficient": True,
            "dynamic_motion_transfer_must_be_tested_separately": True,
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-report", required=True)
    p.add_argument("--target-report", required=True)
    p.add_argument("--support-count", type=int, default=4)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    report = diagnose(
        source_report=Path(a.source_report),
        target_report=Path(a.target_report),
        support_count=a.support_count,
    )
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        "MOTION_WEIGHTED_SUPPORT_FIELD_DIAGNOSTIC",
        json.dumps(
            {
                "rms": report["support_reconstruction_rms"],
                "p95": report["support_reconstruction_p95"],
                "max": report["support_reconstruction_max"],
                "mean_effective_support_count": report[
                    "mean_effective_support_count"
                ],
            },
            sort_keys=True,
        ),
    )


if __name__ == "__main__":
    main()
