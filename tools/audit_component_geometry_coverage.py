from __future__ import annotations

"""Evaluator-only component coverage audit for full-subject geometry.

The tool never feeds source component truth into product inference. It is intended for
FIT/evaluation audits where a source authority is legally available. It catches cases
where aggregate whole-character geometry metrics can hide a missing visible component.

Component ranges/classification are supplied as an external truth JSON so the tool has
no Mage-specific names or rules.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_component_rows(path: Path, vertex_count: int) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("components")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("COMPONENT_TRUTH_EMPTY")
    out = []
    occupied = np.zeros(vertex_count, dtype=bool)
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RuntimeError(f"COMPONENT_TRUTH_ROW_INVALID::{i}")
        vr = row.get("vertex_range")
        if not isinstance(vr, list) or len(vr) != 2:
            raise RuntimeError(f"COMPONENT_VERTEX_RANGE_MISSING::{i}")
        a, b = int(vr[0]), int(vr[1])
        if not (0 <= a < b <= vertex_count):
            raise RuntimeError(f"COMPONENT_VERTEX_RANGE_INVALID::{i}::{a}:{b}")
        if occupied[a:b].any():
            raise RuntimeError(f"COMPONENT_VERTEX_RANGE_OVERLAP::{i}")
        occupied[a:b] = True
        cid = str(row.get("component_family") or row.get("component_id") or f"component_{i}")
        klass = str(row.get("mechanical_class") or "UNKNOWN")
        out.append({**row, "component_id": cid, "mechanical_class": klass, "start": a, "end": b})
    if not occupied.all():
        missing = np.flatnonzero(~occupied)
        raise RuntimeError(f"COMPONENT_TRUTH_DOES_NOT_ACCOUNT_FOR_ALL_SOURCE_VERTICES::{len(missing)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--normalized", type=Path, required=True)
    ap.add_argument("--prediction", type=Path, required=True)
    ap.add_argument("--component-truth", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--normalization-half-extent", type=float, required=True)
    ap.add_argument("--near-tolerance-norm", type=float, default=0.05)
    args = ap.parse_args()

    half = float(args.normalization_half_extent)
    tol = float(args.near_tolerance_norm)
    if not np.isfinite(half) or half <= 0:
        raise RuntimeError("NORMALIZATION_HALF_EXTENT_INVALID")
    if not np.isfinite(tol) or tol <= 0:
        raise RuntimeError("NEAR_TOLERANCE_INVALID")

    with np.load(args.normalized, allow_pickle=False) as d:
        if "vertices_source" not in d.files:
            raise RuntimeError("NORMALIZED_VERTICES_SOURCE_MISSING")
        source = np.asarray(d["vertices_source"], dtype=np.float64)
        skin = np.asarray(d["skin"], dtype=np.float64) if "skin" in d.files else None

    with np.load(args.prediction, allow_pickle=False) as d:
        if "vertices" not in d.files:
            raise RuntimeError("PREDICTION_VERTICES_MISSING")
        pred = np.asarray(d["vertices"], dtype=np.float64)

    if source.ndim != 2 or source.shape[1] != 3 or not np.isfinite(source).all():
        raise RuntimeError("NORMALIZED_SOURCE_VERTEX_SHAPE_INVALID")
    if pred.ndim != 2 or pred.shape[1] != 3 or not np.isfinite(pred).all() or len(pred) < 4:
        raise RuntimeError("PREDICTION_VERTEX_SHAPE_INVALID")

    components = load_component_rows(args.component_truth, len(source))
    pred_tree = cKDTree(pred)
    source_tree = cKDTree(source)

    rows = []
    for comp in components:
        a, b = int(comp["start"]), int(comp["end"])
        src = source[a:b]
        d_src_to_pred, _ = pred_tree.query(src, k=1)
        p95 = float(np.quantile(d_src_to_pred / half, 0.95))
        mean = float(np.mean(d_src_to_pred / half))
        within = float(np.mean((d_src_to_pred / half) <= tol))

        # Prediction->source attribution is diagnostic only. It quantifies how many
        # predicted vertices are nearest to this source component; it does not assign
        # product component identity.
        d_pred_to_source, nn = source_tree.query(pred, k=1)
        attributed = (nn >= a) & (nn < b)
        attr_count = int(attributed.sum())
        attr_p95 = (
            float(np.quantile(d_pred_to_source[attributed] / half, 0.95))
            if attr_count else None
        )

        skin_mass = None
        zero_skin_rows = None
        if skin is not None:
            sub = skin[a:b]
            skin_mass = float(sub.sum())
            zero_skin_rows = int((sub.sum(axis=1) <= 1e-8).sum())

        rows.append({
            "component_id": comp["component_id"],
            "mechanical_class": comp["mechanical_class"],
            "vertex_range": [a, b],
            "source_vertex_count": int(b - a),
            "source_to_prediction_mean_norm": mean,
            "source_to_prediction_p95_norm": p95,
            "source_vertices_within_tolerance_fraction": within,
            "near_tolerance_norm": tol,
            "prediction_vertices_nearest_to_component": attr_count,
            "attributed_prediction_to_source_p95_norm": attr_p95,
            "source_skin_total_mass": skin_mass,
            "source_zero_skin_rows": zero_skin_rows,
            "required_visible_component": bool(comp.get("required_visible_component", comp["mechanical_class"] != "EXCLUDED_SOURCE_COMPONENT")),
        })

    required = [r for r in rows if r["required_visible_component"]]
    min_fraction = min((r["source_vertices_within_tolerance_fraction"] for r in required), default=0.0)
    report = {
        "schema": "RealSaS.ComponentGeometryCoverageAudit.v1",
        "status": "MEASURED__NO_PROMOTION_DECISION",
        "normalized_source_sha256": sha256_file(args.normalized),
        "prediction_sha256": sha256_file(args.prediction),
        "component_truth_sha256": sha256_file(args.component_truth),
        "normalization_half_extent": half,
        "near_tolerance_norm": tol,
        "source_vertex_count": int(len(source)),
        "prediction_vertex_count": int(len(pred)),
        "required_component_count": int(len(required)),
        "min_required_component_within_tolerance_fraction": float(min_fraction),
        "components": rows,
        "teacher_component_truth_used_at_product_inference": False,
        "canonical_mutation": False,
        "interpretation": "Evaluator only. Do not infer product component IDs from nearest-source attribution.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report["status"])
    for row in rows:
        print(
            row["component_id"],
            f"p95={row['source_to_prediction_p95_norm']:.6f}",
            f"within={row['source_vertices_within_tolerance_fraction']:.6f}",
            f"pred_attr={row['prediction_vertices_nearest_to_component']}",
        )
    print(args.out)


if __name__ == "__main__":
    main()
