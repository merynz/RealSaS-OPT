from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from teacher_projection_v1 import project_skeleton_teacher_v1

NEAR_ZERO_REL_BBOX_V1 = 1e-8
SKIN_NEGATIVE_TOL_V1 = -1e-8
ROW_MASS_EPS_V1 = 1e-12


def _json_float(x: float) -> float:
    x = float(x)
    if not np.isfinite(x):
        raise ValueError(f"non-finite summary scalar: {x}")
    return x


def summarize_numeric_v1(values: Iterable[float]) -> dict[str, float | int | None]:
    a = np.asarray(list(values), dtype=np.float64)
    if a.size == 0:
        return {"count": 0, "min": None, "p50": None, "p95": None, "p99": None, "max": None, "mean": None}
    if not np.isfinite(a).all():
        raise ValueError("non-finite values passed to summarize_numeric_v1")
    return {
        "count": int(a.size),
        "min": _json_float(a.min()),
        "p50": _json_float(np.quantile(a, 0.50)),
        "p95": _json_float(np.quantile(a, 0.95)),
        "p99": _json_float(np.quantile(a, 0.99)),
        "max": _json_float(a.max()),
        "mean": _json_float(a.mean()),
    }


def nearest_deform_ancestor_v1(index: int, parents: np.ndarray, deform_mask: np.ndarray) -> int | None:
    p = int(parents[index])
    while p != -1:
        if bool(deform_mask[p]):
            return p
        p = int(parents[p])
    return None


def helper_skip_hops_for_deform_v1(parents: np.ndarray, deform_mask: np.ndarray) -> dict[int, int]:
    out: dict[int, int] = {}
    for i in np.flatnonzero(deform_mask).astype(int).tolist():
        p = int(parents[i])
        skipped = 0
        while p != -1 and not bool(deform_mask[p]):
            skipped += 1
            p = int(parents[p])
        out[i] = skipped
    return out


def _required_rig_arrays(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    missing = [k for k in ("bone_heads", "bone_tails", "parents", "deform_mask") if k not in arrays]
    if missing:
        raise ValueError(f"missing required rig arrays: {missing}")
    H = np.asarray(arrays["bone_heads"], dtype=np.float64)
    T = np.asarray(arrays["bone_tails"], dtype=np.float64)
    P = np.asarray(arrays["parents"], dtype=np.int64)
    D = np.asarray(arrays["deform_mask"], dtype=bool)
    return H, T, P, D


def _skin_audit_v1(skin: np.ndarray, parents: np.ndarray, deform_mask: np.ndarray) -> dict[str, Any]:
    W = np.asarray(skin, dtype=np.float64)
    n = int(len(deform_mask))
    if W.ndim != 2:
        raise ValueError(f"skin must be rank-2 [point,bone], got {W.shape}")
    if W.shape[1] != n:
        raise ValueError(
            f"skin bone axis mismatch: expected shape[1]=={n}, got {W.shape}; "
            "silent transpose/reindex is forbidden"
        )
    if not np.isfinite(W).all():
        raise ValueError("skin contains non-finite values")
    min_w = float(W.min()) if W.size else 0.0
    if min_w < SKIN_NEGATIVE_TOL_V1:
        raise ValueError(f"skin has weight below tolerance: min={min_w}")
    # Tiny negative numerical residue is retained in diagnostics, but clipped only
    # for mass accounting so signed cancellation cannot hide helper-column mass.
    M = np.maximum(W, 0.0)
    row_total = M.sum(axis=1)
    valid_rows = row_total > ROW_MASS_EPS_V1
    helper = ~deform_mask
    helper_row_mass = M[:, helper].sum(axis=1) if np.any(helper) else np.zeros(len(M), dtype=np.float64)
    helper_fraction = np.zeros(len(M), dtype=np.float64)
    helper_fraction[valid_rows] = helper_row_mass[valid_rows] / row_total[valid_rows]
    col_mass = M.sum(axis=0)
    total_mass = float(col_mass.sum())
    helper_mass_total = float(col_mass[helper].sum()) if np.any(helper) else 0.0

    transportable = 0.0
    untransportable = 0.0
    helper_columns_nonzero = 0
    orphan_helper_columns_nonzero = 0
    helper_column_rows: list[dict[str, Any]] = []
    for b in np.flatnonzero(helper).astype(int).tolist():
        mass = float(col_mass[b])
        if mass <= 0.0:
            continue
        helper_columns_nonzero += 1
        anc = nearest_deform_ancestor_v1(b, parents, deform_mask)
        if anc is None:
            untransportable += mass
            orphan_helper_columns_nonzero += 1
        else:
            transportable += mass
        helper_column_rows.append({
            "source_bone_index": b,
            "column_mass": mass,
            "nearest_deform_ancestor": anc,
        })

    return {
        "shape": [int(x) for x in W.shape],
        "min_raw_weight": min_w,
        "negative_entry_count_below_zero": int(np.sum(W < 0.0)),
        "row_sum": summarize_numeric_v1(row_total.tolist()),
        "zero_mass_row_count": int(np.sum(~valid_rows)),
        "helper_mass_fraction_per_row": summarize_numeric_v1(helper_fraction[valid_rows].tolist()),
        "total_mass": total_mass,
        "deform_mass_total": float(col_mass[deform_mask].sum()),
        "nondeform_mass_total": helper_mass_total,
        "nondeform_mass_fraction_total": (helper_mass_total / total_mass) if total_mass > 0.0 else 0.0,
        "helper_columns_nonzero": helper_columns_nonzero,
        "helper_columns_nonzero_without_deform_ancestor": orphan_helper_columns_nonzero,
        "candidate_nearest_ancestor_transportable_mass": transportable,
        "candidate_nearest_ancestor_untransportable_mass": untransportable,
        "candidate_nearest_ancestor_untransportable_fraction_of_helper_mass": (
            untransportable / helper_mass_total if helper_mass_total > 0.0 else 0.0
        ),
        "helper_column_rows": helper_column_rows,
        "transport_was_applied": False,
    }


def audit_projection_asset_v1(
    *,
    canonical_asset_id: str,
    source_registry_id: str | None,
    arrays: dict[str, np.ndarray],
    arachne_capable: bool,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "canonical_asset_id": canonical_asset_id,
        "source_registry_id": source_registry_id,
        "arachne_capable": bool(arachne_capable),
        "status": "FAIL",
        "errors": [],
    }
    try:
        H, T, P, D = _required_rig_arrays(arrays)
        projection = project_skeleton_teacher_v1(H, T, P, D)

        lengths = np.linalg.norm(T - H, axis=1)
        deform_lengths = lengths[D]
        if len(H):
            Q = np.concatenate([H, T], axis=0)
            bbox_diag = float(np.linalg.norm(Q.max(axis=0) - Q.min(axis=0)))
        else:
            bbox_diag = 0.0
        near_eps = NEAR_ZERO_REL_BBOX_V1 * max(bbox_diag, 1.0)
        exact_zero = deform_lengths == 0.0
        near_zero = deform_lengths <= near_eps
        skip = helper_skip_hops_for_deform_v1(P, D)
        helper_hops = list(skip.values())

        out["rig"] = {
            "source_bone_count": int(len(H)),
            "deform_control_count": int(np.sum(D)),
            "skipped_helper_count": int(np.sum(~D)),
            "projected_root_count": int(len(projection.root_control_ids)),
            "multi_root": bool(len(projection.root_control_ids) > 1),
            "helper_skip_hops_per_deform_control": summarize_numeric_v1(helper_hops),
            "helper_skip_hops_histogram": dict(sorted(Counter(helper_hops).items())),
            "deform_controls_with_helper_skip": int(sum(v > 0 for v in helper_hops)),
            "max_helper_skip_hops": int(max(helper_hops, default=0)),
            "skeleton_bbox_diag": bbox_diag,
            "deform_bone_length": summarize_numeric_v1(deform_lengths.tolist()),
            "exact_zero_length_deform_bone_count": int(np.sum(exact_zero)),
            "near_zero_length_deform_bone_count_rel1e8_bbox": int(np.sum(near_zero)),
            "near_zero_epsilon": float(near_eps),
            "projection_metadata": dict(projection.metadata),
        }

        if arachne_capable:
            if "skin" not in arrays:
                raise ValueError("Arachne-capable asset is missing exact skin array")
            out["skin"] = _skin_audit_v1(np.asarray(arrays["skin"]), P, D)
        elif "skin" in arrays:
            out["skin_presence_not_audited_as_authority"] = True

        out["status"] = "PASS"
    except Exception as e:
        out["errors"].append(f"{type(e).__name__}: {e}")
    return out


def aggregate_projection_audit_v1(rows: list[dict[str, Any]]) -> dict[str, Any]:
    passed = [r for r in rows if r.get("status") == "PASS"]
    failed = [r for r in rows if r.get("status") != "PASS"]
    rig = [r["rig"] for r in passed]
    skin = [r["skin"] for r in passed if "skin" in r]

    source_counts = Counter(str(r.get("source_registry_id") or "<missing>") for r in rows)
    global_hops: list[int] = []
    hop_hist = Counter()
    for x in rig:
        for k, v in (x.get("helper_skip_hops_histogram") or {}).items():
            hop = int(k)
            cnt = int(v)
            hop_hist[hop] += cnt
            global_hops.extend([hop] * cnt)

    source_breakdown: dict[str, Any] = {}
    for source in sorted(source_counts):
        rr = [r for r in passed if str(r.get("source_registry_id") or "<missing>") == source]
        rg = [r["rig"] for r in rr]
        source_breakdown[source] = {
            "asset_count": len(rr),
            "deform_control_count": summarize_numeric_v1([x["deform_control_count"] for x in rg]),
            "multi_root_asset_count": int(sum(bool(x["multi_root"]) for x in rg)),
            "exact_zero_length_deform_bone_total": int(
                sum(int(x["exact_zero_length_deform_bone_count"]) for x in rg)
            ),
        }

    out: dict[str, Any] = {
        "asset_count": len(rows),
        "pass_count": len(passed),
        "fail_count": len(failed),
        "source_distribution": dict(sorted(source_counts.items())),
        "source_breakdown": source_breakdown,
        "failure_examples": [
            {"canonical_asset_id": r["canonical_asset_id"], "errors": r.get("errors", [])}
            for r in failed[:100]
        ],
        "deform_control_count": summarize_numeric_v1([x["deform_control_count"] for x in rig]),
        "source_bone_count": summarize_numeric_v1([x["source_bone_count"] for x in rig]),
        "skipped_helper_count": summarize_numeric_v1([x["skipped_helper_count"] for x in rig]),
        "projected_root_count": summarize_numeric_v1([x["projected_root_count"] for x in rig]),
        "multi_root_asset_count": int(sum(bool(x["multi_root"]) for x in rig)),
        "asset_with_exact_zero_length_deform_bone_count": int(
            sum(int(x["exact_zero_length_deform_bone_count"]) > 0 for x in rig)
        ),
        "exact_zero_length_deform_bone_total": int(
            sum(int(x["exact_zero_length_deform_bone_count"]) for x in rig)
        ),
        "asset_with_near_zero_length_deform_bone_count_rel1e8_bbox": int(
            sum(int(x["near_zero_length_deform_bone_count_rel1e8_bbox"]) > 0 for x in rig)
        ),
        "helper_skip_hops_global": summarize_numeric_v1(global_hops),
        "helper_skip_hops_histogram_global": dict(sorted(hop_hist.items())),
        "max_helper_skip_hops": int(max((int(x["max_helper_skip_hops"]) for x in rig), default=0)),
        "assets_with_any_helper_skip": int(sum(int(x["deform_controls_with_helper_skip"]) > 0 for x in rig)),
        "arachne_skin_asset_count": len(skin),
    }

    if skin:
        total_mass = sum(float(x["total_mass"]) for x in skin)
        helper_mass = sum(float(x["nondeform_mass_total"]) for x in skin)
        untransportable = sum(float(x["candidate_nearest_ancestor_untransportable_mass"]) for x in skin)
        out["skin"] = {
            "nondeform_mass_fraction_per_asset": summarize_numeric_v1(
                [float(x["nondeform_mass_fraction_total"]) for x in skin]
            ),
            "corpus_total_mass": total_mass,
            "corpus_nondeform_mass": helper_mass,
            "corpus_nondeform_mass_fraction": helper_mass / total_mass if total_mass > 0.0 else 0.0,
            "candidate_nearest_ancestor_untransportable_mass": untransportable,
            "candidate_nearest_ancestor_untransportable_fraction_of_nondeform_mass": (
                untransportable / helper_mass if helper_mass > 0.0 else 0.0
            ),
            "assets_with_nondeform_mass": int(sum(float(x["nondeform_mass_total"]) > 0.0 for x in skin)),
            "assets_with_untransportable_nondeform_mass": int(
                sum(float(x["candidate_nearest_ancestor_untransportable_mass"]) > 0.0 for x in skin)
            ),
            "transport_was_applied": False,
        }
    return out
