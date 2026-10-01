from __future__ import annotations

from typing import Any, Iterable
import math

import numpy as np
from scipy import ndimage


DEFAULT_DISTANCE_BANDS_PX_V5 = (1.0, 2.0, 4.0, 8.0, 16.0)
DEFAULT_EROSION_RADII_PX_V5 = tuple(float(x) for x in range(0, 9))


def _as_bool_mask(mask: np.ndarray, *, name: str) -> np.ndarray:
    out = np.asarray(mask, dtype=bool)
    if out.ndim != 2 or out.size == 0:
        raise ValueError(f"{name} must be a non-empty 2D mask")
    return out


def overlap_metrics_v5(source: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    src = _as_bool_mask(source, name="source")
    pred = _as_bool_mask(prediction, name="prediction")
    if src.shape != pred.shape:
        raise ValueError("source/prediction shape mismatch")
    tp = int(np.count_nonzero(src & pred))
    fp = int(np.count_nonzero((~src) & pred))
    fn = int(np.count_nonzero(src & (~pred)))
    src_n = int(np.count_nonzero(src))
    pred_n = int(np.count_nonzero(pred))
    if src_n <= 0:
        raise ValueError("source mask has no foreground")
    recall = float(tp / src_n)
    precision = float(tp / pred_n) if pred_n else 0.0
    return {
        "source_foreground_pixels": src_n,
        "predicted_foreground_pixels": pred_n,
        "true_positive_pixels": tp,
        "false_positive_pixels": fp,
        "false_negative_pixels": fn,
        "recall": recall,
        "precision": precision,
        "predicted_to_source_area_ratio": float(pred_n / src_n),
    }


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(v) == 0:
        return {"p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    return {
        "p50": float(np.quantile(v, 0.50)),
        "p90": float(np.quantile(v, 0.90)),
        "p95": float(np.quantile(v, 0.95)),
        "p99": float(np.quantile(v, 0.99)),
        "max": float(np.max(v)),
    }


def error_distance_report_v5(
    source: np.ndarray,
    prediction: np.ndarray,
    *,
    bands_px: Iterable[float] = DEFAULT_DISTANCE_BANDS_PX_V5,
) -> dict[str, Any]:
    src = _as_bool_mask(source, name="source")
    pred = _as_bool_mask(prediction, name="prediction")
    if src.shape != pred.shape:
        raise ValueError("source/prediction shape mismatch")
    bands = tuple(float(x) for x in bands_px)
    if not bands or any((not math.isfinite(x) or x <= 0) for x in bands):
        raise ValueError("bands_px must contain finite positive radii")
    if any(bands[i] >= bands[i + 1] for i in range(len(bands) - 1)):
        raise ValueError("bands_px must be strictly increasing")

    fp = pred & (~src)
    fn = src & (~pred)
    outside_distance = ndimage.distance_transform_edt(~src)
    miss_distance = ndimage.distance_transform_edt(~pred)
    fp_d = outside_distance[fp]
    fn_d = miss_distance[fn]

    def band_payload(values: np.ndarray) -> dict[str, Any]:
        total = int(len(values))
        within = {}
        for r in bands:
            key = (str(int(r)) if float(r).is_integer() else str(r)).replace(".", "_")
            count = int(np.count_nonzero(values <= r))
            within[f"within_{key}px_count"] = count
            within[f"within_{key}px_fraction"] = float(count / total) if total else 1.0
        gt_last = int(np.count_nonzero(values > bands[-1]))
        return {
            "count": total,
            **_quantiles(values),
            **within,
            "beyond_max_band_count": gt_last,
            "beyond_max_band_fraction": float(gt_last / total) if total else 0.0,
        }

    return {
        "distance_semantics": "EUCLIDEAN_PIXEL_CENTER_DISTANCE_TO_NEAREST_OPPOSITE_MASK_FOREGROUND",
        "bands_px": list(bands),
        "false_positive": band_payload(fp_d),
        "false_negative": band_payload(fn_d),
    }


def eroded_prediction_sweep_v5(
    source: np.ndarray,
    prediction: np.ndarray,
    *,
    radii_px: Iterable[float] = DEFAULT_EROSION_RADII_PX_V5,
) -> list[dict[str, Any]]:
    src = _as_bool_mask(source, name="source")
    pred = _as_bool_mask(prediction, name="prediction")
    if src.shape != pred.shape:
        raise ValueError("source/prediction shape mismatch")
    radii = tuple(float(x) for x in radii_px)
    if not radii or any((not math.isfinite(x) or x < 0) for x in radii):
        raise ValueError("radii_px must contain finite non-negative radii")
    inside_distance = ndimage.distance_transform_edt(pred)
    rows = []
    for radius in radii:
        if radius == 0.0:
            eroded = pred
        else:
            eroded = inside_distance > radius
        row = overlap_metrics_v5(src, eroded)
        row["erosion_radius_px"] = float(radius)
        rows.append(row)
    return rows


def component_count_v5(mask: np.ndarray) -> int:
    m = _as_bool_mask(mask, name="mask")
    structure = np.ones((3, 3), dtype=np.uint8)
    _labels, count = ndimage.label(m, structure=structure)
    return int(count)


def diagnose_mask_pair_v5(
    source: np.ndarray,
    prediction: np.ndarray,
    *,
    bands_px: Iterable[float] = DEFAULT_DISTANCE_BANDS_PX_V5,
    erosion_radii_px: Iterable[float] = DEFAULT_EROSION_RADII_PX_V5,
) -> dict[str, Any]:
    src = _as_bool_mask(source, name="source")
    pred = _as_bool_mask(prediction, name="prediction")
    overlap = overlap_metrics_v5(src, pred)
    distance = error_distance_report_v5(src, pred, bands_px=bands_px)
    sweep = eroded_prediction_sweep_v5(src, pred, radii_px=erosion_radii_px)
    return {
        "overlap": overlap,
        "distance": distance,
        "source_component_count_8conn": component_count_v5(src),
        "prediction_component_count_8conn": component_count_v5(pred),
        "false_positive_component_count_8conn": component_count_v5(pred & (~src)) if np.any(pred & (~src)) else 0,
        "false_negative_component_count_8conn": component_count_v5(src & (~pred)) if np.any(src & (~pred)) else 0,
        "prediction_erosion_sweep": sweep,
    }
