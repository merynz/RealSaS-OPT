from __future__ import annotations

"""Pure binary silhouette-distance metrics for current geometry qualification."""

import numpy as np
from scipy.ndimage import distance_transform_edt

from .types import QualificationError


def _boundary(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2:
        raise QualificationError("SILHOUETTE_MASK_DIMENSION_INVALID")
    h, w = m.shape
    out = np.zeros_like(m)
    ys, xs = np.nonzero(m)
    for y, x in zip(ys, xs):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                yy = y + dy
                xx = x + dx
                if yy < 0 or yy >= h or xx < 0 or xx >= w or not m[yy, xx]:
                    out[y, x] = True
                    break
            if out[y, x]:
                break
    return out


def silhouette_distance_metrics(
    source: np.ndarray,
    predicted: np.ndarray,
) -> tuple[float, float, float]:
    a = _boundary(source)
    b = _boundary(predicted)
    if not np.any(a) or not np.any(b):
        raise QualificationError("SILHOUETTE_EMPTY_BOUNDARY")
    dist_to_b = distance_transform_edt(~b)
    dist_to_a = distance_transform_edt(~a)
    values = np.concatenate((dist_to_b[a], dist_to_a[b])).astype(np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise QualificationError("SILHOUETTE_DISTANCE_INVALID")
    return (
        float(np.mean(values)),
        float(np.percentile(values, 95.0)),
        float(np.max(values)),
    )
