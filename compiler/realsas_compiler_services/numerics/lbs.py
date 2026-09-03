from __future__ import annotations

"""Production LBS probe kernel promoted out of experiment space.

This module is a subordinate numerical service.  Its 4x4 matrices are typed
proof/training fixtures, not canonical RealSaS motion.  Shipping motion remains
puppet-local 2D/2.5D and compiler-owned.

Promotion origin:
    experiments/geppetto_arachne_r6_20260901/verified_lbs_v1.py
    Git blob de6fcdf2f2251d401abda8f5bd705f8f25bad393
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VerifiedLBSReportV1:
    rms: float
    p95: float
    max_error: float
    point_count: int
    probe_count: int
    schema_version: str = "RealSaS.VerifiedLBSReport.v1"


def validate_lbs_probe_inputs_v1(
    rest_points: np.ndarray,
    weights: np.ndarray,
    transforms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = np.asarray(rest_points, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    t = np.asarray(transforms, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("rest_points must be [N,3]")
    if w.ndim != 2 or w.shape[0] != len(p):
        raise ValueError("weights must be [N,J]")
    if t.ndim != 4 or t.shape[1] != w.shape[1] or t.shape[2:] != (4, 4):
        raise ValueError("transforms must be [P,J,4,4]")
    if not np.isfinite(p).all() or not np.isfinite(w).all() or not np.isfinite(t).all():
        raise ValueError("LBS inputs must be finite")
    if (w < -1e-8).any() or (np.abs(w.sum(axis=1) - 1.0) > 1e-5).any():
        raise ValueError("LBS weights must be nonnegative simplex")
    expected_last = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    if not np.allclose(t[:, :, 3, :], expected_last, atol=1e-8, rtol=0.0):
        raise ValueError("LBS probe transforms must be affine homogeneous matrices")
    return p, w, t


def apply_lbs_probe_v1(
    rest_points: np.ndarray,
    weights: np.ndarray,
    transforms: np.ndarray,
) -> np.ndarray:
    p, w, t = validate_lbs_probe_inputs_v1(rest_points, weights, transforms)
    hom = np.concatenate([p, np.ones((len(p), 1), dtype=np.float64)], axis=1)
    transformed = np.einsum("pjab,nb->pjna", t, hom)[..., :3]
    return np.einsum("nj,pjna->pna", w, transformed).astype(np.float32)


def lbs_probe_report_v1(
    predicted: np.ndarray,
    expected: np.ndarray,
) -> VerifiedLBSReportV1:
    pred = np.asarray(predicted, dtype=np.float64)
    exp = np.asarray(expected, dtype=np.float64)
    if pred.shape != exp.shape or pred.ndim != 3 or pred.shape[-1] != 3:
        raise ValueError("deformation arrays must share [P,N,3]")
    err = np.linalg.norm(pred - exp, axis=-1).reshape(-1)
    if not np.isfinite(err).all():
        raise ValueError("non-finite deformation error")
    return VerifiedLBSReportV1(
        rms=float(np.sqrt(np.mean(err ** 2))) if len(err) else 0.0,
        p95=float(np.percentile(err, 95)) if len(err) else 0.0,
        max_error=float(err.max()) if len(err) else 0.0,
        point_count=int(pred.shape[1]),
        probe_count=int(pred.shape[0]),
    )
