from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Iterable

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError as exc:
    raise ImportError("rig_quality_v1 requires scipy.optimize.linear_sum_assignment") from exc


def _as_points(x, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 3 or len(a) == 0 or not np.isfinite(a).all():
        raise ValueError(f"{name} must be finite non-empty [N,3]")
    return a


def _parents(x, n: int, name: str) -> np.ndarray:
    p = np.asarray(x, dtype=np.int64)
    if p.shape != (n,):
        raise ValueError(f"{name} parent array shape mismatch")
    for child, parent in enumerate(p.tolist()):
        if parent >= n or parent == child:
            raise ValueError(f"{name} invalid parent index")
    return p


def _bones(points: np.ndarray, parents: np.ndarray) -> np.ndarray:
    rows = []
    for child, parent in enumerate(parents.tolist()):
        if parent >= 0:
            rows.append(np.stack([points[parent], points[child]], axis=0))
    if not rows:
        return np.zeros((0, 2, 3), dtype=np.float64)
    return np.stack(rows, axis=0)


def _pairwise_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)


def _point_segment_distance(points: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """Return [P,S] Euclidean distance from points to finite segments."""
    if len(segments) == 0:
        return np.full((len(points), 0), np.inf, dtype=np.float64)
    a = segments[:, 0]
    b = segments[:, 1]
    ab = b - a
    denom = np.sum(ab * ab, axis=-1)
    ap = points[:, None, :] - a[None, :, :]
    t = np.divide(
        np.einsum("psi,si->ps", ap, ab),
        denom[None, :],
        out=np.zeros((len(points), len(segments)), dtype=np.float64),
        where=denom[None, :] > 1e-15,
    )
    t = np.clip(t, 0.0, 1.0)
    closest = a[None, :, :] + t[..., None] * ab[None, :, :]
    return np.linalg.norm(points[:, None, :] - closest, axis=-1)


def _sample_bones(segments: np.ndarray, samples_per_bone: int) -> np.ndarray:
    if len(segments) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if samples_per_bone < 2:
        raise ValueError("samples_per_bone must be >=2")
    t = np.linspace(0.0, 1.0, samples_per_bone, dtype=np.float64)
    pts = segments[:, 0, None, :] * (1.0 - t[None, :, None]) + segments[:, 1, None, :] * t[None, :, None]
    return pts.reshape(-1, 3)


@dataclass(frozen=True)
class SkeletonReferenceScoresV1:
    tolerance: float
    matched_within_tolerance: int
    predicted_joint_count: int
    reference_joint_count: int
    iou: float
    precision: float
    recall: float
    cd_j2j: float
    cd_j2b: float
    cd_b2b: float

    def to_dict(self):
        return asdict(self)


def skeleton_reference_scores_v1(
    predicted_positions,
    predicted_parents,
    reference_positions,
    reference_parents,
    *,
    tolerance: float,
    bone_samples: int = 17,
) -> SkeletonReferenceScoresV1:
    """RigNet/RigAnything-family skeleton reference metrics.

    IoU/precision/recall use a Hungarian joint matching followed by a geometric
    tolerance.  Chamfer scores are symmetric mean Euclidean distances.  CD-B2B uses
    deterministic dense segment sampling; before publishing cross-paper numbers this
    implementation must be parity-checked against the exact RigNet evaluator.

    These are *reference diagnostics*, not RealSaS product PASS authority.
    """
    if not (math.isfinite(float(tolerance)) and tolerance > 0.0):
        raise ValueError("tolerance must be finite positive")
    pred = _as_points(predicted_positions, "predicted_positions")
    ref = _as_points(reference_positions, "reference_positions")
    pp = _parents(predicted_parents, len(pred), "predicted")
    rp = _parents(reference_parents, len(ref), "reference")

    dist = _pairwise_dist(pred, ref)
    row, col = linear_sum_assignment(dist)
    accepted = int(np.sum(dist[row, col] <= float(tolerance)))
    precision = accepted / float(len(pred))
    recall = accepted / float(len(ref))
    denom = len(pred) + len(ref) - accepted
    iou = accepted / float(max(denom, 1))

    cd_j2j = 0.5 * (float(dist.min(axis=1).mean()) + float(dist.min(axis=0).mean()))
    pb = _bones(pred, pp)
    rb = _bones(ref, rp)
    if len(pb) == 0 or len(rb) == 0:
        cd_j2b = math.inf
        cd_b2b = math.inf
    else:
        pred_to_ref_bone = _point_segment_distance(pred, rb).min(axis=1)
        ref_to_pred_bone = _point_segment_distance(ref, pb).min(axis=1)
        cd_j2b = 0.5 * (float(pred_to_ref_bone.mean()) + float(ref_to_pred_bone.mean()))
        ps = _sample_bones(pb, bone_samples)
        rs = _sample_bones(rb, bone_samples)
        bdist = _pairwise_dist(ps, rs)
        cd_b2b = 0.5 * (float(bdist.min(axis=1).mean()) + float(bdist.min(axis=0).mean()))

    return SkeletonReferenceScoresV1(
        tolerance=float(tolerance),
        matched_within_tolerance=accepted,
        predicted_joint_count=len(pred),
        reference_joint_count=len(ref),
        iou=float(iou),
        precision=float(precision),
        recall=float(recall),
        cd_j2j=float(cd_j2j),
        cd_j2b=float(cd_j2b),
        cd_b2b=float(cd_b2b),
    )


@dataclass(frozen=True)
class MechanicalCoverageScoresV1:
    reference_to_candidate_residual: float
    reference_to_candidate_coverage: float
    candidate_to_reference_residual: float
    candidate_to_reference_coverage: float
    reference_rank: int
    candidate_rank: int

    def to_dict(self):
        return asdict(self)


def _span_residual(source_modes: np.ndarray, basis_modes: np.ndarray, rcond: float) -> tuple[float, int]:
    # Rows are deformation modes; columns are flattened observed displacements.
    source = np.asarray(source_modes, dtype=np.float64)
    basis = np.asarray(basis_modes, dtype=np.float64)
    if source.ndim != 2 or basis.ndim != 2 or source.shape[1] != basis.shape[1]:
        raise ValueError("mode matrices must be [M,D] with common D")
    if not np.isfinite(source).all() or not np.isfinite(basis).all() or len(source) == 0 or len(basis) == 0:
        raise ValueError("mode matrices must be finite non-empty")
    # Project each source mode onto row-span(basis) through a least-squares solve.
    coeff, _, rank, _ = np.linalg.lstsq(basis.T, source.T, rcond=rcond)
    recon = (basis.T @ coeff).T
    denom = float(np.linalg.norm(source))
    residual = float(np.linalg.norm(source - recon) / max(denom, 1e-12))
    return residual, int(rank)


def mechanical_coverage_scores_v1(
    reference_modes,
    candidate_modes,
    *,
    rcond: float = 1e-6,
) -> MechanicalCoverageScoresV1:
    """Compare *functional deformation spans*, independent of joint identity/count.

    A value near 1 for reference_to_candidate_coverage means the compiled candidate
    rig can reproduce the reference rig's infinitesimal deformation subspace even if
    it uses a different number or arrangement of controls.
    """
    ref = np.asarray(reference_modes, dtype=np.float64)
    cand = np.asarray(candidate_modes, dtype=np.float64)
    r2c, cand_rank = _span_residual(ref, cand, rcond)
    c2r, ref_rank = _span_residual(cand, ref, rcond)
    return MechanicalCoverageScoresV1(
        reference_to_candidate_residual=r2c,
        reference_to_candidate_coverage=float(max(0.0, 1.0 - r2c)),
        candidate_to_reference_residual=c2r,
        candidate_to_reference_coverage=float(max(0.0, 1.0 - c2r)),
        reference_rank=ref_rank,
        candidate_rank=cand_rank,
    )


@dataclass(frozen=True)
class FiniteDeformationScoresV1:
    rms: float
    p95: float
    max_error: float

    def to_dict(self):
        return asdict(self)


def finite_deformation_scores_v1(reference_displacements, candidate_displacements) -> FiniteDeformationScoresV1:
    ref = np.asarray(reference_displacements, dtype=np.float64)
    cand = np.asarray(candidate_displacements, dtype=np.float64)
    if ref.shape != cand.shape or ref.ndim < 2 or ref.shape[-1] not in (2, 3):
        raise ValueError("deformation arrays must have matching [...,2|3] shape")
    if not np.isfinite(ref).all() or not np.isfinite(cand).all():
        raise ValueError("deformation arrays must be finite")
    err = np.linalg.norm(cand - ref, axis=-1).reshape(-1)
    return FiniteDeformationScoresV1(
        rms=float(np.sqrt(np.mean(err * err))),
        p95=float(np.quantile(err, 0.95)),
        max_error=float(err.max(initial=0.0)),
    )


def control_unique_fraction_v1(candidate_modes, *, rcond: float = 1e-6) -> np.ndarray:
    """Per-control fraction not explained by the other candidate controls.

    This returns telemetry only.  The threshold separating useful from redundant
    controls must be calibrated from GOOD/BAD rig perturbations, not hard-coded here.
    """
    modes = np.asarray(candidate_modes, dtype=np.float64)
    if modes.ndim != 2 or len(modes) < 1 or not np.isfinite(modes).all():
        raise ValueError("candidate_modes must be finite non-empty [K,D]")
    out = np.ones(len(modes), dtype=np.float64)
    for i in range(len(modes)):
        if len(modes) == 1:
            out[i] = 1.0
            continue
        basis = np.delete(modes, i, axis=0)
        residual, _ = _span_residual(modes[i : i + 1], basis, rcond)
        out[i] = residual
    return out
