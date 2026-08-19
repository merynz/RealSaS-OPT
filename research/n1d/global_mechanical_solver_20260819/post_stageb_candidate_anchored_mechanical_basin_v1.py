"""RealSaS N1D post-Stage-B candidate-anchored basin composition.

This is the canonical compact treatment specification recovered on 2026-08-19.
It intentionally does not contain truth-driven selection. It transforms a frozen
mechanical basin q_i using raster-derived current amplitude only.

The full replay still depends on the frozen V11 runner/candidate reconstruction
and canonical GFDR evaluator stored beside this research line.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-9


def current_amplitude_weights(current: np.ndarray, quantile: float = 0.95) -> tuple[np.ndarray, float]:
    """Return q95-normalized sqrt activity weights from frozen N1D current vectors."""
    current = np.asarray(current, dtype=np.float64)
    if current.ndim != 2 or current.shape[1] != 3:
        raise ValueError(f"current must be [N,3], got {current.shape}")
    amp = np.linalg.norm(current, axis=1)
    scale = float(np.quantile(amp, quantile)) + EPS
    w = np.sqrt(np.clip(amp / scale, 0.0, 1.0))
    return w, scale


def compose_inside_mechanical_basin(
    P_A: np.ndarray,
    q: np.ndarray,
    current: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compose P_B = P_A + w*q without granting scalar evidence free XYZ authority."""
    P_A = np.asarray(P_A, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if P_A.shape != q.shape:
        raise ValueError(f"P_A/q shape mismatch: {P_A.shape} vs {q.shape}")
    w, scale = current_amplitude_weights(current)
    if len(w) != len(P_A):
        raise ValueError("current carrier count does not match basin carrier count")
    P_B = P_A + w[:, None] * q
    return P_B.astype(np.float32), w.astype(np.float32), scale


def candidate_projection_weights(P_A: np.ndarray, q: np.ndarray, candidate_endpoints: list[np.ndarray]) -> list[np.ndarray]:
    """Project each carrier's feasible candidate endpoints onto its frozen basin axis q_i."""
    P_A = np.asarray(P_A, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    out: list[np.ndarray] = []
    for i, endpoints in enumerate(candidate_endpoints):
        E = np.asarray(endpoints, dtype=np.float64)
        D = E - P_A[i]
        den = max(float(q[i] @ q[i]), 1e-12)
        out.append(np.clip((D @ q[i]) / den, 0.0, 1.0))
    return out


def nearest_candidate_weight_distance(w: np.ndarray, projected_candidate_weights: list[np.ndarray]) -> np.ndarray:
    """Distance from each treatment weight to the nearest feasible candidate projection."""
    w = np.asarray(w, dtype=np.float64)
    d = []
    for wi, cand in zip(w, projected_candidate_weights):
        cand = np.asarray(cand, dtype=np.float64)
        d.append(float(np.min(np.abs(cand - wi))) if len(cand) else float("nan"))
    return np.asarray(d, dtype=np.float64)


def summarize_candidate_anchor(w: np.ndarray, projected_candidate_weights: list[np.ndarray]) -> dict[str, float]:
    d = nearest_candidate_weight_distance(w, projected_candidate_weights)
    finite = d[np.isfinite(d)]
    return {
        "median_nearest_candidate_projection_weight_distance": float(np.median(finite)) if len(finite) else float("nan"),
        "p95_nearest_candidate_projection_weight_distance": float(np.quantile(finite, 0.95)) if len(finite) else float("nan"),
    }


if __name__ == "__main__":
    raise SystemExit(
        "Treatment specification only. Use with the frozen V11 candidate reconstruction "
        "and canonical GFDR evaluator in research/n1d/global_mechanical_solver_20260819/."
    )
