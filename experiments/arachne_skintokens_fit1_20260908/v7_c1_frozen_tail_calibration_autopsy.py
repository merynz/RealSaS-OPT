from __future__ import annotations

"""V7-C1 frozen tail/calibration autopsy contract.

Read-only diagnostic bound to the completed Mage A0 V7 C0 2000-step final model.
No training, backward pass, optimizer, architecture change, FSQ, sampler change, or
extra character is permitted. Teacher support projection is diagnostic-only.
"""

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class V7C1FrozenTailCalibrationContract:
    prereg_commit: str = "149376408dba94e41fff50f190dae80e8573576b"
    long_horizon_prereg_commit: str = "e58fc093bd6fc77ec3eca477e7bbe993859d9867"
    final_model_sha256: str = "ad90cc0287963d338703c667f132a3073e034d8c3b6e49e8c290d3e5b0daeaf9"
    config_hash: str = "e9d327cedb206e7ae5b074ae04b28e7de89c0e5caecb5f7c183203dbd8336fa1"
    parameter_count: int = 278_010_880
    cache_sha256: str = "db87c42d65e777072b3a607178a2c7f19ab221a4969c380eac46070db2216edd"
    binding_sha256: str = "ab74756e32ee5c9f4f2d4020cdb56620a110130d80d7b9384c62509af3f193cf"
    total_rows: int = 950
    supervised_rows: int = 934
    joint_count: int = 22
    active_epsilon: float = 1e-8
    alphas: tuple[float, ...] = (0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
    training_permitted: bool = False


CONTRACT = V7C1FrozenTailCalibrationContract()


def stable_sigmoid(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, np.float64)
    out = np.empty_like(a)
    pos = a >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-a[pos]))
    ea = np.exp(a[~pos])
    out[~pos] = ea / (1.0 + ea)
    return out


def normalize_positive_rows(x: np.ndarray, eps: float = 1e-300) -> np.ndarray:
    a = np.asarray(x, np.float64)
    if a.ndim != 2 or np.any(a < 0) or not np.isfinite(a).all():
        raise ValueError("positive finite [N,J] matrix required")
    mass = a.sum(axis=1, keepdims=True)
    if np.any(mass <= float(eps)):
        raise ValueError("zero row mass")
    return a / mass


def stable_softmax(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, np.float64)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError("finite [N,J] matrix required")
    e = np.exp(a - a.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def stable_softplus(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, np.float64)
    return np.maximum(a, 0.0) + np.log1p(np.exp(-np.abs(a)))


def map_sigmoid_normalize(logits: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    return normalize_positive_rows(stable_sigmoid(float(alpha) * np.asarray(logits, np.float64)))


def map_softmax(logits: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    return stable_softmax(float(alpha) * np.asarray(logits, np.float64))


def map_softplus_normalize(logits: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    return normalize_positive_rows(stable_softplus(float(alpha) * np.asarray(logits, np.float64)))


def qstats(x: np.ndarray) -> dict[str, float | int]:
    a = np.asarray(x, np.float64)
    if a.size == 0:
        return {"count": 0}
    return {
        "count": int(a.size),
        "min": float(a.min()), "p05": float(np.quantile(a, .05)),
        "p25": float(np.quantile(a, .25)), "p50": float(np.quantile(a, .50)),
        "p75": float(np.quantile(a, .75)), "p90": float(np.quantile(a, .90)),
        "p95": float(np.quantile(a, .95)), "p99": float(np.quantile(a, .99)),
        "max": float(a.max()), "mean": float(a.mean()), "std": float(a.std()),
    }


def teacher_support_oracle(
    pred_rows: np.ndarray,
    teacher_rows: np.ndarray,
    active_epsilon: float = CONTRACT.active_epsilon,
) -> np.ndarray:
    p = np.asarray(pred_rows, np.float64)
    t = np.asarray(teacher_rows, np.float64)
    if p.shape != t.shape or p.ndim != 2:
        raise ValueError("matching [N,J] matrices required")
    active = t > float(active_epsilon)
    out = p * active
    mass = out.sum(axis=1, keepdims=True)
    if np.any(mass <= 1e-15):
        raise ValueError("teacher-support oracle has zero predicted mass")
    return out / mass


def row_l1_summary(pred_rows: np.ndarray, teacher_rows: np.ndarray) -> dict[str, float]:
    p = np.asarray(pred_rows, np.float64)
    t = np.asarray(teacher_rows, np.float64)
    if p.shape != t.shape or p.ndim != 2:
        raise ValueError("matching [N,J] matrices required")
    row = np.abs(p - t).sum(axis=1)
    k = max(1, int(math.ceil(0.10 * len(row))))
    return {
        "mean": float(row.mean()),
        "p95": float(np.quantile(row, .95)),
        "cvar10": float(np.sort(row)[-k:].mean()),
    }


def run_contract_fixture() -> dict[str, object]:
    z = np.asarray([[4.0, 1.0, -3.0], [-2.0, 3.0, 1.0], [0.5, -1.0, 2.0]], np.float64)
    t = np.asarray([[0.7, 0.3, 0.0], [0.0, 0.6, 0.4], [0.2, 0.0, 0.8]], np.float64)
    raw_dom = z.argmax(axis=1)
    checked = 0
    for alpha in CONTRACT.alphas:
        for fn in (map_sigmoid_normalize, map_softmax, map_softplus_normalize):
            w = fn(z, alpha)
            if not np.isfinite(w).all() or np.any(w < 0):
                raise RuntimeError("fixture nonfinite/negative mapping")
            if np.max(np.abs(w.sum(axis=1) - 1.0)) > 1e-12:
                raise RuntimeError("fixture simplex failure")
            if not np.array_equal(w.argmax(axis=1), raw_dom):
                raise RuntimeError("fixture monotone-rank failure")
            checked += 1
    oracle = teacher_support_oracle(map_sigmoid_normalize(z), t)
    if np.max(np.abs(oracle.sum(axis=1) - 1.0)) > 1e-12:
        raise RuntimeError("fixture support-oracle simplex failure")
    if np.any(oracle[t <= CONTRACT.active_epsilon] != 0.0):
        raise RuntimeError("fixture support-oracle leakage")
    return {"status": "PASS", "mapping_cases": checked, "oracle_rows": int(len(oracle))}


__all__ = [
    "V7C1FrozenTailCalibrationContract", "CONTRACT",
    "stable_sigmoid", "normalize_positive_rows", "stable_softmax", "stable_softplus",
    "map_sigmoid_normalize", "map_softmax", "map_softplus_normalize",
    "qstats", "teacher_support_oracle", "row_l1_summary", "run_contract_fixture",
]
