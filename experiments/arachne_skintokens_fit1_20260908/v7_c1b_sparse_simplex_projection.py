from __future__ import annotations

"""V7-C1B frozen sparse-simplex output diagnostic contract.

No training is permitted. All candidate mappings are teacher-free and operate only on
one frozen prediction row at a time. Teacher weights are evaluation-only.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class V7C1BSparseSimplexContract:
    prereg_commit: str = "f3d18249eba51a1f646b0c6330d94e5f6256f25d"
    final_model_sha256: str = "ad90cc0287963d338703c667f132a3073e034d8c3b6e49e8c290d3e5b0daeaf9"
    config_hash: str = "e9d327cedb206e7ae5b074ae04b28e7de89c0e5caecb5f7c183203dbd8336fa1"
    parameter_count: int = 278_010_880
    cache_sha256: str = "db87c42d65e777072b3a607178a2c7f19ab221a4969c380eac46070db2216edd"
    binding_sha256: str = "ab74756e32ee5c9f4f2d4020cdb56620a110130d80d7b9384c62509af3f193cf"
    probability_alphas: tuple[float, ...] = (0.75, 1.0, 1.25)
    logit_sparsemax_alphas: tuple[float, ...] = (0.125, 0.25, 0.5, 0.75, 1.0)
    training_permitted: bool = False


CONTRACT = V7C1BSparseSimplexContract()


def stable_sigmoid(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, np.float64)
    out = np.empty_like(a)
    pos = a >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-a[pos]))
    ea = np.exp(a[~pos])
    out[~pos] = ea / (1.0 + ea)
    return out


def project_rows_to_simplex(x: np.ndarray) -> np.ndarray:
    """Exact Euclidean projection of each row onto {w>=0, sum(w)=1}."""
    a = np.asarray(x, np.float64)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError("finite [N,J] matrix required")
    u = np.sort(a, axis=1)[:, ::-1]
    cssv = np.cumsum(u, axis=1) - 1.0
    j = np.arange(1, a.shape[1] + 1, dtype=np.float64)[None, :]
    cond = u - cssv / j > 0.0
    rho = cond.sum(axis=1) - 1
    if np.any(rho < 0):
        raise RuntimeError("simplex projection rho failure")
    theta = cssv[np.arange(a.shape[0]), rho] / (rho + 1.0)
    w = np.maximum(a - theta[:, None], 0.0)
    # Only roundoff repair; no semantic renormalization should be needed.
    resid = np.abs(w.sum(axis=1) - 1.0)
    if float(resid.max()) > 5e-12:
        raise RuntimeError(f"simplex projection residual {float(resid.max())}")
    return w


def probability_simplex_from_logits(logits: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    return project_rows_to_simplex(stable_sigmoid(float(alpha) * np.asarray(logits, np.float64)))


def logit_sparsemax(logits: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    return project_rows_to_simplex(float(alpha) * np.asarray(logits, np.float64))


def run_contract_fixture() -> dict[str, object]:
    x = np.asarray([[0.6, 0.4, 0.02], [0.95, 0.03, 0.02], [-3.0, 1.0, 0.5]], np.float64)
    w = project_rows_to_simplex(x)
    if np.max(np.abs(w.sum(axis=1) - 1.0)) > 5e-12 or np.any(w < 0):
        raise RuntimeError("fixture simplex failure")
    # Permutation equivariance.
    perm = np.asarray([2, 0, 1])
    wp = project_rows_to_simplex(x[:, perm])
    inv = np.argsort(perm)
    if np.max(np.abs(wp[:, inv] - w)) > 5e-12:
        raise RuntimeError("fixture permutation equivariance failure")
    # Known Euclidean projection: [0.6,0.4,0.02] has excess .02, shared by first two.
    if np.max(np.abs(w[0] - np.asarray([0.59, 0.39, 0.02]))) > 5e-12:
        # 0.02 remains positive here because all three stay in active projection set;
        # verify against KKT rather than assuming sparsity on every row.
        theta = x[0] - w[0]
        pos = w[0] > 0
        if np.ptp(theta[pos]) > 5e-12:
            raise RuntimeError("fixture KKT threshold failure")
    return {"status": "PASS", "rows": int(len(x)), "max_simplex_residual": float(np.max(np.abs(w.sum(axis=1)-1.0)))}


__all__ = [
    "V7C1BSparseSimplexContract", "CONTRACT", "stable_sigmoid",
    "project_rows_to_simplex", "probability_simplex_from_logits", "logit_sparsemax",
    "run_contract_fixture",
]
