from __future__ import annotations

"""Deterministic SkinTokens production-parity top-4 mapping for Arachne V7-C2.

Scientific scope: frozen post-hoc diagnostic only. No training, no threshold sweep,
no teacher-dependent support selection. K=4 is fixed from the public SkinTokens
production export contract audited at upstream commit
273b691d35989d71cd17ff2895fdc735097b92d1.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class V7C2SkinTokensTop4ParityContract:
    c2_treatment_model_sha256: str = "280d126ecb3177dfd718b651a956a65ca8a96bede1952b0b18f7d9a719bad7d0"
    config_hash: str = "e9d327cedb206e7ae5b074ae04b28e7de89c0e5caecb5f7c183203dbd8336fa1"
    cache_sha256: str = "db87c42d65e777072b3a607178a2c7f19ab221a4969c380eac46070db2216edd"
    binding_sha256: str = "ab74756e32ee5c9f4f2d4020cdb56620a110130d80d7b9384c62509af3f193cf"
    joint_count: int = 22
    supervised_rows: int = 934
    active_epsilon: float = 1e-8
    group_per_vertex: int = 4
    skintokens_upstream_commit: str = "273b691d35989d71cd17ff2895fdc735097b92d1"


CONTRACT = V7C2SkinTokensTop4ParityContract()


def stable_sigmoid_np(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, np.float64)
    out = np.empty_like(a)
    pos = a >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-a[pos]))
    ea = np.exp(a[~pos])
    out[~pos] = ea / (1.0 + ea)
    return out


def skintokens_topk_renormalize_from_scores(
    scores: np.ndarray,
    *,
    k: int = CONTRACT.group_per_vertex,
) -> np.ndarray:
    """Mirror SkinTokens export: keep row-wise top-k positive scores, renormalize retained sum."""
    s = np.asarray(scores, np.float64)
    if s.ndim != 2 or s.shape[1] != CONTRACT.joint_count:
        raise ValueError(f"expected [N,{CONTRACT.joint_count}] scores, got {s.shape}")
    if not np.isfinite(s).all() or np.any(s < 0):
        raise ValueError("scores must be finite and nonnegative")
    if not (1 <= int(k) <= s.shape[1]):
        raise ValueError("invalid top-k")

    # Stable descending sort mirrors argsort(-skin) semantics and fixes ties deterministically.
    order = np.argsort(-s, axis=1, kind="stable")
    keep = order[:, : int(k)]
    out = np.zeros_like(s)
    rows = np.arange(len(s))[:, None]
    out[rows, keep] = s[rows, keep]
    denom = out.sum(axis=1, keepdims=True)
    if np.any(denom <= 0):
        raise RuntimeError("top-k retained zero row mass")
    out /= denom
    return out


def skintokens_top4_from_logits(logits: np.ndarray) -> np.ndarray:
    return skintokens_topk_renormalize_from_scores(stable_sigmoid_np(logits), k=4)


def truth_support_histogram(
    teacher_weights: np.ndarray,
    supervised_mask: np.ndarray,
    *,
    eps: float = CONTRACT.active_epsilon,
) -> dict[int, int]:
    w = np.asarray(teacher_weights, np.float64)
    sup = np.asarray(supervised_mask, bool)
    if w.ndim != 2 or w.shape[1] != CONTRACT.joint_count or sup.shape != (w.shape[0],):
        raise ValueError("truth/support shape drift")
    counts = (w[sup] > float(eps)).sum(axis=1)
    u, c = np.unique(counts, return_counts=True)
    return {int(a): int(b) for a, b in zip(u, c)}


def run_contract_fixture() -> dict[str, object]:
    scores = np.asarray([
        [0.40, 0.30, 0.20, 0.10, 0.05, 0.01],
        [0.01, 0.02, 0.03, 0.04, 0.90, 0.80],
    ], np.float64)
    # Pad to the bound 22-joint contract.
    scores = np.pad(scores, ((0, 0), (0, CONTRACT.joint_count - scores.shape[1])))
    out = skintokens_topk_renormalize_from_scores(scores)
    if float(np.max(np.abs(out.sum(axis=1) - 1.0))) > 1e-12:
        raise RuntimeError("fixture simplex failed")
    if np.any(out < 0) or not np.isfinite(out).all():
        raise RuntimeError("fixture finite/nonnegative failed")
    if not np.all((out > 0).sum(axis=1) <= 4):
        raise RuntimeError("fixture top4 cardinality failed")
    if not np.array_equal(out.argmax(axis=1), scores.argmax(axis=1)):
        raise RuntimeError("fixture dominant rank changed")

    # Row permutation and joint permutation equivariance.
    rp = np.asarray([1, 0])
    jp = np.arange(CONTRACT.joint_count)[::-1]
    if not np.allclose(skintokens_topk_renormalize_from_scores(scores[rp]), out[rp], atol=0, rtol=0):
        raise RuntimeError("fixture row permutation equivariance failed")
    permuted = skintokens_topk_renormalize_from_scores(scores[:, jp])
    if not np.allclose(permuted[:, np.argsort(jp)], out, atol=0, rtol=0):
        raise RuntimeError("fixture joint permutation equivariance failed")

    return {
        "status": "PASS",
        "k": CONTRACT.group_per_vertex,
        "simplex_max_abs_residual": float(np.max(np.abs(out.sum(axis=1) - 1.0))),
        "max_nonzero_support": int((out > 0).sum(axis=1).max()),
        "dominant_preserved": True,
        "row_permutation_equivariant": True,
        "joint_permutation_equivariant": True,
    }


__all__ = [
    "V7C2SkinTokensTop4ParityContract",
    "CONTRACT",
    "stable_sigmoid_np",
    "skintokens_topk_renormalize_from_scores",
    "skintokens_top4_from_logits",
    "truth_support_histogram",
    "run_contract_fixture",
]
