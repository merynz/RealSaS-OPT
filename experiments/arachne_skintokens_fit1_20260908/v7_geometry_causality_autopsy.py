from __future__ import annotations

"""Read-only geometry-causality autopsy contract for Arachne Mage A0 V7 C0.

Scientific question
-------------------
The V7 C0 biased-scalar control learns nontrivial continuous field identities but all
934 supervised rows still choose joint 8 as dominant. This diagnostic asks whether
surface/query geometry and the encoded geometry condition memory causally influence
row-level skin-weight predictions, or whether the decoder is effectively producing a
near-global field bias.

No training is permitted. The diagnostic is bound to the completed C0 final model.

Preregistered interventions
---------------------------
1. BASELINE: exact C0 final decode on all 950 rows and 22 joints.
2. QUERY_PERMUTE: deterministically permute query-geometry rows while preserving
   field tokens and baseline condition tokens. Report both invariance distance
   (shuffled output vs original row) and geometry-following/equivariance distance
   (shuffled output vs baseline prediction at the donor geometry row).
3. CONDITION_ZERO: replace encoded condition tokens with zeros while preserving
   field tokens and exact baseline query geometry.
4. CONDITION_ANCHOR_SCRAMBLE: preserve the full geometry memory set but permute its
   row association before gathering the fixed 384 condition anchors. This perturbs
   condition-query geometry while leaving the decoder query geometry untouched.
5. LOCAL_POSITION_RELOCATION: on deterministic supervised sentinel rows, replace
   only geometry channels 0:3 with the farthest supervised donor position while
   preserving the target row's remaining geometry features, field tokens, and
   baseline condition tokens. Compare relocated output both to the original target
   row and the donor baseline row.

Primary geometry-use metric
---------------------------
The primary baseline statistic is the mean pairwise L1 variation between normalized
predicted 22-joint row vectors divided by the same mean pairwise L1 variation of the
teacher rows. A ratio near zero means the model is spatially almost constant relative
to the target even if its absolute logits vary.

The interventions additionally report raw-logit L1, normalized-row L1, cosine
similarity, dominant-joint flip rate, and row-to-row variation changes.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class V7GeometryCausalityContract:
    model_sha256: str = "0e5f11cd5925b35235a83527b2ec6ef7903f7214734262576e9b8c45a77c9e53"
    model_parameter_count: int = 278_010_880
    model_config_hash: str = "e9d327cedb206e7ae5b074ae04b28e7de89c0e5caecb5f7c183203dbd8336fa1"
    c0_source_commit: str = "f2a573022f718cd390904cc4c6bfe427436bce11"
    cache_sha256: str = "db87c42d65e777072b3a607178a2c7f19ab221a4969c380eac46070db2216edd"
    binding_sha256: str = "ab74756e32ee5c9f4f2d4020cdb56620a110130d80d7b9384c62509af3f193cf"
    supervised_rows: int = 934
    total_rows: int = 950
    joint_count: int = 22
    permutation_seed: int = 20260909
    sentinel_count: int = 64
    geometry_position_channels: tuple[int, int] = (0, 3)
    training_permitted: bool = False


CONTRACT = V7GeometryCausalityContract()


def normalize_rows_from_logits(logits: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    z = np.asarray(logits, np.float64)
    p = 1.0 / (1.0 + np.exp(-z))
    return p / np.maximum(p.sum(axis=1, keepdims=True), float(eps))


def mean_pairwise_l1(rows: np.ndarray) -> float:
    x = np.asarray(rows, np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("rows must be [N,D] with N>=2")
    total = 0.0
    count = 0
    # Streaming O(N^2) without materializing the full [N,N,D] tensor.
    for i in range(x.shape[0] - 1):
        d = np.abs(x[i + 1 :] - x[i]).sum(axis=1)
        total += float(d.sum())
        count += int(d.size)
    return total / float(count)


def row_variation_ratio(pred_rows: np.ndarray, teacher_rows: np.ndarray) -> dict[str, float]:
    pv = mean_pairwise_l1(pred_rows)
    tv = mean_pairwise_l1(teacher_rows)
    return {
        "pred_mean_pairwise_row_l1": float(pv),
        "teacher_mean_pairwise_row_l1": float(tv),
        "pred_over_teacher_pairwise_row_l1_ratio": float(pv / max(tv, 1e-12)),
    }


def compare_row_matrices(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    x = np.asarray(a, np.float64)
    y = np.asarray(b, np.float64)
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("expected matching [N,J] matrices")
    l1 = np.abs(x - y).sum(axis=1)
    xn = np.linalg.norm(x, axis=1)
    yn = np.linalg.norm(y, axis=1)
    cos = (x * y).sum(axis=1) / np.maximum(xn * yn, 1e-12)
    return {
        "mean_row_l1": float(l1.mean()),
        "p95_row_l1": float(np.quantile(l1, 0.95)),
        "mean_cosine": float(cos.mean()),
        "dominant_flip_fraction": float((x.argmax(axis=1) != y.argmax(axis=1)).mean()),
    }


def deterministic_supervised_permutation(supervised: np.ndarray, seed: int = CONTRACT.permutation_seed) -> np.ndarray:
    sup = np.flatnonzero(np.asarray(supervised, bool))
    rng = np.random.RandomState(int(seed))
    donor = sup.copy()
    rng.shuffle(donor)
    # Deterministically eliminate fixed points so every supervised row is intervened.
    fixed = donor == sup
    if fixed.any():
        donor[fixed] = np.roll(donor[fixed], 1)
        # The previous operation can still leave a rare fixed point; fall back to a cyclic permutation.
        if np.any(donor == sup):
            donor = np.roll(sup, 1)
    out = np.arange(len(supervised), dtype=np.int64)
    out[sup] = donor
    return out


__all__ = [
    "V7GeometryCausalityContract",
    "CONTRACT",
    "normalize_rows_from_logits",
    "mean_pairwise_l1",
    "row_variation_ratio",
    "compare_row_matrices",
    "deterministic_supervised_permutation",
]
