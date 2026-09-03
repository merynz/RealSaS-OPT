from __future__ import annotations

"""Controlled counterfactual mutations for causal proof tests.

These helpers operate on detached numerical fixtures only.  They are not repair
operators and cannot mutate a CanonicalPuppetGraphV3 production state.
"""

import numpy as np


def swap_weight_mass_v1(
    weights: np.ndarray,
    *,
    joint_a: int,
    joint_b: int,
    fraction: float = 0.5,
) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64).copy()
    if (
        w.ndim != 2
        or joint_a == joint_b
        or not (0 <= joint_a < w.shape[1])
        or not (0 <= joint_b < w.shape[1])
    ):
        raise ValueError("invalid joint mutation indices")
    if not (0.0 < fraction <= 1.0):
        raise ValueError("fraction outside (0,1]")
    moved = w[:, joint_a] * float(fraction)
    w[:, joint_a] -= moved
    w[:, joint_b] += moved
    return w.astype(np.float32)
