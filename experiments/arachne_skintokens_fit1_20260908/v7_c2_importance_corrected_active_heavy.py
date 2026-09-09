from __future__ import annotations

"""V7-C2 importance-corrected active-heavy warm-start contract.

Two matched arms start from the exact completed 2000-step V7 C0 model. Sampling and
prefix schedules are identical. The treatment changes only the separable BCE/MSE
sample weighting from the biased proposal expectation toward the uniform-supervised
row target. Sampled Dice semantics are deliberately held unchanged between arms.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class V7C2ImportanceCorrectionContract:
    prereg_commit: str = "99038c4128259838b0286f7105501172c18191da"
    base_model_sha256: str = "ad90cc0287963d338703c667f132a3073e034d8c3b6e49e8c290d3e5b0daeaf9"
    config_hash: str = "e9d327cedb206e7ae5b074ae04b28e7de89c0e5caecb5f7c183203dbd8336fa1"
    parameter_count: int = 278_010_880
    cache_sha256: str = "db87c42d65e777072b3a607178a2c7f19ab221a4969c380eac46070db2216edd"
    binding_sha256: str = "ab74756e32ee5c9f4f2d4020cdb56620a110130d80d7b9384c62509af3f193cf"
    supervised_rows: int = 934
    joint_count: int = 22
    query_count: int = 384
    dense_fraction: float = 0.5
    optimizer_steps: int = 384
    learning_rate: float = 2.5e-5
    weight_decay: float = 1e-4
    active_epsilon: float = 1e-8
    mse_weight: float = 0.1
    dice_weight: float = 1.0
    dice_epsilon: float = 1e-4
    seed: int = 20260909
    control_name: str = "C2_CONTROL_BIASED_ACTIVE_HEAVY"
    treatment_name: str = "C2_TREATMENT_IMPORTANCE_CORRECTED_BCE_MSE"


CONTRACT = V7C2ImportanceCorrectionContract()


def proposal_probabilities(
    teacher_joint: np.ndarray,
    supervised: np.ndarray,
    *,
    dense_fraction: float = CONTRACT.dense_fraction,
    active_epsilon: float = CONTRACT.active_epsilon,
) -> np.ndarray:
    """Return q(i) over all rows for the 50% active + 50% supervised proposal."""
    t = np.asarray(teacher_joint, np.float64)
    sup = np.asarray(supervised, bool)
    if t.ndim != 1 or sup.shape != t.shape:
        raise ValueError("teacher_joint/supervised shape drift")
    ns = int(sup.sum())
    if ns <= 0:
        raise ValueError("empty supervised set")
    active = sup & (t > float(active_epsilon))
    na = int(active.sum())
    if na <= 0:
        raise ValueError("empty active support")
    q = np.zeros_like(t, dtype=np.float64)
    q[sup] += (1.0 - float(dense_fraction)) / ns
    q[active] += float(dense_fraction) / na
    if abs(float(q.sum()) - 1.0) > 1e-12:
        raise RuntimeError("proposal probability mass drift")
    return q


def uniform_target_probabilities(supervised: np.ndarray) -> np.ndarray:
    sup = np.asarray(supervised, bool)
    ns = int(sup.sum())
    if ns <= 0:
        raise ValueError("empty supervised set")
    u = np.zeros(len(sup), np.float64)
    u[sup] = 1.0 / ns
    return u


def importance_weights(
    teacher_joint: np.ndarray,
    supervised: np.ndarray,
    *,
    dense_fraction: float = CONTRACT.dense_fraction,
    active_epsilon: float = CONTRACT.active_epsilon,
) -> np.ndarray:
    q = proposal_probabilities(
        teacher_joint, supervised,
        dense_fraction=dense_fraction,
        active_epsilon=active_epsilon,
    )
    u = uniform_target_probabilities(supervised)
    w = np.zeros_like(q)
    m = q > 0
    w[m] = u[m] / q[m]
    if not np.isfinite(w).all() or np.any(w < 0):
        raise RuntimeError("importance weight nonfinite/negative")
    return w


def exact_expectation_identity(
    values: np.ndarray,
    teacher_joint: np.ndarray,
    supervised: np.ndarray,
) -> tuple[float, float]:
    """Return uniform target mean and exact proposal-weighted expectation."""
    f = np.asarray(values, np.float64)
    sup = np.asarray(supervised, bool)
    if f.ndim != 1 or f.shape != sup.shape:
        raise ValueError("values shape drift")
    q = proposal_probabilities(teacher_joint, sup)
    w = importance_weights(teacher_joint, sup)
    target = float(f[sup].mean())
    corrected = float(np.sum(q * w * f))
    return target, corrected


def run_contract_fixture() -> dict[str, object]:
    sup = np.asarray([True, True, True, True, False])
    teacher = np.asarray([1.0, 0.25, 0.0, 0.0, 0.0])
    q = proposal_probabilities(teacher, sup)
    w = importance_weights(teacher, sup)
    u = uniform_target_probabilities(sup)
    if abs(float(q.sum()) - 1.0) > 1e-12:
        raise RuntimeError("fixture q mass")
    if abs(float(np.sum(q * w)) - 1.0) > 1e-12:
        raise RuntimeError("fixture E_q[w] != 1")
    if np.max(np.abs(q * w - u)) > 1e-12:
        raise RuntimeError("fixture pointwise q*w != u")
    f = np.asarray([0.2, 1.3, 2.1, -0.4, 99.0])
    target, corrected = exact_expectation_identity(f, teacher, sup)
    if abs(target - corrected) > 1e-12:
        raise RuntimeError("fixture expectation identity failed")
    active = sup & (teacher > CONTRACT.active_epsilon)
    inactive = sup & ~active
    return {
        "status": "PASS",
        "q_mass": float(q.sum()),
        "eq_weight": float(np.sum(q * w)),
        "max_qw_minus_u": float(np.max(np.abs(q * w - u))),
        "target_mean": target,
        "corrected_expectation": corrected,
        "active_weight_values": sorted(set(map(float, w[active]))),
        "inactive_weight_values": sorted(set(map(float, w[inactive]))),
    }


__all__ = [
    "V7C2ImportanceCorrectionContract",
    "CONTRACT",
    "proposal_probabilities",
    "uniform_target_probabilities",
    "importance_weights",
    "exact_expectation_identity",
    "run_contract_fixture",
]
