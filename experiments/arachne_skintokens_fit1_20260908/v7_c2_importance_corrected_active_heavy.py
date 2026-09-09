from __future__ import annotations

"""V7-C2 importance-corrected active-heavy warm-start contract.

Two matched arms start from the exact completed 2000-step V7 C0 model. Sampling and
prefix schedules are identical. The treatment changes only the separable BCE/MSE
sample weighting from the biased proposal expectation toward the uniform-supervised
row target. Sampled Dice semantics are deliberately held unchanged between arms.
"""

from dataclasses import dataclass
import hashlib
import numpy as np
import torch
import torch.nn.functional as F


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


def build_matched_schedule(
    teacher_weights: np.ndarray,
    supervised: np.ndarray,
    *,
    steps: int = CONTRACT.optimizer_steps,
    query_count: int = CONTRACT.query_count,
    dense_fraction: float = CONTRACT.dense_fraction,
    prefix_min: int = 1,
    prefix_max: int = 4,
    seed: int = CONTRACT.seed,
    active_epsilon: float = CONTRACT.active_epsilon,
) -> tuple[dict[int, list[np.ndarray]], dict[int, list[int]], str]:
    """Build the one frozen sample/prefix schedule consumed by both C2 arms."""
    tw = np.asarray(teacher_weights, np.float64)
    sup = np.asarray(supervised, bool)
    if tw.ndim != 2 or sup.shape != (tw.shape[0],):
        raise ValueError("teacher_weights/supervised shape drift")
    if query_count <= 0 or steps <= 0 or not (0.0 < dense_fraction < 1.0):
        raise ValueError("invalid schedule cardinality")
    rng = np.random.RandomState(int(seed))
    sup_idx = np.flatnonzero(sup).astype(np.int64)
    dense_count = int(round(query_count * dense_fraction))
    uniform_count = query_count - dense_count
    samples: dict[int, list[np.ndarray]] = {}
    prefixes: dict[int, list[int]] = {}
    h = hashlib.sha256()
    for step in range(1, int(steps) + 1):
        ss: list[np.ndarray] = []
        pp: list[int] = []
        for j in range(tw.shape[1]):
            active = np.flatnonzero(sup & (tw[:, j] > float(active_epsilon))).astype(np.int64)
            if active.size == 0:
                raise ValueError(f"joint {j} has no active support")
            dense = rng.choice(active, size=dense_count, replace=(active.size < dense_count))
            uniform = rng.choice(sup_idx, size=uniform_count, replace=False)
            idx = np.concatenate([uniform, dense]).astype(np.int64)
            rng.shuffle(idx)
            prefix = int(rng.randint(int(prefix_min), int(prefix_max) + 1))
            ss.append(idx)
            pp.append(prefix)
            h.update(np.ascontiguousarray(idx, dtype=np.int64).tobytes())
            h.update(prefix.to_bytes(2, "little", signed=False))
        samples[step] = ss
        prefixes[step] = pp
    return samples, prefixes, h.hexdigest()


def sampled_scalar_loss(
    logits: torch.Tensor,
    truth: torch.Tensor,
    supervision_mask: torch.Tensor,
    sample_weights: torch.Tensor,
    *,
    arm: str,
    mse_weight: float = CONTRACT.mse_weight,
    dice_weight: float = CONTRACT.dice_weight,
    dice_epsilon: float = CONTRACT.dice_epsilon,
) -> dict[str, torch.Tensor]:
    """Exact C2 control/treatment loss. Dice semantics are common by contract."""
    if logits.shape != truth.shape or supervision_mask.shape != logits.shape or sample_weights.shape != logits.shape:
        raise ValueError("C2 sampled loss shape drift")
    with torch.autocast(device_type=logits.device.type, enabled=False):
        z = logits.float()
        t = truth.float()
        m = supervision_mask.to(torch.bool)
        p = torch.sigmoid(z)
        bce_el = F.binary_cross_entropy_with_logits(z, t, reduction="none")
        mse_el = (p - t).square()
        if arm == CONTRACT.control_name:
            bce = bce_el[m].mean()
            mse = mse_el[m].mean()
            effective_weight = torch.ones_like(z)
        elif arm == CONTRACT.treatment_name:
            w = sample_weights.float()
            if not torch.isfinite(w).all() or bool((w < 0).any()):
                raise ValueError("invalid C2 importance weights")
            bce = (w[m] * bce_el[m]).mean()
            mse = (w[m] * mse_el[m]).mean()
            effective_weight = w
        else:
            raise ValueError(f"unknown C2 arm: {arm}")

        pm = torch.where(m, p, torch.zeros_like(p))
        tm = torch.where(m, t, torch.zeros_like(t))
        reduce_dims = tuple(range(1, pm.ndim))
        numerator = 2.0 * (pm * tm).sum(dim=reduce_dims) + float(dice_epsilon)
        denominator = pm.square().sum(dim=reduce_dims) + tm.square().sum(dim=reduce_dims) + float(dice_epsilon)
        dice = (1.0 - numerator / denominator).mean()
        total = bce + float(mse_weight) * mse + float(dice_weight) * dice
        return {
            "total": total,
            "bce": bce,
            "mse": mse,
            "dice": dice,
            "l1": (p[m] - t[m]).abs().mean(),
            "mean_weight": effective_weight[m].mean(),
        }


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

    tw = np.stack([teacher, np.asarray([0.0, 1.0, 0.5, 0.0, 0.0])], axis=1)
    samples, prefixes, schedule_sha = build_matched_schedule(
        tw, sup, steps=3, query_count=4, dense_fraction=0.5, prefix_min=1, prefix_max=4, seed=17
    )
    if len(samples) != 3 or len(prefixes) != 3 or not schedule_sha:
        raise RuntimeError("fixture schedule failed")

    z = torch.tensor([[1.0, -2.0, 0.3, -0.8]], dtype=torch.float32)
    t = torch.tensor([[1.0, 0.0, 0.25, 0.0]], dtype=torch.float32)
    m = torch.ones_like(z, dtype=torch.bool)
    sw = torch.tensor([[0.5, 2.0, 0.5, 2.0]], dtype=torch.float32)
    lc = sampled_scalar_loss(z, t, m, sw, arm=CONTRACT.control_name)
    lt = sampled_scalar_loss(z, t, m, sw, arm=CONTRACT.treatment_name)
    if not torch.isfinite(lc["total"]) or not torch.isfinite(lt["total"]):
        raise RuntimeError("fixture torch loss nonfinite")
    if float(lc["dice"]) != float(lt["dice"]):
        raise RuntimeError("fixture Dice semantics drift")
    if float(lc["total"]) == float(lt["total"]):
        raise RuntimeError("fixture treatment has no effect")

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
        "schedule_sha256_fixture": schedule_sha,
        "control_total": float(lc["total"]),
        "treatment_total": float(lt["total"]),
        "shared_dice": float(lc["dice"]),
    }


__all__ = [
    "V7C2ImportanceCorrectionContract",
    "CONTRACT",
    "proposal_probabilities",
    "uniform_target_probabilities",
    "importance_weights",
    "exact_expectation_identity",
    "build_matched_schedule",
    "sampled_scalar_loss",
    "run_contract_fixture",
]
