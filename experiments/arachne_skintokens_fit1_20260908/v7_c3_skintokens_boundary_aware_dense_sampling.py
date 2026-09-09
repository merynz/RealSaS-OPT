from __future__ import annotations

"""V7-C3 matched boundary-aware dense-sampling continuation contract.

The only causal treatment is dense-pool membership:
- control: exact-positive supervised rows for each joint;
- treatment: exact-positive rows plus a point-cloud near-support neighborhood.

The treatment is a proxy port of upstream SkinTokens SamplerMix.sample_on_skin(); the
bound Mage cache has no triangle-face adjacency, so this file does not claim exact
face-mask replay. Public upstream source-code defaults max_distance=0.1 and
rate_distance=0.1 are used without claiming they were the final paper-training
overrides.
"""

from dataclasses import dataclass
import hashlib
import numpy as np


@dataclass(frozen=True)
class V7C3BoundarySamplingContract:
    prereg_commit: str = "fe036d87b41ae3387c8fed9aa3a2ed11ff0ab13f"
    parent_localization_result_commit: str = "26d994b25f83096623ac78d804c715660ede319a"
    base_model_sha256: str = "280d126ecb3177dfd718b651a956a65ca8a96bede1952b0b18f7d9a719bad7d0"
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
    upstream_max_distance_default: float = 0.1
    upstream_rate_distance_default: float = 0.1
    seed: int = 20260909
    control_name: str = "C3_CONTROL_C2_ACTIVE_ONLY"
    treatment_name: str = "C3_TREATMENT_BOUNDARY_PROXY"


CONTRACT = V7C3BoundarySamplingContract()


def _validate_inputs(
    positions: np.ndarray,
    teacher_joint: np.ndarray,
    supervised: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyz = np.asarray(positions, np.float64)
    t = np.asarray(teacher_joint, np.float64)
    sup = np.asarray(supervised, bool)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("positions must be [N,3]")
    if t.ndim != 1 or sup.ndim != 1 or len(t) != len(sup) or len(t) != len(xyz):
        raise ValueError("C3 input shape drift")
    if not np.isfinite(xyz).all() or not np.isfinite(t).all():
        raise ValueError("C3 nonfinite inputs")
    if int(sup.sum()) <= 0:
        raise ValueError("C3 empty supervised set")
    return xyz, t, sup


def exact_positive_pool(
    teacher_joint: np.ndarray,
    supervised: np.ndarray,
    *,
    active_epsilon: float = CONTRACT.active_epsilon,
) -> np.ndarray:
    t = np.asarray(teacher_joint, np.float64)
    sup = np.asarray(supervised, bool)
    if t.ndim != 1 or sup.shape != t.shape:
        raise ValueError("C3 exact-positive shape drift")
    pool = np.flatnonzero(sup & (t > float(active_epsilon))).astype(np.int64)
    if pool.size == 0:
        raise ValueError("C3 joint has no exact-positive support")
    return pool


def support_bbox_diagonal(positions: np.ndarray, support_indices: np.ndarray) -> float:
    xyz = np.asarray(positions, np.float64)
    idx = np.asarray(support_indices, np.int64)
    if idx.ndim != 1 or idx.size == 0:
        raise ValueError("C3 empty support for bbox")
    p = xyz[idx]
    return float(np.linalg.norm(p.max(axis=0) - p.min(axis=0)))


def boundary_radius(
    support_span: float,
    *,
    max_distance: float = CONTRACT.upstream_max_distance_default,
    rate_distance: float = CONTRACT.upstream_rate_distance_default,
) -> float:
    if support_span < 0 or max_distance <= 0 or rate_distance <= 0:
        raise ValueError("C3 invalid boundary radius inputs")
    return float(min(float(max_distance), float(support_span) * float(rate_distance)))


def nearest_support_distance(
    positions: np.ndarray,
    query_indices: np.ndarray,
    support_indices: np.ndarray,
    *,
    block: int = 256,
) -> np.ndarray:
    """Exact Euclidean nearest-support distance without scipy dependency."""
    xyz = np.asarray(positions, np.float64)
    qidx = np.asarray(query_indices, np.int64)
    sidx = np.asarray(support_indices, np.int64)
    if qidx.ndim != 1 or sidx.ndim != 1 or sidx.size == 0:
        raise ValueError("C3 nearest-support index drift")
    support = xyz[sidx]
    out = np.empty(qidx.size, np.float64)
    for s in range(0, qidx.size, int(block)):
        q = xyz[qidx[s:s + int(block)]]
        d2 = ((q[:, None, :] - support[None, :, :]) ** 2).sum(axis=-1)
        out[s:s + len(q)] = np.sqrt(d2.min(axis=1))
    return out


def boundary_proxy_pool(
    positions: np.ndarray,
    teacher_joint: np.ndarray,
    supervised: np.ndarray,
    *,
    active_epsilon: float = CONTRACT.active_epsilon,
    max_distance: float = CONTRACT.upstream_max_distance_default,
    rate_distance: float = CONTRACT.upstream_rate_distance_default,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Support + near-support point-cloud proxy for upstream SkinTokens dense mask."""
    xyz, t, sup = _validate_inputs(positions, teacher_joint, supervised)
    active = exact_positive_pool(t, sup, active_epsilon=active_epsilon)
    sup_idx = np.flatnonzero(sup).astype(np.int64)
    span = support_bbox_diagonal(xyz, active)
    radius = boundary_radius(span, max_distance=max_distance, rate_distance=rate_distance)
    dist = nearest_support_distance(xyz, sup_idx, active)
    # Upstream uses strict distance < threshold. Exact-positive rows are explicitly
    # unioned so a degenerate radius cannot accidentally remove true support.
    near = sup_idx[dist < radius]
    pool = np.unique(np.concatenate([active, near])).astype(np.int64)
    if not np.all(sup[pool]):
        raise RuntimeError("C3 boundary pool admitted unsupervised row")
    if not np.all(np.isin(active, pool)):
        raise RuntimeError("C3 boundary pool lost exact-positive support")
    return pool, {
        "active_count": int(active.size),
        "pool_count": int(pool.size),
        "added_near_support_count": int(pool.size - active.size),
        "support_bbox_diagonal": float(span),
        "radius": float(radius),
    }


def proposal_probabilities_from_pool(
    supervised: np.ndarray,
    dense_pool_indices: np.ndarray,
    *,
    dense_fraction: float = CONTRACT.dense_fraction,
) -> np.ndarray:
    sup = np.asarray(supervised, bool)
    pool = np.asarray(dense_pool_indices, np.int64)
    if not (0.0 < dense_fraction < 1.0):
        raise ValueError("C3 dense_fraction must be in (0,1)")
    ns = int(sup.sum())
    if ns <= 0 or pool.ndim != 1 or pool.size <= 0:
        raise ValueError("C3 proposal empty set")
    if not np.all(sup[pool]):
        raise ValueError("C3 proposal pool includes unsupervised rows")
    if np.unique(pool).size != pool.size:
        raise ValueError("C3 proposal pool must be unique")
    q = np.zeros(len(sup), np.float64)
    q[sup] += (1.0 - float(dense_fraction)) / ns
    q[pool] += float(dense_fraction) / pool.size
    if abs(float(q.sum()) - 1.0) > 1e-12:
        raise RuntimeError("C3 proposal probability mass drift")
    return q


def uniform_target_probabilities(supervised: np.ndarray) -> np.ndarray:
    sup = np.asarray(supervised, bool)
    ns = int(sup.sum())
    if ns <= 0:
        raise ValueError("C3 empty supervised target")
    u = np.zeros(len(sup), np.float64)
    u[sup] = 1.0 / ns
    return u


def importance_weights_from_pool(
    supervised: np.ndarray,
    dense_pool_indices: np.ndarray,
    *,
    dense_fraction: float = CONTRACT.dense_fraction,
) -> np.ndarray:
    q = proposal_probabilities_from_pool(supervised, dense_pool_indices, dense_fraction=dense_fraction)
    u = uniform_target_probabilities(supervised)
    w = np.zeros_like(q)
    m = q > 0
    w[m] = u[m] / q[m]
    if not np.isfinite(w).all() or np.any(w < 0):
        raise RuntimeError("C3 invalid importance weights")
    return w


def _subseed(seed: int, step: int, joint: int, stream: int) -> int:
    payload = f"{int(seed)}:{int(step)}:{int(joint)}:{int(stream)}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little", signed=False)


def build_matched_boundary_schedules(
    teacher_weights: np.ndarray,
    supervised: np.ndarray,
    positions: np.ndarray,
    *,
    steps: int = CONTRACT.optimizer_steps,
    query_count: int = CONTRACT.query_count,
    dense_fraction: float = CONTRACT.dense_fraction,
    prefix_min: int = 1,
    prefix_max: int = 4,
    seed: int = CONTRACT.seed,
) -> tuple[dict[str, dict[int, list[np.ndarray]]], dict[int, list[int]], dict[str, object]]:
    """Build matched C3 schedules.

    Global draws and prefix lengths are exact-common across arms. Dense draws use
    identical deterministic sub-seeds but map into each arm's own dense pool.
    """
    tw = np.asarray(teacher_weights, np.float64)
    sup = np.asarray(supervised, bool)
    xyz = np.asarray(positions, np.float64)
    if tw.ndim != 2 or sup.shape != (tw.shape[0],) or xyz.shape != (tw.shape[0], 3):
        raise ValueError("C3 schedule input drift")
    if steps <= 0 or query_count <= 0 or not (0 < dense_fraction < 1):
        raise ValueError("C3 invalid schedule cardinality")
    sup_idx = np.flatnonzero(sup).astype(np.int64)
    dense_count = int(round(query_count * dense_fraction))
    global_count = query_count - dense_count

    pools_control: list[np.ndarray] = []
    pools_treatment: list[np.ndarray] = []
    pool_summary: list[dict[str, object]] = []
    for j in range(tw.shape[1]):
        c = exact_positive_pool(tw[:, j], sup)
        t, meta = boundary_proxy_pool(xyz, tw[:, j], sup)
        pools_control.append(c)
        pools_treatment.append(t)
        pool_summary.append({"joint": j, **meta})

    schedules = {CONTRACT.control_name: {}, CONTRACT.treatment_name: {}}
    prefixes: dict[int, list[int]] = {}
    hashes = {CONTRACT.control_name: hashlib.sha256(), CONTRACT.treatment_name: hashlib.sha256()}
    h_common = hashlib.sha256()

    for step in range(1, int(steps) + 1):
        global_rows: list[np.ndarray] = []
        prefix_rows: list[int] = []
        for j in range(tw.shape[1]):
            rg = np.random.RandomState(_subseed(seed, step, j, 1))
            global_idx = rg.choice(sup_idx, size=global_count, replace=False).astype(np.int64)
            rp = np.random.RandomState(_subseed(seed, step, j, 2))
            prefix = int(rp.randint(int(prefix_min), int(prefix_max) + 1))
            global_rows.append(global_idx)
            prefix_rows.append(prefix)
        prefixes[step] = prefix_rows

        for arm, pools in ((CONTRACT.control_name, pools_control), (CONTRACT.treatment_name, pools_treatment)):
            ss: list[np.ndarray] = []
            for j, pool in enumerate(pools):
                rd = np.random.RandomState(_subseed(seed, step, j, 3))
                dense = rd.choice(pool, size=dense_count, replace=(pool.size < dense_count)).astype(np.int64)
                idx = np.concatenate([global_rows[j], dense]).astype(np.int64)
                # Common shuffle seed: same permutation positions in both arms.
                rs = np.random.RandomState(_subseed(seed, step, j, 4))
                rs.shuffle(idx)
                ss.append(idx)
                hashes[arm].update(np.ascontiguousarray(idx, dtype=np.int64).tobytes())
                hashes[arm].update(prefix_rows[j].to_bytes(2, "little", signed=False))
                if arm == CONTRACT.control_name:
                    h_common.update(np.ascontiguousarray(global_rows[j], dtype=np.int64).tobytes())
                    h_common.update(prefix_rows[j].to_bytes(2, "little", signed=False))
            schedules[arm][step] = ss

    return schedules, prefixes, {
        "control_schedule_sha256": hashes[CONTRACT.control_name].hexdigest(),
        "treatment_schedule_sha256": hashes[CONTRACT.treatment_name].hexdigest(),
        "common_global_prefix_sha256": h_common.hexdigest(),
        "pool_summary": pool_summary,
        "strict_expansion_joint_count": int(sum(int(x["pool_count"]) > int(x["active_count"]) for x in pool_summary)),
    }


def run_contract_fixture() -> dict[str, object]:
    # Two separated positive clusters with nearby supervised negatives.
    xyz = np.array([
        [0.00, 0.00, 0.00],
        [0.10, 0.00, 0.00],
        [0.12, 0.00, 0.00],
        [0.50, 0.00, 0.00],
        [0.90, 0.00, 0.00],
        [9.00, 9.00, 9.00],
    ], dtype=np.float64)
    sup = np.array([1, 1, 1, 1, 1, 0], dtype=bool)
    teacher = np.array([1.0, 0.4, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    active = exact_positive_pool(teacher, sup)
    pool, meta = boundary_proxy_pool(xyz, teacher, sup, max_distance=0.1, rate_distance=1.0)
    if not np.all(np.isin(active, pool)):
        raise RuntimeError("C3 fixture lost active")
    if 2 not in pool or 4 in pool or 5 in pool:
        raise RuntimeError(f"C3 fixture neighborhood membership failed::{pool.tolist()}")
    q = proposal_probabilities_from_pool(sup, pool)
    w = importance_weights_from_pool(sup, pool)
    u = uniform_target_probabilities(sup)
    if abs(float(q.sum()) - 1.0) > 1e-12:
        raise RuntimeError("C3 fixture q mass")
    if abs(float(np.sum(q * w)) - 1.0) > 1e-12:
        raise RuntimeError("C3 fixture E_q[w]")
    if float(np.max(np.abs(q * w - u))) > 1e-12:
        raise RuntimeError("C3 fixture q*w != u")

    tw = np.stack([teacher, np.array([0.0, 0.0, 1.0, 0.2, 0.0, 0.0])], axis=1)
    schedules, prefixes, summary = build_matched_boundary_schedules(
        tw, sup, xyz, steps=3, query_count=4, dense_fraction=0.5, prefix_min=1, prefix_max=4, seed=17
    )
    if len(schedules[CONTRACT.control_name]) != 3 or len(prefixes) != 3:
        raise RuntimeError("C3 fixture schedule cardinality")
    if int(summary["strict_expansion_joint_count"]) <= 0:
        raise RuntimeError("C3 fixture treatment has no pool expansion")
    return {
        "status": "PASS",
        "active": active.tolist(),
        "boundary_pool": pool.tolist(),
        "radius": float(meta["radius"]),
        "q_mass": float(q.sum()),
        "eq_weight": float(np.sum(q * w)),
        "max_qw_minus_u": float(np.max(np.abs(q * w - u))),
        "strict_expansion_joint_count": int(summary["strict_expansion_joint_count"]),
        "control_schedule_sha256": str(summary["control_schedule_sha256"]),
        "treatment_schedule_sha256": str(summary["treatment_schedule_sha256"]),
        "common_global_prefix_sha256": str(summary["common_global_prefix_sha256"]),
    }


__all__ = [
    "V7C3BoundarySamplingContract",
    "CONTRACT",
    "exact_positive_pool",
    "support_bbox_diagonal",
    "boundary_radius",
    "nearest_support_distance",
    "boundary_proxy_pool",
    "proposal_probabilities_from_pool",
    "uniform_target_probabilities",
    "importance_weights_from_pool",
    "build_matched_boundary_schedules",
    "run_contract_fixture",
]
