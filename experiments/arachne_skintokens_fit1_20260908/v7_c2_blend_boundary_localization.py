"""V7-C2 frozen blend-boundary localization diagnostic.

No training, optimizer, backward, threshold sweep, or model mutation.
The diagnostic localizes top-4 support-displacement errors using only the
sealed C2 predictions, Mage teacher weights for evaluation, and frozen Mage
surface positions.

Important limitation: the cache does not contain the original triangle-face
adjacency used by upstream SkinTokens SamplerMix.sample_on_skin(). Therefore
this module computes a point-cloud boundary-distance proxy on the frozen Mage
surface samples; it must not be described as an exact replay of the upstream
face-mask sampler.
"""

from __future__ import annotations

import numpy as np

ACTIVE_EPS = 1e-8
TOP_K = 4


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
    if a.ndim != 2 or not np.isfinite(a).all() or np.any(a < 0):
        raise ValueError("expected finite nonnegative [N,J] scores")
    s = a.sum(axis=1, keepdims=True)
    if np.any(s <= eps):
        raise ValueError("zero row mass")
    return a / s


def top4_from_scores(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(scores, np.float64)
    if s.ndim != 2 or s.shape[1] != 22:
        raise ValueError("expected [N,22] scores")
    if not np.isfinite(s).all() or np.any(s < 0):
        raise ValueError("scores must be finite and nonnegative")
    order = np.argsort(-s, axis=1, kind="stable")
    keep = order[:, :TOP_K]
    out = np.zeros_like(s)
    rows = np.arange(len(s))[:, None]
    out[rows, keep] = s[rows, keep]
    return normalize_positive_rows(out), keep


def pairwise_euclidean(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError("points must be finite [N,3]")
    q = np.square(p).sum(axis=1, keepdims=True)
    d2 = np.maximum(q + q.T - 2.0 * (p @ p.T), 0.0)
    return np.sqrt(d2)


def empirical_percentile(value: float, population: np.ndarray) -> float:
    a = np.asarray(population, np.float64)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return float("nan")
    # Mid-rank empirical CDF: robust to equal distances.
    lt = float(np.sum(a < value))
    eq = float(np.sum(a == value))
    return (lt + 0.5 * eq) / float(len(a))


def build_boundary_tables(
    positions: np.ndarray,
    truth: np.ndarray,
    supervised_mask: np.ndarray,
    active_eps: float = ACTIVE_EPS,
) -> dict[str, np.ndarray]:
    pos_all = np.asarray(positions, np.float64)
    t_all = np.asarray(truth, np.float64)
    sup = np.asarray(supervised_mask, bool)
    if pos_all.shape[0] != t_all.shape[0] or t_all.ndim != 2:
        raise ValueError("position/truth shape mismatch")
    pos = pos_all[sup]
    t = t_all[sup]
    active = t > float(active_eps)
    dist = pairwise_euclidean(pos)
    n, jn = t.shape
    d_to_active = np.full((n, jn), np.nan, np.float64)
    d_to_inactive = np.full((n, jn), np.nan, np.float64)
    active_pct = np.full((n, jn), np.nan, np.float64)
    inactive_pct = np.full((n, jn), np.nan, np.float64)
    weight_pct = np.full((n, jn), np.nan, np.float64)

    for j in range(jn):
        aidx = np.flatnonzero(active[:, j])
        zidx = np.flatnonzero(~active[:, j])
        if len(aidx) == 0 or len(zidx) == 0:
            continue
        d_to_active[:, j] = dist[:, aidx].min(axis=1)
        d_to_inactive[:, j] = dist[:, zidx].min(axis=1)

        active_boundary = d_to_inactive[aidx, j]
        inactive_near = d_to_active[zidx, j]
        positive_weights = t[aidx, j]
        for i in aidx:
            active_pct[i, j] = empirical_percentile(d_to_inactive[i, j], active_boundary)
            weight_pct[i, j] = empirical_percentile(t[i, j], positive_weights)
        for i in zidx:
            inactive_pct[i, j] = empirical_percentile(d_to_active[i, j], inactive_near)

    return {
        "active": active,
        "d_to_active": d_to_active,
        "d_to_inactive": d_to_inactive,
        "active_boundary_percentile": active_pct,
        "inactive_near_support_percentile": inactive_pct,
        "positive_weight_percentile": weight_pct,
    }


def localize_displacements(
    positions: np.ndarray,
    truth: np.ndarray,
    supervised_mask: np.ndarray,
    raw_sigmoid_scores: np.ndarray,
    active_eps: float = ACTIVE_EPS,
) -> dict:
    truth_all = np.asarray(truth, np.float64)
    sup = np.asarray(supervised_mask, bool)
    scores_all = np.asarray(raw_sigmoid_scores, np.float64)
    if truth_all.shape != scores_all.shape:
        raise ValueError("truth/score shape mismatch")

    top4_all, keep_all = top4_from_scores(scores_all)
    t = truth_all[sup]
    top4 = top4_all[sup]
    keep = keep_all[sup]
    active = t > float(active_eps)
    pred_mask = top4 > 0
    full_contained = np.logical_or(~active, pred_mask).all(axis=1)
    displaced_rows = np.flatnonzero(~full_contained)

    tables = build_boundary_tables(positions, truth_all, sup, active_eps)
    rows = []
    missed_true_records = []
    false_kept_records = []

    for i in displaced_rows:
        missed = np.flatnonzero(active[i] & ~pred_mask[i])
        false = np.flatnonzero((~active[i]) & pred_mask[i])
        rows.append({
            "local_row": int(i),
            "missed_true_count": int(len(missed)),
            "false_kept_count": int(len(false)),
            "truth_support_count": int(active[i].sum()),
        })
        for j in missed:
            missed_true_records.append({
                "local_row": int(i),
                "joint": int(j),
                "teacher_weight": float(t[i, j]),
                "teacher_weight_percentile_within_joint_positive": float(tables["positive_weight_percentile"][i, j]),
                "distance_to_nearest_inactive": float(tables["d_to_inactive"][i, j]),
                "boundary_distance_percentile_within_joint_active": float(tables["active_boundary_percentile"][i, j]),
                "predicted_weight_after_top4": 0.0,
            })
        for j in false:
            false_kept_records.append({
                "local_row": int(i),
                "joint": int(j),
                "teacher_weight": 0.0,
                "distance_to_nearest_true_support": float(tables["d_to_active"][i, j]),
                "near_support_distance_percentile_within_joint_inactive": float(tables["inactive_near_support_percentile"][i, j]),
                "predicted_weight_after_top4": float(top4[i, j]),
            })

    def arr(records, key):
        return np.asarray([r[key] for r in records], np.float64)

    mt_bp = arr(missed_true_records, "boundary_distance_percentile_within_joint_active")
    mt_wp = arr(missed_true_records, "teacher_weight_percentile_within_joint_positive")
    fk_np = arr(false_kept_records, "near_support_distance_percentile_within_joint_inactive")

    def summarize(a):
        a = a[np.isfinite(a)]
        if len(a) == 0:
            return {"n": 0, "median": None, "q25": None, "q75": None, "fraction_le_0p25": None, "fraction_le_0p50": None}
        return {
            "n": int(len(a)),
            "median": float(np.median(a)),
            "q25": float(np.quantile(a, 0.25)),
            "q75": float(np.quantile(a, 0.75)),
            "fraction_le_0p25": float(np.mean(a <= 0.25)),
            "fraction_le_0p50": float(np.mean(a <= 0.50)),
        }

    missed_boundary_summary = summarize(mt_bp)
    missed_weight_summary = summarize(mt_wp)
    false_near_summary = summarize(fk_np)

    # Pre-registered interpretation tiers. Percentiles are joint-normalized
    # empirical ranks against all corresponding active/inactive pairs.
    strong = bool(
        missed_boundary_summary["median"] is not None
        and false_near_summary["median"] is not None
        and missed_weight_summary["median"] is not None
        and missed_boundary_summary["median"] <= 0.25
        and false_near_summary["median"] <= 0.25
        and missed_weight_summary["median"] <= 0.25
    )
    directional = bool(
        missed_boundary_summary["median"] is not None
        and false_near_summary["median"] is not None
        and missed_weight_summary["median"] is not None
        and missed_boundary_summary["median"] < 0.50
        and false_near_summary["median"] < 0.50
        and missed_weight_summary["median"] < 0.50
    )

    return {
        "displaced_row_count": int(len(displaced_rows)),
        "missed_true_pair_count": int(len(missed_true_records)),
        "false_kept_pair_count": int(len(false_kept_records)),
        "missed_true_boundary_percentiles": missed_boundary_summary,
        "missed_true_weight_percentiles": missed_weight_summary,
        "false_kept_near_support_percentiles": false_near_summary,
        "boundary_alignment_directional": directional,
        "boundary_alignment_strong": strong,
        "rows": rows,
        "missed_true_records": missed_true_records,
        "false_kept_records": false_kept_records,
    }
