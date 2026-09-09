from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


SCHEMA = "RealSaS.Arachne.V7RowObjectiveDiagnostic.v1"


def _entropy_rows(w: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(w, np.float64)
    return -(x * np.log(np.clip(x, eps, None))).sum(axis=1)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    a = np.asarray(x, np.float64)
    b = np.asarray(y, np.float64)
    if len(a) < 2 or np.std(a) <= 1e-15 or np.std(b) <= 1e-15:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _rankdata_average(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), np.float64)
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and a[order[j]] == a[order[i]]:
            j += 1
        r = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = r
        i = j
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return _pearson(_rankdata_average(np.asarray(x)), _rankdata_average(np.asarray(y)))


def _qstats(x: np.ndarray) -> dict[str, float]:
    a = np.asarray(x, np.float64)
    return {
        "min": float(np.min(a)), "p05": float(np.quantile(a, 0.05)),
        "p25": float(np.quantile(a, 0.25)), "p50": float(np.quantile(a, 0.50)),
        "p75": float(np.quantile(a, 0.75)), "p90": float(np.quantile(a, 0.90)),
        "p95": float(np.quantile(a, 0.95)), "p99": float(np.quantile(a, 0.99)),
        "max": float(np.max(a)), "mean": float(np.mean(a)), "std": float(np.std(a)),
    }


def expected_sampling_weights(truth: np.ndarray, supervised: np.ndarray, *, active_eps: float = 1e-8,
                              dense_fraction: float = 0.5, sample_count: int = 384):
    """Expected V7 decoder-sampler point distribution q[N,J]."""
    t = np.asarray(truth, np.float64)
    sup = np.asarray(supervised, bool)
    n, jn = t.shape
    if sup.shape != (n,):
        raise ValueError("supervised mask shape drift")
    ns = int(sup.sum())
    if ns <= 0:
        raise ValueError("empty supervision")
    q = np.zeros((n, jn), np.float64)
    active_counts = np.zeros(jn, np.int64)
    ratios = np.zeros(jn, np.float64)
    for j in range(jn):
        active = sup & (t[:, j] > active_eps)
        na = int(active.sum())
        if na <= 0:
            raise ValueError(f"joint {j} has no active supervised support")
        active_counts[j] = na
        q[sup, j] += (1.0 - dense_fraction) / ns
        q[active, j] += dense_fraction / na
        inactive_prob = (1.0 - dense_fraction) / ns
        active_prob = inactive_prob + dense_fraction / na
        ratios[j] = active_prob / inactive_prob
    if not np.allclose(q.sum(axis=0), 1.0, atol=1e-10):
        raise RuntimeError("sampling weights do not sum to one")
    return q, active_counts, ratios


def _objective_tensors(logits: torch.Tensor, truth: torch.Tensor, supervised: torch.Tensor, q_np: np.ndarray,
                       *, mse_weight: float = 0.1, dice_weight: float = 1.0,
                       dice_eps: float = 1e-4, sample_count: int = 384):
    if logits.shape != truth.shape or logits.ndim != 2:
        raise ValueError("logits/truth must be [N,J]")
    sup2 = supervised[:, None].expand_as(logits)
    p = torch.sigmoid(logits)
    z = logits[sup2]
    tflat = truth[sup2]
    pflat = p[sup2]
    ubce = F.binary_cross_entropy_with_logits(z, tflat)
    umse = F.mse_loss(pflat, tflat)
    ps = torch.where(sup2, p, torch.zeros_like(p))
    ts = torch.where(sup2, truth, torch.zeros_like(truth))
    unumer = 2.0 * (ps * ts).sum(dim=0) + float(dice_eps)
    udenom = ps.square().sum(dim=0) + ts.square().sum(dim=0) + float(dice_eps)
    udice = (1.0 - unumer / udenom).mean()
    scalar_uniform = ubce + float(mse_weight) * umse + float(dice_weight) * udice

    q = torch.as_tensor(q_np, dtype=logits.dtype, device=logits.device)
    bce_el = F.binary_cross_entropy_with_logits(logits, truth, reduction="none")
    mse_el = (p - truth).square()
    sbce = (q * bce_el).sum(dim=0).mean()
    smse = (q * mse_el).sum(dim=0).mean()
    counts = q * float(sample_count)
    snumer = 2.0 * (counts * p * truth).sum(dim=0) + float(dice_eps)
    sdenom = (counts * p.square()).sum(dim=0) + (counts * truth.square()).sum(dim=0) + float(dice_eps)
    sdice = (1.0 - snumer / sdenom).mean()
    scalar_sampled_expected = sbce + float(mse_weight) * smse + float(dice_weight) * sdice

    pn = p / p.sum(dim=1, keepdim=True).clamp_min(1e-8)
    row_l1 = (pn - truth).abs().sum(dim=1)
    coupled_mean_l1 = row_l1[supervised].mean()
    k = max(1, int(math.ceil(0.10 * int(supervised.sum().item()))))
    coupled_cvar10_l1 = torch.topk(row_l1[supervised], k=k, largest=True).values.mean()
    coupled_mse = (pn[supervised] - truth[supervised]).square().mean()
    coupled_ce = -(truth[supervised] * torch.log(pn[supervised].clamp_min(1e-8))).sum(dim=1).mean()
    return {
        "scalar_uniform_bce": ubce, "scalar_uniform_mse": umse, "scalar_uniform_dice": udice,
        "scalar_uniform_total": scalar_uniform,
        "scalar_sampled_expected_bce": sbce, "scalar_sampled_expected_mse": smse,
        "scalar_sampled_expected_dice": sdice, "scalar_sampled_expected_total": scalar_sampled_expected,
        "coupled_mean_row_l1": coupled_mean_l1, "coupled_cvar10_row_l1": coupled_cvar10_l1,
        "coupled_row_mse": coupled_mse, "coupled_row_cross_entropy": coupled_ce,
    }


def _grad_for(obj: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(obj, z, retain_graph=True, create_graph=False)[0]


def _cos(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> float:
    m = mask[:, None].expand_as(a)
    x, y = a[m].float().reshape(-1), b[m].float().reshape(-1)
    den = x.norm() * y.norm()
    return float("nan") if float(den) <= 1e-20 else float(torch.dot(x, y) / den)


def _rms(g: torch.Tensor, mask: torch.Tensor) -> float:
    x = g[mask[:, None].expand_as(g)].float()
    return float(torch.sqrt(torch.mean(x.square())).detach().cpu())


def _eval_row_metrics(z: torch.Tensor, truth: torch.Tensor, supervised: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        p = torch.sigmoid(z)
        pn = p / p.sum(dim=1, keepdim=True).clamp_min(1e-8)
        row = (pn - truth).abs().sum(dim=1)[supervised]
        return {
            "mean_row_l1": float(row.mean().cpu()),
            "p95_row_l1": float(torch.quantile(row.float(), 0.95).cpu()),
            "cvar10_row_l1": float(torch.topk(row, k=max(1, int(math.ceil(0.10 * row.numel())))).values.mean().cpu()),
            "raw_mass_mean": float(p.sum(dim=1)[supervised].mean().cpu()),
        }


def analyze_v7_logits(raw_logits: np.ndarray, truth: np.ndarray, supervised: np.ndarray,
                      surface_ids, joint_ids, *, active_eps: float = 1e-8,
                      dense_fraction: float = 0.5, sample_count: int = 384,
                      mse_weight: float = 0.1, dice_weight: float = 1.0,
                      dice_eps: float = 1e-4):
    z_np, t_np, sup = np.asarray(raw_logits, np.float64), np.asarray(truth, np.float64), np.asarray(supervised, bool)
    if z_np.shape != t_np.shape or z_np.ndim != 2:
        raise ValueError("raw_logits/truth shape drift")
    n, jn = z_np.shape
    if len(surface_ids) != n or len(joint_ids) != jn or sup.shape != (n,):
        raise ValueError("id/mask shape drift")
    p = 1.0 / (1.0 + np.exp(-z_np))
    mass = p.sum(axis=1)
    pn = p / np.maximum(mass[:, None], 1e-8)
    row_l1 = np.abs(pn - t_np).sum(axis=1)
    raw_row_l1 = np.abs(p - t_np).sum(axis=1)
    active_count = (t_np > active_eps).sum(axis=1)
    teacher_entropy, pred_entropy = _entropy_rows(t_np), _entropy_rows(pn)
    teacher_dom, pred_dom = np.argmax(t_np, axis=1), np.argmax(pn, axis=1)
    dom_ok = teacher_dom == pred_dom
    pure, blend = active_count == 1, active_count > 1
    q, joint_active_counts, active_ratios = expected_sampling_weights(t_np, sup, active_eps=active_eps,
                                                                      dense_fraction=dense_fraction, sample_count=sample_count)

    def group(mask):
        m = sup & mask
        if not np.any(m):
            return {"count": 0}
        return {
            "count": int(m.sum()), "row_l1": _qstats(row_l1[m]), "raw_scalar_row_l1": _qstats(raw_row_l1[m]),
            "raw_mass": _qstats(mass[m]), "dominant_joint_accuracy": float(dom_ok[m].mean()),
            "teacher_entropy_mean": float(teacher_entropy[m].mean()), "pred_entropy_mean": float(pred_entropy[m].mean()),
        }

    threshold_counts = {str(x): {"count": int(np.sum(sup & (row_l1 > x))),
                                 "fraction": float(np.mean(row_l1[sup] > x))}
                        for x in [0.05, 0.10, 0.25, 0.50, 1.00, 1.50]}
    corr_features = {
        "raw_mass": mass, "raw_mass_abs_error_from_1": np.abs(mass - 1.0),
        "raw_scalar_row_l1": raw_row_l1, "teacher_entropy": teacher_entropy,
        "teacher_active_count": active_count.astype(np.float64), "pred_entropy": pred_entropy,
    }
    correlations = {name: {"pearson_with_normalized_row_l1": _pearson(v[sup], row_l1[sup]),
                           "spearman_with_normalized_row_l1": _spearman(v[sup], row_l1[sup])}
                    for name, v in corr_features.items()}

    z = torch.tensor(z_np, dtype=torch.float32, requires_grad=True)
    t = torch.tensor(t_np, dtype=torch.float32)
    sm = torch.tensor(sup, dtype=torch.bool)
    objs = _objective_tensors(z, t, sm, q, mse_weight=mse_weight, dice_weight=dice_weight,
                              dice_eps=dice_eps, sample_count=sample_count)
    g_uniform = _grad_for(objs["scalar_uniform_total"], z)
    g_sample = _grad_for(objs["scalar_sampled_expected_total"], z)
    g_row = _grad_for(objs["coupled_mean_row_l1"], z)
    g_cvar = _grad_for(objs["coupled_cvar10_row_l1"], z)
    gradient_alignment = {
        "cos_scalar_uniform_vs_coupled_mean_row_l1": _cos(g_uniform, g_row, sm),
        "cos_scalar_sampled_expected_vs_coupled_mean_row_l1": _cos(g_sample, g_row, sm),
        "cos_scalar_uniform_vs_cvar10_row_l1": _cos(g_uniform, g_cvar, sm),
        "cos_scalar_sampled_expected_vs_cvar10_row_l1": _cos(g_sample, g_cvar, sm),
        "cos_scalar_uniform_vs_scalar_sampled_expected": _cos(g_uniform, g_sample, sm),
        "gradient_rms": {"scalar_uniform": _rms(g_uniform, sm), "scalar_sampled_expected": _rms(g_sample, sm),
                         "coupled_mean_row_l1": _rms(g_row, sm), "coupled_cvar10_row_l1": _rms(g_cvar, sm)},
    }
    baseline_local = _eval_row_metrics(z.detach(), t, sm)
    local_steps = {"baseline": baseline_local, "logit_rms_step_sizes": [0.005, 0.02, 0.05]}
    for name, g in {"scalar_uniform_descent": g_uniform, "scalar_sampled_expected_descent": g_sample,
                    "coupled_mean_row_l1_descent": g_row, "coupled_cvar10_row_l1_descent": g_cvar}.items():
        grms = _rms(g, sm)
        runs = []
        for eps in (0.005, 0.02, 0.05):
            zm = z.detach().clone() if not np.isfinite(grms) or grms <= 1e-20 else z.detach() - eps * g.detach() / grms
            mm = _eval_row_metrics(zm, t, sm)
            mm.update({"epsilon": eps,
                       "delta_mean_row_l1": mm["mean_row_l1"] - baseline_local["mean_row_l1"],
                       "delta_p95_row_l1": mm["p95_row_l1"] - baseline_local["p95_row_l1"],
                       "delta_cvar10_row_l1": mm["cvar10_row_l1"] - baseline_local["cvar10_row_l1"]})
            runs.append(mm)
        local_steps[name] = runs

    active_count_groups = {str(k): group(active_count == k) for k in sorted(set(map(int, active_count[sup])))}
    report = {
        "schema": SCHEMA, "shape": {"rows": n, "joints": jn, "supervised_rows": int(sup.sum())},
        "normalized_row_l1": _qstats(row_l1[sup]), "raw_scalar_row_l1": _qstats(raw_row_l1[sup]),
        "raw_scalar_element_mae": float(np.abs(p[sup] - t_np[sup]).mean()), "raw_mass": _qstats(mass[sup]),
        "raw_mass_abs_error_from_1": _qstats(np.abs(mass[sup] - 1.0)), "thresholds": threshold_counts,
        "dominant_joint_accuracy": float(dom_ok[sup].mean()),
        "groups": {"pure": group(pure), "blend": group(blend), "active_count": active_count_groups},
        "correlations": correlations,
        "sampling_bias": {"dense_fraction": dense_fraction, "sample_count": sample_count,
                          "active_support_counts": [int(x) for x in joint_active_counts],
                          "active_vs_inactive_expected_sample_probability_ratio": {
                              "min": float(active_ratios.min()), "median": float(np.median(active_ratios)),
                              "max": float(active_ratios.max()), "per_joint": [float(x) for x in active_ratios]}},
        "objective_values": {k: float(v.detach().cpu()) for k, v in objs.items()},
        "gradient_alignment": gradient_alignment, "local_logit_descent_probe": local_steps,
    }

    rows = []
    for i in range(n):
        teacher_order, pred_order = np.argsort(-t_np[i])[:3], np.argsort(-pn[i])[:3]
        rows.append({
            "surface_index": i, "surface_id": str(surface_ids[i]), "supervised": bool(sup[i]),
            "group": "pure" if pure[i] else ("blend" if blend[i] else "zero"),
            "teacher_active_count": int(active_count[i]), "teacher_entropy": float(teacher_entropy[i]),
            "pred_entropy": float(pred_entropy[i]), "raw_mass": float(mass[i]),
            "raw_mass_abs_error_from_1": float(abs(mass[i] - 1.0)), "raw_scalar_row_l1": float(raw_row_l1[i]),
            "normalized_row_l1": float(row_l1[i]), "teacher_dominant_joint_index": int(teacher_dom[i]),
            "teacher_dominant_joint_id": str(joint_ids[teacher_dom[i]]),
            "teacher_dominant_weight": float(t_np[i, teacher_dom[i]]), "pred_dominant_joint_index": int(pred_dom[i]),
            "pred_dominant_joint_id": str(joint_ids[pred_dom[i]]), "pred_dominant_weight": float(pn[i, pred_dom[i]]),
            "dominant_correct": bool(dom_ok[i]),
            "teacher_top3": [(str(joint_ids[j]), float(t_np[i, j])) for j in teacher_order],
            "pred_top3": [(str(joint_ids[j]), float(pn[i, j])) for j in pred_order],
        })

    per_joint = []
    for j in range(jn):
        act, ina = sup & (t_np[:, j] > active_eps), sup & ~(t_np[:, j] > active_eps)
        per_joint.append({
            "joint_index": j, "joint_id": str(joint_ids[j]), "active_supervised_count": int(act.sum()),
            "active_vs_inactive_expected_sample_probability_ratio": float(active_ratios[j]),
            "raw_field_mae_all_supervised": float(np.abs(p[sup, j] - t_np[sup, j]).mean()),
            "raw_field_mae_active": float(np.abs(p[act, j] - t_np[act, j]).mean()),
            "raw_field_false_positive_mean_inactive": float(p[ina, j].mean()) if np.any(ina) else 0.0,
            "raw_field_pred_mean_all_supervised": float(p[sup, j].mean()),
            "teacher_field_mean_all_supervised": float(t_np[sup, j].mean()),
        })
    rows.sort(key=lambda r: (not r["supervised"], -r["normalized_row_l1"]))
    per_joint.sort(key=lambda r: -r["raw_field_mae_all_supervised"])
    return report, rows, per_joint
