from __future__ import annotations
from typing import Dict
import math
import numpy as np
import torch
import torch.nn.functional as F
from .g1_sampling import sample_g1_predictions


def _rank_np(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    order = np.argsort(x, kind='mergesort'); y = x[order]; r = np.empty(len(x), np.float64); i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and y[j] == y[i]: j += 1
        r[order[i:j]] = 0.5 * (i + j - 1); i = j
    return r


def _spearman_np(a, b):
    if len(a) < 2: return math.nan
    ra, rb = _rank_np(a), _rank_np(b)
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12: return math.nan
    return float(np.corrcoef(ra, rb)[0,1])


def _q(x, q): return float(torch.quantile(x, q)) if x.numel() else math.nan


def _risk_coverage_auc(err, uncertainty):
    if len(err) < 2: return math.nan
    order = np.argsort(uncertainty, kind='mergesort'); e = err[order]
    return float(np.mean(np.cumsum(e) / np.arange(1, len(e)+1)))


def g1_geometry_metrics(outputs: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor], image_size: int = 256) -> Dict[str, float]:
    s = sample_g1_predictions(outputs, target['XY_A'], image_size)
    V = target['V_A'].bool(); P = target['P_A']; N = target['N_A']
    tP = P[:,None].expand_as(s['point']); tN = N[:,None].expand_as(s['normal'])
    point_err = torch.linalg.vector_norm(s['point'] - tP, dim=-1)
    normal_deg = torch.rad2deg(torch.acos((s['normal'] * tN).sum(-1).clamp(-1,1)))
    pe, ne = point_err[V], normal_deg[V]

    m = V.float().unsqueeze(-1)
    consensus = (s['point'] * m).sum(1) / m.sum(1).clamp_min(1.0)
    spread = torch.linalg.vector_norm(s['point'] - consensus[:,None], dim=-1)[V]
    ncons = F.normalize((s['normal'] * m).sum(1) / m.sum(1).clamp_min(1.0), dim=-1, eps=1e-6)
    nspread = torch.rad2deg(torch.acos((s['normal'] * ncons[:,None]).sum(-1).clamp(-1,1)))[V]

    pv = torch.sigmoid(s['visibility_logit'][...,0])
    brier = float(((pv - target['V_A'].float())**2).mean())
    sigma = torch.exp(s['log_sigma'][...,0])[V]
    e_np = pe.detach().cpu().numpy(); u_np = sigma.detach().cpu().numpy()
    direct = target.get('direct_obs_A', target['V_A'].bool().any(1)).bool()
    pred_supported = (pv >= 0.5).any(1)

    return {
        'point_mean': float(pe.mean()) if pe.numel() else math.nan,
        'point_median': float(pe.median()) if pe.numel() else math.nan,
        'point_p90': _q(pe,.90), 'point_p95': _q(pe,.95),
        'normal_median_deg': float(ne.median()) if ne.numel() else math.nan,
        'normal_p90_deg': _q(ne,.90), 'normal_p95_deg': _q(ne,.95),
        'crossview_point_spread_mean': float(spread.mean()) if spread.numel() else math.nan,
        'crossview_point_spread_p95': _q(spread,.95),
        'crossview_normal_spread_median_deg': float(nspread.median()) if nspread.numel() else math.nan,
        'crossview_normal_spread_p95_deg': _q(nspread,.95),
        'visibility_brier': brier,
        'uncertainty_error_spearman': _spearman_np(u_np, e_np),
        'uncertainty_risk_coverage_auc': _risk_coverage_auc(e_np, u_np),
        'direct_surface_support_recall_at_0_5': float((pred_supported & direct).sum() / direct.sum().clamp_min(1)),
        'visible_sample_count': int(V.sum().item()),
        'direct_carrier_count': int(direct.sum().item()),
    }
