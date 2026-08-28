from __future__ import annotations
from typing import Dict, Tuple
import torch
import torch.nn.functional as F
from .g1_sampling import sample_g1_predictions


def _masked_mean(x, mask):
    m = mask.to(x.dtype)
    base_ndim = m.ndim
    extra = 1
    for d in x.shape[base_ndim:]: extra *= int(d)
    while m.ndim < x.ndim: m = m.unsqueeze(-1)
    return (x * m).sum() / (m.sum().clamp_min(1.0) * max(extra, 1))


def _hetero_point(pred, target, log_sigma, mask):
    e = F.smooth_l1_loss(pred, target, reduction='none').sum(-1)
    ls = log_sigma[..., 0]
    return _masked_mean(torch.exp(-ls) * e + 0.05 * ls, mask)


def _normal_loss(pred, target, mask):
    return _masked_mean(1.0 - (pred * target).sum(-1).clamp(-1, 1), mask)


def g1_geometry_loss(outputs: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor], image_size: int = 256) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Geometry-only G1 objective. Descriptor/mechanics have zero loss authority."""
    s = sample_g1_predictions(outputs, target['XY_A'], image_size)
    P, N, V = target['P_A'], target['N_A'], target['V_A'].bool()
    tP = P[:, None].expand(-1, 8, -1, -1)
    tN = N[:, None].expand(-1, 8, -1, -1)
    parts = {
        'P': _hetero_point(s['point'], tP, s['log_sigma'], V),
        'N': _normal_loss(s['normal'], tN, V),
        'V': F.binary_cross_entropy_with_logits(s['visibility_logit'][..., 0], target['V_A'].float()),
    }
    total = 1.5 * parts['P'] + 0.5 * parts['N'] + 0.25 * parts['V']
    parts['total'] = total
    return total, parts
