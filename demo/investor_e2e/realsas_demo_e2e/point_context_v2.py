from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_points(points: torch.Tensor):
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError('points must be [B,N,3]')
    pmin = points.amin(dim=1, keepdim=True)
    pmax = points.amax(dim=1, keepdim=True)
    center = 0.5 * (pmin + pmax)
    scale = (pmax - pmin).amax(dim=-1, keepdim=True).clamp_min(1e-6)
    return (points - center) / scale, center, scale


def sinusoidal_position(index: int, dim: int, *, device, dtype) -> torch.Tensor:
    if dim <= 0:
        raise ValueError('dim must be positive')
    half = dim // 2
    if half == 0:
        return torch.zeros(dim, device=device, dtype=dtype)
    freq = torch.exp(
        torch.arange(half, device=device, dtype=dtype)
        * (-math.log(10000.0) / max(1, half - 1))
    )
    x = torch.tensor(float(index), device=device, dtype=dtype) * freq
    emb = torch.cat([torch.sin(x), torch.cos(x)], dim=0)
    if emb.numel() < dim:
        emb = F.pad(emb, (0, dim - emb.numel()))
    return emb[:dim]


class LocalPointContextEncoder(nn.Module):
    """Dynamic-N local geometry encoder with global set reasoning.

    No specimen identity, semantic part label, or output-cardinality prior enters this
    operator. kNN is recomputed from the current point set and therefore remains a
    deterministic geometric construction rather than learned authority.
    """

    def __init__(self, dim: int = 192, hidden_dim: int = 384, layers: int = 3, heads: int = 8, local_k: int = 16):
        super().__init__()
        self.dim = int(dim)
        self.local_k = int(local_k)
        self.base = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.edge = nn.Sequential(nn.Linear(4, dim), nn.GELU(), nn.Linear(dim, dim))
        self.mix = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.LayerNorm(dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=hidden_dim,
            dropout=0.0, batch_first=True, norm_first=True, activation='gelu'
        )
        self.global_encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, points: torch.Tensor):
        pn, center, scale = normalize_points(points)
        b, n, _ = pn.shape
        base = self.base(pn)
        if n == 1:
            local = torch.zeros_like(base)
        else:
            k = min(max(1, self.local_k), n - 1)
            distance = torch.cdist(pn, pn)
            eye = torch.eye(n, device=pn.device, dtype=torch.bool).unsqueeze(0)
            distance = distance.masked_fill(eye, float('inf'))
            idx = torch.topk(distance, k=k, dim=-1, largest=False).indices
            batch = torch.arange(b, device=pn.device)[:, None, None].expand(b, n, k)
            neigh = pn[batch, idx]
            rel = neigh - pn[:, :, None, :]
            radius = torch.linalg.norm(rel, dim=-1, keepdim=True)
            edge = self.edge(torch.cat([rel, radius], dim=-1))
            local = edge.max(dim=2).values
        z = self.mix(torch.cat([base, local], dim=-1))
        z = self.out_norm(self.global_encoder(z))
        return z, pn, center, scale
