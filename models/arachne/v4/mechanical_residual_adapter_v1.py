from __future__ import annotations

"""Small residual mechanical conditioner for frozen Arachne skin fields.

The adapter consumes only product-available evidence:
- frozen/base Arachne simplex weights,
- deterministic surface geometry7,
- deterministic surface<->joint pair geometry.

It does not consume teacher identity, source bone names, or compiler PASS labels.
The final projection is residual in log-probability space and the output remains a
nonnegative simplex over the admitted joint mask.

The final layer is zero-initialized so step-0 behavior exactly reproduces the
frozen base field (up to floating-point roundoff).
"""

from dataclasses import dataclass
import torch
from torch import nn


@dataclass(frozen=True)
class MechanicalResidualAdapterConfigV1:
    surface_dim: int = 7
    pair_dim: int = 10
    embed_dim: int = 256
    hidden_dim: int = 1024
    architecture_id: str = "RealSaS.Arachne.MechanicalResidualAdapter.v1"


class MechanicalResidualAdapterV1(nn.Module):
    def __init__(self, config: MechanicalResidualAdapterConfigV1 = MechanicalResidualAdapterConfigV1()):
        super().__init__()
        self.config = config
        e, h = int(config.embed_dim), int(config.hidden_dim)
        self.surface = nn.Sequential(
            nn.Linear(int(config.surface_dim), e),
            nn.GELU(),
            nn.LayerNorm(e),
        )
        self.pair = nn.Sequential(
            nn.Linear(int(config.pair_dim), e),
            nn.GELU(),
            nn.LayerNorm(e),
        )
        self.residual = nn.Sequential(
            nn.Linear(2 * e + 1, h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.GELU(),
            nn.Linear(h, 1),
        )
        # Identity-at-initialization: no mechanical correction until learned.
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        *,
        base_weights: torch.Tensor,
        surface_geometry7: torch.Tensor,
        pair_geometry: torch.Tensor,
        surface_mask: torch.Tensor,
        joint_mask: torch.Tensor,
        row_joint_mask: torch.Tensor | None = None,
        surface_chunk_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if base_weights.ndim != 3:
            raise ValueError("base_weights must be [B,N,J]")
        B, N, J = base_weights.shape
        if surface_geometry7.shape != (B, N, self.config.surface_dim):
            raise ValueError("surface_geometry7 shape drift")
        if pair_geometry.shape != (B, N, J, self.config.pair_dim):
            raise ValueError("pair_geometry shape drift")
        if surface_mask.shape != (B, N) or joint_mask.shape != (B, J):
            raise ValueError("mask shape drift")
        if row_joint_mask is not None and row_joint_mask.shape != (B,N,J):
            raise ValueError("row_joint_mask shape drift")

        legal = surface_mask[:, :, None].bool() & joint_mask[:, None, :].bool()
        if row_joint_mask is not None:
            legal = legal & row_joint_mask.bool()
        w = base_weights.float().clamp_min(0.0)
        w = w * legal.to(w.dtype)
        denom = w.sum(-1, keepdim=True)
        if bool((surface_mask & (denom.squeeze(-1) <= 0)).any()):
            raise ValueError("base skin simplex has empty admitted row")
        w = w / denom.clamp_min(1e-12)

        base_log = torch.log(w.clamp_min(1e-8))
        se_all = self.surface(surface_geometry7.float())
        chunk = N if surface_chunk_size is None else int(surface_chunk_size)
        if chunk <= 0:
            raise ValueError("surface_chunk_size must be positive")
        delta_chunks = []
        for start in range(0, N, chunk):
            stop = min(N, start + chunk)
            se = se_all[:, start:stop, None, :].expand(-1, -1, J, -1)
            pe = self.pair(pair_geometry[:, start:stop].float())
            x = torch.cat([se, pe, base_log[:, start:stop, :, None]], dim=-1)
            delta_chunks.append(self.residual(x).squeeze(-1))
        delta = torch.cat(delta_chunks, dim=1)
        delta = delta.masked_fill(~legal, 0.0)

        logits = base_log + delta
        logits = logits.masked_fill(~legal, -1e4)
        out = torch.softmax(logits, dim=-1)
        out = out * legal.to(out.dtype)
        out = out / out.sum(-1, keepdim=True).clamp_min(1e-12)
        out = out * surface_mask[:, :, None].to(out.dtype)
        return out, delta


__all__ = [
    "MechanicalResidualAdapterConfigV1",
    "MechanicalResidualAdapterV1",
]
