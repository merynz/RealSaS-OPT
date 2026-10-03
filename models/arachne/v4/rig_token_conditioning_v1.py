from __future__ import annotations

"""Additive CanonicalRigToken conditioning for Arachne research.

This adapter consumes only ID-free numeric rig-token features plus an integer
joint->token alignment sidecar. It does not see canonical/source joint names.

The final projection is zero-initialized, so inserting the adapter additively
into an existing Arachne joint stream is identity-at-initialization.
"""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class RigTokenConditioningConfigV1:
    token_feature_dim: int = 12
    model_dim: int = 640
    token_dim: int = 160
    attention_heads: int = 5
    transformer_layers: int = 2
    ffn_ratio: int = 4
    dropout: float = 0.0
    architecture_id: str = "RealSaS.Arachne.CanonicalRigTokenConditioning.v1"


class RigTokenConditioningV1(nn.Module):
    def __init__(self, config: RigTokenConditioningConfigV1 = RigTokenConditioningConfigV1()):
        super().__init__()
        self.config = config
        d = int(config.token_dim)
        if d <= 0 or d % int(config.attention_heads):
            raise ValueError("rig-token dim/head mismatch")
        self.in_proj = nn.Sequential(
            nn.Linear(int(config.token_feature_dim), d),
            nn.GELU(),
            nn.LayerNorm(d),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=int(config.attention_heads),
            dim_feedforward=d * int(config.ffn_ratio),
            dropout=float(config.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=int(config.transformer_layers),
            norm=nn.LayerNorm(d),
        )
        self.local_to_model = nn.Linear(d, int(config.model_dim), bias=False)
        self.global_to_model = nn.Linear(d, int(config.model_dim), bias=False)

        # Additive identity-at-initialization.
        nn.init.zeros_(self.local_to_model.weight)
        nn.init.zeros_(self.global_to_model.weight)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        *,
        token_features: torch.Tensor,
        token_mask: torch.Tensor,
        joint_token_indices: torch.Tensor,
        joint_mask: torch.Tensor,
    ) -> torch.Tensor:
        if token_features.ndim != 3:
            raise ValueError("token_features must be [B,T,F]")
        B, T, F = token_features.shape
        if F != int(self.config.token_feature_dim):
            raise ValueError("rig-token feature width drift")
        if token_mask.shape != (B, T):
            raise ValueError("rig-token mask shape drift")
        if joint_token_indices.ndim != 2 or joint_token_indices.shape[0] != B:
            raise ValueError("joint-token index shape drift")
        if joint_mask.shape != joint_token_indices.shape:
            raise ValueError("joint mask shape drift")

        legal_tokens = token_mask.bool()
        if not bool(legal_tokens.any(dim=1).all()):
            raise ValueError("each sample requires at least one rig token")

        x = self.in_proj(token_features.float())
        x = self.encoder(x, src_key_padding_mask=~legal_tokens)
        x = x * legal_tokens[..., None].to(x.dtype)

        denom = legal_tokens.sum(1, keepdim=True).clamp_min(1).to(x.dtype)
        pooled = x.sum(1) / denom

        idx = joint_token_indices.long()
        valid_joint = joint_mask.bool()
        if bool(((idx < 0) & valid_joint).any()) or bool(((idx >= T) & valid_joint).any()):
            raise ValueError("joint-token index outside token axis")
        safe = idx.clamp(0, max(0, T - 1))
        b = torch.arange(B, device=x.device)[:, None]
        local = x[b, safe]
        if bool((~legal_tokens[b, safe] & valid_joint).any()):
            raise ValueError("joint aligned to masked rig token")

        ctx = self.local_to_model(local) + self.global_to_model(pooled)[:, None, :]
        return ctx * valid_joint[..., None].to(ctx.dtype)


__all__ = [
    "RigTokenConditioningConfigV1",
    "RigTokenConditioningV1",
]
