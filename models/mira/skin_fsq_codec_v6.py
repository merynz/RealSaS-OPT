from __future__ import annotations

"""MIRA V6: carrier-native discrete skin-field codec.

This module borrows the representation principle from SkinTokens/TokenRig -- skin
weights are compressed into discrete latent tokens -- without importing their mesh
encoder or skeleton authority. RealSaS conditions on its own mesh/carrier features.

The codec is deliberately not canonical authority. It produces learned proposal
latents that decode to dense weights; Compiler qualification remains downstream.
"""

from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Sequence
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


MIRA_V6_CODEC_ID = "RealSaS.MIRA.CarrierNativeFSQSkinCodec.v6_0"


@dataclass(frozen=True)
class MIRASkinFSQCodecConfig:
    vertex_feature_dim: int
    joint_feature_dim: int
    hidden_dim: int = 192
    tokens_per_joint: int = 4
    levels: tuple[int, ...] = (5, 5, 5, 5)
    eps: float = 1e-8

    @property
    def latent_dim(self) -> int:
        return len(self.levels)

    @property
    def vocab_size(self) -> int:
        return int(reduce(mul, self.levels, 1))


class FiniteScalarQuantizer(nn.Module):
    """Small dependency-free FSQ-style quantizer with mixed-radix token ids."""

    def __init__(self, levels: Sequence[int]):
        super().__init__()
        levels = tuple(int(v) for v in levels)
        if not levels or any(v < 2 for v in levels):
            raise ValueError("MIRA_V6_FSQ_LEVELS_INVALID")
        self.levels = levels
        self.latent_dim = len(levels)
        self.vocab_size = int(reduce(mul, levels, 1))
        radix = []
        running = 1
        for level in levels:
            radix.append(running)
            running *= level
        self.register_buffer("_radix", torch.tensor(radix, dtype=torch.long), persistent=False)
        self.register_buffer("_levels", torch.tensor(levels, dtype=torch.long), persistent=False)

    def _quantize_indices(self, z: Tensor) -> Tensor:
        if z.shape[-1] != self.latent_dim:
            raise ValueError("MIRA_V6_FSQ_LATENT_DIM_DRIFT")
        bounded = torch.tanh(z)
        levels = self._levels.to(device=z.device)
        max_index = (levels - 1).to(dtype=z.dtype)
        indices = torch.round((bounded + 1.0) * 0.5 * max_index).to(torch.long)
        return torch.minimum(torch.maximum(indices, torch.zeros_like(indices)), levels - 1)

    def indices_to_codes(self, indices: Tensor) -> Tensor:
        if indices.shape[-1] != self.latent_dim:
            raise ValueError("MIRA_V6_FSQ_INDEX_DIM_DRIFT")
        levels = self._levels.to(device=indices.device)
        denom = (levels - 1).clamp_min(1).to(torch.float32)
        return 2.0 * indices.to(torch.float32) / denom - 1.0

    def indices_to_token_ids(self, indices: Tensor) -> Tensor:
        radix = self._radix.to(device=indices.device)
        return torch.sum(indices * radix, dim=-1)

    def token_ids_to_indices(self, token_ids: Tensor) -> Tensor:
        if torch.any(token_ids < 0) or torch.any(token_ids >= self.vocab_size):
            raise ValueError("MIRA_V6_FSQ_TOKEN_OUT_OF_RANGE")
        remaining = token_ids.to(torch.long)
        rows = []
        for level in self.levels:
            rows.append(torch.remainder(remaining, level))
            remaining = torch.div(remaining, level, rounding_mode="floor")
        return torch.stack(rows, dim=-1)

    def token_ids_to_codes(self, token_ids: Tensor) -> Tensor:
        return self.indices_to_codes(self.token_ids_to_indices(token_ids))

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        indices = self._quantize_indices(z)
        hard = self.indices_to_codes(indices).to(dtype=z.dtype)
        # Straight-through estimator: decoder sees quantized values, encoder gets identity gradient.
        quantized = z + (hard - z).detach()
        token_ids = self.indices_to_token_ids(indices)
        return quantized, token_ids, hard


class MIRASkinFSQCodecV6(nn.Module):
    """Encode dense carrier skin into per-joint tokens and decode them back.

    Shapes:
      vertex_features: [B, N, Fv]
      joint_features:  [B, J, Fj]
      weights:         [B, N, J]
      token_ids:       [B, J, T]

    The decoder always returns a simplex over J by softmax. Compiler still checks
    exact product legality; this normalization is a model parameterization, not a
    qualification shortcut.
    """

    architecture_id = MIRA_V6_CODEC_ID

    def __init__(self, config: MIRASkinFSQCodecConfig):
        super().__init__()
        self.config = config
        h = config.hidden_dim
        d = config.latent_dim
        t = config.tokens_per_joint
        self.quantizer = FiniteScalarQuantizer(config.levels)

        self.vertex_encode = nn.Sequential(
            nn.Linear(config.vertex_feature_dim + 1, h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.GELU(),
        )
        self.joint_encode = nn.Sequential(
            nn.Linear(config.joint_feature_dim, h),
            nn.GELU(),
        )
        self.latent_head = nn.Linear(3 * h, t * d)

        self.vertex_decode = nn.Sequential(
            nn.Linear(config.vertex_feature_dim, h),
            nn.GELU(),
            nn.Linear(h, h),
        )
        self.joint_decode = nn.Sequential(
            nn.Linear(config.joint_feature_dim, h),
            nn.GELU(),
            nn.Linear(h, h),
        )
        self.code_decode = nn.Sequential(
            nn.Linear(t * d, h),
            nn.GELU(),
            nn.Linear(h, h),
        )
        self.bias_head = nn.Linear(h, 1)

    def _check_inputs(self, vertex_features: Tensor, joint_features: Tensor) -> tuple[int, int, int]:
        if vertex_features.ndim != 3 or joint_features.ndim != 3:
            raise ValueError("MIRA_V6_CODEC_INPUT_RANK_INVALID")
        b, n, fv = vertex_features.shape
        bj, j, fj = joint_features.shape
        if b != bj:
            raise ValueError("MIRA_V6_CODEC_BATCH_DRIFT")
        if fv != self.config.vertex_feature_dim or fj != self.config.joint_feature_dim:
            raise ValueError("MIRA_V6_CODEC_FEATURE_DIM_DRIFT")
        if n < 1 or j < 1:
            raise ValueError("MIRA_V6_CODEC_EMPTY_DOMAIN")
        return b, n, j

    def encode_tokens(self, weights: Tensor, vertex_features: Tensor, joint_features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        b, n, j = self._check_inputs(vertex_features, joint_features)
        if weights.shape != (b, n, j):
            raise ValueError("MIRA_V6_CODEC_WEIGHT_SHAPE_DRIFT")
        if not torch.isfinite(weights).all():
            raise ValueError("MIRA_V6_CODEC_WEIGHT_NONFINITE")
        if torch.any(weights < -self.config.eps):
            raise ValueError("MIRA_V6_CODEC_WEIGHT_NEGATIVE")

        # Per-joint field encoding. Pool both mean and max so a sparse influence
        # does not disappear in a large carrier.
        vf = vertex_features[:, None, :, :].expand(b, j, n, -1)
        wf = weights.transpose(1, 2).unsqueeze(-1)
        encoded = self.vertex_encode(torch.cat([vf, wf], dim=-1))
        mean_pool = encoded.mean(dim=2)
        max_pool = encoded.amax(dim=2)
        joint_state = self.joint_encode(joint_features)
        latent = self.latent_head(torch.cat([mean_pool, max_pool, joint_state], dim=-1))
        latent = latent.view(b, j, self.config.tokens_per_joint, self.config.latent_dim)
        quantized, token_ids, hard_codes = self.quantizer(latent)
        return token_ids, quantized, hard_codes

    def decode_tokens(self, token_ids: Tensor, vertex_features: Tensor, joint_features: Tensor) -> Tensor:
        b, n, j = self._check_inputs(vertex_features, joint_features)
        expected = (b, j, self.config.tokens_per_joint)
        if tuple(token_ids.shape) != expected:
            raise ValueError("MIRA_V6_CODEC_TOKEN_SHAPE_DRIFT")
        codes = self.quantizer.token_ids_to_codes(token_ids).to(dtype=vertex_features.dtype)
        codes = codes.reshape(b, j, self.config.tokens_per_joint * self.config.latent_dim)

        v = self.vertex_decode(vertex_features)  # [B,N,H]
        q = self.joint_decode(joint_features) + self.code_decode(codes)  # [B,J,H]
        logits = torch.einsum("bnh,bjh->bnj", v, q) / math.sqrt(float(self.config.hidden_dim))
        logits = logits + self.bias_head(q).transpose(1, 2)
        return torch.softmax(logits, dim=-1)

    def forward(self, weights: Tensor, vertex_features: Tensor, joint_features: Tensor) -> dict[str, Tensor]:
        token_ids, quantized, hard_codes = self.encode_tokens(weights, vertex_features, joint_features)
        reconstructed = self.decode_tokens(token_ids, vertex_features, joint_features)
        target = weights / weights.sum(dim=-1, keepdim=True).clamp_min(self.config.eps)
        reconstruction_loss = F.smooth_l1_loss(reconstructed, target)
        # FSQ has no learned embedding table; this term merely keeps pre-quantized
        # latents close to their finite scalar cells.
        commitment_loss = F.mse_loss(quantized, hard_codes.detach())
        return {
            "token_ids": token_ids,
            "reconstructed_weights": reconstructed,
            "reconstruction_loss": reconstruction_loss,
            "commitment_loss": commitment_loss,
        }


__all__ = [
    "FiniteScalarQuantizer",
    "MIRASkinFSQCodecConfig",
    "MIRASkinFSQCodecV6",
    "MIRA_V6_CODEC_ID",
]
