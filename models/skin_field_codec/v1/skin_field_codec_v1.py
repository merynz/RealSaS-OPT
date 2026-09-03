from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .candidate_config_v1 import SKIN_FIELD_CODEC_V1, SkinFieldCodecConfigV1


@dataclass
class SkinFieldCodecOutputV1:
    latents: torch.Tensor
    decoded_weights: torch.Tensor
    pair_logits: torch.Tensor


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, layers: int, dropout: float):
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be positive")
        blocks = []
        d = in_dim
        for _ in range(layers - 1):
            blocks.extend([nn.Linear(d, hidden_dim), nn.GELU()])
            if dropout:
                blocks.append(nn.Dropout(dropout))
            d = hidden_dim
        blocks.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SkinFieldCodecV1(nn.Module):
    """Continuous per-joint influence-field codec.

    Encode is teacher-lane only and may see dense W. Decode sees only latent fields
    plus the same admitted surface / qualified-skeleton conditioning used by Arachne.
    The decoder is the single shared W reconstruction head for both R6-A0 and R6-A1.
    """

    def __init__(self, config: SkinFieldCodecConfigV1 = SKIN_FIELD_CODEC_V1):
        super().__init__()
        config.validate()
        self.config = config
        h = config.hidden_dim
        self.surface_embed = _MLP(config.surface_feature_dim, h, h, 2, config.dropout)
        self.joint_embed = _MLP(config.joint_feature_dim, h, h, 2, config.dropout)
        self.weight_embed = _MLP(2, h, h, 2, config.dropout)
        self.encoder = _MLP(3 * h, h, config.latent_dim, config.encoder_layers, config.dropout)
        self.decoder = _MLP(2 * h + config.latent_dim, h, 1, config.decoder_layers, config.dropout)
        self.log_temperature = nn.Parameter(torch.tensor(0.0))

    def _validate_conditioning(self, surface_features: torch.Tensor, joint_features: torch.Tensor, surface_mask: torch.Tensor, joint_mask: torch.Tensor) -> None:
        if surface_features.ndim != 3 or joint_features.ndim != 3:
            raise ValueError("codec features must be batched rank-3 tensors")
        if surface_features.shape[-1] != self.config.surface_feature_dim:
            raise ValueError("surface feature width mismatch")
        if joint_features.shape[-1] != self.config.joint_feature_dim:
            raise ValueError("joint feature width mismatch")
        if surface_mask.shape != surface_features.shape[:2] or joint_mask.shape != joint_features.shape[:2]:
            raise ValueError("codec mask shape mismatch")
        if not surface_mask.any(dim=1).all() or not joint_mask.any(dim=1).all():
            raise ValueError("every codec sample requires surface and joint support")

    def encode_teacher_weights(self, surface_features: torch.Tensor, joint_features: torch.Tensor, dense_weights: torch.Tensor, surface_mask: torch.Tensor, joint_mask: torch.Tensor) -> torch.Tensor:
        self._validate_conditioning(surface_features, joint_features, surface_mask, joint_mask)
        if dense_weights.shape != (surface_features.shape[0], surface_features.shape[1], joint_features.shape[1]):
            raise ValueError("dense teacher weight shape mismatch")
        if not torch.isfinite(dense_weights).all() or (dense_weights < -1e-7).any():
            raise ValueError("invalid dense teacher weights")
        valid_pair = surface_mask[:, :, None].bool() & joint_mask[:, None, :].bool()
        w = dense_weights.clamp_min(0.0) * valid_pair.to(dense_weights.dtype)
        row_sum = w.sum(dim=-1)
        valid_rows = surface_mask.bool()
        if ((row_sum[valid_rows] - 1.0).abs() > 1e-4).any():
            raise ValueError("teacher weights must be simplex on valid surface rows")
        s = self.surface_embed(surface_features)
        j = self.joint_embed(joint_features)
        w_pair = torch.stack([w, torch.sqrt(w.clamp_min(0.0) + 1e-8)], dim=-1)
        we = self.weight_embed(w_pair)
        pair = torch.tanh(s[:, :, None, :] + j[:, None, :, :] + we)
        support = valid_pair.to(pair.dtype)
        denom = support.sum(dim=1).clamp_min(1.0)[:, :, None]
        mean_pair = (pair * support[..., None]).sum(dim=1) / denom
        wdenom = w.sum(dim=1).clamp_min(1e-8)[:, :, None]
        weighted_pair = (pair * w[..., None]).sum(dim=1) / wdenom
        summary = torch.cat([mean_pair, weighted_pair, j], dim=-1)
        latents = self.encoder(summary)
        return latents * joint_mask[..., None].to(latents.dtype)

    def decode_from_latents(self, latents: torch.Tensor, surface_features: torch.Tensor, joint_features: torch.Tensor, surface_mask: torch.Tensor, joint_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_conditioning(surface_features, joint_features, surface_mask, joint_mask)
        if latents.shape != (joint_features.shape[0], joint_features.shape[1], self.config.latent_dim):
            raise ValueError("latent shape mismatch")
        s = self.surface_embed(surface_features)
        j = self.joint_embed(joint_features)
        pair = torch.cat([
            s[:, :, None, :].expand(-1, -1, j.shape[1], -1),
            j[:, None, :, :].expand(-1, s.shape[1], -1, -1),
            latents[:, None, :, :].expand(-1, s.shape[1], -1, -1),
        ], dim=-1)
        logits = self.decoder(pair).squeeze(-1)
        temperature = F.softplus(self.log_temperature) + self.config.temperature_floor
        logits = logits / temperature
        logits = logits.masked_fill(~joint_mask[:, None, :].bool(), -1e4)
        weights = torch.softmax(logits, dim=-1)
        weights = weights * surface_mask[:, :, None].to(weights.dtype)
        return weights, logits

    def forward(self, surface_features: torch.Tensor, joint_features: torch.Tensor, dense_weights: torch.Tensor, surface_mask: torch.Tensor, joint_mask: torch.Tensor) -> SkinFieldCodecOutputV1:
        latents = self.encode_teacher_weights(surface_features, joint_features, dense_weights, surface_mask, joint_mask)
        decoded, logits = self.decode_from_latents(latents, surface_features, joint_features, surface_mask, joint_mask)
        return SkinFieldCodecOutputV1(latents, decoded, logits)


def skin_field_codec_loss_v1(decoded_weights: torch.Tensor, teacher_weights: torch.Tensor, surface_mask: torch.Tensor, joint_mask: torch.Tensor, *, active_threshold: float = 1e-3, active_weight: float = 2.0) -> dict[str, torch.Tensor]:
    """Truth-stationary dense skin reconstruction objective.

    Active influence emphasis is applied as one scalar per simplex row. Applying
    different class multipliers inside a row changes the cross-entropy optimum
    away from the teacher simplex distribution. A row scalar preserves the
    active-emphasis intent while keeping ``decoded_weights == teacher_weights``
    stationary under the row softmax.
    """
    if decoded_weights.shape != teacher_weights.shape:
        raise ValueError("codec loss weight shape mismatch")
    if active_weight < 0.0:
        raise ValueError("active_weight must be non-negative")
    pair_mask = surface_mask[:, :, None].bool() & joint_mask[:, None, :].bool()
    eps = 1e-8
    t = teacher_weights.clamp_min(0.0) * pair_mask.to(teacher_weights.dtype)
    p = decoded_weights.clamp_min(eps)
    ce = -(t * torch.log(p)) * pair_mask.to(p.dtype)

    # Preserve approximately the historical active-emphasis scale without
    # changing relative class targets inside a simplex row. active_mass is
    # teacher-only and therefore constant with respect to decoded logits.
    active_mass = (t * (t >= active_threshold).to(t.dtype)).sum(dim=-1)
    row_scale = 1.0 + float(active_weight) * active_mass
    weighted = ce * row_scale[..., None]
    ce_loss = weighted.sum() / pair_mask.sum().clamp_min(1).to(weighted.dtype)

    l1 = (torch.abs(decoded_weights - teacher_weights) * pair_mask.to(decoded_weights.dtype)).sum() / pair_mask.sum().clamp_min(1)
    total = ce_loss + l1
    return {"total": total, "cross_entropy": ce_loss, "l1": l1}
