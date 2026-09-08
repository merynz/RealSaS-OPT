from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from models.skin_field_codec.v1.skin_field_codec_v1 import _MLP, skin_field_codec_loss_v1
from .config_v2 import SKIN_FIELD_CODEC_V2, SkinFieldCodecConfigV2


@dataclass
class SkinFieldCodecOutputV2:
    latents: torch.Tensor
    decoded_weights: torch.Tensor
    pair_logits: torch.Tensor


class SkinFieldCodecV2(nn.Module):
    """A0 per-joint influence-field codec with explicit point-to-joint geometry.

    The teacher encoder is intentionally unchanged from V1 and never receives
    pair_geometry. Only decode receives the sealed 10D pair geometry, projected
    once to 64D, then concatenated with surface/joint embeddings and the 64D
    per-joint latent field.
    """

    def __init__(self, config: SkinFieldCodecConfigV2 = SKIN_FIELD_CODEC_V2):
        super().__init__()
        config.validate()
        self.config = config
        h = config.hidden_dim
        self.surface_embed = _MLP(config.surface_feature_dim, h, h, 2, config.dropout)
        self.joint_embed = _MLP(config.joint_feature_dim, h, h, 2, config.dropout)
        self.weight_embed = _MLP(2, h, h, 2, config.dropout)
        self.encoder = _MLP(3 * h, h, config.latent_dim, config.encoder_layers, config.dropout)
        self.pair_geometry_embed = nn.Sequential(
            nn.Linear(config.pair_geometry_dim, config.pair_geometry_embed_dim),
            nn.GELU(),
        )
        decoder_input_dim = 2 * h + config.latent_dim + config.pair_geometry_embed_dim
        if decoder_input_dim != 512:
            raise RuntimeError("SKIN_FIELD_CODEC_V2_DECODER_WIDTH_DRIFT")
        self.decoder = _MLP(decoder_input_dim, h, 1, config.decoder_layers, config.dropout)
        self.log_temperature = nn.Parameter(torch.tensor(0.0))

    def _validate_conditioning(
        self,
        surface_features: torch.Tensor,
        joint_features: torch.Tensor,
        surface_mask: torch.Tensor,
        joint_mask: torch.Tensor,
    ) -> None:
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

    def _validate_pair_geometry(
        self,
        pair_geometry: torch.Tensor,
        surface_features: torch.Tensor,
        joint_features: torch.Tensor,
    ) -> None:
        expected = (
            surface_features.shape[0],
            surface_features.shape[1],
            joint_features.shape[1],
            self.config.pair_geometry_dim,
        )
        if pair_geometry.shape != expected:
            raise ValueError(f"pair_geometry shape mismatch: expected {expected}, got {tuple(pair_geometry.shape)}")
        if not torch.isfinite(pair_geometry).all():
            raise ValueError("pair_geometry must be finite")

    def encode_teacher_weights(
        self,
        surface_features: torch.Tensor,
        joint_features: torch.Tensor,
        dense_weights: torch.Tensor,
        surface_mask: torch.Tensor,
        joint_mask: torch.Tensor,
    ) -> torch.Tensor:
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

    def decode_from_latents(
        self,
        latents: torch.Tensor,
        surface_features: torch.Tensor,
        joint_features: torch.Tensor,
        pair_geometry: torch.Tensor,
        surface_mask: torch.Tensor,
        joint_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_conditioning(surface_features, joint_features, surface_mask, joint_mask)
        self._validate_pair_geometry(pair_geometry, surface_features, joint_features)
        if latents.shape != (joint_features.shape[0], joint_features.shape[1], self.config.latent_dim):
            raise ValueError("latent shape mismatch")
        s = self.surface_embed(surface_features)
        j = self.joint_embed(joint_features)
        pg = self.pair_geometry_embed(pair_geometry)
        pair = torch.cat([
            s[:, :, None, :].expand(-1, -1, j.shape[1], -1),
            j[:, None, :, :].expand(-1, s.shape[1], -1, -1),
            latents[:, None, :, :].expand(-1, s.shape[1], -1, -1),
            pg,
        ], dim=-1)
        if pair.shape[-1] != 512:
            raise RuntimeError("SKIN_FIELD_CODEC_V2_DECODER_CONCAT_DRIFT")
        logits = self.decoder(pair).squeeze(-1)
        temperature = F.softplus(self.log_temperature) + self.config.temperature_floor
        logits = logits / temperature
        logits = logits.masked_fill(~joint_mask[:, None, :].bool(), -1e4)
        weights = torch.softmax(logits, dim=-1)
        weights = weights * surface_mask[:, :, None].to(weights.dtype)
        return weights, logits

    def forward(
        self,
        surface_features: torch.Tensor,
        joint_features: torch.Tensor,
        pair_geometry: torch.Tensor,
        dense_weights: torch.Tensor,
        surface_mask: torch.Tensor,
        joint_mask: torch.Tensor,
    ) -> SkinFieldCodecOutputV2:
        latents = self.encode_teacher_weights(surface_features, joint_features, dense_weights, surface_mask, joint_mask)
        decoded, logits = self.decode_from_latents(latents, surface_features, joint_features, pair_geometry, surface_mask, joint_mask)
        return SkinFieldCodecOutputV2(latents, decoded, logits)


def skin_field_codec_loss_v2(
    decoded_weights: torch.Tensor,
    teacher_weights: torch.Tensor,
    surface_mask: torch.Tensor,
    joint_mask: torch.Tensor,
    *,
    active_threshold: float = 1e-3,
    active_weight: float = 2.0,
) -> dict[str, torch.Tensor]:
    """Exact V1 truth-stationary objective, intentionally unchanged for V2."""
    return skin_field_codec_loss_v1(
        decoded_weights,
        teacher_weights,
        surface_mask,
        joint_mask,
        active_threshold=active_threshold,
        active_weight=active_weight,
    )
