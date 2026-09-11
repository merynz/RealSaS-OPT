from __future__ import annotations

"""RealSaS Arachne A1 V5 minimal K4-Z -> direct simplex candidate.

Scientific contract:
- preserve the exact A1 V4 observable-input backbone and K=4, 512D joint field tokens;
- remove the frozen A0 continuous-field decoder from the product prediction path;
- decode each surface row with one shared, permutation-safe point×joint scorer;
- impose joint competition directly with a masked row softmax;
- keep Compiler qualification authoritative; this module proposes dense weights only.

The decoder parameterization is state-dict compatible with the sealed
Z_DIRECT_SIMPLEX diagnostic head from the 16K causal rerun.
"""

from dataclasses import dataclass
import torch
from torch import nn

from models.arachne.v4.arachne_candidate_v4 import ArachneA1ConfigV4, ArachneA1V4


@dataclass(frozen=True)
class ArachneA1ConfigV5:
    backbone_architecture_id: str = "RealSaS.Arachne.A1.RichQualifiedBidirectional.v4"
    architecture_id: str = "RealSaS.Arachne.A1.MinimalK4DirectSimplex.v5"
    field_tokens: int = 4
    latent_channels: int = 512
    query_geometry_dim: int = 7
    pair_geometry_dim: int = 10
    token_hidden_dim: int = 192
    relation_dim: int = 128
    score_hidden_dim: int = 256

    def validate(self) -> None:
        if self.field_tokens != 4 or self.latent_channels != 512:
            raise ValueError("V5 must preserve the sealed K4x512 interface")
        if self.query_geometry_dim != 7 or self.pair_geometry_dim != 10:
            raise ValueError("V5 geometry contract drift")
        if min(self.token_hidden_dim, self.relation_dim, self.score_hidden_dim) <= 0:
            raise ValueError("positive V5 hidden dimensions required")


class DirectSimplexDecoderV5(nn.Module):
    """K4 joint tokens + surface/query geometry -> masked row-simplex weights.

    K-token aggregation is permutation-safe (mean + max). One shared scorer is
    applied to every point×joint pair, so no joint-ID vocabulary is introduced.
    """

    def __init__(self, config: ArachneA1ConfigV5 = ArachneA1ConfigV5()):
        super().__init__()
        config.validate()
        self.config = config
        d = config.latent_channels
        r = config.relation_dim
        self.token_enc = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, config.token_hidden_dim), nn.GELU(),
            nn.Linear(config.token_hidden_dim, r),
        )
        self.query_enc = nn.Sequential(
            nn.Linear(config.query_geometry_dim, r), nn.GELU(), nn.Linear(r, r),
        )
        self.pair_enc = nn.Sequential(
            nn.Linear(config.pair_geometry_dim, r), nn.GELU(), nn.Linear(r, r),
        )
        self.score = nn.Sequential(
            nn.LayerNorm(r * 5), nn.Linear(r * 5, config.score_hidden_dim), nn.GELU(),
            nn.Linear(config.score_hidden_dim, 1),
        )

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        geometry7: torch.Tensor,
        pair_geometry: torch.Tensor,
        pair_mask: torch.Tensor,
        joint_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if geometry7.ndim != 3 or geometry7.shape[-1] != self.config.query_geometry_dim:
            raise ValueError("geometry7 must be [B,N,7]")
        if joint_tokens.ndim != 4:
            raise ValueError("joint_tokens must be [B,J,K,512]")
        B, N, _ = geometry7.shape
        Bt, J, K, D = joint_tokens.shape
        if Bt != B or K != self.config.field_tokens or D != self.config.latent_channels:
            raise ValueError("joint token contract drift")
        if pair_geometry.shape != (B, N, J, self.config.pair_geometry_dim):
            raise ValueError("pair geometry contract drift")
        if pair_mask.shape != (B, N, J):
            raise ValueError("pair mask contract drift")
        if not bool(pair_mask.any()):
            raise ValueError("no legal V5 point-joint pairs")

        te = self.token_enc(joint_tokens)       # [B,J,K,R]
        jm = te.mean(2)                         # [B,J,R]
        jx = te.amax(2)                         # [B,J,R]
        q = self.query_enc(geometry7)           # [B,N,R]
        p = self.pair_enc(pair_geometry)        # [B,N,J,R]

        qe = q[:, :, None, :].expand(-1, -1, J, -1)
        jme = jm[:, None, :, :].expand(-1, N, -1, -1)
        jxe = jx[:, None, :, :].expand(-1, N, -1, -1)
        interaction = qe * jme
        feat = torch.cat([qe, jme, jxe, p, interaction], dim=-1)
        logits = self.score(feat).squeeze(-1)
        logits = logits.masked_fill(~pair_mask.bool(), -1e4)
        weights = torch.softmax(logits, dim=-1) * pair_mask.to(logits.dtype)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        return logits, weights


@dataclass
class ArachneA1OutputV5:
    weights: torch.Tensor
    logits: torch.Tensor
    field_tokens: torch.Tensor
    surface_memory: torch.Tensor
    joint_memory: torch.Tensor


class ArachneA1V5(nn.Module):
    """Minimal V5 product candidate: exact V4 backbone + direct K4 simplex decoder."""

    def __init__(
        self,
        backbone: ArachneA1V4 | None = None,
        decoder: DirectSimplexDecoderV5 | None = None,
        config: ArachneA1ConfigV5 = ArachneA1ConfigV5(),
    ):
        super().__init__()
        config.validate()
        self.config = config
        self.backbone = backbone if backbone is not None else ArachneA1V4(ArachneA1ConfigV4())
        self.decoder = decoder if decoder is not None else DirectSimplexDecoderV5(config)
        if self.backbone.config.field_tokens != config.field_tokens:
            raise ValueError("V4/V5 field-token contract drift")
        if self.backbone.config.latent_channels != config.latent_channels:
            raise ValueError("V4/V5 latent-channel contract drift")

    @staticmethod
    def geometry7_from_surface(
        surface_positions_normalized: torch.Tensor,
        surface_normals: torch.Tensor,
        surface_normal_valid: torch.Tensor,
    ) -> torch.Tensor:
        # Exact V4 conditioning contract: codec/query geometry stores 2*normalized xyz.
        return torch.cat(
            [
                2.0 * surface_positions_normalized,
                surface_normals,
                surface_normal_valid[..., None].to(surface_positions_normalized.dtype),
            ],
            dim=-1,
        )

    def forward(
        self,
        *,
        surface_positions_normalized,
        surface_normals,
        surface_normal_valid,
        surface_support_views,
        surface_raster_xy,
        surface_raster_valid,
        surface_observed,
        surface_completed,
        surface_mask,
        edge_index,
        edge_features,
        edge_mask,
        joint_positions_normalized,
        joint_mask,
        parent_indices,
        root_mask,
        deform_root_mask,
        joint_depth_normalized,
        support_anchor_matrix,
        pair_geometry,
        pair_mask,
        view_yaw_fourier,
    ) -> ArachneA1OutputV5:
        legal_pair = (
            pair_mask.bool()
            & surface_mask[:, :, None].bool()
            & joint_mask[:, None, :].bool()
        )
        raw = self.backbone(
            surface_positions_normalized=surface_positions_normalized,
            surface_normals=surface_normals,
            surface_normal_valid=surface_normal_valid,
            surface_support_views=surface_support_views,
            surface_raster_xy=surface_raster_xy,
            surface_raster_valid=surface_raster_valid,
            surface_observed=surface_observed,
            surface_completed=surface_completed,
            surface_mask=surface_mask,
            edge_index=edge_index,
            edge_features=edge_features,
            edge_mask=edge_mask,
            joint_positions_normalized=joint_positions_normalized,
            joint_mask=joint_mask,
            parent_indices=parent_indices,
            root_mask=root_mask,
            deform_root_mask=deform_root_mask,
            joint_depth_normalized=joint_depth_normalized,
            support_anchor_matrix=support_anchor_matrix,
            pair_geometry=pair_geometry,
            pair_mask=pair_mask,
            view_yaw_fourier=view_yaw_fourier,
        )
        geometry7 = self.geometry7_from_surface(
            surface_positions_normalized, surface_normals, surface_normal_valid
        )
        logits, weights = self.decoder(geometry7, pair_geometry, legal_pair, raw.field_tokens)
        weights = weights * surface_mask[..., None].to(weights.dtype)
        return ArachneA1OutputV5(
            weights=weights,
            logits=logits,
            field_tokens=raw.field_tokens,
            surface_memory=raw.surface_memory,
            joint_memory=raw.joint_memory,
        )

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


__all__ = [
    "ArachneA1ConfigV5",
    "DirectSimplexDecoderV5",
    "ArachneA1OutputV5",
    "ArachneA1V5",
]
