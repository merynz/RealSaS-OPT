from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class SceneFirstSignedGeometryConfigV3:
    """Canonical scene-first signed geometry core.

    Upstream image tokens are intentionally external to this module. The contract is:
    frozen/qualified per-patch image evidence + exact camera context -> one shared scene
    memory -> arbitrary-P signed field queries. No candidate-local image lookup and no
    categorical view identity are permitted here.
    """

    image_token_dim: int = 384
    camera_feature_dim: int = 13
    scene_dim: int = 256
    camera_embed_dim: int = 96
    plane_size: int = 24
    plane_channels: int = 64
    field_hidden_dim: int = 192
    attention_heads: int = 8
    scene_layers: int = 4
    feedforward_multiplier: int = 3
    dropout: float = 0.0

    def validate(self) -> None:
        if min(
            self.image_token_dim,
            self.camera_feature_dim,
            self.scene_dim,
            self.camera_embed_dim,
            self.plane_size,
            self.plane_channels,
            self.field_hidden_dim,
            self.attention_heads,
            self.scene_layers,
            self.feedforward_multiplier,
        ) <= 0:
            raise ValueError("scene-first config dimensions must be positive")
        if self.scene_dim % self.attention_heads:
            raise ValueError("scene_dim must be divisible by attention_heads")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in [0,1)")


def _plane_query_coordinates(size: int) -> torch.Tensor:
    axis = torch.linspace(-1.0, 1.0, int(size), dtype=torch.float32)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    xy = torch.stack([xx, yy], dim=-1)
    plane_id = torch.eye(3, dtype=torch.float32)[:, None, None, :].expand(3, size, size, 3)
    return torch.cat([xy[None].expand(3, -1, -1, -1), plane_id], dim=-1)


class CameraAwareViewTokenBankV3(nn.Module):
    """All-view/all-patch memory with exact camera conditioning before fusion."""

    def __init__(self, cfg: SceneFirstSignedGeometryConfigV3):
        super().__init__()
        d = cfg.scene_dim
        c = cfg.camera_embed_dim
        self.image_projection = nn.Linear(cfg.image_token_dim, d)
        self.camera_embedding = nn.Sequential(
            nn.Linear(cfg.camera_feature_dim, c), nn.GELU(), nn.Linear(c, d)
        )
        self.token_xy_embedding = nn.Sequential(nn.Linear(2, c), nn.GELU(), nn.Linear(c, d))
        self.norm = nn.LayerNorm(d)

    def forward(
        self,
        image_tokens: torch.Tensor,
        token_xy: torch.Tensor,
        camera_features: torch.Tensor,
    ) -> torch.Tensor:
        if image_tokens.ndim != 4:
            raise ValueError("image_tokens must be [B,V,T,C]")
        b, v, t, _ = image_tokens.shape
        if token_xy.shape != (b, v, t, 2):
            raise ValueError("token_xy must be [B,V,T,2]")
        if camera_features.ndim != 3 or camera_features.shape[:2] != (b, v):
            raise ValueError("camera_features must be [B,V,Ccam]")
        z = (
            self.image_projection(image_tokens)
            + self.camera_embedding(camera_features)[:, :, None, :]
            + self.token_xy_embedding(token_xy)
        )
        z = self.norm(z)
        return z.reshape(b, v * t, z.shape[-1])


class SharedTriplaneSceneFusionV3(nn.Module):
    """Learnable scene queries cross-attend to the complete observation memory."""

    def __init__(self, cfg: SceneFirstSignedGeometryConfigV3):
        super().__init__()
        d = cfg.scene_dim
        p = cfg.plane_size
        self.cfg = cfg
        self.scene_queries = nn.Parameter(torch.randn(3, p, p, d) / math.sqrt(d))
        self.register_buffer("scene_query_coordinates", _plane_query_coordinates(p), persistent=False)
        self.query_pos = nn.Sequential(nn.Linear(5, d), nn.GELU(), nn.Linear(d, d))
        layer = nn.TransformerDecoderLayer(
            d_model=d,
            nhead=cfg.attention_heads,
            dim_feedforward=d * cfg.feedforward_multiplier,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(layer, cfg.scene_layers, norm=nn.LayerNorm(d))
        self.upsample = nn.ConvTranspose2d(d, cfg.plane_channels, kernel_size=2, stride=2)

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        if memory.ndim != 3:
            raise ValueError("memory must be [B,M,D]")
        b = memory.shape[0]
        p = self.cfg.plane_size
        d = self.cfg.scene_dim
        q = self.scene_queries + self.query_pos(self.scene_query_coordinates.to(memory))
        q = q.reshape(1, 3 * p * p, d).expand(b, -1, -1)
        scene = self.decoder(q, memory)
        scene = scene.reshape(b, 3, p, p, d).permute(0, 1, 4, 2, 3).contiguous()
        up = self.upsample(scene.reshape(b * 3, d, p, p))
        return up.reshape(b, 3, self.cfg.plane_channels, up.shape[-2], up.shape[-1])


class ContinuousSignedSurfaceFieldV3(nn.Module):
    """Arbitrary-P signed field over the shared scene representation."""

    def __init__(self, cfg: SceneFirstSignedGeometryConfigV3):
        super().__init__()
        width = 3 * cfg.plane_channels
        h = cfg.field_hidden_dim
        self.body = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
        )
        self.sdf_head = nn.Linear(h, 1)
        self.uncertainty_head = nn.Linear(h, 1)

    @staticmethod
    def _sample_triplanes(planes: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        if planes.ndim != 5 or planes.shape[1] != 3:
            raise ValueError("planes must be [B,3,C,H,W]")
        if points.ndim != 3 or points.shape[0] != planes.shape[0] or points.shape[-1] != 3:
            raise ValueError("points must be [B,N,3]")
        b, _, c, h, w = planes.shape
        x, y, z = points.unbind(dim=-1)
        grids = torch.stack(
            [torch.stack([x, y], dim=-1), torch.stack([x, z], dim=-1), torch.stack([y, z], dim=-1)],
            dim=1,
        )
        sampled = F.grid_sample(
            planes.reshape(b * 3, c, h, w),
            grids.reshape(b * 3, points.shape[1], 1, 2),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return (
            sampled.squeeze(-1)
            .reshape(b, 3, c, points.shape[1])
            .permute(0, 3, 1, 2)
            .reshape(b, points.shape[1], 3 * c)
        )

    def forward(self, planes: torch.Tensor, points: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.body(self._sample_triplanes(planes, points))
        return {
            "sdf": self.sdf_head(h).squeeze(-1),
            "log_uncertainty": self.uncertainty_head(h).squeeze(-1).clamp(-8.0, 4.0),
        }


class SceneFirstSignedGeometryV3(nn.Module):
    architecture_id = "RealSaS.IRIS.SceneFirstSignedGeometry.v3"
    evidence_policy = "ALL_VIEW_ALL_PATCH_CAMERA_AWARE_MEMORY_BEFORE_QUERY_P"
    decoder_policy = "SIGNED_FIELD_ZERO_LEVEL_SURFACE"

    def __init__(self, config: SceneFirstSignedGeometryConfigV3 = SceneFirstSignedGeometryConfigV3()):
        super().__init__()
        config.validate()
        self.config = config
        self.view_bank = CameraAwareViewTokenBankV3(config)
        self.scene_fusion = SharedTriplaneSceneFusionV3(config)
        self.surface_field = ContinuousSignedSurfaceFieldV3(config)

    def encode_scene(
        self,
        image_tokens: torch.Tensor,
        token_xy: torch.Tensor,
        camera_features: torch.Tensor,
    ) -> torch.Tensor:
        return self.scene_fusion(self.view_bank(image_tokens, token_xy, camera_features))

    def query(self, scene_planes: torch.Tensor, points_normalized: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.surface_field(scene_planes, points_normalized)

    def forward(
        self,
        image_tokens: torch.Tensor,
        token_xy: torch.Tensor,
        camera_features: torch.Tensor,
        points_normalized: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        planes = self.encode_scene(image_tokens, token_xy, camera_features)
        out = self.query(planes, points_normalized)
        out["scene_planes"] = planes
        return out
