from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class SceneFirstSignedGeometryConfigV4:
    """Demo-focused scene-first signed geometry producer.

    V4 deliberately keeps the V3 observation/camera fusion width and transformer depth
    frozen. The first demo pivot spends capacity only where Arm C left the strongest
    evidence of a bottleneck: spatial triplane bandwidth and continuous-field decoder
    depth.
    """

    image_token_dim: int = 384
    camera_feature_dim: int = 13
    scene_dim: int = 256
    camera_embed_dim: int = 96
    plane_size: int = 32
    plane_channels: int = 64
    field_hidden_dim: int = 192
    field_residual_blocks: int = 4
    attention_heads: int = 8
    scene_layers: int = 4
    feedforward_multiplier: int = 3
    dropout: float = 0.0

    def validate(self) -> None:
        numeric = (
            self.image_token_dim,
            self.camera_feature_dim,
            self.scene_dim,
            self.camera_embed_dim,
            self.plane_size,
            self.plane_channels,
            self.field_hidden_dim,
            self.field_residual_blocks,
            self.attention_heads,
            self.scene_layers,
            self.feedforward_multiplier,
        )
        if min(numeric) <= 0:
            raise ValueError("scene-first V4 config dimensions must be positive")
        if self.scene_dim % self.attention_heads:
            raise ValueError("scene_dim must be divisible by attention_heads")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in [0,1)")


def _plane_query_coordinates(size: int) -> torch.Tensor:
    axis = torch.linspace(-1.0, 1.0, int(size), dtype=torch.float32)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    xy = torch.stack([xx, yy], dim=-1)
    plane_id = torch.eye(3, dtype=torch.float32)[:, None, None, :].expand(
        3, size, size, 3
    )
    return torch.cat([xy[None].expand(3, -1, -1, -1), plane_id], dim=-1)


class CameraAwareViewTokenBankV4(nn.Module):
    """Exact V3 camera-aware evidence fusion retained for the first V4 arm."""

    def __init__(self, cfg: SceneFirstSignedGeometryConfigV4):
        super().__init__()
        d = cfg.scene_dim
        c = cfg.camera_embed_dim
        self.image_projection = nn.Linear(cfg.image_token_dim, d)
        self.camera_embedding = nn.Sequential(
            nn.Linear(cfg.camera_feature_dim, c),
            nn.GELU(),
            nn.Linear(c, d),
        )
        self.token_xy_embedding = nn.Sequential(
            nn.Linear(2, c),
            nn.GELU(),
            nn.Linear(c, d),
        )
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
        return self.norm(z).reshape(b, v * t, z.shape[-1])


class SharedTriplaneSceneFusionV4(nn.Module):
    """V3 fusion semantics with a real 32x32 learnable query lattice -> 64x64 planes."""

    def __init__(self, cfg: SceneFirstSignedGeometryConfigV4):
        super().__init__()
        d = cfg.scene_dim
        p = cfg.plane_size
        self.cfg = cfg
        self.scene_queries = nn.Parameter(torch.randn(3, p, p, d) / math.sqrt(d))
        self.register_buffer(
            "scene_query_coordinates",
            _plane_query_coordinates(p),
            persistent=False,
        )
        self.query_pos = nn.Sequential(
            nn.Linear(5, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        layer = nn.TransformerDecoderLayer(
            d_model=d,
            nhead=cfg.attention_heads,
            dim_feedforward=d * cfg.feedforward_multiplier,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(
            layer,
            cfg.scene_layers,
            norm=nn.LayerNorm(d),
        )
        self.upsample = nn.ConvTranspose2d(
            d,
            cfg.plane_channels,
            kernel_size=2,
            stride=2,
        )

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        if memory.ndim != 3:
            raise ValueError("memory must be [B,M,D]")
        b = memory.shape[0]
        p = self.cfg.plane_size
        d = self.cfg.scene_dim
        q = self.scene_queries + self.query_pos(
            self.scene_query_coordinates.to(memory)
        )
        q = q.reshape(1, 3 * p * p, d).expand(b, -1, -1)
        scene = self.decoder(q, memory)
        scene = (
            scene.reshape(b, 3, p, p, d)
            .permute(0, 1, 4, 2, 3)
            .contiguous()
        )
        up = self.upsample(scene.reshape(b * 3, d, p, p))
        return up.reshape(
            b,
            3,
            self.cfg.plane_channels,
            up.shape[-2],
            up.shape[-1],
        )


class ResidualFieldBlockV4(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.residual_scale = 1.0 / math.sqrt(2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = F.silu(self.fc1(h))
        h = self.fc2(h)
        return (x + h) * self.residual_scale


class ContinuousSignedSurfaceFieldV4(nn.Module):
    """Continuous signed field with four residual nonlinear refinement blocks."""

    def __init__(self, cfg: SceneFirstSignedGeometryConfigV4):
        super().__init__()
        width = 3 * cfg.plane_channels
        hidden = cfg.field_hidden_dim
        self.input_norm = nn.LayerNorm(width)
        self.input_projection = nn.Linear(width, hidden)
        self.blocks = nn.ModuleList(
            [ResidualFieldBlockV4(hidden) for _ in range(cfg.field_residual_blocks)]
        )
        self.final_norm = nn.LayerNorm(hidden)
        self.sdf_head = nn.Linear(hidden, 1)
        self.uncertainty_head = nn.Linear(hidden, 1)

    @staticmethod
    def _sample_triplanes(
        planes: torch.Tensor,
        points: torch.Tensor,
    ) -> torch.Tensor:
        if planes.ndim != 5 or planes.shape[1] != 3:
            raise ValueError("planes must be [B,3,C,H,W]")
        if (
            points.ndim != 3
            or points.shape[0] != planes.shape[0]
            or points.shape[-1] != 3
        ):
            raise ValueError("points must be [B,N,3]")
        b, _, c, h, w = planes.shape
        x, y, z = points.unbind(dim=-1)
        grids = torch.stack(
            [
                torch.stack([x, y], dim=-1),
                torch.stack([x, z], dim=-1),
                torch.stack([y, z], dim=-1),
            ],
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

    def forward(
        self,
        planes: torch.Tensor,
        points: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        features = self._sample_triplanes(planes, points)
        h = F.silu(self.input_projection(self.input_norm(features)))
        for block in self.blocks:
            h = block(h)
        h = self.final_norm(h)
        return {
            "sdf": self.sdf_head(h).squeeze(-1),
            "log_uncertainty": self.uncertainty_head(h).squeeze(-1).clamp(-8.0, 4.0),
        }


class SceneFirstSignedGeometryV4(nn.Module):
    architecture_id = "RealSaS.IRIS.SceneFirstSignedGeometry.v4.demo64"
    evidence_policy = "ALL_VIEW_ALL_PATCH_CAMERA_AWARE_MEMORY_BEFORE_QUERY_P__V3_SEMANTICS"
    representation_policy = "LEARNED_32_QUERY_LATTICE__64X64_TRIPLANE__64_CHANNELS"
    field_policy = "FOUR_RESIDUAL_SILU_BLOCKS__SIGNED_ZERO_LEVEL_SURFACE"
    decoder_policy = "SIGNED_FIELD_ZERO_LEVEL_SURFACE"

    def __init__(
        self,
        config: SceneFirstSignedGeometryConfigV4 = SceneFirstSignedGeometryConfigV4(),
    ):
        super().__init__()
        config.validate()
        self.config = config
        self.view_bank = CameraAwareViewTokenBankV4(config)
        self.scene_fusion = SharedTriplaneSceneFusionV4(config)
        self.surface_field = ContinuousSignedSurfaceFieldV4(config)

    def encode_scene(
        self,
        image_tokens: torch.Tensor,
        token_xy: torch.Tensor,
        camera_features: torch.Tensor,
    ) -> torch.Tensor:
        return self.scene_fusion(
            self.view_bank(image_tokens, token_xy, camera_features)
        )

    def query(
        self,
        scene_planes: torch.Tensor,
        points_normalized: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
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
