from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class IRISTopologySceneMemoryConfigV1:
    """Bounded adapter from rich IRIS scene planes to topology-decoder memory.

    This is intentionally an information bridge, not a second scene-understanding
    model.  The source is the same IRIS scene representation used by the signed
    field.  Fixed learned queries bound downstream topology-decoder cost while
    preserving direct dependence on rich scene evidence.
    """

    scene_channels: int = 64
    source_width: int = 256
    topology_width: int = 768
    memory_tokens: int = 384
    attention_heads: int = 12
    refinement_layers: int = 2
    mlp_ratio: int = 4
    dropout: float = 0.0
    expected_planes: int = 3

    def validate(self) -> None:
        numeric = (
            self.scene_channels,
            self.source_width,
            self.topology_width,
            self.memory_tokens,
            self.attention_heads,
            self.mlp_ratio,
            self.expected_planes,
        )
        if min(numeric) <= 0:
            raise ValueError("IRIS_TOPOLOGY_MEMORY_DIMENSION_INVALID")
        if self.refinement_layers < 0:
            raise ValueError("IRIS_TOPOLOGY_MEMORY_REFINEMENT_NEGATIVE")
        if self.topology_width % self.attention_heads:
            raise ValueError("IRIS_TOPOLOGY_MEMORY_HEAD_WIDTH_MISMATCH")
        if not (0.0 <= float(self.dropout) < 1.0):
            raise ValueError("IRIS_TOPOLOGY_MEMORY_DROPOUT_INVALID")
        if self.expected_planes != 3:
            raise ValueError("IRIS_TOPOLOGY_MEMORY_REQUIRES_TRIPLANE")


def _triplane_position_features(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if height < 1 or width < 1:
        raise ValueError("IRIS_TOPOLOGY_MEMORY_SPATIAL_SHAPE_INVALID")
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    xy = torch.stack([xx, yy], dim=-1)
    plane_id = torch.eye(3, device=device, dtype=dtype)[:, None, None, :].expand(
        3, height, width, 3
    )
    return torch.cat([xy[None].expand(3, -1, -1, -1), plane_id], dim=-1)


class IRISTopologySceneMemoryV1(nn.Module):
    """Compress IRIS triplanes into fixed topology-decoder memory.

    Input is the existing `SceneFirstSignedGeometryV4.encode_scene()` output:
    `[B,3,C,H,W]`.  No GSA/TESSA 17-D conditioning is accepted here.
    """

    architecture_id = "RealSaS.IRIS.TopologySceneMemory.v1"
    source_semantics = "DIRECT_IRIS_SCENE_PLANES"
    legacy_gsa_17d_used = False

    def __init__(
        self,
        config: IRISTopologySceneMemoryConfigV1 = IRISTopologySceneMemoryConfigV1(),
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.source_projection = nn.Sequential(
            nn.Linear(config.scene_channels + 5, config.source_width),
            nn.LayerNorm(config.source_width),
            nn.GELU(),
        )
        self.memory_queries = nn.Parameter(
            torch.randn(config.memory_tokens, config.topology_width)
            / math.sqrt(config.topology_width)
        )
        self.query_norm = nn.LayerNorm(config.topology_width)
        self.source_norm = nn.LayerNorm(config.source_width)
        self.cross = nn.MultiheadAttention(
            embed_dim=config.topology_width,
            num_heads=config.attention_heads,
            dropout=config.dropout,
            kdim=config.source_width,
            vdim=config.source_width,
            batch_first=True,
        )
        if config.refinement_layers:
            layer = nn.TransformerEncoderLayer(
                d_model=config.topology_width,
                nhead=config.attention_heads,
                dim_feedforward=config.topology_width * config.mlp_ratio,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.refinement = nn.TransformerEncoder(
                layer,
                num_layers=config.refinement_layers,
            )
        else:
            self.refinement = nn.Identity()
        self.out_norm = nn.LayerNorm(config.topology_width)

    def source_tokens(self, scene_planes: torch.Tensor) -> torch.Tensor:
        if scene_planes.ndim != 5:
            raise ValueError("IRIS_TOPOLOGY_SCENE_PLANES_RANK_INVALID")
        b, p, c, h, w = scene_planes.shape
        if p != self.config.expected_planes:
            raise ValueError("IRIS_TOPOLOGY_SCENE_PLANE_COUNT_INVALID")
        if c != self.config.scene_channels:
            raise ValueError("IRIS_TOPOLOGY_SCENE_CHANNEL_COUNT_INVALID")
        if not torch.is_floating_point(scene_planes):
            raise ValueError("IRIS_TOPOLOGY_SCENE_PLANES_DTYPE_INVALID")
        if not bool(torch.isfinite(scene_planes).all()):
            raise ValueError("IRIS_TOPOLOGY_SCENE_PLANES_NONFINITE")

        values = scene_planes.permute(0, 1, 3, 4, 2).contiguous()
        pos = _triplane_position_features(
            h,
            w,
            device=scene_planes.device,
            dtype=scene_planes.dtype,
        )
        pos = pos.unsqueeze(0).expand(b, -1, -1, -1, -1)
        source = torch.cat([values, pos], dim=-1).reshape(b, p * h * w, c + 5)
        return self.source_projection(source)

    def forward(self, scene_planes: torch.Tensor) -> torch.Tensor:
        source = self.source_tokens(scene_planes)
        batch = source.shape[0]
        q = self.memory_queries.unsqueeze(0).expand(batch, -1, -1)
        update, _ = self.cross(
            query=self.query_norm(q),
            key=self.source_norm(source),
            value=self.source_norm(source),
            need_weights=False,
        )
        memory = q + update
        memory = self.refinement(memory)
        return self.out_norm(memory)
