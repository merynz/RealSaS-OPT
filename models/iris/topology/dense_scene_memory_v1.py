from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class IRISDenseTopologySceneMemoryConfigV1:
    """Bounded information bridge from an IRIS dense 3-D carrier to TESSA memory.

    The adapter does not reconstruct a second scene.  It preserves direct dependence on the
    existing dense carrier, spatially bounds source-token cost with deterministic average pooling,
    attaches cell-centre XYZ coordinates matching the dense carrier convention, and lets fixed
    learned topology queries cross-attend to that evidence.
    """

    scene_channels: int
    pooled_resolution: int = 12
    source_width: int = 256
    topology_width: int = 768
    memory_tokens: int = 384
    attention_heads: int = 12
    refinement_layers: int = 2
    mlp_ratio: int = 4
    dropout: float = 0.0

    def validate(self) -> None:
        numeric = (
            self.scene_channels,
            self.pooled_resolution,
            self.source_width,
            self.topology_width,
            self.memory_tokens,
            self.attention_heads,
            self.mlp_ratio,
        )
        if min(numeric) <= 0:
            raise ValueError("IRIS_DENSE_TOPOLOGY_MEMORY_DIMENSION_INVALID")
        if self.refinement_layers < 0:
            raise ValueError("IRIS_DENSE_TOPOLOGY_MEMORY_REFINEMENT_NEGATIVE")
        if self.topology_width % self.attention_heads:
            raise ValueError("IRIS_DENSE_TOPOLOGY_MEMORY_HEAD_WIDTH_MISMATCH")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("IRIS_DENSE_TOPOLOGY_MEMORY_DROPOUT_INVALID")


def _cell_center_xyz(resolution: int, *, device, dtype) -> torch.Tensor:
    if int(resolution) <= 0:
        raise ValueError("IRIS_DENSE_TOPOLOGY_MEMORY_RESOLUTION_INVALID")
    # DenseObservationEncoderV31 uses voxel centres (i + .5) * 2/R - 1.  Preserve that exact
    # convention instead of endpoint linspace coordinates, which would silently shift evidence
    # relative to align_corners=False sampling.
    axis = (torch.arange(resolution, device=device, dtype=dtype) + 0.5) * (2.0 / resolution) - 1.0
    z, y, x = torch.meshgrid(axis, axis, axis, indexing="ij")
    return torch.stack((x, y, z), dim=-1)


class IRISDenseTopologySceneMemoryV1(nn.Module):
    architecture_id = "RealSaS.IRIS.DenseTopologySceneMemory.v1"
    source_semantics = "DIRECT_IRIS_DENSE_3D_CARRIER"
    output_authority = "TOPOLOGY_DECODER_MEMORY_ONLY"
    legacy_gsa_17d_used = False

    def __init__(self, config: IRISDenseTopologySceneMemoryConfigV1) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.source_projection = nn.Sequential(
            nn.Linear(config.scene_channels + 3, config.source_width),
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
            self.refinement = nn.TransformerEncoder(layer, num_layers=config.refinement_layers)
        else:
            self.refinement = nn.Identity()
        self.out_norm = nn.LayerNorm(config.topology_width)

    def source_tokens(self, carrier: torch.Tensor) -> torch.Tensor:
        if carrier.ndim != 5:
            raise ValueError("IRIS_DENSE_TOPOLOGY_CARRIER_RANK_INVALID")
        batch, channels, depth, height, width = carrier.shape
        if channels != self.config.scene_channels:
            raise ValueError("IRIS_DENSE_TOPOLOGY_CARRIER_CHANNEL_COUNT_INVALID")
        if min(depth, height, width) <= 0:
            raise ValueError("IRIS_DENSE_TOPOLOGY_CARRIER_SPATIAL_SHAPE_INVALID")
        if not torch.is_floating_point(carrier) or not bool(torch.isfinite(carrier).all()):
            raise ValueError("IRIS_DENSE_TOPOLOGY_CARRIER_INVALID")

        r = self.config.pooled_resolution
        pooled = F.adaptive_avg_pool3d(carrier, output_size=(r, r, r))
        values = pooled.permute(0, 2, 3, 4, 1).contiguous()
        xyz = _cell_center_xyz(r, device=carrier.device, dtype=carrier.dtype)
        xyz = xyz.unsqueeze(0).expand(batch, -1, -1, -1, -1)
        source = torch.cat((values, xyz), dim=-1).reshape(batch, r**3, channels + 3)
        return self.source_projection(source)

    def forward(self, carrier: torch.Tensor) -> torch.Tensor:
        source = self.source_tokens(carrier)
        batch = source.shape[0]
        query = self.memory_queries.unsqueeze(0).expand(batch, -1, -1)
        update, _ = self.cross(
            query=self.query_norm(query),
            key=self.source_norm(source),
            value=self.source_norm(source),
            need_weights=False,
        )
        memory = self.refinement(query + update)
        return self.out_norm(memory)
