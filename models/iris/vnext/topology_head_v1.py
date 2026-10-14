from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tessa.v1.model_v1 import TESSAConfigV1, TESSADecoderBlockV1


@dataclass(frozen=True)
class UnifiedTopologyHeadConfigV1:
    scene_dim: int = 1024
    point_feature_dim: int = 256
    d_model: int = 1024
    n_heads: int = 16
    decoder_layers: int = 12
    coordinate_bins: int = 1024
    local_attention_window: int = 2048
    query_chunk_size: int = 256
    max_faces: int = 32768
    max_vertices: int = 32768
    coordinate_residual_limit: float = 1.0 / 1024.0

    def as_tessa_decoder_config(self) -> TESSAConfigV1:
        cfg = TESSAConfigV1(
            input_dim=self.d_model,
            d_model=self.d_model,
            n_heads=self.n_heads,
            surface_layers=1,  # unused: vNext has no TESSASurfaceEncoderV1
            decoder_layers=self.decoder_layers,
            coordinate_bins=self.coordinate_bins,
            local_attention_window=self.local_attention_window,
            query_chunk_size=self.query_chunk_size,
            max_faces=self.max_faces,
            max_vertices=self.max_vertices,
            surface_latent_count=384,
        )
        cfg.scaling_policy()
        return cfg

    def validate(self) -> None:
        if self.scene_dim < 1024 or self.d_model < 1024:
            raise ValueError("UNIFIED_IRIS_TOPOLOGY_CAPACITY_BELOW_FLOOR")
        if self.d_model % self.n_heads:
            raise ValueError("UNIFIED_IRIS_TOPOLOGY_DIM_NOT_DIVISIBLE_BY_HEADS")
        if self.decoder_layers < 12:
            raise ValueError("UNIFIED_IRIS_TOPOLOGY_DECODER_TOO_SHALLOW")
        if self.coordinate_bins < 1024:
            raise ValueError("UNIFIED_IRIS_TOPOLOGY_COORDINATE_BINS_TOO_SMALL")
        if self.coordinate_residual_limit <= 0:
            raise ValueError("UNIFIED_IRIS_TOPOLOGY_RESIDUAL_LIMIT_INVALID")


@dataclass
class UnifiedTopologyHeadOutputV1:
    logits: torch.Tensor
    coordinate_residual: torch.Tensor
    field_binding_query: torch.Tensor
    decoder_hidden: torch.Tensor
    loss: torch.Tensor | None = None


class UnifiedTopologyHeadV1(nn.Module):
    """TESSA mesh grammar over Unified IRIS scene memory.

    This intentionally preserves TESSA's strong mesh-token grammar and bounded
    causal decoder while removing the old 17-D GSA -> TESSASurfaceEncoder scene
    bottleneck. Memory comes directly from the shared scene and optional sampled
    G/A/S/F point features.
    """

    def __init__(self, config: UnifiedTopologyHeadConfigV1 | None = None):
        super().__init__()
        self.config = config or UnifiedTopologyHeadConfigV1()
        self.config.validate()
        self.decoder_cfg = self.config.as_tessa_decoder_config()

        self.scene_projection = nn.Sequential(
            nn.LayerNorm(self.config.scene_dim),
            nn.Linear(self.config.scene_dim, self.config.d_model),
        )
        self.point_projection = nn.Sequential(
            nn.LayerNorm(self.config.point_feature_dim),
            nn.Linear(self.config.point_feature_dim, self.config.d_model),
        )
        self.memory_type_embedding = nn.Parameter(torch.zeros(2, self.config.d_model))

        self.token_embedding = nn.Embedding(self.decoder_cfg.vocab_size, self.config.d_model)
        self.position_embedding = nn.Embedding(
            self.config.local_attention_window,
            self.config.d_model,
        )
        self.decoder = nn.ModuleList(
            [TESSADecoderBlockV1(self.decoder_cfg) for _ in range(self.config.decoder_layers)]
        )
        self.final_norm = nn.LayerNorm(self.config.d_model)
        self.lm_head = nn.Linear(self.config.d_model, self.decoder_cfg.vocab_size, bias=False)

        # Continuous correction augments, rather than replaces, the 1024-bin
        # coordinate grammar. Field-binding queries let the Compiler bind a
        # proposed vertex back into the same shared field instead of recreating
        # identity by nearest-neighbor search on an unrelated surface summary.
        self.coordinate_residual_head = nn.Linear(self.config.d_model, 3)
        self.field_binding_query_head = nn.Linear(self.config.d_model, self.config.d_model)

    def build_memory(
        self,
        *,
        scene_memory: torch.Tensor,
        point_memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if scene_memory.ndim != 3 or scene_memory.shape[-1] != self.config.scene_dim:
            raise ValueError("UNIFIED_IRIS_TOPOLOGY_SCENE_MEMORY_SHAPE_INVALID")
        if not torch.isfinite(scene_memory).all():
            raise ValueError("UNIFIED_IRIS_TOPOLOGY_SCENE_MEMORY_NONFINITE")
        scene = self.scene_projection(scene_memory) + self.memory_type_embedding[0]
        if point_memory is None:
            return scene
        if point_memory.ndim != 3 or point_memory.shape[0] != scene_memory.shape[0]:
            raise ValueError("UNIFIED_IRIS_TOPOLOGY_POINT_MEMORY_SHAPE_INVALID")
        if point_memory.shape[-1] != self.config.point_feature_dim:
            raise ValueError("UNIFIED_IRIS_TOPOLOGY_POINT_FEATURE_DIM_INVALID")
        if not torch.isfinite(point_memory).all():
            raise ValueError("UNIFIED_IRIS_TOPOLOGY_POINT_MEMORY_NONFINITE")
        points = self.point_projection(point_memory) + self.memory_type_embedding[1]
        return torch.cat([scene, points], dim=1)

    def forward(
        self,
        *,
        scene_memory: torch.Tensor,
        input_ids: torch.Tensor,
        point_memory: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        sequence_position_offset: int = 0,
    ) -> UnifiedTopologyHeadOutputV1:
        if input_ids.ndim != 2:
            raise ValueError("UNIFIED_IRIS_TOPOLOGY_INPUT_IDS_SHAPE_INVALID")
        if input_ids.numel() and (
            int(input_ids.min()) < 0 or int(input_ids.max()) >= self.decoder_cfg.vocab_size
        ):
            raise ValueError("UNIFIED_IRIS_TOPOLOGY_INPUT_TOKEN_OUT_OF_RANGE")
        if sequence_position_offset < 0:
            raise ValueError("UNIFIED_IRIS_TOPOLOGY_POSITION_OFFSET_NEGATIVE")

        memory = self.build_memory(scene_memory=scene_memory, point_memory=point_memory)
        pos = (
            torch.arange(input_ids.shape[1], device=input_ids.device)
            + int(sequence_position_offset)
        ) % self.config.local_attention_window
        x = self.token_embedding(input_ids) + self.position_embedding(pos)[None]
        for block in self.decoder:
            x = block(x, memory)
        hidden = self.final_norm(x)
        logits = self.lm_head(hidden)
        residual = self.config.coordinate_residual_limit * torch.tanh(
            self.coordinate_residual_head(hidden)
        )
        binding = F.normalize(self.field_binding_query_head(hidden), dim=-1, eps=1e-6)

        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("UNIFIED_IRIS_TOPOLOGY_LABEL_SHAPE_INVALID")
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=self.decoder_cfg.PAD,
            )
        return UnifiedTopologyHeadOutputV1(
            logits=logits,
            coordinate_residual=residual,
            field_binding_query=binding,
            decoder_hidden=hidden,
            loss=loss,
        )


__all__ = [
    "UnifiedTopologyHeadConfigV1",
    "UnifiedTopologyHeadOutputV1",
    "UnifiedTopologyHeadV1",
]
