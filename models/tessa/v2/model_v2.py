from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config_v2 import TESSAStructuralizerConfigV2
from .objective_v2 import (
    TESSAObjectiveResultV2,
    TESSAObjectiveWeightsV2,
    topology_sequence_objective_v2,
)


@dataclass
class TESSAStructuralizerOutputV2:
    logits: torch.Tensor
    conditioning_memory: torch.Tensor
    loss: torch.Tensor | None = None
    objective: TESSAObjectiveResultV2 | None = None


class _TypedMemoryStreamV2(nn.Module):
    """Perceiver-style resampler for one typed structural evidence stream."""

    def __init__(
        self,
        *,
        input_dim: int,
        token_count: int,
        type_index: int,
        config: TESSAStructuralizerConfigV2,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or token_count <= 0 or type_index < 0:
            raise ValueError("TESSA_V2_MEMORY_STREAM_CONFIG_INVALID")
        self.config = config
        self.input_dim = int(input_dim)
        self.token_count = int(token_count)
        self.source_projection = nn.Sequential(
            nn.Linear(self.input_dim, config.memory_source_width),
            nn.LayerNorm(config.memory_source_width),
            nn.GELU(),
        )
        self.queries = nn.Parameter(
            torch.randn(self.token_count, config.d_model) / math.sqrt(config.d_model)
        )
        self.type_embedding = nn.Parameter(
            torch.randn(config.d_model) / math.sqrt(config.d_model)
        )
        self.query_norm = nn.LayerNorm(config.d_model)
        self.source_norm = nn.LayerNorm(config.memory_source_width)
        self.cross = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_heads,
            dropout=config.dropout,
            kdim=config.memory_source_width,
            vdim=config.memory_source_width,
            batch_first=True,
        )
        if config.memory_refinement_layers:
            layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.n_heads,
                dim_feedforward=config.feedforward_dim,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.refinement = nn.TransformerEncoder(
                layer,
                num_layers=config.memory_refinement_layers,
            )
        else:
            self.refinement = nn.Identity()
        self.out_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        source: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if source.ndim != 3 or source.shape[-1] != self.input_dim:
            raise ValueError("TESSA_V2_MEMORY_SOURCE_SHAPE_INVALID")
        if not torch.is_floating_point(source) or not bool(torch.isfinite(source).all()):
            raise ValueError("TESSA_V2_MEMORY_SOURCE_NUMERIC_INVALID")
        if source.shape[1] <= 0:
            raise ValueError("TESSA_V2_MEMORY_SOURCE_EMPTY")

        key_padding_mask = None
        if valid_mask is not None:
            if valid_mask.shape != source.shape[:2]:
                raise ValueError("TESSA_V2_MEMORY_MASK_SHAPE_INVALID")
            if not bool(valid_mask.any(dim=1).all()):
                raise ValueError("TESSA_V2_MEMORY_SAMPLE_HAS_NO_VALID_SOURCE")
            key_padding_mask = ~valid_mask.bool()

        projected = self.source_projection(source)
        batch = projected.shape[0]
        q = self.queries.unsqueeze(0).expand(batch, -1, -1)
        update, _ = self.cross(
            query=self.query_norm(q),
            key=self.source_norm(projected),
            value=self.source_norm(projected),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        memory = q + update + self.type_embedding[None, None, :]
        memory = self.refinement(memory)
        return self.out_norm(memory)


class TESSADualStreamMemoryV2(nn.Module):
    """Keep IRIS scene and explicit surface/mechanics evidence separately typed."""

    architecture_id = "RealSaS.TESSA.DualStreamStructuralMemory.v2"

    def __init__(
        self,
        *,
        scene_input_dim: int,
        surface_input_dim: int,
        config: TESSAStructuralizerConfigV2 = TESSAStructuralizerConfigV2(),
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.scene = _TypedMemoryStreamV2(
            input_dim=scene_input_dim,
            token_count=config.scene_memory_tokens,
            type_index=0,
            config=config,
        )
        self.surface = _TypedMemoryStreamV2(
            input_dim=surface_input_dim,
            token_count=config.surface_memory_tokens,
            type_index=1,
            config=config,
        )
        self.fusion_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        *,
        scene_source: torch.Tensor,
        surface_source: torch.Tensor,
        scene_valid_mask: torch.Tensor | None = None,
        surface_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scene_memory = self.scene(scene_source, scene_valid_mask)
        surface_memory = self.surface(surface_source, surface_valid_mask)
        if scene_memory.shape[0] != surface_memory.shape[0]:
            raise ValueError("TESSA_V2_MEMORY_BATCH_MISMATCH")
        memory = torch.cat((scene_memory, surface_memory), dim=1)
        if memory.shape[1] != self.config.total_conditioning_tokens:
            raise AssertionError("TESSA_V2_MEMORY_TOKEN_COUNT_DRIFT")
        return self.fusion_norm(memory)


class WindowedCausalSelfAttentionV2(nn.Module):
    """Exact local causal attention; global-history claims require a separate court."""

    def __init__(self, config: TESSAStructuralizerConfigV2) -> None:
        super().__init__()
        self.config = config
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, config.d_model * 3, bias=False)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.config.d_model:
            raise ValueError("TESSA_V2_SELF_ATTN_INPUT_SHAPE_INVALID")
        b, length, dim = x.shape
        if length <= 0:
            raise ValueError("TESSA_V2_SEQUENCE_EMPTY")
        qkv = self.qkv(x).view(
            b, length, 3, self.config.n_heads, self.head_dim
        )
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        chunks: list[torch.Tensor] = []
        chunk = int(self.config.query_chunk_size)
        window = int(self.config.local_attention_window)
        for start in range(0, length, chunk):
            end = min(length, start + chunk)
            key_start = max(0, end - window)
            q_chunk = q[:, :, start:end]
            k_chunk = k[:, :, key_start:end]
            v_chunk = v[:, :, key_start:end]

            q_pos = torch.arange(start, end, device=x.device)[:, None]
            k_pos = torch.arange(key_start, end, device=x.device)[None, :]
            allow = k_pos <= q_pos
            mask = torch.zeros(
                (end - start, end - key_start),
                dtype=x.dtype,
                device=x.device,
            ).masked_fill(~allow, float("-inf"))
            y = F.scaled_dot_product_attention(
                q_chunk,
                k_chunk,
                v_chunk,
                attn_mask=mask[None, None],
                dropout_p=self.config.dropout if self.training else 0.0,
            )
            chunks.append(y)
        y = (
            torch.cat(chunks, dim=2)
            .transpose(1, 2)
            .contiguous()
            .view(b, length, dim)
        )
        return self.out(y)


class TESSADecoderBlockV2(nn.Module):
    def __init__(self, config: TESSAStructuralizerConfigV2) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.self_attn = WindowedCausalSelfAttentionV2(config)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.cross = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm3 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.feedforward_dim),
            nn.GELU(),
            nn.Linear(config.feedforward_dim, config.d_model),
        )

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x))
        q = self.norm2(x)
        cross, _ = self.cross(q, memory, memory, need_weights=False)
        x = x + cross
        x = x + self.mlp(self.norm3(x))
        return x


class TESSAStructuralizerV2(nn.Module):
    """Reference-capacity learned retopology/structuralization layer.

    This module emits topology-sequence proposal logits only.  It cannot mint
    canonical M and does not consume teacher topology at inference.
    """

    architecture_id = "RealSaS.TESSA.Structuralizer.v2"
    output_authority = "LEARNED_MESH_CANDIDATE_PROPOSAL_ONLY"

    def __init__(
        self,
        *,
        scene_input_dim: int,
        surface_input_dim: int,
        config: TESSAStructuralizerConfigV2 = TESSAStructuralizerConfigV2(),
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.memory = TESSADualStreamMemoryV2(
            scene_input_dim=scene_input_dim,
            surface_input_dim=surface_input_dim,
            config=config,
        )
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(
            config.local_attention_window,
            config.d_model,
        )
        self.decoder = nn.ModuleList(
            [TESSADecoderBlockV2(config) for _ in range(config.decoder_layers)]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def _position_ids(
        self,
        length: int,
        device: torch.device,
        *,
        offset: int | torch.Tensor = 0,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        base = torch.arange(length, device=device)
        if isinstance(offset, torch.Tensor):
            value = offset.to(device=device, dtype=torch.long)
            if value.ndim != 1 or batch_size is None or value.shape != (batch_size,):
                raise ValueError("TESSA_V2_POSITION_OFFSET_BATCH_SHAPE_INVALID")
            if value.numel() and int(value.min()) < 0:
                raise ValueError("TESSA_V2_POSITION_OFFSET_NEGATIVE")
            return (base[None, :] + value[:, None]) % self.config.local_attention_window
        if int(offset) < 0:
            raise ValueError("TESSA_V2_POSITION_OFFSET_NEGATIVE")
        return (base + int(offset)) % self.config.local_attention_window

    def encode_memory(
        self,
        *,
        scene_source: torch.Tensor,
        surface_source: torch.Tensor,
        scene_valid_mask: torch.Tensor | None = None,
        surface_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.memory(
            scene_source=scene_source,
            surface_source=surface_source,
            scene_valid_mask=scene_valid_mask,
            surface_valid_mask=surface_valid_mask,
        )

    def decode_tokens(
        self,
        *,
        memory: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        sequence_position_offset: int | torch.Tensor = 0,
        objective_weights: TESSAObjectiveWeightsV2 = TESSAObjectiveWeightsV2(),
    ) -> TESSAStructuralizerOutputV2:
        if memory.ndim != 3 or memory.shape[-1] != self.config.d_model:
            raise ValueError("TESSA_V2_MEMORY_SHAPE_INVALID")
        if input_ids.ndim != 2 or input_ids.shape[0] != memory.shape[0]:
            raise ValueError("TESSA_V2_INPUT_ID_SHAPE_INVALID")
        if input_ids.numel() and (
            int(input_ids.min()) < 0 or int(input_ids.max()) >= self.config.vocab_size
        ):
            raise ValueError("TESSA_V2_INPUT_TOKEN_OUT_OF_RANGE")

        pos = self._position_ids(
            input_ids.shape[1],
            input_ids.device,
            offset=sequence_position_offset,
            batch_size=input_ids.shape[0],
        )
        position = self.position_embedding(pos)
        if pos.ndim == 1:
            position = position[None]
        x = self.token_embedding(input_ids) + position

        for block in self.decoder:
            if self.training and self.config.gradient_checkpointing:
                x = checkpoint(block, x, memory, use_reentrant=False)
            else:
                x = block(x, memory)

        logits = self.lm_head(self.final_norm(x))
        objective = None
        loss = None
        if labels is not None:
            objective = topology_sequence_objective_v2(
                logits,
                labels,
                config=self.config,
                weights=objective_weights,
            )
            loss = objective.total
        return TESSAStructuralizerOutputV2(
            logits=logits,
            conditioning_memory=memory,
            loss=loss,
            objective=objective,
        )

    def forward(
        self,
        *,
        scene_source: torch.Tensor,
        surface_source: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        scene_valid_mask: torch.Tensor | None = None,
        surface_valid_mask: torch.Tensor | None = None,
        sequence_position_offset: int | torch.Tensor = 0,
        objective_weights: TESSAObjectiveWeightsV2 = TESSAObjectiveWeightsV2(),
    ) -> TESSAStructuralizerOutputV2:
        memory = self.encode_memory(
            scene_source=scene_source,
            surface_source=surface_source,
            scene_valid_mask=scene_valid_mask,
            surface_valid_mask=surface_valid_mask,
        )
        return self.decode_tokens(
            memory=memory,
            input_ids=input_ids,
            labels=labels,
            sequence_position_offset=sequence_position_offset,
            objective_weights=objective_weights,
        )
