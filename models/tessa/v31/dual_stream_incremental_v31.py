"""Exact KV-reuse decoder for the Truth139 dual-stream V31 structuralizer."""
from __future__ import annotations

import torch

from models.tessa.v2.windowed_incremental_generation_v2 import (
    _WindowedLayerCacheV2,
    _append_stable_kv,
    _cross_attention_cached,
    _token_fast_self_attention,
)
from .dual_stream_v31 import TESSADualStreamStructuralizerV31


class DualStreamIncrementalDecoderV31:
    def __init__(self, model, memory, *, sequence_position_offset: int = 0):
        if not isinstance(model, TESSADualStreamStructuralizerV31) or model.training:
            raise ValueError("TESSA_DUAL_V31_INCREMENTAL_REQUIRES_EVAL_MODEL")
        if memory.shape != (1, model.config.total_conditioning_tokens, model.config.d_model):
            raise ValueError("TESSA_DUAL_V31_INCREMENTAL_MEMORY_SHAPE")
        if not isinstance(sequence_position_offset, int) or sequence_position_offset < 0:
            raise ValueError("TESSA_DUAL_V31_INCREMENTAL_POSITION_OFFSET")
        self.model = model
        self.memory = memory.detach().clone()
        self.offset = sequence_position_offset
        self.position = 0
        self.layers = [_WindowedLayerCacheV2() for _ in model.decoder]
        self.parameter_versions = [(p, p._version) for p in model.parameters()]

    @torch.inference_mode()
    def step(self, token: int) -> torch.Tensor:
        model = self.model
        if model.training or any(p._version != version for p, version in self.parameter_versions):
            raise ValueError("TESSA_DUAL_V31_INCREMENTAL_MODEL_CHANGED")
        if isinstance(token, bool) or not isinstance(token, int) or not 0 <= token < model.config.vocab_size:
            raise ValueError("TESSA_DUAL_V31_INCREMENTAL_TOKEN")
        device = self.memory.device
        ids = torch.tensor([[token]], device=device)
        pos = torch.tensor([(self.position + self.offset) % model.config.local_attention_window], device=device)
        x = model.token_embedding(ids) + model.position_embedding(pos)[None]
        for block, cache in zip(model.decoder, self.layers, strict=True):
            y, k, v = _token_fast_self_attention(
                block.self_attn,
                block.norm1(x),
                cache=cache,
                absolute_position=self.position,
            )
            x = x + y
            x = x + _cross_attention_cached(block, block.norm2(x), cache=cache, memory=self.memory)
            x = x + block.mlp(block.norm3(x))
            _append_stable_kv(
                cache,
                k,
                v,
                new_stable_count=self.position + 1,
                window=model.config.local_attention_window,
            )
        self.position += 1
        return model.lm_head(model.final_norm(x))[:, -1]
