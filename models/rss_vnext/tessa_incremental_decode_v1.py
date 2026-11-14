from __future__ import annotations

"""Exact V3.1 per-token evaluation outside the immutable relocated TESSA source.

This first-party adapter uses the live TESSA weights and preserves the archived
expert relocation manifest; no model source or state_dict is rewritten.
"""
from dataclasses import dataclass
import torch
from torch.nn import functional as F
from models.tessa.v31.model_v31 import TESSAStructuralizerV31

@dataclass(frozen=True)
class TESSAV31IncrementalCache:
    memory: torch.Tensor
    position: int
    keys: tuple[torch.Tensor, ...]
    values: tuple[torch.Tensor, ...]

@torch.no_grad()
def decode_next_token_v1(
    model: TESSAStructuralizerV31, *, memory: torch.Tensor, token: torch.Tensor,
    cache: TESSAV31IncrementalCache | None = None,
) -> tuple[torch.Tensor, TESSAV31IncrementalCache]:
    """Exact unbatched-prefix V3.1 next-token logits with rolling K/V.

    At each layer the query attends to the previous window-1 entries plus
    itself, equivalent to V3.1 full-prefix SDPA. The positional embedding
    uses the same modulo-window scheme as decode_tokens(). Inference-only;
    independent model/Compiler proof still decides topology admission.
    """
    if model.training:
        raise RuntimeError("TESSA_V31_INCREMENTAL_REQUIRES_EVAL")
    if (
        token.ndim != 2 or token.shape[1] != 1
        or token.dtype != torch.long
        or memory.ndim != 3
        or memory.shape[0] != token.shape[0]
        or memory.shape[-1] != model.config.d_model
        or token.device != memory.device
        or not bool(((token >= 0) & (token < model.config.vocab_size)).all())
    ):
        raise ValueError("TESSA_V31_INCREMENTAL_TOKEN_OR_MEMORY_DOMAIN")
    if cache is None:
        position = 0
        old_keys = (None,) * len(model.decoder)
        old_values = (None,) * len(model.decoder)
    else:
        if (
            type(cache) is not TESSAV31IncrementalCache
            or cache.memory is not memory
            or len(cache.keys) != len(model.decoder)
            or len(cache.values) != len(model.decoder)
            or cache.position < 1
        ):
            raise ValueError("TESSA_V31_INCREMENTAL_FOREIGN_CACHE")
        position = cache.position
        old_keys, old_values = cache.keys, cache.values
    width = model.config.local_attention_window
    pos = torch.full_like(token, position % width)
    x = model.token_embedding(token) + model.position_embedding(pos)
    updated_k, updated_v = [], []
    for layer_id, layer in enumerate(model.decoder):
        normalized = layer.norm1(x)
        b, _, d = normalized.shape
        h = model.config.n_heads
        head_width = d // h
        qkv = layer.self_attn.qkv(normalized).reshape(
            b, 1, 3, h, head_width,
        )
        q, k, v = (
            z.transpose(1, 2)
            for z in qkv.unbind(dim=2)
        )
        if old_keys[layer_id] is not None:
            k = torch.cat((old_keys[layer_id], k), dim=2)
            v = torch.cat((old_values[layer_id], v), dim=2)
        k = k[:, :, -width:, :]
        v = v[:, :, -width:, :]
        attended = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0., is_causal=False,
        )
        projected = layer.self_attn.out(
            attended.transpose(1, 2).contiguous().reshape(b, 1, d),
        )
        x = x + projected
        cross, _ = layer.cross(
            layer.norm2(x), memory, memory, need_weights=False,
        )
        x = x + cross
        x = x + layer.mlp(layer.norm3(x))
        updated_k.append(k)
        updated_v.append(v)
    logits = model.lm_head(model.final_norm(x))[:, 0]
    return logits, TESSAV31IncrementalCache(
        memory=memory, position=position+1,
        keys=tuple(updated_k), values=tuple(updated_v),
    )


__all__ = ["TESSAV31IncrementalCache", "decode_next_token_v1"]
