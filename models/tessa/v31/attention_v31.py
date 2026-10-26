"""Chunk-independent rolling causal attention, including the current token.

The V2/V3 key_start=end-window implementation is deliberately not modified:
historical checkpoints have different attention semantics. A V3.1 checkpoint
must bind this semantics identity as well as all architecture parameters.
"""
from __future__ import annotations

from functools import lru_cache
import torch
from torch.nn import functional as F
from models.tessa.v2.model_v2 import WindowedCausalSelfAttentionV2

SEMANTICS_ID = "ROLLING_CAUSAL_SELF_INCLUSIVE_V31"


def rolling_sdpa(q, k, v, *, window, query_chunk_size):
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4 or q.shape[2] <= 0:
        raise ValueError("TESSA_V31_QKV_SHAPE")
    if min(window, query_chunk_size) <= 0:
        raise ValueError("TESSA_V31_WINDOW_CHUNK")
    length = q.shape[2]
    if length <= window:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0)
    chunks = []
    for start in range(0, length, query_chunk_size):
        end = min(length, start+query_chunk_size)
        key_start = max(0, start-window+1)
        qp = torch.arange(start, end, device=q.device)[:, None]
        kp = torch.arange(key_start, end, device=q.device)[None, :]
        allow = (kp <= qp) & (kp > qp-window)
        chunks.append(F.scaled_dot_product_attention(q[:, :, start:end], k[:, :, key_start:end],
                                                    v[:, :, key_start:end], attn_mask=allow, dropout_p=0))
    return torch.cat(chunks, 2)


@lru_cache(maxsize=16)
def _flex_mask(length, window, device):
    from torch.nn.attention.flex_attention import create_block_mask
    def allow(batch, head, q, k):
        return (q >= k) & (q-k < window)
    return create_block_mask(allow, None, None, length, length, device=device)


@lru_cache(maxsize=1)
def _flex_kernel():
    from torch.nn.attention.flex_attention import flex_attention
    return torch.compile(flex_attention, dynamic=False)


class WindowedCausalSelfAttentionV31(WindowedCausalSelfAttentionV2):
    semantics_id = SEMANTICS_ID

    def __init__(self, config, *, backend="SDPA"):
        super().__init__(config)
        if config.dropout != 0:
            raise ValueError("TESSA_V31_STOCHASTIC_BATCH_EQUIVALENCE_UNSEALED")
        if backend not in {"SDPA", "FLEX"}:
            raise ValueError("TESSA_V31_BACKEND")
        self.backend = backend

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != self.config.d_model or x.shape[1] <= 0:
            raise ValueError("TESSA_V31_SELF_ATTENTION_SHAPE")
        b, length, dim = x.shape
        q, k, v = (t.transpose(1, 2) for t in self.qkv(x).reshape(b, length, 3, self.config.n_heads, self.head_dim).unbind(2))
        if self.backend == "FLEX":
            if not x.is_cuda:
                raise ValueError("TESSA_V31_FLEX_REQUIRES_CUDA")
            y = _flex_kernel()(q, k, v, block_mask=_flex_mask(length, self.config.local_attention_window, str(x.device)))
        else:
            y = rolling_sdpa(q, k, v, window=self.config.local_attention_window,
                             query_chunk_size=self.config.query_chunk_size)
        return self.out(y.transpose(1, 2).contiguous().reshape(b, length, dim))
