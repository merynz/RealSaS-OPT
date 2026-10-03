"""Bound CUDA BF16 inference attention memory without dropping keys or queries.

Audit execution adapter for pre-Ampere CUDA devices lacking BF16 fused SDPA.
Only noncausal, zero-dropout inference is admitted. Model parameters are unchanged.
"""
from contextlib import contextmanager
from unittest.mock import patch

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend


@contextmanager
def query_chunked_cuda_bf16_sdpa(query_chunk):
    if query_chunk <= 0:
        raise ValueError('query_chunk must be positive')
    original = torch.nn.functional.scaled_dot_product_attention
    telemetry = {'intercepted_calls': 0, 'query_blocks': 0, 'maximum_query_block': 0}

    def apply(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, *, scale=None, enable_gqa=False):
        options = dict(attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal,
                       scale=scale, enable_gqa=enable_gqa)
        if query.device.type != 'cuda' or query.dtype != torch.bfloat16:
            return original(query, key, value, **options)
        if torch.is_grad_enabled() or dropout_p != 0.0 or is_causal:
            raise RuntimeError('QUERY_CHUNK_ADAPTER_REQUIRES_NONCAUSAL_ZERO_DROPOUT_INFERENCE')
        length = query.shape[-2]
        if length == 0:
            raise ValueError('empty attention query')
        telemetry['intercepted_calls'] += 1
        outputs = []
        with sdpa_kernel([SDPBackend.MATH]):
            for start in range(0, length, query_chunk):
                stop = min(start + query_chunk, length)
                mask = attn_mask
                if mask is not None and mask.ndim >= 2 and mask.shape[-2] == length:
                    mask = mask[..., start:stop, :]
                # Every block sees the complete original K/V sequence. This is
                # query partitioning, not local/windowed or sampled attention.
                outputs.append(original(query[..., start:stop, :], key, value,
                                        attn_mask=mask, dropout_p=0.0, is_causal=False,
                                        scale=scale, enable_gqa=enable_gqa))
                telemetry['query_blocks'] += 1
                telemetry['maximum_query_block'] = max(telemetry['maximum_query_block'], stop - start)
        return torch.cat(outputs, dim=-2)

    with patch('torch.nn.functional.scaled_dot_product_attention', apply):
        yield telemetry
