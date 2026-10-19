from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .court_v2 import evaluate_decoded_topology_v2
from .generation_v2 import (
    TESSAGeneratedCandidateV2,
    TESSAGenerationPolicyV2,
    TESSAGrammarStateV2,
    _choose_token,
)
from .global_attention_v2 import HierarchicalCausalSelfAttentionV2
from .model_global_v2 import TESSAStructuralizerGlobalV2


@dataclass
class _LayerIncrementalCacheV2:
    summary_k: torch.Tensor | None = None
    summary_v: torch.Tensor | None = None
    local_k: torch.Tensor | None = None
    local_v: torch.Tensor | None = None
    local_start: int = 0
    total_tokens: int = 0


@dataclass
class TESSAIncrementalStateV2:
    layers: list[_LayerIncrementalCacheV2]
    token_count: int = 0

    @classmethod
    def for_model(cls, model: TESSAStructuralizerGlobalV2) -> "TESSAIncrementalStateV2":
        return cls(layers=[_LayerIncrementalCacheV2() for _ in model.decoder])


def _append_cache(
    cache: _LayerIncrementalCacheV2,
    k: torch.Tensor,
    v: torch.Tensor,
    attention: HierarchicalCausalSelfAttentionV2,
) -> tuple[torch.Tensor, torch.Tensor]:
    # k/v are [B,H,1,D]. Generation is single-asset but keeping B generic makes
    # parity tests easier and does not change cache semantics.
    if cache.local_k is None:
        cache.local_k = k
        cache.local_v = v
    else:
        cache.local_k = torch.cat((cache.local_k, k), dim=2)
        cache.local_v = torch.cat((cache.local_v, v), dim=2)
    cache.total_tokens += 1

    window = int(attention.config.local_attention_window)
    block = int(attention.history.block_size)
    desired_local_start = max(0, cache.total_tokens - window)
    summary_end = (desired_local_start // block) * block
    while cache.local_start < summary_end:
        if cache.local_k is None or cache.local_k.shape[2] < block:
            raise AssertionError("TESSA_V2_INCREMENTAL_CACHE_BLOCK_UNDERFLOW")
        block_k = cache.local_k[:, :, :block]
        block_v = cache.local_v[:, :, :block]
        mean_k = block_k.mean(dim=2, keepdim=True)
        mean_v = block_v.mean(dim=2, keepdim=True)
        cache.summary_k = mean_k if cache.summary_k is None else torch.cat((cache.summary_k, mean_k), dim=2)
        cache.summary_v = mean_v if cache.summary_v is None else torch.cat((cache.summary_v, mean_v), dim=2)
        cache.local_k = cache.local_k[:, :, block:]
        cache.local_v = cache.local_v[:, :, block:]
        cache.local_start += block

    expected_summaries = summary_end // block
    actual_summaries = 0 if cache.summary_k is None else int(cache.summary_k.shape[2])
    if actual_summaries != expected_summaries:
        raise AssertionError("TESSA_V2_INCREMENTAL_SUMMARY_COUNT_DRIFT")
    if actual_summaries > int(attention.history.max_summary_tokens):
        raise RuntimeError("TESSA_V2_GLOBAL_HISTORY_CAPACITY_EXCEEDED")

    if cache.summary_k is None:
        key, value = cache.local_k, cache.local_v
    else:
        key = torch.cat((cache.summary_k, cache.local_k), dim=2)
        value = torch.cat((cache.summary_v, cache.local_v), dim=2)
    return key, value


def _incremental_self_attention(
    attention: HierarchicalCausalSelfAttentionV2,
    x: torch.Tensor,
    cache: _LayerIncrementalCacheV2,
) -> torch.Tensor:
    if x.ndim != 3 or x.shape[1] != 1 or x.shape[-1] != attention.config.d_model:
        raise ValueError("TESSA_V2_INCREMENTAL_SELF_ATTN_SHAPE_INVALID")
    b = x.shape[0]
    qkv = attention.qkv(x).view(
        b, 1, 3, attention.config.n_heads, attention.head_dim
    )
    q, k, v = qkv.unbind(dim=2)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    key, value = _append_cache(cache, k, v, attention)
    y = F.scaled_dot_product_attention(
        q,
        key,
        value,
        dropout_p=0.0,
    )
    y = y.transpose(1, 2).contiguous().view(b, 1, attention.config.d_model)
    return attention.out(y)


@torch.no_grad()
def incremental_decode_step_v2(
    model: TESSAStructuralizerGlobalV2,
    *,
    memory: torch.Tensor,
    token_ids: torch.Tensor,
    state: TESSAIncrementalStateV2,
) -> torch.Tensor:
    """Consume one token per batch row and return logits for the next token."""

    if model.training:
        raise ValueError("TESSA_V2_INCREMENTAL_MODEL_MUST_BE_EVAL")
    if token_ids.ndim != 1 or token_ids.shape[0] != memory.shape[0]:
        raise ValueError("TESSA_V2_INCREMENTAL_TOKEN_BATCH_INVALID")
    if len(state.layers) != len(model.decoder):
        raise ValueError("TESSA_V2_INCREMENTAL_LAYER_CACHE_COUNT_INVALID")
    if token_ids.numel() and (int(token_ids.min()) < 0 or int(token_ids.max()) >= model.config.vocab_size):
        raise ValueError("TESSA_V2_INCREMENTAL_TOKEN_OUT_OF_RANGE")

    position = state.token_count % int(model.config.local_attention_window)
    pos = torch.full_like(token_ids, position)
    x = model.token_embedding(token_ids)[:, None, :] + model.position_embedding(pos)[:, None, :]

    for layer_index, block in enumerate(model.decoder):
        if not isinstance(block.self_attn, HierarchicalCausalSelfAttentionV2):
            raise TypeError("TESSA_V2_INCREMENTAL_REQUIRES_GLOBAL_HISTORY_MODEL")
        normed = block.norm1(x)
        x = x + _incremental_self_attention(block.self_attn, normed, state.layers[layer_index])
        q = block.norm2(x)
        cross, _ = block.cross(q, memory, memory, need_weights=False)
        x = x + cross
        x = x + block.mlp(block.norm3(x))

    state.token_count += 1
    return model.lm_head(model.final_norm(x))[:, 0, :]


@torch.no_grad()
def incremental_prefix_logits_v2(
    model: TESSAStructuralizerGlobalV2,
    *,
    memory: torch.Tensor,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Reference helper used by parity tests and engineering diagnostics."""

    if input_ids.ndim != 2 or input_ids.shape[0] != memory.shape[0]:
        raise ValueError("TESSA_V2_INCREMENTAL_PREFIX_SHAPE_INVALID")
    previous = model.training
    model.eval()
    try:
        state = TESSAIncrementalStateV2.for_model(model)
        rows = []
        for index in range(input_ids.shape[1]):
            rows.append(
                incremental_decode_step_v2(
                    model,
                    memory=memory,
                    token_ids=input_ids[:, index],
                    state=state,
                )
            )
        return torch.stack(rows, dim=1)
    finally:
        model.train(previous)


@torch.no_grad()
def generate_tessa_candidate_incremental_v2(
    model: TESSAStructuralizerGlobalV2,
    *,
    memory: torch.Tensor,
    policy: TESSAGenerationPolicyV2 = TESSAGenerationPolicyV2(),
    seed: int | None = None,
) -> TESSAGeneratedCandidateV2:
    """Strict grammar-constrained free-running generation with exact KV caching.

    This path may replace the prefix-recompute reference generator only while the
    dedicated parity court remains green.
    """

    policy.validate()
    if memory.ndim != 3 or memory.shape[0] != 1 or memory.shape[-1] != model.config.d_model:
        raise ValueError("TESSA_V2_INCREMENTAL_GENERATION_REQUIRES_SINGLE_ASSET_MEMORY")
    previous = model.training
    model.eval()
    grammar = TESSAGrammarStateV2()
    cache = TESSAIncrementalStateV2.for_model(model)
    tokens = [model.config.BOS]
    logp_sum = 0.0
    generator = None
    if seed is not None:
        generator = torch.Generator(device=memory.device)
        generator.manual_seed(int(seed))
    stop_reason = "MAXIMUM_NEW_TOKENS_REACHED"
    try:
        logits = incremental_decode_step_v2(
            model,
            memory=memory,
            token_ids=torch.tensor([model.config.BOS], device=memory.device, dtype=torch.long),
            state=cache,
        )[0]
        for _ in range(policy.maximum_new_tokens):
            allowed = grammar.allowed_token_mask(config=model.config, device=memory.device)
            token, logp = _choose_token(logits, allowed, policy=policy, generator=generator)
            tokens.append(token)
            logp_sum += logp
            grammar.advance(token, config=model.config)
            if grammar.complete:
                stop_reason = "EOS"
                break
            logits = incremental_decode_step_v2(
                model,
                memory=memory,
                token_ids=torch.tensor([token], device=memory.device, dtype=torch.long),
                state=cache,
            )[0]
    finally:
        model.train(previous)

    court = evaluate_decoded_topology_v2(tokens, config=model.config)
    return TESSAGeneratedCandidateV2(
        token_ids=tuple(tokens),
        topology_court=court,
        completed=bool(grammar.complete),
        stop_reason=stop_reason,
        token_log_probability_sum=float(logp_sum),
    )
