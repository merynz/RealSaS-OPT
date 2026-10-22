from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .model_v2 import TESSAStructuralizerV2, WindowedCausalSelfAttentionV2


@dataclass
class _ProjectedCrossMemoryV2:
    key: torch.Tensor
    value: torch.Tensor


@dataclass
class _WindowedLayerCacheV2:
    stable_k: torch.Tensor | None = None
    stable_v: torch.Tensor | None = None
    stable_start: int = 0
    cross_memory: _ProjectedCrossMemoryV2 | None = None


@dataclass
class TESSAWindowedIncrementalStateV2:
    """Exact cached state for the shipped windowed TESSA decoder.

    There are two exact execution regimes:

    1. PRE_WINDOW token-fast path. While total consumed tokens are <= the local
       attention window, future tokens are causally masked and no prior token can
       change. Each token is therefore immediately stable and can use ordinary
       per-token KV caching.

    2. POST_WINDOW chunk-exact path. Once the prefix exceeds the local window,
       WindowedCausalSelfAttentionV2 anchors key_start to the *current chunk end*.
       Growing a partial chunk can therefore change earlier hidden states inside
       that same chunk. Completed chunks remain stable, but the active partial
       chunk must be recomputed exactly on every token.

    Cross-attention memory is immutable during generation. Its K/V projections
    are cached once per decoder layer and reused in both regimes.
    """

    layers: list[_WindowedLayerCacheV2]
    stable_token_count: int = 0
    current_token_ids: list[int] | None = None

    @classmethod
    def for_model(cls, model: TESSAStructuralizerV2) -> "TESSAWindowedIncrementalStateV2":
        return cls(
            layers=[_WindowedLayerCacheV2() for _ in model.decoder],
            stable_token_count=0,
            current_token_ids=[],
        )


def _split_mha_projection(mha, *, query: torch.Tensor | None = None, memory: torch.Tensor | None = None):
    """Project nn.MultiheadAttention tensors using its packed Q/K/V weights."""

    if mha.in_proj_weight is None:
        raise TypeError("TESSA_V2_WINDOWED_INCREMENTAL_REQUIRES_PACKED_MHA_PROJECTION")
    embed = int(mha.embed_dim)
    wq, wk, wv = mha.in_proj_weight.split(embed, dim=0)
    if mha.in_proj_bias is None:
        bq = bk = bv = None
    else:
        bq, bk, bv = mha.in_proj_bias.split(embed, dim=0)
    q = F.linear(query, wq, bq) if query is not None else None
    k = F.linear(memory, wk, bk) if memory is not None else None
    v = F.linear(memory, wv, bv) if memory is not None else None
    return q, k, v


def _reshape_heads(x: torch.Tensor, *, n_heads: int) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError("TESSA_V2_WINDOWED_INCREMENTAL_HEAD_RESHAPE_INVALID")
    b, length, dim = x.shape
    if dim % n_heads:
        raise ValueError("TESSA_V2_WINDOWED_INCREMENTAL_HEAD_DIM_INVALID")
    return x.view(b, length, n_heads, dim // n_heads).transpose(1, 2)


def _cross_attention_cached(block, q_input: torch.Tensor, *, cache: _WindowedLayerCacheV2, memory: torch.Tensor) -> torch.Tensor:
    mha = block.cross
    if cache.cross_memory is None:
        _, k, v = _split_mha_projection(mha, memory=memory)
        assert k is not None and v is not None
        cache.cross_memory = _ProjectedCrossMemoryV2(
            key=_reshape_heads(k, n_heads=int(mha.num_heads)).detach(),
            value=_reshape_heads(v, n_heads=int(mha.num_heads)).detach(),
        )
    q, _, _ = _split_mha_projection(mha, query=q_input)
    assert q is not None
    qh = _reshape_heads(q, n_heads=int(mha.num_heads))
    y = F.scaled_dot_product_attention(
        qh,
        cache.cross_memory.key,
        cache.cross_memory.value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
    )
    y = y.transpose(1, 2).contiguous().view(q_input.shape[0], q_input.shape[1], int(mha.embed_dim))
    return mha.out_proj(y)


def _append_stable_kv(cache: _WindowedLayerCacheV2, k: torch.Tensor, v: torch.Tensor, *, new_stable_count: int, window: int) -> None:
    if cache.stable_k is None:
        combined_k = k
        combined_v = v
        combined_start = new_stable_count - k.shape[2]
    else:
        combined_k = torch.cat((cache.stable_k, k), dim=2)
        combined_v = torch.cat((cache.stable_v, v), dim=2)
        combined_start = cache.stable_start
    desired_start = max(0, new_stable_count - window)
    trim = desired_start - combined_start
    if trim < 0 or trim > combined_k.shape[2]:
        raise AssertionError("TESSA_V2_WINDOWED_INCREMENTAL_STABLE_APPEND_TRIM_INVALID")
    cache.stable_k = combined_k[:, :, trim:].detach()
    cache.stable_v = combined_v[:, :, trim:].detach()
    cache.stable_start = desired_start


def _token_fast_self_attention(
    attention: WindowedCausalSelfAttentionV2,
    x: torch.Tensor,
    *,
    cache: _WindowedLayerCacheV2,
    absolute_position: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if x.ndim != 3 or x.shape[:2] != (1, 1):
        raise ValueError("TESSA_V2_WINDOWED_INCREMENTAL_FAST_TOKEN_SHAPE_INVALID")
    qkv = attention.qkv(x).view(
        1, 1, 3, attention.config.n_heads, attention.head_dim
    )
    q, current_k, current_v = qkv.unbind(dim=2)
    q = q.transpose(1, 2)
    current_k = current_k.transpose(1, 2)
    current_v = current_v.transpose(1, 2)

    key_start = max(0, absolute_position + 1 - int(attention.config.local_attention_window))
    previous_k = previous_v = None
    if absolute_position > key_start:
        if cache.stable_k is None or cache.stable_v is None:
            raise AssertionError("TESSA_V2_WINDOWED_INCREMENTAL_FAST_CACHE_MISSING")
        if cache.stable_start > key_start:
            raise AssertionError("TESSA_V2_WINDOWED_INCREMENTAL_FAST_CACHE_UNDERFLOW")
        offset = key_start - cache.stable_start
        count = absolute_position - key_start
        previous_k = cache.stable_k[:, :, offset : offset + count]
        previous_v = cache.stable_v[:, :, offset : offset + count]
        if previous_k.shape[2] != count or previous_v.shape[2] != count:
            raise AssertionError("TESSA_V2_WINDOWED_INCREMENTAL_FAST_CACHE_LENGTH_DRIFT")

    if previous_k is None:
        key = current_k
        value = current_v
    else:
        key = torch.cat((previous_k, current_k), dim=2)
        value = torch.cat((previous_v, current_v), dim=2)

    y = F.scaled_dot_product_attention(q, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)
    y = y.transpose(1, 2).contiguous().view(1, 1, attention.config.d_model)
    return attention.out(y), current_k, current_v


def _current_chunk_attention(
    attention: WindowedCausalSelfAttentionV2,
    x: torch.Tensor,
    *,
    cache: _WindowedLayerCacheV2,
    stable_token_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if x.ndim != 3 or x.shape[0] != 1 or x.shape[-1] != attention.config.d_model:
        raise ValueError("TESSA_V2_WINDOWED_INCREMENTAL_INPUT_SHAPE_INVALID")

    _, current_len, dim = x.shape
    if current_len <= 0 or current_len > int(attention.config.query_chunk_size):
        raise ValueError("TESSA_V2_WINDOWED_INCREMENTAL_CURRENT_CHUNK_INVALID")

    qkv = attention.qkv(x).view(
        1,
        current_len,
        3,
        attention.config.n_heads,
        attention.head_dim,
    )
    q, current_k, current_v = qkv.unbind(dim=2)
    q = q.transpose(1, 2)
    current_k = current_k.transpose(1, 2)
    current_v = current_v.transpose(1, 2)

    end = stable_token_count + current_len
    key_start = max(0, end - int(attention.config.local_attention_window))

    previous_k = None
    previous_v = None
    if stable_token_count > key_start:
        if cache.stable_k is None or cache.stable_v is None:
            raise AssertionError("TESSA_V2_WINDOWED_INCREMENTAL_STABLE_CACHE_MISSING")
        if cache.stable_start > key_start:
            raise AssertionError("TESSA_V2_WINDOWED_INCREMENTAL_STABLE_CACHE_UNDERFLOW")
        offset = key_start - cache.stable_start
        count = stable_token_count - key_start
        previous_k = cache.stable_k[:, :, offset : offset + count]
        previous_v = cache.stable_v[:, :, offset : offset + count]
        if previous_k.shape[2] != count or previous_v.shape[2] != count:
            raise AssertionError("TESSA_V2_WINDOWED_INCREMENTAL_STABLE_CACHE_LENGTH_DRIFT")

    if previous_k is None:
        key = current_k
        value = current_v
    else:
        key = torch.cat((previous_k, current_k), dim=2)
        value = torch.cat((previous_v, current_v), dim=2)

    q_pos = torch.arange(stable_token_count, end, device=x.device)[:, None]
    k_pos = torch.arange(key_start, end, device=x.device)[None, :]
    allow = k_pos <= q_pos
    mask = torch.zeros(
        (current_len, end - key_start),
        dtype=x.dtype,
        device=x.device,
    ).masked_fill(~allow, float("-inf"))

    y = F.scaled_dot_product_attention(
        q,
        key,
        value,
        attn_mask=mask[None, None],
        dropout_p=0.0,
    )
    y = y.transpose(1, 2).contiguous().view(1, current_len, dim)
    return attention.out(y), current_k, current_v


def _finalize_chunk(
    state: TESSAWindowedIncrementalStateV2,
    *,
    layer_current_kv: list[tuple[torch.Tensor, torch.Tensor]],
    config,
) -> None:
    chunk = int(config.query_chunk_size)
    if state.current_token_ids is None or len(state.current_token_ids) != chunk:
        raise AssertionError("TESSA_V2_WINDOWED_INCREMENTAL_FINALIZE_PARTIAL_CHUNK")

    old_stable_count = state.stable_token_count
    new_stable_count = old_stable_count + chunk
    desired_start = max(0, new_stable_count - int(config.local_attention_window))

    for cache, (current_k, current_v) in zip(state.layers, layer_current_kv, strict=True):
        if cache.stable_k is None:
            combined_k = current_k
            combined_v = current_v
            combined_start = old_stable_count
        else:
            combined_k = torch.cat((cache.stable_k, current_k), dim=2)
            combined_v = torch.cat((cache.stable_v, current_v), dim=2)
            combined_start = cache.stable_start

        trim = desired_start - combined_start
        if trim < 0 or trim > combined_k.shape[2]:
            raise AssertionError("TESSA_V2_WINDOWED_INCREMENTAL_TRIM_INVALID")
        cache.stable_k = combined_k[:, :, trim:].detach()
        cache.stable_v = combined_v[:, :, trim:].detach()
        cache.stable_start = desired_start

    state.stable_token_count = new_stable_count
    state.current_token_ids.clear()


@torch.no_grad()
def incremental_decode_step_windowed_v2(
    model: TESSAStructuralizerV2,
    *,
    memory: torch.Tensor,
    token_ids: torch.Tensor,
    state: TESSAWindowedIncrementalStateV2,
) -> torch.Tensor:
    """Consume one token and return next-token logits with exact reference parity."""

    if model.training:
        raise ValueError("TESSA_V2_WINDOWED_INCREMENTAL_MODEL_MUST_BE_EVAL")
    if memory.ndim != 3 or memory.shape[0] != 1 or memory.shape[-1] != model.config.d_model:
        raise ValueError("TESSA_V2_WINDOWED_INCREMENTAL_MEMORY_SHAPE_INVALID")
    if token_ids.ndim != 1 or token_ids.shape != (1,):
        raise ValueError("TESSA_V2_WINDOWED_INCREMENTAL_TOKEN_BATCH_INVALID")
    if len(state.layers) != len(model.decoder):
        raise ValueError("TESSA_V2_WINDOWED_INCREMENTAL_LAYER_CACHE_COUNT_INVALID")
    if int(token_ids[0]) < 0 or int(token_ids[0]) >= model.config.vocab_size:
        raise ValueError("TESSA_V2_WINDOWED_INCREMENTAL_TOKEN_OUT_OF_RANGE")
    if state.current_token_ids is None:
        state.current_token_ids = []

    for block in model.decoder:
        if not isinstance(block.self_attn, WindowedCausalSelfAttentionV2):
            raise TypeError("TESSA_V2_WINDOWED_INCREMENTAL_REQUIRES_WINDOWED_MODEL")

    window = int(model.config.local_attention_window)
    chunk = int(model.config.query_chunk_size)
    consumed = state.stable_token_count + len(state.current_token_ids)

    # Exact token-fast regime. Before the local window is exceeded, adding a
    # future token cannot change any previous token because causal masking is the
    # only active restriction. Every token is therefore immediately stable.
    if consumed < window and not state.current_token_ids:
        absolute_position = state.stable_token_count
        pos = torch.tensor([absolute_position % window], dtype=torch.long, device=token_ids.device)
        x = model.token_embedding(token_ids[None, :]) + model.position_embedding(pos)[None]
        layer_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_index, block in enumerate(model.decoder):
            normed = block.norm1(x)
            self_update, current_k, current_v = _token_fast_self_attention(
                block.self_attn,
                normed,
                cache=state.layers[layer_index],
                absolute_position=absolute_position,
            )
            layer_kv.append((current_k, current_v))
            x = x + self_update
            q = block.norm2(x)
            x = x + _cross_attention_cached(
                block,
                q,
                cache=state.layers[layer_index],
                memory=memory,
            )
            x = x + block.mlp(block.norm3(x))
        logits = model.lm_head(model.final_norm(x))[:, -1, :]
        new_count = state.stable_token_count + 1
        for cache, (k, v) in zip(state.layers, layer_kv, strict=True):
            _append_stable_kv(cache, k, v, new_stable_count=new_count, window=window)
        state.stable_token_count = new_count
        return logits

    # Once the local window is exceeded, exact shipped semantics require active
    # partial-chunk recomputation because key_start depends on current chunk end.
    if state.stable_token_count % chunk:
        raise AssertionError("TESSA_V2_WINDOWED_INCREMENTAL_POST_WINDOW_ALIGNMENT_DRIFT")
    state.current_token_ids.append(int(token_ids[0]))
    if len(state.current_token_ids) > chunk:
        raise AssertionError("TESSA_V2_WINDOWED_INCREMENTAL_CHUNK_OVERFLOW")

    current_ids = torch.tensor(
        state.current_token_ids,
        dtype=torch.long,
        device=token_ids.device,
    )[None, :]
    current_len = current_ids.shape[1]
    absolute_positions = (
        torch.arange(
            state.stable_token_count,
            state.stable_token_count + current_len,
            device=token_ids.device,
            dtype=torch.long,
        )
        % window
    )
    x = model.token_embedding(current_ids) + model.position_embedding(absolute_positions)[None]

    layer_current_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer_index, block in enumerate(model.decoder):
        normed = block.norm1(x)
        self_update, current_k, current_v = _current_chunk_attention(
            block.self_attn,
            normed,
            cache=state.layers[layer_index],
            stable_token_count=state.stable_token_count,
        )
        layer_current_kv.append((current_k, current_v))
        x = x + self_update
        q = block.norm2(x)
        x = x + _cross_attention_cached(
            block,
            q,
            cache=state.layers[layer_index],
            memory=memory,
        )
        x = x + block.mlp(block.norm3(x))

    logits = model.lm_head(model.final_norm(x))[:, -1, :]

    if current_len == chunk:
        _finalize_chunk(
            state,
            layer_current_kv=layer_current_kv,
            config=model.config,
        )

    return logits


@torch.no_grad()
def assert_windowed_incremental_reference_parity_v2(
    *,
    device: torch.device | str = "cpu",
    atol: float = 2.0e-5,
    rtol: float = 2.0e-5,
) -> dict[str, float | int | str]:
    """Deterministic proof against model.decode_tokens across chunk/window boundaries."""

    from .config_v2 import TESSAStructuralizerConfigV2

    torch.manual_seed(26091991)
    cfg = TESSAStructuralizerConfigV2(
        d_model=32,
        n_heads=4,
        decoder_layers=2,
        mlp_ratio=2,
        dropout=0.0,
        coordinate_bins=128,
        max_faces=64,
        max_vertices=64,
        scene_memory_tokens=2,
        surface_memory_tokens=2,
        memory_source_width=16,
        memory_refinement_layers=0,
        local_attention_window=8,
        query_chunk_size=4,
        gradient_checkpointing=False,
    )
    model = TESSAStructuralizerV2(
        scene_input_dim=8,
        surface_input_dim=8,
        config=cfg,
    ).to(device)
    model.eval()
    memory = torch.randn(1, cfg.total_conditioning_tokens, cfg.d_model, device=device)
    tokens = torch.randint(0, cfg.vocab_size, (17,), device=device)
    state = TESSAWindowedIncrementalStateV2.for_model(model)

    max_abs = 0.0
    max_rel = 0.0
    checked_prefixes: list[int] = []
    for index in range(tokens.numel()):
        inc = incremental_decode_step_windowed_v2(
            model,
            memory=memory,
            token_ids=tokens[index : index + 1],
            state=state,
        )[0]
        ref = model.decode_tokens(
            memory=memory,
            input_ids=tokens[: index + 1][None, :],
        ).logits[0, -1]
        diff = (inc - ref).abs()
        max_abs = max(max_abs, float(diff.max().item()))
        denom = ref.abs().clamp_min(1.0e-8)
        max_rel = max(max_rel, float((diff / denom).max().item()))
        checked_prefixes.append(index + 1)
        if not torch.allclose(inc, ref, atol=atol, rtol=rtol):
            raise AssertionError(
                "TESSA_V2_WINDOWED_INCREMENTAL_REFERENCE_PARITY_FAIL:"
                f"prefix={index + 1}:max_abs={max_abs:.9g}:max_rel={max_rel:.9g}"
            )

    # 17 prefixes exercise: token-fast regime 1..8, first post-window chunk
    # prefixes 9..12, a completed post-window chunk boundary at 12, and another
    # partial chunk 13..17. In production this corresponds exactly to proving
    # the semantics around 4095/4096/4097 and subsequent 256-token boundaries.
    return {
        "status": "PASS__WINDOWED_INCREMENTAL_REFERENCE_PARITY",
        "prefix_count": int(tokens.numel()),
        "query_chunk_size": int(cfg.query_chunk_size),
        "local_attention_window": int(cfg.local_attention_window),
        "token_fast_prefixes": int(cfg.local_attention_window),
        "post_window_prefixes": int(tokens.numel() - cfg.local_attention_window),
        "max_abs_error": max_abs,
        "max_relative_error": max_rel,
    }
