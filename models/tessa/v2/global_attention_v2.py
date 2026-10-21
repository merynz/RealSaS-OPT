from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .config_v2 import TESSAStructuralizerConfigV2


@dataclass(frozen=True)
class TESSAGlobalHistoryPolicyV2:
    """Causal global-history policy for long topology sequences.

    Exact token attention is retained in the local trailing region. Older
    completed token blocks are represented by one K/V summary each. The
    coarse/local partition is anchored to an absolute query-chunk start rather
    than the currently available prefix end. This is essential: an already
    generated token must never change its representation merely because another
    token later arrives in the same execution chunk.
    """

    block_size: int = 256
    max_summary_tokens: int = 2048

    def validate(self) -> None:
        if self.block_size <= 0 or self.max_summary_tokens <= 0:
            raise ValueError("TESSA_V2_GLOBAL_HISTORY_POLICY_INVALID")


class HierarchicalCausalSelfAttentionV2(nn.Module):
    """Prefix-stable local exact attention plus coarse causal global history.

    The execution query chunk is only a vectorization unit. It must not change
    the mathematical history available to a token when the prefix later grows.
    Therefore every query in one absolute chunk uses the coarse/local partition
    fixed when that chunk begins. Causal masking still prevents future-token
    access inside the chunk.
    """

    def __init__(
        self,
        config: TESSAStructuralizerConfigV2,
        history: TESSAGlobalHistoryPolicyV2 = TESSAGlobalHistoryPolicyV2(),
    ) -> None:
        super().__init__()
        config.validate()
        history.validate()
        if config.d_model % config.n_heads:
            raise ValueError("TESSA_V2_GLOBAL_ATTN_HEAD_DIM_INVALID")
        self.config = config
        self.history = history
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, config.d_model * 3, bias=False)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)

    def summary_end_for_query_position(self, position: int) -> int:
        """Return the prefix-stable exclusive end of summarized history.

        `position` is the absolute zero-based query position. All queries in the
        same execution chunk intentionally share the same summary boundary. The
        block-floor keeps any partial history block exact so no token disappears
        at the coarse/local boundary.
        """

        if position < 0:
            raise ValueError("TESSA_V2_GLOBAL_ATTN_POSITION_NEGATIVE")
        query_chunk = int(self.config.query_chunk_size)
        local_window = int(self.config.local_attention_window)
        block = int(self.history.block_size)
        chunk_start = (int(position) // query_chunk) * query_chunk
        prefix_stable_end = chunk_start + 1
        desired_local_start = max(0, prefix_stable_end - local_window)
        return (desired_local_start // block) * block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.config.d_model:
            raise ValueError("TESSA_V2_GLOBAL_ATTN_INPUT_SHAPE_INVALID")
        b, length, dim = x.shape
        if length <= 0:
            raise ValueError("TESSA_V2_GLOBAL_ATTN_EMPTY_SEQUENCE")

        qkv = self.qkv(x).view(
            b,
            length,
            3,
            self.config.n_heads,
            self.head_dim,
        )
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        chunks: list[torch.Tensor] = []
        query_chunk = int(self.config.query_chunk_size)
        block = int(self.history.block_size)

        for start in range(0, length, query_chunk):
            end = min(length, start + query_chunk)
            summary_end = self.summary_end_for_query_position(start)
            local_start = summary_end
            summary_count = summary_end // block
            if summary_count > int(self.history.max_summary_tokens):
                raise RuntimeError(
                    "TESSA_V2_GLOBAL_HISTORY_CAPACITY_EXCEEDED:"
                    f"required={summary_count}:max={self.history.max_summary_tokens}"
                )

            q_chunk = q[:, :, start:end]
            local_k = k[:, :, local_start:end]
            local_v = v[:, :, local_start:end]

            if summary_count:
                history_k = k[:, :, :summary_end].reshape(
                    b,
                    self.config.n_heads,
                    summary_count,
                    block,
                    self.head_dim,
                ).mean(dim=3)
                history_v = v[:, :, :summary_end].reshape(
                    b,
                    self.config.n_heads,
                    summary_count,
                    block,
                    self.head_dim,
                ).mean(dim=3)
                key = torch.cat((history_k, local_k), dim=2)
                value = torch.cat((history_v, local_v), dim=2)
            else:
                key = local_k
                value = local_v

            q_pos = torch.arange(start, end, device=x.device)[:, None]
            local_pos = torch.arange(local_start, end, device=x.device)[None, :]
            local_allow = local_pos <= q_pos
            history_allow = torch.ones(
                (end - start, summary_count),
                dtype=torch.bool,
                device=x.device,
            )
            allow = torch.cat((history_allow, local_allow), dim=1)
            mask = torch.zeros(
                allow.shape,
                dtype=x.dtype,
                device=x.device,
            ).masked_fill(~allow, float("-inf"))

            y = F.scaled_dot_product_attention(
                q_chunk,
                key,
                value,
                attn_mask=mask[None, None],
                dropout_p=self.config.dropout if self.training else 0.0,
            )
            chunks.append(y)

        combined = (
            torch.cat(chunks, dim=2)
            .transpose(1, 2)
            .contiguous()
            .view(b, length, dim)
        )
        return self.out(combined)
