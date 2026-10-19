from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .config_v2 import TESSAStructuralizerConfigV2


@dataclass(frozen=True)
class TESSAGlobalHistoryPolicyV2:
    """Causal global-history policy for long topology sequences.

    Exact token attention is retained in the local trailing window. Older
    completed token blocks are represented by one K/V summary each. Summaries
    contain only positions strictly earlier than the local region, so training
    remains causal and matches incremental generation semantics.
    """

    block_size: int = 256
    max_summary_tokens: int = 2048

    def validate(self) -> None:
        if self.block_size <= 0 or self.max_summary_tokens <= 0:
            raise ValueError("TESSA_V2_GLOBAL_HISTORY_POLICY_INVALID")


class HierarchicalCausalSelfAttentionV2(nn.Module):
    """Local exact causal attention plus persistent coarse global history.

    This is deliberately simpler than importing a paper-specific sparse-attention
    implementation. The required invariant is causal structural memory:

      recent topology -> exact token context
      old topology    -> bounded summaries of every completed history block

    No future token contributes to a history summary seen by a query.
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
        local_window = int(self.config.local_attention_window)
        block = int(self.history.block_size)

        for start in range(0, length, query_chunk):
            end = min(length, start + query_chunk)
            desired_local_start = max(0, end - local_window)

            # Everything before summary_end is represented once per completed
            # block. The partial remainder stays exact in local attention, so no
            # past token disappears at the coarse/local boundary.
            summary_end = (desired_local_start // block) * block
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

            # History summaries are all strictly in the past. Only the exact
            # local token suffix requires a triangular causal mask.
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
