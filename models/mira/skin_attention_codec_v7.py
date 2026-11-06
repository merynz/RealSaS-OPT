from __future__ import annotations

"""MIRA V7 learned-query FSQ-CVAE research candidate.

Separately implemented after auditing official SkinTokens attention encoders:
https://github.com/VAST-AI-Research/SkinTokens/blob/main/src/model/skin_vae/autoencoders/skin_fsq_cvae_model.py

- Active-support sampled per-joint surface context (TRAIN target only).
- Learned query attention compresses local influence distributions into T tokens.
- Straight-through FSQ and conditional carrier-native decoding inherited from V7.
- No source body-part naming, no segmentation labels, no graphics target.
- Memory-bounded sampling: B x J x K x H, not B x J x N x H.
This is NOT a port or checkpoint-compatible clone of upstream SkinTokens.
"""

from dataclasses import dataclass, asdict

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .skin_fsq_codec_v7 import (
    MIRASkinCodecV7, MIRACodecConfigV7,
    MIRA_V7_ARCHITECTURE_ID,
)

MIRA_V7_ATTENTION_ARCHITECTURE_ID = (
    "RealSaS.MIRA.AttentionCarrierCVAE.FSQ.AuthorizedMechanics.v7_0"
)


@dataclass(frozen=True)
class MIRAAttentionConfigV7(MIRACodecConfigV7):
    attention_heads: int = 8
    attention_layers: int = 4
    active_samples_per_joint: int = 128
    attention_dropout: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        if (self.attention_heads < 1 or self.attention_layers < 1
                or self.active_samples_per_joint < 1
                or self.hidden_dim % self.attention_heads):
            raise ValueError("MIRA_V7_ATTENTION_CONFIG_INVALID")
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError("MIRA_V7_ATTENTION_DROPOUT_INVALID")


class _JointLocalAttentionLayer(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float):
        super().__init__()
        self.norm_q = nn.LayerNorm(width)
        self.norm_kv = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(4 * width, width),
        )

    def forward(self, queries: Tensor, sampled_context: Tensor) -> Tensor:
        q = self.norm_q(queries)
        kv = self.norm_kv(sampled_context)
        value, _ = self.attn(q, kv, kv, need_weights=False)
        out = queries + value
        return out + self.ffn(self.norm_ffn(out))


class MIRAAttentionSkinCodecV7(MIRASkinCodecV7):
    architecture_id = MIRA_V7_ATTENTION_ARCHITECTURE_ID

    def __init__(self, config: MIRAAttentionConfigV7 = MIRAAttentionConfigV7()):
        super().__init__(config)
        h, t = config.hidden_dim, config.tokens_per_joint
        self.learned_queries = nn.Parameter(torch.empty(t, h))
        nn.init.normal_(self.learned_queries, std=0.02)
        self.local_weight_embedding = nn.Sequential(nn.Linear(1, h), nn.GELU(), nn.Linear(h, h))
        self.attention = nn.ModuleList([
            _JointLocalAttentionLayer(h, config.attention_heads, config.attention_dropout)
            for _ in range(config.attention_layers)
        ])
        self.attention_post_norm = nn.LayerNorm(h)
        self.attention_latent = nn.Linear(h, config.latent_dim)
        # Note: prior V7 regional latent encoder remains a registered baseline,
        # but not on this attention candidate's active forward path.

    def _active_samples(self, target: Tensor) -> tuple[Tensor, Tensor]:
        # Deterministic max-influence subset per joint. This is teacher-only
        # sampling; downstream decoder does NOT need teacher weights.
        _, n, _ = target.shape
        count = min(n, self.config.active_samples_per_joint)
        selected_weights, indices = torch.topk(
            target.transpose(1, 2).contiguous(), k=count,
            dim=-1, largest=True, sorted=True,
        )
        return indices, selected_weights

    def _latent(self, target: Tensor, vertices: Tensor, joints: Tensor) -> Tensor:
        b, n, j = self._shape(vertices, joints)
        sampled_ids, weights = self._active_samples(target)
        h, t = self.config.hidden_dim, self.config.tokens_per_joint
        vertex_embed = self.vertex_encoder(vertices)
        joint_embed = self.joint_encoder(joints)
        selected = torch.gather(
            vertex_embed[:, None, :, :].expand(b, j, n, h),
            2, sampled_ids[..., None].expand(b, j, sampled_ids.shape[-1], h),
        )
        context = selected + self.local_weight_embedding(weights[..., None])
        q = (self.learned_queries[None, None, :, :] +
             joint_embed[:, :, None, :])
        query_batch = q.reshape(b * j, t, h)
        sampled_context = context.reshape(b * j, sampled_ids.shape[-1], h)
        for layer in self.attention:
            query_batch = layer(query_batch, sampled_context)
        latent = self.attention_latent(self.attention_post_norm(query_batch))
        return latent.reshape(b, j, t, self.config.latent_dim)


__all__ = [
    "MIRA_V7_ATTENTION_ARCHITECTURE_ID",
    "MIRAAttentionConfigV7",
    "MIRAAttentionSkinCodecV7",
]
