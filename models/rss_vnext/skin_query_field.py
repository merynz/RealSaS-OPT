from __future__ import annotations

"""SkinTokens-inspired spatial query decoder on the SAME RSS 3D carrier.

Original SkinTokens: Michelangelo point/normal mesh encoding → causal Qwen3
skeleton+FSQ skin sequence → Tripo2 per-joint SkinCVAE query decoder.

RSS deliberate changes:
- the exact indexed structural mesh is the query domain; no second private
  Michelangelo scene memory or detached mesh rasterization;
- all skin code tokens are visible *only through predicted causal logits*;
- exact vertex H(x), XYZ and indexed mesh normals address output weights;
- cross attention exposes each of the T codes for each of J joints, rather
  than pooling them all into one compressed 7D vertex bottleneck;
- this is a learned skin-field hypothesis. Compiler alone qualifies and
  materializes the controller/skin bindings.

This is a research ablation, NOT a copy of SkinTokens' pretrained VAE decoder.
"""
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class RSSvNextSkinQueryOutput:
    weights: Tensor             # [B,V,J], exact structural mesh vertex indexing
    unnormalized_logits: Tensor # [B,V,J]
    valid_joint_mask: Tensor    # [B,J]
    authority: str = "UNQUALIFIED_INDEXED_SKIN_PROPOSAL"


class RSSvNextJointSkinQueryField(nn.Module):
    """Chunked vertex→(joint, FSQ token) cross attention with typed masks.

    Complexity O(V * J*T) with a bounded live vertex query chunk. The
    point-wise output preserves vertex ordering and arbitrary J (not a
    humanoid-only skeleton size).
    """

    def __init__(
        self,
        vertex_feature_dim: int,
        joint_feature_dim: int,
        code_dim: int,
        *,
        tokens_per_joint: int,
        hidden_dim: int,
        heads: int,
        max_query_chunk: int = 1024,
    ) -> None:
        super().__init__()
        if (
            min(vertex_feature_dim, joint_feature_dim, code_dim,
                tokens_per_joint, hidden_dim, heads, max_query_chunk) <= 0
            or hidden_dim % heads
            or max_query_chunk > 8192
            or tokens_per_joint > 32
        ):
            raise ValueError("RSS_VNEXT_SKIN_QUERY_FIELD_CONFIG_INVALID")
        self.vertex_feature_dim = vertex_feature_dim
        self.joint_feature_dim = joint_feature_dim
        self.code_dim = code_dim
        self.tokens_per_joint = tokens_per_joint
        self.max_query_chunk = max_query_chunk
        self.vertex_projection = nn.Sequential(
            nn.LayerNorm(vertex_feature_dim + 6),
            nn.Linear(vertex_feature_dim + 6, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.joint_projection = nn.Sequential(
            nn.LayerNorm(joint_feature_dim),
            nn.Linear(joint_feature_dim, hidden_dim),
        )
        self.code_projection = nn.Sequential(
            nn.LayerNorm(code_dim), nn.Linear(code_dim, hidden_dim),
        )
        self.code_position = nn.Parameter(
            torch.zeros(1, 1, tokens_per_joint, hidden_dim),
        )
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, heads, batch_first=True, dropout=0,
        )
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.joint_norm = nn.LayerNorm(hidden_dim)
        self.pair_score = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        vertex_h: Tensor,
        vertex_xyz: Tensor,
        joint_h: Tensor,
        code_features: Tensor,
        joint_mask: Tensor,
        *,
        vertex_normals: Tensor | None = None,
    ) -> RSSvNextSkinQueryOutput:
        if vertex_h.ndim != 3:
            raise ValueError("RSS_VNEXT_SKIN_QUERY_VERTEX_RANK")
        b, v, features = vertex_h.shape
        if vertex_xyz.shape != (b, v, 3) or features != self.vertex_feature_dim:
            raise ValueError("RSS_VNEXT_SKIN_QUERY_VERTEX_ADDRESS")
        if joint_h.ndim != 3 or joint_h.shape[:1] != (b,) or joint_h.shape[-1] != self.joint_feature_dim:
            raise ValueError("RSS_VNEXT_SKIN_QUERY_JOINT_ADDRESS")
        j = joint_h.shape[1]
        if (
            code_features.shape != (b, j, self.tokens_per_joint, self.code_dim)
        ):
            raise ValueError("RSS_VNEXT_SKIN_QUERY_CODE_ADDRESS")
        if min(b, v, j) <= 0:
            raise ValueError("RSS_VNEXT_SKIN_QUERY_EMPTY_DOMAIN")
        if joint_mask.shape != (b, j) or joint_mask.dtype != torch.bool:
            raise ValueError("RSS_VNEXT_SKIN_QUERY_JOINT_MASK")
        if not bool(joint_mask.any(dim=-1).all()):
            raise ValueError("RSS_VNEXT_SKIN_QUERY_ALL_JOINTS_UNKNOWN")
        if vertex_normals is None:
            # This means *not measured* rather than "the normals are zero".
            # Caller must provide indexed normals once structural mesh exists.
            vertex_normals = torch.zeros_like(vertex_xyz)
        if vertex_normals.shape != (b, v, 3):
            raise ValueError("RSS_VNEXT_SKIN_QUERY_NORMAL_ADDRESS")
        tensors = (vertex_h, vertex_xyz, vertex_normals, joint_h, code_features)
        if not all(bool(torch.isfinite(x).all()) for x in tensors):
            raise ValueError("RSS_VNEXT_SKIN_QUERY_NONFINITE")
        if len({x.device for x in tensors}) != 1:
            raise ValueError("RSS_VNEXT_SKIN_QUERY_DEVICE_DRIFT")

        vertex_queries = self.vertex_projection(
            torch.cat((vertex_h, vertex_xyz, vertex_normals), dim=-1),
        )
        jbase = self.joint_projection(joint_h)
        ctokens = (
            self.code_projection(code_features)
            + jbase[:, :, None, :]
            + self.code_position
        )
        # A query may attend every joint's every predicted code: no loss of
        # source token order, and padded joints cannot be selected.
        _, _, _, h = ctokens.shape
        kv = ctokens.reshape(b, j * self.tokens_per_joint, h)
        key_mask = (~joint_mask)[:, :, None].expand(
            b, j, self.tokens_per_joint,
        ).reshape(b, -1)
        per_joint = self.joint_norm(jbase + ctokens.mean(dim=2))
        out = []
        for start in range(0, v, self.max_query_chunk):
            q = vertex_queries[:, start : start + self.max_query_chunk]
            attended, _ = self.cross_attention(
                q, kv, kv, key_padding_mask=key_mask,
                need_weights=False,
            )
            q = self.query_norm(q + attended)
            # Pairwise learned compatibility makes each joint independently
            # queryable; masked joints cannot steal skin weight mass.
            n = q.shape[1]
            qp = q[:, :, None, :].expand(b, n, j, h)
            jp = per_joint[:, None].expand(b, n, j, h)
            logits = self.pair_score(
                torch.cat((qp * jp, qp - jp), dim=-1),
            ).squeeze(-1)
            logits = logits.masked_fill(
                ~joint_mask[:, None, :], torch.finfo(logits.dtype).min,
            )
            out.append(logits)
        logits = torch.cat(out, dim=1)
        weights = F.softmax(logits, dim=-1)
        weights = torch.where(
            joint_mask[:, None], weights, torch.zeros_like(weights),
        )
        if not bool(torch.isfinite(weights).all()):
            raise RuntimeError("RSS_VNEXT_SKIN_QUERY_NONFINITE_RESULT")
        return RSSvNextSkinQueryOutput(
            weights=weights, unnormalized_logits=logits,
            valid_joint_mask=joint_mask,
        )


__all__ = [
    "RSSvNextSkinQueryOutput",
    "RSSvNextJointSkinQueryField",
]
