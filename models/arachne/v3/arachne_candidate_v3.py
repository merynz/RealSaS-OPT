from __future__ import annotations

"""V7-native Arachne A1 predictor.

Product-time inputs are restricted to the rich, qualified conditioning boundary
defined by ``conditioning_v3``. No teacher skin, source mesh, semantic source
bone labels, character identity, provenance hashes or rejected Compiler
alternatives are neural inputs.

The model is deliberately non-autoregressive. It predicts a *set* of K=4
continuous field tokens per qualified joint. Token slot order is not semantic
authority; the frozen V7 decoder is full-set permutation invariant in FP32.
"""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math

import torch
from torch import nn


def _hash(payload: object) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class ArachneA1ConfigV3:
    model_dim: int = 512
    attention_heads: int = 8
    surface_graph_layers: int = 3
    surface_transformer_layers: int = 4
    skeleton_graph_layers: int = 3
    token_self_layers: int = 4
    token_surface_cross_layers: int = 3
    ffn_ratio: int = 4
    dropout: float = 0.0
    field_tokens: int = 4
    latent_channels: int = 512
    pair_geometry_dim: int = 10
    edge_feature_dim: int = 4
    architecture_id: str = "RealSaS.Arachne.A1.RichQualifiedSurfaceSkeleton.v3"

    def validate(self) -> None:
        if self.model_dim <= 0 or self.model_dim % self.attention_heads:
            raise ValueError("model_dim/head contract drift")
        if min(
            self.surface_graph_layers,
            self.surface_transformer_layers,
            self.skeleton_graph_layers,
            self.token_self_layers,
            self.token_surface_cross_layers,
            self.ffn_ratio,
            self.field_tokens,
            self.latent_channels,
        ) <= 0:
            raise ValueError("positive architecture cardinality required")
        if self.field_tokens != 4 or self.latent_channels != 512:
            raise ValueError("frozen K4 V7 output interface drift")
        if self.pair_geometry_dim != 10 or self.edge_feature_dim != 4:
            raise ValueError("typed relation width drift")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("invalid dropout")

    @property
    def config_hash(self) -> str:
        self.validate()
        return _hash(asdict(self))


class FeedForward(nn.Module):
    def __init__(self, dim: int, ratio: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EdgeMessageBlock(nn.Module):
    """Permutation-equivariant exact-edge message passing."""

    def __init__(self, dim: int, edge_dim: int, ratio: int, dropout: float):
        super().__init__()
        self.src_norm = nn.LayerNorm(dim)
        self.msg = nn.Sequential(
            nn.Linear(dim * 2 + edge_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.out_norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, ratio, dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        edge_mask: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, N, _ = x.shape
        out = torch.zeros_like(x)
        deg = torch.zeros((B, N, 1), device=x.device, dtype=x.dtype)
        xn = self.src_norm(x)
        for b in range(B):
            m = edge_mask[b].bool()
            if not bool(m.any()):
                continue
            e = edge_index[b, m]
            ef = edge_features[b, m]
            a = e[:, 0].long()
            c = e[:, 1].long()
            if (a < 0).any() or (c < 0).any() or (a >= N).any() or (c >= N).any():
                raise ValueError("edge endpoint outside padded surface")
            ma = self.msg(torch.cat([xn[b, a], xn[b, c], ef], dim=-1))
            mc = self.msg(torch.cat([xn[b, c], xn[b, a], ef], dim=-1))
            out[b].index_add_(0, a, ma)
            out[b].index_add_(0, c, mc)
            one = torch.ones((len(a), 1), device=x.device, dtype=x.dtype)
            deg[b].index_add_(0, a, one)
            deg[b].index_add_(0, c, one)
        out = out / deg.clamp_min(1.0)
        y = x + out
        y = y + self.ff(self.out_norm(y))
        return y * node_mask[..., None].to(y.dtype)


class TreeMessageBlock(nn.Module):
    """Qualified-tree message passing; topology is input authority, never predicted."""

    def __init__(self, dim: int, ratio: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.up = nn.Linear(dim * 2, dim)
        self.down = nn.Linear(dim * 2, dim)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, ratio, dropout)

    def forward(
        self,
        x: torch.Tensor,
        parent_indices: torch.Tensor,
        joint_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, J, _ = x.shape
        xn = self.norm(x)
        agg = torch.zeros_like(x)
        deg = torch.zeros((B, J, 1), device=x.device, dtype=x.dtype)
        for b in range(B):
            for j in range(J):
                if not bool(joint_mask[b, j]):
                    continue
                p = int(parent_indices[b, j].item())
                if p < 0:
                    continue
                if p >= J or not bool(joint_mask[b, p]):
                    raise ValueError("qualified parent outside joint mask")
                child_msg = self.up(torch.cat([xn[b, j], xn[b, p]], dim=-1))
                parent_msg = self.down(torch.cat([xn[b, p], xn[b, j]], dim=-1))
                agg[b, j] += child_msg
                agg[b, p] += parent_msg
                deg[b, j] += 1.0
                deg[b, p] += 1.0
        y = x + agg / deg.clamp_min(1.0)
        y = y + self.ff(self.ff_norm(y))
        return y * joint_mask[..., None].to(y.dtype)


class RelationBiasedCrossBlock(nn.Module):
    """Dense joint-field -> full-surface attention with legal relation bias."""

    def __init__(
        self,
        dim: int,
        heads: int,
        pair_dim: int,
        ratio: int,
        dropout: float,
        field_tokens: int,
    ):
        super().__init__()
        if dim % heads:
            raise ValueError("attention dimension/head mismatch")
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = dim // heads
        self.field_tokens = int(field_tokens)
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.pair_bias = nn.Sequential(
            nn.Linear(pair_dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, heads),
        )
        self.anchor_bias = nn.Parameter(torch.full((heads,), 0.25))
        self.drop = nn.Dropout(dropout)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, ratio, dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        surface: torch.Tensor,
        pair_geometry: torch.Tensor,
        support_anchor_matrix: torch.Tensor,
        surface_mask: torch.Tensor,
        joint_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, J, K, D = tokens.shape
        N = surface.shape[1]
        if K != self.field_tokens:
            raise ValueError("field-token count drift")
        Q = J * K
        tq = self.q(self.q_norm(tokens.reshape(B, Q, D)))
        sk = self.k(self.kv_norm(surface))
        sv = self.v(self.kv_norm(surface))

        q = tq.view(B, Q, self.heads, self.head_dim).transpose(1, 2)
        k = sk.view(B, N, self.heads, self.head_dim).transpose(1, 2)
        v = sv.view(B, N, self.heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        pb = self.pair_bias(pair_geometry)
        pb = pb.permute(0, 3, 2, 1)
        pb = pb[:, :, :, None, :].expand(-1, -1, -1, K, -1).reshape(B, self.heads, Q, N)
        ab = support_anchor_matrix[:, :, None, :].expand(-1, -1, K, -1).reshape(B, Q, N)
        ab = ab[:, None].to(scores.dtype) * self.anchor_bias[None, :, None, None]
        scores = scores + pb + ab

        valid_q = joint_mask[:, :, None].expand(-1, -1, K).reshape(B, Q)
        scores = scores.masked_fill(~surface_mask[:, None, None, :].bool(), -1e4)
        scores = scores.masked_fill(~valid_q[:, None, :, None], -1e4)
        attn = torch.softmax(scores, dim=-1)
        attn = attn * surface_mask[:, None, None, :].to(attn.dtype)
        attn = attn * valid_q[:, None, :, None].to(attn.dtype)
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        ctx = torch.matmul(self.drop(attn), v)
        ctx = ctx.transpose(1, 2).contiguous().view(B, Q, D)
        y = tokens.reshape(B, Q, D) + self.out(ctx)
        y = y + self.ff(self.ff_norm(y))
        y = y * valid_q[..., None].to(y.dtype)
        return y.reshape(B, J, K, D)


@dataclass
class ArachneA1RawOutputV3:
    field_tokens: torch.Tensor
    surface_memory: torch.Tensor
    joint_memory: torch.Tensor


class ArachneA1V3(nn.Module):
    """Rich S+Qualified-G -> unordered K4 latent-set predictor."""

    def __init__(self, config: ArachneA1ConfigV3 = ArachneA1ConfigV3()):
        super().__init__()
        config.validate()
        self.config = config
        d = config.model_dim

        # 3 P + 3 N + 1 N-valid + 8 support + 16 raster XY + 8 raster-valid
        # + 1 observed + 1 completed = 41.
        self.surface_in = nn.Sequential(nn.Linear(41, d), nn.GELU(), nn.Linear(d, d))

        # Shared view encoder + deterministic yaw code; no learned absolute view slot.
        self.view_in = nn.Sequential(nn.Linear(8, d // 2), nn.GELU(), nn.Linear(d // 2, d))

        self.surface_graph = nn.ModuleList([
            EdgeMessageBlock(d, config.edge_feature_dim, config.ffn_ratio, config.dropout)
            for _ in range(config.surface_graph_layers)
        ])
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.attention_heads,
            dim_feedforward=d * config.ffn_ratio,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.surface_global = nn.TransformerEncoder(
            enc_layer, num_layers=config.surface_transformer_layers, norm=nn.LayerNorm(d)
        )

        # position3 + root + deform-root + depth + parent-delta3 + parent-length = 10.
        self.joint_in = nn.Sequential(nn.Linear(10, d), nn.GELU(), nn.Linear(d, d))
        self.tree = nn.ModuleList([
            TreeMessageBlock(d, config.ffn_ratio, config.dropout)
            for _ in range(config.skeleton_graph_layers)
        ])
        self.anchor_context = nn.Linear(d, d, bias=False)

        self.token_prototypes = nn.Parameter(torch.randn(config.field_tokens, d) * 0.02)
        token_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.attention_heads,
            dim_feedforward=d * config.ffn_ratio,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.token_self = nn.TransformerEncoder(
            token_layer, num_layers=config.token_self_layers, norm=nn.LayerNorm(d)
        )
        self.cross = nn.ModuleList([
            RelationBiasedCrossBlock(
                d,
                config.attention_heads,
                config.pair_geometry_dim,
                config.ffn_ratio,
                config.dropout,
                config.field_tokens,
            )
            for _ in range(config.token_surface_cross_layers)
        ])
        self.out_norm = nn.LayerNorm(d)
        self.to_v7_latent = nn.Linear(d, config.latent_channels)

    @staticmethod
    def _yaw_code(device, dtype) -> torch.Tensor:
        yaw = torch.arange(8, device=device, dtype=dtype) * (math.pi / 4.0)
        return torch.stack(
            [torch.sin(yaw), torch.cos(yaw), torch.sin(2 * yaw), torch.cos(2 * yaw)], dim=-1
        )

    def _surface_features(
        self,
        positions: torch.Tensor,
        normals: torch.Tensor,
        normal_valid: torch.Tensor,
        support: torch.Tensor,
        raster_xy: torch.Tensor,
        raster_valid: torch.Tensor,
        observed: torch.Tensor,
        completed: torch.Tensor,
    ) -> torch.Tensor:
        flat = torch.cat([
            positions,
            normals,
            normal_valid[..., None].to(positions.dtype),
            support.to(positions.dtype),
            raster_xy.reshape(*raster_xy.shape[:-2], 16),
            raster_valid.to(positions.dtype),
            observed[..., None].to(positions.dtype),
            completed[..., None].to(positions.dtype),
        ], dim=-1)
        if flat.shape[-1] != 41:
            raise ValueError(f"surface neural width drift:{flat.shape[-1]}")
        base = self.surface_in(flat)

        yaw = self._yaw_code(positions.device, positions.dtype)
        yaw = yaw[None, None].expand(positions.shape[0], positions.shape[1], -1, -1)
        vf = torch.cat([
            support[..., None].to(positions.dtype),
            raster_valid[..., None].to(positions.dtype),
            raster_xy,
            yaw,
        ], dim=-1)
        vt = self.view_in(vf)
        present = (support | raster_valid)[..., None].to(vt.dtype)
        view_ctx = (vt * present).sum(dim=2) / present.sum(dim=2).clamp_min(1.0)
        return base + view_ctx

    def _joint_features(
        self,
        joint_positions: torch.Tensor,
        parent_indices: torch.Tensor,
        root_mask: torch.Tensor,
        deform_root_mask: torch.Tensor,
        depth: torch.Tensor,
        joint_mask: torch.Tensor,
    ) -> torch.Tensor:
        safe = parent_indices.clamp_min(0)
        parent = torch.gather(joint_positions, 1, safe[..., None].expand(-1, -1, 3))
        has_parent = (parent_indices >= 0) & joint_mask.bool()
        delta = (joint_positions - parent) * has_parent[..., None].to(joint_positions.dtype)
        length = torch.linalg.norm(delta, dim=-1, keepdim=True)
        feat = torch.cat([
            joint_positions,
            root_mask[..., None].to(joint_positions.dtype),
            deform_root_mask[..., None].to(joint_positions.dtype),
            depth[..., None].to(joint_positions.dtype),
            delta,
            length,
        ], dim=-1)
        if feat.shape[-1] != 10:
            raise AssertionError("joint feature width drift")
        return self.joint_in(feat) * joint_mask[..., None].to(feat.dtype)

    def forward(
        self,
        *,
        surface_positions_normalized: torch.Tensor,
        surface_normals: torch.Tensor,
        surface_normal_valid: torch.Tensor,
        surface_support_views: torch.Tensor,
        surface_raster_xy: torch.Tensor,
        surface_raster_valid: torch.Tensor,
        surface_observed: torch.Tensor,
        surface_completed: torch.Tensor,
        surface_mask: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        edge_mask: torch.Tensor,
        joint_positions_normalized: torch.Tensor,
        joint_mask: torch.Tensor,
        parent_indices: torch.Tensor,
        root_mask: torch.Tensor,
        deform_root_mask: torch.Tensor,
        joint_depth_normalized: torch.Tensor,
        support_anchor_matrix: torch.Tensor,
        pair_geometry: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> ArachneA1RawOutputV3:
        B, N, _ = surface_positions_normalized.shape
        J = joint_positions_normalized.shape[1]
        if pair_geometry.shape != (B, N, J, self.config.pair_geometry_dim):
            raise ValueError("pair geometry shape drift")
        if pair_mask.shape != (B, N, J):
            raise ValueError("pair mask shape drift")
        if support_anchor_matrix.shape != (B, J, N):
            raise ValueError("support anchor matrix shape drift")

        surface = self._surface_features(
            surface_positions_normalized,
            surface_normals,
            surface_normal_valid,
            surface_support_views,
            surface_raster_xy,
            surface_raster_valid,
            surface_observed,
            surface_completed,
        )
        surface = surface * surface_mask[..., None].to(surface.dtype)
        for block in self.surface_graph:
            surface = block(surface, edge_index, edge_features, edge_mask, surface_mask)
        surface = self.surface_global(surface, src_key_padding_mask=~surface_mask.bool())
        surface = surface * surface_mask[..., None].to(surface.dtype)

        joint = self._joint_features(
            joint_positions_normalized,
            parent_indices,
            root_mask,
            deform_root_mask,
            joint_depth_normalized,
            joint_mask,
        )
        for block in self.tree:
            joint = block(joint, parent_indices, joint_mask)

        # Exact sparse anchor identity is relation evidence, never a hard skin mask.
        a = support_anchor_matrix.to(surface.dtype)
        anchor_ctx = torch.einsum("bjn,bnd->bjd", a, surface) / a.sum(dim=-1, keepdim=True).clamp_min(1.0)
        joint = joint + self.anchor_context(anchor_ctx)
        joint = joint * joint_mask[..., None].to(joint.dtype)

        tokens = joint[:, :, None, :] + self.token_prototypes[None, None, :, :]
        valid_tok = joint_mask[:, :, None].expand(-1, -1, self.config.field_tokens)
        flat = tokens.reshape(B, J * self.config.field_tokens, self.config.model_dim)
        flat_mask = valid_tok.reshape(B, J * self.config.field_tokens)
        flat = self.token_self(flat, src_key_padding_mask=~flat_mask.bool())
        tokens = flat.reshape(B, J, self.config.field_tokens, self.config.model_dim)

        for block in self.cross:
            tokens = block(
                tokens,
                surface,
                pair_geometry,
                support_anchor_matrix,
                surface_mask,
                joint_mask,
            )

        z = self.to_v7_latent(self.out_norm(tokens))
        z = z * valid_tok[..., None].to(z.dtype)
        return ArachneA1RawOutputV3(z, surface, joint)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


__all__ = ["ArachneA1ConfigV3", "ArachneA1RawOutputV3", "ArachneA1V3"]
