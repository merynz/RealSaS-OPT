from __future__ import annotations

"""Arachne V7-native A1 predictor v4.

Design goals:
- no historical 20D/8D conditioning bottleneck;
- view-order safe shared view evidence with explicit yaw sidecar;
- exact GSA graph + qualified tree + sparse support anchors + dense 10D pair geometry;
- dense bidirectional surface <-> joint-field reasoning;
- non-autoregressive unordered K=4 latent-set output for frozen V7 decoder;
- no joint-ID / fixed-bone vocabulary / hard nearest-joint mask.
"""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math

import torch
from torch import nn


def _hash(payload: object) -> str:
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


@dataclass(frozen=True)
class ArachneA1ConfigV4:
    model_dim: int = 640
    attention_heads: int = 10
    surface_graph_layers: int = 4
    surface_transformer_layers: int = 6
    skeleton_graph_layers: int = 4
    token_self_layers: int = 6
    bidirectional_fusion_rounds: int = 4
    ffn_ratio: int = 4
    dropout: float = 0.0
    field_tokens: int = 4
    latent_channels: int = 512
    pair_geometry_dim: int = 10
    edge_feature_dim: int = 4
    view_feature_dim: int = 8
    architecture_id: str = "RealSaS.Arachne.A1.RichQualifiedBidirectional.v4"

    def validate(self) -> None:
        if self.model_dim <= 0 or self.model_dim % self.attention_heads:
            raise ValueError("model_dim/head contract drift")
        vals = (
            self.surface_graph_layers, self.surface_transformer_layers,
            self.skeleton_graph_layers, self.token_self_layers,
            self.bidirectional_fusion_rounds, self.ffn_ratio,
            self.field_tokens, self.latent_channels,
        )
        if min(vals) <= 0:
            raise ValueError("positive architecture cardinality required")
        if self.field_tokens != 4 or self.latent_channels != 512:
            raise ValueError("frozen K4 V7 output interface drift")
        if self.pair_geometry_dim != 10 or self.edge_feature_dim != 4 or self.view_feature_dim != 8:
            raise ValueError("typed relation width drift")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("invalid dropout")

    @property
    def config_hash(self) -> str:
        self.validate(); return _hash(asdict(self))


class FeedForward(nn.Module):
    def __init__(self, dim: int, ratio: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * ratio), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * ratio, dim), nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)


class EdgeMessageBlock(nn.Module):
    def __init__(self, dim: int, edge_dim: int, ratio: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.msg = nn.Sequential(nn.Linear(dim * 2 + edge_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, ratio, dropout)

    def forward(self, x, edge_index, edge_features, edge_mask, node_mask):
        B, N, _ = x.shape
        xn = self.norm(x)
        out = torch.zeros_like(x)
        deg = torch.zeros((B, N, 1), device=x.device, dtype=x.dtype)
        for b in range(B):
            m = edge_mask[b].bool()
            if not bool(m.any()):
                continue
            e = edge_index[b, m]
            ef = edge_features[b, m]
            a, c = e[:, 0].long(), e[:, 1].long()
            if (a < 0).any() or (c < 0).any() or (a >= N).any() or (c >= N).any():
                raise ValueError("edge endpoint outside padded surface")
            ma = self.msg(torch.cat([xn[b, a], xn[b, c], ef], -1))
            mc = self.msg(torch.cat([xn[b, c], xn[b, a], ef], -1))
            out[b].index_add_(0, a, ma); out[b].index_add_(0, c, mc)
            one = torch.ones((len(a), 1), device=x.device, dtype=x.dtype)
            deg[b].index_add_(0, a, one); deg[b].index_add_(0, c, one)
        y = x + out / deg.clamp_min(1.0)
        y = y + self.ff(self.ff_norm(y))
        return y * node_mask[..., None].to(y.dtype)


class TreeMessageBlock(nn.Module):
    def __init__(self, dim: int, ratio: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.child = nn.Linear(dim * 2, dim)
        self.parent = nn.Linear(dim * 2, dim)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, ratio, dropout)

    def forward(self, x, parent_indices, joint_mask):
        B, J, _ = x.shape
        xn = self.norm(x)
        out = torch.zeros_like(x)
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
                out[b, j] += self.child(torch.cat([xn[b, j], xn[b, p]], -1))
                out[b, p] += self.parent(torch.cat([xn[b, p], xn[b, j]], -1))
                deg[b, j] += 1; deg[b, p] += 1
        y = x + out / deg.clamp_min(1.0)
        y = y + self.ff(self.ff_norm(y))
        return y * joint_mask[..., None].to(y.dtype)


class DenseRelationAttention(nn.Module):
    """Manual multi-head attention so pair/anchor bias is explicit and auditable."""
    def __init__(self, dim: int, heads: int, pair_dim: int, ratio: int, dropout: float, field_tokens: int):
        super().__init__()
        if dim % heads: raise ValueError("dim/head mismatch")
        self.dim, self.heads, self.hd, self.K = dim, heads, dim // heads, field_tokens
        self.qn = nn.LayerNorm(dim); self.kn = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim, bias=False); self.k = nn.Linear(dim, dim, bias=False); self.v = nn.Linear(dim, dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)
        self.pair_bias = nn.Sequential(nn.Linear(pair_dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, heads))
        self.anchor_bias = nn.Parameter(torch.full((heads,), 0.25))
        self.drop = nn.Dropout(dropout)
        self.ffn = nn.LayerNorm(dim); self.ff = FeedForward(dim, ratio, dropout)

    def token_reads_surface(self, tokens, surface, pair_geometry, anchor, pair_mask, surface_mask, joint_mask):
        B, J, K, D = tokens.shape; N = surface.shape[1]; Q = J * K
        tq = self.q(self.qn(tokens.reshape(B, Q, D)))
        sk = self.k(self.kn(surface)); sv = self.v(self.kn(surface))
        q = tq.view(B, Q, self.heads, self.hd).transpose(1, 2)
        k = sk.view(B, N, self.heads, self.hd).transpose(1, 2)
        v = sv.view(B, N, self.heads, self.hd).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.hd)
        pb = self.pair_bias(pair_geometry).permute(0, 3, 2, 1)
        pb = pb[:, :, :, None, :].expand(-1, -1, -1, K, -1).reshape(B, self.heads, Q, N)
        ab = anchor[:, :, None, :].expand(-1, -1, K, -1).reshape(B, Q, N)
        scores = scores + pb + ab[:, None].to(scores.dtype) * self.anchor_bias[None, :, None, None]
        legal = pair_mask.permute(0, 2, 1)[:, :, None, :].expand(-1, -1, K, -1).reshape(B, Q, N)
        scores = scores.masked_fill(~legal[:, None].bool(), -1e4)
        valid_q = joint_mask[:, :, None].expand(-1, -1, K).reshape(B, Q)
        scores = scores.masked_fill(~surface_mask[:, None, None, :].bool(), -1e4)
        scores = scores.masked_fill(~valid_q[:, None, :, None], -1e4)
        attn = torch.softmax(scores, -1)
        attn = attn * surface_mask[:, None, None, :].to(attn.dtype) * valid_q[:, None, :, None].to(attn.dtype)
        attn = attn / attn.sum(-1, keepdim=True).clamp_min(1e-8)
        ctx = torch.matmul(self.drop(attn), v).transpose(1, 2).contiguous().view(B, Q, D)
        y = tokens.reshape(B, Q, D) + self.o(ctx)
        y = y + self.ff(self.ffn(y))
        y = y * valid_q[..., None].to(y.dtype)
        return y.reshape(B, J, K, D)

    def surface_reads_tokens(self, surface, tokens, pair_geometry, anchor, pair_mask, surface_mask, joint_mask):
        B, N, D = surface.shape; J, K = tokens.shape[1:3]; Q = J * K
        sq = self.q(self.qn(surface))
        tk = self.k(self.kn(tokens.reshape(B, Q, D))); tv = self.v(self.kn(tokens.reshape(B, Q, D)))
        q = sq.view(B, N, self.heads, self.hd).transpose(1, 2)
        k = tk.view(B, Q, self.heads, self.hd).transpose(1, 2)
        v = tv.view(B, Q, self.heads, self.hd).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.hd)
        pb = self.pair_bias(pair_geometry).permute(0, 3, 1, 2)
        pb = pb[:, :, :, :, None].expand(-1, -1, -1, -1, K).reshape(B, self.heads, N, Q)
        ab = anchor.permute(0, 2, 1)[:, :, :, None].expand(-1, -1, -1, K).reshape(B, N, Q)
        scores = scores + pb + ab[:, None].to(scores.dtype) * self.anchor_bias[None, :, None, None]
        legal = pair_mask[:, :, :, None].expand(-1, -1, -1, K).reshape(B, N, Q)
        scores = scores.masked_fill(~legal[:, None].bool(), -1e4)
        valid_k = joint_mask[:, :, None].expand(-1, -1, K).reshape(B, Q)
        scores = scores.masked_fill(~valid_k[:, None, None, :], -1e4)
        scores = scores.masked_fill(~surface_mask[:, None, :, None].bool(), -1e4)
        attn = torch.softmax(scores, -1)
        attn = attn * valid_k[:, None, None, :].to(attn.dtype) * surface_mask[:, None, :, None].to(attn.dtype)
        attn = attn / attn.sum(-1, keepdim=True).clamp_min(1e-8)
        ctx = torch.matmul(self.drop(attn), v).transpose(1, 2).contiguous().view(B, N, D)
        y = surface + self.o(ctx)
        y = y + self.ff(self.ffn(y))
        return y * surface_mask[..., None].to(y.dtype)


class BidirectionalFusionRound(nn.Module):
    def __init__(self, dim, heads, pair_dim, ratio, dropout, field_tokens):
        super().__init__()
        self.t2s = DenseRelationAttention(dim, heads, pair_dim, ratio, dropout, field_tokens)
        self.s2t = DenseRelationAttention(dim, heads, pair_dim, ratio, dropout, field_tokens)
    def forward(self, tokens, surface, pair_geometry, anchor, pair_mask, surface_mask, joint_mask):
        tokens = self.t2s.token_reads_surface(tokens, surface, pair_geometry, anchor, pair_mask, surface_mask, joint_mask)
        surface = self.s2t.surface_reads_tokens(surface, tokens, pair_geometry, anchor, pair_mask, surface_mask, joint_mask)
        return tokens, surface


@dataclass
class ArachneA1RawOutputV4:
    field_tokens: torch.Tensor
    surface_memory: torch.Tensor
    joint_memory: torch.Tensor


class ArachneA1V4(nn.Module):
    def __init__(self, config: ArachneA1ConfigV4 = ArachneA1ConfigV4()):
        super().__init__(); config.validate(); self.config = config; d = config.model_dim
        self.surface_in = nn.Sequential(nn.Linear(11, d), nn.GELU(), nn.Linear(d, d))
        self.view_in = nn.Sequential(nn.Linear(8, d // 2), nn.GELU(), nn.Linear(d // 2, d))
        self.view_gate = nn.Sequential(nn.Linear(8, d // 4), nn.GELU(), nn.Linear(d // 4, 1))
        self.surface_graph = nn.ModuleList([EdgeMessageBlock(d, config.edge_feature_dim, config.ffn_ratio, config.dropout) for _ in range(config.surface_graph_layers)])
        enc = nn.TransformerEncoderLayer(d, config.attention_heads, d * config.ffn_ratio, config.dropout, activation="gelu", batch_first=True, norm_first=True)
        self.surface_global = nn.TransformerEncoder(enc, config.surface_transformer_layers, norm=nn.LayerNorm(d))
        self.joint_in = nn.Sequential(nn.Linear(10, d), nn.GELU(), nn.Linear(d, d))
        self.tree = nn.ModuleList([TreeMessageBlock(d, config.ffn_ratio, config.dropout) for _ in range(config.skeleton_graph_layers)])
        self.anchor_context = nn.Linear(d, d, bias=False)
        self.token_prototypes = nn.Parameter(torch.randn(config.field_tokens, d) * 0.02)
        tok = nn.TransformerEncoderLayer(d, config.attention_heads, d * config.ffn_ratio, config.dropout, activation="gelu", batch_first=True, norm_first=True)
        self.token_self = nn.TransformerEncoder(tok, config.token_self_layers, norm=nn.LayerNorm(d))
        self.fusion = nn.ModuleList([BidirectionalFusionRound(d, config.attention_heads, config.pair_geometry_dim, config.ffn_ratio, config.dropout, config.field_tokens) for _ in range(config.bidirectional_fusion_rounds)])
        self.out_norm = nn.LayerNorm(d)
        self.to_v7_latent = nn.Linear(d, config.latent_channels)

    def _surface_features(self, positions, normals, normal_valid, support, raster_xy, raster_valid, observed, completed, view_yaw_fourier):
        if view_yaw_fourier.ndim == 2:
            view_yaw_fourier = view_yaw_fourier[None].expand(positions.shape[0], -1, -1)
        if view_yaw_fourier.shape != (positions.shape[0], 8, 4):
            raise ValueError("view_yaw_fourier must be [B,8,4]")
        support_frac = support.to(positions.dtype).mean(-1, keepdim=True)
        raster_frac = raster_valid.to(positions.dtype).mean(-1, keepdim=True)
        base = torch.cat([positions, normals, normal_valid[..., None].to(positions.dtype), observed[..., None].to(positions.dtype), completed[..., None].to(positions.dtype), support_frac, raster_frac], -1)
        if base.shape[-1] != 11: raise AssertionError("surface base width drift")
        ycode = view_yaw_fourier[:, None].expand(-1, positions.shape[1], -1, -1).to(positions.dtype)
        vf = torch.cat([support[..., None].to(positions.dtype), raster_valid[..., None].to(positions.dtype), raster_xy, ycode], -1)
        if vf.shape[-1] != 8: raise AssertionError("view feature width drift")
        ve = self.view_in(vf)
        gate = self.view_gate(vf).squeeze(-1)
        present = (support | raster_valid)
        gate = gate.masked_fill(~present, -1e4)
        w = torch.softmax(gate, dim=2) * present.to(gate.dtype)
        w = w / w.sum(dim=2, keepdim=True).clamp_min(1e-8)
        vctx = (ve * w[..., None]).sum(dim=2)
        return self.surface_in(base) + vctx

    def _joint_features(self, pos, parent_indices, root_mask, deform_root_mask, depth, joint_mask):
        safe = parent_indices.clamp_min(0)
        parent = torch.gather(pos, 1, safe[..., None].expand(-1, -1, 3))
        has = (parent_indices >= 0) & joint_mask.bool()
        delta = (pos - parent) * has[..., None].to(pos.dtype)
        length = torch.linalg.norm(delta, dim=-1, keepdim=True)
        feat = torch.cat([pos, root_mask[..., None].to(pos.dtype), deform_root_mask[..., None].to(pos.dtype), depth[..., None].to(pos.dtype), delta, length], -1)
        return self.joint_in(feat) * joint_mask[..., None].to(pos.dtype)

    def forward(self, *, surface_positions_normalized, surface_normals, surface_normal_valid,
                surface_support_views, surface_raster_xy, surface_raster_valid,
                surface_observed, surface_completed, surface_mask,
                edge_index, edge_features, edge_mask,
                joint_positions_normalized, joint_mask, parent_indices, root_mask,
                deform_root_mask, joint_depth_normalized, support_anchor_matrix,
                pair_geometry, pair_mask, view_yaw_fourier):
        B, N, _ = surface_positions_normalized.shape; J = joint_positions_normalized.shape[1]
        if pair_geometry.shape != (B, N, J, self.config.pair_geometry_dim): raise ValueError("pair geometry shape drift")
        if pair_mask.shape != (B, N, J): raise ValueError("pair mask shape drift")
        if support_anchor_matrix.shape != (B, J, N): raise ValueError("support anchor shape drift")
        legal_pair = pair_mask & surface_mask[:, :, None].bool() & joint_mask[:, None, :].bool()
        if not bool(legal_pair.any()): raise ValueError("no legal surface-joint pairs")
        surface = self._surface_features(surface_positions_normalized, surface_normals, surface_normal_valid,
                                         surface_support_views, surface_raster_xy, surface_raster_valid,
                                         surface_observed, surface_completed, view_yaw_fourier)
        surface = surface * surface_mask[..., None].to(surface.dtype)
        for blk in self.surface_graph: surface = blk(surface, edge_index, edge_features, edge_mask, surface_mask)
        surface = self.surface_global(surface, src_key_padding_mask=~surface_mask.bool())
        surface = surface * surface_mask[..., None].to(surface.dtype)
        joint = self._joint_features(joint_positions_normalized, parent_indices, root_mask, deform_root_mask, joint_depth_normalized, joint_mask)
        for blk in self.tree: joint = blk(joint, parent_indices, joint_mask)
        a = support_anchor_matrix.to(surface.dtype)
        actx = torch.einsum("bjn,bnd->bjd", a, surface) / a.sum(-1, keepdim=True).clamp_min(1.0)
        joint = (joint + self.anchor_context(actx)) * joint_mask[..., None].to(joint.dtype)
        tokens = joint[:, :, None, :] + self.token_prototypes[None, None, :, :]
        valid_tok = joint_mask[:, :, None].expand(-1, -1, self.config.field_tokens)
        flat = tokens.reshape(B, J * self.config.field_tokens, self.config.model_dim)
        flat = self.token_self(flat, src_key_padding_mask=~valid_tok.reshape(B, -1).bool())
        tokens = flat.reshape(B, J, self.config.field_tokens, self.config.model_dim)
        pg = pair_geometry * legal_pair[..., None].to(pair_geometry.dtype)
        for blk in self.fusion:
            tokens, surface = blk(tokens, surface, pg, support_anchor_matrix, legal_pair, surface_mask, joint_mask)
        z = self.to_v7_latent(self.out_norm(tokens)) * valid_tok[..., None].to(tokens.dtype)
        return ArachneA1RawOutputV4(z, surface, joint)

    @property
    def parameter_count(self): return sum(p.numel() for p in self.parameters())


__all__ = ["ArachneA1ConfigV4", "ArachneA1RawOutputV4", "ArachneA1V4"]
