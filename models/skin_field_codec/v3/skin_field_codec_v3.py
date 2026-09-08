from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from .config_v3 import SkinTokensStrengthConfigV3


class FrequencyXYZEmbeddingV3(nn.Module):
    def __init__(self, num_frequencies: int = 8, include_input: bool = True, include_pi: bool = False):
        super().__init__()
        freq = 2.0 ** torch.arange(num_frequencies, dtype=torch.float32)
        if include_pi:
            freq = freq * torch.pi
        self.register_buffer("frequencies", freq, persistent=False)
        self.include_input = bool(include_input)
        self.num_frequencies = int(num_frequencies)

    @property
    def out_dim(self) -> int:
        return 3 * (2 * self.num_frequencies + (1 if self.include_input else 0))

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        if xyz.shape[-1] != 3:
            raise ValueError("xyz must end in 3")
        e = (xyz[..., None] * self.frequencies).reshape(*xyz.shape[:-1], -1)
        parts = [e.sin(), e.cos()]
        if self.include_input:
            parts.insert(0, xyz)
        return torch.cat(parts, dim=-1)


class FeedForwardV3(nn.Module):
    def __init__(self, dim: int, ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = dim * ratio
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SelfAttentionBlockV3(nn.Module):
    def __init__(self, dim: int, heads: int, ratio: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True, bias=False)
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        self.ff = FeedForwardV3(dim, ratio, dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.norm1(x)
        a, _ = self.attn(q, q, q, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + a
        return x + self.ff(self.norm2(x))


class CrossAttentionBlockV3(nn.Module):
    def __init__(self, q_dim: int, kv_dim: int, dim: int, heads: int, ratio: int, dropout: float):
        super().__init__()
        self.q_proj = nn.Linear(q_dim, dim)
        self.kv_proj = nn.Linear(kv_dim, dim)
        self.q_norm = nn.LayerNorm(dim, eps=1e-5)
        self.kv_norm = nn.LayerNorm(dim, eps=1e-5)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True, bias=False)
        self.ff_norm = nn.LayerNorm(dim, eps=1e-5)
        self.ff = FeedForwardV3(dim, ratio, dropout)

    def forward(self, q_input: torch.Tensor, kv_input: torch.Tensor, kv_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.q_proj(q_input)
        kv = self.kv_proj(kv_input)
        qn = self.q_norm(q)
        kvn = self.kv_norm(kv)
        a, _ = self.attn(qn, kvn, kvn, key_padding_mask=kv_padding_mask, need_weights=False)
        x = q + a
        return x + self.ff(self.ff_norm(x))


class LatentSetEncoderV3(nn.Module):
    def __init__(self, query_dim: int, memory_dim: int, width: int, heads: int, layers: int, ratio: int, dropout: float):
        super().__init__()
        self.cross = CrossAttentionBlockV3(query_dim, memory_dim, width, heads, ratio, dropout)
        self.blocks = nn.ModuleList([SelfAttentionBlockV3(width, heads, ratio, dropout) for _ in range(layers)])
        self.out_norm = nn.LayerNorm(width, eps=1e-5)

    def forward(self, q: torch.Tensor, memory: torch.Tensor, memory_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.cross(q, memory, memory_padding_mask)
        for block in self.blocks:
            x = block(x)
        return self.out_norm(x)


class ScalarFieldQueryDecoderV3(nn.Module):
    def __init__(self, query_dim: int, latent_dim: int, width: int, heads: int, layers: int, ratio: int, dropout: float):
        super().__init__()
        self.memory_in = nn.Linear(latent_dim, width)
        self.memory_blocks = nn.ModuleList([SelfAttentionBlockV3(width, heads, ratio, dropout) for _ in range(layers)])
        self.query_cross = CrossAttentionBlockV3(query_dim, width, width, heads, ratio, dropout)
        self.out_norm = nn.LayerNorm(width, eps=1e-5)
        self.out = nn.Linear(width, 1)

    def prepare_memory(self, latent_memory: torch.Tensor) -> torch.Tensor:
        x = self.memory_in(latent_memory)
        for block in self.memory_blocks:
            x = block(x)
        return x

    def query(self, prepared_memory: torch.Tensor, geometry_queries: torch.Tensor) -> torch.Tensor:
        x = self.query_cross(geometry_queries, prepared_memory)
        return torch.sigmoid(self.out(self.out_norm(x))).squeeze(-1)

    def forward(self, latent_memory: torch.Tensor, geometry_queries: torch.Tensor) -> torch.Tensor:
        return self.query(self.prepare_memory(latent_memory), geometry_queries)


class FiniteScalarQuantizerV3(nn.Module):
    def __init__(self, dim: int, levels: tuple[int, ...]):
        super().__init__()
        self.levels_tuple = tuple(int(x) for x in levels)
        levels_t = torch.tensor(self.levels_tuple, dtype=torch.int64)
        basis = torch.cumprod(torch.tensor((1, *self.levels_tuple[:-1]), dtype=torch.int64), dim=0)
        self.register_buffer("levels", levels_t, persistent=False)
        self.register_buffer("basis", basis, persistent=False)
        code_dim = len(self.levels_tuple)
        self.project_in = nn.Linear(dim, code_dim)
        self.project_out = nn.Linear(code_dim, dim)

    @property
    def codebook_size(self) -> int:
        return math.prod(self.levels_tuple)

    def _bound(self, z: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        levels = self.levels.to(device=z.device, dtype=z.dtype)
        half_l = (levels - 1) * (1.0 + eps) / 2.0
        offset = torch.where((self.levels.to(z.device) % 2) == 0, torch.tensor(0.5, device=z.device, dtype=z.dtype), torch.tensor(0.0, device=z.device, dtype=z.dtype))
        shift = torch.atanh(offset / half_l)
        return torch.tanh(z + shift) * half_l - offset

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scalar = self.project_in(z)
        bounded = self._bound(scalar.float()).to(scalar.dtype)
        rounded = bounded + (bounded.round() - bounded).detach()
        half_width = (self.levels.to(device=z.device, dtype=rounded.dtype) // 2).clamp_min(1)
        codes = rounded / half_width
        non_centered = codes * half_width + half_width
        indices = (non_centered.round().to(torch.int64) * self.basis.to(z.device)).sum(dim=-1)
        return self.project_out(codes), indices

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        idx = indices.to(torch.int64)[..., None]
        levels = self.levels.to(indices.device)
        basis = self.basis.to(indices.device)
        level_idx = (idx // basis) % levels
        half_width = (levels // 2).clamp_min(1)
        codes = (level_idx.to(torch.float32) - half_width.to(torch.float32)) / half_width.to(torch.float32)
        return self.project_out(codes.to(self.project_out.weight.dtype))


class SkinFieldCodecV3(nn.Module):
    """SkinTokens-strength RealSaS codec. Joint/pair geometry never enters this decoder."""
    def __init__(self, config: SkinTokensStrengthConfigV3 = SkinTokensStrengthConfigV3()):
        super().__init__()
        config.validate(); self.config = config
        self.xyz_embed = FrequencyXYZEmbeddingV3(config.num_frequencies, config.include_input, config.include_pi)
        self.field_queries = nn.Parameter(torch.randn(config.field_tokens, config.encoder_width) * 0.02)
        self.field_encoder = LatentSetEncoderV3(config.encoder_width, config.field_embed_input_dim, config.encoder_width, config.attention_heads, config.encoder_layers, config.ffn_ratio, config.dropout)
        self.condition_encoder = LatentSetEncoderV3(config.geometry_embed_input_dim, config.geometry_embed_input_dim, config.encoder_width, config.attention_heads, config.encoder_layers, config.ffn_ratio, config.dropout)
        self.field_to_latent = nn.Linear(config.encoder_width, config.latent_channels)
        self.condition_to_latent = nn.Linear(config.encoder_width, config.latent_channels)
        self.fsq = FiniteScalarQuantizerV3(config.latent_channels, config.fsq_levels)
        self.decoder = ScalarFieldQueryDecoderV3(config.geometry_embed_input_dim, config.latent_channels, config.decoder_width, config.attention_heads, config.decoder_layers, config.ffn_ratio, config.dropout)

    def geometry_embedding(self, geometry: torch.Tensor) -> torch.Tensor:
        if geometry.shape[-1] != 7:
            raise ValueError("geometry must be xyz+normal+normal_valid [..,7]")
        return torch.cat([self.xyz_embed(geometry[..., :3]), geometry[..., 3:]], dim=-1)

    def field_embedding(self, field_observations: torch.Tensor) -> torch.Tensor:
        if field_observations.shape[-1] != 8:
            raise ValueError("field observations must be geometry7+scalar weight")
        return torch.cat([self.geometry_embedding(field_observations[..., :7]), field_observations[..., 7:8]], dim=-1)

    def encode_field(self, field_observations: torch.Tensor, memory_padding_mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        memory = self.field_embedding(field_observations)
        q = self.field_queries[None].expand(memory.shape[0], -1, -1)
        h = self.field_encoder(q, memory, memory_padding_mask)
        continuous = self.field_to_latent(h)
        quantized, indices = self.fsq(continuous)
        return continuous, quantized, indices

    def encode_condition(self, geometry: torch.Tensor, query_indices: torch.Tensor, memory_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        memory = self.geometry_embedding(geometry)
        if query_indices.ndim != 2 or query_indices.shape[0] != geometry.shape[0]:
            raise ValueError("query_indices must be [B,K]")
        gather = query_indices[..., None].expand(-1, -1, geometry.shape[-1])
        q_geom = torch.gather(geometry, 1, gather)
        h = self.condition_encoder(self.geometry_embedding(q_geom), memory, memory_padding_mask)
        return self.condition_to_latent(h)

    def decode_field(self, quantized_field_tokens: torch.Tensor, condition_tokens: torch.Tensor, query_geometry: torch.Tensor) -> torch.Tensor:
        if quantized_field_tokens.shape[1] != self.config.field_tokens:
            raise ValueError("field token count drift")
        if condition_tokens.shape[1] != self.config.condition_tokens:
            raise ValueError("condition token count drift")
        memory = torch.cat([quantized_field_tokens, condition_tokens], dim=1)
        return self.decoder(memory, self.geometry_embedding(query_geometry))

    def forward_field(self, field_observations: torch.Tensor, full_geometry: torch.Tensor, condition_query_indices: torch.Tensor, field_memory_padding_mask: Optional[torch.Tensor] = None, geometry_memory_padding_mask: Optional[torch.Tensor] = None):
        continuous, quantized, ids = self.encode_field(field_observations, field_memory_padding_mask)
        cond = self.encode_condition(full_geometry, condition_query_indices, geometry_memory_padding_mask)
        pred = self.decode_field(quantized, cond, full_geometry)
        return pred, continuous, quantized, ids, cond

    @staticmethod
    def normalize_joint_fields(fields: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        if fields.ndim != 3:
            raise ValueError("fields must be [B,N,J]")
        denom = fields.sum(dim=-1, keepdim=True)
        safe = fields / denom.clamp_min(eps)
        if torch.any(denom <= eps):
            fallback = torch.full_like(fields, 1.0 / fields.shape[-1])
            safe = torch.where((denom <= eps).expand_as(fields), fallback, safe)
        return safe


def scalar_field_loss_v3(pred: torch.Tensor, truth: torch.Tensor, supervision_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    if pred.shape != truth.shape:
        raise ValueError("field loss shape mismatch")
    m = supervision_mask.to(dtype=torch.bool)
    if m.shape != pred.shape:
        if m.shape == pred.shape[:-1] and pred.ndim == m.ndim + 1:
            m = m[..., None].expand_as(pred)
        else:
            raise ValueError("field loss mask mismatch")
    p = pred[m].clamp(1e-6, 1 - 1e-6)
    t = truth[m]
    if p.numel() == 0:
        raise ValueError("empty field supervision")
    bce = F.binary_cross_entropy(p, t)
    l1 = (p - t).abs().mean()
    return {"bce": bce, "l1": l1, "total": bce + l1}
