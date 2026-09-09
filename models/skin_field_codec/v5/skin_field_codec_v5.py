from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import torch
from hashlib import sha256 as _sha256
from typing import Optional
from torch import nn
import torch.nn.functional as F


def _hash(payload: object) -> str:
    return _sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


@dataclass(frozen=True)
class ArachneSkinFieldConfigV5:
    geometry_dim: int = 7
    field_observation_dim: int = 8
    num_frequencies: int = 8
    include_input: bool = True
    include_pi: bool = False

    field_tokens: int = 4
    condition_tokens: int = 384
    latent_channels: int = 512
    encoder_width: int = 512
    decoder_width: int = 1024
    attention_heads: int = 8
    encoder_layers: int = 8
    decoder_layers: int = 16
    ffn_ratio: int = 4
    dropout: float = 0.0
    fsq_levels: tuple[int, ...] = (8, 8, 8, 5, 5, 5)

    loss_bce_weight: float = 1.0
    loss_mse_weight: float = 0.1
    loss_dice_weight: float = 1.0
    dice_epsilon: float = 1e-4
    decoder_query_count: int = 384
    decoder_dense_fraction: float = 0.5
    active_weight_epsilon: float = 1e-8
    nested_dropout_min_prefix: int = 1
    architecture_id: str = "RealSaS.Arachne.SkinFieldCodec.v5"
    strict_contract: bool = True

    def validate(self) -> None:
        if self.geometry_dim != 7 or self.field_observation_dim != 8:
            raise ValueError("V5 input contract drift")
        if min(self.field_tokens, self.condition_tokens, self.latent_channels, self.encoder_width, self.decoder_width, self.attention_heads, self.encoder_layers, self.decoder_layers, self.ffn_ratio, self.decoder_query_count, self.nested_dropout_min_prefix) <= 0:
            raise ValueError("V5 positive architecture cardinality required")
        if self.encoder_width % self.attention_heads or self.decoder_width % self.attention_heads:
            raise ValueError("attention head divisibility drift")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("invalid dropout")
        if not (0.0 < self.decoder_dense_fraction < 1.0):
            raise ValueError("decoder_dense_fraction must be in (0,1)")
        if not (1 <= self.nested_dropout_min_prefix <= self.field_tokens):
            raise ValueError("nested dropout prefix contract drift")
        if min(self.loss_bce_weight, self.loss_mse_weight, self.loss_dice_weight) < 0:
            raise ValueError("loss weights must be non-negative")
        if self.loss_bce_weight + self.loss_mse_weight + self.loss_dice_weight <= 0:
            raise ValueError("at least one reconstruction loss must be enabled")
        if self.dice_epsilon <= 0 or self.active_weight_epsilon < 0:
            raise ValueError("invalid numerical epsilon")
        if self.strict_contract:
            if self.field_tokens != 4:
                raise ValueError("V5 requires four field tokens per joint")
            if self.condition_tokens != 384 or self.latent_channels != 512:
                raise ValueError("V5 token/latent contract drift")
            if (self.encoder_width, self.decoder_width) != (512, 1024):
                raise ValueError("V5 width contract drift")
            if (self.encoder_layers, self.decoder_layers, self.attention_heads) != (8, 16, 8):
                raise ValueError("V5 attention depth/head contract drift")
            if self.ffn_ratio != 4 or self.dropout != 0.0:
                raise ValueError("V5 transformer contract drift")
            if self.num_frequencies != 8 or not self.include_input or self.include_pi:
                raise ValueError("V5 frequency embedding contract drift")
            if tuple(self.fsq_levels) != (8, 8, 8, 5, 5, 5):
                raise ValueError("V5 FSQ level contract drift")
            if math.prod(self.fsq_levels) != 64000:
                raise ValueError("V5 FSQ codebook cardinality drift")
            if self.decoder_query_count != 384 or self.decoder_dense_fraction != 0.5:
                raise ValueError("V5 sparse sampling contract drift")
            if (self.loss_bce_weight, self.loss_mse_weight, self.loss_dice_weight) != (1.0, 0.1, 1.0):
                raise ValueError("V5 sparse loss weighting drift")
            if self.dice_epsilon != 1e-4 or self.nested_dropout_min_prefix != 1:
                raise ValueError("V5 numerical/nested-dropout contract drift")

    @property
    def xyz_frequency_dim(self) -> int:
        return 3 * (2 * self.num_frequencies + (1 if self.include_input else 0))

    @property
    def geometry_embed_input_dim(self) -> int:
        return self.xyz_frequency_dim + 4

    @property
    def field_embed_input_dim(self) -> int:
        return self.geometry_embed_input_dim + 1

    @property
    def fsq_codebook_size(self) -> int:
        return math.prod(self.fsq_levels)

    @property
    def config_hash(self) -> str:
        return _hash(asdict(self))


class FrequencyXYZEmbeddingV5(nn.Module):
    def __init__(self, num_frequencies: int, include_input: bool, include_pi: bool):
        super().__init__()
        self.num_frequencies = int(num_frequencies)
        self.include_input = bool(include_input)
        self.include_pi = bool(include_pi)
        self.register_buffer("freqs", 2.0 ** torch.arange(self.num_frequencies, dtype=torch.float32), persistent=False)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        parts = [xyz] if self.include_input else []
        scale = math.pi if self.include_pi else 1.0
        angles = xyz[..., None] * self.freqs.to(device=xyz.device, dtype=xyz.dtype) * scale
        parts += [torch.sin(angles).flatten(-2), torch.cos(angles).flatten(-2)]
        return torch.cat(parts, dim=-1)


class CrossAttentionBlockV5(nn.Module):
    def __init__(self, q_dim: int, kv_dim: int, width: int, heads: int, ratio: int, dropout: float):
        super().__init__()
        self.q_in = nn.Linear(q_dim, width)
        self.kv_in = nn.Linear(kv_dim, width)
        self.q_norm = nn.LayerNorm(width, eps=1e-5)
        self.kv_norm = nn.LayerNorm(width, eps=1e-5)
        self.attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.ff_norm = nn.LayerNorm(width, eps=1e-5)
        self.ff = nn.Sequential(nn.Linear(width, width * ratio), nn.GELU(), nn.Dropout(dropout), nn.Linear(width * ratio, width))
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.q_in(q)
        m = self.kv_in(kv)
        y, _ = self.attn(self.q_norm(x), self.kv_norm(m), self.kv_norm(m), key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.dropout(y)
        return x + self.dropout(self.ff(self.ff_norm(x)))


class SelfAttentionBlockV5(nn.Module):
    def __init__(self, width: int, heads: int, ratio: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, eps=1e-5)
        self.attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(width, eps=1e-5)
        self.ff = nn.Sequential(nn.Linear(width, width * ratio), nn.GELU(), nn.Dropout(dropout), nn.Linear(width * ratio, width))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.norm1(x)
        y, _ = self.attn(n, n, n, need_weights=False)
        x = x + self.dropout(y)
        return x + self.dropout(self.ff(self.norm2(x)))


class LatentSetEncoderV5(nn.Module):
    def __init__(self, query_dim: int, memory_dim: int, width: int, heads: int, layers: int, ratio: int, dropout: float):
        super().__init__()
        self.cross = CrossAttentionBlockV5(query_dim, memory_dim, width, heads, ratio, dropout)
        self.blocks = nn.ModuleList([SelfAttentionBlockV5(width, heads, ratio, dropout) for _ in range(layers)])
        self.out_norm = nn.LayerNorm(width, eps=1e-5)

    def forward(self, queries: torch.Tensor, memory: torch.Tensor, memory_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.cross(queries, memory, memory_padding_mask)
        for block in self.blocks:
            x = block(x)
        return self.out_norm(x)


class ScalarFieldQueryDecoderV5(nn.Module):
    def __init__(self, query_dim: int, latent_dim: int, width: int, heads: int, layers: int, ratio: int, dropout: float):
        super().__init__()
        self.memory_in = nn.Linear(latent_dim, width)
        self.memory_blocks = nn.ModuleList([SelfAttentionBlockV5(width, heads, ratio, dropout) for _ in range(layers)])
        self.query_cross = CrossAttentionBlockV5(query_dim, width, width, heads, ratio, dropout)
        self.out_norm = nn.LayerNorm(width, eps=1e-5)
        self.out = nn.Linear(width, 1)

    def prepare_memory(self, latent_memory: torch.Tensor) -> torch.Tensor:
        x = self.memory_in(latent_memory)
        for block in self.memory_blocks:
            x = block(x)
        return x

    def query_logits(self, prepared_memory: torch.Tensor, geometry_queries: torch.Tensor) -> torch.Tensor:
        x = self.query_cross(geometry_queries, prepared_memory)
        return self.out(self.out_norm(x)).squeeze(-1)

    def query(self, prepared_memory: torch.Tensor, geometry_queries: torch.Tensor) -> torch.Tensor:
        return self.query_logits(prepared_memory, geometry_queries)

    def forward(self, latent_memory: torch.Tensor, geometry_queries: torch.Tensor) -> torch.Tensor:
        return self.query_logits(self.prepare_memory(latent_memory), geometry_queries)


class FiniteScalarQuantizerV5(nn.Module):
    def __init__(self, dim: int, levels: tuple[int, ...]):
        super().__init__()
        self.dim = int(dim)
        self.levels_tuple = tuple(int(x) for x in levels)
        code_dim = len(self.levels_tuple)
        self.project_in = nn.Linear(dim, code_dim)
        self.project_out = nn.Linear(code_dim, dim)
        levels_t = torch.tensor(self.levels_tuple, dtype=torch.int64)
        basis = torch.cumprod(torch.cat([torch.ones(1, dtype=torch.int64), levels_t[:-1]]), dim=0)
        self.register_buffer("levels", levels_t, persistent=False)
        self.register_buffer("basis", basis, persistent=False)

    def _bound(self, z: torch.Tensor) -> torch.Tensor:
        half_l = (self.levels - 1) * (1.0 + 1e-3) / 2.0
        offset = torch.where(self.levels % 2 == 0, 0.5, 0.0)
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


class SkinFieldCodecV5(nn.Module):
    def __init__(self, config: ArachneSkinFieldConfigV5 = ArachneSkinFieldConfigV5()):
        super().__init__()
        config.validate()
        self.config = config
        self.xyz_embed = FrequencyXYZEmbeddingV5(config.num_frequencies, config.include_input, config.include_pi)
        self.field_queries = nn.Parameter(torch.randn(config.field_tokens, config.encoder_width) * 0.02)
        self.field_encoder = LatentSetEncoderV5(config.encoder_width, config.field_embed_input_dim, config.encoder_width, config.attention_heads, config.encoder_layers, config.ffn_ratio, config.dropout)
        self.condition_encoder = LatentSetEncoderV5(config.geometry_embed_input_dim, config.geometry_embed_input_dim, config.encoder_width, config.attention_heads, config.encoder_layers, config.ffn_ratio, config.dropout)
        self.field_to_latent = nn.Linear(config.encoder_width, config.latent_channels)
        self.condition_to_latent = nn.Linear(config.encoder_width, config.latent_channels)
        self.fsq = FiniteScalarQuantizerV5(config.latent_channels, config.fsq_levels)
        self.decoder = ScalarFieldQueryDecoderV5(config.geometry_embed_input_dim, config.latent_channels, config.decoder_width, config.attention_heads, config.decoder_layers, config.ffn_ratio, config.dropout)

    def geometry_embedding(self, geometry: torch.Tensor) -> torch.Tensor:
        if geometry.shape[-1] != 7:
            raise ValueError("geometry must be xyz+normal+normal_valid [..,7]")
        return torch.cat([self.xyz_embed(geometry[..., :3]), geometry[..., 3:]], dim=-1)

    def field_embedding(self, field_observations: torch.Tensor) -> torch.Tensor:
        if field_observations.shape[-1] != 8:
            raise ValueError("field observations must be geometry7+scalar weight")
        return torch.cat([self.geometry_embedding(field_observations[..., :7]), field_observations[..., 7:8]], dim=-1)

    def encode_field(self, field_observations: torch.Tensor, memory_padding_mask: Optional[torch.Tensor] = None):
        memory = self.field_embedding(field_observations)
        q = self.field_queries[None].expand(memory.shape[0], -1, -1)
        h = self.field_encoder(q, memory, memory_padding_mask)
        continuous = self.field_to_latent(h)
        quantized, indices = self.fsq(continuous)
        return continuous, quantized, indices

    def encode_condition(self, geometry: torch.Tensor, query_indices: torch.Tensor, memory_padding_mask: Optional[torch.Tensor] = None):
        memory = self.geometry_embedding(geometry)
        if query_indices.ndim != 2 or query_indices.shape[0] != geometry.shape[0]:
            raise ValueError("query_indices must be [B,K]")
        gather = query_indices[..., None].expand(-1, -1, geometry.shape[-1])
        q_geom = torch.gather(geometry, 1, gather)
        h = self.condition_encoder(self.geometry_embedding(q_geom), memory, memory_padding_mask)
        return self.condition_to_latent(h)

    def decode_field(self, quantized_field_tokens: torch.Tensor, condition_tokens: torch.Tensor, query_geometry: torch.Tensor, field_token_prefix_len: int | None = None):
        if quantized_field_tokens.shape[1] != self.config.field_tokens:
            raise ValueError("field token count drift")
        if condition_tokens.shape[1] != self.config.condition_tokens:
            raise ValueError("condition token count drift")
        prefix = self.config.field_tokens if field_token_prefix_len is None else int(field_token_prefix_len)
        if not (self.config.nested_dropout_min_prefix <= prefix <= self.config.field_tokens):
            raise ValueError("nested field-token prefix out of range")
        memory = torch.cat([quantized_field_tokens[:, :prefix], condition_tokens], dim=1)
        return self.decoder(memory, self.geometry_embedding(query_geometry))

    def forward_field(self, field_observations: torch.Tensor, full_geometry: torch.Tensor, condition_query_indices: torch.Tensor, query_geometry: torch.Tensor | None = None, field_token_prefix_len: int | None = None, field_memory_padding_mask: Optional[torch.Tensor] = None, geometry_memory_padding_mask: Optional[torch.Tensor] = None):
        continuous, quantized, ids = self.encode_field(field_observations, field_memory_padding_mask)
        cond = self.encode_condition(full_geometry, condition_query_indices, geometry_memory_padding_mask)
        query_geometry = full_geometry if query_geometry is None else query_geometry
        logits = self.decode_field(quantized, cond, query_geometry, field_token_prefix_len=field_token_prefix_len)
        return logits, continuous, quantized, ids, cond

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


def scalar_field_loss_v5(logits: torch.Tensor, truth: torch.Tensor, supervision_mask: torch.Tensor | None = None, *, bce_weight: float = 1.0, mse_weight: float = 0.1, dice_weight: float = 1.0, dice_epsilon: float = 1e-4) -> dict[str, torch.Tensor]:
    if logits.shape != truth.shape:
        raise ValueError("field loss shape mismatch")
    if supervision_mask is None:
        m = torch.ones_like(logits, dtype=torch.bool)
    else:
        m = supervision_mask.to(dtype=torch.bool)
        if m.shape != logits.shape:
            if m.shape == logits.shape[:-1] and logits.ndim == m.ndim + 1:
                m = m[..., None].expand_as(logits)
            else:
                raise ValueError("field loss mask mismatch")
    with torch.autocast(device_type=logits.device.type, enabled=False):
        z_all = logits.float()
        t_all = truth.float()
        z = z_all[m]
        t = t_all[m]
        if z.numel() == 0:
            raise ValueError("empty field supervision")
        bce = F.binary_cross_entropy_with_logits(z, t)
        p_all = torch.sigmoid(z_all)
        p = p_all[m]
        mse = F.mse_loss(p, t)
        pm = torch.where(m, p_all, torch.zeros_like(p_all))
        tm = torch.where(m, t_all, torch.zeros_like(t_all))
        reduce_dims = tuple(range(1, pm.ndim))
        numerator = 2.0 * (pm * tm).sum(dim=reduce_dims) + float(dice_epsilon)
        denominator = pm.square().sum(dim=reduce_dims) + tm.square().sum(dim=reduce_dims) + float(dice_epsilon)
        dice = (1.0 - numerator / denominator).mean()
        l1 = (p - t).abs().mean()
        total = float(bce_weight) * bce + float(mse_weight) * mse + float(dice_weight) * dice
    return {"bce": bce, "mse": mse, "dice": dice, "l1": l1, "total": total}


__all__ = ["ArachneSkinFieldConfigV5", "SkinFieldCodecV5", "scalar_field_loss_v5"]
