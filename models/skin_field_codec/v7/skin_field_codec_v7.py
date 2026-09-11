# Arachne Skin Field V7 — continuous field-token transport

import json
import math
import torch
from dataclasses import asdict, dataclass
from hashlib import sha256 as _sha256
from typing import Optional
from torch import nn
import torch.nn.functional as F


def _hash(payload: object) -> str:
    return _sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


@dataclass(frozen=True)
class ArachneSkinFieldConfigV7:
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

    # Sparse-field reconstruction contract.
    loss_bce_weight: float = 1.0
    loss_mse_weight: float = 0.1
    loss_dice_weight: float = 1.0
    dice_epsilon: float = 1e-4

    # Decoder importance sampling: half global/supervised, half active-field support.
    decoder_query_count: int = 384
    decoder_dense_fraction: float = 0.5
    active_weight_epsilon: float = 1e-8

    # SkinTokens-style nested prefix dropout over the four field tokens.
    nested_dropout_min_prefix: int = 1

    # Scalar logits have no field-independent bypass.
    field_conditioning_mode: str = "FORCED_BILINEAR_FIELD_CROSS"
    field_token_transport: str = "CONTINUOUS_NO_QUANTIZER"

    architecture_id: str = "RealSaS.Arachne.SkinFieldCodec.v7"
    strict_contract: bool = True

    def validate(self) -> None:
        if self.geometry_dim != 7 or self.field_observation_dim != 8:
            raise ValueError("V7 input contract drift")
        if min(
            self.field_tokens, self.condition_tokens, self.latent_channels,
            self.encoder_width, self.decoder_width, self.attention_heads,
            self.encoder_layers, self.decoder_layers, self.ffn_ratio,
            self.decoder_query_count, self.nested_dropout_min_prefix,
        ) <= 0:
            raise ValueError("V7 positive architecture cardinality required")
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
        if self.field_conditioning_mode != "FORCED_BILINEAR_FIELD_CROSS":
            raise ValueError("V7 forced field-conditioning contract drift")
        if self.field_token_transport != "CONTINUOUS_NO_QUANTIZER":
            raise ValueError("V7 continuous field-token transport contract drift")

        if self.strict_contract:
            if self.field_tokens != 4:
                raise ValueError("V7 requires four field tokens per joint")
            if self.condition_tokens != 384 or self.latent_channels != 512:
                raise ValueError("V7 token/latent contract drift")
            if (self.encoder_width, self.decoder_width) != (512, 1024):
                raise ValueError("V7 width contract drift")
            if (self.encoder_layers, self.decoder_layers, self.attention_heads) != (8, 16, 8):
                raise ValueError("V7 attention depth/head contract drift")
            if self.ffn_ratio != 4 or self.dropout != 0.0:
                raise ValueError("V7 transformer contract drift")
            if self.num_frequencies != 8 or not self.include_input or self.include_pi:
                raise ValueError("V7 frequency embedding contract drift")
            if self.decoder_query_count != 384:
                raise ValueError("V7 decoder query count drift")
            if self.decoder_dense_fraction != 0.5:
                raise ValueError("V7 importance sampling ratio drift")
            if (self.loss_bce_weight, self.loss_mse_weight, self.loss_dice_weight) != (1.0, 0.1, 1.0):
                raise ValueError("V7 sparse loss weighting drift")
            if self.dice_epsilon != 1e-4:
                raise ValueError("V7 Dice epsilon drift")
            if self.nested_dropout_min_prefix != 1:
                raise ValueError("V7 nested dropout contract drift")

    @property
    def xyz_frequency_dim(self) -> int:
        return 3 * (2 * self.num_frequencies + (1 if self.include_input else 0))

    @property
    def geometry_embed_input_dim(self) -> int:
        return self.xyz_frequency_dim + 4

    @property
    def field_embed_input_dim(self) -> int:
        return self.xyz_frequency_dim + 5

    @property
    def config_hash(self) -> str:
        self.validate()
        return _hash(asdict(self))


class FrequencyXYZEmbeddingV7(nn.Module):
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

class FeedForwardV7(nn.Module):
    def __init__(self, dim: int, ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = dim * ratio
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class SelfAttentionBlockV7(nn.Module):
    def __init__(self, dim: int, heads: int, ratio: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True, bias=False)
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        self.ff = FeedForwardV7(dim, ratio, dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.norm1(x)
        a, _ = self.attn(q, q, q, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + a
        return x + self.ff(self.norm2(x))

class CrossAttentionBlockV7(nn.Module):
    def __init__(self, q_dim: int, kv_dim: int, dim: int, heads: int, ratio: int, dropout: float):
        super().__init__()
        self.q_proj = nn.Linear(q_dim, dim)
        self.kv_proj = nn.Linear(kv_dim, dim)
        self.q_norm = nn.LayerNorm(dim, eps=1e-5)
        self.kv_norm = nn.LayerNorm(dim, eps=1e-5)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True, bias=False)
        self.ff_norm = nn.LayerNorm(dim, eps=1e-5)
        self.ff = FeedForwardV7(dim, ratio, dropout)

    def forward(self, q_input: torch.Tensor, kv_input: torch.Tensor, kv_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.q_proj(q_input)
        kv = self.kv_proj(kv_input)
        qn = self.q_norm(q)
        kvn = self.kv_norm(kv)
        a, _ = self.attn(qn, kvn, kvn, key_padding_mask=kv_padding_mask, need_weights=False)
        x = q + a
        return x + self.ff(self.ff_norm(x))

class LatentSetEncoderV7(nn.Module):
    def __init__(self, query_dim: int, memory_dim: int, width: int, heads: int, layers: int, ratio: int, dropout: float):
        super().__init__()
        self.cross = CrossAttentionBlockV7(query_dim, memory_dim, width, heads, ratio, dropout)
        self.blocks = nn.ModuleList([SelfAttentionBlockV7(width, heads, ratio, dropout) for _ in range(layers)])
        self.out_norm = nn.LayerNorm(width, eps=1e-5)

    def forward(self, q: torch.Tensor, memory: torch.Tensor, memory_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.cross(q, memory, memory_padding_mask)
        for block in self.blocks:
            x = block(x)
        return self.out_norm(x)

class ForcedFieldCrossReadoutV7(nn.Module):
    """Joint-specific field memory is structurally mandatory for every scalar logit."""
    def __init__(self, latent_dim: int, width: int, heads: int, dropout: float):
        super().__init__()
        self.field_in = nn.Linear(latent_dim, width, bias=False)
        self.q_norm = nn.LayerNorm(width, eps=1e-5)
        self.field_norm = nn.LayerNorm(width, eps=1e-5, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True, bias=False)
        self.field_out_norm = nn.LayerNorm(width, eps=1e-5, elementwise_affine=False)
        self.query_out_norm = nn.LayerNorm(width, eps=1e-5)
        self.field_bias = nn.Linear(width, 1, bias=False)
        self.scale = float(width) ** -0.5
    def forward(self, query_features: torch.Tensor, field_tokens: torch.Tensor) -> torch.Tensor:
        field_memory = self.field_in(field_tokens)
        qn = self.q_norm(query_features)
        fn = self.field_norm(field_memory)
        field_ctx, _ = self.attn(qn, fn, fn, need_weights=False)
        qv = self.query_out_norm(query_features)
        fv = self.field_out_norm(field_ctx)
        bilinear = (qv * fv).sum(dim=-1) * self.scale
        return bilinear + self.field_bias(fv).squeeze(-1)

class ScalarFieldQueryDecoderV7(nn.Module):
    def __init__(self, query_dim: int, latent_dim: int, width: int, heads: int, layers: int, ratio: int, dropout: float):
        super().__init__()
        self.condition_in = nn.Linear(latent_dim, width)
        self.condition_blocks = nn.ModuleList([SelfAttentionBlockV7(width, heads, ratio, dropout) for _ in range(layers)])
        self.query_condition = CrossAttentionBlockV7(query_dim, width, width, heads, ratio, dropout)
        self.field_readout = ForcedFieldCrossReadoutV7(latent_dim, width, heads, dropout)
    def prepare_condition(self, condition_tokens: torch.Tensor) -> torch.Tensor:
        x = self.condition_in(condition_tokens)
        for block in self.condition_blocks: x = block(x)
        return x
    def query_logits(self, prepared_condition: torch.Tensor, field_tokens: torch.Tensor, geometry_queries: torch.Tensor) -> torch.Tensor:
        query_features = self.query_condition(geometry_queries, prepared_condition)
        return self.field_readout(query_features, field_tokens)
    def forward(self, field_tokens: torch.Tensor, condition_tokens: torch.Tensor, geometry_queries: torch.Tensor) -> torch.Tensor:
        return self.query_logits(self.prepare_condition(condition_tokens), field_tokens, geometry_queries)

class SkinFieldCodecV7(nn.Module):
    """Arachne sparse skin-field codec. Joint/pair geometry never enters this decoder."""
    def __init__(self, config: ArachneSkinFieldConfigV7 = ArachneSkinFieldConfigV7()):
        super().__init__()
        config.validate()
        self.config = config
        self.xyz_embed = FrequencyXYZEmbeddingV7(config.num_frequencies, config.include_input, config.include_pi)
        self.field_queries = nn.Parameter(torch.randn(config.field_tokens, config.encoder_width) * 0.02)
        self.field_encoder = LatentSetEncoderV7(
            config.encoder_width, config.field_embed_input_dim, config.encoder_width,
            config.attention_heads, config.encoder_layers, config.ffn_ratio, config.dropout
        )
        self.condition_encoder = LatentSetEncoderV7(
            config.geometry_embed_input_dim, config.geometry_embed_input_dim, config.encoder_width,
            config.attention_heads, config.encoder_layers, config.ffn_ratio, config.dropout
        )
        self.field_to_latent = nn.Linear(config.encoder_width, config.latent_channels)
        self.condition_to_latent = nn.Linear(config.encoder_width, config.latent_channels)
        self.decoder = ScalarFieldQueryDecoderV7(
            config.geometry_embed_input_dim, config.latent_channels, config.decoder_width,
            config.attention_heads, config.decoder_layers, config.ffn_ratio, config.dropout
        )

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
        return continuous

    def encode_condition(self, geometry: torch.Tensor, query_indices: torch.Tensor, memory_padding_mask: Optional[torch.Tensor] = None):
        memory = self.geometry_embedding(geometry)
        if query_indices.ndim != 2 or query_indices.shape[0] != geometry.shape[0]:
            raise ValueError("query_indices must be [B,K]")
        gather = query_indices[..., None].expand(-1, -1, geometry.shape[-1])
        q_geom = torch.gather(geometry, 1, gather)
        h = self.condition_encoder(self.geometry_embedding(q_geom), memory, memory_padding_mask)
        return self.condition_to_latent(h)

    def decode_field(
        self,
        continuous_field_tokens: torch.Tensor,
        condition_tokens: torch.Tensor,
        query_geometry: torch.Tensor,
        field_token_prefix_len: int | None = None,
    ):
        if continuous_field_tokens.shape[1] != self.config.field_tokens:
            raise ValueError("field token count drift")
        if condition_tokens.shape[1] != self.config.condition_tokens:
            raise ValueError("condition token count drift")
        prefix = self.config.field_tokens if field_token_prefix_len is None else int(field_token_prefix_len)
        if not (self.config.nested_dropout_min_prefix <= prefix <= self.config.field_tokens):
            raise ValueError("nested field-token prefix out of range")
        field_tokens = continuous_field_tokens[:, :prefix]
        return self.decoder(field_tokens, condition_tokens, self.geometry_embedding(query_geometry))

    def forward_field(
        self,
        field_observations: torch.Tensor,
        full_geometry: torch.Tensor,
        condition_query_indices: torch.Tensor,
        query_geometry: torch.Tensor | None = None,
        field_token_prefix_len: int | None = None,
        field_memory_padding_mask: Optional[torch.Tensor] = None,
        geometry_memory_padding_mask: Optional[torch.Tensor] = None,
    ):
        continuous = self.encode_field(field_observations, field_memory_padding_mask)
        cond = self.encode_condition(full_geometry, condition_query_indices, geometry_memory_padding_mask)
        query_geometry = full_geometry if query_geometry is None else query_geometry
        pred = self.decode_field(
            continuous,
            cond,
            query_geometry,
            field_token_prefix_len=field_token_prefix_len,
        )
        return pred, continuous, cond

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


def scalar_field_loss_v7(
    logits: torch.Tensor,
    truth: torch.Tensor,
    supervision_mask: torch.Tensor | None = None,
    *,
    bce_weight: float = 1.0,
    mse_weight: float = 0.1,
    dice_weight: float = 1.0,
    dice_epsilon: float = 1e-4,
) -> dict[str, torch.Tensor]:
    """Logits-first sparse reconstruction: BCEWithLogits + probability-space MSE/Dice."""
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

    # The whole objective is FP32. BCE stays in logits space, so saturation never
    # creates a zero-gradient clamp dead-zone.
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



__all__ = ["ArachneSkinFieldConfigV7", "SkinFieldCodecV7", "scalar_field_loss_v7"]
