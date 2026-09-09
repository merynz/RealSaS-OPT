from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from torch import nn

from ..v5.skin_field_codec_v5 import (
    ArachneSkinFieldConfigV5,
    CrossAttentionBlockV5,
    SelfAttentionBlockV5,
    SkinFieldCodecV5,
    scalar_field_loss_v5,
)


@dataclass(frozen=True)
class ArachneSkinFieldConfigV6(ArachneSkinFieldConfigV5):
    """V5 sparse/logits contract plus mandatory joint-specific decoder conditioning."""

    field_conditioning_mode: str = "FORCED_BILINEAR_FIELD_CROSS"
    architecture_id: str = "RealSaS.Arachne.SkinFieldCodec.v6"

    def validate(self) -> None:
        super().validate()
        if self.field_conditioning_mode != "FORCED_BILINEAR_FIELD_CROSS":
            raise ValueError("V6 forced field-conditioning contract drift")


class ForcedFieldCrossReadoutV6(nn.Module):
    """No scalar-logit path can bypass joint-specific field tokens.

    Geometry/condition features provide only the query side. Attention values come
    only from field tokens, and the final scalar is bilinear in query and field
    context. Bias-free/non-affine field transforms preserve the invariant that
    zero field tokens produce exactly zero logits even after training.
    """

    def __init__(self, latent_dim: int, width: int, heads: int, dropout: float):
        super().__init__()
        self.field_in = nn.Linear(latent_dim, width, bias=False)
        self.q_norm = nn.LayerNorm(width, eps=1e-5)
        self.field_norm = nn.LayerNorm(width, eps=1e-5, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            width,
            heads,
            dropout=dropout,
            batch_first=True,
            bias=False,
        )
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


class ScalarFieldQueryDecoderV6(nn.Module):
    def __init__(
        self,
        query_dim: int,
        latent_dim: int,
        width: int,
        heads: int,
        layers: int,
        ratio: int,
        dropout: float,
    ):
        super().__init__()
        self.condition_in = nn.Linear(latent_dim, width)
        self.condition_blocks = nn.ModuleList(
            [SelfAttentionBlockV5(width, heads, ratio, dropout) for _ in range(layers)]
        )
        self.query_condition = CrossAttentionBlockV5(
            query_dim,
            width,
            width,
            heads,
            ratio,
            dropout,
        )
        self.field_readout = ForcedFieldCrossReadoutV6(
            latent_dim,
            width,
            heads,
            dropout,
        )

    def prepare_condition(self, condition_tokens: torch.Tensor) -> torch.Tensor:
        x = self.condition_in(condition_tokens)
        for block in self.condition_blocks:
            x = block(x)
        return x

    def query_logits(
        self,
        prepared_condition: torch.Tensor,
        field_tokens: torch.Tensor,
        geometry_queries: torch.Tensor,
    ) -> torch.Tensor:
        query_features = self.query_condition(geometry_queries, prepared_condition)
        return self.field_readout(query_features, field_tokens)

    def forward(
        self,
        field_tokens: torch.Tensor,
        condition_tokens: torch.Tensor,
        geometry_queries: torch.Tensor,
    ) -> torch.Tensor:
        return self.query_logits(
            self.prepare_condition(condition_tokens),
            field_tokens,
            geometry_queries,
        )


class SkinFieldCodecV6(SkinFieldCodecV5):
    def __init__(self, config: ArachneSkinFieldConfigV6 = ArachneSkinFieldConfigV6()):
        config.validate()
        super().__init__(config)
        self.decoder = ScalarFieldQueryDecoderV6(
            config.geometry_embed_input_dim,
            config.latent_channels,
            config.decoder_width,
            config.attention_heads,
            config.decoder_layers,
            config.ffn_ratio,
            config.dropout,
        )

    def decode_field(
        self,
        quantized_field_tokens: torch.Tensor,
        condition_tokens: torch.Tensor,
        query_geometry: torch.Tensor,
        field_token_prefix_len: int | None = None,
    ) -> torch.Tensor:
        if quantized_field_tokens.shape[1] != self.config.field_tokens:
            raise ValueError("field token count drift")
        if condition_tokens.shape[1] != self.config.condition_tokens:
            raise ValueError("condition token count drift")
        prefix = self.config.field_tokens if field_token_prefix_len is None else int(field_token_prefix_len)
        if not (self.config.nested_dropout_min_prefix <= prefix <= self.config.field_tokens):
            raise ValueError("nested field-token prefix out of range")
        field_tokens = quantized_field_tokens[:, :prefix]
        return self.decoder(field_tokens, condition_tokens, self.geometry_embedding(query_geometry))


def scalar_field_loss_v6(*args, **kwargs):
    return scalar_field_loss_v5(*args, **kwargs)


__all__ = [
    "ArachneSkinFieldConfigV6",
    "ForcedFieldCrossReadoutV6",
    "ScalarFieldQueryDecoderV6",
    "SkinFieldCodecV6",
    "scalar_field_loss_v6",
]
