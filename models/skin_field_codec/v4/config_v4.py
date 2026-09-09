from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math


def _hash(payload: object) -> str:
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


@dataclass(frozen=True)
class ArachneSkinFieldConfigV4:
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

    architecture_id: str = "RealSaS.Arachne.SkinFieldCodec.v4"
    strict_contract: bool = True

    def validate(self) -> None:
        if self.geometry_dim != 7 or self.field_observation_dim != 8:
            raise ValueError("V4 input contract drift")
        if min(
            self.field_tokens, self.condition_tokens, self.latent_channels,
            self.encoder_width, self.decoder_width, self.attention_heads,
            self.encoder_layers, self.decoder_layers, self.ffn_ratio,
            self.decoder_query_count, self.nested_dropout_min_prefix,
        ) <= 0:
            raise ValueError("V4 positive architecture cardinality required")
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
                raise ValueError("V4 requires four field tokens per joint")
            if self.condition_tokens != 384 or self.latent_channels != 512:
                raise ValueError("V4 token/latent contract drift")
            if (self.encoder_width, self.decoder_width) != (512, 1024):
                raise ValueError("V4 width contract drift")
            if (self.encoder_layers, self.decoder_layers, self.attention_heads) != (8, 16, 8):
                raise ValueError("V4 attention depth/head contract drift")
            if self.ffn_ratio != 4 or self.dropout != 0.0:
                raise ValueError("V4 transformer contract drift")
            if self.num_frequencies != 8 or not self.include_input or self.include_pi:
                raise ValueError("V4 frequency embedding contract drift")
            if tuple(self.fsq_levels) != (8, 8, 8, 5, 5, 5):
                raise ValueError("V4 FSQ level contract drift")
            if math.prod(self.fsq_levels) != 64000:
                raise ValueError("V4 FSQ codebook cardinality drift")
            if self.decoder_query_count != 384:
                raise ValueError("V4 decoder query count drift")
            if self.decoder_dense_fraction != 0.5:
                raise ValueError("V4 importance sampling ratio drift")
            if (self.loss_bce_weight, self.loss_mse_weight, self.loss_dice_weight) != (1.0, 0.1, 1.0):
                raise ValueError("V4 sparse loss weighting drift")
            if self.dice_epsilon != 1e-4:
                raise ValueError("V4 Dice epsilon drift")
            if self.nested_dropout_min_prefix != 1:
                raise ValueError("V4 nested dropout contract drift")

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
    def fsq_codebook_size(self) -> int:
        return math.prod(self.fsq_levels)

    @property
    def config_hash(self) -> str:
        self.validate()
        return _hash(asdict(self))
