from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json


def _hash(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return sha256(raw).hexdigest()


@dataclass(frozen=True)
class GeppettoCandidateConfigV1:
    surface_feature_dim: int = 20
    model_dim: int = 256
    encoder_layers: int = 4
    decoder_layers: int = 4
    attention_heads: int = 8
    feedforward_dim: int = 768
    max_joint_queries: int = 64
    support_topk: int = 8
    dropout: float = 0.0
    position_scale: float = 1.25
    architecture_id: str = "RealSaS.GeppettoCandidate.SetProposalTransformer.v1"

    def validate(self) -> None:
        if self.surface_feature_dim != 20:
            raise ValueError("Geppetto V1 conditioner feature width must remain 20")
        if self.model_dim <= 0 or self.model_dim % self.attention_heads != 0:
            raise ValueError("model_dim must be positive and divisible by attention_heads")
        if min(self.encoder_layers, self.decoder_layers, self.attention_heads, self.feedforward_dim, self.max_joint_queries) <= 0:
            raise ValueError("all Geppetto architecture cardinalities must be positive")
        if self.support_topk <= 0 or self.support_topk > self.max_joint_queries:
            raise ValueError("support_topk outside valid range")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout outside [0,1)")
        if self.position_scale <= 0.0:
            raise ValueError("position_scale must be positive")

    @property
    def config_hash(self) -> str:
        self.validate()
        return _hash(asdict(self))


@dataclass(frozen=True)
class SkinFieldCodecConfigV1:
    surface_feature_dim: int = 20
    joint_feature_dim: int = 8
    hidden_dim: int = 192
    latent_dim: int = 64
    encoder_layers: int = 3
    decoder_layers: int = 3
    dropout: float = 0.0
    temperature_floor: float = 0.35
    architecture_id: str = "RealSaS.SkinFieldCodec.ContinuousJointField.v1"

    def validate(self) -> None:
        if self.surface_feature_dim != 20 or self.joint_feature_dim != 8:
            raise ValueError("SkinFieldCodec V1 conditioner widths must remain 20/8")
        if min(self.hidden_dim, self.latent_dim, self.encoder_layers, self.decoder_layers) <= 0:
            raise ValueError("codec dimensions/layers must be positive")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout outside [0,1)")
        if self.temperature_floor <= 0.0:
            raise ValueError("temperature_floor must be positive")

    @property
    def config_hash(self) -> str:
        self.validate()
        return _hash(asdict(self))


@dataclass(frozen=True)
class ArachneCandidateConfigV1:
    surface_feature_dim: int = 20
    joint_feature_dim: int = 8
    model_dim: int = 256
    encoder_layers: int = 4
    cross_attention_layers: int = 3
    attention_heads: int = 8
    feedforward_dim: int = 768
    dropout: float = 0.0
    architecture_id: str = "RealSaS.ArachneCandidate.JointFieldPredictor.v1"

    def validate(self) -> None:
        if self.surface_feature_dim != 20 or self.joint_feature_dim != 8:
            raise ValueError("Arachne V1 conditioner widths must remain 20/8")
        if self.model_dim <= 0 or self.model_dim % self.attention_heads != 0:
            raise ValueError("model_dim must be positive and divisible by attention_heads")
        if min(self.encoder_layers, self.cross_attention_layers, self.attention_heads, self.feedforward_dim) <= 0:
            raise ValueError("Arachne architecture cardinalities must be positive")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout outside [0,1)")

    @property
    def config_hash(self) -> str:
        self.validate()
        return _hash(asdict(self))


GEPPETTO_V1 = GeppettoCandidateConfigV1()
SKIN_FIELD_CODEC_V1 = SkinFieldCodecConfigV1()
ARACHNE_V1 = ArachneCandidateConfigV1()
CANDIDATE_STACK_HASH_V1 = _hash({
    "geppetto": GEPPETTO_V1.config_hash,
    "skin_field_codec": SKIN_FIELD_CODEC_V1.config_hash,
    "arachne": ARACHNE_V1.config_hash,
    "product_ontology": "THREE_D_EQUIVALENT_MECHANICS_TO_DIRECTIONAL_2D_2P5D_PUPPET",
})
