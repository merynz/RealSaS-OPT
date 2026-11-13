from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json


def _hash(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return sha256(raw).hexdigest()


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


SKIN_FIELD_CODEC_V1 = SkinFieldCodecConfigV1()
