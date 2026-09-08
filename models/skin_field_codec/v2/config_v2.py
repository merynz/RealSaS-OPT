from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json


def _hash(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return sha256(raw).hexdigest()


@dataclass(frozen=True)
class SkinFieldCodecConfigV2:
    surface_feature_dim: int = 20
    joint_feature_dim: int = 8
    pair_geometry_dim: int = 10
    pair_geometry_embed_dim: int = 64
    hidden_dim: int = 192
    latent_dim: int = 64
    encoder_layers: int = 3
    decoder_layers: int = 3
    dropout: float = 0.0
    temperature_floor: float = 0.35
    architecture_id: str = "RealSaS.SkinFieldCodec.ContinuousJointField.PairGeometry.v2"

    def validate(self) -> None:
        if self.surface_feature_dim != 20 or self.joint_feature_dim != 8:
            raise ValueError("SkinFieldCodec V2 conditioner widths must remain 20/8")
        if self.pair_geometry_dim != 10 or self.pair_geometry_embed_dim != 64:
            raise ValueError("SkinFieldCodec V2 pair geometry widths must remain 10/64")
        if self.hidden_dim != 192 or self.latent_dim != 64:
            raise ValueError("SkinFieldCodec V2 hidden/latent widths must remain 192/64")
        if self.encoder_layers != 3 or self.decoder_layers != 3:
            raise ValueError("SkinFieldCodec V2 encoder/decoder depths must remain 3/3")
        if self.dropout != 0.0:
            raise ValueError("SkinFieldCodec V2 dropout must remain zero")
        if self.temperature_floor != 0.35:
            raise ValueError("SkinFieldCodec V2 temperature floor must remain 0.35")
        if self.architecture_id != "RealSaS.SkinFieldCodec.ContinuousJointField.PairGeometry.v2":
            raise ValueError("SkinFieldCodec V2 architecture id drift")

    @property
    def config_hash(self) -> str:
        self.validate()
        return _hash(asdict(self))


SKIN_FIELD_CODEC_V2 = SkinFieldCodecConfigV2()
EXPECTED_SKIN_FIELD_CODEC_V2_CONFIG_HASH = "8282eca35a1983363be8709ea3d9ea6af435dda0c424cfe0143c3e77b0dd8d2a"

if SKIN_FIELD_CODEC_V2.config_hash != EXPECTED_SKIN_FIELD_CODEC_V2_CONFIG_HASH:
    raise RuntimeError("SKIN_FIELD_CODEC_V2_CONFIG_HASH_DRIFT")
