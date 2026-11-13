"""Compatibility import for the byte-preserved V1 codec source.

Only the SkinFieldCodec configuration is current in this semantic package.
"""

from .config_v1 import SKIN_FIELD_CODEC_V1, SkinFieldCodecConfigV1

__all__ = ["SKIN_FIELD_CODEC_V1", "SkinFieldCodecConfigV1"]
