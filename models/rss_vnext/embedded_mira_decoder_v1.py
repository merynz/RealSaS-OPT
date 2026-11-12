from __future__ import annotations

"""Pinned MIRA decoder carved from the SAME RSS.vNext inference checkpoint.

SkinTokens' crucial lesson: learned FSQ IDs mean nothing without the exact
learned decoder and joint order. RSS does NOT require a second independent
MIRA checkpoint: producer exports its embedded, trained submodule once as a
CAS child. A distinct subartifact SHA is real serialized decoder bytes, NOT
an arbitrary synthetic "codec checkpoint hash".

Includes both FSQ token decoder and point-query spatial decoder; never
pretends soft expected-code training forward is identical to hard ID inference.
This supplies ONLY a structural codec binding, not validated skinning, rig,
source attachment, neural skill, or editable product authority.
"""

from hashlib import sha256
from io import BytesIO
from typing import Any
from zipfile import ZipFile

import torch

from models.axis.rig_program_decode_v6 import AXISV6CodecBinding
from models.mira.skin_fsq_codec_v6 import MIRA_V6_CODEC_ID
from .contracts import RSSvNextConfig
from .inference_checkpoint_v1 import _validated_state
from .model_config_archive_v1 import _digest as config_sha256
from .unified_character_model import (
    RSSvNextSkinCoupler, RSSvNextUnifiedCharacterModel,
)

SCHEMA = "RealSaS.RSSvNext.EmbeddedMIRADecoder.v1"
MAX_BYTES = 512 * 1024 * 1024
_FIELDS = {
    "schema", "parent_unified_checkpoint_sha256", "unified_model_config_sha256",
    "codec_architecture_id", "codebook_size", "tokens_per_joint",
    "skin_decoder_mode", "fsq_levels", "decoder_state_dict",
    "product_skin_authority", "soft_to_discrete_runtime_parity_qualified",
    "independent_model_release_required",
}


def _sha(value: Any) -> bool:
    return type(value) is str and len(value) == 64 and all(
        ch in "0123456789abcdef" for ch in value
    )


def encode_embedded_mira_decoder_v1(
    model: RSSvNextUnifiedCharacterModel, *,
    parent_unified_checkpoint_sha256: str,
) -> bytes:
    if (
        type(model) is not RSSvNextUnifiedCharacterModel
        or not _sha(parent_unified_checkpoint_sha256)
        or model.training or any(p.requires_grad for p in model.parameters())
    ):
        raise RuntimeError("RSS_VNEXT_EMBEDDED_MIRA_FROZEN_PARENT_REQUIRED")
    cfg = model.cfg
    cfg.validate()
    decoder = model.mira
    if (
        type(decoder) is not RSSvNextSkinCoupler
        or decoder.codec.architecture_id != MIRA_V6_CODEC_ID
        or decoder.codec.config.vocab_size != cfg.axis_config().skin_vocab_size
        or decoder.codec.config.tokens_per_joint != cfg.skin_tokens_per_joint
        or tuple(decoder.codec.config.levels) != tuple(cfg.skin_fsq_levels)
    ):
        raise RuntimeError("RSS_VNEXT_EMBEDDED_MIRA_DECODER_CONFIG_DRIFT")
    payload = {
        "schema": SCHEMA,
        "parent_unified_checkpoint_sha256": parent_unified_checkpoint_sha256,
        "unified_model_config_sha256": config_sha256(cfg),
        "codec_architecture_id": MIRA_V6_CODEC_ID,
        "codebook_size": decoder.codec.config.vocab_size,
        "tokens_per_joint": decoder.codec.config.tokens_per_joint,
        "skin_decoder_mode": cfg.skin_decoder_mode,
        "fsq_levels": tuple(cfg.skin_fsq_levels),
        "decoder_state_dict": {
            key: value.detach().cpu().contiguous().clone()
            for key, value in decoder.state_dict().items()
        },
        "product_skin_authority": False,
        "soft_to_discrete_runtime_parity_qualified": False,
        "independent_model_release_required": False,
    }
    stream = BytesIO()
    torch.save(payload, stream)
    data = stream.getvalue()
    if len(data) > MAX_BYTES:
        raise RuntimeError("RSS_VNEXT_EMBEDDED_MIRA_RESOURCE_LIMIT")
    return data


def load_embedded_mira_decoder_v1(
    data: bytes, *,
    expected_parent_unified_checkpoint_sha256: str,
    expected_subcheckpoint_sha256: str,
    expected_model_config_sha256: str,
    model_config: RSSvNextConfig,
) -> tuple[RSSvNextSkinCoupler, AXISV6CodecBinding]:
    if (
        type(data) is not bytes or not 1 <= len(data) <= MAX_BYTES
        or not all(map(_sha, (
            expected_parent_unified_checkpoint_sha256,
            expected_subcheckpoint_sha256, expected_model_config_sha256,
        )))
    ):
        raise RuntimeError("RSS_VNEXT_EMBEDDED_MIRA_RESOURCE_OR_PIN_INVALID")
    if sha256(data).hexdigest() != expected_subcheckpoint_sha256:
        raise RuntimeError("RSS_VNEXT_EMBEDDED_MIRA_CAS_BYTES_DRIFT")
    try:
        with ZipFile(BytesIO(data)) as archive:
            items = archive.infolist()
            if (
                not items or len(items) != len({x.filename for x in items})
                or sum(x.file_size for x in items) > MAX_BYTES
                or any(x.file_size > MAX_BYTES for x in items)
            ):
                raise RuntimeError("RSS_VNEXT_EMBEDDED_MIRA_ARCHIVE_LIMIT")
        payload = torch.load(BytesIO(data), map_location="cpu", weights_only=True)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError("RSS_VNEXT_EMBEDDED_MIRA_DECODE_INVALID") from exc
    if type(payload) is not dict or set(payload) != _FIELDS:
        raise RuntimeError("RSS_VNEXT_EMBEDDED_MIRA_SCHEMA_FIELDS_DRIFT")
    if (
        payload["schema"] != SCHEMA
        or payload["parent_unified_checkpoint_sha256"] != expected_parent_unified_checkpoint_sha256
        or payload["unified_model_config_sha256"] != expected_model_config_sha256
        or payload["codec_architecture_id"] != MIRA_V6_CODEC_ID
        or payload["product_skin_authority"] is not False
        or payload["soft_to_discrete_runtime_parity_qualified"] is not False
        or payload["independent_model_release_required"] is not False
    ):
        raise RuntimeError("RSS_VNEXT_EMBEDDED_MIRA_PARENT_OR_AUTHORITY_DRIFT")
    if (
        type(payload["codebook_size"]) is not int
        or type(payload["tokens_per_joint"]) is not int
        or type(payload["skin_decoder_mode"]) is not str
        or type(payload["fsq_levels"]) not in (tuple, list)
        or any(type(x) is not int or x < 2 for x in payload["fsq_levels"])
    ):
        raise RuntimeError("RSS_VNEXT_EMBEDDED_MIRA_CODEC_DOMAIN_INVALID")
    if (
        type(model_config) is not RSSvNextConfig
        or config_sha256(model_config) != expected_model_config_sha256
        or payload["codebook_size"] != model_config.axis_config().skin_vocab_size
        or payload["tokens_per_joint"] != model_config.skin_tokens_per_joint
        or payload["skin_decoder_mode"] != model_config.skin_decoder_mode
        or tuple(payload["fsq_levels"]) != tuple(model_config.skin_fsq_levels)
    ):
        raise RuntimeError("RSS_VNEXT_EMBEDDED_MIRA_PINNED_CONFIG_DRIFT")
    decoder = RSSvNextSkinCoupler(model_config)
    weights = _validated_state(payload["decoder_state_dict"])
    if set(weights) != set(decoder.state_dict()):
        raise RuntimeError("RSS_VNEXT_EMBEDDED_MIRA_STATE_KEYS_DRIFT")
    try:
        decoder.load_state_dict(weights, strict=True)
    except RuntimeError as exc:
        raise RuntimeError("RSS_VNEXT_EMBEDDED_MIRA_STATE_SHAPES_DRIFT") from exc
    decoder.eval()
    decoder.requires_grad_(False)
    if any(p.requires_grad for p in decoder.parameters()):
        raise RuntimeError("RSS_VNEXT_EMBEDDED_MIRA_NOT_FROZEN")
    return decoder, AXISV6CodecBinding(
        architecture_id=MIRA_V6_CODEC_ID,
        checkpoint_hash=expected_subcheckpoint_sha256,
        codebook_size=decoder.codec.config.vocab_size,
        tokens_per_joint=decoder.codec.config.tokens_per_joint,
    )


__all__ = [
    "SCHEMA", "encode_embedded_mira_decoder_v1",
    "load_embedded_mira_decoder_v1",
]
