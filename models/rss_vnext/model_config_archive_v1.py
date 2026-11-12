from __future__ import annotations

"""Exact RSS.vNext architecture config sidecar for independent Compiler readers.

The Compiler needs a TESSA codec, 3D witness geometry and relation-slot schema
matched to the ONE inference checkpoint. It must never guess the topology
quantizer bins or read a second, unrelated IRIS/TESSA release. This small
sidecar is signed by the inference receipt, source-parent index and checkpoint
sha; it grants no output or model-quality authority.
"""

from dataclasses import asdict
from hashlib import sha256
import json
from typing import Any

from .contracts import RSSvNextConfig
from .inference_checkpoint_v1 import _load_config

SCHEMA = "RealSaS.RSSvNext.UnifiedModelConfigArtifact.v1"
MAX_BYTES = 64 * 1024


def _digest(cfg: RSSvNextConfig) -> str:
    return sha256(json.dumps(
        asdict(cfg), sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf8")).hexdigest()


def _sha(value: Any) -> bool:
    return type(value) is str and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def encode_rss_vnext_unified_model_config_v1(
    cfg: RSSvNextConfig, *,
    checkpoint_sha256: str,
) -> bytes:
    if type(cfg) is not RSSvNextConfig or not _sha(checkpoint_sha256):
        raise RuntimeError("RSS_VNEXT_CONFIG_ARCHIVE_PARENT_INVALID")
    cfg.validate()
    payload = {
        "schema": SCHEMA,
        "checkpoint_sha256": checkpoint_sha256,
        "unified_model_config_sha256": _digest(cfg),
        "config": asdict(cfg),
        "product_authority": False,
    }
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf8")
    if len(raw) > MAX_BYTES:
        raise RuntimeError("RSS_VNEXT_CONFIG_ARCHIVE_RESOURCE_LIMIT")
    return raw


def decode_rss_vnext_unified_model_config_v1(
    raw: bytes, *,
    expected_checkpoint_sha256: str,
    expected_config_sha256: str,
) -> RSSvNextConfig:
    if type(raw) is not bytes or not 1 <= len(raw) <= MAX_BYTES:
        raise RuntimeError("RSS_VNEXT_CONFIG_ARCHIVE_RESOURCE_LIMIT")
    if not _sha(expected_checkpoint_sha256) or not _sha(expected_config_sha256):
        raise RuntimeError("RSS_VNEXT_CONFIG_ARCHIVE_EXPECTED_PARENT_INVALID")
    try:
        payload = json.loads(raw.decode("utf8"))
    except (UnicodeError, ValueError) as exc:
        raise RuntimeError("RSS_VNEXT_CONFIG_ARCHIVE_JSON_INVALID") from exc
    if (
        type(payload) is not dict
        or set(payload) != {
            "schema", "checkpoint_sha256", "unified_model_config_sha256",
            "config", "product_authority",
        }
        or payload["schema"] != SCHEMA
        or payload["product_authority"] is not False
        or payload["checkpoint_sha256"] != expected_checkpoint_sha256
        or payload["unified_model_config_sha256"] != expected_config_sha256
    ):
        raise RuntimeError("RSS_VNEXT_CONFIG_ARCHIVE_PARENT_OR_AUTHORITY_DRIFT")
    try:
        cfg = _load_config(payload["config"])
    except (TypeError, ValueError, RuntimeError) as exc:
        raise RuntimeError("RSS_VNEXT_CONFIG_ARCHIVE_CONFIG_INVALID") from exc
    if _digest(cfg) != expected_config_sha256:
        raise RuntimeError("RSS_VNEXT_CONFIG_ARCHIVE_CONFIG_DIGEST_DRIFT")
    return cfg


__all__ = [
    "SCHEMA", "MAX_BYTES",
    "encode_rss_vnext_unified_model_config_v1",
    "decode_rss_vnext_unified_model_config_v1",
]
