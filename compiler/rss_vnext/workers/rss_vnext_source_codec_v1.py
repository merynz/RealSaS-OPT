from __future__ import annotations

"""CAS transport for TWO unmodified observed RSS.vNext images and cameras.

Never consume the legacy eight-view SourceObservation.v1 as RSS input.
No RGB flattening, crop, synthesized back camera, material or rights grant.
Torch-free at import time and safe to use by both Input and Shape workers.
"""

from hashlib import sha256
import io
import json
from typing import Any
from zipfile import ZipFile

import numpy as np

from compiler.realsas_compiler_core.rss_vnext_source_observation_v1 import (
    RSSvNextTwoViewSourceObservationV1,
    SCHEMA as SOURCE_SCHEMA,
    MAX_BYTES as MAX_SOURCE_BYTES,
    canonicalize_two_view_source_v1,
)

ARTIFACT_TYPE = "RealSaS.RSSvNextTwoViewSource"
SCHEMA_VERSION = "v1"
_ARCHIVE_FILES = frozenset({
    "schema.npy", "native_rgba.npy", "camera_features.npy",
    "source_rights_claim_sha256.npy",
})
MAX_COMPRESSED_BYTES = 128 * 1024 * 1024


def _validate_archive(payload: bytes) -> None:
    if type(payload) is not bytes or not 1 <= len(payload) <= MAX_COMPRESSED_BYTES:
        raise RuntimeError("RSS_VNEXT_SOURCE_CAS_BYTES_INVALID")
    try:
        with ZipFile(io.BytesIO(payload)) as reader:
            entries = reader.infolist()
            if (
                len(entries) != len(_ARCHIVE_FILES)
                or {row.filename for row in entries} != _ARCHIVE_FILES
                or sum(row.file_size for row in entries) > MAX_SOURCE_BYTES + 8192
                or any(row.file_size > MAX_SOURCE_BYTES for row in entries)
            ):
                raise RuntimeError("RSS_VNEXT_SOURCE_CAS_ARCHIVE_LIMIT_OR_FIELDS")
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError("RSS_VNEXT_SOURCE_CAS_ARCHIVE_INVALID") from exc


def encode_rss_vnext_two_view_source_v1(
    source: RSSvNextTwoViewSourceObservationV1,
) -> bytes:
    if type(source) is not RSSvNextTwoViewSourceObservationV1:
        raise RuntimeError("RSS_VNEXT_SOURCE_TYPED_CANONICAL_REQUIRED")
    stream = io.BytesIO()
    np.savez_compressed(
        stream,
        schema=np.asarray([SOURCE_SCHEMA]),
        native_rgba=source.native_rgba,
        camera_features=source.camera_features,
        source_rights_claim_sha256=np.asarray([source.source_rights_claim_sha256]),
    )
    result = stream.getvalue()
    _validate_archive(result)
    return result


def decode_rss_vnext_two_view_source_v1(
    payload: bytes,
) -> RSSvNextTwoViewSourceObservationV1:
    _validate_archive(payload)
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if set(archive.files) != {
                "schema", "native_rgba", "camera_features", "source_rights_claim_sha256",
            }:
                raise RuntimeError("RSS_VNEXT_SOURCE_CAS_FIELDS_INVALID")
            header = archive["schema"]
            rights = archive["source_rights_claim_sha256"]
            if (
                header.shape != (1,) or str(header[0]) != SOURCE_SCHEMA
                or rights.shape != (1,)
            ):
                raise RuntimeError("RSS_VNEXT_SOURCE_CAS_SCHEMA_INVALID")
            native = np.asarray(archive["native_rgba"]).copy()
            cameras = np.asarray(archive["camera_features"]).copy()
            rights_digest = str(rights[0])
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError("RSS_VNEXT_SOURCE_CAS_DECODE_INVALID") from exc
    return canonicalize_two_view_source_v1(
        native, cameras, source_rights_claim_sha256=rights_digest,
    )


def validate_rss_vnext_source_ref_v1(
    artifact_ref: dict[str, Any], payload: bytes,
) -> RSSvNextTwoViewSourceObservationV1:
    if (
        type(artifact_ref) is not dict
        or artifact_ref.get("artifact_type") != ARTIFACT_TYPE
        or artifact_ref.get("schema_version") != SCHEMA_VERSION
    ):
        raise RuntimeError("RSS_VNEXT_SOURCE_CAS_REF_SCHEMA_INVALID")
    digest = sha256(payload).hexdigest()
    if (
        artifact_ref.get("content_sha256") != digest
        or type(artifact_ref.get("size_bytes")) is not int
        or artifact_ref["size_bytes"] != len(payload)
    ):
        raise RuntimeError("RSS_VNEXT_SOURCE_CAS_REF_CONTENT_DRIFT")
    return decode_rss_vnext_two_view_source_v1(payload)


__all__ = [
    "ARTIFACT_TYPE", "SCHEMA_VERSION",
    "encode_rss_vnext_two_view_source_v1",
    "decode_rss_vnext_two_view_source_v1",
    "validate_rss_vnext_source_ref_v1",
]
