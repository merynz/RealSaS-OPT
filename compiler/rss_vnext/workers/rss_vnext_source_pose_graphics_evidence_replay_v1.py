from __future__ import annotations

"""Independently re-read raw source/native image buffers for Graphics court.

CAS checksums are not authority. The qualified source and native execution
registries are injected by the caller OUTSIDE the untrusted MODULE request.
Their independently pinned SHA-256s bind two real raster bundles to the same
canonical M*/Graphics/frame and registered camera. Never evaluate only a
rehashed SourcePoseGraphicsCourtV1 JSON as if it proved its own pixels.
"""

from hashlib import sha256
import io
from typing import Any, Mapping
import zipfile

import numpy as np

from compiler.realsas_compiler_core.rss_vnext_source_pose_graphics_court_v1 import (
    MAX_SIDE,
    SourcePoseGraphicsThresholdsV1,
    SourcePoseGraphicsCourtV1,
    evaluate_registered_source_pose_graphics_v1,
    reopen_source_pose_graphics_court_v1,
)
from compiler.realsas_compiler_core.types import QualificationError

SOURCE_BUNDLE_SCHEMA = "RealSaS.RSSvNext.SourcePoseGraphicsRasterBundle.v1"
NATIVE_BUNDLE_SCHEMA = "RealSaS.RSSvNext.NativePoseGraphicsRasterBundle.v1"

_SOURCE_KEYS = frozenset((
    "target_rgba_u8", "target_face_owner_i32", "target_known_mask",
))
_NATIVE_KEYS = frozenset((
    "native_rgba_u8", "native_front_face_i32",
    "native_color_authority_u8", "native_unknown_occlusion_u8",
))
_MAX_COMPRESSED_BYTES = 96 * 1024 * 1024
_MAX_DECODED_BYTES = 128 * 1024 * 1024


def _reopen_bounded_npz(data: bytes, keys: frozenset[str]) -> dict[str, np.ndarray]:
    if type(data) is not bytes or not 0 < len(data) <= _MAX_COMPRESSED_BYTES:
        raise QualificationError("RSS_SOURCE_GRAPHICS_RASTER_CAS_BUDGET")
    try:
        with zipfile.ZipFile(io.BytesIO(data), "r") as archive:
            infos = archive.infolist()
            if (
                len(infos) != len(keys)
                or {x.filename for x in infos}
                    != {k + ".npy" for k in keys}
                or any(
                    x.compress_type not in (
                        zipfile.ZIP_DEFLATED, zipfile.ZIP_STORED,
                    )
                    or x.flag_bits & 1
                    or x.file_size > 64 * 1024 * 1024
                    for x in infos
                )
                or sum(x.file_size for x in infos) > _MAX_DECODED_BYTES
            ):
                raise QualificationError("RSS_SOURCE_GRAPHICS_RASTER_ARCHIVE_DOMAIN")
        with np.load(io.BytesIO(data), allow_pickle=False) as loaded:
            if set(loaded.files) != keys:
                raise QualificationError("RSS_SOURCE_GRAPHICS_RASTER_ARRAY_KEYS")
            result = {key:loaded[key] for key in sorted(keys)}
    except (
        zipfile.BadZipFile, OSError, EOFError,
        ValueError, KeyError, OverflowError, MemoryError,
    ) as exc:
        raise QualificationError("RSS_SOURCE_GRAPHICS_RASTER_ARCHIVE_INVALID") from exc
    if (
        any(not isinstance(a,np.ndarray) or a.dtype.hasobject
            for a in result.values())
        or sum(a.nbytes for a in result.values()) > _MAX_DECODED_BYTES
    ):
        raise QualificationError("RSS_SOURCE_GRAPHICS_RASTER_DTYPE_OR_BUDGET")
    return result


def independently_replay_registered_source_pose_graphics_v1(
    *,
    source_raster_bundle: bytes,
    native_raster_bundle: bytes,
    native_face_count: int,
    live_pose_receipt: Mapping[str, Any],
    source_witness_key: str,
    renderer_witness_key: str,
    source_view_index: int,
    frame_index: int,
    source_target_kind: str,
    original_source_asset_sha256: str,
    original_source_file_sha256: str,
    source_registration_sha256: str,
    native_renderer_binary_sha256: str,
    canonical_mesh_lineage_hash: str,
    canonical_rig_lineage_hash: str,
    canonical_skin_lineage_hash: str,
    compiled_graphics_hash: str,
    canonical_motion_lineage_hash: str,
    frozen_thresholds: SourcePoseGraphicsThresholdsV1,
    trusted_source_registry: Mapping[str, Mapping[str, Any]],
    trusted_native_renderer_registry: Mapping[str, Mapping[str, Any]],
    claimed_court: Mapping[str, Any] | None = None,
) -> SourcePoseGraphicsCourtV1:
    """Recompute real arrays and court, not accept an arbitrary hash claim.

    Registries are injected as trusted host parameters, never copied from an
    untrusted request's capability_metadata. They must independently bind
    the actual NPZ bytes, source raster extraction and renderer execution.
    """
    source_row = trusted_source_registry.get(source_witness_key)
    native_row = trusted_native_renderer_registry.get(renderer_witness_key)
    source_digest = (
        sha256(source_raster_bundle).hexdigest()
        if type(source_raster_bundle) is bytes else None
    )
    native_digest = (
        sha256(native_raster_bundle).hexdigest()
        if type(native_raster_bundle) is bytes else None
    )
    if (
        not isinstance(source_row,Mapping) or not isinstance(native_row,Mapping)
        or source_row.get("raster_bundle_schema") != SOURCE_BUNDLE_SCHEMA
        or native_row.get("raster_bundle_schema") != NATIVE_BUNDLE_SCHEMA
        or source_row.get("raster_bundle_sha256") != source_digest
        or native_row.get("raster_bundle_sha256") != native_digest
        or source_row.get("source_raster_extraction_independently_verified")
            is not True
        or native_row.get("cpp_native_raster_execution_independently_verified")
            is not True
        or source_row.get("raster_artifact_not_student_prediction") is not True
        or native_row.get("raster_artifact_not_student_prediction") is not True
        or source_witness_key == renderer_witness_key
    ):
        raise QualificationError("RSS_SOURCE_GRAPHICS_RAW_INDEPENDENT_RASTER_MISSING")
    source = _reopen_bounded_npz(source_raster_bundle,_SOURCE_KEYS)
    native = _reopen_bounded_npz(native_raster_bundle,_NATIVE_KEYS)
    result=evaluate_registered_source_pose_graphics_v1(
        target_rgba_u8=source["target_rgba_u8"],
        target_face_owner_i32=source["target_face_owner_i32"],
        target_known_mask=source["target_known_mask"],
        native_rgba_u8=native["native_rgba_u8"],
        native_front_face_i32=native["native_front_face_i32"],
        native_color_authority_u8=native["native_color_authority_u8"],
        native_unknown_occlusion_u8=native["native_unknown_occlusion_u8"],
        native_face_count=native_face_count,
        live_pose_receipt=live_pose_receipt,
        source_witness_key=source_witness_key,
        renderer_witness_key=renderer_witness_key,
        source_view_index=source_view_index,
        frame_index=frame_index,
        source_target_kind=source_target_kind,
        original_source_asset_sha256=original_source_asset_sha256,
        original_source_file_sha256=original_source_file_sha256,
        source_registration_sha256=source_registration_sha256,
        native_renderer_binary_sha256=native_renderer_binary_sha256,
        canonical_mesh_lineage_hash=canonical_mesh_lineage_hash,
        canonical_rig_lineage_hash=canonical_rig_lineage_hash,
        canonical_skin_lineage_hash=canonical_skin_lineage_hash,
        compiled_graphics_hash=compiled_graphics_hash,
        canonical_motion_lineage_hash=canonical_motion_lineage_hash,
        thresholds=frozen_thresholds,
        trusted_source_registry=trusted_source_registry,
        trusted_native_renderer_registry=trusted_native_renderer_registry,
    )
    if claimed_court is not None:
        certified=reopen_source_pose_graphics_court_v1(
            claimed_court,
            canonical_mesh_lineage_hash=canonical_mesh_lineage_hash,
            canonical_rig_lineage_hash=canonical_rig_lineage_hash,
            canonical_skin_lineage_hash=canonical_skin_lineage_hash,
            compiled_graphics_hash=compiled_graphics_hash,
            canonical_motion_lineage_hash=canonical_motion_lineage_hash,
        )
        if certified != result:
            raise QualificationError("RSS_SOURCE_GRAPHICS_CLAIMED_COURT_REPLAY_DRIFT")
    return result


__all__ = [
    "SOURCE_BUNDLE_SCHEMA", "NATIVE_BUNDLE_SCHEMA",
    "independently_replay_registered_source_pose_graphics_v1",
]
