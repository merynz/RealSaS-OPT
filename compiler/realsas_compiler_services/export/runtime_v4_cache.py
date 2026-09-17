from __future__ import annotations

"""Persistent content-addressed cache wrapper for Runtime-v4 archives."""

from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Iterable, Sequence
import json
import os
import struct
import zlib

import numpy as np

from compiler.realsas_compiler_core.playback_runtime_v4 import (
    DEFAULT_VIEWS,
    RuntimeV4Clip,
    RuntimeV4PlaybackContract,
    validate_playback_runtime_v4_contract,
    validate_runtime_v4_clip,
)
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_services.cache.content_addressed import (
    ContentAddressedStageCache,
)
from compiler.realsas_compiler_services.export.runtime_v3 import RuntimeV3TexturePayload
from compiler.realsas_compiler_services.export.runtime_v4 import (
    RUNTIME_V4_BINARY_SCHEMA,
    RUNTIME_V4_PACKAGE_SCHEMA,
    materialize_runtime_v4_archive,
)

RUNTIME_V4_CACHE_STAGE = "runtime_v4_archive"
RUNTIME_V4_CACHE_PRODUCER = (
    "RealSaS.RuntimeV4ArchiveCacheProducer.p2.v1"
    f"|binary={RUNTIME_V4_BINARY_SCHEMA}|package={RUNTIME_V4_PACKAGE_SCHEMA}"
)


def _default_cache_root() -> Path:
    override = os.environ.get("REALSAS_CACHE_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / ".cache" / "realsas").resolve()


def _sha_crc_file(path: Path) -> tuple[str, int, int]:
    h = sha256()
    crc = 0
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
            crc = zlib.crc32(chunk, crc)
            size += len(chunk)
    return h.hexdigest(), crc & 0xFFFFFFFF, size


def _canonical_json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _update_array(h, value, *, dtype: str) -> None:
    arr = np.ascontiguousarray(value, dtype=dtype)
    h.update(struct.pack("<I", arr.ndim))
    for dim in arr.shape:
        h.update(struct.pack("<Q", int(dim)))
    h.update(memoryview(arr).cast("B"))


def _clip_set_digest(
    contract: RuntimeV4PlaybackContract,
    clips: tuple[RuntimeV4Clip, ...],
    *,
    view_ids: tuple[str, ...],
) -> str:
    h = sha256()
    h.update(b"RealSaS.RuntimeV4ClipSetCacheIdentity.v1\0")
    h.update(struct.pack("<I", len(clips)))
    asset_ids = tuple(asset.asset_id for asset in contract.assets)
    slot_ids = tuple(slot.slot_id for slot in contract.slots)

    for clip in clips:
        validate_runtime_v4_clip(contract, clip, required_view_ids=view_ids)
        h.update(_canonical_json({
            "schema_version": clip.schema_version,
            "clip_id": clip.clip_id,
            "display_name": clip.display_name,
            "intent": clip.intent,
            "duration_seconds": float(clip.duration_seconds),
            "fps": float(clip.fps),
            "loop": bool(clip.loop),
            "runtime_qualified": bool(clip.runtime_qualified),
            "frame_count": len(clip.frames),
        }))
        for frame in clip.frames:
            h.update(struct.pack("<d", float(frame.time_seconds)))
            for asset_id in asset_ids:
                payload = np.asarray(frame.canonical_posed_xyz_by_asset[asset_id], dtype=np.float32)
                _update_array(h, payload, dtype="<f4")
            for view_id in view_ids:
                composition = frame.composition_by_view[view_id]
                h.update(_canonical_json({
                    "view_id": view_id,
                    "draw_order_slot_ids": list(composition.draw_order_slot_ids),
                    "active_attachment_by_slot": [
                        [slot_id, composition.active_attachment_by_slot[slot_id]]
                        for slot_id in slot_ids
                    ],
                    "clip_intervals": [
                        {
                            "clip_attachment_id": row.clip_attachment_id,
                            "start_slot_id": row.start_slot_id,
                            "end_slot_id": row.end_slot_id,
                            "inverse": bool(row.inverse),
                        }
                        for row in composition.clip_intervals
                    ],
                }))
    return h.hexdigest()


def _texture_identities(
    textures: Iterable[RuntimeV3TexturePayload],
    *,
    texture_root: Path,
    view_ids: tuple[str, ...],
) -> list[dict]:
    rows = tuple(textures)
    by_view = {row.view_id: row for row in rows}
    if len(by_view) != len(rows) or set(by_view) != set(view_ids):
        raise QualificationError("RUNTIME_V4_CACHE_TEXTURE_VIEW_SET_MISMATCH")
    root = texture_root.resolve()
    out = []
    for view_id in view_ids:
        row = by_view[view_id]
        path = (root / row.texture_path).resolve()
        if root not in path.parents and path != root:
            raise QualificationError("RUNTIME_V4_CACHE_TEXTURE_PATH_ESCAPES_ROOT")
        if not path.is_file():
            raise QualificationError(f"RUNTIME_V4_CACHE_TEXTURE_MISSING:{row.texture_path}")
        actual_sha, actual_crc, actual_bytes = _sha_crc_file(path)
        if actual_sha != row.texture_sha256:
            raise QualificationError(f"RUNTIME_V4_CACHE_TEXTURE_SHA_MISMATCH:{view_id}")
        if actual_crc != int(row.texture_crc32):
            raise QualificationError(f"RUNTIME_V4_CACHE_TEXTURE_CRC_MISMATCH:{view_id}")
        out.append({
            "view_id": view_id,
            "texture_path": row.texture_path.replace("\\", "/"),
            "sha256": actual_sha,
            "crc32": actual_crc,
            "bytes": actual_bytes,
            "width": int(row.width),
            "height": int(row.height),
        })
    return out


def runtime_v4_cache_identity(
    *,
    texture_root: str | Path,
    contract: RuntimeV4PlaybackContract,
    textures: Iterable[RuntimeV3TexturePayload],
    clips: Iterable[RuntimeV4Clip],
    source_product_state_hash: str,
    source_proof_bundle_hash: str,
    required_views: Sequence[str] = DEFAULT_VIEWS,
    cache_root: str | Path | None = None,
) -> tuple[ContentAddressedStageCache, str, dict]:
    view_ids = tuple(map(str, required_views))
    if len(source_product_state_hash) != 64 or len(source_proof_bundle_hash) != 64:
        raise QualificationError("RUNTIME_V4_CACHE_SOURCE_IDENTITIES_MUST_BE_SHA256")
    contract_hash = validate_playback_runtime_v4_contract(contract, required_view_ids=view_ids)
    clip_rows = tuple(clips)
    if not clip_rows or len({clip.clip_id for clip in clip_rows}) != len(clip_rows):
        raise QualificationError("RUNTIME_V4_CACHE_CLIP_SET_INVALID")
    clip_digest = _clip_set_digest(contract, clip_rows, view_ids=view_ids)
    texture_rows = _texture_identities(
        textures,
        texture_root=Path(texture_root).resolve(),
        view_ids=view_ids,
    )
    inputs = {
        "runtime_binary_schema": RUNTIME_V4_BINARY_SCHEMA,
        "runtime_package_schema": RUNTIME_V4_PACKAGE_SCHEMA,
        "playback_contract_hash": contract_hash,
        "reference_raster_contract_hash": contract.raster.contract_hash,
        "source_product_state_hash": source_product_state_hash,
        "source_proof_bundle_hash": source_proof_bundle_hash,
        "required_views": list(view_ids),
        "textures": texture_rows,
        "clip_set_sha256": clip_digest,
        "clip_ids": [clip.clip_id for clip in clip_rows],
    }
    cache = ContentAddressedStageCache(
        cache_root if cache_root is not None else _default_cache_root(),
        stage=RUNTIME_V4_CACHE_STAGE,
        producer_fingerprint=RUNTIME_V4_CACHE_PRODUCER,
    )
    return cache, cache.key(inputs), inputs


def materialize_runtime_v4_archive_cached(
    *,
    out_path: str | Path,
    texture_root: str | Path,
    contract: RuntimeV4PlaybackContract,
    textures: Iterable[RuntimeV3TexturePayload],
    clips: Iterable[RuntimeV4Clip],
    source_product_state_hash: str,
    source_proof_bundle_hash: str,
    required_views: Sequence[str] = DEFAULT_VIEWS,
    cache_root: str | Path | None = None,
) -> dict:
    started = perf_counter()
    texture_rows = tuple(textures)
    clip_rows = tuple(clips)
    identity_started = perf_counter()
    cache, cache_key, _ = runtime_v4_cache_identity(
        texture_root=texture_root,
        contract=contract,
        textures=texture_rows,
        clips=clip_rows,
        source_product_state_hash=source_product_state_hash,
        source_proof_bundle_hash=source_proof_bundle_hash,
        required_views=required_views,
        cache_root=cache_root,
    )
    identity_seconds = perf_counter() - identity_started

    target = Path(out_path).expanduser().resolve()
    lookup_started = perf_counter()
    restored = cache.restore(cache_key, target)
    lookup_seconds = perf_counter() - lookup_started
    if restored.hit:
        result = dict(restored.metadata or {})
        prior_wall = result.get("wall_seconds")
        if prior_wall is not None:
            result["cold_materializer_wall_seconds"] = prior_wall
        result.update({
            "archive_path": str(target),
            "archive_sha256": restored.artifact_sha256,
            "package_total_bytes": restored.artifact_bytes,
            "cache_hit": True,
            "cache_key": cache_key,
            "cache_reason": restored.reason,
            "cache_output_already_current": restored.output_already_current,
            "cache_identity_seconds": identity_seconds,
            "cache_lookup_seconds": lookup_seconds,
            "wall_seconds": perf_counter() - started,
        })
        return result

    fresh = materialize_runtime_v4_archive(
        out_path=target,
        texture_root=texture_root,
        contract=contract,
        textures=texture_rows,
        clips=clip_rows,
        source_product_state_hash=source_product_state_hash,
        source_proof_bundle_hash=source_proof_bundle_hash,
        required_views=required_views,
    )
    materializer_wall = float(fresh.get("wall_seconds", 0.0))
    metadata = {k: v for k, v in fresh.items() if k != "archive_path"}
    store_started = perf_counter()
    stored = cache.store(cache_key, target, metadata=metadata)
    store_seconds = perf_counter() - store_started

    result = dict(fresh)
    result.update({
        "cache_hit": False,
        "cache_key": cache_key,
        "cache_reason": restored.reason,
        "cache_output_already_current": False,
        "cache_identity_seconds": identity_seconds,
        "cache_lookup_seconds": lookup_seconds,
        "cache_store_seconds": store_seconds,
        "cache_entry_path": stored.entry_path,
        "cold_materializer_wall_seconds": materializer_wall,
        "wall_seconds": perf_counter() - started,
    })
    return result
