from __future__ import annotations

"""Streaming Runtime-v4 P0/P1 writer.

The writer never reprojects canonical posed XYZ per view and never materializes the
full .rsr payload in Python memory. Large numeric arrays are emitted as contiguous
little-endian blocks while CRC32/SHA-256 are updated in the same pass.
"""

from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Iterable, Sequence
import json
import math
import os
import struct
import tempfile
import zipfile
import zlib

import numpy as np

from compiler.realsas_compiler_core.playback_runtime_v3 import AttachmentKind, TopologyClass
from compiler.realsas_compiler_core.playback_runtime_v4 import (
    DEFAULT_VIEWS,
    RuntimeV4Clip,
    RuntimeV4PlaybackContract,
    validate_playback_runtime_v4_contract,
    validate_runtime_v4_clip,
)
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_services.export.runtime_v3 import RuntimeV3TexturePayload

RUNTIME_V4_BINARY_MAGIC = b"RSRT\x00\x04\x00\x00"
RUNTIME_V4_BINARY_VERSION = 4
RUNTIME_V4_BINARY_SCHEMA = "RealSaS.RuntimeBinary.rsr.v4"
RUNTIME_V4_PACKAGE_SCHEMA = "RealSaS.RuntimePackage.native_v4.v1"
RUNTIME_V4_COORDINATE_SYSTEM = "canonical_3d.shared_pose__view_projection_runtime"
RUNTIME_V4_UNITS_PER_PIXEL = 1.0

FEATURE_RGBA8_TEXTURES = 1 << 0
FEATURE_CANONICAL_XYZ_FRAMES = 1 << 1
FEATURE_DYNAMIC_SLOT_DRAW_ORDER = 1 << 2
FEATURE_MULTI_VIEW_8 = 1 << 3
FEATURE_REFERENCE_DEPTH_RENDERER = 1 << 4
FEATURE_RUNTIME_VIEW_PROJECTION = 1 << 5
FEATURE_FULL_SURFACE_AUTHORITY = 1 << 6
FEATURE_SLOT_ATTACHMENTS = 1 << 7
FEATURE_CLIP_INTERVALS = 1 << 8
FEATURE_APPEARANCE_PROVENANCE = 1 << 9
FEATURE_SHARED_CANONICAL_GEOMETRY = 1 << 10
FEATURE_STREAMED_BULK_ARRAYS = 1 << 11
RUNTIME_V4_FEATURE_FLAGS = sum(1 << i for i in range(12))

_ATTACHMENT_KIND_CODE = {
    AttachmentKind.DEFORMABLE_BODY: 0,
    AttachmentKind.RIGID_COMPONENT: 1,
    AttachmentKind.CLIPPING: 2,
}
_TOPOLOGY_CLASS_CODE = {
    TopologyClass.STATIC: 0,
    TopologyClass.ATTACHMENT_DYNAMIC: 1,
    TopologyClass.CLIP_DYNAMIC: 2,
    TopologyClass.DRAW_ORDER_DYNAMIC: 3,
}


class _StreamWriter:
    def __init__(self, path: Path):
        self.path = path
        self.file = path.open("wb")
        self.body_crc = 0
        self.sha = sha256()
        self.bytes_written = 0
        self._finalized = False

    def raw(self, payload) -> None:
        view = memoryview(payload).cast("B")
        self.file.write(view)
        self.body_crc = zlib.crc32(view, self.body_crc) & 0xFFFFFFFF
        self.sha.update(view)
        self.bytes_written += len(view)

    def u8(self, value: int) -> None:
        self.raw(struct.pack("<B", int(value)))

    def i16_array(self, value) -> None:
        arr = np.ascontiguousarray(value, dtype="<i2")
        self.raw(arr)

    def u32(self, value: int) -> None:
        self.raw(struct.pack("<I", int(value)))

    def f32(self, value: float) -> None:
        value = float(value)
        if not math.isfinite(value):
            raise QualificationError("RUNTIME_V4_NONFINITE_BINARY_VALUE")
        self.raw(struct.pack("<f", value))

    def f32_array(self, value) -> None:
        arr = np.ascontiguousarray(value, dtype="<f4")
        if not np.isfinite(arr).all():
            raise QualificationError("RUNTIME_V4_NONFINITE_ARRAY_VALUE")
        self.raw(arr)

    def u32_array(self, value) -> None:
        arr = np.ascontiguousarray(value, dtype="<u4")
        self.raw(arr)

    def u8_array(self, value) -> None:
        arr = np.ascontiguousarray(value, dtype=np.uint8)
        self.raw(arr)

    def string(self, value: str) -> None:
        payload = str(value).encode("utf-8")
        self.u32(len(payload))
        self.raw(payload)

    def finalize(self) -> tuple[str, int]:
        if self._finalized:
            raise RuntimeError("Runtime-v4 stream already finalized")
        trailer = struct.pack("<I", self.body_crc)
        self.file.write(trailer)
        self.sha.update(trailer)
        self.bytes_written += len(trailer)
        self.file.flush()
        os.fsync(self.file.fileno())
        self.file.close()
        self._finalized = True
        return self.sha.hexdigest(), self.bytes_written

    def abort(self) -> None:
        if not self.file.closed:
            self.file.close()


def _sha_file(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _crc_file(path: Path) -> int:
    crc = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            crc = zlib.crc32(chunk, crc)
    return crc & 0xFFFFFFFF


def _png_size(path: Path) -> tuple[int, int]:
    raw = path.read_bytes()[:24]
    if len(raw) < 24 or raw[:8] != b"\x89PNG\r\n\x1a\n" or raw[12:16] != b"IHDR":
        raise QualificationError(f"RUNTIME_V4_TEXTURE_NOT_PNG:{path}")
    return struct.unpack(">II", raw[16:24])


def _validate_textures(
    textures: Iterable[RuntimeV3TexturePayload],
    *,
    texture_root: Path,
    required_views: tuple[str, ...],
) -> tuple[RuntimeV3TexturePayload, ...]:
    rows = tuple(textures)
    by_view = {row.view_id: row for row in rows}
    if len(by_view) != len(rows) or set(by_view) != set(required_views):
        raise QualificationError("RUNTIME_V4_TEXTURE_VIEW_SET_MISMATCH")
    root = texture_root.resolve()
    for view_id in required_views:
        row = by_view[view_id]
        path = (root / row.texture_path).resolve()
        if root not in path.parents and path != root:
            raise QualificationError("RUNTIME_V4_TEXTURE_PATH_ESCAPES_ROOT")
        if not path.is_file():
            raise QualificationError(f"RUNTIME_V4_TEXTURE_MISSING:{row.texture_path}")
        if len(row.texture_sha256) != 64 or _sha_file(path) != row.texture_sha256:
            raise QualificationError(f"RUNTIME_V4_TEXTURE_SHA_MISMATCH:{view_id}")
        if _crc_file(path) != int(row.texture_crc32):
            raise QualificationError(f"RUNTIME_V4_TEXTURE_CRC_MISMATCH:{view_id}")
        if _png_size(path) != (int(row.width), int(row.height)):
            raise QualificationError(f"RUNTIME_V4_TEXTURE_SIZE_MISMATCH:{view_id}")
    return tuple(by_view[v] for v in required_views)


def _contract_summary(contract: RuntimeV4PlaybackContract, contract_hash: str) -> dict:
    return {
        "schema_version": contract.schema_version,
        "contract_hash": contract_hash,
        "raster_contract_hash": contract.raster.contract_hash,
        "allow_completion": bool(contract.allow_completion),
        "slots": [asdict(slot) for slot in contract.slots],
        "assets": [
            {
                "asset_id": asset.asset_id,
                "slot_id": asset.slot_id,
                "attachment_id": asset.attachment_id,
                "attachment_kind": asset.attachment_kind.value,
                "topology_class": asset.topology_class.value,
                "vertex_count": asset.vertex_count,
                "face_count": asset.face_count,
                "sealed_source_hash": asset.sealed_source_hash,
                "asset_hash": asset.asset_hash,
            }
            for asset in contract.assets
        ],
        "views": [
            {
                "view_id": view.view_id,
                "view_index": view.view_index,
                "camera": asdict(view.camera),
                "asset_ids": [row.asset_id for row in view.assets],
            }
            for view in contract.views
        ],
    }


def _write_binary(
    *,
    out_path: Path,
    contract: RuntimeV4PlaybackContract,
    textures: tuple[RuntimeV3TexturePayload, ...],
    clips: tuple[RuntimeV4Clip, ...],
    source_binding_sha256: str,
    source_proof_bundle_hash: str,
    required_views: tuple[str, ...],
    contract_hash: str,
) -> tuple[str, int, dict]:
    asset_index = {asset.asset_id: i for i, asset in enumerate(contract.assets)}
    slot_index = {slot.slot_id: i for i, slot in enumerate(contract.slots)}
    texture_by_view = {row.view_id: row for row in textures}

    static_asset_bytes = 0
    per_view_overlay_bytes = 0
    deform_array_bytes_per_frame = sum(asset.vertex_count * 3 * 4 for asset in contract.assets)
    deform_array_bytes_total = deform_array_bytes_per_frame * sum(len(clip.frames) for clip in clips)

    writer = _StreamWriter(out_path)
    try:
        writer.raw(RUNTIME_V4_BINARY_MAGIC)
        writer.u32(RUNTIME_V4_BINARY_VERSION)
        writer.string(RUNTIME_V4_BINARY_SCHEMA)
        writer.string(source_binding_sha256)
        writer.string(source_proof_bundle_hash)
        writer.string(RUNTIME_V4_COORDINATE_SYSTEM)
        writer.f32(RUNTIME_V4_UNITS_PER_PIXEL)
        writer.u32(RUNTIME_V4_FEATURE_FLAGS)
        writer.string(contract_hash)
        writer.string(contract.raster.contract_hash)
        writer.f32(contract.visibility.alpha_cutout_threshold)

        writer.u32(len(contract.slots))
        for slot in contract.slots:
            writer.string(slot.slot_id)
            writer.string(slot.bone_id)
            writer.u32(slot.setup_order)
            writer.string(slot.default_attachment_id or "")

        writer.u32(len(contract.assets))
        for asset in contract.assets:
            xyz, tri = asset.canonical_arrays()
            writer.string(asset.asset_id)
            writer.u32(slot_index[asset.slot_id])
            writer.string(asset.attachment_id)
            writer.u8(_ATTACHMENT_KIND_CODE[asset.attachment_kind])
            writer.u8(_TOPOLOGY_CLASS_CODE[asset.topology_class])
            writer.u8(0); writer.u8(0)
            writer.string(asset.sealed_source_hash)
            writer.u32(len(xyz)); writer.u32(len(tri))
            before = writer.bytes_written
            writer.f32_array(xyz)
            writer.u32_array(tri)
            static_asset_bytes += writer.bytes_written - before

        writer.u32(len(required_views))
        for view_id in required_views:
            view = contract.views[required_views.index(view_id)]
            texture = texture_by_view[view_id]
            writer.string(view_id)
            writer.string(texture.texture_path.replace("\\", "/"))
            writer.u32(texture.texture_crc32)
            writer.u32(texture.width); writer.u32(texture.height)
            camera = view.camera
            writer.f32_array(np.asarray((camera.origin, camera.right, camera.screen_up, camera.forward), dtype=np.float32))
            writer.f32(camera.half_extent); writer.u32(camera.resolution)
            writer.u32(len(view.assets))
            for row in view.assets:
                writer.u32(asset_index[row.asset_id])
                before = writer.bytes_written
                writer.f32_array(row.uv)
                writer.u8_array(row.provenance_codes)
                writer.i16_array(row.donor_view_indices)
                per_view_overlay_bytes += writer.bytes_written - before

        writer.u32(len(clips))
        for clip in clips:
            writer.string(clip.clip_id)
            writer.string(clip.display_name)
            writer.string(clip.intent)
            writer.f32(clip.duration_seconds)
            writer.f32(clip.fps)
            writer.u8(1 if clip.loop else 0)
            writer.u8(1 if clip.runtime_qualified else 0)
            writer.u8(0); writer.u8(0)
            writer.u32(len(clip.frames))
            for frame in clip.frames:
                writer.f32(frame.time_seconds)
                for asset in contract.assets:
                    writer.f32_array(frame.canonical_posed_xyz_by_asset[asset.asset_id])
                for view_id in required_views:
                    composition = frame.composition_by_view[view_id]
                    writer.u32(len(composition.draw_order_slot_ids))
                    for slot_id in composition.draw_order_slot_ids:
                        writer.u32(slot_index[slot_id])
                    writer.u32(len(contract.slots))
                    for slot in contract.slots:
                        writer.string(composition.active_attachment_by_slot[slot.slot_id] or "")
                    writer.u32(len(composition.clip_intervals))
                    for interval in composition.clip_intervals:
                        writer.string(interval.clip_attachment_id)
                        writer.u32(slot_index[interval.start_slot_id])
                        writer.u32(slot_index[interval.end_slot_id])
                        writer.u8(1 if interval.inverse else 0)
                        writer.u8(0); writer.u8(0); writer.u8(0)
        binary_sha, binary_bytes = writer.finalize()
    except Exception:
        writer.abort()
        raise

    return binary_sha, binary_bytes, {
        "static_asset_array_bytes": static_asset_bytes,
        "per_view_overlay_array_bytes": per_view_overlay_bytes,
        "deform_array_bytes_per_frame": deform_array_bytes_per_frame,
        "deform_array_bytes_total": deform_array_bytes_total,
    }


def materialize_runtime_v4_archive(
    *,
    out_path: str | Path,
    texture_root: str | Path,
    contract: RuntimeV4PlaybackContract,
    textures: Iterable[RuntimeV3TexturePayload],
    clips: Iterable[RuntimeV4Clip],
    source_product_state_hash: str,
    source_proof_bundle_hash: str,
    source_authority_kind: str = "PRODUCT_PROOF_BUNDLE",
    required_views: Sequence[str] = DEFAULT_VIEWS,
) -> dict:
    started = perf_counter()
    view_ids = tuple(map(str, required_views))
    contract_hash = validate_playback_runtime_v4_contract(contract, required_view_ids=view_ids)
    clip_rows = tuple(clips)
    if not clip_rows or len({clip.clip_id for clip in clip_rows}) != len(clip_rows):
        raise QualificationError("RUNTIME_V4_CLIP_SET_INVALID")
    for clip in clip_rows:
        validate_runtime_v4_clip(contract, clip, required_view_ids=view_ids)
    if len(source_product_state_hash) != 64 or len(source_proof_bundle_hash) != 64:
        raise QualificationError("RUNTIME_V4_SOURCE_IDENTITIES_MUST_BE_SHA256")
    source_authority_kind = str(source_authority_kind or "").strip()
    if source_authority_kind not in {
        "PRODUCT_PROOF_BUNDLE",
        "RUNTIME_V4_ADMISSION_CERTIFICATE",
    }:
        raise QualificationError("RUNTIME_V4_SOURCE_AUTHORITY_KIND_INVALID")

    root = Path(texture_root).resolve()
    texture_rows = _validate_textures(textures, texture_root=root, required_views=view_ids)
    contract_bytes = json.dumps(
        _contract_summary(contract, contract_hash),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    binding = {
        "schema": "RealSaS.RuntimeV4SourceBinding.v1",
        "source_product_state_hash": source_product_state_hash,
        "source_proof_bundle_hash": source_proof_bundle_hash,
        "source_authority_kind": source_authority_kind,
        "legacy_binary_source_proof_bundle_hash_field_semantics": (
            "PRODUCT_PROOF_BUNDLE_HASH"
            if source_authority_kind == "PRODUCT_PROOF_BUNDLE"
            else "RUNTIME_V4_ADMISSION_CERTIFICATE_HASH__NOT_FULL_PRODUCT_PROOF"
        ),
        "full_product_proof_claimed": source_authority_kind == "PRODUCT_PROOF_BUNDLE",
        "playback_contract_hash": contract_hash,
        "reference_raster_contract_hash": contract.raster.contract_hash,
        "texture_hashes": {row.view_id: row.texture_sha256 for row in texture_rows},
        "clip_ids": [clip.clip_id for clip in clip_rows],
        "representation": "SHARED_CANONICAL_ASSET__VIEW_OVERLAY__CANONICAL_XYZ_ONCE_PER_FRAME",
    }
    binding_bytes = json.dumps(binding, sort_keys=True, separators=(",", ":")).encode("utf-8")
    binding_sha = sha256(binding_bytes).hexdigest()

    target = Path(out_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="realsas_runtime_v4_", dir=str(target.parent)) as temp_dir:
        binary_path = Path(temp_dir) / "realsas_runtime.rsr"
        binary_started = perf_counter()
        binary_sha, binary_bytes, size_rows = _write_binary(
            out_path=binary_path,
            contract=contract,
            textures=texture_rows,
            clips=clip_rows,
            source_binding_sha256=binding_sha,
            source_proof_bundle_hash=source_proof_bundle_hash,
            required_views=view_ids,
            contract_hash=contract_hash,
        )
        binary_seconds = perf_counter() - binary_started

        manifest = {
            "schema_version": RUNTIME_V4_PACKAGE_SCHEMA,
            "binary": {
                "path": "runtime/realsas_runtime.rsr",
                "schema": RUNTIME_V4_BINARY_SCHEMA,
                "sha256": binary_sha,
                "bytes": binary_bytes,
            },
            "source_binding": {**binding, "content_sha256": binding_sha},
            "playback_contract": {
                "path": "runtime/playback_contract.json",
                "content_sha256": sha256(contract_bytes).hexdigest(),
            },
            "render_capabilities": {
                "depth_test": True,
                "posed_vertex_depth": True,
                "full_surface_geometry_authority": True,
                "canonical_posed_xyz_once_per_attachment_per_frame": True,
                "runtime_exact_view_projection": True,
                "shared_canonical_geometry": True,
                "semantic_slot_draw_order": True,
                "clip_intervals_transport": True,
                "unseen_face_suppression": True,
                "visibility_by_face_deletion": False,
            },
            "measurement": {
                **size_rows,
                "runtime_binary_bytes": binary_bytes,
                "runtime_binary_write_seconds": binary_seconds,
                "frame_count_total": sum(len(clip.frames) for clip in clip_rows),
                "view_count": len(view_ids),
                "asset_count": len(contract.assets),
            },
            "source_authority": {
                "kind": source_authority_kind,
                "hash": source_proof_bundle_hash,
                "full_product_proof_claimed": source_authority_kind == "PRODUCT_PROOF_BUNDLE",
            },
            "founder_visual_pass_claimed": False,
        }
        manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")

        archive_started = perf_counter()
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            archive.writestr("manifest.json", manifest_bytes)
            archive.writestr("source_binding.json", binding_bytes)
            archive.writestr("runtime/playback_contract.json", contract_bytes)
            archive.write(binary_path, "runtime/realsas_runtime.rsr")
            for texture in texture_rows:
                archive.write(root / texture.texture_path, texture.texture_path.replace("\\", "/"))
        archive_seconds = perf_counter() - archive_started

    total_seconds = perf_counter() - started
    return {
        "archive_path": str(target),
        "archive_sha256": _sha_file(target),
        "package_total_bytes": target.stat().st_size,
        "runtime_binary_sha256": binary_sha,
        "runtime_binary_bytes": binary_bytes,
        "source_binding_sha256": binding_sha,
        "playback_contract_hash": contract_hash,
        "reference_raster_contract_hash": contract.raster.contract_hash,
        "view_count": len(texture_rows),
        "clip_count": len(clip_rows),
        "frame_count_total": sum(len(clip.frames) for clip in clip_rows),
        "runtime_binary_write_seconds": binary_seconds,
        "archive_write_seconds": archive_seconds,
        "wall_seconds": total_seconds,
        **size_rows,
    }
