from __future__ import annotations

"""Pure generic RealSaS native runtime-v3 deployment writer.

Consumes only already-qualified playback-v3 contract data and already-baked posed
XYZ frames. It does not run models, solvers, FK/LBS, visibility repair, appearance
completion or subject-specific logic.
"""

from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Iterable, Mapping
import json
import math
import struct
import zipfile
import zlib

from compiler.realsas_compiler_core.playback_runtime_v3 import (
    AttachmentKind,
    RuntimeV3FrameComposition,
    RuntimeV3PlaybackContract,
    TopologyClass,
    validate_frame_composition_v3,
    validate_playback_runtime_v3_contract,
)
from compiler.realsas_compiler_core.types import QualificationError


RUNTIME_V3_BINARY_MAGIC = b"RSRT\x00\x03\x00\x00"
RUNTIME_V3_BINARY_VERSION = 3
RUNTIME_V3_BINARY_SCHEMA = "RealSaS.RuntimeBinary.rsr.v3"
RUNTIME_V3_PACKAGE_SCHEMA = "RealSaS.RuntimePackage.native_v3.v1"
RUNTIME_V3_COORDINATE_SYSTEM = "screen_2d.top_left.x_right.y_down.depth_camera_forward"
RUNTIME_V3_UNITS_PER_PIXEL = 1.0

FEATURE_RGBA8_TEXTURES = 1 << 0
FEATURE_BAKED_XYZ_FRAMES = 1 << 1
FEATURE_DYNAMIC_SLOT_DRAW_ORDER = 1 << 2
FEATURE_MULTI_VIEW_8 = 1 << 3
FEATURE_REFERENCE_DEPTH_RENDERER = 1 << 4
FEATURE_POSED_VERTEX_DEPTH = 1 << 5
FEATURE_FULL_SURFACE_AUTHORITY = 1 << 6
FEATURE_SLOT_ATTACHMENTS = 1 << 7
FEATURE_CLIP_INTERVALS = 1 << 8
FEATURE_APPEARANCE_PROVENANCE = 1 << 9
RUNTIME_V3_FEATURE_FLAGS = sum(1 << i for i in range(10))

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


@dataclass(frozen=True)
class RuntimeV3TexturePayload:
    view_id: str
    texture_path: str
    texture_sha256: str
    texture_crc32: int
    width: int
    height: int


@dataclass(frozen=True)
class RuntimeV3Frame:
    time_seconds: float
    posed_xyz_by_mesh: Mapping[str, tuple[tuple[float, float, float], ...]]
    composition_by_view: Mapping[str, RuntimeV3FrameComposition]


@dataclass(frozen=True)
class RuntimeV3Clip:
    clip_id: str
    display_name: str
    intent: str
    duration_seconds: float
    fps: float
    loop: bool
    frames: tuple[RuntimeV3Frame, ...]
    runtime_qualified: bool = True


class _Writer:
    def __init__(self):
        self.b = BytesIO()

    def raw(self, payload: bytes) -> None:
        self.b.write(payload)

    def u8(self, value: int) -> None:
        self.raw(struct.pack("<B", int(value)))

    def u32(self, value: int) -> None:
        self.raw(struct.pack("<I", int(value)))

    def f32(self, value: float) -> None:
        value = float(value)
        if not math.isfinite(value):
            raise QualificationError("RUNTIME_V3_NONFINITE_BINARY_VALUE")
        self.raw(struct.pack("<f", value))

    def string(self, value: str) -> None:
        payload = str(value).encode("utf-8")
        self.u32(len(payload))
        self.raw(payload)

    def finish(self) -> bytes:
        return self.b.getvalue()


def _sha_file(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _png_size(path: Path) -> tuple[int, int]:
    raw = path.read_bytes()[:24]
    if len(raw) < 24 or raw[:8] != b"\x89PNG\r\n\x1a\n" or raw[12:16] != b"IHDR":
        raise QualificationError(f"RUNTIME_V3_TEXTURE_NOT_PNG:{path}")
    return struct.unpack(">II", raw[16:24])


def _json_default(value):
    if isinstance(value, Enum):
        return value.value
    raise TypeError(type(value).__name__)


def _contract_dict(contract: RuntimeV3PlaybackContract) -> dict:
    return json.loads(json.dumps(asdict(contract), default=_json_default, sort_keys=True))


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(str(name).replace("\\", "/"), date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = (0o100644 & 0xFFFF) << 16
    info.create_system = 3
    return info


def _zip_write(archive: zipfile.ZipFile, name: str, payload: bytes) -> None:
    archive.writestr(_zip_info(name), payload, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)


def _validate_textures(
    textures: Iterable[RuntimeV3TexturePayload],
    *,
    texture_root: Path,
    required_views: tuple[str, ...],
) -> tuple[RuntimeV3TexturePayload, ...]:
    rows = tuple(sorted(textures, key=lambda x: x.view_id))
    by_view = {row.view_id: row for row in rows}
    if len(by_view) != len(rows) or set(by_view) != set(required_views):
        raise QualificationError("RUNTIME_V3_TEXTURE_VIEW_SET_MISMATCH")
    root = texture_root.resolve()
    for view_id in required_views:
        row = by_view[view_id]
        p = (root / row.texture_path).resolve()
        if root not in p.parents and p != root:
            raise QualificationError("RUNTIME_V3_TEXTURE_PATH_ESCAPES_ROOT")
        if not p.is_file():
            raise QualificationError(f"RUNTIME_V3_TEXTURE_MISSING:{row.texture_path}")
        if len(row.texture_sha256) != 64 or _sha_file(p) != row.texture_sha256:
            raise QualificationError(f"RUNTIME_V3_TEXTURE_SHA_MISMATCH:{view_id}")
        if (zlib.crc32(p.read_bytes()) & 0xFFFFFFFF) != int(row.texture_crc32):
            raise QualificationError(f"RUNTIME_V3_TEXTURE_CRC_MISMATCH:{view_id}")
        if _png_size(p) != (int(row.width), int(row.height)):
            raise QualificationError(f"RUNTIME_V3_TEXTURE_SIZE_MISMATCH:{view_id}")
    return tuple(by_view[v] for v in required_views)


def _validate_clips(
    contract: RuntimeV3PlaybackContract,
    clips: Iterable[RuntimeV3Clip],
    *,
    required_views: tuple[str, ...],
) -> tuple[RuntimeV3Clip, ...]:
    rows = tuple(clips)
    if not rows or len({c.clip_id for c in rows}) != len(rows):
        raise QualificationError("RUNTIME_V3_CLIP_SET_INVALID")
    mesh_by_id = {m.mesh_id: m for m in contract.meshes}
    expected_mesh_ids = set(mesh_by_id)
    for clip in rows:
        if clip.duration_seconds <= 0 or clip.fps <= 0 or not clip.frames:
            raise QualificationError(f"RUNTIME_V3_CLIP_INVALID:{clip.clip_id}")
        prior = -1.0
        for frame in clip.frames:
            if not math.isfinite(frame.time_seconds) or frame.time_seconds < prior:
                raise QualificationError(f"RUNTIME_V3_FRAME_TIME_INVALID:{clip.clip_id}")
            prior = float(frame.time_seconds)
            if set(frame.posed_xyz_by_mesh) != expected_mesh_ids:
                raise QualificationError("RUNTIME_V3_FRAME_MESH_SET_MISMATCH")
            if set(frame.composition_by_view) != set(required_views):
                raise QualificationError("RUNTIME_V3_FRAME_COMPOSITION_VIEW_SET_MISMATCH")
            for mesh_id, points in frame.posed_xyz_by_mesh.items():
                if len(points) != len(mesh_by_id[mesh_id].vertices):
                    raise QualificationError(f"RUNTIME_V3_FRAME_VERTEX_COUNT_MISMATCH:{mesh_id}")
                for point in points:
                    if len(point) != 3 or not all(math.isfinite(float(v)) for v in point):
                        raise QualificationError("RUNTIME_V3_FRAME_XYZ_INVALID")
            for view_id in required_views:
                composition = frame.composition_by_view[view_id]
                if composition.view_id != view_id:
                    raise QualificationError("RUNTIME_V3_FRAME_COMPOSITION_VIEW_ID_MISMATCH")
                validate_frame_composition_v3(contract, composition)
    return rows


def write_runtime_v3_binary(
    *,
    contract: RuntimeV3PlaybackContract,
    textures: tuple[RuntimeV3TexturePayload, ...],
    clips: tuple[RuntimeV3Clip, ...],
    source_binding_sha256: str,
    source_proof_bundle_hash: str,
    required_views: tuple[str, ...] = tuple(f"V{i}" for i in range(8)),
) -> bytes:
    if len(source_binding_sha256) != 64 or len(source_proof_bundle_hash) != 64:
        raise QualificationError("RUNTIME_V3_SOURCE_IDENTITIES_MUST_BE_SHA256")

    mesh_by_view = {
        view_id: tuple(m for m in contract.meshes if m.view_id == view_id)
        for view_id in required_views
    }
    slot_index = {slot.slot_id: i for i, slot in enumerate(contract.slots)}

    w = _Writer()
    w.raw(RUNTIME_V3_BINARY_MAGIC)
    w.u32(RUNTIME_V3_BINARY_VERSION)
    w.string(RUNTIME_V3_BINARY_SCHEMA)
    w.string(source_binding_sha256)
    w.string(source_proof_bundle_hash)
    w.string(RUNTIME_V3_COORDINATE_SYSTEM)
    w.f32(RUNTIME_V3_UNITS_PER_PIXEL)
    w.u32(RUNTIME_V3_FEATURE_FLAGS)
    w.string(contract.contract_hash)
    w.string(contract.raster.contract_hash)
    w.f32(contract.visibility.alpha_cutout_threshold)

    w.u32(len(contract.slots))
    for slot in contract.slots:
        w.string(slot.slot_id)
        w.string(slot.bone_id)
        w.u32(slot.setup_order)
        w.string(slot.default_attachment_id or "")

    texture_by_view = {t.view_id: t for t in textures}
    w.u32(len(required_views))
    for view_id in required_views:
        texture = texture_by_view[view_id]
        w.string(view_id)
        w.string(texture.texture_path.replace("\\", "/"))
        w.u32(texture.texture_crc32)
        w.u32(texture.width)
        w.u32(texture.height)
        meshes = mesh_by_view[view_id]
        w.u32(len(meshes))
        for mesh in meshes:
            w.string(mesh.mesh_id)
            w.u32(slot_index[mesh.slot_id])
            w.string(mesh.attachment_id)
            w.u8(_ATTACHMENT_KIND_CODE[mesh.attachment_kind])
            w.u8(_TOPOLOGY_CLASS_CODE[mesh.topology_class])
            w.u8(0)
            w.u8(0)
            w.u32(len(mesh.vertices))
            w.u32(len(mesh.triangles))
            for v in mesh.vertices:
                w.f32(v.x); w.f32(v.y); w.f32(v.z); w.f32(v.u); w.f32(v.v)
            for a, b, c in mesh.triangles:
                w.u32(a); w.u32(b); w.u32(c)

    w.u32(len(clips))
    for clip in clips:
        w.string(clip.clip_id)
        w.string(clip.display_name)
        w.string(clip.intent)
        w.f32(clip.duration_seconds)
        w.f32(clip.fps)
        w.u8(1 if clip.loop else 0)
        w.u8(1 if clip.runtime_qualified else 0)
        w.u8(0); w.u8(0)
        w.u32(len(clip.frames))
        for frame in clip.frames:
            w.f32(frame.time_seconds)
            for view_id in required_views:
                meshes = mesh_by_view[view_id]
                for mesh in meshes:
                    for x, y, z in frame.posed_xyz_by_mesh[mesh.mesh_id]:
                        w.f32(x); w.f32(y); w.f32(z)

                composition = frame.composition_by_view[view_id]
                w.u32(len(composition.draw_order_slot_ids))
                for slot_id in composition.draw_order_slot_ids:
                    w.u32(slot_index[slot_id])

                w.u32(len(contract.slots))
                for slot in contract.slots:
                    w.string(composition.active_attachment_by_slot[slot.slot_id] or "")

                w.u32(len(composition.clip_intervals))
                for interval in composition.clip_intervals:
                    w.string(interval.clip_attachment_id)
                    w.u32(slot_index[interval.start_slot_id])
                    w.u32(slot_index[interval.end_slot_id])
                    w.u8(1 if interval.inverse else 0)
                    w.u8(0); w.u8(0); w.u8(0)

    raw = w.finish()
    return raw + struct.pack("<I", zlib.crc32(raw) & 0xFFFFFFFF)


def materialize_runtime_v3_archive(
    *,
    out_path: str | Path,
    texture_root: str | Path,
    contract: RuntimeV3PlaybackContract,
    textures: Iterable[RuntimeV3TexturePayload],
    clips: Iterable[RuntimeV3Clip],
    source_product_state_hash: str,
    source_proof_bundle_hash: str,
    required_views: tuple[str, ...] = tuple(f"V{i}" for i in range(8)),
) -> dict:
    contract_hash = validate_playback_runtime_v3_contract(contract, required_view_ids=required_views)
    root = Path(texture_root).resolve()
    texture_rows = _validate_textures(textures, texture_root=root, required_views=required_views)
    clip_rows = _validate_clips(contract, clips, required_views=required_views)
    if len(source_product_state_hash) != 64 or len(source_proof_bundle_hash) != 64:
        raise QualificationError("RUNTIME_V3_SOURCE_IDENTITIES_MUST_BE_SHA256")

    contract_json = _contract_dict(contract)
    contract_bytes = json.dumps(contract_json, sort_keys=True, separators=(",", ":")).encode("utf-8")
    binding = {
        "schema": "RealSaS.RuntimeV3SourceBinding.v1",
        "source_product_state_hash": source_product_state_hash,
        "source_proof_bundle_hash": source_proof_bundle_hash,
        "playback_contract_hash": contract_hash,
        "reference_raster_contract_hash": contract.raster.contract_hash,
        "texture_hashes": {t.view_id: t.texture_sha256 for t in texture_rows},
        "clip_ids": [c.clip_id for c in clip_rows],
    }
    binding_bytes = json.dumps(binding, sort_keys=True, separators=(",", ":")).encode("utf-8")
    binding_sha = sha256(binding_bytes).hexdigest()
    binary = write_runtime_v3_binary(
        contract=contract,
        textures=texture_rows,
        clips=clip_rows,
        source_binding_sha256=binding_sha,
        source_proof_bundle_hash=source_proof_bundle_hash,
        required_views=required_views,
    )

    provenance_counts: dict[str, int] = {}
    provenance_face_counts: dict[str, int] = {}
    for patch in contract.appearance_patches:
        key = patch.provenance.value
        provenance_counts[key] = provenance_counts.get(key, 0) + 1
        provenance_face_counts[key] = provenance_face_counts.get(key, 0) + len(patch.face_indices)

    manifest = {
        "schema_version": RUNTIME_V3_PACKAGE_SCHEMA,
        "binary": {
            "path": "runtime/realsas_runtime.rsr",
            "schema": RUNTIME_V3_BINARY_SCHEMA,
            "sha256": sha256(binary).hexdigest(),
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
            "semantic_slot_draw_order": True,
            "clip_intervals": True,
            "visibility_by_face_deletion": False,
        },
        "reference_raster": {
            **asdict(contract.raster),
            "contract_hash": contract.raster.contract_hash,
        },
        "appearance_policy": {
            "completion_allowed": bool(contract.allow_completion),
            "patch_counts_by_provenance": provenance_counts,
            "face_counts_by_provenance": provenance_face_counts,
        },
        "views": [
            {
                "view_id": t.view_id,
                "texture_path": t.texture_path,
                "texture_sha256": t.texture_sha256,
                "texture_crc32": f"{int(t.texture_crc32):08x}",
                "width": t.width,
                "height": t.height,
            }
            for t in texture_rows
        ],
        "clips": [
            {
                "clip_id": c.clip_id,
                "duration_seconds": c.duration_seconds,
                "fps": c.fps,
                "loop": c.loop,
                "runtime_qualified": c.runtime_qualified,
                "frame_count": len(c.frames),
            }
            for c in clip_rows
        ],
        "product_goal": "AUTOMATIC_8_DIRECTION_SPINE_CLASS_PUPPET_PLAYBACK",
        "subject_specific_pipeline_policy": False,
        "founder_visual_pass_claimed": False,
    }

    target = Path(out_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        _zip_write(z, "manifest.json", (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        _zip_write(z, "source_binding.json", binding_bytes)
        _zip_write(z, "runtime/playback_contract.json", contract_bytes)
        _zip_write(z, "runtime/realsas_runtime.rsr", binary)
        for texture in texture_rows:
            _zip_write(z, texture.texture_path, (root / texture.texture_path).read_bytes())

    return {
        "archive_path": str(target),
        "archive_sha256": _sha_file(target),
        "runtime_binary_sha256": sha256(binary).hexdigest(),
        "source_binding_sha256": binding_sha,
        "playback_contract_hash": contract_hash,
        "reference_raster_contract_hash": contract.raster.contract_hash,
        "view_count": len(texture_rows),
        "clip_count": len(clip_rows),
    }
