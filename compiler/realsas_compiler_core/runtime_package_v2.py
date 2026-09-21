from __future__ import annotations

"""Compact dependency-free RealSaS V2 runtime package writer/reader."""

from collections import OrderedDict
import hashlib
import io
from pathlib import Path
import struct

import numpy as np
from PIL import Image

from .runtime_authority_v2 import RuntimeProjectionV2IR
from .types import QualificationError

MAGIC = b"RSASV2R1"


def _entry_bytes(entries: OrderedDict[str, bytes]) -> bytes:
    out = io.BytesIO()
    out.write(MAGIC)
    out.write(struct.pack("<I", len(entries)))
    for name, payload in entries.items():
        encoded = name.encode("utf-8")
        if not encoded or len(encoded) > 65535:
            raise QualificationError("RSS_V2_ENTRY_NAME_INVALID")
        out.write(struct.pack("<H", len(encoded)))
        out.write(encoded)
        out.write(struct.pack("<Q", len(payload)))
        out.write(payload)
    return out.getvalue()


def write_rss_v2(path: Path, entries: OrderedDict[str, bytes]) -> dict:
    raw = _entry_bytes(entries)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return {
        "archive_sha256": hashlib.sha256(raw).hexdigest(),
        "archive_bytes": len(raw),
        "entry_names": tuple(entries.keys()),
        "package_format": "REALSAS_RSS_V2_UNCOMPRESSED_CONTAINER",
    }


def read_rss_v2(path: Path) -> OrderedDict[str, bytes]:
    raw = path.read_bytes()
    offset = 0
    if raw[: len(MAGIC)] != MAGIC:
        raise QualificationError("RSS_V2_MAGIC_INVALID")
    offset += len(MAGIC)
    if offset + 4 > len(raw):
        raise QualificationError("RSS_V2_TRUNCATED")
    count = struct.unpack_from("<I", raw, offset)[0]
    offset += 4
    entries: OrderedDict[str, bytes] = OrderedDict()
    for _ in range(count):
        if offset + 2 > len(raw):
            raise QualificationError("RSS_V2_TRUNCATED_NAME")
        name_len = struct.unpack_from("<H", raw, offset)[0]
        offset += 2
        if offset + name_len + 8 > len(raw):
            raise QualificationError("RSS_V2_TRUNCATED_ENTRY_HEADER")
        name = raw[offset : offset + name_len].decode("utf-8")
        offset += name_len
        size = struct.unpack_from("<Q", raw, offset)[0]
        offset += 8
        if offset + size > len(raw):
            raise QualificationError("RSS_V2_TRUNCATED_ENTRY_DATA")
        if name in entries:
            raise QualificationError("RSS_V2_DUPLICATE_ENTRY")
        entries[name] = raw[offset : offset + size]
        offset += size
    if offset != len(raw):
        raise QualificationError("RSS_V2_TRAILING_BYTES")
    return entries


def _mesh_payload(arrays: dict) -> bytes:
    vertices = np.asarray(arrays["vertices"], dtype="<f8")
    faces = np.asarray(arrays["faces"], dtype="<u4")
    uv = np.asarray(arrays["face_uv"], dtype="<f8")
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise QualificationError("RSS_V2_VERTEX_SHAPE_INVALID")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise QualificationError("RSS_V2_FACE_SHAPE_INVALID")
    if uv.shape != (len(faces), 3, 2):
        raise QualificationError("RSS_V2_UV_SHAPE_INVALID")
    return (
        struct.pack("<II", len(vertices), len(faces))
        + vertices.tobytes(order="C")
        + faces.tobytes(order="C")
        + uv.tobytes(order="C")
    )


def _camera_payload(projection: RuntimeProjectionV2IR) -> bytes:
    out = io.BytesIO()
    out.write(struct.pack("<I", len(projection.views)))
    for view in sorted(projection.views, key=lambda row: row.view_index):
        camera = dict(view.camera)
        values = (
            tuple(map(float, camera["origin"]))
            + tuple(map(float, camera["right"]))
            + tuple(map(float, camera["screen_up"]))
            + tuple(map(float, camera["forward"]))
            + (float(camera["half_extent"]),)
        )
        if len(values) != 13:
            raise QualificationError("RSS_V2_CAMERA_VECTOR_INVALID")
        out.write(struct.pack("<13dI", *values, int(camera["resolution"])))
    return out.getvalue()


def _texture_payload(projection: RuntimeProjectionV2IR) -> bytes:
    images = []
    shape = None
    for view in sorted(projection.views, key=lambda row: row.view_index):
        path = Path(view.texture_path)
        image = np.asarray(Image.open(path).convert("RGBA"), dtype=np.uint8)
        if shape is None:
            shape = image.shape
        if image.shape != shape:
            raise QualificationError("RSS_V2_TEXTURE_DIMENSION_DRIFT")
        images.append(image)
    stacked = np.stack(images, axis=0)
    return (
        struct.pack("<III", stacked.shape[0], stacked.shape[1], stacked.shape[2])
        + stacked.tobytes(order="C")
    )


def _provenance_payload(projection: RuntimeProjectionV2IR) -> bytes:
    path = Path(projection.provenance_npz_path)
    with np.load(path, allow_pickle=False) as data:
        if "provenance" not in data.files:
            raise QualificationError("RSS_V2_PROVENANCE_ARRAY_MISSING")
        value = np.asarray(data["provenance"], dtype=np.uint8)
    if value.ndim != 3 or value.shape[0] != 8:
        raise QualificationError("RSS_V2_PROVENANCE_SHAPE_INVALID")
    return (
        struct.pack("<III", value.shape[0], value.shape[1], value.shape[2])
        + value.tobytes(order="C")
    )


def _clip_payload(times: np.ndarray, positions: np.ndarray) -> bytes:
    times = np.asarray(times, dtype="<f8")
    positions = np.asarray(positions, dtype="<f8")
    if times.ndim != 1 or positions.ndim != 3 or positions.shape[0] != len(times) or positions.shape[2] != 3:
        raise QualificationError("RSS_V2_CLIP_ARRAY_SHAPE_INVALID")
    return (
        struct.pack("<II", len(times), positions.shape[1])
        + times.tobytes(order="C")
        + positions.tobytes(order="C")
    )


def build_rss_v2_entries(projection: RuntimeProjectionV2IR) -> OrderedDict[str, bytes]:
    with np.load(projection.projection_npz_path, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]).copy() for name in data.files}
    required = {"vertices", "faces", "face_uv"}
    if not required.issubset(arrays):
        raise QualificationError("RSS_V2_PROJECTION_ARRAYS_MISSING")

    entries: OrderedDict[str, bytes] = OrderedDict()
    manifest = [
        "schema=RealSaS.RuntimePackage.v2",
        f"projection_hash={projection.projection_hash}",
        f"mesh_hash={projection.mesh_binding_hash}",
        f"appearance_asset_hash={projection.appearance_asset_binding_hash}",
        f"dynamic_motion_hash={projection.dynamic_motion_binding_hash}",
        f"visibility_contract_hash={projection.visibility_contract_hash}",
        "playback_sampling_contract=SEALED_FRAME_INDEX_ONLY",
        "host_interpolation_authorized=0",
        "geometry_uv_position_precision=IEEE754_FLOAT64",
        f"clip_count={len(projection.clips)}",
        f"view_count={len(projection.views)}",
    ]
    entries["mesh.bin"] = _mesh_payload(arrays)
    entries["cameras.bin"] = _camera_payload(projection)
    entries["textures.bin"] = _texture_payload(projection)
    entries["provenance.bin"] = _provenance_payload(projection)

    for index, clip in enumerate(projection.clips):
        times_key = f"{clip.array_prefix}_times"
        positions_key = f"{clip.array_prefix}_positions"
        if times_key not in arrays or positions_key not in arrays:
            raise QualificationError("RSS_V2_CLIP_ARRAY_MISSING")
        entry_name = f"clip_{index}.bin"
        entries[entry_name] = _clip_payload(
            arrays[times_key], arrays[positions_key]
        )
        manifest.extend(
            [
                f"clip.{index}.id={clip.clip_id}",
                f"clip.{index}.entry={entry_name}",
                f"clip.{index}.frame_count={clip.frame_count}",
                f"clip.{index}.duration_seconds={clip.duration_seconds:.17g}",
                f"clip.{index}.loop={1 if clip.loop else 0}",
            ]
        )
    entries["manifest.txt"] = ("\n".join(manifest) + "\n").encode("utf-8")
    return entries
