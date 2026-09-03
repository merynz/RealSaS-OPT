from __future__ import annotations

"""Compact engine-neutral handoff for MotionProof runtime frames.

The dynamic compiler already executes every authored clip frame required by its
bounded temporal proof.  This contract serializes those exact frames once so
runtime package export never replays deformation and never copies millions of
JSON coordinate scalars into several artifacts.
"""

from base64 import b64decode, b64encode
from hashlib import sha256
from io import BytesIO
import struct
import zlib
from typing import Any, Iterable

SCHEMA_VERSION = "realSaS.RuntimeDeployBake.v2"
BINARY_MAGIC = b"RSBK\x00\x02\x00\x00"
BINARY_VERSION = 2


class _Writer:
    def __init__(self) -> None:
        self.buffer = BytesIO()

    def raw(self, value: bytes) -> None:
        self.buffer.write(value)

    def u8(self, value: int) -> None:
        self.raw(struct.pack("<B", int(value)))

    def u32(self, value: int) -> None:
        self.raw(struct.pack("<I", int(value)))

    def f32(self, value: float) -> None:
        self.raw(struct.pack("<f", float(value)))

    def string(self, value: str) -> None:
        encoded = str(value).encode("utf-8")
        self.u32(len(encoded))
        self.raw(encoded)

    def finish(self) -> bytes:
        return self.buffer.getvalue()


class _Reader:
    def __init__(self, payload: bytes) -> None:
        self.buffer = BytesIO(payload)

    def raw(self, count: int) -> bytes:
        value = self.buffer.read(count)
        if len(value) != count:
            raise RuntimeError("Truncated runtime deploy bake payload")
        return value

    def u8(self) -> int:
        return struct.unpack("<B", self.raw(1))[0]

    def u32(self) -> int:
        return struct.unpack("<I", self.raw(4))[0]

    def f32(self) -> float:
        return struct.unpack("<f", self.raw(4))[0]

    def string(self) -> str:
        return self.raw(self.u32()).decode("utf-8")

    def at_end(self) -> bool:
        return self.buffer.read(1) == b""


def _frame_fields(frame: Any) -> tuple[dict[str, Any], dict[str, Any], str]:
    if isinstance(frame, dict):
        meshes = dict(frame.get("mesh_vertices_by_id") or {})
        render = dict(frame.get("render_order_by_view") or {})
        semantic = str(frame.get("semantic_sha256") or "")
    else:
        meshes = dict(getattr(frame, "mesh_vertices_by_id", {}) or {})
        render = dict(getattr(frame, "render_order_by_view", {}) or {})
        semantic = str(getattr(frame, "semantic_sha256", "") or "")
    return meshes, render, semantic


def encode_runtime_deploy_bake(
    *,
    clip_id: str,
    duration_seconds: float,
    fps: float,
    loop: bool,
    sample_times_seconds: Iterable[float],
    frames: Iterable[Any],
    evaluator_semantic_version: str,
    sampling_policy: str,
) -> dict[str, Any]:
    frame_rows = list(frames)
    times = [float(value) for value in sample_times_seconds]
    if not frame_rows or len(frame_rows) != len(times):
        raise ValueError("Runtime deploy bake requires one time for every frame")

    first_meshes, first_render, _ = _frame_fields(frame_rows[0])
    mesh_ids = tuple(sorted(str(value) for value in first_meshes))
    if not mesh_ids:
        raise ValueError("Runtime deploy bake has no meshes")
    vertex_counts = {
        mesh_id: len(list(first_meshes.get(mesh_id) or ())) for mesh_id in mesh_ids
    }
    view_ids = tuple(sorted({
        str(view_id)
        for frame in frame_rows
        for view_id in _frame_fields(frame)[1]
    } | {str(view_id) for view_id in first_render}))
    mesh_index = {mesh_id: index for index, mesh_id in enumerate(mesh_ids)}

    writer = _Writer()
    writer.raw(BINARY_MAGIC)
    writer.u32(BINARY_VERSION)
    writer.string(str(clip_id))
    writer.string(str(evaluator_semantic_version))
    writer.string(str(sampling_policy))
    writer.f32(float(duration_seconds))
    writer.f32(float(fps))
    writer.u8(1 if loop else 0)
    writer.u8(0); writer.u8(0); writer.u8(0)
    writer.u32(len(mesh_ids))
    for mesh_id in mesh_ids:
        writer.string(mesh_id)
        writer.u32(vertex_counts[mesh_id])
    writer.u32(len(view_ids))
    for view_id in view_ids:
        writer.string(view_id)
    writer.u32(len(frame_rows))

    for time_seconds, frame in zip(times, frame_rows):
        meshes, render, semantic = _frame_fields(frame)
        if tuple(sorted(str(value) for value in meshes)) != mesh_ids:
            raise ValueError("Runtime deploy bake mesh identity drift")
        writer.f32(time_seconds)
        try:
            semantic_bytes = bytes.fromhex(semantic)
        except ValueError as exc:
            raise ValueError("Runtime deploy bake frame hash is not hexadecimal") from exc
        if len(semantic_bytes) != 32:
            raise ValueError("Runtime deploy bake frame hash must be SHA-256")
        writer.raw(semantic_bytes)
        for mesh_id in mesh_ids:
            vertices = list(meshes.get(mesh_id) or ())
            if len(vertices) != vertex_counts[mesh_id]:
                raise ValueError(f"Runtime deploy bake vertex count drift: {mesh_id}")
            for point in vertices:
                writer.f32(float(point[0]))
                writer.f32(float(point[1]))
        for view_id in view_ids:
            order = [
                mesh_index[str(mesh_id)]
                for mesh_id in list(render.get(view_id) or ())
                if str(mesh_id) in mesh_index
            ]
            writer.u32(len(order))
            for index in order:
                writer.u32(index)

    raw = writer.finish()
    compressed = zlib.compress(raw, level=6)
    return {
        "schema_version": SCHEMA_VERSION,
        "binary_version": BINARY_VERSION,
        "clip_id": str(clip_id),
        "duration_seconds": float(duration_seconds),
        "fps": float(fps),
        "loop": bool(loop),
        "sampling_policy": str(sampling_policy),
        "evaluator_semantic_version": str(evaluator_semantic_version),
        "coordinate_encoding": "source_pixel_float32_little_endian",
        "compression": "zlib_level6_base64",
        "frame_count": len(frame_rows),
        "mesh_count": len(mesh_ids),
        "view_count": len(view_ids),
        "uncompressed_byte_length": len(raw),
        "compressed_byte_length": len(compressed),
        "payload_sha256": sha256(raw).hexdigest(),
        "payload_zlib_base64": b64encode(compressed).decode("ascii"),
    }


def decode_runtime_deploy_bake(value: dict[str, Any]) -> dict[str, Any]:
    bake = dict(value or {})
    if str(bake.get("schema_version") or "") != SCHEMA_VERSION:
        raise RuntimeError("Unsupported runtime deploy bake schema")
    if str(bake.get("compression") or "") != "zlib_level6_base64":
        raise RuntimeError("Unsupported runtime deploy bake compression")
    try:
        compressed = b64decode(str(bake.get("payload_zlib_base64") or ""), validate=True)
        raw = zlib.decompress(compressed)
    except Exception as exc:
        raise RuntimeError("Runtime deploy bake payload cannot be decoded") from exc
    if len(compressed) != int(bake.get("compressed_byte_length", -1)):
        raise RuntimeError("Runtime deploy bake compressed length mismatch")
    if len(raw) != int(bake.get("uncompressed_byte_length", -1)):
        raise RuntimeError("Runtime deploy bake raw length mismatch")
    if sha256(raw).hexdigest() != str(bake.get("payload_sha256") or ""):
        raise RuntimeError("Runtime deploy bake payload hash mismatch")

    reader = _Reader(raw)
    if reader.raw(len(BINARY_MAGIC)) != BINARY_MAGIC:
        raise RuntimeError("Runtime deploy bake magic mismatch")
    if reader.u32() != BINARY_VERSION:
        raise RuntimeError("Runtime deploy bake binary version mismatch")
    clip_id = reader.string()
    evaluator_semantic_version = reader.string()
    sampling_policy = reader.string()
    duration = reader.f32()
    fps = reader.f32()
    loop = bool(reader.u8())
    reader.raw(3)
    mesh_ids: list[str] = []
    vertex_counts: list[int] = []
    for _ in range(reader.u32()):
        mesh_ids.append(reader.string())
        vertex_counts.append(reader.u32())
    view_ids = [reader.string() for _ in range(reader.u32())]
    frames: list[dict[str, Any]] = []
    for _ in range(reader.u32()):
        time_seconds = reader.f32()
        semantic = reader.raw(32).hex()
        meshes: dict[str, list[list[float]]] = {}
        for mesh_id, vertex_count in zip(mesh_ids, vertex_counts):
            meshes[mesh_id] = [[reader.f32(), reader.f32()] for _ in range(vertex_count)]
        render: dict[str, list[str]] = {}
        for view_id in view_ids:
            indices = [reader.u32() for _ in range(reader.u32())]
            if any(index >= len(mesh_ids) for index in indices):
                raise RuntimeError("Runtime deploy bake draw-order index is out of range")
            render[view_id] = [mesh_ids[index] for index in indices]
        frames.append({
            "time_seconds": float(time_seconds),
            "semantic_sha256": semantic,
            "mesh_vertices_by_id": meshes,
            "render_order_by_view": render,
        })
    if not reader.at_end():
        raise RuntimeError("Runtime deploy bake has trailing bytes")
    if clip_id != str(bake.get("clip_id") or ""):
        raise RuntimeError("Runtime deploy bake clip identity mismatch")
    if len(frames) != int(bake.get("frame_count", -1)):
        raise RuntimeError("Runtime deploy bake frame count mismatch")
    return {
        "schema_version": SCHEMA_VERSION,
        "clip_id": clip_id,
        "duration_seconds": float(duration),
        "fps": float(fps),
        "loop": loop,
        "sampling_policy": sampling_policy,
        "evaluator_semantic_version": evaluator_semantic_version,
        "frames": frames,
    }
