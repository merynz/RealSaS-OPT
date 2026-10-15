from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from tools.corpus.gltf_bind_scene_v1 import SourceGLTF, Y_TO_Z
from tools.corpus.fit8_gltf_static_carrier_v1 import (
    _canonical_skin_signature,
    build_gltf_static_carrier_v1,
)


SCHEMA_VERSION = "RealSaS.FIT8GLTFDynamicPose.v1"
SUPPORTED_INTERPOLATION = "LINEAR"
SUPPORTED_TARGET_PATHS = {"translation", "rotation", "scale"}


@dataclass(frozen=True)
class PrimitiveMotionRecordV1:
    render_vertex_offset: int
    node_index: int
    skin_index: int | None
    local_positions: np.ndarray
    normalized_skin_rows: tuple[dict[int, float], ...]
    joint_node_ids: tuple[int, ...]
    inverse_bind_matrices: np.ndarray | None


@dataclass(frozen=True)
class GLTFDynamicSourceV1:
    source_path: Path
    source: SourceGLTF
    carrier_manifest: dict
    carrier_arrays: dict[str, np.ndarray]
    primitive_records: tuple[PrimitiveMotionRecordV1, ...]
    render_groups_by_geometric_vertex: tuple[tuple[int, ...], ...]

    @property
    def render_vertex_count(self) -> int:
        return int(len(self.carrier_arrays["render_to_geometric"]))

    @property
    def geometric_vertex_count(self) -> int:
        return int(len(self.carrier_arrays["vertices"]))


@dataclass(frozen=True)
class DynamicPoseResultV1:
    clip_name: str
    clip_index: int
    authored_time: float
    normalized_time: float
    raw_render_xyz: np.ndarray
    geometric_xyz: np.ndarray
    quotient_worst_separation: float


def _inverse_bind_matrices(source: SourceGLTF, skin_index: int) -> tuple[tuple[int, ...], np.ndarray]:
    skin = source.doc["skins"][skin_index]
    joints = tuple(int(value) for value in skin["joints"])
    if "inverseBindMatrices" not in skin:
        matrices = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], len(joints), axis=0)
    else:
        raw = np.asarray(source.accessor(skin["inverseBindMatrices"]), dtype=np.float64)
        if raw.shape != (len(joints), 16):
            raise ValueError("FIT8_GLTF_DYNAMIC_IBM_SHAPE_INVALID")
        # glTF MAT4 accessor elements are column-major.
        matrices = raw.reshape(-1, 4, 4).transpose(0, 2, 1).copy()
    if not np.isfinite(matrices).all():
        raise ValueError("FIT8_GLTF_DYNAMIC_IBM_NONFINITE")
    return joints, matrices


def build_gltf_dynamic_source_v1(source_path: Path) -> GLTFDynamicSourceV1:
    source_path = Path(source_path)
    source = SourceGLTF(source_path)
    carrier_manifest, carrier_arrays = build_gltf_static_carrier_v1(source_path)
    mapping = np.asarray(carrier_arrays["render_to_geometric"], dtype=np.int64)
    records: list[PrimitiveMotionRecordV1] = []
    render_offset = 0

    for node_index, node in enumerate(source.doc.get("nodes", [])):
        if "mesh" not in node:
            continue
        mesh = source.doc["meshes"][node["mesh"]]
        skin_index = node.get("skin")
        for primitive in mesh.get("primitives", []):
            if primitive.get("mode", 4) != 4:
                raise ValueError("FIT8_GLTF_DYNAMIC_NON_TRIANGLE_PRIMITIVE")
            position = source.accessor(primitive["attributes"]["POSITION"]).astype(np.float64)
            morph_weights = node.get("weights", mesh.get("weights", []))
            for morph_weight, target in zip(morph_weights, primitive.get("targets", [])):
                if "POSITION" in target:
                    position += float(morph_weight) * source.accessor(target["POSITION"])
            _, normalized_skin = _canonical_skin_signature(
                source,
                primitive,
                skin_index,
                len(position),
            )
            if skin_index is None:
                joints: tuple[int, ...] = ()
                ibm = None
            else:
                joints, ibm = _inverse_bind_matrices(source, int(skin_index))
                if any(not row for row in normalized_skin):
                    raise ValueError("FIT8_GLTF_DYNAMIC_ZERO_WEIGHT_SKIN_VERTEX")
            records.append(
                PrimitiveMotionRecordV1(
                    render_vertex_offset=render_offset,
                    node_index=int(node_index),
                    skin_index=None if skin_index is None else int(skin_index),
                    local_positions=np.asarray(position, dtype=np.float64),
                    normalized_skin_rows=tuple(dict(row) for row in normalized_skin),
                    joint_node_ids=joints,
                    inverse_bind_matrices=ibm,
                )
            )
            render_offset += len(position)

    if render_offset != len(mapping):
        raise ValueError("FIT8_GLTF_DYNAMIC_RENDER_ORDER_DRIFT")
    groups: list[list[int]] = [[] for _ in range(len(carrier_arrays["vertices"]))]
    for render_vertex_id, geometric_vertex_id in enumerate(mapping.tolist()):
        groups[int(geometric_vertex_id)].append(int(render_vertex_id))
    if any(not row for row in groups):
        raise ValueError("FIT8_GLTF_DYNAMIC_EMPTY_EQUIVALENCE_GROUP")
    return GLTFDynamicSourceV1(
        source_path=source_path,
        source=source,
        carrier_manifest=carrier_manifest,
        carrier_arrays=carrier_arrays,
        primitive_records=tuple(records),
        render_groups_by_geometric_vertex=tuple(tuple(row) for row in groups),
    )


def animation_time_range_v1(dynamic_source: GLTFDynamicSourceV1, clip_index: int) -> tuple[float, float]:
    source = dynamic_source.source
    animations = source.doc.get("animations", [])
    if not 0 <= int(clip_index) < len(animations):
        raise ValueError("FIT8_GLTF_DYNAMIC_CLIP_INDEX_INVALID")
    starts: list[float] = []
    ends: list[float] = []
    for sampler in animations[int(clip_index)].get("samplers", []):
        if sampler.get("interpolation", SUPPORTED_INTERPOLATION) != SUPPORTED_INTERPOLATION:
            raise ValueError("FIT8_GLTF_DYNAMIC_INTERPOLATION_UNSUPPORTED")
        time = np.asarray(source.accessor(sampler["input"]), dtype=np.float64).reshape(-1)
        if len(time):
            if not np.all(np.diff(time) >= 0.0):
                raise ValueError("FIT8_GLTF_DYNAMIC_SAMPLER_TIME_NOT_MONOTONIC")
            starts.append(float(time[0]))
            ends.append(float(time[-1]))
    return (min(starts), max(ends)) if starts else (0.0, 0.0)


def _sample_channel_value(source: SourceGLTF, sampler: dict[str, Any], path: str, time: float) -> np.ndarray:
    if sampler.get("interpolation", SUPPORTED_INTERPOLATION) != SUPPORTED_INTERPOLATION:
        raise ValueError("FIT8_GLTF_DYNAMIC_INTERPOLATION_UNSUPPORTED")
    times = np.asarray(source.accessor(sampler["input"]), dtype=np.float64).reshape(-1)
    values = np.asarray(source.accessor(sampler["output"]), dtype=np.float64)
    if not len(times) or values.shape[0] != len(times):
        raise ValueError("FIT8_GLTF_DYNAMIC_SAMPLER_SHAPE_INVALID")
    t = float(np.clip(float(time), float(times[0]), float(times[-1])))
    hi = int(np.searchsorted(times, t, side="right"))
    if hi <= 0:
        value = values[0]
    elif hi >= len(times):
        value = values[-1]
    else:
        lo = hi - 1
        duration = float(times[hi] - times[lo])
        alpha = 0.0 if duration <= 0.0 else float((t - times[lo]) / duration)
        if path == "rotation":
            rotation = Rotation.from_quat(np.stack((values[lo], values[hi]), axis=0))
            value = Slerp([0.0, 1.0], rotation)([alpha]).as_quat()[0]
        else:
            value = (1.0 - alpha) * values[lo] + alpha * values[hi]
    if path == "rotation":
        norm = float(np.linalg.norm(value))
        if norm <= 0.0:
            raise ValueError("FIT8_GLTF_DYNAMIC_ROTATION_ZERO")
        value = value / norm
    return np.asarray(value, dtype=np.float64)


def _animation_overrides(dynamic_source: GLTFDynamicSourceV1, clip_index: int, time: float) -> dict[int, dict[str, np.ndarray]]:
    source = dynamic_source.source
    animation = source.doc["animations"][int(clip_index)]
    overrides: dict[int, dict[str, np.ndarray]] = {}
    for channel in animation.get("channels", []):
        target = channel["target"]
        path = str(target["path"])
        if path not in SUPPORTED_TARGET_PATHS:
            raise ValueError(f"FIT8_GLTF_DYNAMIC_TARGET_PATH_UNSUPPORTED:{path}")
        node_index = int(target["node"])
        if "matrix" in source.doc["nodes"][node_index]:
            raise ValueError("FIT8_GLTF_DYNAMIC_TARGETS_MATRIX_NODE")
        sampler = animation["samplers"][int(channel["sampler"])]
        overrides.setdefault(node_index, {})[path] = _sample_channel_value(source, sampler, path, time)
    return overrides


def _local_matrix(source: SourceGLTF, node_index: int, overrides: dict[int, dict[str, np.ndarray]]) -> np.ndarray:
    row = source.doc["nodes"][node_index]
    if "matrix" in row:
        if node_index in overrides:
            raise ValueError("FIT8_GLTF_DYNAMIC_TARGETS_MATRIX_NODE")
        return np.asarray(row["matrix"], dtype=np.float64).reshape(4, 4).T
    translation = np.asarray(row.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64)
    rotation = np.asarray(row.get("rotation", [0.0, 0.0, 0.0, 1.0]), dtype=np.float64)
    scale = np.asarray(row.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
    if node_index in overrides:
        values = overrides[node_index]
        translation = np.asarray(values.get("translation", translation), dtype=np.float64)
        rotation = np.asarray(values.get("rotation", rotation), dtype=np.float64)
        scale = np.asarray(values.get("scale", scale), dtype=np.float64)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_quat(rotation).as_matrix()
    matrix[:3, :3] *= scale[None, :]
    matrix[:3, 3] = translation
    return matrix


def _world_matrices(dynamic_source: GLTFDynamicSourceV1, overrides: dict[int, dict[str, np.ndarray]]) -> tuple[np.ndarray, ...]:
    source = dynamic_source.source
    cache: dict[int, np.ndarray] = {}

    def world(node_index: int, stack: tuple[int, ...] = ()) -> np.ndarray:
        if node_index in cache:
            return cache[node_index]
        if node_index in stack:
            raise ValueError("FIT8_GLTF_DYNAMIC_NODE_CYCLE")
        matrix = _local_matrix(source, node_index, overrides)
        if node_index in source.parents:
            matrix = world(int(source.parents[node_index]), (*stack, node_index)) @ matrix
        cache[node_index] = matrix
        return matrix

    return tuple(world(index) for index in range(len(source.doc.get("nodes", []))))


def evaluate_authored_pose_v1(
    dynamic_source: GLTFDynamicSourceV1,
    *,
    clip_index: int,
    normalized_time: float,
) -> DynamicPoseResultV1:
    if not math.isfinite(float(normalized_time)) or not 0.0 <= float(normalized_time) <= 1.0:
        raise ValueError("FIT8_GLTF_DYNAMIC_NORMALIZED_TIME_INVALID")
    lo, hi = animation_time_range_v1(dynamic_source, clip_index)
    authored_time = float(lo + float(normalized_time) * (hi - lo))
    overrides = _animation_overrides(dynamic_source, clip_index, authored_time)
    worlds = _world_matrices(dynamic_source, overrides)
    raw = np.empty((dynamic_source.render_vertex_count, 3), dtype=np.float64)

    for record in dynamic_source.primitive_records:
        position = record.local_positions
        count = len(position)
        offset = int(record.render_vertex_offset)
        if record.skin_index is None:
            matrix = worlds[int(record.node_index)]
            posed = position @ matrix[:3, :3].T + matrix[:3, 3]
        else:
            assert record.inverse_bind_matrices is not None
            joint_to_ibm = {
                int(joint): record.inverse_bind_matrices[index]
                for index, joint in enumerate(record.joint_node_ids)
            }
            homogeneous = np.concatenate((position, np.ones((count, 1), dtype=np.float64)), axis=1)
            posed = np.zeros((count, 3), dtype=np.float64)
            for vertex_id, weights in enumerate(record.normalized_skin_rows):
                if not weights:
                    raise ValueError("FIT8_GLTF_DYNAMIC_ZERO_WEIGHT_SKIN_VERTEX")
                value = np.zeros(4, dtype=np.float64)
                for joint_node_id, weight in weights.items():
                    value += float(weight) * (
                        worlds[int(joint_node_id)]
                        @ joint_to_ibm[int(joint_node_id)]
                        @ homogeneous[vertex_id]
                    )
                posed[vertex_id] = value[:3]
        raw[offset : offset + count] = posed @ Y_TO_Z.T

    geometric = np.empty((dynamic_source.geometric_vertex_count, 3), dtype=np.float64)
    worst = 0.0
    for geometric_vertex_id, render_ids in enumerate(dynamic_source.render_groups_by_geometric_vertex):
        points = raw[list(render_ids)]
        geometric[geometric_vertex_id] = points[0]
        if len(render_ids) > 1:
            separation = float(np.max(np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)))
            worst = max(worst, separation)
    if not np.isfinite(raw).all() or not np.isfinite(geometric).all():
        raise ValueError("FIT8_GLTF_DYNAMIC_POSE_NONFINITE")
    animation = dynamic_source.source.doc["animations"][int(clip_index)]
    return DynamicPoseResultV1(
        clip_name=str(animation.get("name", clip_index)),
        clip_index=int(clip_index),
        authored_time=authored_time,
        normalized_time=float(normalized_time),
        raw_render_xyz=raw,
        geometric_xyz=geometric,
        quotient_worst_separation=worst,
    )


__all__ = [
    "DynamicPoseResultV1",
    "GLTFDynamicSourceV1",
    "PrimitiveMotionRecordV1",
    "SCHEMA_VERSION",
    "animation_time_range_v1",
    "build_gltf_dynamic_source_v1",
    "evaluate_authored_pose_v1",
]
