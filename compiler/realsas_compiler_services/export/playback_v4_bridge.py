from __future__ import annotations

"""D1 canonical motion -> Runtime-v4 shared-frame bridge."""

from dataclasses import dataclass
from hashlib import sha256
from typing import Mapping, Sequence
import math

import numpy as np

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.motion_3d_adapter_v1 import D1CompiledClipV1
from compiler.realsas_compiler_core.motion_3d_v1 import apply_lbs_matrix_v1
from compiler.realsas_compiler_core.playback_runtime_v3 import RuntimeV3FrameComposition
from compiler.realsas_compiler_core.playback_runtime_v4 import RuntimeV4Clip, RuntimeV4Frame
from compiler.realsas_compiler_core.types import QualificationError

PLAYBACK_V4_D1_BRIDGE_SCHEMA = "RealSaS.PlaybackV4D1Bridge.v1"


@dataclass(frozen=True)
class PlaybackV4D1BridgeReport:
    clip_id: str
    frame_count: int
    vertex_count: int
    joint_count: int
    view_count: int
    nominal_fps: float
    max_weight_row_sum_error: float
    canonical_pose_bytes_per_frame: int
    avoided_v3_view_multiplier: int
    runtime_qualified: bool
    bridge_hash: str
    schema_version: str = PLAYBACK_V4_D1_BRIDGE_SCHEMA


def _array_sha256(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(value)
    h = sha256()
    h.update(str(arr.dtype).encode("ascii")); h.update(b"|")
    h.update("x".join(map(str, arr.shape)).encode("ascii")); h.update(b"|")
    h.update(memoryview(arr).cast("B"))
    return h.hexdigest()


def build_runtime_v4_clip_from_d1(
    *,
    d1_clip: D1CompiledClipV1,
    dense_vertices,
    dense_weights,
    weight_joint_ids: Sequence[str],
    asset_id: str,
    composition_by_frame: Sequence[Mapping[str, RuntimeV3FrameComposition]],
    display_name: str,
    intent: str,
    nominal_fps: float,
    runtime_qualified: bool = False,
    required_view_ids: Sequence[str] = tuple(f"V{i}" for i in range(8)),
) -> tuple[RuntimeV4Clip, PlaybackV4D1BridgeReport]:
    fps = float(nominal_fps)
    if not math.isfinite(fps) or fps <= 0:
        raise QualificationError("PLAYBACK_V4_D1_NOMINAL_FPS_INVALID")
    if not asset_id.strip():
        raise QualificationError("PLAYBACK_V4_D1_ASSET_ID_REQUIRED")
    if len(d1_clip.times) != len(d1_clip.poses) or not d1_clip.poses:
        raise QualificationError("PLAYBACK_V4_D1_CLIP_FRAME_CARDINALITY_DRIFT")
    if len(composition_by_frame) != len(d1_clip.poses):
        raise QualificationError("PLAYBACK_V4_D1_COMPOSITION_FRAME_CARDINALITY_DRIFT")

    p = np.asarray(dense_vertices, dtype=np.float64)
    w = np.asarray(dense_weights, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise QualificationError("PLAYBACK_V4_D1_DENSE_VERTICES_INVALID")
    if tuple(map(str, weight_joint_ids)) != tuple(map(str, d1_clip.joint_ids)):
        raise QualificationError("PLAYBACK_V4_D1_WEIGHT_JOINT_ORDER_DRIFT")
    if w.shape != (len(p), len(d1_clip.joint_ids)) or not np.isfinite(w).all() or np.any(w < -1e-12):
        raise QualificationError("PLAYBACK_V4_D1_DENSE_WEIGHTS_INVALID")
    row_sum = w.sum(axis=1)
    weight_error = float(np.max(np.abs(row_sum - 1.0), initial=0.0))
    if weight_error > 1e-9:
        raise QualificationError("PLAYBACK_V4_D1_DENSE_WEIGHT_SIMPLEX_DRIFT")

    view_ids = tuple(map(str, required_view_ids))
    frames: list[RuntimeV4Frame] = []
    posed_hashes: list[str] = []
    for frame_index, (time_seconds, pose) in enumerate(zip(d1_clip.times, d1_clip.poses)):
        posed = np.ascontiguousarray(apply_lbs_matrix_v1(p, w, pose.skin_matrices), dtype=np.float32)
        posed_hashes.append(_array_sha256(posed))
        composition = dict(composition_by_frame[frame_index])
        if set(composition) != set(view_ids):
            raise QualificationError("PLAYBACK_V4_D1_COMPOSITION_VIEW_SET_MISMATCH")
        frames.append(RuntimeV4Frame(
            time_seconds=float(time_seconds),
            canonical_posed_xyz_by_asset={str(asset_id): posed},
            composition_by_view=composition,
        ))

    qualified = bool(runtime_qualified)
    clip = RuntimeV4Clip(
        clip_id=d1_clip.clip_id,
        display_name=str(display_name),
        intent=str(intent),
        duration_seconds=float(d1_clip.duration_seconds),
        fps=fps,
        loop=bool(d1_clip.loop),
        frames=tuple(frames),
        runtime_qualified=qualified,
    )
    bridge_hash = content_sha256({
        "schema": PLAYBACK_V4_D1_BRIDGE_SCHEMA,
        "d1_compile_hash": d1_clip.compile_hash,
        "asset_id": str(asset_id),
        "frame_count": len(frames),
        "vertex_count": len(p),
        "view_ids": list(view_ids),
        "posed_hashes": posed_hashes,
        "runtime_qualified": qualified,
    })
    report = PlaybackV4D1BridgeReport(
        clip_id=d1_clip.clip_id,
        frame_count=len(frames),
        vertex_count=len(p),
        joint_count=len(d1_clip.joint_ids),
        view_count=len(view_ids),
        nominal_fps=fps,
        max_weight_row_sum_error=weight_error,
        canonical_pose_bytes_per_frame=len(p) * 3 * 4,
        avoided_v3_view_multiplier=len(view_ids),
        runtime_qualified=qualified,
        bridge_hash=bridge_hash,
    )
    return clip, report
