from __future__ import annotations

"""Generic D1 canonical motion -> Runtime-v3 baked-frame bridge.

This service layer connects qualified 3D motion mechanics to the deployment writer.
It does not choose topology, appearance donors, visibility subsets, or subject policy.
"""

from dataclasses import dataclass
from typing import Mapping, Sequence
import math

import numpy as np

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.motion_3d_adapter_v1 import D1CompiledClipV1
from compiler.realsas_compiler_core.motion_3d_v1 import apply_lbs_matrix_v1
from compiler.realsas_compiler_core.playback_full_surface_v3 import project_posed_full_surface_frames_v3
from compiler.realsas_compiler_core.playback_runtime_v3 import RuntimeV3FrameComposition
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_services.export.runtime_v3 import RuntimeV3Clip, RuntimeV3Frame


PLAYBACK_V3_D1_BRIDGE_SCHEMA = "RealSaS.PlaybackV3D1Bridge.v1"


@dataclass(frozen=True)
class PlaybackV3D1BridgeReport:
    clip_id: str
    frame_count: int
    vertex_count: int
    joint_count: int
    view_count: int
    nominal_fps: float
    max_weight_row_sum_error: float
    bridge_hash: str
    schema_version: str = PLAYBACK_V3_D1_BRIDGE_SCHEMA


def _validate_dense_skin(
    dense_vertices,
    dense_weights,
    *,
    weight_joint_ids: Sequence[str],
    expected_joint_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, float]:
    p = np.asarray(dense_vertices, dtype=np.float64)
    w = np.asarray(dense_weights, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise QualificationError("PLAYBACK_V3_D1_DENSE_VERTICES_INVALID")
    if tuple(map(str, weight_joint_ids)) != tuple(map(str, expected_joint_ids)):
        raise QualificationError("PLAYBACK_V3_D1_WEIGHT_JOINT_ORDER_DRIFT")
    if w.shape != (len(p), len(expected_joint_ids)) or not np.isfinite(w).all():
        raise QualificationError("PLAYBACK_V3_D1_DENSE_WEIGHTS_INVALID")
    if np.any(w < -1.0e-12):
        raise QualificationError("PLAYBACK_V3_D1_DENSE_WEIGHTS_NEGATIVE")
    row_sum = w.sum(axis=1)
    max_error = float(np.max(np.abs(row_sum - 1.0), initial=0.0))
    if max_error > 1.0e-9:
        raise QualificationError(f"PLAYBACK_V3_D1_DENSE_WEIGHT_SIMPLEX_DRIFT:{max_error}")
    return p, w, max_error


def build_runtime_v3_clip_from_d1(
    *,
    d1_clip: D1CompiledClipV1,
    dense_vertices,
    dense_weights,
    weight_joint_ids: Sequence[str],
    cameras: Mapping[str, Mapping],
    attachment_id: str,
    composition_by_frame: Sequence[Mapping[str, RuntimeV3FrameComposition]],
    display_name: str,
    intent: str,
    nominal_fps: float,
    required_view_ids: Sequence[str] = tuple(f"V{i}" for i in range(8)),
) -> tuple[RuntimeV3Clip, PlaybackV3D1BridgeReport]:
    """Bake one D1 clip into posed Runtime-v3 XYZ frames.

    X/Y and Z always come from the same deformed canonical 3D vertices. Composition
    is caller-owned and must already be qualified by the Runtime-v3 contract.
    """

    fps = float(nominal_fps)
    if not math.isfinite(fps) or fps <= 0.0:
        raise QualificationError("PLAYBACK_V3_D1_NOMINAL_FPS_INVALID")
    if not str(attachment_id).strip():
        raise QualificationError("PLAYBACK_V3_D1_ATTACHMENT_ID_REQUIRED")
    if len(d1_clip.times) != len(d1_clip.poses) or not d1_clip.poses:
        raise QualificationError("PLAYBACK_V3_D1_CLIP_FRAME_CARDINALITY_DRIFT")
    if len(composition_by_frame) != len(d1_clip.poses):
        raise QualificationError("PLAYBACK_V3_D1_COMPOSITION_FRAME_CARDINALITY_DRIFT")
    if abs(float(d1_clip.times[0])) > 1.0e-9 or abs(float(d1_clip.times[-1]) - float(d1_clip.duration_seconds)) > 1.0e-9:
        raise QualificationError("PLAYBACK_V3_D1_CLIP_MUST_INCLUDE_ENDPOINTS")

    view_ids = tuple(map(str, required_view_ids))
    if set(map(str, cameras.keys())) != set(view_ids):
        raise QualificationError("PLAYBACK_V3_D1_CAMERA_VIEW_SET_MISMATCH")
    p, w, weight_error = _validate_dense_skin(
        dense_vertices,
        dense_weights,
        weight_joint_ids=weight_joint_ids,
        expected_joint_ids=d1_clip.joint_ids,
    )

    frames: list[RuntimeV3Frame] = []
    posed_hashes: list[str] = []
    for frame_index, (time_seconds, pose) in enumerate(zip(d1_clip.times, d1_clip.poses)):
        posed_3d = apply_lbs_matrix_v1(p, w, pose.skin_matrices)
        posed_hashes.append(content_sha256({
            "frame_index": frame_index,
            "time_seconds": float(time_seconds),
            "shape": list(posed_3d.shape),
            "values": posed_3d.tolist(),
        }))
        posed_by_mesh = project_posed_full_surface_frames_v3(
            posed_3d,
            cameras,
            attachment_id=str(attachment_id),
            expected_vertex_count=len(p),
            required_view_ids=view_ids,
        )
        composition = dict(composition_by_frame[frame_index])
        if set(composition) != set(view_ids):
            raise QualificationError("PLAYBACK_V3_D1_COMPOSITION_VIEW_SET_MISMATCH")
        for view_id in view_ids:
            if composition[view_id].view_id != view_id:
                raise QualificationError("PLAYBACK_V3_D1_COMPOSITION_VIEW_ID_DRIFT")
        frames.append(RuntimeV3Frame(
            time_seconds=float(time_seconds),
            posed_xyz_by_mesh=posed_by_mesh,
            composition_by_view=composition,
        ))

    runtime_clip = RuntimeV3Clip(
        clip_id=d1_clip.clip_id,
        display_name=str(display_name),
        intent=str(intent),
        duration_seconds=float(d1_clip.duration_seconds),
        fps=fps,
        loop=bool(d1_clip.loop),
        frames=tuple(frames),
        runtime_qualified=True,
    )
    bridge_hash = content_sha256({
        "schema": PLAYBACK_V3_D1_BRIDGE_SCHEMA,
        "d1_compile_hash": d1_clip.compile_hash,
        "clip_id": d1_clip.clip_id,
        "attachment_id": str(attachment_id),
        "frame_count": len(frames),
        "vertex_count": len(p),
        "joint_ids": list(d1_clip.joint_ids),
        "view_ids": list(view_ids),
        "nominal_fps": fps,
        "max_weight_row_sum_error": weight_error,
        "posed_frame_hashes": posed_hashes,
    })
    report = PlaybackV3D1BridgeReport(
        clip_id=d1_clip.clip_id,
        frame_count=len(frames),
        vertex_count=len(p),
        joint_count=len(d1_clip.joint_ids),
        view_count=len(view_ids),
        nominal_fps=fps,
        max_weight_row_sum_error=weight_error,
        bridge_hash=bridge_hash,
    )
    return runtime_clip, report
