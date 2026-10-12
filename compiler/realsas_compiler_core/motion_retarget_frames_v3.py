"""Retarget motion through rest frames, including skipped source ancestors.

Correspondence, target frames and object-basis alignment are explicit inputs.
This operator never changes target geometry, hierarchy or skin weights.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .joint_frames_v1 import derive_joint_frames_from_rows
from .types import QualificationError


def compile_retargeted_keys(payload, *, target_frames, target_to_source,
                            object_rotation, target_body_scale, sample_times=None):
    C = np.asarray(object_rotation, dtype=float)
    if C.shape != (3, 3) or not np.isfinite(C).all() or not np.allclose(
        C.T @ C, np.eye(3), atol=1e-10
    ) or np.linalg.det(C) < .999999:
        raise QualificationError("MOTION_RETARGET_OBJECT_BASIS_INVALID")
    scale = float(payload["metadata"]["source_body_scale"])
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(target_body_scale) or target_body_scale <= 0:
        raise QualificationError("MOTION_RETARGET_BODY_SCALE_INVALID")
    rows = tuple(payload["source_skeleton"])
    sf = derive_joint_frames_from_rows(rows, joint_id_key="source_joint_id",
        parent_id_key="parent_source_joint_id", position_key="rest_position")
    rest = {}
    for sid, frame in sf.items():
        rest[sid] = np.eye(4)
        rest[sid][:3, :3] = frame.rotation_matrix
        rest[sid][:3, 3] = frame.rest_position
    raw_tracks = {t["source_joint_id"]: t["keyframes"] for t in payload["tracks"]}
    if set(raw_tracks) != set(sf):
        raise QualificationError("MOTION_RETARGET_SOURCE_TRACK_ACCOUNTING")
    times = (sorted({float(k["time_seconds"]) for ks in raw_tracks.values() for k in ks})
             if sample_times is None else np.asarray(sample_times, float).tolist())
    if not times or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise QualificationError("MOTION_RETARGET_SAMPLE_TIMES_INVALID")
    order = []
    pending = set(sf)
    while pending:
        ready = sorted(s for s in pending if sf[s].parent_id is None or sf[s].parent_id in order)
        if not ready:
            raise QualificationError("MOTION_RETARGET_SOURCE_TREE_INVALID")
        order.extend(ready)
        pending.difference_update(ready)
    samples = {}
    for sid, ks in raw_tracks.items():
        ts = np.asarray([k["time_seconds"] for k in ks], float)
        qs = np.asarray([k["local_rotation_quat_xyzw"] for k in ks], float)
        if np.any(np.diff(ts) <= 0) or times[0] < ts[0] or times[-1] > ts[-1]:
            raise QualificationError("MOTION_RETARGET_SOURCE_TIME_RANGE")
        if any(not np.allclose(k.get("local_scale_xyz", [1, 1, 1]), 1., atol=1e-10) for k in ks):
            raise QualificationError("MOTION_RETARGET_ANIMATED_SCALE_UNSUPPORTED")
        rotations = (np.repeat(Rotation.from_quat(qs).as_matrix(), len(times), axis=0)
                     if len(ts) == 1 else Slerp(ts, Rotation.from_quat(qs))(times).as_matrix())
        tr = np.asarray([k.get("local_translation_xyz", [0, 0, 0]) for k in ks], float)
        samples[sid] = rotations, np.column_stack([np.interp(times, ts, tr[:, a]) for a in range(3)]) * scale
    mapped_parent = {}
    for tid, sid in target_to_source.items():
        if tid not in target_frames or sid not in sf:
            raise QualificationError("MOTION_RETARGET_CORRESPONDENCE_UNKNOWN")
        pid = target_frames[tid].parent_id
        while pid is not None and pid not in target_to_source:
            pid = target_frames[pid].parent_id
        psid = None if pid is None else target_to_source[pid]
        if psid is not None:
            ancestors = []
            cur = sid
            while cur is not None:
                ancestors.append(cur)
                cur = sf[cur].parent_id
            if psid not in ancestors:
                raise QualificationError("MOTION_RETARGET_CORRESPONDENCE_ANCESTRY")
        mapped_parent[tid] = psid
    out = {tid: [] for tid in target_to_source}
    for fi, time in enumerate(times):
        posed, skin = {}, {}
        for sid in order:
            pid = sf[sid].parent_id
            D = np.eye(4)
            D[:3, :3] = samples[sid][0][fi]
            D[:3, 3] = samples[sid][1][fi]
            posed[sid] = (rest[sid] if pid is None else posed[pid] @ np.linalg.inv(rest[pid]) @ rest[sid]) @ D
            skin[sid] = posed[sid] @ np.linalg.inv(rest[sid])
        for tid, sid in target_to_source.items():
            psid = mapped_parent[tid]
            P = np.eye(4) if psid is None else skin[psid]
            # The relative skin rotation folds every skipped source ancestor.
            delta_R = P[:3, :3].T @ skin[sid][:3, :3]
            delta_t = (np.linalg.inv(P) @ np.r_[posed[sid][:3, 3], 1.])[:3] - rest[sid][:3, 3]
            F = np.asarray(target_frames[tid].rotation_matrix)
            q = Rotation.from_matrix(F.T @ C @ delta_R @ C.T @ F).as_quat()
            if out[tid] and np.dot(q, out[tid][-1]["local_rotation_quat_xyzw"]) < 0:
                q = -q
            out[tid].append(dict(time_seconds=float(time),
                local_rotation_quat_xyzw=q.tolist(),
                local_translation_xyz=(F.T @ C @ delta_t * (float(target_body_scale) / scale)).tolist(),
                local_scale_xyz=[1., 1., 1.]))
    return out
