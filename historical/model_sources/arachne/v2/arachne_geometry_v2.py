from __future__ import annotations

import numpy as np

PAIR_GEOMETRY_CONTRACT_V2 = (
    "point_control_dx",
    "point_control_dy",
    "point_control_dz",
    "point_control_distance",
    "point_segment_distance",
    "segment_t",
    "segment_length",
    "normal_axis_abs_cos",
    "normal_axis_valid",
    "parent_exists",
)


def arachne_pair_geometry_v2(
    surface_positions_normalized: np.ndarray,
    surface_normals: np.ndarray,
    surface_normal_valid: np.ndarray,
    joint_positions_normalized: np.ndarray,
    parent_indices: np.ndarray,
    surface_mask: np.ndarray,
    joint_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Point↔control and parent-segment geometry for skin-field prediction."""
    sp = np.asarray(surface_positions_normalized, np.float32)
    sn = np.asarray(surface_normals, np.float32)
    sv = np.asarray(surface_normal_valid, bool)
    jp = np.asarray(joint_positions_normalized, np.float32)
    pi = np.asarray(parent_indices, np.int64)
    sm = np.asarray(surface_mask, bool)
    jm = np.asarray(joint_mask, bool)
    if sp.ndim != 3 or sp.shape[-1] != 3 or jp.ndim != 3 or jp.shape[-1] != 3:
        raise ValueError("position tensors must be [B,N/J,3]")
    if sn.shape != sp.shape or sv.shape != sp.shape[:2] or sm.shape != sp.shape[:2] or jm.shape != jp.shape[:2] or pi.shape != jm.shape:
        raise ValueError("Arachne geometry mask/shape mismatch")
    if sp.shape[0] != jp.shape[0]:
        raise ValueError("Arachne geometry batch mismatch")
    B, N, _ = sp.shape; J = jp.shape[1]
    out = np.zeros((B, N, J, len(PAIR_GEOMETRY_CONTRACT_V2)), np.float32)
    pair_mask = sm[:, :, None] & jm[:, None, :]
    for b in range(B):
        for j in range(J):
            if not jm[b, j]:
                continue
            control = jp[b, j]
            delta = sp[b] - control[None]
            pc_dist = np.linalg.norm(delta, axis=1)
            parent = int(pi[b, j])
            if parent >= 0:
                if parent >= J or not jm[b, parent]:
                    raise ValueError("parent index references invalid joint")
                a = jp[b, parent]; seg = control - a; seg_len = float(np.linalg.norm(seg))
                if seg_len > 1e-8:
                    axis = seg / seg_len
                    raw_t = ((sp[b] - a[None]) @ axis) / seg_len
                    t = np.clip(raw_t, 0.0, 1.0)
                    closest = a[None] + t[:, None] * seg[None]
                    seg_dist = np.linalg.norm(sp[b] - closest, axis=1)
                    cos = np.abs(sn[b] @ axis)
                    axis_valid = sv[b].astype(np.float32)
                else:
                    t = np.zeros(N, np.float32); seg_dist = np.linalg.norm(sp[b] - a[None], axis=1); cos = np.zeros(N, np.float32); axis_valid = np.zeros(N, np.float32)
                parent_exists = np.ones(N, np.float32)
            else:
                seg_len = 0.0; t = np.zeros(N, np.float32); seg_dist = pc_dist.copy(); cos = np.zeros(N, np.float32); axis_valid = np.zeros(N, np.float32); parent_exists = np.zeros(N, np.float32)
            out[b, :, j, :] = np.stack([
                delta[:, 0], delta[:, 1], delta[:, 2], pc_dist, seg_dist, t,
                np.full(N, seg_len, np.float32), cos.astype(np.float32), axis_valid, parent_exists,
            ], axis=-1)
    out *= pair_mask[..., None].astype(np.float32)
    if not np.isfinite(out).all():
        raise ValueError("non-finite Arachne pair geometry")
    return out, pair_mask
