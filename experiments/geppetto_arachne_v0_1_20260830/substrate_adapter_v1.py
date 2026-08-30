from __future__ import annotations

import hashlib
from typing import Iterable, Mapping

import numpy as np

from contracts_v0_1 import ConsumerTokenV1, PathStateFeaturesV1, VolumeState


def _view_mask(support_views: Iterable[int]) -> tuple[int, ...]:
    m = [0] * 8
    for v in support_views:
        iv = int(v)
        if iv < 0 or iv >= 8:
            raise ValueError(f"support view out of range: {iv}")
        m[iv] = 1
    return tuple(m)


def surface_tokens_from_rigging_surface_v1(
    surface,
    *,
    geometry_uncertainty_by_surface_id: Mapping[str, float] | None = None,
) -> tuple[ConsumerTokenV1, ...]:
    """Adapt current RiggingSurfaceIR-like nodes without requiring future IRIS risk.

    `geometry_uncertainty_by_surface_id` is optional/versioned. When absent,
    uncertainty is explicitly invalid rather than silently set to authoritative 0.
    """
    uncertainty = geometry_uncertainty_by_surface_id or {}
    out: list[ConsumerTokenV1] = []
    for node in surface.surface_nodes:
        n = getattr(node, "derived_normal", None)
        n_valid = n is not None and np.isfinite(np.asarray(n, float)).all()
        if n_valid:
            nv = np.asarray(n, np.float32)
            ln = float(np.linalg.norm(nv))
            n_valid = ln > 1e-8
            if n_valid:
                nv = nv / ln
            else:
                nv = np.zeros(3, np.float32)
        else:
            nv = np.zeros(3, np.float32)
        has_u = node.surface_id in uncertainty
        u = float(uncertainty[node.surface_id]) if has_u else 0.0
        if has_u and (not np.isfinite(u) or u < 0):
            raise ValueError(f"bad geometry uncertainty for {node.surface_id}: {u}")
        out.append(
            ConsumerTokenV1(
                token_id=f"S:{node.surface_id}",
                position=tuple(map(float, node.P)),
                token_type="SURFACE",
                normal=tuple(map(float, nv)),
                normal_valid=bool(n_valid),
                support_view_mask=_view_mask(node.support_views),
                geometry_uncertainty=u,
                geometry_uncertainty_valid=has_u,
                medialness=float(getattr(node, "metadata", {}).get("medialness", 0.0)),
                local_thickness=float(getattr(node, "metadata", {}).get("local_thickness", 0.0)),
                branch_likelihood=float(getattr(node, "metadata", {}).get("branch_likelihood", 0.0)),
                volume_state=VolumeState.SUPPORTED_SURFACE_BAND,
                provenance={
                    "surface_id": node.surface_id,
                    "geometry_lineage_hash": getattr(surface, "geometry_lineage_hash", ""),
                    "uncertainty_optional_channel_used": has_u,
                },
            )
        )
    return tuple(out)


def _grid_coordinates(bounds_min, bounds_max, shape) -> np.ndarray:
    b0 = np.asarray(bounds_min, np.float32)
    b1 = np.asarray(bounds_max, np.float32)
    axes = [np.linspace(b0[i], b1[i], int(shape[i]), dtype=np.float32) for i in range(3)]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    return np.stack([X, Y, Z], axis=-1)


def classify_volume_state_v1(vol, ijk: tuple[int, int, int]) -> VolumeState:
    i, j, k = map(int, ijk)
    if bool(vol.certain_outside[i, j, k]):
        return VolumeState.CERTAIN_OUTSIDE
    if bool(vol.supported_surface[i, j, k]):
        return VolumeState.SUPPORTED_SURFACE_BAND
    if bool(vol.high_confidence_interior[i, j, k]):
        return VolumeState.HIGH_CONFIDENCE_INTERIOR
    if bool(vol.unknown_concavity[i, j, k]):
        return VolumeState.UNKNOWN_CONCAVITY
    if bool(vol.possible_interior[i, j, k]):
        return VolumeState.POSSIBLE_INTERIOR
    return VolumeState.CERTAIN_OUTSIDE


def _fps(points: np.ndarray, count: int, seed_index: int) -> np.ndarray:
    if count >= len(points):
        return np.arange(len(points), dtype=np.int64)
    chosen = np.empty(count, np.int64)
    chosen[0] = int(seed_index) % len(points)
    d2 = np.sum((points - points[chosen[0]]) ** 2, axis=1)
    for t in range(1, count):
        j = int(np.argmax(d2))
        chosen[t] = j
        d2 = np.minimum(d2, np.sum((points - points[j]) ** 2, axis=1))
    return chosen


def sample_interior_tokens_v1(
    vol,
    interior,
    *,
    count: int,
    stream_id: str,
    candidate_pool_max: int = 8192,
) -> tuple[ConsumerTokenV1, ...]:
    """Deterministically sample interior evidence; never samples CERTAIN_OUTSIDE.

    Candidate pool is spread lexicographically before FPS to keep runtime bounded.
    Medialness/thickness influence only the deterministic first seed, not authority.
    """
    if count <= 0:
        return ()
    shape = vol.possible_interior.shape
    if len(shape) != 3:
        raise ValueError("volume must be 3D")
    candidate = np.argwhere(vol.possible_interior & ~vol.supported_surface & ~vol.certain_outside)
    if not len(candidate):
        return ()
    if len(candidate) > candidate_pool_max:
        # Deterministic coverage subset; no random state hidden in the sampler.
        idx = np.linspace(0, len(candidate) - 1, candidate_pool_max, dtype=np.int64)
        candidate = candidate[idx]
    coords_all = _grid_coordinates(vol.bounds_min, vol.bounds_max, shape)
    points = coords_all[tuple(candidate.T)]
    med = np.asarray(interior.medialness)[tuple(candidate.T)]
    thick = np.asarray(interior.local_thickness)[tuple(candidate.T)]
    score = med + 1e-3 * thick
    # Tie-break by a stable stream hash so two consumer profiles can use distinct
    # deterministic samples without data-dependent randomness.
    h = int(hashlib.sha256(stream_id.encode("utf-8")).hexdigest()[:16], 16)
    top = np.flatnonzero(score == score.max())
    seed = int(top[h % len(top)])
    chosen = _fps(points, min(count, len(points)), seed)

    out: list[ConsumerTokenV1] = []
    for rank, q in enumerate(chosen.tolist()):
        ijk = tuple(map(int, candidate[q]))
        st = classify_volume_state_v1(vol, ijk)
        if st in {VolumeState.CERTAIN_OUTSIDE, VolumeState.SUPPORTED_SURFACE_BAND}:
            raise AssertionError("interior sampler admitted forbidden state")
        out.append(
            ConsumerTokenV1(
                token_id=f"I:{rank:05d}:{ijk[0]}:{ijk[1]}:{ijk[2]}",
                position=tuple(map(float, points[q])),
                token_type="INTERIOR",
                normal=(0.0, 0.0, 0.0),
                normal_valid=False,
                support_view_mask=(0, 0, 0, 0, 0, 0, 0, 0),
                geometry_uncertainty=0.0,
                geometry_uncertainty_valid=False,
                medialness=float(np.asarray(interior.medialness)[ijk]),
                local_thickness=float(np.asarray(interior.local_thickness)[ijk]),
                branch_likelihood=float(np.asarray(interior.branch_likelihood)[ijk]),
                volume_state=st,
                provenance={
                    "voxel_ijk": ijk,
                    "volume_lineage_sha256": getattr(vol, "lineage_sha256", ""),
                    "interior_lineage_sha256": getattr(interior, "lineage_sha256", ""),
                },
            )
        )
    return tuple(out)


def _point_to_index(point, bounds_min, bounds_max, shape) -> tuple[int, int, int]:
    p = np.asarray(point, np.float64)
    b0 = np.asarray(bounds_min, np.float64)
    b1 = np.asarray(bounds_max, np.float64)
    t = (p - b0) / np.maximum(b1 - b0, 1e-12)
    q = np.rint(t * (np.asarray(shape, np.float64) - 1.0)).astype(np.int64)
    q = np.clip(q, 0, np.asarray(shape) - 1)
    return tuple(map(int, q))


def path_state_features_v1(
    point,
    target,
    vol,
    *,
    samples: int = 33,
) -> PathStateFeaturesV1:
    """Typed point→control connectivity evidence for A0.1.

    This is deterministic substrate evidence, not a learned or teacher-only field.
    """
    if samples < 2:
        raise ValueError("samples must be >=2")
    a = np.asarray(point, np.float64)
    b = np.asarray(target, np.float64)
    total = float(np.linalg.norm(b - a))
    ts = np.linspace(0.0, 1.0, samples, dtype=np.float64)
    xyz = a[None] * (1.0 - ts[:, None]) + b[None] * ts[:, None]
    states = [classify_volume_state_v1(vol, _point_to_index(p, vol.bounds_min, vol.bounds_max, vol.possible_interior.shape)) for p in xyz]
    counts = np.bincount(np.asarray(states, np.int64), minlength=5).astype(np.float64)
    frac = counts / counts.sum()

    # Midpoint state approximates each equal-length interval's state.
    mid = (xyz[:-1] + xyz[1:]) * 0.5
    mids = [classify_volume_state_v1(vol, _point_to_index(p, vol.bounds_min, vol.bounds_max, vol.possible_interior.shape)) for p in mid]
    seg = total / float(samples - 1)
    lengths = np.bincount(np.asarray(mids, np.int64), minlength=5).astype(np.float64) * seg
    return PathStateFeaturesV1(
        state_fraction=tuple(map(float, frac)),
        length_by_state=tuple(map(float, lengths)),
        total_length=total,
        certain_outside_fraction=float(frac[int(VolumeState.CERTAIN_OUTSIDE)]),
        unknown_concavity_fraction=float(frac[int(VolumeState.UNKNOWN_CONCAVITY)]),
    )


def point_control_geometry_v1(point, control, *, parent=None, normal=None) -> np.ndarray:
    """Production-available point/control pair geometry; no source bone tails."""
    p = np.asarray(point, np.float32)
    c = np.asarray(control, np.float32)
    if parent is None:
        # Root has no compiler parent segment. Retain point distance and mark
        # segment-dependent terms invalid rather than invent an axis.
        dist = float(np.linalg.norm(p - c))
        return np.asarray([dist, 0.0, 0.0, 0.0], np.float32)
    a = np.asarray(parent, np.float32)
    ab = c - a
    den = float(np.dot(ab, ab))
    if den <= 1e-12:
        t = 0.0
        closest = a
        axis = np.zeros(3, np.float32)
        axis_valid = 0.0
    else:
        t = float(np.clip(np.dot(p - a, ab) / den, 0.0, 1.0))
        closest = a + t * ab
        axis = ab / np.sqrt(den)
        axis_valid = 1.0
    dist = float(np.linalg.norm(p - closest))
    if normal is None or axis_valid == 0.0:
        angle_cos = 0.0
        normal_axis_valid = 0.0
    else:
        n = np.asarray(normal, np.float32)
        nn = float(np.linalg.norm(n))
        if nn <= 1e-8:
            angle_cos = 0.0
            normal_axis_valid = 0.0
        else:
            angle_cos = float(abs(np.dot(n / nn, axis)))
            normal_axis_valid = 1.0
    return np.asarray([dist, t, angle_cos, normal_axis_valid], np.float32)
