from __future__ import annotations

"""Endpoint-frequency diagnostics for H-GEO-P3.

Pure analysis helpers: no optimizer, no model mutation, no TESSA.
"""

from dataclasses import dataclass
import math
import numpy as np
import torch

from tools.iris_vnext.h_geo_p1_geometry_v1 import _closest_points_to_triangles


EPS_SCALES_IRIS = (1.0 / 256.0, 1.0 / 128.0, 1.0 / 64.0)
DISTANCE_BINS_IRIS = (0.0, 0.01, 0.02, 0.04, 0.08, 0.1200001)


@dataclass(frozen=True)
class NormalizedCameraV1:
    origin_iris: np.ndarray
    right: np.ndarray
    up: np.ndarray
    forward: np.ndarray
    half_extent_iris: float


def cameras_from_features_v1(camera_features: torch.Tensor | np.ndarray) -> list[NormalizedCameraV1]:
    value = camera_features.detach().cpu().numpy() if torch.is_tensor(camera_features) else np.asarray(camera_features)
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.shape != (8, 13):
        raise ValueError(f"H_GEO_P3_EFREQ_CAMERA_FEATURE_SHAPE_INVALID:{value.shape}")
    out = []
    for row in value.astype(np.float64):
        half = float(row[12])
        if not np.isfinite(row).all() or half <= 0:
            raise ValueError("H_GEO_P3_EFREQ_CAMERA_FEATURE_INVALID")
        out.append(NormalizedCameraV1(row[0:3], row[3:6], row[6:9], row[9:12], half))
    return out


def project_points_v1(points_iris: np.ndarray, camera: NormalizedCameraV1) -> np.ndarray:
    points = np.asarray(points_iris, dtype=np.float64)
    delta = points - camera.origin_iris[None]
    x = delta @ camera.right / camera.half_extent_iris
    y = delta @ camera.up / camera.half_extent_iris
    return np.stack((x, y), axis=-1)


def triangle_view_stats_v1(triangle_iris: np.ndarray, normal_unit: np.ndarray, cameras: list[NormalizedCameraV1], *, resolution: int = 1024) -> dict[str, float]:
    tri = np.asarray(triangle_iris, dtype=np.float64)
    normal = np.asarray(normal_unit, dtype=np.float64)
    if tri.shape != (3, 3) or normal.shape != (3,):
        raise ValueError("H_GEO_P3_EFREQ_TRIANGLE_VIEW_SHAPE_INVALID")
    center = tri.mean(axis=0, keepdims=True)
    in_frame = 0
    informative = 0
    areas = []
    max_edges = []
    for cam in cameras:
        cxy = project_points_v1(center, cam)[0]
        if np.max(np.abs(cxy)) <= 1.0:
            in_frame += 1
        incidence = abs(float(np.dot(normal, cam.forward)))
        if np.max(np.abs(cxy)) <= 1.0 and incidence >= 0.25:
            informative += 1
        xy = project_points_v1(tri, cam)
        px = (xy + 1.0) * (float(resolution) / 2.0)
        area = 0.5 * abs(np.cross(px[1] - px[0], px[2] - px[0]))
        edge = max(np.linalg.norm(px[1] - px[0]), np.linalg.norm(px[2] - px[1]), np.linalg.norm(px[0] - px[2]))
        areas.append(float(area))
        max_edges.append(float(edge))
    return {
        "in_frame_view_count": float(in_frame),
        "informative_view_count": float(informative),
        "maximum_projected_triangle_area_px2": float(max(areas)),
        "median_projected_triangle_area_px2": float(np.median(areas)),
        "maximum_projected_triangle_edge_px": float(max(max_edges)),
    }


def face_vertex_adjacency_v1(faces: np.ndarray) -> list[np.ndarray]:
    faces = np.asarray(faces, dtype=np.int64)
    vertex_to_faces: dict[int, list[int]] = {}
    for fi, face in enumerate(faces):
        for v in face:
            vertex_to_faces.setdefault(int(v), []).append(fi)
    out = []
    for fi, face in enumerate(faces):
        s = {fi}
        for v in face:
            s.update(vertex_to_faces[int(v)])
        out.append(np.asarray(sorted(s), dtype=np.int64))
    return out


def nonadjacent_surface_stats_v1(point_iris: np.ndarray, nearest_face: int, triangles: np.ndarray, adjacency: list[np.ndarray]) -> dict[str, float]:
    p = np.asarray(point_iris, dtype=np.float64).reshape(3)
    tri = np.asarray(triangles, dtype=np.float64)
    mask = np.ones(len(tri), dtype=bool)
    mask[adjacency[int(nearest_face)]] = False
    indices = np.flatnonzero(mask)
    if not len(indices):
        return {"nonadjacent_distance_iris": float("inf"), "nonadjacent_face_index": -1.0}
    q = _closest_points_to_triangles(p, tri[indices])
    d = np.linalg.norm(q - p[None], axis=1)
    j = int(np.argmin(d))
    return {"nonadjacent_distance_iris": float(d[j]), "nonadjacent_face_index": float(indices[j])}


def deterministic_unit_direction_v1(subject_index: int, query_index: int) -> np.ndarray:
    a = float((subject_index + 1) * 0.7548776662466927 + (query_index + 1) * 0.5698402909980532)
    b = float((subject_index + 1) * 0.438289 + (query_index + 1) * 0.317837)
    v = np.asarray((math.sin(2 * math.pi * a), math.cos(2 * math.pi * a), math.sin(2 * math.pi * b)), dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / max(n, 1e-12)


def angular_change_degrees_v1(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(3)
    b = np.asarray(b, dtype=np.float64).reshape(3)
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    c = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def select_tail_and_matched_control_v1(endpoint_error: np.ndarray, distance_iris: np.ndarray, *, maximum_per_group: int = 384) -> tuple[np.ndarray, np.ndarray]:
    err = np.asarray(endpoint_error, dtype=np.float64)
    dist = np.asarray(distance_iris, dtype=np.float64)
    valid = np.isfinite(err) & np.isfinite(dist) & (dist <= 0.12)
    idx = np.flatnonzero(valid)
    if len(idx) < 64:
        raise ValueError("H_GEO_P3_EFREQ_TOO_FEW_VALID_QUERIES")
    threshold = float(np.quantile(err[idx], 0.90))
    tail = idx[err[idx] >= threshold]
    tail = tail[np.argsort(-err[tail], kind="mergesort")][: int(maximum_per_group)]
    controls: list[int] = []
    for lo, hi in zip(DISTANCE_BINS_IRIS[:-1], DISTANCE_BINS_IRIS[1:]):
        t = tail[(dist[tail] >= lo) & (dist[tail] < hi)]
        if not len(t):
            continue
        candidates = idx[(dist[idx] >= lo) & (dist[idx] < hi) & (err[idx] < threshold)]
        if not len(candidates):
            continue
        median = float(np.median(err[candidates]))
        order = candidates[np.argsort(np.abs(err[candidates] - median), kind="mergesort")]
        controls.extend(int(x) for x in order[: len(t)])
    control = np.asarray(controls[: len(tail)], dtype=np.int64)
    if len(control) < max(16, len(tail) // 2):
        candidates = np.setdiff1d(idx, tail, assume_unique=False)
        median = float(np.median(err[candidates]))
        order = candidates[np.argsort(np.abs(err[candidates] - median), kind="mergesort")]
        control = order[: len(tail)]
    return np.asarray(tail, dtype=np.int64), np.asarray(control, dtype=np.int64)


def whitened_rms_delta_v1(a: torch.Tensor, b: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    denom = scale.clamp_min(1e-6)
    return torch.sqrt(torch.mean(((a - b) / denom) ** 2, dim=-1))


def radial_highband_fraction_v1(residual_planes: torch.Tensor, *, cutoff_fraction_of_nyquist: float = 0.5) -> float:
    if residual_planes.ndim != 5 or residual_planes.shape[1] != 3:
        raise ValueError("H_GEO_P3_EFREQ_RESIDUAL_PLANE_SHAPE_INVALID")
    x = residual_planes.detach().float().reshape(-1, residual_planes.shape[-2], residual_planes.shape[-1])
    spec = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
    power = spec.real.square() + spec.imag.square()
    h, w = x.shape[-2:]
    yy = torch.linspace(-1.0, 1.0, h, device=x.device)[:, None]
    xx = torch.linspace(-1.0, 1.0, w, device=x.device)[None, :]
    rr = torch.sqrt(xx.square() + yy.square())
    mask = rr >= float(cutoff_fraction_of_nyquist)
    total = power.sum().clamp_min(1e-20)
    return float(power[..., mask].sum().div(total).item())
