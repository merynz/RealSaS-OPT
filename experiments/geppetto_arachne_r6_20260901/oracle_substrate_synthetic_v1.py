from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.types import ObservationEvidenceIR, ObservationSample, RiggingSurfaceIR, SurfaceNode
from experiments.iris_reprojection_v2_20260831.persistence_adapter_v2 import compile_surface_v2, attach_dtb_nd1_from_evidence
from experiments.geppetto_arachne_r6_20260901.test_arachne_v2_behavioral_panel import Witness


CAPSULE_RADIUS = 0.075
CYLINDER_T_SAMPLES = 3
CYLINDER_AZIMUTH_SAMPLES = 8
SPHERE_LATITUDE_SAMPLES = 4
SPHERE_AZIMUTH_SAMPLES = 8
SHELL_CULL_EPS = 1e-5
DEDUP_GRID = 1e-5
MAX_FULL_SURFACE_SAMPLES = 160
VIEW_AZIMUTHS_DEG = tuple(45.0 * i for i in range(8))
RASTER_RESOLUTION = 64
FRAMING_MARGIN = 1.10
FRONT_FACING_EPS = 1e-6
PERSISTENCE_TOLERANCE = 1e-6


CENTERLINE_EDGES = {
    "chain_blend_3": tuple((i, i + 1) for i in range(8)),
    "branch_blend_4": (
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
        (2, 6), (6, 7), (7, 8),
        (2, 9), (9, 10), (10, 11),
    ),
    "sharp_fork_5": (
        (0, 1), (1, 2), (2, 3), (3, 4),
        (4, 5), (5, 6), (6, 7), (7, 8), (8, 9),
        (4, 10), (10, 11), (11, 12), (12, 13), (13, 14),
    ),
}


@dataclass(frozen=True)
class ShellSample:
    sample_index: int
    P: tuple[float, float, float]
    N: tuple[float, float, float]


@dataclass(frozen=True)
class OrthoCamera:
    view_index: int
    forward: tuple[float, float, float]
    right: tuple[float, float, float]
    up: tuple[float, float, float]
    plane_distance: float
    raster_extent: float
    resolution: int = RASTER_RESOLUTION

    def project_world(self, p) -> tuple[float, float, float, float, float]:
        P = np.asarray(p, np.float64)
        f = np.asarray(self.forward, np.float64)
        r = np.asarray(self.right, np.float64)
        u = np.asarray(self.up, np.float64)
        x = float(np.dot(P, r))
        y = float(np.dot(P, u))
        depth = float(self.plane_distance + np.dot(P, f))
        px = (x / (2.0 * self.raster_extent) + 0.5) * float(self.resolution) - 0.5
        py = (y / (2.0 * self.raster_extent) + 0.5) * float(self.resolution) - 0.5
        return x, y, depth, px, py

    def ray_origin_for_projected(self, x: float, y: float) -> tuple[float, float, float]:
        f = np.asarray(self.forward, np.float64)
        r = np.asarray(self.right, np.float64)
        u = np.asarray(self.up, np.float64)
        O = r * float(x) + u * float(y) - float(self.plane_distance) * f
        return tuple(map(float, O))


def _unit(v) -> np.ndarray:
    a = np.asarray(v, np.float64)
    n = float(np.linalg.norm(a))
    if not math.isfinite(n) or n <= 1e-12:
        raise ValueError("invalid vector")
    return a / n


def _basis(axis) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = _unit(axis)
    ref = np.asarray((0.0, 1.0, 0.0), np.float64)
    if abs(float(np.dot(z, ref))) > 0.90:
        ref = np.asarray((1.0, 0.0, 0.0), np.float64)
    x = _unit(np.cross(z, ref))
    y = _unit(np.cross(z, x))
    return z, x, y


def _distance_to_segment(p, a, b) -> float:
    P = np.asarray(p, np.float64)
    A = np.asarray(a, np.float64)
    B = np.asarray(b, np.float64)
    AB = B - A
    denom = float(np.dot(AB, AB))
    if denom <= 1e-16:
        return float(np.linalg.norm(P - A))
    t = float(np.clip(np.dot(P - A, AB) / denom, 0.0, 1.0))
    return float(np.linalg.norm(P - (A + t * AB)))


def _candidate_shell_points(witness: Witness) -> list[tuple[np.ndarray, np.ndarray]]:
    points = np.asarray(witness.points, np.float64)
    edges = CENTERLINE_EDGES[witness.name]
    candidates: list[tuple[np.ndarray, np.ndarray]] = []

    for ia, ib in edges:
        a, b = points[ia], points[ib]
        axis, e1, e2 = _basis(b - a)
        del axis
        for ti in range(CYLINDER_T_SAMPLES):
            t = float(ti + 1) / float(CYLINDER_T_SAMPLES + 1)
            center = a + t * (b - a)
            for ai in range(CYLINDER_AZIMUTH_SAMPLES):
                angle = 2.0 * math.pi * float(ai) / float(CYLINDER_AZIMUTH_SAMPLES)
                n = math.cos(angle) * e1 + math.sin(angle) * e2
                candidates.append((center + CAPSULE_RADIUS * n, _unit(n)))

    for center in points:
        candidates.append((center + CAPSULE_RADIUS * np.asarray((0.0, 1.0, 0.0)), np.asarray((0.0, 1.0, 0.0))))
        candidates.append((center + CAPSULE_RADIUS * np.asarray((0.0, -1.0, 0.0)), np.asarray((0.0, -1.0, 0.0))))
        for li in range(SPHERE_LATITUDE_SAMPLES):
            theta = math.pi * float(li + 1) / float(SPHERE_LATITUDE_SAMPLES + 1)
            st, ct = math.sin(theta), math.cos(theta)
            for ai in range(SPHERE_AZIMUTH_SAMPLES):
                phi = 2.0 * math.pi * float(ai) / float(SPHERE_AZIMUTH_SAMPLES)
                n = np.asarray((st * math.cos(phi), ct, st * math.sin(phi)), np.float64)
                candidates.append((center + CAPSULE_RADIUS * n, _unit(n)))

    retained = []
    for p, n in candidates:
        strictly_inside = False
        for ia, ib in edges:
            if _distance_to_segment(p, points[ia], points[ib]) < CAPSULE_RADIUS - SHELL_CULL_EPS:
                strictly_inside = True
                break
        if not strictly_inside:
            retained.append((p, n))

    by_key: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}
    for p, n in sorted(retained, key=lambda x: (*map(float, x[0]), *map(float, x[1]))):
        key = tuple(np.rint(np.asarray(p) / DEDUP_GRID).astype(np.int64).tolist())
        by_key.setdefault(key, (p, n))
    return list(by_key.values())


def _farthest_point_subset(candidates: list[tuple[np.ndarray, np.ndarray]], count: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if len(candidates) <= count:
        return sorted(candidates, key=lambda x: tuple(map(float, x[0])))
    order = sorted(range(len(candidates)), key=lambda i: tuple(map(float, candidates[i][0])))
    P = np.stack([candidates[i][0] for i in range(len(candidates))], axis=0)
    selected = [order[0]]
    min_d2 = np.sum((P - P[selected[0]][None]) ** 2, axis=1)
    min_d2[selected[0]] = -1.0
    while len(selected) < count:
        best_value = float(np.max(min_d2))
        ties = np.flatnonzero(np.isclose(min_d2, best_value, rtol=0.0, atol=1e-15)).tolist()
        nxt = min(ties, key=lambda i: tuple(map(float, P[i])))
        selected.append(nxt)
        d2 = np.sum((P - P[nxt][None]) ** 2, axis=1)
        min_d2 = np.minimum(min_d2, d2)
        min_d2[selected] = -1.0
    return [candidates[i] for i in selected]


def build_full_shell_samples(witness: Witness) -> tuple[ShellSample, ...]:
    candidates = _candidate_shell_points(witness)
    chosen = _farthest_point_subset(candidates, MAX_FULL_SURFACE_SAMPLES)
    chosen = sorted(chosen, key=lambda x: tuple(map(float, x[0])))
    return tuple(
        ShellSample(i, tuple(map(float, p)), tuple(map(float, _unit(n))))
        for i, (p, n) in enumerate(chosen)
    )


def build_cameras(samples: tuple[ShellSample, ...]) -> tuple[OrthoCamera, ...]:
    P = np.asarray([s.P for s in samples], np.float64)
    up = np.asarray((0.0, 1.0, 0.0), np.float64)
    forward_right = []
    max_projection = 0.0
    for degrees in VIEW_AZIMUTHS_DEG:
        a = math.radians(float(degrees))
        f = _unit((math.cos(a), 0.0, math.sin(a)))
        r = _unit(np.cross(f, up))
        max_projection = max(
            max_projection,
            float(np.max(np.abs(P @ r))),
            float(np.max(np.abs(P @ up))),
        )
        forward_right.append((f, r))
    raster_extent = max(1e-6, FRAMING_MARGIN * max_projection)
    world_radius = float(np.max(np.linalg.norm(P, axis=1)))
    plane_distance = 1.0 + 3.0 * world_radius
    return tuple(
        OrthoCamera(i, tuple(map(float, f)), tuple(map(float, r)), (0.0, 1.0, 0.0), plane_distance, raster_extent)
        for i, (f, r) in enumerate(forward_right)
    )


def visible_views(samples: tuple[ShellSample, ...], cameras: tuple[OrthoCamera, ...]) -> tuple[tuple[int, ...], ...]:
    by_sample = [set() for _ in samples]
    for cam in cameras:
        f = np.asarray(cam.forward, np.float64)
        owner: dict[tuple[int, int], tuple[float, int]] = {}
        for sample in samples:
            n = np.asarray(sample.N, np.float64)
            if float(np.dot(n, f)) >= -FRONT_FACING_EPS:
                continue
            _, _, depth, px, py = cam.project_world(sample.P)
            ix, iy = int(round(px)), int(round(py))
            if ix < 0 or ix >= cam.resolution or iy < 0 or iy >= cam.resolution or depth <= 0.0:
                continue
            key = (ix, iy)
            rec = (float(depth), int(sample.sample_index))
            if key not in owner or rec < owner[key]:
                owner[key] = rec
        for _, sample_index in owner.values():
            by_sample[sample_index].add(cam.view_index)
    return tuple(tuple(sorted(v)) for v in by_sample)


def build_u0_surface(witness: Witness) -> tuple[RiggingSurfaceIR, tuple[ShellSample, ...], tuple[OrthoCamera, ...], tuple[tuple[int, ...], ...]]:
    samples = build_full_shell_samples(witness)
    cameras = build_cameras(samples)
    visibility = visible_views(samples, cameras)
    nodes = []
    for sample in samples:
        bindings = []
        for cam in cameras:
            _, _, _, px, py = cam.project_world(sample.P)
            bindings.append((cam.view_index, (float(px), float(py))))
        nodes.append(SurfaceNode(
            surface_id=f"U0:{witness.name}:S:{sample.sample_index:03d}",
            P=sample.P,
            support_views=tuple(range(8)),
            provenance_refs=(f"U0_REFERENCE_FULL_SURFACE:{witness.name}:{sample.sample_index}",),
            source_observation_ids=(),
            raster_bindings=tuple(bindings),
            persistence_group_id=f"U0FULL:{sample.sample_index:03d}",
            derived_normal=sample.N,
            metadata={"reference_full_surface": True, "sample_index": sample.sample_index},
        ))
    operator_hash = content_sha256({"operator": "U0_REFERENCE_ANALYTIC_NORMAL_V1", "radius": CAPSULE_RADIUS})
    lineage = content_sha256({"arm": "U0_REFERENCE_FULL_SURFACE", "witness": witness.name, "nodes": [n.to_dict() for n in nodes]})
    surface = RiggingSurfaceIR(
        tuple(nodes),
        geometry_lineage_hash=lineage,
        metadata={
            "raster_coordinate_system": "PIXEL_CENTER_XY",
            "resolution": RASTER_RESOLUTION,
            "Nd_operator_sha256": operator_hash,
            "oracle_arm": "U0_REFERENCE_FULL_SURFACE",
            "closed_mesh_product_authority": False,
        },
    )
    return surface, samples, cameras, visibility


def build_u1_evidence(witness: Witness, samples: tuple[ShellSample, ...], cameras: tuple[OrthoCamera, ...], visibility: tuple[tuple[int, ...], ...]) -> ObservationEvidenceIR:
    evidence_samples = []
    groups: dict[str, list[str]] = {}
    anchors: dict[str, dict] = {}
    mode_sigma: dict[str, float] = {}
    for sample in samples:
        views = visibility[sample.sample_index]
        if not views:
            continue
        gid = f"G{sample.sample_index:03d}"
        obs_ids = []
        for view in views:
            cam = cameras[view]
            x, y, depth, px, py = cam.project_world(sample.P)
            obs_id = f"U1:V{view}:S{sample.sample_index:03d}:M0"
            evidence_samples.append(ObservationSample(
                observation_id=obs_id,
                view_index=view,
                raster_xy=(float(px), float(py)),
                ray_origin=cam.ray_origin_for_projected(x, y),
                ray_forward=cam.forward,
                depth=float(depth),
                support=True,
                provenance_ref=f"U1_OBSERVATION_ORACLE:{witness.name}:S{sample.sample_index:03d}:V{view}",
                validity_flags=(),
            ))
            obs_ids.append(obs_id)
        groups[gid] = obs_ids
        anchor_view = min(views)
        _, _, _, apx, apy = cameras[anchor_view].project_world(sample.P)
        anchors[gid] = {"view_index": int(anchor_view), "raster_xy": (float(apx), float(apy))}
        mode_sigma[f"S{sample.sample_index:03d}:M0"] = 0.0
    return ObservationEvidenceIR(
        tuple(evidence_samples),
        metadata={
            "hypothesis_groups": groups,
            "hypothesis_anchor_raster": anchors,
            "mode_sigma": mode_sigma,
            "raster_coordinate_system": "PIXEL_CENTER_XY",
            "resolution": RASTER_RESOLUTION,
            "oracle_arm": "U1_OBSERVATION_ORACLE_SUBSTRATE",
            "hidden_surface_completion": False,
            "source_mesh_consumer_input": False,
        },
    )


def build_u1_surface(witness: Witness, samples: tuple[ShellSample, ...], cameras: tuple[OrthoCamera, ...], visibility: tuple[tuple[int, ...], ...]):
    evidence = build_u1_evidence(witness, samples, cameras, visibility)
    surface = compile_surface_v2(evidence, max_common_frame_error=PERSISTENCE_TOLERANCE)
    surface = attach_dtb_nd1_from_evidence(evidence, surface)
    return surface, evidence


def substrate_telemetry(u0: RiggingSurfaceIR, u1: RiggingSurfaceIR, visibility: tuple[tuple[int, ...], ...]) -> dict:
    visible_count = sum(bool(v) for v in visibility)
    support_counts = [len(n.support_views) for n in u1.surface_nodes]
    normal_count = sum(n.derived_normal is not None for n in u1.surface_nodes)
    return {
        "u0_node_count": len(u0.surface_nodes),
        "u1_node_count": len(u1.surface_nodes),
        "visible_full_sample_count": visible_count,
        "visible_fraction": float(visible_count / max(len(u0.surface_nodes), 1)),
        "u1_normal_count": normal_count,
        "u1_normal_fraction": float(normal_count / max(len(u1.surface_nodes), 1)),
        "u1_support_view_min": min(support_counts, default=0),
        "u1_support_view_mean": float(np.mean(support_counts)) if support_counts else 0.0,
        "u1_support_view_max": max(support_counts, default=0),
        "u1_local_relation_count": len(u1.local_relations),
    }
