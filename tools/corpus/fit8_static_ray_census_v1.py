from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from tools.corpus.fit8_rendered_mechanical_census_contract_v1 import (
    SurfaceClassVisibilitySweepV1,
    ViewVisibilityStatsV1,
)

SCHEMA_VERSION = "RealSaS.FIT8StaticRayCensus.v1"
VIEW_ORDER = ("S", "SE", "E", "NE", "N", "NW", "W", "SW")
VIEW_YAW_DEG = (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)
SURFACE_CLASS_TO_CODE = {"CLOSED": 0, "OPEN": 1, "NONMANIFOLD": 2, "AMBIGUOUS": 3}
CODE_TO_SURFACE_CLASS = {value: key for key, value in SURFACE_CLASS_TO_CODE.items()}


def _canonical_json_sha256(payload: Mapping) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


@dataclass(frozen=True)
class CanonicalOrbitFrameV1:
    center_xyz: tuple[float, float, float]
    half_extent: float
    frame_scale: float
    source_bbox_min_xyz: tuple[float, float, float]
    source_bbox_max_xyz: tuple[float, float, float]
    max_screen_abs_before_margin: float
    max_xyz_abs_before_margin: float
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("FIT8_STATIC_FRAME_SCHEMA_INVALID")
        if len(self.center_xyz) != 3 or not all(math.isfinite(v) for v in self.center_xyz):
            raise ValueError("FIT8_STATIC_FRAME_CENTER_INVALID")
        if not math.isfinite(self.half_extent) or self.half_extent <= 0.0:
            raise ValueError("FIT8_STATIC_FRAME_HALF_INVALID")
        if not math.isfinite(self.frame_scale) or self.frame_scale <= 1.0:
            raise ValueError("FIT8_STATIC_FRAME_SCALE_INVALID")
        if self.max_screen_abs_before_margin <= 0.0 or self.max_xyz_abs_before_margin <= 0.0:
            raise ValueError("FIT8_STATIC_FRAME_EXTENT_INVALID")

    def canonical_payload(self) -> dict:
        self.validate()
        return asdict(self)

    def sha256(self) -> str:
        return _canonical_json_sha256(self.canonical_payload())


@dataclass(frozen=True)
class StaticRayCensusPolicyV1:
    resolution: int
    frame_scale: float = 1.12
    barycentric_epsilon: float = 1.0e-10
    depth_merge_tolerance_fraction_of_extent: float = 1.0e-10
    provenance: str = "FIT8_STATIC_RAY_CENSUS_V1"

    def validate(self) -> None:
        if self.resolution < 32:
            raise ValueError("FIT8_STATIC_RAY_RESOLUTION_TOO_SMALL")
        if not math.isfinite(self.frame_scale) or self.frame_scale <= 1.0:
            raise ValueError("FIT8_STATIC_RAY_FRAME_SCALE_INVALID")
        if not math.isfinite(self.barycentric_epsilon) or self.barycentric_epsilon < 0.0:
            raise ValueError("FIT8_STATIC_RAY_BARY_EPS_INVALID")
        if not math.isfinite(self.depth_merge_tolerance_fraction_of_extent) or self.depth_merge_tolerance_fraction_of_extent < 0.0:
            raise ValueError("FIT8_STATIC_RAY_DEPTH_EPS_INVALID")
        if not self.provenance:
            raise ValueError("FIT8_STATIC_RAY_PROVENANCE_MISSING")

    def sha256(self) -> str:
        self.validate()
        return _canonical_json_sha256(asdict(self))


def _basis(yaw_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta = math.radians(float(yaw_deg))
    radial = np.asarray([math.sin(theta), -math.cos(theta), 0.0], dtype=np.float64)
    radial /= np.linalg.norm(radial)
    forward = -radial
    up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    if abs(float(np.dot(forward, up))) > 1.0e-12 or abs(float(np.dot(right, up))) > 1.0e-12:
        raise RuntimeError("FIT8_STATIC_RAY_CAMERA_BASIS_DRIFT")
    return radial, forward, right, up


def solve_canonical_orbit_frame_v1(vertices: np.ndarray, *, frame_scale: float = 1.12) -> CanonicalOrbitFrameV1:
    xyz = np.asarray(vertices, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError("FIT8_STATIC_RAY_VERTICES_INVALID")
    if not math.isfinite(frame_scale) or frame_scale <= 1.0:
        raise ValueError("FIT8_STATIC_RAY_FRAME_SCALE_INVALID")
    lo = xyz.min(axis=0)
    hi = xyz.max(axis=0)
    center = (lo + hi) * 0.5
    q = xyz - center[None, :]
    max_screen = 0.0
    for yaw in VIEW_YAW_DEG:
        _, _, right, up = _basis(yaw)
        max_screen = max(
            max_screen,
            float(np.max(np.abs(q @ right))),
            float(np.max(np.abs(q @ up))),
        )
    max_xyz = float(np.max(np.abs(q)))
    half = max(max_screen, max_xyz) * float(frame_scale)
    value = CanonicalOrbitFrameV1(
        center_xyz=tuple(float(v) for v in center),
        half_extent=float(half),
        frame_scale=float(frame_scale),
        source_bbox_min_xyz=tuple(float(v) for v in lo),
        source_bbox_max_xyz=tuple(float(v) for v in hi),
        max_screen_abs_before_margin=float(max_screen),
        max_xyz_abs_before_margin=float(max_xyz),
    )
    value.validate()
    return value


def face_edge_components_v1(faces: np.ndarray) -> np.ndarray:
    tri = np.asarray(faces, dtype=np.int64)
    if tri.ndim != 2 or tri.shape[1] != 3 or not len(tri) or np.any(tri < 0):
        raise ValueError("FIT8_STATIC_RAY_FACES_INVALID")
    edge_map: dict[tuple[int, int], list[int]] = {}
    for fi, row in enumerate(tri.tolist()):
        for a, b in ((row[0], row[1]), (row[1], row[2]), (row[2], row[0])):
            edge_map.setdefault(tuple(sorted((int(a), int(b)))), []).append(fi)
    parent = np.arange(len(tri), dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            if rb < ra:
                ra, rb = rb, ra
            parent[rb] = ra

    for rows in edge_map.values():
        if len(rows) > 1:
            seed = rows[0]
            for other in rows[1:]:
                union(seed, other)
    roots = [find(i) for i in range(len(tri))]
    root_to_id = {root: idx for idx, root in enumerate(sorted(set(roots)))}
    return np.asarray([root_to_id[root] for root in roots], dtype=np.int32)


def component_surface_classes_v1(faces: np.ndarray, face_component_id: np.ndarray) -> dict[int, str]:
    tri = np.asarray(faces, dtype=np.int64)
    comp = np.asarray(face_component_id, dtype=np.int64)
    if comp.shape != (len(tri),):
        raise ValueError("FIT8_STATIC_RAY_COMPONENT_SHAPE_INVALID")
    out: dict[int, str] = {}
    for cid in sorted(set(int(v) for v in comp.tolist())):
        rows = tri[comp == cid]
        edge_counts: dict[tuple[int, int], int] = {}
        for row in rows.tolist():
            for a, b in ((row[0], row[1]), (row[1], row[2]), (row[2], row[0])):
                key = tuple(sorted((int(a), int(b))))
                edge_counts[key] = edge_counts.get(key, 0) + 1
        counts = tuple(edge_counts.values())
        if any(value > 2 for value in counts):
            out[cid] = "NONMANIFOLD"
        elif any(value == 1 for value in counts):
            out[cid] = "OPEN"
        elif counts and all(value == 2 for value in counts):
            out[cid] = "CLOSED"
        else:
            out[cid] = "AMBIGUOUS"
    return out


def _validate_face_labels(name: str, values: np.ndarray, face_count: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.int32)
    if array.shape != (face_count,):
        raise ValueError(f"FIT8_STATIC_RAY_LABEL_SHAPE_INVALID:{name}")
    if np.any(array < -1):
        raise ValueError(f"FIT8_STATIC_RAY_LABEL_VALUE_INVALID:{name}")
    return array


def _face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangle = vertices[faces]
    normal = np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
    length = np.linalg.norm(normal, axis=1)
    if np.any(length <= 1.0e-14):
        raise ValueError("FIT8_STATIC_RAY_DEGENERATE_FACE")
    return normal / length[:, None]


def _raster_hits_one_view(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    face_component_id: np.ndarray,
    face_material_id: np.ndarray,
    face_owner_id: np.ndarray,
    frame: CanonicalOrbitFrameV1,
    yaw_deg: float,
    policy: StaticRayCensusPolicyV1,
) -> dict[str, np.ndarray | dict]:
    res = int(policy.resolution)
    center = np.asarray(frame.center_xyz, dtype=np.float64)
    _, forward, right, up = _basis(yaw_deg)
    q = vertices - center[None, :]
    sx = q @ right
    sy = q @ up
    depth = q @ forward
    half = float(frame.half_extent)
    px = (sx / half + 1.0) * 0.5 * res - 0.5
    py = (sy / half + 1.0) * 0.5 * res - 0.5
    screen = np.stack((px, py), axis=1)
    normals = _face_normals(vertices, faces)

    hit_pixels: list[np.ndarray] = []
    hit_depths: list[np.ndarray] = []
    hit_triangles: list[np.ndarray] = []
    eps = float(policy.barycentric_epsilon)

    for fi, (ia, ib, ic) in enumerate(faces.tolist()):
        a = screen[ia]
        b = screen[ib]
        c = screen[ic]
        x0 = max(0, int(math.ceil(min(a[0], b[0], c[0]) - eps)))
        x1 = min(res - 1, int(math.floor(max(a[0], b[0], c[0]) + eps)))
        y0 = max(0, int(math.ceil(min(a[1], b[1], c[1]) - eps)))
        y1 = min(res - 1, int(math.floor(max(a[1], b[1], c[1]) + eps)))
        if x1 < x0 or y1 < y0:
            continue
        ab = b - a
        ac = c - a
        denom = ab[0] * ac[1] - ab[1] * ac[0]
        if abs(float(denom)) <= 1.0e-15:
            continue
        xs = np.arange(x0, x1 + 1, dtype=np.float64)
        ys = np.arange(y0, y1 + 1, dtype=np.float64)
        gx, gy = np.meshgrid(xs, ys, indexing="xy")
        p = np.stack((gx.ravel(), gy.ravel()), axis=1)
        ap = p - a[None, :]
        v = (ap[:, 0] * ac[1] - ap[:, 1] * ac[0]) / denom
        w = (ab[0] * ap[:, 1] - ab[1] * ap[:, 0]) / denom
        u = 1.0 - v - w
        inside = (u >= -eps) & (v >= -eps) & (w >= -eps)
        if not np.any(inside):
            continue
        p = p[inside]
        u = u[inside]
        v = v[inside]
        w = w[inside]
        z = u * depth[ia] + v * depth[ib] + w * depth[ic]
        ix = p[:, 0].astype(np.int64)
        iy = p[:, 1].astype(np.int64)
        hit_pixels.append((iy * res + ix).astype(np.int64))
        hit_depths.append(z.astype(np.float64))
        hit_triangles.append(np.full(len(z), fi, dtype=np.int32))

    if not hit_pixels:
        raise ValueError("FIT8_STATIC_RAY_EMPTY_RASTER")
    pixel = np.concatenate(hit_pixels)
    z = np.concatenate(hit_depths)
    triangle_id = np.concatenate(hit_triangles)
    order = np.lexsort((triangle_id.astype(np.int64), z, pixel))
    pixel = pixel[order]
    z = z[order]
    triangle_id = triangle_id[order]

    depth_eps = float(frame.half_extent) * float(policy.depth_merge_tolerance_fraction_of_extent)
    keep = np.ones(len(pixel), dtype=bool)
    if len(pixel) > 1 and depth_eps >= 0.0:
        same_pixel = pixel[1:] == pixel[:-1]
        same_depth = np.abs(z[1:] - z[:-1]) <= depth_eps
        same_component = face_component_id[triangle_id[1:]] == face_component_id[triangle_id[:-1]]
        keep[1:] = ~(same_pixel & same_depth & same_component)
    pixel = pixel[keep]
    z = z[keep]
    triangle_id = triangle_id[keep]

    counts = np.bincount(pixel, minlength=res * res).astype(np.int64)
    offsets = np.empty(res * res + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    foreground = counts > 0
    first = offsets[:-1][foreground]
    front_triangle = np.full(res * res, -1, dtype=np.int32)
    front_depth = np.full(res * res, np.nan, dtype=np.float64)
    front_triangle[foreground] = triangle_id[first]
    front_depth[foreground] = z[first]
    front_component = np.full(res * res, -1, dtype=np.int32)
    front_material = np.full(res * res, -1, dtype=np.int32)
    front_owner = np.full(res * res, -1, dtype=np.int32)
    front_normal = np.full((res * res, 3), np.nan, dtype=np.float32)
    front_component[foreground] = face_component_id[front_triangle[foreground]]
    front_material[foreground] = face_material_id[front_triangle[foreground]]
    front_owner[foreground] = face_owner_id[front_triangle[foreground]]
    front_normal[foreground] = normals[front_triangle[foreground]].astype(np.float32)

    pair_counts: dict[tuple[int, int], int] = {}
    stack_sizes: list[int] = []
    for pix in np.flatnonzero(foreground).tolist():
        start, end = int(offsets[pix]), int(offsets[pix + 1])
        raw = face_component_id[triangle_id[start:end]].tolist()
        ordered_unique: list[int] = []
        seen: set[int] = set()
        for cid in raw:
            value = int(cid)
            if value not in seen:
                seen.add(value)
                ordered_unique.append(value)
        stack_sizes.append(len(ordered_unique))
        for i, a in enumerate(ordered_unique):
            for b in ordered_unique[i + 1 :]:
                pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1

    pair_rows = np.asarray(
        [[a, b, count] for (a, b), count in sorted(pair_counts.items())],
        dtype=np.int64,
    ) if pair_counts else np.empty((0, 3), dtype=np.int64)
    mixed_pairs: list[tuple[int, int, int, int]] = []
    unordered_keys = sorted({tuple(sorted((int(a), int(b)))) for a, b in pair_counts})
    for lo, hi in unordered_keys:
        forward_count = int(pair_counts.get((lo, hi), 0))
        reverse_count = int(pair_counts.get((hi, lo), 0))
        if forward_count and reverse_count:
            mixed_pairs.append((lo, hi, forward_count, reverse_count))
    mixed_rows = np.asarray(mixed_pairs, dtype=np.int64) if mixed_pairs else np.empty((0, 4), dtype=np.int64)

    return {
        "hit_offsets": offsets,
        "hit_depth": z.astype(np.float32),
        "hit_triangle_id": triangle_id.astype(np.int32),
        "hit_component_id": face_component_id[triangle_id].astype(np.int32),
        "hit_material_id": face_material_id[triangle_id].astype(np.int32),
        "hit_owner_id": face_owner_id[triangle_id].astype(np.int32),
        "front_depth": front_depth.reshape(res, res).astype(np.float32),
        "front_triangle_id": front_triangle.reshape(res, res),
        "front_component_id": front_component.reshape(res, res),
        "front_material_id": front_material.reshape(res, res),
        "front_owner_id": front_owner.reshape(res, res),
        "front_normal_xyz": front_normal.reshape(res, res, 3),
        "ordered_component_pair_counts": pair_rows,
        "mixed_component_pair_counts": mixed_rows,
        "stack_sizes": np.asarray(stack_sizes, dtype=np.int16),
    }


def build_static_ray_census_v1(
    *,
    subject_key: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    face_material_id: np.ndarray | None = None,
    face_owner_id: np.ndarray | None = None,
    face_component_id: np.ndarray | None = None,
    policy: StaticRayCensusPolicyV1,
) -> tuple[dict, dict[str, np.ndarray]]:
    policy.validate()
    if not subject_key:
        raise ValueError("FIT8_STATIC_RAY_SUBJECT_MISSING")
    xyz = np.asarray(vertices, dtype=np.float64)
    tri = np.asarray(faces, dtype=np.int64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError("FIT8_STATIC_RAY_VERTICES_INVALID")
    if tri.ndim != 2 or tri.shape[1] != 3 or not len(tri) or np.any(tri < 0) or np.any(tri >= len(xyz)):
        raise ValueError("FIT8_STATIC_RAY_FACES_INVALID")
    component = face_edge_components_v1(tri) if face_component_id is None else _validate_face_labels("component", face_component_id, len(tri))
    material = np.zeros(len(tri), dtype=np.int32) if face_material_id is None else _validate_face_labels("material", face_material_id, len(tri))
    owner = np.full(len(tri), -1, dtype=np.int32) if face_owner_id is None else _validate_face_labels("owner", face_owner_id, len(tri))
    surface_classes = component_surface_classes_v1(tri, component)
    frame = solve_canonical_orbit_frame_v1(xyz, frame_scale=policy.frame_scale)

    arrays: dict[str, np.ndarray] = {}
    view_stats: list[ViewVisibilityStatsV1] = []
    class_visible_totals = {key: 0 for key in SURFACE_CLASS_TO_CODE}
    total_foreground = 0
    pair_evidence: dict[str, dict] = {}
    for view_index, yaw in enumerate(VIEW_YAW_DEG):
        result = _raster_hits_one_view(
            vertices=xyz,
            faces=tri,
            face_component_id=component,
            face_material_id=material,
            face_owner_id=owner,
            frame=frame,
            yaw_deg=yaw,
            policy=policy,
        )
        prefix = f"v{view_index}_"
        for key, value in result.items():
            if key != "stack_sizes":
                arrays[prefix + key] = np.asarray(value)
        front_component = np.asarray(result["front_component_id"])
        front_material = np.asarray(result["front_material_id"])
        foreground = front_component >= 0
        fg_count = int(np.count_nonzero(foreground))
        total_foreground += fg_count
        visible_components = set(int(v) for v in front_component[foreground].tolist())
        visible_materials = set(int(v) for v in front_material[foreground].tolist() if int(v) >= 0)
        stack = np.asarray(result["stack_sizes"], dtype=np.int64)
        pair_rows = np.asarray(result["ordered_component_pair_counts"], dtype=np.int64)
        for cid in front_component[foreground].tolist():
            class_visible_totals[surface_classes[int(cid)]] += 1
        row = ViewVisibilityStatsV1(
            view_index=view_index,
            foreground_pixels=fg_count,
            visible_component_count=len(visible_components),
            visible_material_region_count=len(visible_materials),
            max_distinct_component_stack=int(stack.max()),
            p95_distinct_component_stack=float(np.percentile(stack, 95.0)),
            pairwise_ordered_component_relation_count=int(len(pair_rows)),
        )
        row.validate()
        view_stats.append(row)
        pair_evidence[str(view_index)] = {
            "mixed_pair_count": int(len(np.asarray(result["mixed_component_pair_counts"]))),
            "mixed_pairs": np.asarray(result["mixed_component_pair_counts"], dtype=np.int64).tolist(),
        }

    visible_fraction = {
        key: float(class_visible_totals[key] / total_foreground)
        for key in sorted(class_visible_totals)
    }
    sweep = SurfaceClassVisibilitySweepV1(
        resolution=policy.resolution,
        visible_fraction_by_surface_class=visible_fraction,
    )
    sweep.validate()
    camera_contract = {
        "contract": "realsas.level_orthographic_z_orbit.v1",
        "view_order": list(VIEW_ORDER),
        "yaw_deg": list(VIEW_YAW_DEG),
        "up": [0.0, 0.0, 1.0],
        "center_xyz": list(frame.center_xyz),
        "half_extent": frame.half_extent,
        "resolution": policy.resolution,
    }
    summary = {
        "schema": SCHEMA_VERSION,
        "subject_key": subject_key,
        "policy": asdict(policy),
        "policy_sha256": policy.sha256(),
        "frame": frame.canonical_payload(),
        "frame_sha256": frame.sha256(),
        "camera_contract": camera_contract,
        "camera_contract_sha256": _canonical_json_sha256(camera_contract),
        "component_count": int(len(set(component.tolist()))),
        "material_region_count": int(len(set(int(v) for v in material.tolist() if int(v) >= 0))),
        "owner_region_count": int(len(set(int(v) for v in owner.tolist() if int(v) >= 0))),
        "surface_class_by_component": {str(k): v for k, v in sorted(surface_classes.items())},
        "static_views": [asdict(row) for row in view_stats],
        "surface_class_visibility": asdict(sweep),
        "pair_evidence": pair_evidence,
        "claims": {
            "source_rgba_material_parity": False,
            "mechanical_owner_authority": bool(np.any(owner >= 0)),
            "dynamic_mechanics": False,
            "static_geometry_visibility_only": True,
        },
    }
    arrays["face_component_id"] = component.astype(np.int32)
    arrays["face_material_id"] = material.astype(np.int32)
    arrays["face_owner_id"] = owner.astype(np.int32)
    arrays["component_surface_class_code"] = np.asarray(
        [SURFACE_CLASS_TO_CODE[surface_classes[cid]] for cid in range(len(surface_classes))],
        dtype=np.int8,
    )
    return summary, arrays


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--carrier-npz", required=True, type=Path)
    parser.add_argument("--subject-key", required=True)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--frame-scale", type=float, default=1.12)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    with np.load(args.carrier_npz, allow_pickle=False) as archive:
        vertices = archive["vertices"]
        faces = archive["faces"]
        material = archive["face_material_id"] if "face_material_id" in archive.files else None
        owner = archive["face_owner_id"] if "face_owner_id" in archive.files else None
        component = archive["face_component_id"] if "face_component_id" in archive.files else None
    summary, arrays = build_static_ray_census_v1(
        subject_key=args.subject_key,
        vertices=vertices,
        faces=faces,
        face_material_id=material,
        face_owner_id=owner,
        face_component_id=component,
        policy=StaticRayCensusPolicyV1(
            resolution=args.resolution,
            frame_scale=args.frame_scale,
            provenance="CLI_STATIC_RAY_CENSUS_V1",
        ),
    )
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **arrays)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


__all__ = [
    "CanonicalOrbitFrameV1",
    "StaticRayCensusPolicyV1",
    "build_static_ray_census_v1",
    "component_surface_classes_v1",
    "face_edge_components_v1",
    "solve_canonical_orbit_frame_v1",
]

if __name__ == "__main__":
    main()
