from __future__ import annotations

"""Build source-defensible all-hit relation evidence for Fit8.

This producer is deliberately upstream of any V10 training corpus. For each
admitted Stage01 orthographic camera it samples deterministic foreground rays,
recovers the complete ordered rest-mesh hit stack, and emits every unordered hit
pair in both orientations.

The evidence separates two concepts that Relation Truth V2 coupled together:

* ``front`` is pairwise depth order;
* ``visibility`` is whether the sampled point is the globally frontmost hit on
  that ray.

For a deep pair B/C behind A, B can be in front of C while both B and C are not
visible. Those rows are the principal new evidence family this producer exists
to expose.

Pair points are normalized with the exact legacy IRIS ``center_xyz`` and
``normalization_half_extent`` used by the prepared cohort. Teacher-bbox
normalization is forbidden because it would put correct source evidence into the
wrong frozen-H query coordinate system.

No static owner/cut/attachment truth is inferred. No projected RGB/alpha
boundary label is assigned to hidden pairs. Occlusion evidence here is positive
same-ray overlap support only; this file does not manufacture negative examples
or authorize a binary occlusion court.

This tool never authorizes optimizer steps.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.iris.vnext.cohort_v1 import SUBJECT_IDS
from tools.iris_vnext.build_fit8_vnext_shards_v1 import (
    SUBJECT_SOURCE_IDS,
    _camera_rays,
    _normalize_source_world,
    _seed,
    _sha,
)
from tools.iris_vnext.build_fit8_vnext_shards_v2 import _parallel_projection_basis

_SCHEMA = "RealSaS.IRISVNextFit8AllHitRelationEvidence.v1"
_MANIFEST_SCHEMA = "RealSaS.IRISVNextFit8AllHitRelationEvidenceManifest.v1"
_STATUS = "MATERIALIZED__EVIDENCE_ONLY__NO_TRAINING_AUTHORIZATION"
_GRID_SIZE = 32


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-shard-dir", type=Path, required=True)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--stage01-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-rays-per-view", type=int, default=8192)
    return parser.parse_args()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _source_provenance() -> dict[str, str]:
    repo = Path(__file__).resolve().parents[2]
    paths = (
        Path(__file__).resolve(),
        repo / "models/iris/vnext/cohort_v1.py",
        repo / "tools/iris_vnext/build_fit8_vnext_shards_v1.py",
        repo / "tools/iris_vnext/build_fit8_vnext_shards_v2.py",
    )
    return {str(path.relative_to(repo)): _sha(path) for path in paths}


def _face_edge_components(faces: np.ndarray) -> np.ndarray:
    tri = np.asarray(faces, dtype=np.int64)
    if tri.ndim != 2 or tri.shape[1] != 3 or not len(tri) or np.any(tri < 0):
        raise ValueError("VNEXT_ALL_HIT_FACES_INVALID")
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for face_index, row in enumerate(tri.tolist()):
        for a, b in ((row[0], row[1]), (row[1], row[2]), (row[2], row[0])):
            edge_faces.setdefault(tuple(sorted((int(a), int(b)))), []).append(face_index)
    parent = np.arange(len(tri), dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rb < ra:
            ra, rb = rb, ra
        parent[rb] = ra

    for rows in edge_faces.values():
        if len(rows) > 1:
            for other in rows[1:]:
                union(rows[0], other)
    roots = [find(i) for i in range(len(tri))]
    mapping = {root: idx for idx, root in enumerate(sorted(set(roots)))}
    return np.asarray([mapping[root] for root in roots], dtype=np.int32)


def _ray_triangle_all_parallel_grid(
    origins: np.ndarray,
    directions: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    face_component: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return deterministic CSR all-hit stacks for parallel rays.

    Candidate-face acceleration matches the existing V2 projected 32x32 grid.
    Same-component hits at numerically identical depth are merged to avoid shared
    triangle-edge duplicates; distinct depths from the same component remain.
    """

    o = np.asarray(origins, dtype=np.float64)
    d = np.asarray(directions, dtype=np.float64)
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    comp = np.asarray(face_component, dtype=np.int32)
    if o.ndim != 2 or o.shape[1] != 3 or d.shape != o.shape or len(o) == 0:
        raise ValueError("VNEXT_ALL_HIT_RAY_SHAPE_INVALID")
    if v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("VNEXT_ALL_HIT_MESH_SHAPE_INVALID")
    if comp.shape != (len(f),):
        raise ValueError("VNEXT_ALL_HIT_COMPONENT_SHAPE_INVALID")
    if not np.isfinite(o).all() or not np.isfinite(d).all() or not np.isfinite(v).all():
        raise ValueError("VNEXT_ALL_HIT_NONFINITE")
    if np.any(f < 0) or np.any(f >= len(v)):
        raise ValueError("VNEXT_ALL_HIT_FACE_INDEX_INVALID")
    if not np.allclose(d, d[0][None, :], rtol=1.0e-12, atol=1.0e-12):
        raise ValueError("VNEXT_ALL_HIT_REQUIRES_PARALLEL_RAYS")

    unit_d, direction_norm, axis_u, axis_v = _parallel_projection_basis(d[0])
    ray_u = o @ axis_u
    ray_v = o @ axis_v
    ray_w = o @ unit_d
    vertex_u = v @ axis_u
    vertex_v = v @ axis_v
    vertex_w = v @ unit_d
    tri_u = vertex_u[f]
    tri_v = vertex_v[f]
    tri_w = vertex_w[f]
    u0, u1, u2 = tri_u[:, 0], tri_u[:, 1], tri_u[:, 2]
    v0, v1, v2 = tri_v[:, 0], tri_v[:, 1], tri_v[:, 2]
    den = (v1 - v2) * (u0 - u2) + (u2 - u1) * (v0 - v2)
    face_valid = np.abs(den) * direction_norm > 1.0e-10
    min_u, max_u = np.min(tri_u, axis=1), np.max(tri_u, axis=1)
    min_v, max_v = np.min(tri_v, axis=1), np.max(tri_v, axis=1)

    projection_scale = max(
        1.0,
        float(np.max(np.abs(tri_u))),
        float(np.max(np.abs(tri_v))),
        float(np.max(np.abs(ray_u))),
        float(np.max(np.abs(ray_v))),
    )
    projection_eps = 1.0e-8 * projection_scale
    grid_size = int(_GRID_SIZE)
    lo_u, hi_u = float(np.min(ray_u) - projection_eps), float(np.max(ray_u) + projection_eps)
    lo_v, hi_v = float(np.min(ray_v) - projection_eps), float(np.max(ray_v) + projection_eps)
    if hi_u <= lo_u:
        lo_u -= 1.0e-9
        hi_u += 1.0e-9
    if hi_v <= lo_v:
        lo_v -= 1.0e-9
        hi_v += 1.0e-9
    scale_u = grid_size / (hi_u - lo_u)
    scale_v = grid_size / (hi_v - lo_v)

    cells: list[list[int]] = [[] for _ in range(grid_size * grid_size)]
    relevant = np.flatnonzero(
        face_valid
        & (max_u >= lo_u)
        & (min_u <= hi_u)
        & (max_v >= lo_v)
        & (min_v <= hi_v)
    )
    for face_index in relevant.tolist():
        gx0 = max(0, min(grid_size - 1, int(np.floor((min_u[face_index] - projection_eps - lo_u) * scale_u))))
        gx1 = max(0, min(grid_size - 1, int(np.floor((max_u[face_index] + projection_eps - lo_u) * scale_u))))
        gy0 = max(0, min(grid_size - 1, int(np.floor((min_v[face_index] - projection_eps - lo_v) * scale_v))))
        gy1 = max(0, min(grid_size - 1, int(np.floor((max_v[face_index] + projection_eps - lo_v) * scale_v))))
        for gy in range(gy0, gy1 + 1):
            for gx in range(gx0, gx1 + 1):
                cells[gy * grid_size + gx].append(int(face_index))
    cell_arrays = tuple(np.asarray(row, dtype=np.int64) for row in cells)

    gx = np.clip(np.floor((ray_u - lo_u) * scale_u).astype(np.int64), 0, grid_size - 1)
    gy = np.clip(np.floor((ray_v - lo_v) * scale_v).astype(np.int64), 0, grid_size - 1)
    ray_cell = gy * grid_size + gx
    ray_order = np.argsort(ray_cell, kind="stable")
    sorted_cells = ray_cell[ray_order]
    starts = np.r_[0, np.flatnonzero(sorted_cells[1:] != sorted_cells[:-1]) + 1, len(sorted_cells)]

    hit_faces_by_ray: list[np.ndarray] = [np.zeros(0, dtype=np.int64) for _ in range(len(o))]
    hit_depths_by_ray: list[np.ndarray] = [np.zeros(0, dtype=np.float64) for _ in range(len(o))]
    depth_scale = max(1.0, float(np.max(v) - np.min(v)), float(np.max(np.abs(v))))
    depth_merge_eps = depth_scale * 1.0e-10

    for group_index in range(len(starts) - 1):
        start, end = int(starts[group_index]), int(starts[group_index + 1])
        rays = ray_order[start:end]
        candidates = cell_arrays[int(sorted_cells[start])]
        if not len(candidates):
            continue
        pu = ray_u[rays][:, None]
        pv = ray_v[rays][:, None]
        bbox = (
            (pu >= min_u[candidates][None, :] - projection_eps)
            & (pu <= max_u[candidates][None, :] + projection_eps)
            & (pv >= min_v[candidates][None, :] - projection_eps)
            & (pv <= max_v[candidates][None, :] + projection_eps)
        )
        dd = den[candidates][None, :]
        w0 = (
            (v1[candidates] - v2[candidates])[None, :] * (pu - u2[candidates][None, :])
            + (u2[candidates] - u1[candidates])[None, :] * (pv - v2[candidates][None, :])
        ) / dd
        w1 = (
            (v2[candidates] - v0[candidates])[None, :] * (pu - u2[candidates][None, :])
            + (u0[candidates] - u2[candidates])[None, :] * (pv - v2[candidates][None, :])
        ) / dd
        w2 = 1.0 - w0 - w1
        inside = bbox & (w0 >= -1.0e-9) & (w1 >= -1.0e-9) & (w2 >= -1.0e-9)
        depth = (
            w0 * tri_w[candidates, 0][None, :]
            + w1 * tri_w[candidates, 1][None, :]
            + w2 * tri_w[candidates, 2][None, :]
            - ray_w[rays][:, None]
        ) / direction_norm
        depth = np.where(inside & (depth > 1.0e-8), depth, np.inf)

        for local_index, ray_index in enumerate(rays.tolist()):
            finite = np.flatnonzero(np.isfinite(depth[local_index]))
            if not len(finite):
                continue
            face_ids = candidates[finite]
            depths = depth[local_index, finite]
            order = np.lexsort((face_ids, depths))
            face_ids = face_ids[order]
            depths = depths[order]
            keep_faces: list[int] = []
            keep_depths: list[float] = []
            for face_id, hit_depth in zip(face_ids.tolist(), depths.tolist()):
                if keep_faces:
                    previous_face = keep_faces[-1]
                    previous_depth = keep_depths[-1]
                    if (
                        abs(float(hit_depth) - float(previous_depth)) <= depth_merge_eps
                        and int(comp[face_id]) == int(comp[previous_face])
                    ):
                        continue
                keep_faces.append(int(face_id))
                keep_depths.append(float(hit_depth))
            hit_faces_by_ray[ray_index] = np.asarray(keep_faces, dtype=np.int64)
            hit_depths_by_ray[ray_index] = np.asarray(keep_depths, dtype=np.float64)

    offsets = np.zeros(len(o) + 1, dtype=np.int64)
    for ray_index, row in enumerate(hit_faces_by_ray):
        offsets[ray_index + 1] = offsets[ray_index] + len(row)
    faces_flat = np.concatenate(hit_faces_by_ray) if offsets[-1] else np.zeros(0, dtype=np.int64)
    depth_flat = np.concatenate(hit_depths_by_ray) if offsets[-1] else np.zeros(0, dtype=np.float64)
    ray_ids = np.repeat(np.arange(len(o), dtype=np.int64), np.diff(offsets))
    points_flat = (
        o[ray_ids] + depth_flat[:, None] * d[ray_ids]
        if len(depth_flat)
        else np.zeros((0, 3), dtype=np.float64)
    )
    return {
        "offsets": offsets,
        "face": faces_flat,
        "depth": depth_flat,
        "point": points_flat,
    }


def _sample_pixels(mask_path: Path, resolution: int, *, source_id: str, view_index: int, budget: int) -> np.ndarray:
    if budget <= 0:
        raise ValueError("VNEXT_ALL_HIT_RAY_BUDGET_NONPOSITIVE")
    mask = np.frombuffer(mask_path.read_bytes(), dtype=np.uint8)
    if mask.size != resolution * resolution:
        raise ValueError("VNEXT_ALL_HIT_MASK_SIZE_DRIFT")
    mask = mask.reshape(resolution, resolution)
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("VNEXT_ALL_HIT_MASK_EMPTY")
    rng = np.random.default_rng(_seed(f"relation-all-hit-v1:{source_id}:view:{view_index}"))
    choose = rng.permutation(len(xs))[: min(len(xs), int(budget))]
    return np.stack((xs[choose], ys[choose]), axis=1).astype(np.int64)


def _oriented_pair_evidence(
    *,
    all_hits: dict[str, np.ndarray],
    pixels: np.ndarray,
    center: np.ndarray,
    half: float,
) -> dict[str, np.ndarray]:
    offsets = np.asarray(all_hits["offsets"], dtype=np.int64)
    faces = np.asarray(all_hits["face"], dtype=np.int64)
    points = np.asarray(all_hits["point"], dtype=np.float64)
    if offsets.shape != (len(pixels) + 1,):
        raise ValueError("VNEXT_ALL_HIT_OFFSETS_SHAPE_DRIFT")
    if len(faces) != len(points):
        raise ValueError("VNEXT_ALL_HIT_FLAT_SHAPE_DRIFT")

    point_a: list[np.ndarray] = []
    point_b: list[np.ndarray] = []
    face_a: list[int] = []
    face_b: list[int] = []
    rank_a: list[int] = []
    rank_b: list[int] = []
    ray_index: list[int] = []
    pixel_xy: list[tuple[int, int]] = []
    front: list[float] = []
    visibility_a: list[float] = []
    visibility_b: list[float] = []
    occlusion_positive: list[float] = []
    base_pair_id: list[int] = []

    pair_id = 0
    for ray in range(len(pixels)):
        start, end = int(offsets[ray]), int(offsets[ray + 1])
        stack_count = end - start
        if stack_count < 2:
            continue
        stack_faces = faces[start:end]
        stack_points = points[start:end]
        for near_rank in range(stack_count - 1):
            for far_rank in range(near_rank + 1, stack_count):
                near_point = _normalize_source_world(stack_points[near_rank], center, half).astype(np.float32)
                far_point = _normalize_source_world(stack_points[far_rank], center, half).astype(np.float32)
                for reverse in (False, True):
                    if not reverse:
                        pa, pb = near_point, far_point
                        fa, fb = int(stack_faces[near_rank]), int(stack_faces[far_rank])
                        ra, rb = near_rank, far_rank
                    else:
                        pa, pb = far_point, near_point
                        fa, fb = int(stack_faces[far_rank]), int(stack_faces[near_rank])
                        ra, rb = far_rank, near_rank
                    point_a.append(pa)
                    point_b.append(pb)
                    face_a.append(fa)
                    face_b.append(fb)
                    rank_a.append(int(ra))
                    rank_b.append(int(rb))
                    ray_index.append(int(ray))
                    pixel_xy.append((int(pixels[ray, 0]), int(pixels[ray, 1])))
                    front.append(float(ra < rb))
                    visibility_a.append(float(ra == 0))
                    visibility_b.append(float(rb == 0))
                    occlusion_positive.append(1.0)
                    base_pair_id.append(pair_id)
                pair_id += 1

    count = len(front)
    if count % 2:
        raise RuntimeError("VNEXT_ALL_HIT_ORIENTATION_CARDINALITY_DRIFT")
    return {
        "point_a": np.asarray(point_a, dtype=np.float32).reshape(count, 3),
        "point_b": np.asarray(point_b, dtype=np.float32).reshape(count, 3),
        "face_a": np.asarray(face_a, dtype=np.int64),
        "face_b": np.asarray(face_b, dtype=np.int64),
        "rank_a": np.asarray(rank_a, dtype=np.int16),
        "rank_b": np.asarray(rank_b, dtype=np.int16),
        "ray_index": np.asarray(ray_index, dtype=np.int32),
        "pixel_xy": np.asarray(pixel_xy, dtype=np.int32).reshape(count, 2),
        "front_probability": np.asarray(front, dtype=np.float32),
        "visibility_probability_a": np.asarray(visibility_a, dtype=np.float32),
        "visibility_probability_b": np.asarray(visibility_b, dtype=np.float32),
        "occlusion_positive_support": np.asarray(occlusion_positive, dtype=np.float32),
        "base_pair_id": np.asarray(base_pair_id, dtype=np.int64),
    }


def _view_stats(*, all_hits: dict[str, np.ndarray], pairs: dict[str, np.ndarray], ray_count: int) -> dict[str, int | float]:
    offsets = np.asarray(all_hits["offsets"], dtype=np.int64)
    stack = np.diff(offsets)
    rows = int(len(pairs["front_probability"]))
    if rows:
        front = pairs["front_probability"] >= 0.5
        visa = pairs["visibility_probability_a"] >= 0.5
        visb = pairs["visibility_probability_b"] >= 0.5
        decoupled = int(np.count_nonzero((front & ~visa) | ((~front) & ~visb)))
        deep_pair_rows = int(np.count_nonzero((pairs["rank_a"] > 0) & (pairs["rank_b"] > 0)))
    else:
        decoupled = 0
        deep_pair_rows = 0
    return {
        "sampled_rays": int(ray_count),
        "foreground_hit_rays": int(np.count_nonzero(stack >= 1)),
        "two_or_more_hit_rays": int(np.count_nonzero(stack >= 2)),
        "three_or_more_hit_rays": int(np.count_nonzero(stack >= 3)),
        "maximum_stack_depth": int(stack.max()) if len(stack) else 0,
        "total_hits": int(offsets[-1]),
        "base_pair_count": rows // 2,
        "oriented_pair_rows": rows,
        "deep_hidden_pair_rows": deep_pair_rows,
        "front_visibility_decoupled_rows": decoupled,
        "front_visibility_decoupled_fraction": float(decoupled / rows) if rows else 0.0,
    }


def main() -> None:
    args = _args()
    if int(args.candidate_rays_per_view) <= 0:
        raise ValueError("VNEXT_ALL_HIT_POLICY_NONPOSITIVE")
    legacy_dir = args.legacy_shard_dir.expanduser().resolve()
    teacher_dir = args.teacher_dir.expanduser().resolve()
    stage01 = args.stage01_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    bootstrap_path = stage01 / "BOOTSTRAP_MANIFEST.json"
    bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    bootstrap_rows = {row["subject_id"]: row for row in bootstrap["subjects"]}
    subjects: list[dict[str, Any]] = []

    for display_id in SUBJECT_IDS:
        source_id = SUBJECT_SOURCE_IDS[display_id]
        legacy_path = legacy_dir / f"{source_id}.pt"
        teacher_path = teacher_dir / f"{source_id}.teacher.npz"
        receipt_path = teacher_path.with_suffix(teacher_path.suffix + ".json")
        if not legacy_path.is_file() or not teacher_path.is_file() or not receipt_path.is_file():
            raise FileNotFoundError("VNEXT_ALL_HIT_INPUT_MISSING:" + display_id)
        boot = bootstrap_rows[source_id]
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("subject_id") != source_id:
            raise ValueError("VNEXT_ALL_HIT_TEACHER_SUBJECT_DRIFT:" + display_id)
        if receipt.get("source", {}).get("sha256") != boot["raw_source"]["sha256"]:
            raise ValueError("VNEXT_ALL_HIT_TEACHER_SOURCE_HASH_DRIFT:" + display_id)

        legacy = torch.load(legacy_path, map_location="cpu", weights_only=True)
        if str(legacy.get("subject_id")) != source_id:
            raise ValueError("VNEXT_ALL_HIT_LEGACY_SUBJECT_DRIFT:" + display_id)
        center = np.asarray(legacy["center_xyz"], dtype=np.float64).reshape(3)
        half = float(torch.as_tensor(legacy["normalization_half_extent"]).item())
        if not np.isfinite(center).all() or not np.isfinite(half) or half <= 0.0:
            raise ValueError("VNEXT_ALL_HIT_LEGACY_NORMALIZATION_INVALID:" + display_id)

        with np.load(teacher_path, allow_pickle=False) as data:
            rest = np.asarray(data["rest_vertices_source_world"], dtype=np.float64)
            faces = np.asarray(data["faces"], dtype=np.int64)
        if rest.ndim != 2 or rest.shape[1] != 3 or not np.isfinite(rest).all():
            raise ValueError("VNEXT_ALL_HIT_REST_INVALID:" + display_id)
        if faces.ndim != 2 or faces.shape[1] != 3 or np.any(faces < 0) or np.any(faces >= len(rest)):
            raise ValueError("VNEXT_ALL_HIT_FACES_INVALID:" + display_id)
        face_component = _face_edge_components(faces)

        camera_bundle_path = stage01 / boot["source_scene_evidence"]["camera_bundle"]["path"]
        cameras = list(json.loads(camera_bundle_path.read_text(encoding="utf-8"))["cameras"])
        if len(cameras) != 8:
            raise ValueError("VNEXT_ALL_HIT_CAMERA_COUNT_DRIFT:" + display_id)

        arrays: dict[str, np.ndarray] = {}
        views: list[dict[str, Any]] = []
        observation_root = stage01 / "observations" / source_id
        for view_index, camera in enumerate(cameras):
            resolution = int(camera["resolution"])
            pixels = _sample_pixels(
                observation_root / f"V{view_index}.mask.bin",
                resolution,
                source_id=source_id,
                view_index=view_index,
                budget=int(args.candidate_rays_per_view),
            )
            origins, directions = _camera_rays(camera, pixels)
            all_hits = _ray_triangle_all_parallel_grid(origins, directions, rest, faces, face_component)
            pairs = _oriented_pair_evidence(
                all_hits=all_hits,
                pixels=pixels,
                center=center,
                half=half,
            )
            prefix = f"v{view_index}_"
            arrays[prefix + "pixels_xy"] = pixels.astype(np.int32)
            arrays[prefix + "stack_offsets"] = all_hits["offsets"].astype(np.int64)
            arrays[prefix + "stack_face"] = all_hits["face"].astype(np.int64)
            arrays[prefix + "stack_depth"] = all_hits["depth"].astype(np.float64)
            for name, value in pairs.items():
                arrays[prefix + name] = value
            stats = _view_stats(all_hits=all_hits, pairs=pairs, ray_count=len(pixels))
            stats["view_index"] = int(view_index)
            stats["camera_sha256"] = _canonical_sha(camera)
            stats["mask_sha256"] = _sha(observation_root / f"V{view_index}.mask.bin")
            views.append(stats)

        out_path = output / f"{display_id.lower().replace('-', '').replace(' ', '')}.all_hit_relation_evidence_v1.npz"
        np.savez_compressed(out_path, **arrays)
        subjects.append(
            {
                "subject_id": display_id,
                "source_subject_id": source_id,
                "file": out_path.name,
                "sha256": _sha(out_path),
                "bytes": out_path.stat().st_size,
                "teacher_export_sha256": _sha(teacher_path),
                "teacher_receipt_sha256": _sha(receipt_path),
                "legacy_shard_sha256": _sha(legacy_path),
                "source_asset_sha256": boot["raw_source"]["sha256"],
                "camera_bundle_sha256": _sha(camera_bundle_path),
                "normalization": {
                    "authority": "EXACT_LEGACY_IRIS_SHARD",
                    "center_xyz": [float(x) for x in center],
                    "half_extent": half,
                },
                "views": views,
            }
        )

    manifest = {
        "schema": _MANIFEST_SCHEMA,
        "status": _STATUS,
        "subject_order": list(SUBJECT_IDS),
        "subjects": subjects,
        "policy": {
            "candidate_rays_per_view": int(args.candidate_rays_per_view),
            "grid_size": int(_GRID_SIZE),
            "same_component_same_depth_duplicates_merged": True,
            "all_unordered_stack_pairs_emitted_both_orientations": True,
            "pair_point_coordinate_domain": "EXACT_LEGACY_IRIS_NORMALIZED_SOURCE_WORLD",
        },
        "authority": {
            "front": "DETERMINISTIC_DERIVED__FULL_RAY_STACK_DEPTH_ORDER",
            "visibility": "DETERMINISTIC_DERIVED__GLOBAL_STACK_RANK_ZERO_ONLY",
            "occlusion": "POSITIVE_SUPPORT_ONLY__SAME_RAY_PAIR__NO_BINARY_NEGATIVE_AUTHORITY",
            "projected_boundary": "NOT_EMITTED__KEEP_SEPARATE_CURRENT_VIEW_RGBA_AUTHORITY",
            "static_same_cut_attachment": "NOT_EMITTED__NO_NEW_AUTHORITY",
        },
        "training_authorized": False,
        "scientific_fit_started": False,
        "teacher_inputs_in_inference_forward": False,
        "source_bindings": {
            "stage01_bootstrap_sha256": _sha(bootstrap_path),
        },
        "source_provenance": _source_provenance(),
    }
    manifest["manifest_value_sha256"] = _canonical_sha(manifest)
    manifest_path = output / "ALL_HIT_RELATION_EVIDENCE_V1_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    total_decoupled = sum(
        int(view["front_visibility_decoupled_rows"])
        for subject in subjects
        for view in subject["views"]
    )
    total_rows = sum(
        int(view["oriented_pair_rows"])
        for subject in subjects
        for view in subject["views"]
    )
    print(
        "VNEXT_FIT8_ALL_HIT_RELATION_EVIDENCE_PASS",
        "subjects=8",
        f"oriented_rows={total_rows}",
        f"front_visibility_decoupled_rows={total_decoupled}",
        f"manifest={manifest['manifest_value_sha256']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
