from __future__ import annotations

"""Active IRIS vNext Fit8 cohort builder.

V2 reuses the proven V1 source/mechanical bank machinery. Topology supervision
has an additional training-only authority court: mechanically equivalent authored
seam duplicates are safely welded, remaining TESSA quantization identity
collisions receive a bounded deterministic identity lift, and serialization still
requires zero residual coordinate collisions. Raw authored faces/owners/materials
remain unchanged for appearance, relations and mechanical-demand truth.

The eight source views of one subject are independent deterministic courts. Their
orthographic rays are intersected with an exact-equivalent projected triangle
uniform grid rather than an O(rays*faces) brute-force sweep. First/second face tie
ordering and Moller-Trumbore tolerances remain preserved. Views are materialized
concurrently while output ordering remains V0..V7.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from models.iris.vnext.cohort_v1 import SUBJECT_IDS
from models.iris.vnext.contracts_v1 import IRISVNextConfigV1
from models.iris.vnext.topology_teacher_v1 import canonicalize_vnext_topology_teacher_v1
from models.tessa.v1.training_data_v1 import (
    build_teacher_asset_sequence_v1,
    build_truncated_teacher_windows_v1,
)
import tools.iris_vnext.build_fit8_vnext_shards_v1 as shard_v1
from tools.iris_vnext.build_fit8_vnext_shards_v1 import (
    SUBJECT_SOURCE_IDS,
    _build_view_banks,
    _mechanical_bank,
    _normalize_source_world,
    _pad_topology,
    _sample_surface,
    _seed,
    _sha,
    _tensor,
)

_RAY_GRID_SIZE = 32
_RAY_BACKEND = "PARALLEL_ORTHOGRAPHIC_PROJECTED_GRID_EXACT_EQUIVALENT_V1"


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--legacy-shard-dir", type=Path, required=True)
    p.add_argument("--teacher-dir", type=Path, required=True)
    p.add_argument("--stage01-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--surface-points", type=int, default=4096)
    p.add_argument("--mechanical-points", type=int, default=8192)
    p.add_argument("--appearance-points-per-view", type=int, default=4096)
    p.add_argument("--relation-pairs-per-view", type=int, default=2048)
    p.add_argument("--topology-target-tokens", type=int, default=512)
    p.add_argument("--topology-trajectory-tolerance", type=float, default=1.0e-7)
    p.add_argument("--ray-chunk", type=int, default=64)
    p.add_argument("--view-workers", type=int, default=8)
    return p.parse_args()


def _topology_windows_v2(rest_iris, posed_iris, faces, target_tokens: int, trajectory_tolerance: float):
    cfg = IRISVNextConfigV1().tessa_decoder_config()
    teacher = canonicalize_vnext_topology_teacher_v1(
        rest_iris,
        posed_iris,
        faces,
        coordinate_bins=cfg.coordinate_bins,
        trajectory_tolerance=trajectory_tolerance,
    )
    collision = teacher.metadata["quantized_collision_census"]
    if int(collision["collision_group_count"]) != 0:
        raise ValueError(
            "VNEXT_TESSA_UNRESOLVED_POSITION_COLLISION:"
            + json.dumps(collision, sort_keys=True, separators=(",", ":"))
        )
    sequence = build_teacher_asset_sequence_v1(
        teacher.vertices_iris * 0.5, teacher.faces, cfg=cfg
    )
    windows = build_truncated_teacher_windows_v1(
        sequence, cfg=cfg, target_tokens=target_tokens
    )
    return sequence, windows, cfg, teacher.metadata


def _source_provenance_v2() -> dict:
    repo = Path(__file__).resolve().parents[2]
    paths = (
        Path(__file__).resolve(),
        repo / "models/iris/vnext/contracts_v1.py",
        repo / "models/iris/vnext/topology_teacher_v1.py",
        repo / "models/iris/vnext/teacher_truth_v1.py",
        repo / "models/tessa/v1/training_data_v1.py",
        repo / "tools/iris_vnext/build_fit8_vnext_shards_v1.py",
    )
    return {str(path.relative_to(repo)): _sha(path) for path in paths}


def _parallel_projection_basis(direction: np.ndarray):
    d = np.asarray(direction, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(d))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("VNEXT_RAY_ACCEL_DIRECTION_INVALID")
    unit = d / norm
    helper = np.zeros(3, dtype=np.float64)
    helper[int(np.argmin(np.abs(unit)))] = 1.0
    axis_u = np.cross(unit, helper)
    axis_u_norm = float(np.linalg.norm(axis_u))
    if axis_u_norm <= 1.0e-15:
        raise ValueError("VNEXT_RAY_ACCEL_SCREEN_BASIS_INVALID")
    axis_u /= axis_u_norm
    axis_v = np.cross(unit, axis_u)
    return unit, norm, axis_u, axis_v


def _ray_triangle_first_two_parallel_grid(
    origins,
    directions,
    vertices,
    faces,
    *,
    chunk=64,
):
    """Exact-equivalent first-two intersections for the parallel prep ray court.

    Orthographic rays share one direction. Quotienting 3-D space along that
    direction gives an exact 2-D triangle containment problem. A uniform grid
    rejects triangles whose projected AABBs cannot contain a ray before applying
    the same barycentric and positive-depth tolerances as the legacy
    Moller-Trumbore implementation. ``chunk`` is retained only for CLI/source
    compatibility with the V1 preparation contract.
    """

    del chunk
    o = np.asarray(origins, dtype=np.float64)
    d = np.asarray(directions, dtype=np.float64)
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if o.ndim != 2 or o.shape[1] != 3 or d.shape != o.shape or len(o) == 0:
        raise ValueError("VNEXT_RAY_ACCEL_RAY_SHAPE_INVALID")
    if v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("VNEXT_RAY_ACCEL_MESH_SHAPE_INVALID")
    if not np.isfinite(o).all() or not np.isfinite(d).all() or not np.isfinite(v).all():
        raise ValueError("VNEXT_RAY_ACCEL_NONFINITE")
    if not np.allclose(d, d[0][None, :], rtol=1.0e-12, atol=1.0e-12):
        raise ValueError("VNEXT_RAY_ACCEL_REQUIRES_PARALLEL_RAYS")

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
    denominator = (v1 - v2) * (u0 - u2) + (u2 - u1) * (v0 - v2)
    face_valid = np.abs(denominator) * direction_norm > 1.0e-10

    min_u = np.min(tri_u, axis=1)
    max_u = np.max(tri_u, axis=1)
    min_v = np.min(tri_v, axis=1)
    max_v = np.max(tri_v, axis=1)
    projected_scale = max(
        1.0,
        float(np.max(np.abs(tri_u))),
        float(np.max(np.abs(tri_v))),
        float(np.max(np.abs(ray_u))),
        float(np.max(np.abs(ray_v))),
    )
    bbox_eps = 1.0e-8 * projected_scale

    grid_size = int(_RAY_GRID_SIZE)
    lo_u = float(np.min(ray_u) - bbox_eps)
    hi_u = float(np.max(ray_u) + bbox_eps)
    lo_v = float(np.min(ray_v) - bbox_eps)
    hi_v = float(np.max(ray_v) + bbox_eps)
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
        gx0 = int(np.floor((min_u[face_index] - bbox_eps - lo_u) * scale_u))
        gx1 = int(np.floor((max_u[face_index] + bbox_eps - lo_u) * scale_u))
        gy0 = int(np.floor((min_v[face_index] - bbox_eps - lo_v) * scale_v))
        gy1 = int(np.floor((max_v[face_index] + bbox_eps - lo_v) * scale_v))
        gx0 = max(0, min(grid_size - 1, gx0))
        gx1 = max(0, min(grid_size - 1, gx1))
        gy0 = max(0, min(grid_size - 1, gy0))
        gy1 = max(0, min(grid_size - 1, gy1))
        for gy in range(gy0, gy1 + 1):
            base = gy * grid_size
            for gx in range(gx0, gx1 + 1):
                cells[base + gx].append(int(face_index))
    cell_arrays = tuple(np.asarray(cell, dtype=np.int64) for cell in cells)

    ray_gx = np.floor((ray_u - lo_u) * scale_u).astype(np.int64)
    ray_gy = np.floor((ray_v - lo_v) * scale_v).astype(np.int64)
    ray_gx = np.clip(ray_gx, 0, grid_size - 1)
    ray_gy = np.clip(ray_gy, 0, grid_size - 1)

    ray_count = len(o)
    first_face = np.full(ray_count, -1, dtype=np.int64)
    second_face = np.full(ray_count, -1, dtype=np.int64)
    first_point = np.zeros((ray_count, 3), dtype=np.float64)
    second_point = np.zeros((ray_count, 3), dtype=np.float64)
    first_depth = np.full(ray_count, np.inf, dtype=np.float64)
    second_depth = np.full(ray_count, np.inf, dtype=np.float64)

    for ray_index in range(ray_count):
        candidates = cell_arrays[int(ray_gy[ray_index]) * grid_size + int(ray_gx[ray_index])]
        if not len(candidates):
            continue
        pu = float(ray_u[ray_index])
        pv = float(ray_v[ray_index])
        candidates = candidates[
            (pu >= min_u[candidates] - bbox_eps)
            & (pu <= max_u[candidates] + bbox_eps)
            & (pv >= min_v[candidates] - bbox_eps)
            & (pv <= max_v[candidates] + bbox_eps)
        ]
        if not len(candidates):
            continue

        den = denominator[candidates]
        weight0 = (
            (v1[candidates] - v2[candidates]) * (pu - u2[candidates])
            + (u2[candidates] - u1[candidates]) * (pv - v2[candidates])
        ) / den
        weight1 = (
            (v2[candidates] - v0[candidates]) * (pu - u2[candidates])
            + (u0[candidates] - u2[candidates]) * (pv - v2[candidates])
        ) / den
        weight2 = 1.0 - weight0 - weight1
        inside = (
            (weight0 >= -1.0e-9)
            & (weight1 >= -1.0e-9)
            & (weight2 >= -1.0e-9)
        )
        candidates = candidates[inside]
        weight0 = weight0[inside]
        weight1 = weight1[inside]
        weight2 = weight2[inside]
        if not len(candidates):
            continue

        distance_along_unit = (
            weight0 * tri_w[candidates, 0]
            + weight1 * tri_w[candidates, 1]
            + weight2 * tri_w[candidates, 2]
            - ray_w[ray_index]
        )
        depth = distance_along_unit / direction_norm
        positive = depth > 1.0e-8
        candidates = candidates[positive]
        depth = depth[positive]
        if not len(candidates):
            continue

        # Legacy finite-face enumeration is ascending face index and uses stable
        # depth sorting. Lexsort reproduces that exact tie rule.
        order = np.lexsort((candidates, depth))
        candidates = candidates[order]
        depth = depth[order]

        face0 = int(candidates[0])
        first_face[ray_index] = face0
        first_depth[ray_index] = float(depth[0])
        first_point[ray_index] = o[ray_index] + first_depth[ray_index] * d[ray_index]
        if len(candidates) > 1:
            face1 = int(candidates[1])
            second_face[ray_index] = face1
            second_depth[ray_index] = float(depth[1])
            second_point[ray_index] = o[ray_index] + second_depth[ray_index] * d[ray_index]

    return {
        "first_face": first_face,
        "second_face": second_face,
        "first_point": first_point,
        "second_point": second_point,
        "first_depth": first_depth,
        "second_depth": second_depth,
    }


def _build_subject_view_banks(
    *,
    source_id: str,
    cameras: list[dict],
    stage01: Path,
    rest_source: np.ndarray,
    faces: np.ndarray,
    face_owner: np.ndarray,
    face_material: np.ndarray,
    owner_rows,
    center: np.ndarray,
    half: float,
    appearance_count: int,
    relation_count: int,
    ray_chunk: int,
    view_workers: int,
):
    """Build V0..V7 independently while preserving deterministic output order."""
    base = stage01 / "observations" / source_id

    def one(view_index: int):
        camera = cameras[view_index]
        return _build_view_banks(
            subject_id=source_id,
            view_index=view_index,
            camera=camera,
            png_path=base / f"V{view_index}.png",
            mask_path=base / f"V{view_index}.mask.bin",
            vertices_source=rest_source,
            faces=faces,
            face_owner=face_owner,
            face_material=face_material,
            owner_rows=owner_rows,
            center=center,
            half=half,
            appearance_count=appearance_count,
            relation_count=relation_count,
            ray_chunk=ray_chunk,
        )

    legacy_raycast = shard_v1._ray_triangle_first_two
    if getattr(legacy_raycast, "__name__", "") != "_ray_triangle_first_two":
        raise RuntimeError("VNEXT_RAY_ACCEL_LEGACY_BACKEND_DRIFT")
    shard_v1._ray_triangle_first_two = _ray_triangle_first_two_parallel_grid
    try:
        workers = min(int(view_workers), len(cameras))
        if workers == 1:
            return [one(view_index) for view_index in range(len(cameras))]
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="vnext-view") as pool:
            # executor.map preserves input order, so the serialized bank remains V0..V7.
            return list(pool.map(one, range(len(cameras))))
    finally:
        shard_v1._ray_triangle_first_two = legacy_raycast


def main():
    args = _args()
    for count in (
        args.surface_points,
        args.mechanical_points,
        args.appearance_points_per_view,
        args.relation_pairs_per_view,
        args.topology_target_tokens,
        args.ray_chunk,
        args.view_workers,
    ):
        if int(count) <= 0:
            raise ValueError("VNEXT_SHARD_POLICY_NONPOSITIVE")
    if int(args.view_workers) > 8:
        raise ValueError("VNEXT_SHARD_VIEW_WORKERS_EXCEED_VIEW_COUNT")
    if not (0.0 <= args.topology_trajectory_tolerance <= 1.0e-3):
        raise ValueError("VNEXT_SHARD_TOPOLOGY_TRAJECTORY_TOLERANCE_INVALID")

    legacy_dir = args.legacy_shard_dir.resolve()
    teacher_dir = args.teacher_dir.resolve()
    stage01 = args.stage01_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    bootstrap_path = stage01 / "BOOTSTRAP_MANIFEST.json"
    bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    bootstrap_rows = {row["subject_id"]: row for row in bootstrap["subjects"]}
    subject_rows = []
    topology_cfg = None

    for display in SUBJECT_IDS:
        subject_started = time.perf_counter()
        source_id = SUBJECT_SOURCE_IDS[display]
        legacy_path = legacy_dir / f"{source_id}.pt"
        teacher_path = teacher_dir / f"{source_id}.teacher.npz"
        teacher_receipt_path = teacher_path.with_suffix(teacher_path.suffix + ".json")
        if not legacy_path.is_file() or not teacher_path.is_file() or not teacher_receipt_path.is_file():
            raise FileNotFoundError(f"VNEXT_SHARD_INPUT_MISSING:{display}")

        legacy = torch.load(legacy_path, map_location="cpu", weights_only=True)
        if str(legacy.get("subject_id")) != source_id:
            raise ValueError("VNEXT_SHARD_LEGACY_SUBJECT_DRIFT:" + display)
        teacher_receipt = json.loads(teacher_receipt_path.read_text(encoding="utf-8"))
        boot = bootstrap_rows[source_id]
        if teacher_receipt["source"]["sha256"] != boot["raw_source"]["sha256"]:
            raise ValueError("VNEXT_SHARD_TEACHER_SOURCE_HASH_DRIFT:" + display)

        with np.load(teacher_path, allow_pickle=False) as data:
            rest_source = np.asarray(data["rest_vertices_source_world"], dtype=np.float64)
            faces = np.asarray(data["faces"], dtype=np.int64)
            face_owner = np.asarray(data["face_owner"], dtype=np.int32)
            face_material = np.asarray(data["face_material"], dtype=np.int32)
            posed_source = np.asarray(data["posed_vertices_source_world"], dtype=np.float64)

        center = np.asarray(legacy["center_xyz"], dtype=np.float64).reshape(3)
        half = float(torch.as_tensor(legacy["normalization_half_extent"]).item())
        if not np.isfinite(half) or half <= 0:
            raise ValueError("VNEXT_SHARD_LEGACY_NORMALIZATION_INVALID")
        rest_iris = _normalize_source_world(rest_source, center, half)
        posed_iris = _normalize_source_world(posed_source, center, half)
        if float(np.abs(rest_iris).max()) > 1.00001:
            raise ValueError("VNEXT_SHARD_REST_OUTSIDE_IRIS_DOMAIN:" + display)

        rng = np.random.default_rng(_seed(display))
        surface_points, _, _ = _sample_surface(rest_iris, faces, args.surface_points, rng)

        mechanical_started = time.perf_counter()
        mechanical = _mechanical_bank(
            rest_iris, posed_iris, faces, args.mechanical_points, rng
        )
        mechanical_seconds = time.perf_counter() - mechanical_started

        topology_started = time.perf_counter()
        sequence, windows, cfg, topology_canonicalization = _topology_windows_v2(
            rest_iris,
            posed_iris,
            faces,
            args.topology_target_tokens,
            args.topology_trajectory_tolerance,
        )
        topology_seconds = time.perf_counter() - topology_started
        if topology_cfg is None:
            topology_cfg = cfg
        elif topology_cfg != cfg:
            raise AssertionError("VNEXT_SHARD_TOPOLOGY_CONFIG_DRIFT")

        camera_bundle_path = stage01 / boot["source_scene_evidence"]["camera_bundle"]["path"]
        camera_bundle = json.loads(camera_bundle_path.read_text(encoding="utf-8"))
        cameras = list(camera_bundle["cameras"])
        if len(cameras) != 8:
            raise ValueError("VNEXT_SHARD_CAMERA_COUNT_DRIFT")

        views_started = time.perf_counter()
        view_banks = _build_subject_view_banks(
            source_id=source_id,
            cameras=cameras,
            stage01=stage01,
            rest_source=rest_source,
            faces=faces,
            face_owner=face_owner,
            face_material=face_material,
            owner_rows=teacher_receipt["owners"],
            center=center,
            half=half,
            appearance_count=args.appearance_points_per_view,
            relation_count=args.relation_pairs_per_view,
            ray_chunk=args.ray_chunk,
            view_workers=args.view_workers,
        )
        views_seconds = time.perf_counter() - views_started
        subject_seconds = time.perf_counter() - subject_started
        print(
            "VNEXT_FIT8_SUBJECT_PREP_TIMING",
            f"subject={display}",
            f"faces={len(faces)}",
            f"frames={len(posed_iris)}",
            f"mechanical_s={mechanical_seconds:.3f}",
            f"topology_s={topology_seconds:.3f}",
            f"views_s={views_seconds:.3f}",
            f"total_s={subject_seconds:.3f}",
            f"ray_backend={_RAY_BACKEND}",
            flush=True,
        )

        row = {
            "subject_id": display,
            "source_subject_id": source_id,
            "teacher_inputs_in_inference_forward": False,
            "image_tokens": legacy["image_tokens"].squeeze(0).contiguous(),
            "token_xy": legacy["token_xy"].squeeze(0).contiguous(),
            "camera_features": legacy["camera_features"].squeeze(0).contiguous(),
            "geometry_points": legacy["replay_points"].contiguous(),
            "geometry_targets": legacy["replay_targets"].contiguous(),
            "teacher_surface_points_iris": _tensor(surface_points, torch.float32),
            **{
                key: _tensor(value, torch.float32)
                for key, value in mechanical.items()
                if key != "mechanical_truth_metadata"
            },
            "appearance_points": _tensor(
                np.stack([view["appearance_points"] for view in view_banks]), torch.float32
            ),
            "appearance_rgba": _tensor(
                np.stack([view["appearance_rgba"] for view in view_banks]), torch.float32
            ),
            "appearance_alpha_boundary": _tensor(
                np.stack([view["appearance_alpha_boundary"] for view in view_banks]), torch.float32
            ),
            "appearance_boundary": _tensor(
                np.stack([view["appearance_boundary"] for view in view_banks]), torch.float32
            ),
            "_windows": windows,
            "_window_count": len(windows),
            "metadata": {
                "schema": "RealSaS.IRISVNextFit8ShardMetadata.v2",
                "legacy_shard_sha256": _sha(legacy_path),
                "teacher_export_sha256": _sha(teacher_path),
                "teacher_export_receipt_sha256": _sha(teacher_receipt_path),
                "stage01_bootstrap_sha256": _sha(bootstrap_path),
                "source_asset_sha256": boot["raw_source"]["sha256"],
                "mechanical_truth": mechanical["mechanical_truth_metadata"],
                "topology_teacher": {
                    "face_count": sequence.face_count,
                    "vertex_count": sequence.vertex_count,
                    "component_count": sequence.component_count,
                    "restart_count": sequence.restart_count,
                    "token_count": len(sequence.token_ids),
                    "window_count": len(windows),
                    "coordinate_bins": int(cfg.coordinate_bins),
                    "canonicalization": topology_canonicalization,
                },
                "anonymous_drawable_ids_exposed_to_inference": False,
                "teacher_rig_skin_motion_exposed_to_inference": False,
            },
        }
        relation_keys = [key for key in view_banks[0] if key.startswith("relation_")]
        for key in relation_keys:
            row[key] = _tensor(
                np.stack([view[key] for view in view_banks]), torch.float32
            )
        subject_rows.append(row)

    if topology_cfg is None:
        raise RuntimeError("VNEXT_SHARD_TOPOLOGY_CONFIG_MISSING")
    max_windows, max_length = _pad_topology(subject_rows, topology_cfg)

    manifest_rows = []
    for row in subject_rows:
        display = row["subject_id"]
        metadata = row.pop("metadata")
        source_subject_id = row.pop("source_subject_id")
        path = output / f"{display.lower().replace('-', '').replace(' ', '')}.vnext.pt"
        torch.save(row, path)
        manifest_rows.append(
            {
                "subject_id": display,
                "source_subject_id": source_subject_id,
                "shard_file": path.name,
                "shard_sha256": _sha(path),
                "shard_bytes": path.stat().st_size,
                "metadata": metadata,
            }
        )

    manifest = {
        "schema": "RealSaS.IRISVNextFit8Manifest.v4",
        "builder": "tools/iris_vnext/build_fit8_vnext_shards_v2.py",
        "source_provenance": _source_provenance_v2(),
        "subjects": manifest_rows,
        "subject_order": list(SUBJECT_IDS),
        "teacher_inputs_in_inference_forward": False,
        "legacy_optimizer_state_reused": False,
        "legacy_5400_checkpoint_reused": False,
        "frozen_legacy_source_evidence_reused": True,
        "topology_window_shape": [max_windows, max_length],
        "topology_pad_token": int(topology_cfg.PAD),
        "topology_coordinate_bins": int(topology_cfg.coordinate_bins),
        "topology_teacher_collision_policy": "MOTION_EQUIVALENT_EXACT_SEAM_WELD_THEN_BOUNDED_TRAINING_ONLY_QUANTIZED_IDENTITY_LIFT_THEN_ZERO_RESIDUAL_COLLISIONS_REQUIRED",
        "policy": {
            "surface_points": int(args.surface_points),
            "mechanical_points": int(args.mechanical_points),
            "appearance_points_per_view": int(args.appearance_points_per_view),
            "relation_pairs_per_view": int(args.relation_pairs_per_view),
            "topology_target_tokens": int(args.topology_target_tokens),
            "topology_trajectory_tolerance": float(args.topology_trajectory_tolerance),
            "ray_chunk": int(args.ray_chunk),
            "ray_chunk_legacy_compatibility_only": True,
            "ray_intersection_backend": _RAY_BACKEND,
            "ray_grid_size": int(_RAY_GRID_SIZE),
            "view_workers": int(args.view_workers),
        },
    }
    manifest["manifest_value_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (output / "COHORT_MANIFEST.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8"
    )
    print(
        "VNEXT_FIT8_SHARDS_V2_PASS",
        "subjects=8",
        f"topology_windows={max_windows}",
        f"topology_length={max_length}",
        f"coordinate_bins={topology_cfg.coordinate_bins}",
        f"view_workers={args.view_workers}",
        f"ray_backend={_RAY_BACKEND}",
        f"manifest={manifest['manifest_value_sha256']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
