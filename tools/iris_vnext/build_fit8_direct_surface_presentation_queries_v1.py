#!/usr/bin/env python3
from __future__ import annotations

"""Materialize inference-admissible presentation queries from Direct IRIS surface.

Inputs are deliberately limited to:
- a cryptographically bound Stage01 observation bundle;
- model-produced Direct IRIS T512->sparse-T1024 surfaces;
- the frozen structural-evidence input binding for subject/camera authority.

There is no teacher-mesh, teacher-topology, rig/skin, material-owner, or V2
all-hit truth input. The output contains only model-surface ray geometry and
source-observed RGBA/boundary/support. Pair front and global visibility are
computed later deterministically from these complete model-surface ray stacks.

This tool performs zero optimizer/backward steps and cannot claim scientific PASS.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
from typing import Any
import zipfile

import numpy as np
from PIL import Image

from models.iris.vnext.cohort_v1 import SUBJECT_IDS
from tools.iris_vnext.build_fit8_relation_all_hit_evidence_v1 import (
    _face_edge_components,
    _ray_triangle_all_parallel_grid,
)
from tools.iris_vnext.build_fit8_vnext_shards_v1 import _boundary_flags, _seed


SCHEMA = "RealSaS.DirectIRISPresentationQueries.v1"
MANIFEST_SCHEMA = "RealSaS.DirectIRISPresentationQueriesManifest.v1"
STATUS = "MATERIALIZED_MODEL_SURFACE_PRESENTATION_QUERIES__NO_TRAINING"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _value_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("DIRECT_PRESENTATION_JSON_OBJECT_REQUIRED:" + str(path))
    return value


def _git_head(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip().lower()


def _depth_tie_epsilon(vertices: np.ndarray) -> float:
    value = np.asarray(vertices, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 3 or not len(value) or not np.isfinite(value).all():
        raise ValueError("DIRECT_PRESENTATION_SURFACE_VERTICES_INVALID")
    scale = max(1.0, float(np.max(value) - np.min(value)), float(np.max(np.abs(value))))
    return float(scale * 1.0e-10)


def _sample_pixels(mask_bytes: bytes, resolution: int, *, source_id: str, view_index: int, budget: int) -> np.ndarray:
    if int(budget) <= 0:
        raise ValueError("DIRECT_PRESENTATION_RAY_BUDGET_NONPOSITIVE")
    mask = np.frombuffer(mask_bytes, dtype=np.uint8)
    if mask.size != resolution * resolution:
        raise ValueError("DIRECT_PRESENTATION_MASK_SIZE_DRIFT")
    mask = mask.reshape(resolution, resolution)
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("DIRECT_PRESENTATION_SOURCE_MASK_EMPTY")
    # Intentionally share the V2 all-hit deterministic source-ray sampling seed.
    rng = np.random.default_rng(_seed(f"relation-all-hit-v1:{source_id}:view:{view_index}"))
    choose = rng.permutation(len(xs))[: min(len(xs), int(budget))]
    return np.stack((xs[choose], ys[choose]), axis=1).astype(np.int64)


def _canonical_camera(
    camera: dict[str, Any], *, center: np.ndarray, normalization_half: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    resolution = int(camera["resolution"])
    half_world = float(camera["half_extent"])
    if resolution <= 0 or not np.isfinite(half_world) or half_world <= 0.0:
        raise ValueError("DIRECT_PRESENTATION_CAMERA_SCALE_INVALID")
    origin_world = np.asarray(camera["origin"], dtype=np.float64).reshape(3)
    right = np.asarray(camera["right"], dtype=np.float64).reshape(3)
    up = np.asarray(camera["screen_up"], dtype=np.float64).reshape(3)
    forward = np.asarray(camera["forward"], dtype=np.float64).reshape(3)
    right /= np.linalg.norm(right)
    up /= np.linalg.norm(up)
    forward /= np.linalg.norm(forward)
    origin = (origin_world - center) / float(normalization_half)
    half_ratio = half_world / float(normalization_half)
    feature = np.concatenate((origin, right, up, forward, np.asarray([half_ratio]))).astype(np.float32)
    return feature, right, up


def _canonical_rays(
    camera: dict[str, Any],
    pixels_xy: np.ndarray,
    *,
    center: np.ndarray,
    normalization_half: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature, right, up = _canonical_camera(
        camera, center=center, normalization_half=normalization_half
    )
    resolution = int(camera["resolution"])
    half_ratio = float(feature[12])
    x = pixels_xy[:, 0].astype(np.float64)
    y = pixels_xy[:, 1].astype(np.float64)
    sx = ((x + 0.5) / resolution * 2.0 - 1.0) * half_ratio
    sy = (1.0 - (y + 0.5) / resolution * 2.0) * half_ratio
    origin = feature[0:3].astype(np.float64)
    forward = feature[9:12].astype(np.float64)
    origins = origin[None, :] + sx[:, None] * right[None, :] + sy[:, None] * up[None, :]
    directions = np.broadcast_to(forward, origins.shape).copy()
    return origins, directions, feature


def _oriented_pairs(all_hits: dict[str, np.ndarray], pixels: np.ndarray) -> dict[str, np.ndarray]:
    offsets = np.asarray(all_hits["offsets"], dtype=np.int64)
    points = np.asarray(all_hits["point"], dtype=np.float32)
    faces = np.asarray(all_hits["face"], dtype=np.int64)
    if offsets.shape != (len(pixels) + 1,):
        raise ValueError("DIRECT_PRESENTATION_STACK_OFFSETS_SHAPE_DRIFT")
    point_a: list[np.ndarray] = []
    point_b: list[np.ndarray] = []
    face_a: list[int] = []
    face_b: list[int] = []
    rank_a: list[int] = []
    rank_b: list[int] = []
    ray_index: list[int] = []
    base_pair_id: list[int] = []
    pair_id = 0
    for ray in range(len(pixels)):
        start, end = int(offsets[ray]), int(offsets[ray + 1])
        count = end - start
        for a in range(count - 1):
            for b in range(a + 1, count):
                for reverse in (False, True):
                    ra, rb = (b, a) if reverse else (a, b)
                    point_a.append(points[start + ra])
                    point_b.append(points[start + rb])
                    face_a.append(int(faces[start + ra]))
                    face_b.append(int(faces[start + rb]))
                    rank_a.append(int(ra))
                    rank_b.append(int(rb))
                    ray_index.append(int(ray))
                    base_pair_id.append(int(pair_id))
                pair_id += 1
    n = len(point_a)
    return {
        "pair_point_a_iris": np.asarray(point_a, dtype=np.float32).reshape(n, 3),
        "pair_point_b_iris": np.asarray(point_b, dtype=np.float32).reshape(n, 3),
        "pair_face_a": np.asarray(face_a, dtype=np.int64),
        "pair_face_b": np.asarray(face_b, dtype=np.int64),
        "pair_rank_a": np.asarray(rank_a, dtype=np.int16),
        "pair_rank_b": np.asarray(rank_b, dtype=np.int16),
        "pair_ray_index": np.asarray(ray_index, dtype=np.int32),
        "pair_base_pair_id": np.asarray(base_pair_id, dtype=np.int64),
    }


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    expected_source = str(args.expected_source_sha).strip().lower()
    if _git_head(repo_root) != expected_source:
        raise RuntimeError("DIRECT_PRESENTATION_CHECKOUT_SHA_DRIFT")

    binding_path = Path(args.input_binding).resolve()
    binding = _json(binding_path)
    if binding.get("schema") != "RealSaS.StructuralEvidenceInputBinding.v2":
        raise ValueError("DIRECT_PRESENTATION_BINDING_SCHEMA_INVALID")
    source_inputs = binding.get("source_inputs") or {}
    stage_spec = source_inputs.get("stage01") or {}
    stage_path = Path(args.stage01_bundle).resolve()
    if _sha(stage_path) != str(stage_spec.get("sha256") or "").lower():
        raise ValueError("DIRECT_PRESENTATION_STAGE01_SHA_DRIFT")
    legacy_rows = list((source_inputs.get("legacy_fit8") or {}).get("subjects") or ())
    if tuple(row.get("subject_id") for row in legacy_rows) != SUBJECT_IDS:
        raise ValueError("DIRECT_PRESENTATION_BINDING_SUBJECT_ORDER_DRIFT")
    legacy_by_subject = {str(row["subject_id"]): str(row["legacy_subject_id"]) for row in legacy_rows}

    direct_root = Path(args.direct_surface_root).resolve()
    direct_batch_path = direct_root / "FIT8_DIRECT_IRIS_SURFACE_BATCH_RECEIPT.json"
    direct_batch = _json(direct_batch_path)
    if direct_batch.get("schema") != "RealSaS.Fit8DirectIRISSurfaceMaterialization.v1":
        raise ValueError("DIRECT_PRESENTATION_SURFACE_BATCH_SCHEMA_INVALID")
    if tuple(direct_batch.get("subject_order") or ()) != SUBJECT_IDS:
        raise ValueError("DIRECT_PRESENTATION_SURFACE_SUBJECT_ORDER_DRIFT")
    if direct_batch.get("teacher_truth_used") is not False or direct_batch.get("tessa_invoked") is not False:
        raise ValueError("DIRECT_PRESENTATION_SURFACE_AUTHORITY_CONTAMINATED")
    direct_rows = {str(row["subject_id"]): row for row in direct_batch.get("subjects") or ()}

    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("DIRECT_PRESENTATION_OUTPUT_MUST_BE_EMPTY")
    output.mkdir(parents=True, exist_ok=True)
    subject_results: list[dict[str, Any]] = []

    with zipfile.ZipFile(stage_path, "r") as archive:
        for subject_id in SUBJECT_IDS:
            legacy_id = legacy_by_subject[subject_id]
            prefix = f"observations/{legacy_id}/"
            camera_bytes = archive.read(prefix + "camera_bundle.json")
            cameras = json.loads(camera_bytes)
            if cameras.get("per_view_pose_change") or cameras.get("per_view_recenter"):
                raise ValueError("DIRECT_PRESENTATION_CAMERA_MUTATION_DRIFT:" + subject_id)
            center = np.asarray(cameras["common_center_xyz"], dtype=np.float64).reshape(3)
            normalization_half = float(cameras["common_half_extent"])
            if not np.isfinite(normalization_half) or normalization_half <= 0.0:
                raise ValueError("DIRECT_PRESENTATION_COMMON_HALF_INVALID:" + subject_id)

            subject_root = direct_root / "subjects" / subject_id
            receipt_path = subject_root / "DIRECT_IRIS_SURFACE_RECEIPT.json"
            surface_path = subject_root / "DIRECT_IRIS_SURFACE.stage12.npz"
            receipt = _json(receipt_path)
            row = direct_rows.get(subject_id) or {}
            if _sha(surface_path) != str(receipt.get("stage12_npz_sha256") or "").lower():
                raise ValueError("DIRECT_PRESENTATION_SURFACE_SHA_DRIFT:" + subject_id)
            if row.get("stage12_npz_sha256") != receipt.get("stage12_npz_sha256"):
                raise ValueError("DIRECT_PRESENTATION_SURFACE_BATCH_DRIFT:" + subject_id)
            with np.load(surface_path, allow_pickle=False) as data:
                vertices = np.asarray(data["vertices_normalized"], dtype=np.float64)
                faces = np.asarray(data["faces"], dtype=np.int64)
            if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
                raise ValueError("DIRECT_PRESENTATION_SURFACE_SHAPE_INVALID:" + subject_id)
            components = _face_edge_components(faces)
            depth_tie_epsilon = _depth_tie_epsilon(vertices)

            view_files: list[dict[str, Any]] = []
            for camera in cameras.get("cameras") or ():
                view_index = int(camera["view_index"])
                resolution = int(camera["resolution"])
                mask_bytes = archive.read(prefix + f"V{view_index}.mask.bin")
                png_bytes = archive.read(prefix + f"V{view_index}.png")
                rgba = np.asarray(Image.open(io.BytesIO(png_bytes)).convert("RGBA"), dtype=np.uint8)
                if rgba.shape != (resolution, resolution, 4):
                    raise ValueError("DIRECT_PRESENTATION_RGBA_SHAPE_DRIFT")
                pixels = _sample_pixels(
                    mask_bytes,
                    resolution,
                    source_id=legacy_id,
                    view_index=view_index,
                    budget=int(args.candidate_rays_per_view),
                )
                origins, directions, camera_feature = _canonical_rays(
                    camera,
                    pixels,
                    center=center,
                    normalization_half=normalization_half,
                )
                all_hits = _ray_triangle_all_parallel_grid(
                    origins, directions, vertices, faces, components
                )
                pairs = _oriented_pairs(all_hits, pixels)
                alpha_boundary, appearance_boundary = _boundary_flags(rgba, pixels)
                source_rgba = rgba[pixels[:, 1], pixels[:, 0]].astype(np.float32) / 255.0
                stack_count = np.diff(all_hits["offsets"]).astype(np.int32)
                file_name = f"{subject_id.lower().replace('-', '').replace(' ', '')}.v{view_index}.direct_presentation_queries_v1.npz"
                path = output / file_name
                np.savez_compressed(
                    path,
                    schema=np.asarray(SCHEMA),
                    subject_id=np.asarray(subject_id),
                    legacy_subject_id=np.asarray(legacy_id),
                    view_index=np.asarray(view_index, dtype=np.int16),
                    direct_surface_depth_tie_epsilon=np.asarray(depth_tie_epsilon, dtype=np.float64),
                    camera_features=np.asarray(camera_feature, dtype=np.float32),
                    pixels_xy=pixels.astype(np.int32),
                    source_rgba=source_rgba.astype(np.float32),
                    source_alpha_boundary=alpha_boundary.astype(np.float32),
                    source_appearance_boundary=appearance_boundary.astype(np.float32),
                    source_support=np.ones(len(pixels), dtype=np.float32),
                    stack_offsets=np.asarray(all_hits["offsets"], dtype=np.int64),
                    stack_face=np.asarray(all_hits["face"], dtype=np.int64),
                    stack_depth=np.asarray(all_hits["depth"], dtype=np.float64),
                    stack_point_iris=np.asarray(all_hits["point"], dtype=np.float32),
                    stack_count=stack_count,
                    **pairs,
                )
                view_files.append(
                    {
                        "view_index": view_index,
                        "file": file_name,
                        "sha256": _sha(path),
                        "bytes": path.stat().st_size,
                        "sampled_source_foreground_rays": int(len(pixels)),
                        "direct_surface_hit_rays": int(np.count_nonzero(stack_count >= 1)),
                        "multi_hit_rays": int(np.count_nonzero(stack_count >= 2)),
                        "total_direct_surface_hits": int(all_hits["offsets"][-1]),
                        "oriented_pair_rows": int(len(pairs["pair_base_pair_id"])),
                    }
                )
            if len(view_files) != 8:
                raise ValueError("DIRECT_PRESENTATION_VIEW_COUNT_DRIFT:" + subject_id)
            subject_results.append(
                {
                    "subject_id": subject_id,
                    "legacy_subject_id": legacy_id,
                    "direct_surface_sha256": receipt["stage12_npz_sha256"],
                    "direct_surface_candidate_binding_hash": receipt["candidate_binding_hash"],
                    "direct_surface_depth_tie_epsilon": depth_tie_epsilon,
                    "camera_bundle_sha256": _sha_bytes(camera_bytes),
                    "views": view_files,
                }
            )

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": STATUS,
        "execution_source_sha": expected_source,
        "input_binding_sha256": _sha(binding_path),
        "stage01_bundle_sha256": _sha(stage_path),
        "direct_surface_batch_receipt_sha256": _sha(direct_batch_path),
        "subject_order": list(SUBJECT_IDS),
        "subjects": subject_results,
        "policy": {
            "candidate_rays_per_view": int(args.candidate_rays_per_view),
            "source_ray_sampling_seed_contract": "relation-all-hit-v1:{legacy_subject_id}:view:{view_index}",
            "surface_anchor": "DIRECT_IRIS_MODEL_SURFACE_ONLY",
            "source_observations": "ADMITTED_STAGE01_RGBA_MASK_CAMERA",
            "pair_geometry": "ALL_UNORDERED_DIRECT_SURFACE_HIT_PAIRS_BOTH_ORIENTATIONS",
            "complete_stack_geometry": True,
            "depth_tie_epsilon": "MAX_1_OR_SURFACE_SCALE_TIMES_1E_MINUS_10",
            "tessa_coordinate_conversion": "MULTIPLY_IRIS_XYZ_BY_0.5_AT_MEMORY_BUILD",
        },
        "teacher_mesh_input": False,
        "teacher_topology_input": False,
        "teacher_rig_skin_input": False,
        "materialized_v2_truth_input": False,
        "optimizer_steps_executed": 0,
        "backward_steps_executed": 0,
        "training_authorized": False,
        "scientific_pass_claimed": False,
        "next_action": "DETERMINISTICALLY_TYPE_DIRECT_SURFACE_PRESENTATION_MEMORY",
    }
    manifest["value_sha256"] = _value_sha(manifest)
    manifest_path = output / "DIRECT_IRIS_PRESENTATION_QUERIES_V1_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", required=True)
    p.add_argument("--expected-source-sha", required=True)
    p.add_argument(
        "--input-binding",
        default="canonical/STRUCTURAL_EVIDENCE_INPUT_BINDING_V2.json",
    )
    p.add_argument("--stage01-bundle", required=True)
    p.add_argument("--direct-surface-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--candidate-rays-per-view", type=int, default=8192)
    return p


def main() -> int:
    result = materialize(parser().parse_args())
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
