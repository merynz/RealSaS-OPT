#!/usr/bin/env python3
"""Build deterministic cross-view visibility/occlusion truth from dense Graphics G-buffers.

The producer samples source-visible surface rows without outcome mining, reprojects each canonical
surface point into every target camera, and classifies the target relation using exact dense
front-surface geometry.  It emits only primitive geometric evidence: no body-part, material, owner,
attachment or global layer labels.

The independent CPU ray-stack oracle remains a separate court.  This file creates teacher truth;
it does not self-certify it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.corpus.build_truth139_dense_graphics_gbuffer_v1 import PROVENANCE_CODES, sha256
from tools.corpus.build_truth139_dense_appearance_gbuffer_v1 import _load_camera_bundle

SCHEMA = "RealSaS.Truth139VisibilityOcclusionTeacher.v1"
GRAPHICS_SCHEMA = "RealSaS.Truth139DenseGraphicsGBuffer.v1"
VISIBILITY = {
    "OUTSIDE_VIEW": 0,
    "VISIBLE_SAME_SURFACE": 1,
    "OCCLUDED_BY_FRONT_SURFACE": 2,
    "PROJECTED_TO_BACKGROUND_OR_UNSUPPORTED": 3,
    "RASTER_GEOMETRY_DISAGREEMENT_OR_AMBIGUOUS": 4,
}


def digest_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def deterministic_stratified_rows(row_count: int, maximum: int, *, subject_id: str, source_view: int) -> np.ndarray:
    if row_count < 0 or maximum <= 0 or source_view < 0 or not subject_id:
        raise ValueError("TRUTH139_VISIBILITY_SAMPLE_ARGUMENT")
    if row_count <= maximum:
        return np.arange(row_count, dtype=np.int32)
    seed_bytes = hashlib.sha256(f"{subject_id}:V{source_view}".encode()).digest()[:8]
    seed = int.from_bytes(seed_bytes, "big", signed=False)
    rng = np.random.default_rng(seed)
    edges = np.linspace(0, row_count, maximum + 1, dtype=np.int64)
    picks = np.empty(maximum, dtype=np.int32)
    for i in range(maximum):
        lo, hi = int(edges[i]), int(edges[i + 1])
        if hi <= lo:
            picks[i] = min(lo, row_count - 1)
        else:
            picks[i] = int(rng.integers(lo, hi))
    if len(np.unique(picks)) != len(picks):
        raise AssertionError("TRUTH139_VISIBILITY_STRATIFIED_DUPLICATE")
    return picks


def project_source_point(point_source_world: np.ndarray, camera: dict) -> tuple[bool, float, float, float]:
    p = np.asarray(point_source_world, dtype=np.float64)
    origin = np.asarray(camera["origin"], dtype=np.float64)
    right = np.asarray(camera["right"], dtype=np.float64)
    up = np.asarray(camera["screen_up"], dtype=np.float64)
    forward = np.asarray(camera["forward"], dtype=np.float64)
    half = float(camera["half_extent"])
    resolution = int(camera["resolution"])
    if p.shape != (3,) or not np.isfinite(p).all() or half <= 0 or resolution <= 0:
        raise ValueError("TRUTH139_VISIBILITY_PROJECTION_ARGUMENT")
    rel = p - origin
    sx = float(rel @ right)
    sy = float(rel @ up)
    depth = float(rel @ forward)
    x_cont = ((sx / half + 1.0) * 0.5) * resolution - 0.5
    y_cont = ((1.0 - sy / half) * 0.5) * resolution - 0.5
    inside = bool(-0.5 <= x_cont < resolution - 0.5 and -0.5 <= y_cont < resolution - 0.5 and depth > 0.0)
    return inside, x_cont, y_cont, depth


def classify_target(
    *,
    sample_canonical: np.ndarray,
    sample_depth: float,
    projected_x: int,
    projected_y: int,
    row_lookup: np.ndarray,
    target_xyz_canonical: np.ndarray,
    target_depth: np.ndarray,
    target_triangle: np.ndarray,
    target_facing: np.ndarray,
    target_source_support: np.ndarray,
    surface_tolerance: float,
    depth_epsilon: float,
    search_radius_px: int,
) -> tuple[int, int, int, float, float, bool, bool]:
    """Return class, pixel, front-triangle, depth-gap, facing, support, same-surface."""
    h, w = row_lookup.shape
    if not (0 <= projected_x < w and 0 <= projected_y < h):
        return VISIBILITY["OUTSIDE_VIEW"], -1, -1, np.nan, 0.0, False, False
    central_row = int(row_lookup[projected_y, projected_x])
    central_linear = projected_y * w + projected_x
    if central_row < 0:
        return VISIBILITY["PROJECTED_TO_BACKGROUND_OR_UNSUPPORTED"], central_linear, -1, np.nan, 0.0, False, False

    # Visible correspondence is allowed to search a tiny declared raster neighborhood because the
    # projected continuous point need not land at the same pixel centre as the target raster row.
    best_row = -1
    best_dist = np.inf
    for yy in range(max(0, projected_y - search_radius_px), min(h, projected_y + search_radius_px + 1)):
        for xx in range(max(0, projected_x - search_radius_px), min(w, projected_x + search_radius_px + 1)):
            row = int(row_lookup[yy, xx])
            if row < 0:
                continue
            dist = float(np.linalg.norm(np.asarray(target_xyz_canonical[row], dtype=np.float64) - sample_canonical))
            if dist < best_dist:
                best_dist = dist
                best_row = row
    if best_row >= 0 and best_dist <= surface_tolerance:
        pixel = int(np.flatnonzero(row_lookup.reshape(-1) == best_row)[0])
        return (
            VISIBILITY["VISIBLE_SAME_SURFACE"], pixel, int(target_triangle[best_row]),
            float(sample_depth - target_depth[best_row]), float(target_facing[best_row]),
            bool(target_source_support[best_row]), True,
        )

    depth_gap = float(sample_depth - target_depth[central_row])
    if depth_gap > depth_epsilon:
        return (
            VISIBILITY["OCCLUDED_BY_FRONT_SURFACE"], central_linear, int(target_triangle[central_row]),
            depth_gap, float(target_facing[central_row]), bool(target_source_support[central_row]), False,
        )
    return (
        VISIBILITY["RASTER_GEOMETRY_DISAGREEMENT_OR_AMBIGUOUS"], central_linear, int(target_triangle[central_row]),
        depth_gap, float(target_facing[central_row]), bool(target_source_support[central_row]), False,
    )


def _load_graphics_view(path: Path, resolution: int) -> dict:
    with np.load(path, allow_pickle=False) as z:
        required = {
            "pixel_linear_index", "triangle_id", "surface_xyz_source_world", "surface_xyz_canonical",
            "depth_source_camera", "view_facing_confidence", "source_foreground",
            "observation_provenance",
        }
        if not required.issubset(set(z.files)):
            raise ValueError("TRUTH139_VISIBILITY_GRAPHICS_FIELDS_MISSING:" + str(path))
        value = {name: np.asarray(z[name]) for name in required}
    n = len(value["pixel_linear_index"])
    if any(len(value[name]) != n for name in required if name != "pixel_linear_index"):
        raise ValueError("TRUTH139_VISIBILITY_GRAPHICS_ROW_COUNT")
    pixels = value["pixel_linear_index"].astype(np.int64)
    if np.any(pixels < 0) or np.any(pixels >= resolution * resolution) or len(np.unique(pixels)) != len(pixels):
        raise ValueError("TRUTH139_VISIBILITY_GRAPHICS_PIXEL_INDEX")
    lookup = np.full(resolution * resolution, -1, dtype=np.int32)
    lookup[pixels] = np.arange(n, dtype=np.int32)
    value["row_lookup"] = lookup.reshape(resolution, resolution)
    value["pixel_linear_index"] = pixels.astype(np.int32)
    provenance = value["observation_provenance"].astype(np.uint8)
    source_code = PROVENANCE_CODES["SOURCE_OBSERVATION"]
    value["source_support"] = value["source_foreground"].astype(bool) & (provenance == source_code)
    return value


def _uniform_canonical_scale(graphics_manifest: dict) -> float:
    matrix = np.asarray(graphics_manifest["coordinate_contract"]["source_world_to_canonical"], dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("TRUTH139_VISIBILITY_COORDINATE_MATRIX")
    singular = np.linalg.svd(matrix[:3, :3], compute_uv=False)
    if np.max(singular) - np.min(singular) > 1.0e-6 * max(1.0, float(np.max(singular))):
        raise ValueError("TRUTH139_VISIBILITY_NONUNIFORM_CANONICAL_SCALE")
    return float(np.mean(singular))


def build(
    *,
    subject_id: str,
    graphics_manifest_path: Path,
    camera_bundle_path: Path,
    output_dir: Path,
    maximum_samples_per_source_view: int,
    search_radius_px: int,
    surface_tolerance_pixels: float,
    depth_epsilon_pixels: float,
) -> dict:
    graphics_manifest_path = graphics_manifest_path.resolve()
    camera_bundle_path = camera_bundle_path.resolve()
    output_dir = output_dir.resolve()
    manifest = json.loads(graphics_manifest_path.read_text())
    if manifest.get("schema") != GRAPHICS_SCHEMA or str(manifest.get("subject_id")) != subject_id:
        raise ValueError("TRUTH139_VISIBILITY_GRAPHICS_MANIFEST")
    cameras = _load_camera_bundle(camera_bundle_path)
    if len(cameras) != 8:
        raise ValueError("TRUTH139_VISIBILITY_CAMERA_COUNT")
    canonical_scale = _uniform_canonical_scale(manifest)
    resolution = int(cameras[0]["resolution"])
    views = []
    for row in sorted(manifest.get("views") or (), key=lambda item: int(item["view_index"])):
        views.append(_load_graphics_view(Path(row["gbuffer_path"]), resolution))
    if len(views) != 8:
        raise ValueError("TRUTH139_VISIBILITY_GRAPHICS_VIEW_COUNT")

    sample_rows = []
    for source_view, source in enumerate(views):
        picks = deterministic_stratified_rows(
            len(source["pixel_linear_index"]), maximum_samples_per_source_view,
            subject_id=subject_id, source_view=source_view,
        )
        for source_row in picks.tolist():
            sample_rows.append((source_view, int(source_row)))
    n = len(sample_rows)
    source_view_index = np.asarray([row[0] for row in sample_rows], dtype=np.uint8)
    source_surface_row = np.asarray([row[1] for row in sample_rows], dtype=np.int32)
    source_xyz = np.stack([views[v]["surface_xyz_source_world"][r] for v, r in sample_rows]).astype(np.float32)
    source_canonical = np.stack([views[v]["surface_xyz_canonical"][r] for v, r in sample_rows]).astype(np.float32)

    classes = np.full((n, 8), VISIBILITY["RASTER_GEOMETRY_DISAGREEMENT_OR_AMBIGUOUS"], dtype=np.uint8)
    projected_pixel = np.full((n, 8), -1, dtype=np.int32)
    front_triangle = np.full((n, 8), -1, dtype=np.int32)
    depth_gap = np.full((n, 8), np.nan, dtype=np.float32)
    facing = np.zeros((n, 8), dtype=np.float32)
    source_support = np.zeros((n, 8), dtype=np.bool_)
    same_surface = np.zeros((n, 8), dtype=np.bool_)

    for i in range(n):
        point_source = source_xyz[i]
        point_canonical = source_canonical[i]
        for target_view, camera in enumerate(cameras):
            inside, x_cont, y_cont, sample_depth = project_source_point(point_source, camera)
            if not inside:
                classes[i, target_view] = VISIBILITY["OUTSIDE_VIEW"]
                continue
            x = int(np.clip(np.rint(x_cont), 0, resolution - 1))
            y = int(np.clip(np.rint(y_cont), 0, resolution - 1))
            pixel_world = 2.0 * float(camera["half_extent"]) / resolution
            surface_tolerance = float(surface_tolerance_pixels) * pixel_world * canonical_scale
            depth_epsilon = float(depth_epsilon_pixels) * pixel_world
            target = views[target_view]
            result = classify_target(
                sample_canonical=point_canonical,
                sample_depth=sample_depth,
                projected_x=x,
                projected_y=y,
                row_lookup=target["row_lookup"],
                target_xyz_canonical=target["surface_xyz_canonical"],
                target_depth=target["depth_source_camera"],
                target_triangle=target["triangle_id"],
                target_facing=target["view_facing_confidence"],
                target_source_support=target["source_support"],
                surface_tolerance=surface_tolerance,
                depth_epsilon=depth_epsilon,
                search_radius_px=search_radius_px,
            )
            cls, pix, tri, gap, face_conf, support, same = result
            classes[i, target_view] = cls
            projected_pixel[i, target_view] = pix
            front_triangle[i, target_view] = tri
            depth_gap[i, target_view] = gap
            facing[i, target_view] = face_conf
            source_support[i, target_view] = support
            same_surface[i, target_view] = same

    supporting_view_mask = np.zeros(n, dtype=np.uint8)
    for view in range(8):
        supporting_view_mask |= (same_surface[:, view] & source_support[:, view]).astype(np.uint8) << view
    support_count = np.sum(same_surface & source_support, axis=1).astype(np.uint8)

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "VISIBILITY_OCCLUSION_TEACHER.npz"
    staged = out.with_suffix(out.suffix + ".partial.npz")
    np.savez_compressed(
        staged,
        source_view_index=source_view_index,
        source_surface_row=source_surface_row,
        surface_xyz_source_world=source_xyz,
        surface_xyz_canonical=source_canonical,
        target_visibility_class=classes,
        target_projected_pixel_linear_index=projected_pixel,
        target_front_triangle_id=front_triangle,
        target_depth_gap=depth_gap,
        target_view_facing_confidence=facing,
        target_source_support=source_support,
        cross_view_same_surface_match=same_surface,
        supporting_view_mask=supporting_view_mask,
        support_count=support_count,
    )
    staged.replace(out)

    class_counts = {
        name: int(np.count_nonzero(classes == code))
        for name, code in VISIBILITY.items()
    }
    payload = {
        "schema": SCHEMA,
        "date": "2026-10-29",
        "status": "PASS_VISIBILITY_OCCLUSION_TEACHER__INDEPENDENT_CPU_ORACLE_COURT_PENDING",
        "subject_id": subject_id,
        "graphics_manifest": {"path": str(graphics_manifest_path), "sha256": sha256(graphics_manifest_path)},
        "camera_bundle": {"path": str(camera_bundle_path), "sha256": sha256(camera_bundle_path)},
        "teacher_npz": {"path": str(out), "sha256": sha256(out)},
        "visibility_class_codebook": {str(code): name for name, code in VISIBILITY.items()},
        "sampling": {
            "policy": "DETERMINISTIC_STRATIFIED_DENSE_SOURCE_ROWS__NO_OUTCOME_MINING",
            "seed": "SHA256_SUBJECT_ID_AND_SOURCE_VIEW",
            "maximum_samples_per_source_view": int(maximum_samples_per_source_view),
            "sample_count": int(n),
        },
        "tolerances": {
            "search_radius_px": int(search_radius_px),
            "surface_tolerance_pixels": float(surface_tolerance_pixels),
            "depth_epsilon_pixels": float(depth_epsilon_pixels),
            "canonical_uniform_scale": canonical_scale,
        },
        "counts": {
            "samples": int(n),
            "target_evaluations": int(n * 8),
            "visibility_classes": class_counts,
            "samples_with_source_support_in_at_least_two_views": int(np.count_nonzero(support_count >= 2)),
        },
        "semantic_part_taxonomy_used": False,
        "material_or_owner_labels_used": False,
        "independent_cpu_ray_oracle_court_pending": True,
        "training_performed": False,
        "qualification_minted": False,
    }
    payload["teacher_identity_sha256"] = digest_json(payload)
    manifest_out = output_dir / "VISIBILITY_OCCLUSION_TEACHER_MANIFEST.json"
    manifest_out.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "status": payload["status"],
        "subject_id": subject_id,
        "sample_count": n,
        "class_counts": class_counts,
        "teacher_identity_sha256": payload["teacher_identity_sha256"],
    }, sort_keys=True), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--graphics-manifest", required=True, type=Path)
    parser.add_argument("--camera-bundle", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--maximum-samples-per-source-view", type=int, default=8192)
    parser.add_argument("--search-radius-px", type=int, default=1)
    parser.add_argument("--surface-tolerance-pixels", type=float, default=2.5)
    parser.add_argument("--depth-epsilon-pixels", type=float, default=0.25)
    args = parser.parse_args()
    if args.maximum_samples_per_source_view <= 0 or args.search_radius_px < 0:
        raise SystemExit("TRUTH139_VISIBILITY_INVALID_SAMPLING_ARGUMENT")
    if args.surface_tolerance_pixels <= 0 or args.depth_epsilon_pixels <= 0:
        raise SystemExit("TRUTH139_VISIBILITY_INVALID_TOLERANCE_ARGUMENT")
    build(
        subject_id=args.subject_id,
        graphics_manifest_path=args.graphics_manifest,
        camera_bundle_path=args.camera_bundle,
        output_dir=args.output_dir,
        maximum_samples_per_source_view=args.maximum_samples_per_source_view,
        search_radius_px=args.search_radius_px,
        surface_tolerance_pixels=args.surface_tolerance_pixels,
        depth_epsilon_pixels=args.depth_epsilon_pixels,
    )


if __name__ == "__main__":
    main()
