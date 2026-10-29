#!/usr/bin/env python3
"""Build Truth139 cross-view visibility/occlusion truth with silhouette-safe correspondence search.

V2 preserves the V1 schema and storage contract but fixes one classification-order defect: a
projected centre pixel may be background while a declared one-pixel neighbourhood contains the
same surface because the continuous projection falls on a raster/silhouette boundary.  V2 searches
for same-surface support first, then decides background/occlusion from the centre pixel.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.corpus import build_truth139_visibility_occlusion_teacher_v1 as v1
from tools.corpus.build_truth139_dense_graphics_gbuffer_v1 import sha256

SCHEMA = v1.SCHEMA
VISIBILITY = v1.VISIBILITY
PRODUCER_REVISION = "V2_SILHOUETTE_NEIGHBOR_FIRST"


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
    if search_radius_px < 0 or surface_tolerance <= 0 or depth_epsilon <= 0:
        raise ValueError("TRUTH139_VISIBILITY_V2_CLASSIFICATION_ARGUMENT")

    central_row = int(row_lookup[projected_y, projected_x])
    central_linear = projected_y * w + projected_x

    # Search correspondence before declaring the centre background.  This is essential at a
    # silhouette/raster boundary where the projected continuous point may round onto the adjacent
    # background pixel even though a nearby front raster row reconstructs the same 3D surface.
    best_row = -1
    best_pixel = -1
    best_dist = np.inf
    for yy in range(max(0, projected_y - search_radius_px), min(h, projected_y + search_radius_px + 1)):
        for xx in range(max(0, projected_x - search_radius_px), min(w, projected_x + search_radius_px + 1)):
            row = int(row_lookup[yy, xx])
            if row < 0:
                continue
            dist = float(np.linalg.norm(np.asarray(target_xyz_canonical[row], dtype=np.float64) - sample_canonical))
            pixel = yy * w + xx
            if dist < best_dist or (dist == best_dist and pixel < best_pixel):
                best_dist = dist
                best_row = row
                best_pixel = pixel
    if best_row >= 0 and best_dist <= surface_tolerance:
        return (
            VISIBILITY["VISIBLE_SAME_SURFACE"], int(best_pixel), int(target_triangle[best_row]),
            float(sample_depth - target_depth[best_row]), float(target_facing[best_row]),
            bool(target_source_support[best_row]), True,
        )

    if central_row < 0:
        return VISIBILITY["PROJECTED_TO_BACKGROUND_OR_UNSUPPORTED"], central_linear, -1, np.nan, 0.0, False, False

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
    if manifest.get("schema") != v1.GRAPHICS_SCHEMA or str(manifest.get("subject_id")) != subject_id:
        raise ValueError("TRUTH139_VISIBILITY_V2_GRAPHICS_MANIFEST")
    cameras = v1._load_camera_bundle(camera_bundle_path)
    if len(cameras) != 8:
        raise ValueError("TRUTH139_VISIBILITY_V2_CAMERA_COUNT")
    canonical_scale = v1._uniform_canonical_scale(manifest)
    resolution = int(cameras[0]["resolution"])
    views = [
        v1._load_graphics_view(Path(row["gbuffer_path"]), resolution)
        for row in sorted(manifest.get("views") or (), key=lambda item: int(item["view_index"]))
    ]
    if len(views) != 8:
        raise ValueError("TRUTH139_VISIBILITY_V2_GRAPHICS_VIEW_COUNT")

    sample_rows: list[tuple[int, int]] = []
    for source_view, source in enumerate(views):
        picks = v1.deterministic_stratified_rows(
            len(source["pixel_linear_index"]), maximum_samples_per_source_view,
            subject_id=subject_id, source_view=source_view,
        )
        sample_rows.extend((source_view, int(source_row)) for source_row in picks.tolist())
    if not sample_rows:
        raise ValueError("TRUTH139_VISIBILITY_V2_NO_SURFACE_ROWS")

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
            inside, x_cont, y_cont, sample_depth = v1.project_source_point(point_source, camera)
            if not inside:
                classes[i, target_view] = VISIBILITY["OUTSIDE_VIEW"]
                continue
            x = int(np.clip(np.rint(x_cont), 0, resolution - 1))
            y = int(np.clip(np.rint(y_cont), 0, resolution - 1))
            pixel_world = 2.0 * float(camera["half_extent"]) / resolution
            target = views[target_view]
            cls, pix, tri, gap, face_conf, support, same = classify_target(
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
                surface_tolerance=float(surface_tolerance_pixels) * pixel_world * canonical_scale,
                depth_epsilon=float(depth_epsilon_pixels) * pixel_world,
                search_radius_px=search_radius_px,
            )
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

    class_counts = {name: int(np.count_nonzero(classes == code)) for name, code in VISIBILITY.items()}
    payload = {
        "schema": SCHEMA,
        "date": "2026-10-29",
        "status": "PASS_VISIBILITY_OCCLUSION_TEACHER__INDEPENDENT_CPU_ORACLE_COURT_PENDING",
        "producer_revision": PRODUCER_REVISION,
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
        "classification_order": "SAME_SURFACE_NEIGHBOR_SEARCH_THEN_BACKGROUND_THEN_OCCLUSION",
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
    payload["teacher_identity_sha256"] = v1.digest_json(payload)
    manifest_out = output_dir / "VISIBILITY_OCCLUSION_TEACHER_MANIFEST.json"
    manifest_out.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "status": payload["status"],
        "producer_revision": PRODUCER_REVISION,
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
        raise SystemExit("TRUTH139_VISIBILITY_V2_INVALID_SAMPLING_ARGUMENT")
    if args.surface_tolerance_pixels <= 0 or args.depth_epsilon_pixels <= 0:
        raise SystemExit("TRUTH139_VISIBILITY_V2_INVALID_TOLERANCE_ARGUMENT")
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
