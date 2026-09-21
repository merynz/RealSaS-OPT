from __future__ import annotations

"""VF-11 subject-free thin-feature survival calibration.

This extends the exact Stage13 numerical-floor apparatus. Source truth is analytic
box-prism geometry rasterized by the same HALF_INTEGER_TOP_LEFT product rasterizer.
The measured path is:

    analytic source geometry -> source raster
    exact signed CSG field @ 256^3 -> ZeroSurfaceDecoder.MarchingCubes.v3
    -> same 1024 product rasterizer -> feature-survival measurements

No subject asset, learned output, Knight result, Mage result, or teacher geometry is
consumed. This tool measures the decoder/discretization floor only and deliberately
does not select a shipping threshold.
"""

import gc
import json
import math
import os
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.mesh.product_coverage_v1 import (
    coverage_metrics,
    rasterize_triangles_half_integer_top_left,
    source_connected_component_recall_metrics,
)
from compiler.realsas_compiler_core.playback_full_surface_v3 import (
    CameraProjectionV3,
    project_points_xyz_v3,
)
from compiler.realsas_compiler_core.rest_preservation_v1 import (
    silhouette_distance_metrics,
)
from models.iris.v3.zero_surface_decoder_v3 import extract_zero_surface_mesh_v3


OUT = Path(os.environ.get("REALSAS_VF11_OUT", "vf11_out"))
OUT.mkdir(parents=True, exist_ok=True)
REFERENCE_DECODER_RESOLUTION = 256
R = int(os.environ.get("REALSAS_VF11_DECODER_RESOLUTION", "256"))
RES = 1024
BOUNDS = (-1.0, 1.0)
if R < REFERENCE_DECODER_RESOLUTION:
    raise RuntimeError("VF11_DECODER_RESOLUTION_BELOW_REFERENCE_FORBIDDEN")
REFERENCE_VOXEL = (
    (BOUNDS[1] - BOUNDS[0]) / float(REFERENCE_DECODER_RESOLUTION - 1)
)
VOXEL = (BOUNDS[1] - BOUNDS[0]) / float(R - 1)
WIDTH_LADDER_REFERENCE_VOXELS = (0.75, 1.0, 1.5, 2.0)
FROZEN_STAGE13_PROFILE = {
    "min_recall": 0.999,
    "min_precision": 0.999,
    "max_largest_coherent_hole_fraction": 0.00025,
    "max_interior_uncovered_fraction": 0.0005,
    "min_component_recall": 0.999,
    "max_silhouette_edge_p95_px": 0.5,
}


def _write(name: str, payload: dict) -> None:
    (OUT / name).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _cameras() -> tuple[CameraProjectionV3, ...]:
    rows = []
    for view in range(8):
        yaw = math.radians(45.0 * view)
        forward = np.asarray(
            [-math.sin(yaw), -math.cos(yaw), 0.0],
            dtype=np.float64,
        )
        right = np.asarray(
            [-math.cos(yaw), math.sin(yaw), 0.0],
            dtype=np.float64,
        )
        rows.append(
            CameraProjectionV3(
                f"V{view}",
                view,
                tuple((-4.0 * forward).tolist()),
                tuple(right.tolist()),
                (0.0, 0.0, 1.0),
                tuple(forward.tolist()),
                1.0,
                RES,
            )
        )
    return tuple(rows)


def _box(
    center: tuple[float, float, float],
    half: tuple[float, float, float],
) -> dict:
    return {
        "center": tuple(map(float, center)),
        "half": tuple(map(float, half)),
    }


def _case_boxes(
    family: str,
    width_reference_voxels: float,
) -> tuple[tuple[dict, ...], tuple[dict, ...], str]:
    # World-space feature width is frozen to the original R256 ladder so
    # resolution sweeps change only signed-field sampling density.
    w = float(width_reference_voxels) * REFERENCE_VOXEL
    core = _box((0.0, 0.0, -0.10), (0.30, 0.22, 0.34))

    if family == "THIN_STRAP":
        feature = _box((0.0, 0.0, 0.0), (0.5 * w, 0.20, 0.45))
        return (feature,), (feature,), "POSITIVE_FEATURE_RECALL"

    if family == "FORK_GAP":
        prong = 0.11
        base = _box((0.0, 0.0, -0.36), (0.36, 0.22, 0.10))
        x = 0.5 * w + 0.5 * prong
        left = _box((-x, 0.0, 0.02), (0.5 * prong, 0.22, 0.48))
        right = _box((x, 0.0, 0.02), (0.5 * prong, 0.22, 0.48))
        gap = _box((0.0, 0.0, 0.02), (0.5 * w, 0.23, 0.48))
        return (base, left, right), (gap,), "NEGATIVE_SPACE_PRESERVATION"

    if family == "CONCAVE_NOTCH":
        # C-shaped prism: the open right-side notch has narrow vertical width w.
        spine = _box((-0.22, 0.0, 0.0), (0.10, 0.22, 0.46))
        upper_z = 0.25 * (0.92 + w)
        band_half_z = 0.25 * (0.92 - w)
        upper = _box((0.04, 0.0, upper_z), (0.26, 0.22, band_half_z))
        lower = _box((0.04, 0.0, -upper_z), (0.26, 0.22, band_half_z))
        notch = _box((0.17, 0.0, 0.0), (0.13, 0.23, 0.5 * w))
        return (spine, upper, lower), (notch,), "NEGATIVE_SPACE_PRESERVATION"

    if family == "THIN_BLADE":
        blade = _box((0.50, 0.0, 0.12), (0.20, 0.12, 0.5 * w))
        return (core, blade), (blade,), "POSITIVE_FEATURE_RECALL"

    if family == "SMALL_COMPONENT":
        half = 0.5 * w
        small = _box((0.55, 0.0, 0.42), (half, half, half))
        return (core, small), (small,), "POSITIVE_FEATURE_RECALL"

    raise KeyError(family)


def _box_sdf(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    box: dict,
) -> np.ndarray:
    cx, cy, cz = box["center"]
    hx, hy, hz = box["half"]
    qx = np.abs(x - cx) - hx
    qy = np.abs(y - cy) - hy
    qz = np.abs(z - cz) - hz
    outside = np.sqrt(
        np.maximum(qx, 0.0) ** 2
        + np.maximum(qy, 0.0) ** 2
        + np.maximum(qz, 0.0) ** 2
    )
    inside = np.minimum(np.maximum.reduce((qx, qy, qz)), 0.0)
    return (outside + inside).astype(np.float32)


def _field(boxes: tuple[dict, ...]) -> np.ndarray:
    axis = np.linspace(BOUNDS[0], BOUNDS[1], R, dtype=np.float32)
    z, y, x = np.meshgrid(axis, axis, axis, indexing="ij")
    field = np.full((R, R, R), np.inf, dtype=np.float32)
    for box in boxes:
        field = np.minimum(field, _box_sdf(x, y, z, box))
    return field


def _box_mesh(box: dict) -> tuple[np.ndarray, np.ndarray]:
    c = np.asarray(box["center"], dtype=np.float64)
    h = np.asarray(box["half"], dtype=np.float64)
    signs = np.asarray(
        [
            (-1, -1, -1),
            (1, -1, -1),
            (1, 1, -1),
            (-1, 1, -1),
            (-1, -1, 1),
            (1, -1, 1),
            (1, 1, 1),
            (-1, 1, 1),
        ],
        dtype=np.float64,
    )
    vertices = c[None, :] + signs * h[None, :]
    faces = np.asarray(
        [
            (0, 2, 1), (0, 3, 2),
            (4, 5, 6), (4, 6, 7),
            (0, 1, 5), (0, 5, 4),
            (1, 2, 6), (1, 6, 5),
            (2, 3, 7), (2, 7, 6),
            (3, 0, 4), (3, 4, 7),
        ],
        dtype=np.int64,
    )
    return vertices, faces


def _render_triangle_sets(
    triangle_sets: tuple[tuple[np.ndarray, np.ndarray], ...],
    camera: CameraProjectionV3,
) -> np.ndarray:
    triangles = []
    for vertices, faces in triangle_sets:
        projected = project_points_xyz_v3(vertices, camera)
        for face in faces:
            a, b, c = (projected[int(i)] for i in face)
            triangles.append(
                (
                    (float(a[0]), float(a[1])),
                    (float(b[0]), float(b[1])),
                    (float(c[0]), float(c[1])),
                )
            )
    raw = rasterize_triangles_half_integer_top_left(
        triangles,
        width=RES,
        height=RES,
    )
    return np.frombuffer(raw, dtype=np.uint8).reshape(RES, RES).astype(bool)


def _render_boxes(boxes: tuple[dict, ...], camera: CameraProjectionV3) -> np.ndarray:
    return _render_triangle_sets(tuple(_box_mesh(box) for box in boxes), camera)


def _render_decoder_mesh(mesh, camera: CameraProjectionV3) -> np.ndarray:
    vertices = np.asarray(mesh.vertices_normalized, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    return _render_triangle_sets(((vertices, faces),), camera)


def _coverage(source: np.ndarray, predicted: np.ndarray) -> dict:
    raw_s = np.asarray(source, dtype=np.uint8).reshape(-1).tobytes()
    raw_p = np.asarray(predicted, dtype=np.uint8).reshape(-1).tobytes()
    cov = coverage_metrics(raw_s, raw_p, width=RES, height=RES)
    if np.any(source) and np.any(predicted):
        edge_mean, edge_p95, edge_max = silhouette_distance_metrics(
            source,
            predicted,
        )
    elif np.array_equal(source, predicted):
        edge_mean = edge_p95 = edge_max = 0.0
    else:
        # Empty predicted/source silhouette is catastrophic, but it is still a
        # valid measurement. Do not turn the scientifically important failure
        # into an apparatus exception.
        edge_mean = edge_p95 = edge_max = math.hypot(RES, RES)
    component = source_connected_component_recall_metrics(
        raw_s,
        raw_p,
        width=RES,
        height=RES,
        minimum_foreground_fraction=0.0,
    )
    return {
        "recall": float(cov["recall"]),
        "precision": float(cov["precision"]),
        "largest_coherent_hole_fraction": float(cov["largest_coherent_hole_fraction"]),
        "interior_uncovered_fraction": float(cov["interior_uncovered_fraction"]),
        "silhouette_edge_mean_px": float(edge_mean),
        "silhouette_edge_p95_px": float(edge_p95),
        "silhouette_edge_max_px": float(edge_max),
        "source_component_count": int(component["source_component_count"]),
        "minimum_component_recall": float(
            component["minimum_eligible_component_recall"]
        ),
    }


def _stage13_profile_pass(metrics: dict) -> bool:
    return bool(
        metrics["recall"] >= FROZEN_STAGE13_PROFILE["min_recall"]
        and metrics["precision"] >= FROZEN_STAGE13_PROFILE["min_precision"]
        and metrics["largest_coherent_hole_fraction"]
        <= FROZEN_STAGE13_PROFILE["max_largest_coherent_hole_fraction"]
        and metrics["interior_uncovered_fraction"]
        <= FROZEN_STAGE13_PROFILE["max_interior_uncovered_fraction"]
        and metrics["minimum_component_recall"]
        >= FROZEN_STAGE13_PROFILE["min_component_recall"]
        and metrics["silhouette_edge_p95_px"]
        <= FROZEN_STAGE13_PROFILE["max_silhouette_edge_p95_px"]
    )


def _feature_metric(
    *,
    mode: str,
    feature_boxes: tuple[dict, ...],
    source: np.ndarray,
    predicted: np.ndarray,
    camera: CameraProjectionV3,
) -> dict:
    probe = _render_boxes(feature_boxes, camera)
    if mode == "POSITIVE_FEATURE_RECALL":
        visible_feature = probe & source
        count = int(np.count_nonzero(visible_feature))
        kept = int(np.count_nonzero(predicted & visible_feature))
        return {
            "mode": mode,
            "source_visible_feature_pixels": count,
            "preserved_feature_pixels": kept,
            "feature_recall": 1.0 if count == 0 else float(kept) / float(count),
        }

    if mode == "NEGATIVE_SPACE_PRESERVATION":
        visible_gap = probe & ~source
        count = int(np.count_nonzero(visible_gap))
        filled = int(np.count_nonzero(predicted & visible_gap))
        return {
            "mode": mode,
            "source_visible_gap_pixels": count,
            "incorrectly_filled_gap_pixels": filled,
            "gap_fill_fraction": 0.0 if count == 0 else float(filled) / float(count),
            "gap_preservation": 1.0 if count == 0 else 1.0 - float(filled) / float(count),
        }
    raise KeyError(mode)


def main() -> int:
    import skimage

    families = (
        "THIN_STRAP",
        "FORK_GAP",
        "CONCAVE_NOTCH",
        "THIN_BLADE",
        "SMALL_COMPONENT",
    )
    cameras = _cameras()
    output = {
        "schema": "RealSaS.VF11GeometryFeatureSurvivalPanel.v2",
        "status": "MEASURED_SUBJECT_FREE__FROZEN_STAGE13_PROFILE_EVALUATED",
        "subject_inputs_used": False,
        "knight_result_used": False,
        "mage_result_used": False,
        "teacher_geometry_used": False,
        "reference_decoder_resolution": REFERENCE_DECODER_RESOLUTION,
        "decoder_resolution": R,
        "raster_resolution": RES,
        "bounds": list(BOUNDS),
        "reference_voxel_size": REFERENCE_VOXEL,
        "decoder_voxel_size": VOXEL,
        "pixels_per_reference_voxel_at_cardinal_unit_extent": (
            REFERENCE_VOXEL * (RES / 2.0)
        ),
        "decoder_id": "RealSaS.ZeroSurfaceDecoder.MarchingCubes.v3",
        "rasterizer": "HALF_INTEGER_TOP_LEFT__PRODUCT_COVERAGE_V1",
        "skimage_version": skimage.__version__,
        "width_ladder_reference_voxels": list(
            WIDTH_LADDER_REFERENCE_VOXELS
        ),
        "frozen_stage13_profile": dict(FROZEN_STAGE13_PROFILE),
        "new_threshold_selected": False,
        "cases": [],
    }

    for family in families:
        for width_reference_voxels in WIDTH_LADDER_REFERENCE_VOXELS:
            boxes, feature_boxes, feature_mode = _case_boxes(
                family,
                width_reference_voxels,
            )
            grid = _field(boxes)
            fmin = float(np.min(grid))
            fmax = float(np.max(grid))
            decoder_status = "EXTRACTED"
            mesh = None
            try:
                mesh = extract_zero_surface_mesh_v3(
                    grid,
                    bounds=BOUNDS,
                    level=0.0,
                )
            except ValueError as exc:
                if "zero level is not bracketed" not in str(exc):
                    raise
                decoder_status = "NO_ZERO_BRACKET"

            view_rows = []
            for camera in cameras:
                source = _render_boxes(boxes, camera)
                predicted = (
                    np.zeros_like(source)
                    if mesh is None
                    else _render_decoder_mesh(mesh, camera)
                )
                metrics = _coverage(source, predicted)
                metrics["view_index"] = int(camera.view_index)
                metrics["source_foreground_pixels"] = int(np.count_nonzero(source))
                metrics["predicted_foreground_pixels"] = int(np.count_nonzero(predicted))
                metrics["feature"] = _feature_metric(
                    mode=feature_mode,
                    feature_boxes=feature_boxes,
                    source=source,
                    predicted=predicted,
                    camera=camera,
                )
                metrics["frozen_stage13_profile_passed"] = (
                    _stage13_profile_pass(metrics)
                )
                view_rows.append(metrics)

            feature_recall = [
                row["feature"]["feature_recall"]
                for row in view_rows
                if row["feature"]["mode"] == "POSITIVE_FEATURE_RECALL"
                and row["feature"]["source_visible_feature_pixels"] > 0
            ]
            gap_preservation = [
                row["feature"]["gap_preservation"]
                for row in view_rows
                if row["feature"]["mode"] == "NEGATIVE_SPACE_PRESERVATION"
                and row["feature"]["source_visible_gap_pixels"] > 0
            ]
            output["cases"].append(
                {
                    "case_id": (
                        f"{family}__"
                        f"{str(width_reference_voxels).replace('.', 'p')}REFVX"
                        f"__R{R}"
                    ),
                    "family": family,
                    "feature_width_reference_voxels": float(
                        width_reference_voxels
                    ),
                    "feature_width_decoder_voxels": float(
                        width_reference_voxels * REFERENCE_VOXEL / VOXEL
                    ),
                    "feature_width_world": (
                        float(width_reference_voxels) * REFERENCE_VOXEL
                    ),
                    "feature_width_cardinal_pixels": (
                        float(width_reference_voxels)
                        * REFERENCE_VOXEL
                        * (RES / 2.0)
                    ),
                    "decoder_status": decoder_status,
                    "field_min": fmin,
                    "field_max": fmax,
                    "vertex_count": 0 if mesh is None else int(len(mesh.vertices_normalized)),
                    "face_count": 0 if mesh is None else int(len(mesh.faces)),
                    "views": view_rows,
                    "worst": {
                        "min_recall": min(row["recall"] for row in view_rows),
                        "min_precision": min(row["precision"] for row in view_rows),
                        "max_largest_coherent_hole_fraction": max(
                            row["largest_coherent_hole_fraction"] for row in view_rows
                        ),
                        "max_interior_uncovered_fraction": max(
                            row["interior_uncovered_fraction"] for row in view_rows
                        ),
                        "max_silhouette_edge_p95_px": max(
                            row["silhouette_edge_p95_px"] for row in view_rows
                        ),
                        "min_component_recall": min(
                            row["minimum_component_recall"] for row in view_rows
                        ),
                        "min_feature_recall": (
                            min(feature_recall) if feature_recall else None
                        ),
                        "min_gap_preservation": (
                            min(gap_preservation) if gap_preservation else None
                        ),
                        "all_views_frozen_stage13_profile_passed": bool(
                            all(
                                row["frozen_stage13_profile_passed"]
                                for row in view_rows
                            )
                        ),
                    },
                }
            )
            del grid
            del mesh
            gc.collect()

    by_family = {}
    for family in families:
        rows = [case for case in output["cases"] if case["family"] == family]
        by_family[family] = [
            {
                "feature_width_reference_voxels": case[
                    "feature_width_reference_voxels"
                ],
                "feature_width_decoder_voxels": case[
                    "feature_width_decoder_voxels"
                ],
                "feature_width_cardinal_pixels": case[
                    "feature_width_cardinal_pixels"
                ],
                "decoder_status": case["decoder_status"],
                **case["worst"],
            }
            for case in rows
        ]
    output["summary_by_family"] = by_family

    _write("VF11_GEOMETRY_FEATURE_SURVIVAL_PANEL.json", output)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
