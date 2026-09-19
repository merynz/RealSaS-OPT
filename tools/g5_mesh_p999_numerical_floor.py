from __future__ import annotations

import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from compiler.realsas_compiler_core.mesh.product_coverage_v1 import (
    coverage_metrics,
    rasterize_visible_component_masks,
)
from compiler.realsas_compiler_core.playback_full_surface_v3 import CameraProjectionV3
from compiler.realsas_compiler_core.rest_preservation_v1 import silhouette_distance_metrics
from models.iris.v3.zero_surface_decoder_v3 import extract_zero_surface_mesh_v3

OUT = Path(os.environ.get("REALSAS_G5_FLOOR_OUT", "g5_floor_out"))
OUT.mkdir(parents=True, exist_ok=True)

GRID_RESOLUTION = 256
RASTER_RESOLUTION = 1024
COMPONENT_ID = "C0"

P999 = {
    "min_recall": 0.999,
    "min_precision": 0.999,
    "max_largest_coherent_hole_fraction": 0.00025,
    "max_interior_uncovered_fraction": 0.0005,
    "max_silhouette_edge_p95_px": 0.5,
}


def _write(name: str, payload: dict) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cams():
    rows = []
    for view in range(8):
        yaw = math.radians(45.0 * view)
        forward = np.asarray([-math.sin(yaw), -math.cos(yaw), 0.0], dtype=np.float64)
        right = np.asarray([-math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float64)
        rows.append(
            CameraProjectionV3(
                view_id=f"V{view}",
                view_index=view,
                origin=tuple((-4.0 * forward).tolist()),
                right=tuple(right.tolist()),
                screen_up=(0.0, 0.0, 1.0),
                forward=tuple(forward.tolist()),
                half_extent=1.0,
                resolution=RASTER_RESOLUTION,
            )
        )
    return rows


def _analytic_grid(kind: str) -> np.ndarray:
    axis = np.linspace(-1.0, 1.0, GRID_RESOLUTION, dtype=np.float32)
    z, y, x = np.meshgrid(axis, axis, axis, indexing="ij")
    if kind == "SPHERE":
        return np.sqrt(x * x + y * y + z * z).astype(np.float32) - np.float32(0.65)
    if kind == "ELLIPSOID":
        a, b, c = np.float32(0.68), np.float32(0.42), np.float32(0.78)
        return (np.sqrt((x / a) ** 2 + (y / b) ** 2 + (z / c) ** 2) - 1.0).astype(np.float32)
    raise KeyError(kind)


def _source_mask(kind: str, camera: CameraProjectionV3) -> np.ndarray:
    yy, xx = np.mgrid[0:RASTER_RESOLUTION, 0:RASTER_RESOLUTION]
    X = xx + 0.5
    Y = yy + 0.5
    cx = cy = RASTER_RESOLUTION / 2.0

    if kind == "SPHERE":
        rx = ry = 0.65 * (RASTER_RESOLUTION / 2.0)
    elif kind == "ELLIPSOID":
        a, b, c = 0.68, 0.42, 0.78
        right = np.asarray(camera.right, dtype=np.float64)
        rx = math.sqrt((a * right[0]) ** 2 + (b * right[1]) ** 2 + (c * right[2]) ** 2) * (RASTER_RESOLUTION / 2.0)
        ry = c * (RASTER_RESOLUTION / 2.0)
    else:
        raise KeyError(kind)

    return (((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2) <= 1.0


def _product_mesh(mesh):
    vertices = []
    vertex_ids = []
    for i, p in enumerate(np.asarray(mesh.vertices_normalized, dtype=np.float64)):
        vid = f"v{i}"
        vertex_ids.append(vid)
        vertices.append(
            SimpleNamespace(
                canonical_mesh_vertex_id=vid,
                component_id=COMPONENT_ID,
                P=tuple(map(float, p)),
            )
        )
    faces = tuple(tuple(vertex_ids[int(i)] for i in tri) for tri in np.asarray(mesh.faces, dtype=np.int64))
    return SimpleNamespace(vertices=tuple(vertices), faces=faces)


def _mask_bytes(mask: np.ndarray) -> bytes:
    return np.asarray(mask, dtype=np.uint8).reshape(-1).tobytes()


def _metrics(source: np.ndarray, predicted: np.ndarray) -> dict:
    cov = coverage_metrics(
        _mask_bytes(source),
        _mask_bytes(predicted),
        width=RASTER_RESOLUTION,
        height=RASTER_RESOLUTION,
    )
    edge_mean, edge_p95, edge_max = silhouette_distance_metrics(source, predicted)
    return {
        "recall": float(cov["recall"]),
        "precision": float(cov["precision"]),
        "largest_coherent_hole_fraction": float(cov["largest_coherent_hole_fraction"]),
        "interior_uncovered_fraction": float(cov["interior_uncovered_fraction"]),
        "silhouette_edge_mean_px": float(edge_mean),
        "silhouette_edge_p95_px": float(edge_p95),
        "silhouette_edge_max_px": float(edge_max),
        "foreground_pixel_count": int(cov["foreground_pixel_count"]),
        "predicted_pixel_count": int(cov["predicted_pixel_count"]),
    }


def _passes(m: dict) -> bool:
    return (
        m["recall"] >= P999["min_recall"]
        and m["precision"] >= P999["min_precision"]
        and m["largest_coherent_hole_fraction"] <= P999["max_largest_coherent_hole_fraction"]
        and m["interior_uncovered_fraction"] <= P999["max_interior_uncovered_fraction"]
        and m["silhouette_edge_p95_px"] <= P999["max_silhouette_edge_p95_px"]
    )


def main() -> int:
    import skimage

    result = {
        "schema": "RealSaS.G5MeshP999NumericalFloor.v1",
        "status": "PASS",
        "subject_inputs_used": False,
        "knight_result_used": False,
        "mage_result_used": False,
        "carrier_class": "MESH",
        "component_count": 1,
        "decoder_id": "RealSaS.ZeroSurfaceDecoder.MarchingCubes.v3",
        "decoder_resolution": GRID_RESOLUTION,
        "raster_resolution": RASTER_RESOLUTION,
        "g5_rasterizer": "rasterize_visible_component_masks__CANONICAL_Z_BUFFER_VISIBLE_OWNER_V1",
        "skimage_version": skimage.__version__,
        "tested_profile": P999,
        "shapes": {},
    }

    overall = True
    cameras = _cams()
    for kind in ("SPHERE", "ELLIPSOID"):
        zero = extract_zero_surface_mesh_v3(_analytic_grid(kind), bounds=(-1.0, 1.0), level=0.0)
        product = _product_mesh(zero)
        views = []
        for camera in cameras:
            predicted_by_component = rasterize_visible_component_masks(
                product,
                camera,
                width=RASTER_RESOLUTION,
                height=RASTER_RESOLUTION,
            )
            predicted = np.frombuffer(
                predicted_by_component[COMPONENT_ID],
                dtype=np.uint8,
            ).reshape(RASTER_RESOLUTION, RASTER_RESOLUTION).astype(bool)
            source = _source_mask(kind, camera)
            m = _metrics(source, predicted)
            m["view_index"] = int(camera.view_index)
            m["passed"] = bool(_passes(m))
            views.append(m)
            overall &= bool(m["passed"])

        result["shapes"][kind] = {
            "vertex_count": int(len(zero.vertices_normalized)),
            "face_count": int(len(zero.faces)),
            "views": views,
            "worst": {
                "min_recall": min(v["recall"] for v in views),
                "min_precision": min(v["precision"] for v in views),
                "max_largest_coherent_hole_fraction": max(v["largest_coherent_hole_fraction"] for v in views),
                "max_interior_uncovered_fraction": max(v["interior_uncovered_fraction"] for v in views),
                "max_silhouette_edge_p95_px": max(v["silhouette_edge_p95_px"] for v in views),
                "max_silhouette_edge_max_px": max(v["silhouette_edge_max_px"] for v in views),
            },
        }

    result["status"] = "PASS" if overall else "FAIL_P999_BELOW_EXACT_G5_MESH_NUMERICAL_FLOOR"
    _write("G5_MESH_P999_NUMERICAL_FLOOR.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if overall else 2


if __name__ == "__main__":
    raise SystemExit(main())
