from __future__ import annotations

"""Raw dense zero-surface rest reprojection diagnostic.

Purpose: isolate upstream geometry/camera/raster continuity from all downstream
materialization, component partition, skin, motion, appearance-donor and completion
logic. The exact sealed dense zero-surface is rasterized in rest pose with the same
Runtime-v3/v4 camera and pixel-center/depth primitive used by the native runtime.

The source image is NOT used to choose or delete faces. It is used only after
rasterization to classify geometry coverage and to copy DIRECT_SOURCE pixels into a
diagnostic source-preserving render.

This script is diagnostic-only. It does not claim PRODUCT_PASS and does not mutate
any product authority.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import struct
import subprocess

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from compiler.realsas_compiler_core.playback_full_surface_v3 import (
    project_points_xyz_v3,
    qualify_camera_v3,
)


SCHEMA = "RealSaS.RawDenseRestReprojectionDiagnostic.v1"
EXPECTED_ZERO_SURFACE_SHA256 = (
    "56073e8b348b828350c812ac44982b823237196d5ec2f361241877e9ae301925"
)
_INPUT_MAGIC = 0x31524452
_OUTPUT_MAGIC = 0x314F4452
_VERSION = 1


def _sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_zero_surface(path: Path):
    actual = _sha(path)
    if actual != EXPECTED_ZERO_SURFACE_SHA256:
        raise RuntimeError(f"RAW_DENSE_ZERO_SURFACE_SHA_DRIFT:{actual}")
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != {"vertices", "faces", "normals"}:
            raise RuntimeError(
                f"RAW_DENSE_ZERO_SURFACE_PAYLOAD_DRIFT:{sorted(z.files)}"
            )
        vertices = np.asarray(z["vertices"], dtype=np.float64)
        faces = np.asarray(z["faces"], dtype=np.uint32)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or faces.ndim != 2
        or faces.shape[1] != 3
        or np.any(faces >= len(vertices))
        or not np.isfinite(vertices).all()
    ):
        raise RuntimeError("RAW_DENSE_ZERO_SURFACE_INVALID")
    return vertices, faces


def _camera(path: Path, view: int) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or int(value.get("view_index", -1)) != int(view):
        raise RuntimeError(f"RAW_DENSE_CAMERA_VIEW_DRIFT:V{view}")
    return value


def _observation(path: Path, resolution: int) -> np.ndarray:
    with Image.open(path) as image:
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    if rgba.shape != (resolution, resolution, 4):
        raise RuntimeError(
            f"RAW_DENSE_OBSERVATION_SHAPE_DRIFT:{path}:{rgba.shape}"
        )
    return rgba


def _coverage(authority: np.ndarray, predicted: np.ndarray) -> dict:
    foreground = int(np.count_nonzero(authority))
    predicted_count = int(np.count_nonzero(predicted))
    tp = int(np.count_nonzero(authority & predicted))
    fp = int(np.count_nonzero(predicted & ~authority))
    fn = int(np.count_nonzero(authority & ~predicted))
    recall = float(tp / foreground) if foreground else 1.0
    precision = float(tp / predicted_count) if predicted_count else 1.0
    denom = tp + fp + fn
    iou = float(tp / denom) if denom else 1.0

    structure = np.asarray(
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8
    )
    missing = authority & ~predicted
    labels, count = ndimage.label(missing, structure=structure)
    if count:
        sizes = np.bincount(labels.ravel())[1:]
        largest_missing = int(sizes.max(initial=0))
    else:
        largest_missing = 0

    extra = predicted & ~authority
    extra_labels, extra_count = ndimage.label(extra, structure=structure)
    if extra_count:
        extra_sizes = np.bincount(extra_labels.ravel())[1:]
        largest_extra = int(extra_sizes.max(initial=0))
    else:
        largest_extra = 0

    return {
        "foreground_pixel_count": foreground,
        "predicted_geometry_pixel_count": predicted_count,
        "true_positive_pixel_count": tp,
        "false_positive_geometry_pixel_count": fp,
        "false_negative_source_pixel_count": fn,
        "source_alpha_recall": recall,
        "geometry_precision_inside_source_alpha": precision,
        "alpha_iou": iou,
        "largest_missing_component_pixel_count": largest_missing,
        "largest_missing_component_fraction": (
            float(largest_missing) / float(foreground) if foreground else 0.0
        ),
        "largest_extra_geometry_component_pixel_count": largest_extra,
    }


def _largest_missing(authority: np.ndarray, predicted: np.ndarray) -> dict:
    structure = np.asarray(
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8
    )
    missing = authority & ~predicted
    labels, count = ndimage.label(missing, structure=structure)
    if not count:
        return {"pixel_count": 0, "bbox_xyxy": None, "centroid_xy": None}
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    label_id = int(np.argmax(sizes))
    mask = labels == label_id
    ys, xs = np.nonzero(mask)
    return {
        "pixel_count": int(sizes[label_id]),
        "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        "centroid_xy": [float(xs.mean()), float(ys.mean())],
    }


def _write_raster_input(
    path: Path,
    projected_xyz: np.ndarray,
    faces: np.ndarray,
    resolution: int,
) -> None:
    xyz = np.ascontiguousarray(projected_xyz, dtype="<f4")
    tri = np.ascontiguousarray(faces, dtype="<u4")
    with path.open("wb") as f:
        f.write(
            struct.pack(
                "<IIIIQQ",
                _INPUT_MAGIC,
                _VERSION,
                int(resolution),
                int(resolution),
                int(len(xyz)),
                int(len(tri)),
            )
        )
        xyz.tofile(f)
        tri.tofile(f)


def _read_raster_output(path: Path, resolution: int):
    with path.open("rb") as f:
        header = f.read(struct.calcsize("<IIII"))
        if len(header) != struct.calcsize("<IIII"):
            raise RuntimeError("RAW_DENSE_RASTER_OUTPUT_HEADER_TRUNCATED")
        magic, version, width, height = struct.unpack("<IIII", header)
        if (
            magic != _OUTPUT_MAGIC
            or version != _VERSION
            or width != resolution
            or height != resolution
        ):
            raise RuntimeError("RAW_DENSE_RASTER_OUTPUT_HEADER_INVALID")
        count = int(width) * int(height)
        face = np.fromfile(f, dtype="<i4", count=count)
        depth = np.fromfile(f, dtype="<f4", count=count)
        if len(face) != count or len(depth) != count:
            raise RuntimeError("RAW_DENSE_RASTER_OUTPUT_PAYLOAD_TRUNCATED")
    return face.reshape(height, width), depth.reshape(height, width)


def _write_images(
    source_rgba: np.ndarray,
    geometry_mask: np.ndarray,
    output_prefix: Path,
) -> dict:
    alpha = source_rgba[..., 3] >= 8
    visible_source = geometry_mask & alpha

    direct = np.zeros_like(source_rgba)
    direct[visible_source] = source_rgba[visible_source]
    direct_path = output_prefix.with_name(
        output_prefix.name + "_DIRECT_SOURCE_REPROJECTION.png"
    )
    Image.fromarray(direct, mode="RGBA").save(direct_path)

    overlay = source_rgba.copy()
    missing = alpha & ~geometry_mask
    extra = geometry_mask & ~alpha
    overlay[missing, 0] = 255
    overlay[missing, 1] = (overlay[missing, 1].astype(np.uint16) // 5).astype(np.uint8)
    overlay[missing, 2] = (overlay[missing, 2].astype(np.uint16) // 5).astype(np.uint8)
    overlay[missing, 3] = 255

    # Cyan shows geometry that exists in front projection where the source is transparent.
    overlay[extra, 0] = 0
    overlay[extra, 1] = 255
    overlay[extra, 2] = 255
    overlay[extra, 3] = 180
    overlay_path = output_prefix.with_name(
        output_prefix.name + "_COVERAGE_OVERLAY.png"
    )
    Image.fromarray(overlay, mode="RGBA").save(overlay_path)

    return {
        "direct_source_reprojection_path": str(direct_path),
        "coverage_overlay_path": str(overlay_path),
    }


def _write_contact_sheet(rows: list[dict], output: Path) -> None:
    panels = []
    for row in rows:
        view = int(row["view"])
        path = Path(row["images"]["direct_source_reprojection_path"])
        with Image.open(path) as im:
            panel = im.convert("RGBA").resize((256, 256), Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", (256, 282), (20, 20, 20, 255))
        canvas.alpha_composite(panel, (0, 26))
        draw = ImageDraw.Draw(canvas)
        cov = row["coverage"]
        label = (
            f"V{view}  R={cov['source_alpha_recall']:.4f}  "
            f"P={cov['geometry_precision_inside_source_alpha']:.4f}"
        )
        draw.text((6, 6), label, fill=(255, 255, 255, 255))
        panels.append(canvas)

    cols = 4
    rows_n = (len(panels) + cols - 1) // cols
    sheet = Image.new("RGBA", (cols * 256, rows_n * 282), (10, 10, 10, 255))
    for i, panel in enumerate(panels):
        sheet.alpha_composite(panel, ((i % cols) * 256, (i // cols) * 282))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def run(args) -> dict:
    vertices, faces = _load_zero_surface(Path(args.zero_surface))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    requested_views = tuple(int(v) for v in args.views)
    if not requested_views or any(v < 0 or v > 7 for v in requested_views):
        raise RuntimeError("RAW_DENSE_VIEW_SET_INVALID")

    rows = []
    for view in requested_views:
        raw_camera = _camera(Path(args.cameras[view]), view)
        camera = qualify_camera_v3(raw_camera, view_id=f"V{view}", view_index=view)
        source = _observation(Path(args.observations[view]), int(camera.resolution))
        alpha = source[..., 3] >= 8

        projected = project_points_xyz_v3(vertices, camera)
        in_bin = out_dir / f"V{view}_RAW_DENSE_RASTER_INPUT.bin"
        out_bin = out_dir / f"V{view}_RAW_DENSE_RASTER_OUTPUT.bin"
        _write_raster_input(in_bin, projected, faces, int(camera.resolution))
        subprocess.run(
            [str(Path(args.rasterizer)), str(in_bin), str(out_bin)],
            check=True,
        )
        face_id, depth = _read_raster_output(out_bin, int(camera.resolution))
        geometry = face_id >= 0
        coverage = _coverage(alpha, geometry)
        largest = _largest_missing(alpha, geometry)

        prefix = out_dir / f"V{view}_RAW_DENSE"
        image_paths = _write_images(source, geometry, prefix)

        finite_depth = depth[np.isfinite(depth)]
        row = {
            "view": view,
            "camera_schema": camera.schema_version,
            "vertex_count": int(len(vertices)),
            "face_count": int(len(faces)),
            "visible_first_hit_face_count": int(len(np.unique(face_id[face_id >= 0]))),
            "geometry_pixel_count": int(np.count_nonzero(geometry)),
            "depth_min": float(finite_depth.min()) if len(finite_depth) else None,
            "depth_max": float(finite_depth.max()) if len(finite_depth) else None,
            "coverage": coverage,
            "largest_missing_component": largest,
            "images": image_paths,
            "source_pixel_copy_is_exact_by_construction": True,
            "source_alpha_used_for_face_selection": False,
            "source_rgb_used_for_geometry": False,
        }
        rows.append(row)
        print(
            "RAW_DENSE_REST_V"
            + str(view)
            + "="
            + json.dumps(
                {
                    "recall": coverage["source_alpha_recall"],
                    "precision": coverage["geometry_precision_inside_source_alpha"],
                    "iou": coverage["alpha_iou"],
                    "largest_missing_fraction": coverage[
                        "largest_missing_component_fraction"
                    ],
                    "largest_missing": largest,
                    "geometry_pixels": int(np.count_nonzero(geometry)),
                    "visible_faces": row["visible_first_hit_face_count"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

        if not args.keep_binary:
            in_bin.unlink(missing_ok=True)
            out_bin.unlink(missing_ok=True)

    contact = out_dir / "RAW_DENSE_DIRECT_SOURCE_CONTACT_SHEET.png"
    _write_contact_sheet(rows, contact)

    result = {
        "schema": SCHEMA,
        "status": "PASS__RAW_DENSE_REST_REPROJECTION_DIAGNOSTIC_COMPLETED",
        "hypothesis": (
            "FULL_SEALED_DENSE_ZERO_SURFACE_WITH_RUNTIME_CAMERA_AND_DEPTH_SEMANTICS_"
            "SHOULD_CARRY_THE_REST_SOURCE_SILHOUETTE_WITHOUT_DOWNSTREAM_FACE_SELECTION"
        ),
        "interpretation_rule": {
            "diagnostic_only": True,
            "product_pass_claimed": False,
            "if_coherent_missing_region_disappears": (
                "DOWNSTREAM_DRAWABLE_MATERIALIZATION_OR_FACE_SELECTION_IS_CAUSAL"
            ),
            "if_same_coherent_missing_region_persists": (
                "UPSTREAM_DENSE_GEOMETRY_CAMERA_OR_RASTER_ALIGNMENT_REQUIRES_DIAGNOSIS"
            ),
        },
        "zero_surface_sha256": EXPECTED_ZERO_SURFACE_SHA256,
        "views": rows,
        "contact_sheet_path": str(contact),
        "teacher_truth_used": False,
        "source_owner_raster_used": False,
        "component_partition_used": False,
        "rig_used": False,
        "skin_used": False,
        "motion_used": False,
        "completion_used": False,
        "source_alpha_used_for_face_selection": False,
        "runtime_reference_raster_semantics_used": True,
    }
    output = out_dir / "RAW_DENSE_REST_REPROJECTION_DIAGNOSTIC.json"
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"RAW_DENSE_REST_REPORT={output}", flush=True)
    print(f"RAW_DENSE_REST_CONTACT={contact}", flush=True)
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zero-surface", required=True)
    p.add_argument("--cameras", nargs=8, required=True)
    p.add_argument("--observations", nargs=8, required=True)
    p.add_argument("--rasterizer", required=True)
    p.add_argument("--views", nargs="+", default=tuple(str(v) for v in range(8)))
    p.add_argument("--output-dir", required=True)
    p.add_argument("--keep-binary", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
