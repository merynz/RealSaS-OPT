from __future__ import annotations

"""Causal diagnostic: corrected H1 raw zero surface vs product visual-hull clip.

This experiment isolates whether artist-visible rest coverage is already absent from
the learned raw zero surface or is removed by the deterministic product clipping
step. It uses exact source observations/cameras only for measurement. It does not
use teacher geometry, owner rasters, rig, skin, component partition, motion or
completion, and it does not mutate or promote product authority.

Colors in causal overlays:
  yellow = source-visible pixel lost by product clipping but carried by raw surface
  red    = source-visible pixel missing even from raw surface
  cyan   = raw geometry first-hit where source alpha is transparent
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


SCHEMA = "RealSaS.H1RawVsProductClipCausalDiagnostic.v1"
RAW_SHA256 = "c2e12e34007a300be3e46730284dabb9a61b26dd61590e8c35d1d63e0084f39a"
CLIPPED_SHA256 = "56073e8b348b828350c812ac44982b823237196d5ec2f361241877e9ae301925"
_INPUT_MAGIC = 0x31524452
_OUTPUT_MAGIC = 0x314F4452
_VERSION = 1


def _sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_mesh(path: Path, expected_sha: str, *, raw: bool):
    actual = _sha(path)
    if actual != expected_sha:
        raise RuntimeError(f"H1_RAW_CLIP_SHA_DRIFT:{path.name}:{actual}")
    with np.load(path, allow_pickle=False) as z:
        expected = {"vertices", "faces", "normals", "vertices_norm"} if raw else {
            "vertices", "faces", "normals"
        }
        if set(z.files) != expected:
            raise RuntimeError(
                f"H1_RAW_CLIP_PAYLOAD_DRIFT:{path.name}:{sorted(z.files)}"
            )
        vertices = np.asarray(z["vertices"], dtype=np.float32)
        faces = np.asarray(z["faces"], dtype=np.uint32)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or faces.ndim != 2
        or faces.shape[1] != 3
        or np.any(faces >= len(vertices))
        or not np.isfinite(vertices).all()
    ):
        raise RuntimeError(f"H1_RAW_CLIP_MESH_INVALID:{path.name}")
    return vertices, faces


def _camera(path: Path, view: int) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or int(value.get("view_index", -1)) != int(view):
        raise RuntimeError(f"H1_RAW_CLIP_CAMERA_VIEW_DRIFT:V{view}")
    return value


def _source(path: Path, resolution: int) -> np.ndarray:
    with Image.open(path) as image:
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    if rgba.shape != (resolution, resolution, 4):
        raise RuntimeError(f"H1_RAW_CLIP_SOURCE_SHAPE_DRIFT:{path}:{rgba.shape}")
    return rgba


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
            raise RuntimeError("H1_RAW_CLIP_RASTER_HEADER_TRUNCATED")
        magic, version, width, height = struct.unpack("<IIII", header)
        if (
            magic != _OUTPUT_MAGIC
            or version != _VERSION
            or width != resolution
            or height != resolution
        ):
            raise RuntimeError("H1_RAW_CLIP_RASTER_HEADER_INVALID")
        count = int(width) * int(height)
        face = np.fromfile(f, dtype="<i4", count=count)
        depth = np.fromfile(f, dtype="<f4", count=count)
        if len(face) != count or len(depth) != count:
            raise RuntimeError("H1_RAW_CLIP_RASTER_PAYLOAD_TRUNCATED")
    return face.reshape(height, width), depth.reshape(height, width)


def _rasterize(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    camera,
    rasterizer: Path,
    scratch: Path,
    label: str,
):
    projected = project_points_xyz_v3(vertices, camera)
    inp = scratch / f"{label}.input.bin"
    out = scratch / f"{label}.output.bin"
    _write_raster_input(inp, projected, faces, int(camera.resolution))
    subprocess.run([str(rasterizer), str(inp), str(out)], check=True)
    face, depth = _read_raster_output(out, int(camera.resolution))
    inp.unlink(missing_ok=True)
    out.unlink(missing_ok=True)
    return projected, face, depth


def _coverage(authority: np.ndarray, predicted: np.ndarray) -> dict:
    fg = int(np.count_nonzero(authority))
    pred = int(np.count_nonzero(predicted))
    tp = int(np.count_nonzero(authority & predicted))
    fp = int(np.count_nonzero(predicted & ~authority))
    fn = int(np.count_nonzero(authority & ~predicted))
    structure = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    labels, count = ndimage.label(authority & ~predicted, structure=structure)
    if count:
        sizes = np.bincount(labels.ravel())[1:]
        largest = int(sizes.max(initial=0))
    else:
        largest = 0
    return {
        "foreground_pixel_count": fg,
        "predicted_pixel_count": pred,
        "true_positive_pixel_count": tp,
        "false_positive_pixel_count": fp,
        "false_negative_pixel_count": fn,
        "source_alpha_recall": float(tp / fg) if fg else 1.0,
        "precision_inside_source_alpha": float(tp / pred) if pred else 1.0,
        "alpha_iou": float(tp / (tp + fp + fn)) if (tp + fp + fn) else 1.0,
        "largest_missing_component_pixel_count": largest,
        "largest_missing_component_fraction": float(largest / fg) if fg else 0.0,
    }


def _largest_mask(authority: np.ndarray, predicted: np.ndarray):
    structure = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    labels, count = ndimage.label(authority & ~predicted, structure=structure)
    if not count:
        return np.zeros_like(authority), {
            "pixel_count": 0,
            "bbox_xyxy": None,
            "centroid_xy": None,
        }
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    label_id = int(np.argmax(sizes))
    mask = labels == label_id
    ys, xs = np.nonzero(mask)
    return mask, {
        "pixel_count": int(sizes[label_id]),
        "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        "centroid_xy": [float(xs.mean()), float(ys.mean())],
    }


def _face_key(face) -> tuple[int, int, int]:
    return tuple(sorted(map(int, face)))


def _raw_lineage_maps(raw_vertices, raw_faces, clipped_vertices, clipped_faces):
    raw_by_bytes = {raw_vertices[i].tobytes(): i for i in range(len(raw_vertices))}
    clipped_to_raw = np.asarray(
        [raw_by_bytes.get(v.tobytes(), -1) for v in clipped_vertices],
        dtype=np.int64,
    )
    if np.any(clipped_to_raw < 0) or len(set(clipped_to_raw.tolist())) != len(clipped_vertices):
        raise RuntimeError("H1_RAW_CLIP_VERTEX_SUBSET_IDENTITY_FAIL")
    kept_vertex = np.zeros(len(raw_vertices), dtype=np.bool_)
    kept_vertex[clipped_to_raw] = True

    clipped_faces_raw = clipped_to_raw[clipped_faces]
    clipped_face_keys = {_face_key(face) for face in clipped_faces_raw}
    kept_face = np.asarray(
        [_face_key(face) in clipped_face_keys for face in raw_faces],
        dtype=np.bool_,
    )
    if int(np.count_nonzero(kept_face)) != int(len(clipped_faces)):
        raise RuntimeError("H1_RAW_CLIP_FACE_SUBSET_IDENTITY_FAIL")
    return clipped_to_raw, kept_vertex, kept_face


def _out_of_frame_counts(points, cameras) -> dict[str, int]:
    out = {}
    for view, camera in enumerate(cameras):
        p = project_points_xyz_v3(points, camera)
        r = int(camera.resolution)
        outside = (
            (p[:, 0] < 0.0)
            | (p[:, 0] > float(r))
            | (p[:, 1] < 0.0)
            | (p[:, 1] > float(r))
        )
        out[f"V{view}"] = int(np.count_nonzero(outside))
    return out


def _write_overlay(
    source_rgba: np.ndarray,
    raw_mask: np.ndarray,
    clipped_mask: np.ndarray,
    path: Path,
) -> None:
    alpha = source_rgba[..., 3] >= 8
    yellow = alpha & ~clipped_mask & raw_mask
    red = alpha & ~raw_mask
    cyan = raw_mask & ~alpha
    out = source_rgba.copy()
    out[yellow] = np.asarray([255, 255, 0, 255], dtype=np.uint8)
    out[red] = np.asarray([255, 0, 0, 255], dtype=np.uint8)
    out[cyan] = np.asarray([0, 255, 255, 180], dtype=np.uint8)
    Image.fromarray(out, mode="RGBA").save(path)


def _write_contact(rows: list[dict], output: Path) -> None:
    panels = []
    for row in rows:
        with Image.open(row["causal_overlay_path"]) as im:
            panel = im.convert("RGBA").resize((256, 256), Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", (256, 282), (20, 20, 20, 255))
        canvas.alpha_composite(panel, (0, 26))
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (5, 5),
            (
                f"V{row['view']} rawR={row['raw_coverage']['source_alpha_recall']:.4f} "
                f"clipR={row['clipped_coverage']['source_alpha_recall']:.4f}"
            ),
            fill=(255, 255, 255, 255),
        )
        panels.append(canvas)
    sheet = Image.new("RGBA", (1024, 564), (10, 10, 10, 255))
    for i, panel in enumerate(panels):
        sheet.alpha_composite(panel, ((i % 4) * 256, (i // 4) * 282))
    sheet.save(output)


def run(args) -> dict:
    raw_vertices, raw_faces = _load_mesh(Path(args.raw_zero_surface), RAW_SHA256, raw=True)
    clipped_vertices, clipped_faces = _load_mesh(
        Path(args.clipped_zero_surface), CLIPPED_SHA256, raw=False
    )
    _, kept_vertex, kept_face = _raw_lineage_maps(
        raw_vertices, raw_faces, clipped_vertices, clipped_faces
    )

    cameras = tuple(
        qualify_camera_v3(
            _camera(Path(args.cameras[view]), view),
            view_id=f"V{view}",
            view_index=view,
        )
        for view in range(8)
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = out_dir / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)

    rows = []
    for view in range(8):
        source = _source(Path(args.observations[view]), int(cameras[view].resolution))
        alpha = source[..., 3] >= 8
        raw_xyz, raw_hit, _ = _rasterize(
            vertices=raw_vertices,
            faces=raw_faces,
            camera=cameras[view],
            rasterizer=Path(args.rasterizer),
            scratch=scratch,
            label=f"V{view}_raw",
        )
        _clip_xyz, clipped_hit, _ = _rasterize(
            vertices=clipped_vertices,
            faces=clipped_faces,
            camera=cameras[view],
            rasterizer=Path(args.rasterizer),
            scratch=scratch,
            label=f"V{view}_clip",
        )
        raw_mask = raw_hit >= 0
        clipped_mask = clipped_hit >= 0
        raw_cov = _coverage(alpha, raw_mask)
        clipped_cov = _coverage(alpha, clipped_mask)

        hole_mask, hole = _largest_mask(alpha, clipped_mask)
        recovered = hole_mask & raw_mask
        still_missing = hole_mask & ~raw_mask
        recovering_face_ids = np.unique(raw_hit[recovered])
        recovering_face_ids = recovering_face_ids[recovering_face_ids >= 0]
        recovering_removed = recovering_face_ids[~kept_face[recovering_face_ids]]
        recovering_vertices = (
            np.unique(raw_faces[recovering_removed].reshape(-1))
            if len(recovering_removed)
            else np.asarray([], dtype=np.int64)
        )
        removed_recovering_vertices = recovering_vertices[~kept_vertex[recovering_vertices]]
        out_of_frame = (
            _out_of_frame_counts(raw_vertices[removed_recovering_vertices], cameras)
            if len(removed_recovering_vertices)
            else {f"V{i}": 0 for i in range(8)}
        )

        overlay = out_dir / f"V{view}_RAW_VS_PRODUCT_CLIP_CAUSAL_OVERLAY.png"
        _write_overlay(source, raw_mask, clipped_mask, overlay)
        row = {
            "view": view,
            "raw_coverage": raw_cov,
            "clipped_coverage": clipped_cov,
            "largest_clipped_missing_component": {
                **hole,
                "raw_recovered_pixel_count": int(np.count_nonzero(recovered)),
                "raw_still_missing_pixel_count": int(np.count_nonzero(still_missing)),
                "raw_recovered_fraction": (
                    float(np.count_nonzero(recovered) / int(hole["pixel_count"]))
                    if int(hole["pixel_count"]) > 0
                    else 1.0
                ),
                "raw_first_hit_face_count_recovering": int(len(recovering_face_ids)),
                "recovering_faces_removed_by_product_clip": int(len(recovering_removed)),
                "recovering_removed_vertex_count": int(len(removed_recovering_vertices)),
                "recovering_removed_vertices_out_of_frame_by_view": out_of_frame,
            },
            "causal_overlay_path": str(overlay),
        }
        rows.append(row)
        print(
            "H1_RAW_CLIP_V"
            + str(view)
            + "="
            + json.dumps(
                {
                    "raw_recall": raw_cov["source_alpha_recall"],
                    "raw_precision": raw_cov["precision_inside_source_alpha"],
                    "raw_iou": raw_cov["alpha_iou"],
                    "raw_largest_missing_fraction": raw_cov[
                        "largest_missing_component_fraction"
                    ],
                    "clip_recall": clipped_cov["source_alpha_recall"],
                    "clip_precision": clipped_cov["precision_inside_source_alpha"],
                    "clip_iou": clipped_cov["alpha_iou"],
                    "clip_largest_missing_fraction": clipped_cov[
                        "largest_missing_component_fraction"
                    ],
                    "largest_clip_hole": row["largest_clipped_missing_component"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    contact = out_dir / "RAW_VS_PRODUCT_CLIP_CAUSAL_CONTACT.png"
    _write_contact(rows, contact)
    result = {
        "schema": SCHEMA,
        "status": "PASS__H1_RAW_VS_PRODUCT_CLIP_CAUSAL_DIAGNOSTIC_COMPLETED",
        "raw_zero_surface_sha256": RAW_SHA256,
        "clipped_zero_surface_sha256": CLIPPED_SHA256,
        "raw_vertex_count": int(len(raw_vertices)),
        "raw_face_count": int(len(raw_faces)),
        "clipped_vertex_count": int(len(clipped_vertices)),
        "clipped_face_count": int(len(clipped_faces)),
        "removed_vertex_count": int(len(raw_vertices) - len(clipped_vertices)),
        "removed_face_count": int(len(raw_faces) - len(clipped_faces)),
        "removed_face_fraction": float(1.0 - len(clipped_faces) / len(raw_faces)),
        "views": rows,
        "contact_sheet_path": str(contact),
        "teacher_truth_used": False,
        "source_owner_raster_used": False,
        "rig_used": False,
        "skin_used": False,
        "component_partition_used": False,
        "motion_used": False,
        "completion_used": False,
        "product_pass_claimed": False,
        "interpretation": {
            "yellow": "SOURCE_VISIBLE__PRESENT_IN_RAW_ZERO_SURFACE__REMOVED_BY_PRODUCT_CLIP",
            "red": "SOURCE_VISIBLE__MISSING_FROM_RAW_ZERO_SURFACE",
            "cyan": "RAW_ZERO_SURFACE_FIRST_HIT_OUTSIDE_SOURCE_ALPHA",
        },
    }
    output = out_dir / "H1_RAW_VS_PRODUCT_CLIP_CAUSAL_DIAGNOSTIC.json"
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"H1_RAW_CLIP_REPORT={output}", flush=True)
    print(f"H1_RAW_CLIP_CONTACT={contact}", flush=True)
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-zero-surface", required=True)
    p.add_argument("--clipped-zero-surface", required=True)
    p.add_argument("--cameras", nargs=8, required=True)
    p.add_argument("--observations", nargs=8, required=True)
    p.add_argument("--rasterizer", required=True)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
