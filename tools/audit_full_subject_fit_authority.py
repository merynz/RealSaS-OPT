from __future__ import annotations

"""Fail-closed audit for raster/teacher subject-set agreement before any geometry FIT.

The learned geometry model may use source geometry as FIT truth, but the source geometry
admitted as teacher authority must explain the same visible subject rendered into the
product observations. This audit detects the historical failure mode where the renderer
showed components that the signed-field teacher silently omitted.

The tool is model-agnostic. It requires only normalized source geometry/skin, the exact
8 orthographic cameras and the exact 8 RGBA observations. It performs no optimization.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


SCHEMA = "RealSaS.FullSubjectFitAuthorityAudit.v1"


def _sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_camera(path: Path) -> dict:
    row = json.loads(path.read_text(encoding="utf-8"))
    required = ("origin", "right", "screen_up", "forward", "half_extent", "resolution", "center")
    missing = [k for k in required if k not in row]
    if missing:
        raise RuntimeError(f"CAMERA_CONTRACT_INCOMPLETE::{path}::{missing}")
    if int(row["resolution"]) <= 0 or float(row["half_extent"]) <= 0:
        raise RuntimeError(f"CAMERA_CONTRACT_INVALID::{path}")
    return row


def _project(points: np.ndarray, camera: dict) -> tuple[np.ndarray, np.ndarray]:
    origin = np.asarray(camera["origin"], dtype=np.float64)
    right = np.asarray(camera["right"], dtype=np.float64)
    up = np.asarray(camera["screen_up"], dtype=np.float64)
    forward = np.asarray(camera["forward"], dtype=np.float64)
    right /= np.linalg.norm(right)
    up /= np.linalg.norm(up)
    forward /= np.linalg.norm(forward)
    half = float(camera["half_extent"])
    resolution = int(camera["resolution"])
    d = np.asarray(points, dtype=np.float64) - origin[None, :]
    gx = (d @ right) / half
    gy = -(d @ up) / half
    depth = d @ forward
    raster = np.stack(
        ((gx + 1.0) * 0.5 * resolution - 0.5, (gy + 1.0) * 0.5 * resolution - 0.5),
        axis=1,
    )
    return raster, depth


def _rasterize_faces(vertices: np.ndarray, faces: np.ndarray, camera: dict) -> np.ndarray:
    resolution = int(camera["resolution"])
    xy, depth = _project(vertices, camera)
    canvas = Image.new("1", (resolution, resolution), 0)
    draw = ImageDraw.Draw(canvas)
    for tri in np.asarray(faces, dtype=np.int64):
        if np.any(depth[tri] <= 0.0):
            continue
        draw.polygon([tuple(xy[int(i)]) for i in tri], fill=1)
    return np.asarray(canvas, dtype=bool).copy()


def _mask_metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    inter = int(np.logical_and(pred, truth).sum())
    union = int(np.logical_or(pred, truth).sum())
    pp = int(pred.sum())
    tp = int(truth.sum())
    return {
        "teacher_pixels": pp,
        "observation_alpha_pixels": tp,
        "intersection_pixels": inter,
        "recall": float(inter / max(tp, 1)),
        "precision": float(inter / max(pp, 1)),
        "iou": float(inter / max(union, 1)),
    }


def audit(
    normalized_path: Path,
    camera_paths: tuple[Path, ...],
    observation_paths: tuple[Path, ...],
    *,
    alpha_threshold: int,
    min_recall: float,
    min_precision: float,
    min_iou: float,
    skin_mass_epsilon: float,
) -> dict:
    if len(camera_paths) != 8 or len(observation_paths) != 8:
        raise RuntimeError("EXACTLY_8_CAMERAS_AND_8_OBSERVATIONS_REQUIRED")

    with np.load(normalized_path, allow_pickle=False) as d:
        required = {"vertices_source", "faces", "skin"}
        if not required.issubset(d.files):
            raise RuntimeError(f"NORMALIZED_SOURCE_FIELDS_MISSING::{sorted(required-set(d.files))}")
        vertices = np.asarray(d["vertices_source"], dtype=np.float64)
        faces = np.asarray(d["faces"], dtype=np.int64)
        skin = np.asarray(d["skin"], dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise RuntimeError("NORMALIZED_VERTICES_INVALID")
    if faces.ndim != 2 or faces.shape[1] != 3 or np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise RuntimeError("NORMALIZED_FACES_INVALID")
    if skin.ndim != 2 or skin.shape[0] != len(vertices):
        raise RuntimeError("NORMALIZED_SKIN_INVALID")
    if not np.isfinite(vertices).all() or not np.isfinite(skin).all() or np.any(skin < -1e-8):
        raise RuntimeError("NORMALIZED_SOURCE_NONFINITE_OR_NEGATIVE_SKIN")

    mechanical_vertex = skin.sum(axis=1) > float(skin_mass_epsilon)
    mechanical_faces = faces[mechanical_vertex[faces].all(axis=1)]
    excluded_faces = faces[~mechanical_vertex[faces].all(axis=1)]
    if len(mechanical_faces) == 0:
        raise RuntimeError("MECHANICAL_TEACHER_EMPTY")

    cameras = tuple(_load_camera(p) for p in camera_paths)
    centers = np.asarray([c["center"] for c in cameras], dtype=np.float64)
    halves = np.asarray([float(c["half_extent"]) for c in cameras], dtype=np.float64)
    resolutions = tuple(int(c["resolution"]) for c in cameras)
    if not np.allclose(centers, centers[0], rtol=0.0, atol=1e-9):
        raise RuntimeError("CAMERA_NORMALIZATION_CENTER_DRIFT")
    if not np.allclose(halves, halves[0], rtol=0.0, atol=1e-12):
        raise RuntimeError("CAMERA_HALF_EXTENT_DRIFT")
    if len(set(resolutions)) != 1:
        raise RuntimeError("CAMERA_RESOLUTION_DRIFT")

    views = []
    failures = []
    for view, (camera, obs_path) in enumerate(zip(cameras, observation_paths)):
        with Image.open(obs_path) as im:
            if im.mode != "RGBA" or im.size != (resolutions[view], resolutions[view]):
                raise RuntimeError(f"OBSERVATION_CONTRACT_DRIFT_V{view}")
            rgba = np.asarray(im, dtype=np.uint8)
        alpha = rgba[..., 3] >= int(alpha_threshold)
        teacher_mask = _rasterize_faces(vertices, mechanical_faces, camera)
        metrics = _mask_metrics(teacher_mask, alpha)
        metrics.update({
            "view": view,
            "camera_sha256": _sha(camera_paths[view]),
            "observation_sha256": _sha(obs_path),
        })
        view_fail = []
        if metrics["recall"] < float(min_recall):
            view_fail.append("RECALL")
        if metrics["precision"] < float(min_precision):
            view_fail.append("PRECISION")
        if metrics["iou"] < float(min_iou):
            view_fail.append("IOU")
        metrics["gate_failures"] = view_fail
        failures.extend(f"V{view}_{x}" for x in view_fail)
        views.append(metrics)

    status = "PASS__FULL_SUBJECT_FIT_AUTHORITY" if not failures else "FAIL__RASTER_TEACHER_SUBJECT_SCOPE_MISMATCH"
    report = {
        "schema": SCHEMA,
        "status": status,
        "optimizer_constructed": False,
        "teacher_policy": "SKIN_SUPPORTED_SOURCE_GEOMETRY__NO_SILENT_RENDERED_COMPONENT_OMISSION",
        "normalized_source_sha256": _sha(normalized_path),
        "source_vertex_count": int(len(vertices)),
        "source_face_count": int(len(faces)),
        "mechanically_supported_vertex_count": int(mechanical_vertex.sum()),
        "mechanically_supported_face_count": int(len(mechanical_faces)),
        "zero_or_unsupported_vertex_count": int((~mechanical_vertex).sum()),
        "faces_not_fully_mechanically_supported": int(len(excluded_faces)),
        "camera_center": centers[0].tolist(),
        "camera_half_extent": float(halves[0]),
        "resolution": int(resolutions[0]),
        "gates": {
            "alpha_threshold": int(alpha_threshold),
            "min_recall_per_view": float(min_recall),
            "min_precision_per_view": float(min_precision),
            "min_iou_per_view": float(min_iou),
            "skin_mass_epsilon": float(skin_mass_epsilon),
        },
        "views": views,
        "failures": failures,
        "fit_authorized": not failures,
        "product_pass_claimed": False,
    }
    return report


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--normalized", required=True)
    p.add_argument("--cameras", nargs=8, required=True)
    p.add_argument("--observations", nargs=8, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--alpha-threshold", type=int, default=8)
    p.add_argument("--min-recall", type=float, default=0.98)
    p.add_argument("--min-precision", type=float, default=0.98)
    p.add_argument("--min-iou", type=float, default=0.97)
    p.add_argument("--skin-mass-epsilon", type=float, default=1e-8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = audit(
        Path(args.normalized),
        tuple(Path(x) for x in args.cameras),
        tuple(Path(x) for x in args.observations),
        alpha_threshold=args.alpha_threshold,
        min_recall=args.min_recall,
        min_precision=args.min_precision,
        min_iou=args.min_iou,
        skin_mass_epsilon=args.skin_mass_epsilon,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report["status"])
    for row in report["views"]:
        print(
            f"V{row['view']}: recall={row['recall']:.6f} precision={row['precision']:.6f} iou={row['iou']:.6f}"
        )
    print(out)
    if not report["fit_authorized"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
