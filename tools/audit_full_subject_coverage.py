from __future__ import annotations

"""Fail-closed CPU audit for full-subject observation/geometry coverage.

This tool is intentionally evaluator-only. It does not train, optimize, infer component
semantics, or mutate canonical artifacts. It compares known source/evaluation geometry
and predicted zero-surface geometry under the exact product cameras and RGBA alpha masks.

Typical use during Mage reclosure:

python tools/audit_full_subject_coverage.py \
  --normalized normalized.npz \
  --zero-surface ZERO_SURFACE_PRODUCT_CLIPPED.npz \
  --observations-dir SOURCE_OBSERVATIONS_20260912 \
  --cameras-dir CAMERA_DIR \
  --out MAGE_FULL_SUBJECT_COVERAGE_AUDIT.json
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def load_camera(path: Path) -> dict:
    c = json.loads(path.read_text(encoding="utf-8"))
    required = ("origin", "right", "screen_up", "forward", "half_extent", "resolution")
    if any(k not in c for k in required):
        raise RuntimeError(f"CAMERA_CONTRACT_INCOMPLETE::{path}")
    if int(c["resolution"]) <= 0 or float(c["half_extent"]) <= 0:
        raise RuntimeError(f"CAMERA_CONTRACT_INVALID::{path}")
    return c


def project(points: np.ndarray, camera: dict) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(points, dtype=np.float64)
    origin = np.asarray(camera["origin"], dtype=np.float64)
    right = np.asarray(camera["right"], dtype=np.float64)
    up = np.asarray(camera["screen_up"], dtype=np.float64)
    forward = np.asarray(camera["forward"], dtype=np.float64)
    half = float(camera["half_extent"])
    res = int(camera["resolution"])
    d = p - origin[None, :]
    gx = (d @ right) / half
    gy = -(d @ up) / half
    depth = d @ forward
    raster = np.stack(((gx + 1.0) * 0.5 * res - 0.5, (gy + 1.0) * 0.5 * res - 0.5), axis=1)
    return raster, depth


def rasterize_faces(vertices: np.ndarray, faces: np.ndarray, camera: dict) -> np.ndarray:
    res = int(camera["resolution"])
    xy, _depth = project(vertices, camera)
    im = Image.new("1", (res, res), 0)
    draw = ImageDraw.Draw(im)
    for tri in np.asarray(faces, dtype=np.int64):
        draw.polygon([tuple(xy[int(i)]) for i in tri], fill=1)
    return np.asarray(im, dtype=bool).copy()


def mask_metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    pred = np.asarray(pred, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    inter = int(np.logical_and(pred, truth).sum())
    union = int(np.logical_or(pred, truth).sum())
    return {
        "pred_pixels": int(pred.sum()),
        "truth_pixels": int(truth.sum()),
        "intersection_pixels": inter,
        "precision": float(inter / max(int(pred.sum()), 1)),
        "recall": float(inter / max(int(truth.sum()), 1)),
        "iou": float(inter / max(union, 1)),
    }


def load_observation(path: Path, *, alpha_threshold: int) -> np.ndarray:
    with Image.open(path) as im:
        if im.mode != "RGBA":
            raise RuntimeError(f"OBSERVATION_NOT_RGBA::{path}")
        a = np.asarray(im, dtype=np.uint8)[..., 3]
    return a >= int(alpha_threshold)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--normalized", type=Path, required=True)
    ap.add_argument("--zero-surface", type=Path, required=True)
    ap.add_argument("--observations-dir", type=Path, required=True)
    ap.add_argument("--cameras-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--alpha-threshold", type=int, default=8)
    ap.add_argument("--min-full-source-recall", type=float, default=0.98)
    args = ap.parse_args()

    with np.load(args.normalized, allow_pickle=False) as d:
        required = ("vertices_source", "faces", "skin")
        if any(k not in d.files for k in required):
            raise RuntimeError("NORMALIZED_SOURCE_REQUIRED_FIELDS_MISSING")
        full_v = np.asarray(d["vertices_source"], dtype=np.float64)
        full_f = np.asarray(d["faces"], dtype=np.int64)
        skin = np.asarray(d["skin"], dtype=np.float64)

    if skin.ndim != 2 or skin.shape[0] != len(full_v):
        raise RuntimeError("NORMALIZED_SKIN_SHAPE_DRIFT")
    if full_f.ndim != 2 or full_f.shape[1] != 3:
        raise RuntimeError("NORMALIZED_FACE_SHAPE_DRIFT")

    supported_v = skin.sum(axis=1) > 1e-8
    supported_f = full_f[supported_v[full_f].all(axis=1)]
    used = np.flatnonzero(supported_v)
    remap = np.full(len(full_v), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    supported_vertices = full_v[used]
    supported_faces = remap[supported_f]

    with np.load(args.zero_surface, allow_pickle=False) as d:
        if "vertices" not in d.files or "faces" not in d.files:
            raise RuntimeError("ZERO_SURFACE_REQUIRED_FIELDS_MISSING")
        zero_v = np.asarray(d["vertices"], dtype=np.float64)
        zero_f = np.asarray(d["faces"], dtype=np.int64)

    views = []
    for v in range(8):
        obs_path = args.observations_dir / f"V{v}.png"
        cam_path = args.cameras_dir / f"V{v}.camera.json"
        if not obs_path.is_file():
            raise RuntimeError(f"OBSERVATION_MISSING_V{v}")
        if not cam_path.is_file():
            raise RuntimeError(f"CAMERA_MISSING_V{v}")
        camera = load_camera(cam_path)
        if int(camera.get("view_index", v)) != v:
            raise RuntimeError(f"CAMERA_VIEW_INDEX_DRIFT_V{v}")
        if int(camera["resolution"]) != Image.open(obs_path).size[0]:
            raise RuntimeError(f"CAMERA_OBSERVATION_RESOLUTION_DRIFT_V{v}")

        alpha = load_observation(obs_path, alpha_threshold=args.alpha_threshold)
        full_mask = rasterize_faces(supported_vertices, supported_faces, camera)
        zero_mask = rasterize_faces(zero_v, zero_f, camera)
        full_metrics = mask_metrics(full_mask, alpha)
        zero_metrics = mask_metrics(zero_mask, alpha)
        views.append({
            "view_index": v,
            "camera_sha256": sha256_file(cam_path),
            "observation_sha256": sha256_file(obs_path),
            "full_deformation_supported_source_vs_alpha": full_metrics,
            "zero_surface_vs_alpha": zero_metrics,
        })

    min_full = min(x["full_deformation_supported_source_vs_alpha"]["recall"] for x in views)
    status = "PASS__OBSERVATION_CAMERA_FULL_SOURCE_CONTRACT" if min_full >= args.min_full_source_recall else "FAIL__FULL_SOURCE_DOES_NOT_REPLAY_PRODUCT_OBSERVATION"
    report = {
        "schema": "RealSaS.FullSubjectCoverageAudit.v1",
        "status": status,
        "alpha_threshold": int(args.alpha_threshold),
        "min_full_source_recall_gate": float(args.min_full_source_recall),
        "normalized_source_sha256": sha256_file(args.normalized),
        "zero_surface_sha256": sha256_file(args.zero_surface),
        "source": {
            "vertices_total": int(len(full_v)),
            "faces_total": int(len(full_f)),
            "deformation_supported_vertices": int(len(supported_vertices)),
            "deformation_supported_faces": int(len(supported_faces)),
            "zero_skin_vertices": int((~supported_v).sum()),
        },
        "views": views,
        "teacher_used_at_product_inference": False,
        "canonical_mutation": False,
        "note": "This is an evaluator-only replay. Full source geometry is evaluation authority, never product inference input.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(status)
    for row in views:
        f = row["full_deformation_supported_source_vs_alpha"]
        z = row["zero_surface_vs_alpha"]
        print(f"V{row['view_index']}: full_recall={f['recall']:.6f} zero_recall={z['recall']:.6f} zero_precision={z['precision']:.6f}")
    print(args.out)
    if not status.startswith("PASS"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
