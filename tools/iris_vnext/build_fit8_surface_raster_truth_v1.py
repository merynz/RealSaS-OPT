from __future__ import annotations

"""Materialize source-defensible SURF-A1 mask/depth/normal training truth.

The authored teacher mesh and Stage01 cameras/masks are training truth only.  The
output is never an IRIS inference input.  Mask authority remains the exact Stage01
foreground bytes; depth and normal are rendered from the exact authored rest mesh
at pixel centers and are valid only where that mesh coverage intersects source
foreground.
"""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

SCHEMA = "RealSaS.SurfA1RasterTruthBatch.v1"
SUBJECT_SCHEMA = "RealSaS.SurfA1RasterTruthSubject.v1"
EXPECTED_SUBJECTS = ("Snake", "Bat", "T-Rex", "Horse", "Spider", "GreenBlob", "Shark", "George")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("SURF_A1_RASTER_JSON_OBJECT_REQUIRED")
    return value


def _find_one(archive: zipfile.ZipFile, suffix: str) -> str:
    matches = [name for name in archive.namelist() if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"SURF_A1_RASTER_ARCHIVE_CARDINALITY:{suffix}:{len(matches)}")
    return matches[0]


def _downsample_mask(mask: np.ndarray, target: int) -> np.ndarray:
    if mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
        raise ValueError("SURF_A1_RASTER_MASK_SHAPE_INVALID")
    source = int(mask.shape[0])
    if target <= 0 or source % int(target) != 0:
        raise ValueError("SURF_A1_RASTER_MASK_RESOLUTION_NOT_DIVISIBLE")
    factor = source // int(target)
    return mask.reshape(target, factor, target, factor).max(axis=(1, 3))


def _unit(value) -> np.ndarray:
    a = np.asarray(value, dtype=np.float64)
    n = float(np.linalg.norm(a))
    if not np.isfinite(n) or n <= 1e-12:
        raise ValueError("SURF_A1_RASTER_CAMERA_AXIS_INVALID")
    return a / n


def _raster_depth_normal(
    vertices_world: np.ndarray,
    faces: np.ndarray,
    camera: dict,
    *,
    resolution: int,
    common_half_extent: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = np.asarray(vertices_world, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("SURF_A1_RASTER_TEACHER_GEOMETRY_SHAPE_INVALID")
    if len(vertices) == 0 or len(faces) == 0 or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("SURF_A1_RASTER_TEACHER_GEOMETRY_INVALID")
    half = float(camera["half_extent"])
    common_half = float(common_half_extent)
    if not np.isfinite(half) or not np.isfinite(common_half) or min(half, common_half) <= 0:
        raise ValueError("SURF_A1_RASTER_CAMERA_SCALE_INVALID")
    origin = np.asarray(camera["origin"], dtype=np.float64)
    right = _unit(camera["right"])
    up = _unit(camera["screen_up"])
    forward = _unit(camera["forward"])
    delta = vertices - origin[None]
    x = (delta @ right / half + 1.0) * (0.5 * resolution)
    y = (-(delta @ up) / half + 1.0) * (0.5 * resolution)
    screen = np.stack((x, y), axis=1)
    depth_world = delta @ forward

    tri_world = vertices[faces]
    tri_normal = np.cross(tri_world[:, 1] - tri_world[:, 0], tri_world[:, 2] - tri_world[:, 0])
    norm = np.linalg.norm(tri_normal, axis=1)
    safe = norm > 1e-12
    tri_normal[safe] /= norm[safe, None]
    tri_normal[~safe] = 0.0

    zbuf = np.full((resolution, resolution), np.inf, dtype=np.float64)
    normal = np.zeros((resolution, resolution, 3), dtype=np.float32)
    eps = 1e-9
    for face_index, face in enumerate(faces):
        if not safe[face_index]:
            continue
        tri = screen[face]
        lo = np.maximum(np.floor(tri.min(axis=0)).astype(np.int64), 0)
        hi = np.minimum(np.ceil(tri.max(axis=0)).astype(np.int64) - 1, resolution - 1)
        if np.any(lo > hi):
            continue
        x0, y0 = tri[0]
        x1, y1 = tri[1]
        x2, y2 = tri[2]
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(float(denom)) <= 1e-12:
            continue
        yy, xx = np.mgrid[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1]
        px = xx.astype(np.float64) + 0.5
        py = yy.astype(np.float64) + 0.5
        w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denom
        w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -eps) & (w1 >= -eps) & (w2 >= -eps)
        if not inside.any():
            continue
        d = w0 * depth_world[face[0]] + w1 * depth_world[face[1]] + w2 * depth_world[face[2]]
        clip_start = float(camera.get("clip_start", 0.0))
        clip_end = float(camera.get("clip_end", np.inf))
        inside &= np.isfinite(d) & (d >= clip_start) & (d <= clip_end)
        if not inside.any():
            continue
        view = zbuf[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1]
        update = inside & (d < view)
        if update.any():
            view[update] = d[update]
            nview = normal[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1]
            nview[update] = tri_normal[face_index].astype(np.float32)

    valid = np.isfinite(zbuf)
    depth = np.zeros_like(zbuf, dtype=np.float32)
    depth[valid] = (zbuf[valid] / common_half).astype(np.float32)
    return depth, normal, valid


def build(args: argparse.Namespace) -> dict:
    binding = load_json(Path(args.binding).resolve())
    if binding.get("schema") != "RealSaS.SurfA1FrozenHInputBinding.v1":
        raise ValueError("SURF_A1_RASTER_BINDING_SCHEMA_INVALID")
    stage_zip = Path(args.stage01_bundle).resolve()
    expected_stage_sha = str(binding["stage01"]["sha256"])
    if sha256_file(stage_zip) != expected_stage_sha:
        raise ValueError("SURF_A1_RASTER_STAGE01_SHA_DRIFT")
    teacher_root = Path(args.teacher_dir).resolve()
    teacher_batch_path = Path(args.teacher_batch_receipt).resolve()
    expected_batch = str(binding["authored_teachers"]["batch_receipt"]["sha256"])
    if sha256_file(teacher_batch_path) != expected_batch:
        raise ValueError("SURF_A1_RASTER_TEACHER_BATCH_SHA_DRIFT")
    teacher_batch = load_json(teacher_batch_path)
    if teacher_batch.get("schema") != "RealSaS.IRISVNextFit8TeacherBatch.v1":
        raise ValueError("SURF_A1_RASTER_TEACHER_BATCH_SCHEMA_INVALID")
    bind_rows = {str(r["subject_id"]): r for r in binding["authored_teachers"]["subjects"]}
    batch_rows = {str(r["subject_id"]): r for r in teacher_batch.get("subjects") or ()}
    if tuple(bind_rows) != EXPECTED_SUBJECTS or tuple(batch_rows) != EXPECTED_SUBJECTS:
        raise ValueError("SURF_A1_RASTER_SUBJECT_ORDER_DRIFT")
    output_root = Path(args.output_dir).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError("SURF_A1_RASTER_OUTPUT_DIR_MUST_BE_EMPTY")
    output_root.mkdir(parents=True, exist_ok=True)
    target = int(args.resolution)
    subject_rows = []

    with zipfile.ZipFile(stage_zip, "r") as archive:
        for subject_id in EXPECTED_SUBJECTS:
            bind = bind_rows[subject_id]
            batch = batch_rows[subject_id]
            if bind["legacy_subject_id"] != batch["legacy_subject_id"] or bind["file"] != batch["teacher_file"]:
                raise ValueError("SURF_A1_RASTER_TEACHER_BINDING_DRIFT:" + subject_id)
            if bind["sha256"] != batch["teacher_sha256"]:
                raise ValueError("SURF_A1_RASTER_TEACHER_HASH_METADATA_DRIFT:" + subject_id)
            teacher_path = (teacher_root / str(bind["file"])).resolve()
            if not teacher_path.is_relative_to(teacher_root) or sha256_file(teacher_path) != bind["sha256"]:
                raise ValueError("SURF_A1_RASTER_TEACHER_FILE_DRIFT:" + subject_id)
            with np.load(teacher_path, allow_pickle=False) as data:
                vertices = np.asarray(data["rest_vertices_source_world"], dtype=np.float64)
                faces = np.asarray(data["faces"], dtype=np.int64)
            legacy = str(bind["legacy_subject_id"])
            camera_name = _find_one(archive, f"observations/{legacy}/camera_bundle.json")
            cameras_bytes = archive.read(camera_name)
            cameras = json.loads(cameras_bytes)
            camera_rows = list(cameras.get("cameras") or ())
            if len(camera_rows) != 8 or cameras.get("per_view_pose_change") or cameras.get("per_view_recenter"):
                raise ValueError("SURF_A1_RASTER_CAMERA_CONTRACT_DRIFT:" + subject_id)
            common_half = float(cameras["common_half_extent"])
            masks, depths, normals, valids = [], [], [], []
            source_mask_hashes = []
            valid_fractions = []
            base_prefix = camera_name[: -len("camera_bundle.json")]
            for camera in camera_rows:
                view_index = int(camera["view_index"])
                source_resolution = int(camera["resolution"])
                mask_name = base_prefix + f"V{view_index}.mask.bin"
                raw_mask = archive.read(mask_name)
                source_mask_hashes.append(sha256_bytes(raw_mask))
                mask = np.frombuffer(raw_mask, dtype=np.uint8).reshape(source_resolution, source_resolution) > 0
                mask = _downsample_mask(mask, target)
                depth, normal, valid = _raster_depth_normal(vertices, faces, camera, resolution=target, common_half_extent=common_half)
                valid &= mask
                depth[~valid] = 0.0
                normal[~valid] = 0.0
                masks.append(mask.astype(np.uint8))
                depths.append(depth)
                normals.append(normal)
                valids.append(valid.astype(np.uint8))
                valid_fractions.append(float(valid.mean()))
            payload_path = output_root / f"{subject_id}.surf_a1_raster_truth.npz"
            np.savez_compressed(
                payload_path,
                mask=np.stack(masks),
                depth_normalized=np.stack(depths),
                normal_world=np.stack(normals),
                depth_normal_valid=np.stack(valids),
            )
            row = {
                "schema": SUBJECT_SCHEMA,
                "subject_id": subject_id,
                "legacy_subject_id": legacy,
                "teacher_sha256": bind["sha256"],
                "camera_bundle_sha256": sha256_bytes(cameras_bytes),
                "source_mask_sha256": source_mask_hashes,
                "resolution": target,
                "view_count": 8,
                "mean_depth_normal_valid_fraction": float(np.mean(valid_fractions)),
                "payload_file": payload_path.name,
                "payload_sha256": sha256_file(payload_path),
                "teacher_truth_in_inference_forward": False,
                "optimizer_steps_executed": 0,
                "scientific_pass_claimed": False,
            }
            receipt_path = output_root / f"{subject_id}.surf_a1_raster_truth.json"
            receipt_path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            row["receipt_sha256"] = sha256_file(receipt_path)
            subject_rows.append(row)
            print(f"SURF_A1_RASTER_TRUTH {subject_id} res={target} valid={row['mean_depth_normal_valid_fraction']:.4f}", flush=True)

    result = {
        "schema": SCHEMA,
        "status": "MATERIALIZED_SURF_A1_RASTER_TRUTH__TRAINING_ONLY",
        "input_binding": str(Path(args.binding)),
        "stage01_sha256": expected_stage_sha,
        "teacher_batch_sha256": expected_batch,
        "resolution": target,
        "subject_order": list(EXPECTED_SUBJECTS),
        "subjects": subject_rows,
        "teacher_truth_in_inference_forward": False,
        "optimizer_steps_executed": 0,
        "training_authorized": False,
        "scientific_pass_claimed": False,
        "next_action": "RUN_SURF_A1_ZERO_STEP_RASTER_AND_NATIVE_R24_PARITY_PREFLIGHT",
    }
    batch_path = output_root / "SURF_A1_RASTER_TRUTH_BATCH_RECEIPT.json"
    batch_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binding", default="canonical/SURF_A1_FROZEN_H_INPUT_BINDING_V1.json")
    p.add_argument("--stage01-bundle", required=True)
    p.add_argument("--teacher-dir", required=True)
    p.add_argument("--teacher-batch-receipt", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--resolution", type=int, default=256)
    result = build(p.parse_args())
    print(json.dumps({"status": result["status"], "subjects": len(result["subjects"]), "resolution": result["resolution"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
