from __future__ import annotations

"""Runtime I/O helpers shared by SURF-A1 preflight, trainer and courts."""

import json
from pathlib import Path
import subprocess
import zipfile
from typing import Any

import numpy as np
import torch

from models.iris.vnext.cohort_v1 import SUBJECT_IDS
from tools.iris_vnext.materialize_fit8_direct_surface_v1 import load_json, sha256_file


def require_git_identity_v1(root: Path, *, repository: str, commit: str, code: str) -> None:
    root = root.resolve()
    if not (root / ".git").exists():
        raise RuntimeError(code + ":GIT_MISSING")
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    remote = subprocess.check_output(["git", "-C", str(root), "config", "--get", "remote.origin.url"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip()
    if head != commit:
        raise RuntimeError(f"{code}:COMMIT_DRIFT:{head}:{commit}")
    normalized = remote.removesuffix("/").removesuffix(".git")
    expected = repository.removesuffix("/").removesuffix(".git")
    if normalized != expected:
        raise RuntimeError(f"{code}:REMOTE_DRIFT:{remote}")
    if dirty:
        raise RuntimeError(code + ":SOURCE_DIRTY")


def load_surf_a1_binding_v1(path: Path) -> dict[str, Any]:
    value = load_json(path.resolve())
    if value.get("schema") != "RealSaS.SurfA1FrozenHInputBinding.v1":
        raise ValueError("SURF_A1_INPUT_BINDING_SCHEMA_INVALID")
    rows = list(value.get("authored_teachers", {}).get("subjects") or ())
    if tuple(row.get("subject_id") for row in rows) != SUBJECT_IDS:
        raise ValueError("SURF_A1_INPUT_BINDING_SUBJECT_ORDER_DRIFT")
    return value


def load_camera_bundles_v1(stage01_bundle: Path, binding: dict[str, Any]) -> dict[str, dict[str, Any]]:
    stage01_bundle = stage01_bundle.resolve()
    if sha256_file(stage01_bundle) != str(binding["stage01"]["sha256"]):
        raise ValueError("SURF_A1_STAGE01_SHA_DRIFT")
    legacy = {str(row["subject_id"]): str(row["legacy_subject_id"]) for row in binding["authored_teachers"]["subjects"]}
    out: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(stage01_bundle, "r") as archive:
        names = archive.namelist()
        for subject_id in SUBJECT_IDS:
            suffix = f"observations/{legacy[subject_id]}/camera_bundle.json"
            matches = [name for name in names if name.endswith(suffix)]
            if len(matches) != 1:
                raise ValueError(f"SURF_A1_CAMERA_BUNDLE_CARDINALITY:{subject_id}:{len(matches)}")
            value = json.loads(archive.read(matches[0]))
            if len(value.get("cameras") or ()) != 8:
                raise ValueError("SURF_A1_CAMERA_VIEW_COUNT_DRIFT:" + subject_id)
            if value.get("per_view_pose_change") or value.get("per_view_recenter"):
                raise ValueError("SURF_A1_CAMERA_MUTATION_DRIFT:" + subject_id)
            out[subject_id] = value
    return out


def camera_tensors_v1(bundle: dict[str, Any], view_index: int, device: torch.device) -> dict[str, torch.Tensor | float]:
    cameras = list(bundle.get("cameras") or ())
    camera = next((row for row in cameras if int(row["view_index"]) == int(view_index)), None)
    if camera is None:
        raise ValueError("SURF_A1_CAMERA_VIEW_MISSING")
    f = lambda value: torch.as_tensor(value, device=device, dtype=torch.float32)
    return {
        "common_center_world": f(bundle["common_center_xyz"]),
        "common_half_extent": float(bundle["common_half_extent"]),
        "origin_world": f(camera["origin"]),
        "right_world": f(camera["right"]),
        "screen_up_world": f(camera["screen_up"]),
        "forward_world": f(camera["forward"]),
        "half_extent": float(camera["half_extent"]),
        "clip_start": float(camera.get("clip_start", 0.0)),
        "clip_end": float(camera.get("clip_end", 1.0e30)),
    }


def load_raster_truth_v1(root: Path, subject_id: str, device: torch.device) -> dict[str, torch.Tensor]:
    path = root.resolve() / f"{subject_id}.surf_a1_raster_truth.npz"
    receipt_path = root.resolve() / f"{subject_id}.surf_a1_raster_truth.json"
    if not path.is_file() or not receipt_path.is_file():
        raise FileNotFoundError("SURF_A1_RASTER_TRUTH_MISSING:" + subject_id)
    receipt = load_json(receipt_path)
    if receipt.get("schema") != "RealSaS.SurfA1RasterTruthSubject.v1" or receipt.get("subject_id") != subject_id:
        raise ValueError("SURF_A1_RASTER_TRUTH_RECEIPT_DRIFT:" + subject_id)
    if sha256_file(path) != receipt.get("payload_sha256"):
        raise ValueError("SURF_A1_RASTER_TRUTH_PAYLOAD_SHA_DRIFT:" + subject_id)
    with np.load(path, allow_pickle=False) as data:
        value = {
            "mask": torch.from_numpy(np.asarray(data["mask"], dtype=np.uint8)).to(device=device),
            "teacher_center_coverage": torch.from_numpy(np.asarray(data["teacher_center_coverage"], dtype=np.uint8)).to(device=device),
            "depth_normalized": torch.from_numpy(np.asarray(data["depth_normalized"], dtype=np.float32)).to(device=device),
            "normal_world": torch.from_numpy(np.asarray(data["normal_world"], dtype=np.float32)).to(device=device),
            "depth_normal_valid": torch.from_numpy(np.asarray(data["depth_normal_valid"], dtype=np.uint8)).to(device=device),
        }
    if value["mask"].shape != (8, 256, 256) or value["normal_world"].shape != (8, 256, 256, 3):
        raise ValueError("SURF_A1_RASTER_TRUTH_SHAPE_DRIFT:" + subject_id)
    return value


def load_teacher_rest_iris_v1(teacher_path: Path, camera_bundle: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    with np.load(teacher_path.resolve(), allow_pickle=False) as data:
        vertices_world = np.asarray(data["rest_vertices_source_world"], dtype=np.float64)
        faces = np.asarray(data["faces"], dtype=np.int64)
    center = np.asarray(camera_bundle["common_center_xyz"], dtype=np.float64).reshape(3)
    half = float(camera_bundle["common_half_extent"])
    if not np.isfinite(half) or half <= 0.0:
        raise ValueError("SURF_A1_TEACHER_NORMALIZATION_INVALID")
    vertices_iris = (vertices_world - center[None]) / half
    return vertices_iris, faces
