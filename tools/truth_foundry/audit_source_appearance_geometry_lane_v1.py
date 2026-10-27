#!/usr/bin/env python3
from __future__ import annotations

"""Separate packed-geometry drift from source-appearance replay drift.

This diagnostic projects both the packed Truth Foundry rest geometry and the
observation lane's independently materialized rest geometry through the exact
sealed camera bundle, then compares both masks with the replayed alpha mask.
It mints no qualification and is not itself a PASS/FAIL authority court.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from tools.corpus.iris_observation_admission_v1 import project_mask

SCHEMA = "RealSaS.SourceAppearanceGeometryLaneDiagnostic.v1"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _geometry(value: dict, label: str) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(value.get("vertices"), dtype=np.float64)
    faces = np.asarray(value.get("faces"), dtype=np.int64)
    if (
        vertices.ndim != 2
        or vertices.shape[1:] != (3,)
        or not len(vertices)
        or not np.isfinite(vertices).all()
        or faces.ndim != 2
        or faces.shape[1:] != (3,)
        or not len(faces)
        or np.any(faces < 0)
        or np.any(faces >= len(vertices))
    ):
        raise RuntimeError("SOURCE_APPEARANCE_GEOMETRY_LANE_INVALID_" + label)
    return vertices, faces


def _mask(path: Path, resolution: int) -> np.ndarray:
    raw = np.frombuffer(path.read_bytes(), np.uint8)
    if raw.size != resolution * resolution:
        raise RuntimeError("SOURCE_APPEARANCE_GEOMETRY_LANE_MASK_LAYOUT:" + path.name)
    value = raw.reshape(resolution, resolution) > 0
    if not value.any():
        raise RuntimeError("SOURCE_APPEARANCE_GEOMETRY_LANE_EMPTY_MASK:" + path.name)
    return value


def _gap(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape or not a.any() or not b.any():
        raise RuntimeError("SOURCE_APPEARANCE_GEOMETRY_LANE_GAP_DOMAIN")
    return float(max(distance_transform_edt(~b)[a].max(), distance_transform_edt(~a)[b].max()))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--subject-id", required=True)
    p.add_argument("--camera-bundle", required=True, type=Path)
    p.add_argument("--imported-geometry", required=True, type=Path)
    p.add_argument("--packed-geometry", required=True, type=Path)
    p.add_argument("--replay-dir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()

    cameras_value = _load(a.camera_bundle)
    imported_value = _load(a.imported_geometry)
    packed_value = _load(a.packed_geometry)
    replay_receipt_path = a.replay_dir / "SOURCE_APPEARANCE_POLICY_REPLAY_V1.json"
    replay_receipt = _load(replay_receipt_path)
    if cameras_value.get("schema") != "RealSaS.CameraProjectionBundle.v1":
        raise RuntimeError("SOURCE_APPEARANCE_GEOMETRY_LANE_CAMERA_SCHEMA")
    if packed_value.get("schema") != "RealSaS.PackedRestGeometryDiagnostic.v1":
        raise RuntimeError("SOURCE_APPEARANCE_GEOMETRY_LANE_PACKED_SCHEMA")
    if packed_value.get("subject_id") != a.subject_id or replay_receipt.get("subject_id") != a.subject_id:
        raise RuntimeError("SOURCE_APPEARANCE_GEOMETRY_LANE_SUBJECT_DRIFT")

    imported_vertices, imported_faces = _geometry(imported_value, "OBSERVATION_IMPORT")
    packed_vertices, packed_faces = _geometry(packed_value, "PACKED")
    cameras = tuple(cameras_value.get("cameras") or ())
    if len(cameras) != 8 or {int(row.get("view_index", -1)) for row in cameras} != set(range(8)):
        raise RuntimeError("SOURCE_APPEARANCE_GEOMETRY_LANE_VIEW_SET")

    rows = []
    max_replay_packed = 0.0
    max_packed_imported = 0.0
    max_replay_imported = 0.0
    for camera in sorted(cameras, key=lambda row: int(row["view_index"])):
        index = int(camera["view_index"])
        resolution = int(camera["resolution"])
        replay = _mask(a.replay_dir / f"V{index}.mask.bin", resolution)
        packed = project_mask(packed_vertices, packed_faces, camera)
        imported = project_mask(imported_vertices, imported_faces, camera)
        if not packed.any() or not imported.any():
            raise RuntimeError("SOURCE_APPEARANCE_GEOMETRY_LANE_PROJECTED_EMPTY")
        replay_packed = _gap(replay, packed)
        packed_imported = _gap(packed, imported)
        replay_imported = _gap(replay, imported)
        max_replay_packed = max(max_replay_packed, replay_packed)
        max_packed_imported = max(max_packed_imported, packed_imported)
        max_replay_imported = max(max_replay_imported, replay_imported)
        rows.append({
            "view_index": index,
            "replay_foreground_pixel_count": int(replay.sum()),
            "packed_geometry_reference_pixel_count": int(packed.sum()),
            "observation_import_geometry_reference_pixel_count": int(imported.sum()),
            "replay_to_packed_geometry_hausdorff_px": replay_packed,
            "packed_to_observation_import_geometry_hausdorff_px": packed_imported,
            "replay_to_observation_import_geometry_hausdorff_px": replay_imported,
        })

    result = {
        "schema": SCHEMA,
        "subject_id": a.subject_id,
        "camera_bundle_sha256": _sha(a.camera_bundle),
        "observation_import_geometry_sha256": _sha(a.imported_geometry),
        "packed_geometry_sha256": _sha(a.packed_geometry),
        "replay_receipt_sha256": _sha(replay_receipt_path),
        "observation_import_vertex_count": len(imported_vertices),
        "observation_import_triangle_count": len(imported_faces),
        "packed_vertex_count": len(packed_vertices),
        "packed_triangle_count": len(packed_faces),
        "vertex_count_equal": len(imported_vertices) == len(packed_vertices),
        "triangle_count_equal": len(imported_faces) == len(packed_faces),
        "maximum_replay_to_packed_geometry_hausdorff_px": max_replay_packed,
        "maximum_packed_to_observation_import_geometry_hausdorff_px": max_packed_imported,
        "maximum_replay_to_observation_import_geometry_hausdorff_px": max_replay_imported,
        "sealed_mask_gap_px_for_reference_only": 2.0,
        "diagnostic_replay_matches_packed_geometry_within_sealed_gap": max_replay_packed <= 2.0 + 1.0e-12,
        "diagnostic_packed_matches_observation_import_geometry_within_sealed_gap": max_packed_imported <= 2.0 + 1.0e-12,
        "authority": "DIAGNOSTIC_ONLY__SEPARATES_GEOMETRY_LANE_FROM_APPEARANCE_REPLAY",
        "qualification_minted": False,
        "student_target_allowed": False,
        "training_allowed": False,
        "product_pass_claimed": False,
        "views": rows,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "views"}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
