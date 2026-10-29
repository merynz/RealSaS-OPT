#!/usr/bin/env python3
"""Validate subject-local source component masks against geometry and peel parents."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from tools.corpus.build_truth139_dense_appearance_gbuffer_v1 import sha256

SCHEMA = "RealSaS.Truth139SourceComponentMasksCourt.v1"
PARENT_SCHEMA = "RealSaS.Truth139SourceComponentMasksTeacher.v1"
PARENT_STATUS = "PASS_ANONYMOUS_SOURCE_COMPONENT_MASK_CANDIDATE__INDEPENDENT_REBINDING_COURT_REQUIRED"
LAYER_SCHEMA = "RealSaS.Truth139GraphicsLayerPeelTeacher.v1"


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _bound_path(ref: Mapping[str, object], code: str) -> Path:
    path = Path(str(ref.get("path") or "")).resolve()
    expected = str(ref.get("sha256") or "")
    if not path.is_file() or len(expected) != 64 or sha256(path) != expected:
        raise ValueError(code)
    return path


def _face_owner(geometry_path: Path, face_count: int) -> np.ndarray:
    with np.load(geometry_path, allow_pickle=False) as z:
        key = next((name for name in (
            "face_owner_privileged", "face_owner", "primitive_owner_index", "source_face_owner"
        ) if name in z.files), None)
        if key is None:
            raise ValueError("TRUTH139_COMPONENT_MASK_COURT_FACE_OWNER_MISSING")
        owner = np.asarray(z[key], dtype=np.int64).reshape(-1)
    if len(owner) != face_count or np.any(owner < 0):
        raise ValueError("TRUTH139_COMPONENT_MASK_COURT_FACE_OWNER_LAYOUT")
    return owner


def _segments(ids: np.ndarray, offsets: np.ndarray, pixels: np.ndarray) -> dict[int, np.ndarray]:
    ids = np.asarray(ids, dtype=np.int32).reshape(-1)
    offsets = np.asarray(offsets, dtype=np.int64).reshape(-1)
    pixels = np.asarray(pixels, dtype=np.int32).reshape(-1)
    if len(offsets) != len(ids) + 1 or len(offsets) == 0 or offsets[0] != 0 or offsets[-1] != len(pixels):
        raise ValueError("TRUTH139_COMPONENT_MASK_COURT_SEGMENT_LAYOUT")
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("TRUTH139_COMPONENT_MASK_COURT_SEGMENT_OFFSETS")
    return {int(component): pixels[int(offsets[i]): int(offsets[i + 1])] for i, component in enumerate(ids)}


def validate(manifest: dict) -> dict:
    if manifest.get("schema") != PARENT_SCHEMA or manifest.get("status") != PARENT_STATUS:
        raise ValueError("TRUTH139_COMPONENT_MASK_COURT_PARENT_NOT_CANDIDATE")
    subject_id = str(manifest.get("subject_id") or "")
    if not subject_id:
        raise ValueError("TRUTH139_COMPONENT_MASK_COURT_SUBJECT")

    parents = dict(manifest.get("parents") or {})
    layer_path = _bound_path(dict(parents.get("layer_peel_teacher_manifest") or {}), "TRUTH139_COMPONENT_MASK_COURT_LAYER_DRIFT")
    geometry_path = _bound_path(dict(parents.get("geometry_npz") or {}), "TRUTH139_COMPONENT_MASK_COURT_GEOMETRY_DRIFT")
    layer = json.loads(layer_path.read_text())
    if layer.get("schema") != LAYER_SCHEMA or str(layer.get("subject_id") or "") != subject_id:
        raise ValueError("TRUTH139_COMPONENT_MASK_COURT_LAYER_BINDING")
    if str(layer.get("layer_peel_teacher_identity_sha256") or "") != str(parents.get("layer_peel_teacher_identity_sha256") or ""):
        raise ValueError("TRUTH139_COMPONENT_MASK_COURT_LAYER_IDENTITY")

    with np.load(geometry_path, allow_pickle=False) as z:
        faces = np.asarray(z["faces"], dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1:] != (3,) or len(faces) == 0:
        raise ValueError("TRUTH139_COMPONENT_MASK_COURT_FACE_LAYOUT")
    owner = _face_owner(geometry_path, len(faces))
    unique_owner = np.unique(owner)
    owner_to_local = {int(value): i for i, value in enumerate(unique_owner.tolist())}
    face_component = np.asarray([owner_to_local[int(value)] for value in owner], dtype=np.int32)
    component_count = len(unique_owner)
    if int(manifest.get("component_count", -1)) != component_count:
        raise ValueError("TRUTH139_COMPONENT_MASK_COURT_COMPONENT_COUNT")

    layer_views = sorted(list(layer.get("views") or ()), key=lambda row: int(row.get("view_index", -1)))
    mask_views = sorted(list(manifest.get("views") or ()), key=lambda row: int(row.get("view_index", -1)))
    if len(layer_views) != 8 or len(mask_views) != 8:
        raise ValueError("TRUTH139_COMPONENT_MASK_COURT_VIEW_COUNT")

    total_visible = total_any = total_hidden = 0
    view_results = []
    for view_index in range(8):
        peel_path = _bound_path(
            {"path": layer_views[view_index].get("path"), "sha256": layer_views[view_index].get("sha256")},
            "TRUTH139_COMPONENT_MASK_COURT_PEEL_VIEW_DRIFT",
        )
        mask_path = _bound_path(
            {"path": mask_views[view_index].get("path"), "sha256": mask_views[view_index].get("sha256")},
            "TRUTH139_COMPONENT_MASK_COURT_MASK_VIEW_DRIFT",
        )
        with np.load(peel_path, allow_pickle=False) as z:
            pixels = np.asarray(z["pixel_linear_index"], dtype=np.int32).reshape(-1)
            valid = np.asarray(z["hit_valid"], dtype=np.bool_)
            tri = np.asarray(z["hit_triangle_id"], dtype=np.int64)
            complete = np.asarray(z["stack_complete"], dtype=np.bool_).reshape(-1)
        if valid.shape != tri.shape or valid.ndim != 2 or valid.shape[0] != len(pixels) or complete.shape != (len(pixels),):
            raise ValueError("TRUTH139_COMPONENT_MASK_COURT_PEEL_LAYOUT")
        if np.any(tri[valid] < 0) or np.any(tri[valid] >= len(faces)):
            raise ValueError("TRUTH139_COMPONENT_MASK_COURT_TRIANGLE_RANGE")
        hit_component = np.full(tri.shape, -1, dtype=np.int32)
        hit_component[valid] = face_component[tri[valid]]
        visible_component = np.full(len(pixels), -1, dtype=np.int32)
        front = valid[:, 0]
        visible_component[front] = hit_component[front, 0]

        with np.load(mask_path, allow_pickle=False) as z:
            stored_owner = np.asarray(z["source_owner_value_by_local_component"], dtype=np.int64).reshape(-1)
            stored_face_component = np.asarray(z["face_local_component"], dtype=np.int32).reshape(-1)
            stored_hit_component = np.asarray(z["hit_local_component"], dtype=np.int32)
            stored_complete = np.asarray(z["pixel_stack_complete"], dtype=np.bool_).reshape(-1)
            visible_segments = _segments(z["visible_component_id"], z["visible_offsets"], z["visible_pixel_linear_index"])
            any_segments = _segments(z["any_depth_component_id"], z["any_depth_offsets"], z["any_depth_pixel_linear_index"])
            hidden_segments = _segments(z["hidden_only_component_id"], z["hidden_only_offsets"], z["hidden_only_pixel_linear_index"])
            occupancy = np.asarray(z["component_depth_slot_occupancy"], dtype=np.int64)
        if not np.array_equal(stored_owner, unique_owner):
            raise ValueError("TRUTH139_COMPONENT_MASK_COURT_OWNER_REMAP_DRIFT")
        if not np.array_equal(stored_face_component, face_component):
            raise ValueError("TRUTH139_COMPONENT_MASK_COURT_FACE_COMPONENT_DRIFT")
        if not np.array_equal(stored_hit_component, hit_component):
            raise ValueError("TRUTH139_COMPONENT_MASK_COURT_HIT_COMPONENT_DRIFT")
        if not np.array_equal(stored_complete, complete):
            raise ValueError("TRUTH139_COMPONENT_MASK_COURT_COMPLETENESS_DRIFT")
        if occupancy.shape != (component_count, valid.shape[1]):
            raise ValueError("TRUTH139_COMPONENT_MASK_COURT_OCCUPANCY_LAYOUT")

        visible_count = any_count = hidden_count = 0
        for component in range(component_count):
            visible_expected = pixels[visible_component == component]
            any_expected = pixels[np.any(valid & (hit_component == component), axis=1)]
            hidden_expected = np.setdiff1d(any_expected, visible_expected, assume_unique=True)
            visible_stored = visible_segments.get(component, np.empty(0, dtype=np.int32))
            any_stored = any_segments.get(component, np.empty(0, dtype=np.int32))
            hidden_stored = hidden_segments.get(component, np.empty(0, dtype=np.int32))
            if not np.array_equal(visible_stored, visible_expected):
                raise ValueError(f"TRUTH139_COMPONENT_MASK_COURT_VISIBLE_DRIFT:V{view_index}:C{component}")
            if not np.array_equal(any_stored, any_expected):
                raise ValueError(f"TRUTH139_COMPONENT_MASK_COURT_ANY_DEPTH_DRIFT:V{view_index}:C{component}")
            if not np.array_equal(hidden_stored, hidden_expected):
                raise ValueError(f"TRUTH139_COMPONENT_MASK_COURT_HIDDEN_DRIFT:V{view_index}:C{component}")
            for slot in range(valid.shape[1]):
                expected = int(np.count_nonzero(valid[:, slot] & (hit_component[:, slot] == component)))
                if int(occupancy[component, slot]) != expected:
                    raise ValueError(f"TRUTH139_COMPONENT_MASK_COURT_OCCUPANCY_DRIFT:V{view_index}:C{component}:S{slot}")
            visible_count += len(visible_expected)
            any_count += len(any_expected)
            hidden_count += len(hidden_expected)

        if int(mask_views[view_index].get("visible_pixel_count", -1)) != visible_count:
            raise ValueError("TRUTH139_COMPONENT_MASK_COURT_VISIBLE_COUNT_METADATA")
        if int(mask_views[view_index].get("any_depth_pixel_membership_count", -1)) != any_count:
            raise ValueError("TRUTH139_COMPONENT_MASK_COURT_ANY_COUNT_METADATA")
        if int(mask_views[view_index].get("hidden_only_pixel_membership_count", -1)) != hidden_count:
            raise ValueError("TRUTH139_COMPONENT_MASK_COURT_HIDDEN_COUNT_METADATA")
        total_visible += visible_count
        total_any += any_count
        total_hidden += hidden_count
        view_results.append({
            "view_index": view_index,
            "visible_pixel_membership_count": int(visible_count),
            "any_depth_pixel_membership_count": int(any_count),
            "hidden_only_pixel_membership_count": int(hidden_count),
            "overflow_unknown_pixel_count": int(np.count_nonzero(~complete)),
        })

    payload = {
        "schema": SCHEMA,
        "date": "2026-10-29",
        "status": "PASS_SOURCE_COMPONENT_MASKS_REBIND_EXACT_FACE_OWNER_AND_PEEL_TRUTH",
        "subject_id": subject_id,
        "component_mask_teacher_identity_sha256": str(manifest.get("component_mask_teacher_identity_sha256") or ""),
        "component_count": int(component_count),
        "view_courts": view_results,
        "counts": {
            "visible_pixel_memberships": int(total_visible),
            "any_depth_pixel_memberships": int(total_any),
            "hidden_only_pixel_memberships": int(total_hidden),
        },
        "policy": {
            "source_owner_values_used_only_for_subject_local_rebinding": True,
            "semantic_names_used": False,
            "cross_subject_component_identity_claimed": False,
            "overflow_unknown_preserved": True,
            "training_performed": False,
        },
        "qualification_minted": False,
    }
    payload["court_identity_sha256"] = _digest(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-component-masks-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = validate(json.loads(args.source_component_masks_manifest.resolve().read_text()))
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = output.with_suffix(output.suffix + ".partial")
    staged.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    staged.replace(output)
    print(json.dumps({
        "status": result["status"],
        "subject_id": result["subject_id"],
        "component_count": result["component_count"],
        "court_identity_sha256": result["court_identity_sha256"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
