#!/usr/bin/env python3
"""Build subject-local anonymous component masks from qualified K-hit layer peel truth.

The source scene's face-owner binding is exact authoring evidence, but owner ids are *not* semantic
classes and are never stable across subjects. This producer maps each measured peel hit to its
anonymous source component and stores sparse visible / any-depth / hidden-only masks per view.

Any-depth support is explicitly limited to the measured K-hit prefix. Pixels whose K stack is full
remain overflow-unknown; the producer never calls an unseen K+1 component absent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from tools.corpus.build_truth139_dense_appearance_gbuffer_v1 import sha256

SCHEMA = "RealSaS.Truth139SourceComponentMasksTeacher.v1"
LAYER_SCHEMA = "RealSaS.Truth139GraphicsLayerPeelTeacher.v1"
LAYER_STATUS = "PASS_K_HIT_GEOMETRIC_LAYER_PEEL_CANDIDATE__CPU_PREFIX_COURT_REQUIRED__RGBA_COMPLETION_SEPARATE"
CONTRACT = "canonical/TRUTH139_GRAPHICS_LAYER_DECOMPOSITION_TEACHER_V1.json"


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


def _write_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(path.suffix + ".partial.npz")
    np.savez_compressed(staged, **arrays)
    staged.replace(path)


def _load_face_owner(geometry_path: Path, face_count: int) -> np.ndarray:
    with np.load(geometry_path, allow_pickle=False) as z:
        candidates = [
            "face_owner_privileged",
            "face_owner",
            "primitive_owner_index",
            "source_face_owner",
        ]
        key = next((name for name in candidates if name in z.files), None)
        if key is None:
            raise ValueError("TRUTH139_COMPONENT_MASK_FACE_OWNER_MISSING")
        owner = np.asarray(z[key], dtype=np.int64).reshape(-1)
    if len(owner) != face_count:
        raise ValueError("TRUTH139_COMPONENT_MASK_FACE_OWNER_LAYOUT")
    if np.any(owner < 0):
        raise ValueError("TRUTH139_COMPONENT_MASK_FACE_OWNER_NEGATIVE")
    return owner


def build(*, layer_peel_teacher_manifest: Path, output_dir: Path) -> dict:
    manifest_path = layer_peel_teacher_manifest.resolve()
    output_dir = output_dir.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError("TRUTH139_COMPONENT_MASK_LAYER_MANIFEST_MISSING")
    layer = json.loads(manifest_path.read_text())
    if layer.get("schema") != LAYER_SCHEMA or layer.get("status") != LAYER_STATUS:
        raise ValueError("TRUTH139_COMPONENT_MASK_LAYER_PARENT_NOT_CANDIDATE")
    subject_id = str(layer.get("subject_id") or "")
    if not subject_id:
        raise ValueError("TRUTH139_COMPONENT_MASK_SUBJECT")

    parents = dict(layer.get("parents") or {})
    geometry_path = _bound_path(dict(parents.get("geometry_npz") or {}), "TRUTH139_COMPONENT_MASK_GEOMETRY_DRIFT")
    with np.load(geometry_path, allow_pickle=False) as z:
        if "faces" not in z.files:
            raise ValueError("TRUTH139_COMPONENT_MASK_FACES_MISSING")
        faces = np.asarray(z["faces"], dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1:] != (3,) or len(faces) == 0:
        raise ValueError("TRUTH139_COMPONENT_MASK_FACE_LAYOUT")
    owner = _load_face_owner(geometry_path, len(faces))

    unique_owner = np.unique(owner)
    owner_to_local = {int(value): index for index, value in enumerate(unique_owner.tolist())}
    face_component = np.asarray([owner_to_local[int(value)] for value in owner], dtype=np.int32)
    component_count = len(unique_owner)

    views = sorted(list(layer.get("views") or ()), key=lambda row: int(row.get("view_index", -1)))
    if len(views) != 8 or [int(row.get("view_index", -1)) for row in views] != list(range(8)):
        raise ValueError("TRUTH139_COMPONENT_MASK_VIEW_COUNT")

    output_dir.mkdir(parents=True, exist_ok=True)
    view_rows = []
    aggregate_visible = np.zeros(component_count, dtype=np.int64)
    aggregate_any = np.zeros(component_count, dtype=np.int64)
    aggregate_hidden = np.zeros(component_count, dtype=np.int64)

    for view_index, meta in enumerate(views):
        peel_path = _bound_path({"path": meta.get("path"), "sha256": meta.get("sha256")}, "TRUTH139_COMPONENT_MASK_PEEL_VIEW_DRIFT")
        with np.load(peel_path, allow_pickle=False) as z:
            pixels = np.asarray(z["pixel_linear_index"], dtype=np.int32).reshape(-1)
            valid = np.asarray(z["hit_valid"], dtype=np.bool_)
            tri = np.asarray(z["hit_triangle_id"], dtype=np.int64)
            complete = np.asarray(z["stack_complete"], dtype=np.bool_).reshape(-1)
        if valid.shape != tri.shape or valid.ndim != 2 or valid.shape[0] != len(pixels) or complete.shape != (len(pixels),):
            raise ValueError("TRUTH139_COMPONENT_MASK_PEEL_LAYOUT")
        if np.any(tri[valid] < 0) or np.any(tri[valid] >= len(faces)):
            raise ValueError("TRUTH139_COMPONENT_MASK_TRIANGLE_RANGE")

        hit_component = np.full(tri.shape, -1, dtype=np.int32)
        hit_component[valid] = face_component[tri[valid]]

        visible_component = np.full(len(pixels), -1, dtype=np.int32)
        has_front = valid[:, 0]
        visible_component[has_front] = hit_component[has_front, 0]

        visible_component_ids = []
        visible_offsets = [0]
        visible_pixels_parts = []
        any_component_ids = []
        any_offsets = [0]
        any_pixels_parts = []
        hidden_component_ids = []
        hidden_offsets = [0]
        hidden_pixels_parts = []
        occupancy = np.zeros((component_count, valid.shape[1]), dtype=np.int64)

        for component in range(component_count):
            visible_mask = visible_component == component
            any_mask = np.any(valid & (hit_component == component), axis=1)
            hidden_mask = any_mask & ~visible_mask
            for slot in range(valid.shape[1]):
                occupancy[component, slot] = int(np.count_nonzero(valid[:, slot] & (hit_component[:, slot] == component)))

            v = pixels[visible_mask].astype(np.int32)
            a = pixels[any_mask].astype(np.int32)
            h = pixels[hidden_mask].astype(np.int32)
            if len(v):
                visible_component_ids.append(component)
                visible_pixels_parts.append(v)
                visible_offsets.append(visible_offsets[-1] + len(v))
            if len(a):
                any_component_ids.append(component)
                any_pixels_parts.append(a)
                any_offsets.append(any_offsets[-1] + len(a))
            if len(h):
                hidden_component_ids.append(component)
                hidden_pixels_parts.append(h)
                hidden_offsets.append(hidden_offsets[-1] + len(h))

            aggregate_visible[component] += len(v)
            aggregate_any[component] += len(a)
            aggregate_hidden[component] += len(h)

        visible_pixels = np.concatenate(visible_pixels_parts) if visible_pixels_parts else np.empty(0, dtype=np.int32)
        any_pixels = np.concatenate(any_pixels_parts) if any_pixels_parts else np.empty(0, dtype=np.int32)
        hidden_pixels = np.concatenate(hidden_pixels_parts) if hidden_pixels_parts else np.empty(0, dtype=np.int32)

        out = output_dir / f"V{view_index}_SOURCE_COMPONENT_MASKS.npz"
        _write_npz(
            out,
            component_count=np.asarray([component_count], dtype=np.int32),
            source_owner_value_by_local_component=unique_owner.astype(np.int64),
            face_local_component=face_component.astype(np.int32),
            hit_local_component=hit_component,
            visible_component_id=np.asarray(visible_component_ids, dtype=np.int32),
            visible_offsets=np.asarray(visible_offsets, dtype=np.int64),
            visible_pixel_linear_index=visible_pixels,
            any_depth_component_id=np.asarray(any_component_ids, dtype=np.int32),
            any_depth_offsets=np.asarray(any_offsets, dtype=np.int64),
            any_depth_pixel_linear_index=any_pixels,
            hidden_only_component_id=np.asarray(hidden_component_ids, dtype=np.int32),
            hidden_only_offsets=np.asarray(hidden_offsets, dtype=np.int64),
            hidden_only_pixel_linear_index=hidden_pixels,
            component_depth_slot_occupancy=occupancy,
            pixel_stack_complete=complete,
        )
        view_rows.append(
            {
                "view_index": view_index,
                "path": str(out),
                "sha256": sha256(out),
                "visible_component_count": int(len(visible_component_ids)),
                "any_depth_component_count": int(len(any_component_ids)),
                "hidden_component_count": int(len(hidden_component_ids)),
                "visible_pixel_count": int(len(visible_pixels)),
                "any_depth_pixel_membership_count": int(len(any_pixels)),
                "hidden_only_pixel_membership_count": int(len(hidden_pixels)),
                "overflow_unknown_pixel_count": int(np.count_nonzero(~complete)),
            }
        )

    payload = {
        "schema": SCHEMA,
        "date": "2026-10-29",
        "status": "PASS_ANONYMOUS_SOURCE_COMPONENT_MASK_CANDIDATE__INDEPENDENT_REBINDING_COURT_REQUIRED",
        "subject_id": subject_id,
        "contract": CONTRACT,
        "parents": {
            "layer_peel_teacher_manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
            "layer_peel_teacher_identity_sha256": str(layer.get("layer_peel_teacher_identity_sha256") or ""),
            "geometry_npz": {"path": str(geometry_path), "sha256": sha256(geometry_path)},
        },
        "component_count": int(component_count),
        "component_identity_policy": "SUBJECT_LOCAL_PERMUTATION_INVARIANT__SOURCE_OWNER_VALUES_TEACHER_ONLY",
        "views": view_rows,
        "aggregate": {
            "visible_pixel_count_by_component": aggregate_visible.astype(int).tolist(),
            "any_depth_pixel_membership_count_by_component": aggregate_any.astype(int).tolist(),
            "hidden_only_pixel_membership_count_by_component": aggregate_hidden.astype(int).tolist(),
        },
        "policy": {
            "source_owner_is_semantic_class": False,
            "source_owner_is_cross_subject_identity": False,
            "source_owner_binding_is_subject_local_authored_partition_truth": True,
            "any_depth_mask_is_complete_when_k_stack_overflow_unknown": False,
            "absence_beyond_measured_k_prefix_is_negative_label": False,
            "training_performed": False,
        },
        "qualification_minted": False,
    }
    payload["component_mask_teacher_identity_sha256"] = _digest(payload)
    manifest_out = output_dir / "SOURCE_COMPONENT_MASKS_MANIFEST.json"
    manifest_out.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer-peel-teacher-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = build(layer_peel_teacher_manifest=args.layer_peel_teacher_manifest, output_dir=args.output_dir)
    print(json.dumps({
        "status": result["status"],
        "subject_id": result["subject_id"],
        "component_count": result["component_count"],
        "component_mask_teacher_identity_sha256": result["component_mask_teacher_identity_sha256"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
