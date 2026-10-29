#!/usr/bin/env python3
"""Build permutation-invariant drawable-affinity truth for Truth139.

The source component teacher provides subject-local authored partitions.  This producer never emits
those integer ids as student labels.  It derives only pairwise same/different-component evidence on
exact mesh adjacency and on unambiguous consecutive hits in the measured K-hit ray prefix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from tools.corpus.build_truth139_dense_appearance_gbuffer_v1 import sha256
from tools.corpus.build_truth139_relations_teacher_v1 import exact_face_edge_adjacency

SCHEMA = "RealSaS.Truth139DrawableAffinityTeacher.v1"
COMPONENT_SCHEMA = "RealSaS.Truth139SourceComponentMasksTeacher.v1"
COMPONENT_STATUS = "PASS_ANONYMOUS_SOURCE_COMPONENT_MASK_CANDIDATE__INDEPENDENT_REBINDING_COURT_REQUIRED"
LAYER_SCHEMA = "RealSaS.Truth139GraphicsLayerPeelTeacher.v1"
LAYER_STATUS = "PASS_K_HIT_GEOMETRIC_LAYER_PEEL_CANDIDATE__CPU_PREFIX_COURT_REQUIRED__RGBA_COMPLETION_SEPARATE"
CONTRACT = "canonical/TRUTH139_DRAWABLE_AFFINITY_TEACHER_V1.json"


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


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


def _component_face_binding(component: dict, face_count: int) -> np.ndarray:
    views = sorted(list(component.get("views") or ()), key=lambda row: int(row.get("view_index", -1)))
    if len(views) != 8 or [int(row.get("view_index", -1)) for row in views] != list(range(8)):
        raise ValueError("TRUTH139_DRAWABLE_COMPONENT_VIEW_COUNT")
    binding = None
    for row in views:
        path = _bound_path({"path": row.get("path"), "sha256": row.get("sha256")}, "TRUTH139_DRAWABLE_COMPONENT_VIEW_DRIFT")
        with np.load(path, allow_pickle=False) as z:
            current = np.asarray(z["face_local_component"], dtype=np.int32).reshape(-1)
        if current.shape != (face_count,) or np.any(current < 0):
            raise ValueError("TRUTH139_DRAWABLE_COMPONENT_FACE_BINDING_LAYOUT")
        if binding is None:
            binding = current
        elif not np.array_equal(binding, current):
            raise ValueError("TRUTH139_DRAWABLE_COMPONENT_FACE_BINDING_VIEW_DRIFT")
    assert binding is not None
    return binding


def build(*, component_mask_manifest: Path, output_dir: Path) -> dict:
    component_path = component_mask_manifest.resolve()
    output_dir = output_dir.resolve()
    if not component_path.is_file():
        raise FileNotFoundError("TRUTH139_DRAWABLE_COMPONENT_MANIFEST_MISSING")
    component = json.loads(component_path.read_text())
    if component.get("schema") != COMPONENT_SCHEMA or component.get("status") != COMPONENT_STATUS:
        raise ValueError("TRUTH139_DRAWABLE_COMPONENT_PARENT_NOT_CANDIDATE")
    subject_id = str(component.get("subject_id") or "")
    if not subject_id:
        raise ValueError("TRUTH139_DRAWABLE_SUBJECT")

    parents = dict(component.get("parents") or {})
    layer_path = _bound_path(dict(parents.get("layer_peel_teacher_manifest") or {}), "TRUTH139_DRAWABLE_LAYER_DRIFT")
    geometry_path = _bound_path(dict(parents.get("geometry_npz") or {}), "TRUTH139_DRAWABLE_GEOMETRY_DRIFT")
    layer = json.loads(layer_path.read_text())
    if layer.get("schema") != LAYER_SCHEMA or layer.get("status") != LAYER_STATUS or str(layer.get("subject_id") or "") != subject_id:
        raise ValueError("TRUTH139_DRAWABLE_LAYER_BINDING")

    with np.load(geometry_path, allow_pickle=False) as z:
        faces = np.asarray(z["faces"], dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1:] != (3,) or len(faces) == 0:
        raise ValueError("TRUTH139_DRAWABLE_FACE_LAYOUT")
    face_component = _component_face_binding(component, len(faces))

    adjacent_pair, adjacent_edge = exact_face_edge_adjacency(faces)
    adjacent_same = (
        face_component[adjacent_pair[:, 0]] == face_component[adjacent_pair[:, 1]]
        if len(adjacent_pair) else np.empty(0, dtype=np.bool_)
    )

    layer_views = sorted(list(layer.get("views") or ()), key=lambda row: int(row.get("view_index", -1)))
    if len(layer_views) != 8 or [int(row.get("view_index", -1)) for row in layer_views] != list(range(8)):
        raise ValueError("TRUTH139_DRAWABLE_LAYER_VIEW_COUNT")

    ray_view, ray_pixel, ray_slot = [], [], []
    ray_front, ray_behind, ray_gap, ray_same = [], [], [], []
    tie_abstain = 0
    for view_index, row in enumerate(layer_views):
        peel_path = _bound_path({"path": row.get("path"), "sha256": row.get("sha256")}, "TRUTH139_DRAWABLE_PEEL_VIEW_DRIFT")
        with np.load(peel_path, allow_pickle=False) as z:
            pixels = np.asarray(z["pixel_linear_index"], dtype=np.int32).reshape(-1)
            valid = np.asarray(z["hit_valid"], dtype=np.bool_)
            tri = np.asarray(z["hit_triangle_id"], dtype=np.int32)
            depth = np.asarray(z["hit_depth"], dtype=np.float32)
            tie = np.asarray(z["adjacent_depth_tie"], dtype=np.bool_)
        if valid.shape != tri.shape or depth.shape != tri.shape or valid.ndim != 2 or valid.shape[0] != len(pixels):
            raise ValueError("TRUTH139_DRAWABLE_PEEL_LAYOUT")
        k = valid.shape[1]
        if tie.shape[0] != len(pixels) or tie.shape[1] < max(k - 1, 1):
            raise ValueError("TRUTH139_DRAWABLE_TIE_LAYOUT")
        for slot in range(k - 1):
            pair_valid = valid[:, slot] & valid[:, slot + 1]
            tied = pair_valid & tie[:, slot]
            tie_abstain += int(np.count_nonzero(tied))
            use = np.flatnonzero(pair_valid & ~tied)
            if not len(use):
                continue
            front = tri[use, slot]
            behind = tri[use, slot + 1]
            if np.any(front < 0) or np.any(behind < 0) or np.any(front >= len(faces)) or np.any(behind >= len(faces)):
                raise ValueError("TRUTH139_DRAWABLE_RAY_FACE_RANGE")
            gap = depth[use, slot + 1] - depth[use, slot]
            if not np.isfinite(gap).all() or np.any(gap <= 0):
                raise ValueError("TRUTH139_DRAWABLE_RAY_DEPTH_GAP")
            ray_view.extend([view_index] * len(use))
            ray_pixel.extend(pixels[use].tolist())
            ray_slot.extend([slot] * len(use))
            ray_front.extend(front.tolist())
            ray_behind.extend(behind.tolist())
            ray_gap.extend(gap.tolist())
            ray_same.extend((face_component[front] == face_component[behind]).tolist())

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "DRAWABLE_AFFINITY_TEACHER.npz"
    _write_npz(
        out,
        adjacent_face_pair=adjacent_pair.astype(np.int32),
        adjacent_shared_edge_vertex=adjacent_edge.astype(np.int32),
        adjacent_same_authored_component=np.asarray(adjacent_same, dtype=np.bool_),
        adjacent_authored_component_boundary=np.asarray(~adjacent_same, dtype=np.bool_),
        ray_view_index=np.asarray(ray_view, dtype=np.int16),
        ray_pixel_linear_index=np.asarray(ray_pixel, dtype=np.int32),
        ray_front_slot=np.asarray(ray_slot, dtype=np.int16),
        ray_front_face=np.asarray(ray_front, dtype=np.int32),
        ray_behind_face=np.asarray(ray_behind, dtype=np.int32),
        ray_positive_depth_gap=np.asarray(ray_gap, dtype=np.float32),
        ray_same_authored_component=np.asarray(ray_same, dtype=np.bool_),
    )

    payload = {
        "schema": SCHEMA,
        "date": "2026-10-29",
        "status": "PASS_DRAWABLE_AFFINITY_CANDIDATE__INDEPENDENT_REBINDING_COURT_REQUIRED",
        "subject_id": subject_id,
        "contract": CONTRACT,
        "parents": {
            "component_mask_manifest": {"path": str(component_path), "sha256": sha256(component_path)},
            "component_mask_teacher_identity_sha256": str(component.get("component_mask_teacher_identity_sha256") or ""),
            "layer_peel_teacher_manifest": {"path": str(layer_path), "sha256": sha256(layer_path)},
            "layer_peel_teacher_identity_sha256": str(layer.get("layer_peel_teacher_identity_sha256") or ""),
            "geometry_npz": {"path": str(geometry_path), "sha256": sha256(geometry_path)},
        },
        "affinity_npz": {"path": str(out), "sha256": sha256(out)},
        "counts": {
            "adjacent_face_pair": int(len(adjacent_pair)),
            "adjacent_same_component": int(np.count_nonzero(adjacent_same)),
            "adjacent_component_boundary": int(np.count_nonzero(~adjacent_same)),
            "ray_order_pair": int(len(ray_view)),
            "ray_order_same_component": int(np.count_nonzero(ray_same)),
            "ray_order_different_component": int(len(ray_same) - np.count_nonzero(ray_same)),
            "depth_tie_abstention": int(tie_abstain),
        },
        "policy": {
            "source_component_integer_emitted_as_student_label": False,
            "pairwise_targets_are_permutation_invariant": True,
            "source_authored_partition_is_optimal_runtime_drawable_claimed": False,
            "depth_ties_abstain": True,
            "unmeasured_or_overflow_pair_is_negative": False,
            "semantic_part_taxonomy_used": False,
            "training_performed": False,
        },
        "qualification_minted": False,
    }
    payload["drawable_affinity_teacher_identity_sha256"] = _digest(payload)
    manifest_out = output_dir / "DRAWABLE_AFFINITY_MANIFEST.json"
    manifest_out.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-mask-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = build(component_mask_manifest=args.component_mask_manifest, output_dir=args.output_dir)
    print(json.dumps({
        "status": result["status"],
        "subject_id": result["subject_id"],
        "counts": result["counts"],
        "drawable_affinity_teacher_identity_sha256": result["drawable_affinity_teacher_identity_sha256"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
