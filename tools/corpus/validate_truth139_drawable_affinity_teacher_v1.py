#!/usr/bin/env python3
"""Validate Truth139 drawable-affinity truth by exact rebinding to component and peel parents."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from tools.corpus.build_truth139_dense_appearance_gbuffer_v1 import sha256
from tools.corpus.build_truth139_relations_teacher_v1 import exact_face_edge_adjacency

SCHEMA = "RealSaS.Truth139DrawableAffinityTeacherCourt.v1"
PARENT_SCHEMA = "RealSaS.Truth139DrawableAffinityTeacher.v1"
PARENT_STATUS = "PASS_DRAWABLE_AFFINITY_CANDIDATE__INDEPENDENT_REBINDING_COURT_REQUIRED"
COMPONENT_SCHEMA = "RealSaS.Truth139SourceComponentMasksTeacher.v1"
LAYER_SCHEMA = "RealSaS.Truth139GraphicsLayerPeelTeacher.v1"


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _bound_path(ref: Mapping[str, object], code: str) -> Path:
    path = Path(str(ref.get("path") or "")).resolve()
    expected = str(ref.get("sha256") or "")
    if not path.is_file() or len(expected) != 64 or sha256(path) != expected:
        raise ValueError(code)
    return path


def _equal(code: str, got: np.ndarray, expected: np.ndarray) -> None:
    if got.shape != expected.shape or not np.array_equal(got, expected):
        raise ValueError("TRUTH139_DRAWABLE_COURT_" + code + "_DRIFT")


def _close(code: str, got: np.ndarray, expected: np.ndarray, atol: float = 1.0e-6) -> None:
    if got.shape != expected.shape or not np.allclose(got, expected, atol=atol, rtol=0.0):
        raise ValueError("TRUTH139_DRAWABLE_COURT_" + code + "_DRIFT")


def validate(manifest: dict) -> dict:
    if manifest.get("schema") != PARENT_SCHEMA or manifest.get("status") != PARENT_STATUS:
        raise ValueError("TRUTH139_DRAWABLE_COURT_PARENT_NOT_CANDIDATE")
    subject_id = str(manifest.get("subject_id") or "")
    parents = dict(manifest.get("parents") or {})
    component_path = _bound_path(dict(parents.get("component_mask_manifest") or {}), "TRUTH139_DRAWABLE_COURT_COMPONENT_DRIFT")
    layer_path = _bound_path(dict(parents.get("layer_peel_teacher_manifest") or {}), "TRUTH139_DRAWABLE_COURT_LAYER_DRIFT")
    geometry_path = _bound_path(dict(parents.get("geometry_npz") or {}), "TRUTH139_DRAWABLE_COURT_GEOMETRY_DRIFT")
    affinity_path = _bound_path(dict(manifest.get("affinity_npz") or {}), "TRUTH139_DRAWABLE_COURT_AFFINITY_DRIFT")

    component = json.loads(component_path.read_text())
    layer = json.loads(layer_path.read_text())
    if component.get("schema") != COMPONENT_SCHEMA or layer.get("schema") != LAYER_SCHEMA:
        raise ValueError("TRUTH139_DRAWABLE_COURT_PARENT_SCHEMA")
    if str(component.get("subject_id") or "") != subject_id or str(layer.get("subject_id") or "") != subject_id:
        raise ValueError("TRUTH139_DRAWABLE_COURT_SUBJECT_DRIFT")
    if str(component.get("component_mask_teacher_identity_sha256") or "") != str(parents.get("component_mask_teacher_identity_sha256") or ""):
        raise ValueError("TRUTH139_DRAWABLE_COURT_COMPONENT_IDENTITY")
    if str(layer.get("layer_peel_teacher_identity_sha256") or "") != str(parents.get("layer_peel_teacher_identity_sha256") or ""):
        raise ValueError("TRUTH139_DRAWABLE_COURT_LAYER_IDENTITY")

    with np.load(geometry_path, allow_pickle=False) as z:
        faces = np.asarray(z["faces"], dtype=np.int64)
    adjacent_pair, adjacent_edge = exact_face_edge_adjacency(faces)

    cviews = sorted(list(component.get("views") or ()), key=lambda row: int(row.get("view_index", -1)))
    if len(cviews) != 8:
        raise ValueError("TRUTH139_DRAWABLE_COURT_COMPONENT_VIEW_COUNT")
    face_component = None
    for row in cviews:
        p = _bound_path({"path": row.get("path"), "sha256": row.get("sha256")}, "TRUTH139_DRAWABLE_COURT_COMPONENT_VIEW_DRIFT")
        with np.load(p, allow_pickle=False) as z:
            current = np.asarray(z["face_local_component"], dtype=np.int32).reshape(-1)
        if current.shape != (len(faces),):
            raise ValueError("TRUTH139_DRAWABLE_COURT_COMPONENT_FACE_LAYOUT")
        if face_component is None:
            face_component = current
        elif not np.array_equal(face_component, current):
            raise ValueError("TRUTH139_DRAWABLE_COURT_COMPONENT_FACE_VIEW_DRIFT")
    assert face_component is not None
    adjacent_same = (
        face_component[adjacent_pair[:, 0]] == face_component[adjacent_pair[:, 1]]
        if len(adjacent_pair) else np.empty(0, dtype=np.bool_)
    )

    lviews = sorted(list(layer.get("views") or ()), key=lambda row: int(row.get("view_index", -1)))
    expected_view, expected_pixel, expected_slot = [], [], []
    expected_front, expected_behind, expected_gap, expected_same = [], [], [], []
    tie_abstain = 0
    for view_index, row in enumerate(lviews):
        p = _bound_path({"path": row.get("path"), "sha256": row.get("sha256")}, "TRUTH139_DRAWABLE_COURT_PEEL_VIEW_DRIFT")
        with np.load(p, allow_pickle=False) as z:
            pixels = np.asarray(z["pixel_linear_index"], dtype=np.int32).reshape(-1)
            valid = np.asarray(z["hit_valid"], dtype=np.bool_)
            tri = np.asarray(z["hit_triangle_id"], dtype=np.int32)
            depth = np.asarray(z["hit_depth"], dtype=np.float32)
            tie = np.asarray(z["adjacent_depth_tie"], dtype=np.bool_)
        for slot in range(valid.shape[1] - 1):
            pair_valid = valid[:, slot] & valid[:, slot + 1]
            tied = pair_valid & tie[:, slot]
            tie_abstain += int(np.count_nonzero(tied))
            use = np.flatnonzero(pair_valid & ~tied)
            if not len(use):
                continue
            front = tri[use, slot]
            behind = tri[use, slot + 1]
            gap = depth[use, slot + 1] - depth[use, slot]
            if np.any(gap <= 0):
                raise ValueError("TRUTH139_DRAWABLE_COURT_PARENT_DEPTH_ORDER")
            expected_view.extend([view_index] * len(use))
            expected_pixel.extend(pixels[use].tolist())
            expected_slot.extend([slot] * len(use))
            expected_front.extend(front.tolist())
            expected_behind.extend(behind.tolist())
            expected_gap.extend(gap.tolist())
            expected_same.extend((face_component[front] == face_component[behind]).tolist())

    with np.load(affinity_path, allow_pickle=False) as z:
        _equal("ADJ_PAIR", np.asarray(z["adjacent_face_pair"], dtype=np.int32), adjacent_pair.astype(np.int32))
        _equal("ADJ_EDGE", np.asarray(z["adjacent_shared_edge_vertex"], dtype=np.int32), adjacent_edge.astype(np.int32))
        _equal("ADJ_SAME", np.asarray(z["adjacent_same_authored_component"], dtype=np.bool_), np.asarray(adjacent_same, dtype=np.bool_))
        _equal("ADJ_BOUNDARY", np.asarray(z["adjacent_authored_component_boundary"], dtype=np.bool_), np.asarray(~adjacent_same, dtype=np.bool_))
        _equal("RAY_VIEW", np.asarray(z["ray_view_index"], dtype=np.int16), np.asarray(expected_view, dtype=np.int16))
        _equal("RAY_PIXEL", np.asarray(z["ray_pixel_linear_index"], dtype=np.int32), np.asarray(expected_pixel, dtype=np.int32))
        _equal("RAY_SLOT", np.asarray(z["ray_front_slot"], dtype=np.int16), np.asarray(expected_slot, dtype=np.int16))
        _equal("RAY_FRONT", np.asarray(z["ray_front_face"], dtype=np.int32), np.asarray(expected_front, dtype=np.int32))
        _equal("RAY_BEHIND", np.asarray(z["ray_behind_face"], dtype=np.int32), np.asarray(expected_behind, dtype=np.int32))
        _close("RAY_GAP", np.asarray(z["ray_positive_depth_gap"], dtype=np.float32), np.asarray(expected_gap, dtype=np.float32))
        _equal("RAY_SAME", np.asarray(z["ray_same_authored_component"], dtype=np.bool_), np.asarray(expected_same, dtype=np.bool_))

    payload = {
        "schema": SCHEMA,
        "date": "2026-10-29",
        "status": "PASS_DRAWABLE_AFFINITY_REBINDS_EXACT_COMPONENT_AND_PEEL_TRUTH",
        "subject_id": subject_id,
        "drawable_affinity_teacher_identity_sha256": str(manifest.get("drawable_affinity_teacher_identity_sha256") or ""),
        "validated_counts": {
            "adjacent_face_pair": int(len(adjacent_pair)),
            "ray_order_pair": int(len(expected_view)),
            "depth_tie_abstention": int(tie_abstain),
        },
        "policy": {
            "component_integer_used_as_student_label": False,
            "pairwise_affinity_recomputed": True,
            "ray_order_recomputed_from_peel_parent": True,
            "semantic_part_taxonomy_used": False,
            "training_performed": False,
        },
        "qualification_minted": False,
    }
    payload["court_identity_sha256"] = _digest(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drawable-affinity-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.drawable_affinity_manifest.resolve().read_text())
    result = validate(manifest)
    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "court_identity_sha256": result["court_identity_sha256"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
