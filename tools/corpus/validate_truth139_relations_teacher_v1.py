#!/usr/bin/env python3
"""Validate Truth139 primitive Relations teacher against exact parent truth.

This court does not re-prove visibility physics; the CPU ray-stack court owns that.  It proves that
Relations truth is an exact rebinding of admitted Graphics arrays, exact mesh adjacency, and optional
hash-bound Stage47 response measurements.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from tools.corpus.build_truth139_dense_appearance_gbuffer_v1 import sha256

SCHEMA = "RealSaS.Truth139RelationsTeacherCourt.v1"
PARENT_SCHEMA = "RealSaS.Truth139RelationsTeacher.v1"
PARENT_STATUS = "PASS_RELATIONS_TEACHER_CANDIDATE__INDEPENDENT_COURT_REQUIRED"
GRAPHICS_SCHEMA = "RealSaS.Truth139DenseGraphicsTeacher.v1"
VISIBLE = 1
OCCLUDED = 2


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


def _adjacency(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f = np.asarray(faces, dtype=np.int64)
    if f.ndim != 2 or f.shape[1:] != (3,) or len(f) == 0:
        raise ValueError("TRUTH139_RELATIONS_COURT_FACE_LAYOUT")
    edge_owners: dict[tuple[int, int], set[int]] = {}
    for fi, tri in enumerate(f.tolist()):
        if len(set(map(int, tri))) != 3:
            raise ValueError("TRUTH139_RELATIONS_COURT_DEGENERATE_FACE")
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge_owners.setdefault(tuple(sorted((int(a), int(b)))), set()).add(int(fi))
    pairs, edges = [], []
    for edge in sorted(edge_owners):
        for a, b in itertools.combinations(sorted(edge_owners[edge]), 2):
            pairs.append((a, b))
            edges.append(edge)
    return np.asarray(pairs, dtype=np.int32).reshape(-1, 2), np.asarray(edges, dtype=np.int32).reshape(-1, 2)


def _load_dense(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        tri = np.asarray(z["triangle_id"], dtype=np.int32).reshape(-1)
        bary = np.asarray(z["barycentric_uv"], dtype=np.float32)
    if bary.shape != (len(tri), 2):
        raise ValueError("TRUTH139_RELATIONS_COURT_DENSE_LAYOUT")
    return {"triangle": tri, "bary": bary}


def _equal(code: str, got: np.ndarray, expected: np.ndarray) -> None:
    if got.shape != expected.shape or not np.array_equal(got, expected):
        raise ValueError("TRUTH139_RELATIONS_COURT_" + code + "_DRIFT")


def _close(code: str, got: np.ndarray, expected: np.ndarray, atol: float = 1.0e-6) -> None:
    if got.shape != expected.shape or not np.allclose(got, expected, atol=atol, rtol=0.0, equal_nan=True):
        raise ValueError("TRUTH139_RELATIONS_COURT_" + code + "_DRIFT")


def validate(manifest: dict) -> dict:
    if manifest.get("schema") != PARENT_SCHEMA or manifest.get("status") != PARENT_STATUS:
        raise ValueError("TRUTH139_RELATIONS_COURT_PARENT_NOT_CANDIDATE")
    subject_id = str(manifest.get("subject_id") or "")
    if not subject_id:
        raise ValueError("TRUTH139_RELATIONS_COURT_SUBJECT")

    parents = dict(manifest.get("parents") or {})
    graphics_path = _bound_path(dict(parents.get("dense_graphics_teacher_manifest") or {}), "TRUTH139_RELATIONS_COURT_GRAPHICS_DRIFT")
    geometry_path = _bound_path(dict(parents.get("geometry_npz") or {}), "TRUTH139_RELATIONS_COURT_GEOMETRY_DRIFT")
    visibility_path = _bound_path(dict(parents.get("visibility_occlusion_teacher") or {}), "TRUTH139_RELATIONS_COURT_VISIBILITY_DRIFT")
    relations_path = _bound_path(dict(manifest.get("relations_npz") or {}), "TRUTH139_RELATIONS_COURT_RELATIONS_DRIFT")

    graphics = json.loads(graphics_path.read_text())
    if graphics.get("schema") != GRAPHICS_SCHEMA or str(graphics.get("subject_id") or "") != subject_id:
        raise ValueError("TRUTH139_RELATIONS_COURT_GRAPHICS_BINDING")
    if str(graphics.get("graphics_teacher_identity_sha256") or "") != str(parents.get("graphics_teacher_identity_sha256") or ""):
        raise ValueError("TRUTH139_RELATIONS_COURT_GRAPHICS_IDENTITY")

    dense_meta = sorted(list(graphics.get("dense_views") or ()), key=lambda row: int(row.get("view_index", -1)))
    if len(dense_meta) != 8 or [int(row.get("view_index", -1)) for row in dense_meta] != list(range(8)):
        raise ValueError("TRUTH139_RELATIONS_COURT_DENSE_COUNT")
    dense = [
        _load_dense(_bound_path({"path": row.get("teacher_path"), "sha256": row.get("teacher_sha256")}, "TRUTH139_RELATIONS_COURT_DENSE_VIEW_DRIFT"))
        for row in dense_meta
    ]

    with np.load(geometry_path, allow_pickle=False) as z:
        expected_adjacent, expected_edges = _adjacency(np.asarray(z["faces"], dtype=np.int64))

    with np.load(visibility_path, allow_pickle=False) as z:
        sv = np.asarray(z["source_view_index"], dtype=np.int16).reshape(-1)
        sr = np.asarray(z["source_surface_row"], dtype=np.int32).reshape(-1)
        stri = np.asarray(z["source_triangle_id"], dtype=np.int32).reshape(-1)
        sbary = np.asarray(z["source_barycentric_uv"], dtype=np.float32)
        cls = np.asarray(z["target_visibility_class"], dtype=np.uint8)
        front = np.asarray(z["target_front_row"], dtype=np.int32)
        gap = np.asarray(z["target_depth_gap"], dtype=np.float32)
        same = np.asarray(z["cross_view_same_surface_match"], dtype=np.bool_)
        match = np.asarray(z["cross_view_match_row"], dtype=np.int32)
        error = np.asarray(z["cross_view_match_surface_error"], dtype=np.float32)
    n = len(sv)
    if not (sr.shape == stri.shape == (n,) and sbary.shape == (n, 2) and cls.shape == front.shape == gap.shape == same.shape == match.shape == error.shape == (n, 8)):
        raise ValueError("TRUTH139_RELATIONS_COURT_VISIBILITY_LAYOUT")

    same_i, behind_i = [], []
    for i in range(n):
        source_view, source_row = int(sv[i]), int(sr[i])
        if source_view < 0 or source_view >= 8 or source_row < 0 or source_row >= len(dense[source_view]["triangle"]):
            raise ValueError("TRUTH139_RELATIONS_COURT_SOURCE_RANGE")
        if int(stri[i]) != int(dense[source_view]["triangle"][source_row]) or not np.allclose(sbary[i], dense[source_view]["bary"][source_row], atol=1.0e-6, rtol=0.0):
            raise ValueError("TRUTH139_RELATIONS_COURT_SOURCE_BINDING_DRIFT")
        for tv in range(8):
            if bool(same[i, tv]):
                tr = int(match[i, tv])
                if int(cls[i, tv]) != VISIBLE or tr < 0 or tr >= len(dense[tv]["triangle"]):
                    raise ValueError("TRUTH139_RELATIONS_COURT_SAME_PARENT_CONTRADICTION")
                same_i.append((source_view, i, source_row, tv, tr, float(error[i, tv])))
            if int(cls[i, tv]) == OCCLUDED:
                fr = int(front[i, tv])
                dg = float(gap[i, tv])
                if fr < 0 or fr >= len(dense[tv]["triangle"]) or not np.isfinite(dg) or dg <= 0:
                    raise ValueError("TRUTH139_RELATIONS_COURT_BEHIND_PARENT_CONTRADICTION")
                behind_i.append((source_view, i, source_row, tv, fr, dg))

    with np.load(relations_path, allow_pickle=False) as z:
        files = set(z.files)
        _equal("ADJACENCY", np.asarray(z["adjacent_face_pair"], dtype=np.int32), expected_adjacent)
        _equal("ADJACENCY_EDGE", np.asarray(z["adjacent_shared_edge_vertex"], dtype=np.int32), expected_edges)

        same_expected = np.asarray(same_i, dtype=np.float64).reshape(-1, 6)
        same_count = len(same_expected)
        for name in ("same_source_view_index", "same_source_sample_index", "same_source_surface_row", "same_target_view_index", "same_target_surface_row", "same_source_triangle_id", "same_target_triangle_id", "same_canonical_surface_error"):
            if len(np.asarray(z[name])) != same_count:
                raise ValueError("TRUTH139_RELATIONS_COURT_SAME_COUNT_DRIFT")
        if same_count:
            _equal("SAME_SV", np.asarray(z["same_source_view_index"], dtype=np.int16), same_expected[:, 0].astype(np.int16))
            _equal("SAME_SAMPLE", np.asarray(z["same_source_sample_index"], dtype=np.int32), same_expected[:, 1].astype(np.int32))
            _equal("SAME_SR", np.asarray(z["same_source_surface_row"], dtype=np.int32), same_expected[:, 2].astype(np.int32))
            _equal("SAME_TV", np.asarray(z["same_target_view_index"], dtype=np.int16), same_expected[:, 3].astype(np.int16))
            _equal("SAME_TR", np.asarray(z["same_target_surface_row"], dtype=np.int32), same_expected[:, 4].astype(np.int32))
            expected_source_tri = np.asarray([stri[int(row[1])] for row in same_expected], dtype=np.int32)
            expected_source_bary = np.asarray([sbary[int(row[1])] for row in same_expected], dtype=np.float32)
            expected_target_tri = np.asarray([dense[int(row[3])]["triangle"][int(row[4])] for row in same_expected], dtype=np.int32)
            expected_target_bary = np.asarray([dense[int(row[3])]["bary"][int(row[4])] for row in same_expected], dtype=np.float32)
            _equal("SAME_SOURCE_TRI", np.asarray(z["same_source_triangle_id"], dtype=np.int32), expected_source_tri)
            _close("SAME_SOURCE_BARY", np.asarray(z["same_source_barycentric_uv"], dtype=np.float32), expected_source_bary)
            _equal("SAME_TARGET_TRI", np.asarray(z["same_target_triangle_id"], dtype=np.int32), expected_target_tri)
            _close("SAME_TARGET_BARY", np.asarray(z["same_target_barycentric_uv"], dtype=np.float32), expected_target_bary)
            _close("SAME_ERROR", np.asarray(z["same_canonical_surface_error"], dtype=np.float32), same_expected[:, 5].astype(np.float32))

        behind_expected = np.asarray(behind_i, dtype=np.float64).reshape(-1, 6)
        behind_count = len(behind_expected)
        for name in ("behind_source_view_index", "behind_source_sample_index", "behind_source_surface_row", "behind_target_view_index", "front_target_surface_row", "behind_triangle_id", "front_triangle_id", "front_behind_depth_gap"):
            if len(np.asarray(z[name])) != behind_count:
                raise ValueError("TRUTH139_RELATIONS_COURT_BEHIND_COUNT_DRIFT")
        if behind_count:
            _equal("BEHIND_SV", np.asarray(z["behind_source_view_index"], dtype=np.int16), behind_expected[:, 0].astype(np.int16))
            _equal("BEHIND_SAMPLE", np.asarray(z["behind_source_sample_index"], dtype=np.int32), behind_expected[:, 1].astype(np.int32))
            _equal("BEHIND_SR", np.asarray(z["behind_source_surface_row"], dtype=np.int32), behind_expected[:, 2].astype(np.int32))
            _equal("BEHIND_TV", np.asarray(z["behind_target_view_index"], dtype=np.int16), behind_expected[:, 3].astype(np.int16))
            _equal("BEHIND_FR", np.asarray(z["front_target_surface_row"], dtype=np.int32), behind_expected[:, 4].astype(np.int32))
            expected_behind_tri = np.asarray([stri[int(row[1])] for row in behind_expected], dtype=np.int32)
            expected_behind_bary = np.asarray([sbary[int(row[1])] for row in behind_expected], dtype=np.float32)
            expected_front_tri = np.asarray([dense[int(row[3])]["triangle"][int(row[4])] for row in behind_expected], dtype=np.int32)
            expected_front_bary = np.asarray([dense[int(row[3])]["bary"][int(row[4])] for row in behind_expected], dtype=np.float32)
            _equal("BEHIND_TRI", np.asarray(z["behind_triangle_id"], dtype=np.int32), expected_behind_tri)
            _close("BEHIND_BARY", np.asarray(z["behind_barycentric_uv"], dtype=np.float32), expected_behind_bary)
            _equal("FRONT_TRI", np.asarray(z["front_triangle_id"], dtype=np.int32), expected_front_tri)
            _close("FRONT_BARY", np.asarray(z["front_barycentric_uv"], dtype=np.float32), expected_front_bary)
            _close("DEPTH_GAP", np.asarray(z["front_behind_depth_gap"], dtype=np.float32), behind_expected[:, 5].astype(np.float32))

        response_names = {"response_pair_face", "response_pair_coherence", "response_pair_jump", "response_pair_valid"}
        response_parent = parents.get("stage47_response_arrays")
        if response_parent is None:
            if files.intersection(response_names):
                raise ValueError("TRUTH139_RELATIONS_COURT_UNBOUND_RESPONSE_ROWS")
            response_count = response_valid = 0
        else:
            response_path = _bound_path(dict(response_parent), "TRUTH139_RELATIONS_COURT_RESPONSE_DRIFT")
            if not response_names.issubset(files):
                raise ValueError("TRUTH139_RELATIONS_COURT_RESPONSE_ROWS_MISSING")
            with np.load(response_path, allow_pickle=False) as p:
                for name in response_names:
                    _equal("RESPONSE_" + name.upper(), np.asarray(z[name]), np.asarray(p[name]))
                response_count = int(len(np.asarray(p["response_pair_face"])))
                response_valid = int(np.count_nonzero(np.asarray(p["response_pair_valid"], dtype=np.bool_)))

    payload = {
        "schema": SCHEMA,
        "date": "2026-10-29",
        "status": "PASS_RELATIONS_TEACHER_REBINDS_EXACT_PARENT_TRUTH",
        "subject_id": subject_id,
        "relations_teacher_identity_sha256": str(manifest.get("relations_teacher_identity_sha256") or ""),
        "validated_counts": {
            "local_surface_adjacency": int(len(expected_adjacent)),
            "same_canonical_surface_across_views": int(len(same_i)),
            "front_behind_on_shared_camera_ray": int(len(behind_i)),
            "response_pair_total": int(response_count),
            "response_pair_valid": int(response_valid),
        },
        "policy": {
            "mesh_adjacency_recomputed": True,
            "graphics_relations_recomputed_from_bound_parent_arrays": True,
            "visibility_physics_owned_by_separate_cpu_ray_oracle_court": True,
            "response_rows_exact_hash_bound_copy_only": True,
            "semantic_part_material_owner_or_bone_name_evidence_used": False,
            "training_performed": False,
        },
        "qualification_minted": False,
    }
    payload["court_identity_sha256"] = _digest(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relations-teacher-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = validate(json.loads(args.relations_teacher_manifest.resolve().read_text()))
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = output.with_suffix(output.suffix + ".partial")
    staged.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    staged.replace(output)
    print(json.dumps({"status": result["status"], "subject_id": result["subject_id"], "validated_counts": result["validated_counts"], "court_identity_sha256": result["court_identity_sha256"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
