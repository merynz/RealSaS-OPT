#!/usr/bin/env python3
"""Validate Truth139 primitive Relations teacher against exact parent truth.

The visibility/occlusion physics is independently checked by the CPU ray oracle court.  This court
has a different responsibility: prove that the Relations package is an exact, lossless rebinding of
that admitted Graphics truth plus exact mesh adjacency and optional Stage47 response measurements.
It recomputes relation rows from the parents instead of trusting the packaged arrays.
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
+PARENT_SCHEMA = "RealSaS.Truth139RelationsTeacher.v1"
+PARENT_STATUS = "PASS_RELATIONS_TEACHER_CANDIDATE__INDEPENDENT_COURT_REQUIRED"
+GRAPHICS_SCHEMA = "RealSaS.Truth139DenseGraphicsTeacher.v1"
+VISIBLE = 1
+OCCLUDED = 2
+
+
+def _digest(value: object) -> str:
+    return hashlib.sha256(
+        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
+    ).hexdigest()
+
+
+def _bound_path(ref: Mapping[str, object], code: str) -> Path:
+    path = Path(str(ref.get("path") or "")).resolve()
+    expected = str(ref.get("sha256") or "")
+    if not path.is_file() or len(expected) != 64 or sha256(path) != expected:
+        raise ValueError(code)
+    return path
+
+
+def _recompute_adjacency(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
+    f = np.asarray(faces, dtype=np.int64)
+    if f.ndim != 2 or f.shape[1:] != (3,) or len(f) == 0:
+        raise ValueError("TRUTH139_RELATIONS_COURT_FACE_LAYOUT")
+    owners: dict[tuple[int, int], set[int]] = {}
+    for fi, tri in enumerate(f.tolist()):
+        if len(set(map(int, tri))) != 3:
+            raise ValueError("TRUTH139_RELATIONS_COURT_DEGENERATE_FACE")
+        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
+            owners.setdefault(tuple(sorted((int(a), int(b)))), set()).add(int(fi))
+    pairs: list[tuple[int, int]] = []
+    edges: list[tuple[int, int]] = []
+    for edge in sorted(owners):
+        for a, b in itertools.combinations(sorted(owners[edge]), 2):
+            pairs.append((a, b))
+            edges.append(edge)
+    return np.asarray(pairs, dtype=np.int32).reshape(-1, 2), np.asarray(edges, dtype=np.int32).reshape(-1, 2)
+
+
+def _load_dense(path: Path) -> dict[str, np.ndarray]:
+    with np.load(path, allow_pickle=False) as z:
+        tri = np.asarray(z["triangle_id"], dtype=np.int64).reshape(-1)
+        bary = np.asarray(z["barycentric_uv"], dtype=np.float64)
+        canonical = np.asarray(z["surface_xyz_canonical"], dtype=np.float64)
+    if bary.shape != (len(tri), 2) or canonical.shape != (len(tri), 3):
+        raise ValueError("TRUTH139_RELATIONS_COURT_DENSE_LAYOUT")
+    return {"triangle": tri, "bary": bary, "canonical": canonical}
+
+
+def _assert_equal(name: str, got: np.ndarray, expected: np.ndarray) -> None:
+    if got.shape != expected.shape or got.dtype.kind != expected.dtype.kind or not np.array_equal(got, expected):
+        raise ValueError(f"TRUTH139_RELATIONS_COURT_{name}_DRIFT")
+
+
+def _assert_close(name: str, got: np.ndarray, expected: np.ndarray, *, atol: float) -> None:
+    if got.shape != expected.shape or not np.allclose(got, expected, atol=atol, rtol=0.0, equal_nan=True):
+        raise ValueError(f"TRUTH139_RELATIONS_COURT_{name}_DRIFT")
+
+
+def validate(manifest: dict) -> dict:
+    if manifest.get("schema") != PARENT_SCHEMA or manifest.get("status") != PARENT_STATUS:
+        raise ValueError("TRUTH139_RELATIONS_COURT_PARENT_NOT_CANDIDATE")
+    subject_id = str(manifest.get("subject_id") or "")
+    if not subject_id:
+        raise ValueError("TRUTH139_RELATIONS_COURT_SUBJECT")
+
+    parents = dict(manifest.get("parents") or {})
+    graphics_manifest_path = _bound_path(
+        dict(parents.get("dense_graphics_teacher_manifest") or {}),
+        "TRUTH139_RELATIONS_COURT_GRAPHICS_MANIFEST_DRIFT",
+    )
+    geometry_path = _bound_path(
+        dict(parents.get("geometry_npz") or {}),
+        "TRUTH139_RELATIONS_COURT_GEOMETRY_DRIFT",
+    )
+    visibility_path = _bound_path(
+        dict(parents.get("visibility_occlusion_teacher") or {}),
+        "TRUTH139_RELATIONS_COURT_VISIBILITY_DRIFT",
+    )
+    relations_path = _bound_path(
+        dict(manifest.get("relations_npz") or {}),
+        "TRUTH139_RELATIONS_COURT_RELATIONS_DRIFT",
+    )
+
+    graphics = json.loads(graphics_manifest_path.read_text())
+    if graphics.get("schema") != GRAPHICS_SCHEMA:
+        raise ValueError("TRUTH139_RELATIONS_COURT_GRAPHICS_SCHEMA")
+    if str(graphics.get("graphics_teacher_identity_sha256") or "") != str(parents.get("graphics_teacher_identity_sha256") or ""):
+        raise ValueError("TRUTH139_RELATIONS_COURT_GRAPHICS_IDENTITY_DRIFT")
+    if str(graphics.get("subject_id") or "") != subject_id:
+        raise ValueError("TRUTH139_RELATIONS_COURT_SUBJECT_BINDING_DRIFT")
+
+    dense_meta = sorted(list(graphics.get("dense_views") or ()), key=lambda row: int(row.get("view_index", -1)))
+    if len(dense_meta) != 8 or [int(row.get("view_index", -1)) for row in dense_meta] != list(range(8)):
+        raise ValueError("TRUTH139_RELATIONS_COURT_DENSE_COUNT")
+    dense = [
+        _load_dense(
+            _bound_path(
+                {"path": row.get("teacher_path"), "sha256": row.get("teacher_sha256")},
+                "TRUTH139_RELATIONS_COURT_DENSE_VIEW_DRIFT",
+            )
+        )
+        for row in dense_meta
+    ]
+
+    with np.load(geometry_path, allow_pickle=False) as z:
+        faces = np.asarray(z["faces"], dtype=np.int64)
+    expected_adjacent, expected_edges = _recompute_adjacency(faces)
+
+    with np.load(visibility_path, allow_pickle=False) as z:
+        source_view = np.asarray(z["source_view_index"], dtype=np.int64).reshape(-1)
+        source_row = np.asarray(z["source_surface_row"], dtype=np.int64).reshape(-1)
+        source_tri = np.asarray(z["source_triangle_id"], dtype=np.int64).reshape(-1)
+        source_bary = np.asarray(z["source_barycentric_uv"], dtype=np.float64)
+        classes = np.asarray(z["target_visibility_class"], dtype=np.uint8)
+        front_row = np.asarray(z["target_front_row"], dtype=np.int64)
+        depth_gap = np.asarray(z["target_depth_gap"], dtype=np.float64)
+        same = np.asarray(z["cross_view_same_surface_match"], dtype=np.bool_)
+        match_row = np.asarray(z["cross_view_match_row"], dtype=np.int64)
+        match_error = np.asarray(z["cross_view_match_surface_error"], dtype=np.float64)
+
+    n = len(source_view)
+    if not (
+        source_row.shape == source_tri.shape == (n,)
+        and source_bary.shape == (n, 2)
+        and classes.shape == front_row.shape == depth_gap.shape == same.shape == match_row.shape == match_error.shape == (n, 8)
+    ):
+        raise ValueError("TRUTH139_RELATIONS_COURT_VISIBILITY_LAYOUT")
+
+    same_rows: list[tuple[int, int, int, int, int, int, int, float, float, int, float, float, float]] = []
+    behind_rows: list[tuple[int, int, int, int, int, int, float, float, int, float, float, float]] = []
+    for sample in range(n):
+        sv = int(source_view[sample])
+        sr = int(source_row[sample])
+        if sv < 0 or sv >= 8 or sr < 0 or sr >= len(dense[sv]["triangle"]):
+            raise ValueError("TRUTH139_RELATIONS_COURT_SOURCE_RANGE")
+        for tv in range(8):
+            if bool(same[sample, tv]):
+                if int(classes[sample, tv]) != VISIBLE:
+                    raise ValueError("TRUTH139_RELATIONS_COURT_SAME_PARENT_CLASS")
+                tr = int(match_row[sample, tv])
+                if tr < 0 or tr >= len(dense[tv]["triangle"]):
+                    raise ValueError("TRUTH139_RELATIONS_COURT_MATCH_ROW_RANGE")
+                same_rows.append(
+                    (
+                        sv, sample, sr, tv, tr, int(source_tri[sample]),
+                        float(source_bary[sample, 0]), float(source_bary[sample, 1]),
+                        int(dense[tv]["triangle"][tr]),
+                        float(dense[tv]["bary"][tr, 0]), float(dense[tv]["bary"][tr, 1]),
+                        float(match_error[sample, tv]),
+                        float(np.linalg.norm(dense[sv]["canonical"][sr] - dense[tv]["canonical"][tr])),
+                    )
+                )
+            if int(classes[sample, tv]) == OCCLUDED:
+                fr = int(front_row[sample, tv])
+                if fr < 0 or fr >= len(dense[tv]["triangle"]):
+                    raise ValueError("TRUTH139_RELATIONS_COURT_FRONT_ROW_RANGE")
+                gap = float(depth_gap[sample, tv])
+                if not np.isfinite(gap) or gap <= 0:
+                    raise ValueError("TRUTH139_RELATIONS_COURT_DEPTH_GAP")
+                behind_rows.append(
+                    (
+                        sv, sample, sr, tv, fr, int(source_tri[sample]),
+                        float(source_bary[sample, 0]), float(source_bary[sample, 1]),
+                        int(dense[tv]["triangle"][fr]),
+                        float(dense[tv]["bary"][fr, 0]), float(dense[tv]["bary"][fr, 1]), gap,
+                    )
+                )
+
+    with np.load(relations_path, allow_pickle=False) as z:
+        files = set(z.files)
+        adjacent = np.asarray(z["adjacent_face_pair"], dtype=np.int32)
+        edges = np.asarray(z["adjacent_shared_edge_vertex"], dtype=np.int32)
+        got_same = {
+            "sv": np.asarray(z["same_source_view_index"], dtype=np.int16),
+            "sample": np.asarray(z["same_source_sample_index"], dtype=np.int32),
+            "sr": np.asarray(z["same_source_surface_row"], dtype=np.int32),
+            "tv": np.asarray(z["same_target_view_index"], dtype=np.int16),
+            "tr": np.asarray(z["same_target_surface_row"], dtype=np.int32),
+            "stri": np.asarray(z["same_source_triangle_id"], dtype=np.int32),
+            "sbary": np.asarray(z["same_source_barycentric_uv"], dtype=np.float64),
+            "ttri": np.asarray(z["same_target_triangle_id"], dtype=np.int32),
+            "tbary": np.asarray(z["same_target_barycentric_uv"], dtype=np.float64),
+            "error": np.asarray(z["same_canonical_surface_error"], dtype=np.float64),
+        }
+        got_behind = {
+            "sv": np.asarray(z["behind_source_view_index"], dtype=np.int16),
+            "sample": np.asarray(z["behind_source_sample_index"], dtype=np.int32),
+            "sr": np.asarray(z["behind_source_surface_row"], dtype=np.int32),
+            "tv": np.asarray(z["behind_target_view_index"], dtype=np.int16),
+            "fr": np.asarray(z["front_target_surface_row"], dtype=np.int32),
+            "btri": np.asarray(z["behind_triangle_id"], dtype=np.int32),
+            "bbary": np.asarray(z["behind_barycentric_uv"], dtype=np.float64),
+            "ftri": np.asarray(z["front_triangle_id"], dtype=np.int32),
+            "fbary": np.asarray(z["front_barycentric_uv"], dtype=np.float64),
+            "gap": np.asarray(z["front_behind_depth_gap"], dtype=np.float64),
+        }
+
+        _assert_equal("ADJACENCY", adjacent, expected_adjacent)
+        _assert_equal("ADJACENCY_EDGE", edges, expected_edges)
+
+        same_count = len(same_rows)
+        if any(len(value) != same_count for value in got_same.values()):
+            raise ValueError("TRUTH139_RELATIONS_COURT_SAME_COUNT_DRIFT")
+        if same_count:
+            expected = np.asarray(same_rows, dtype=np.float64)
+            _assert_equal("SAME_SV", got_same["sv"], expected[:, 0].astype(np.int16))
+            _assert_equal("SAME_SAMPLE", got_same["sample"], expected[:, 1].astype(np.int32))
+            _assert_equal("SAME_SR", got_same["sr"], expected[:, 2].astype(np.int32))
+            _assert_equal("SAME_TV", got_same["tv"], expected[:, 3].astype(np.int16))
+            _assert_equal("SAME_TR", got_same["tr"], expected[:, 4].astype(np.int32))
+            _assert_equal("SAME_SOURCE_TRI", got_same["stri"], expected[:, 5].astype(np.int32))
+            _assert_close("SAME_SOURCE_BARY", got_same["sbary"], expected[:, 6:8], atol=1.0e-6)
+            _assert_equal("SAME_TARGET_TRI", got_same["ttri"], expected[:, 8].astype(np.int32))
+            _assert_close("SAME_TARGET_BARY", got_same["tbary"], expected[:, 9:11], atol=1.0e-6)
+            _assert_close("SAME_ERROR", got_same["error"], expected[:, 11], atol=1.0e-6)
+            surface_error = expected[:, 12]
+            declared_tolerance = float((graphics.get("visibility_occlusion") or {}).get("surface_tolerance", 0.0))
+            if np.any(surface_error > declared_tolerance + 1.0e-6):
+                raise ValueError("TRUTH139_RELATIONS_COURT_SAME_CANONICAL_RECONSTRUCTION_DRIFT")
+
+        behind_count = len(behind_rows)
+        if any(len(value) != behind_count for value in got_behind.values()):
+            raise ValueError("TRUTH139_RELATIONS_COURT_BEHIND_COUNT_DRIFT")
+        if behind_count:
+            expected = np.asarray(behind_rows, dtype=np.float64)
+            _assert_equal("BEHIND_SV", got_behind["sv"], expected[:, 0].astype(np.int16))
+            _assert_equal("BEHIND_SAMPLE", got_behind["sample"], expected[:, 1].astype(np.int32))
+            _assert_equal("BEHIND_SR", got_behind["sr"], expected[:, 2].astype(np.int32))
+            _assert_equal("BEHIND_TV", got_behind["tv"], expected[:, 3].astype(np.int16))
+            _assert_equal("BEHIND_FR", got_behind["fr"], expected[:, 4].astype(np.int32))
+            _assert_equal("BEHIND_TRI", got_behind["btri"], expected[:, 5].astype(np.int32))
+            _assert_close("BEHIND_BARY", got_behind["bbary"], expected[:, 6:8], atol=1.0e-6)
+            _assert_equal("FRONT_TRI", got_behind["ftri"], expected[:, 8].astype(np.int32))
+            _assert_close("FRONT_BARY", got_behind["fbary"], expected[:, 9:11], atol=1.0e-6)
+            _assert_close("DEPTH_GAP", got_behind["gap"], expected[:, 11], atol=1.0e-6)
+
+        response_parent = parents.get("stage47_response_arrays")
+        response_names = {"response_pair_face", "response_pair_coherence", "response_pair_jump", "response_pair_valid"}
+        if response_parent is None:
+            if files.intersection(response_names):
+                raise ValueError("TRUTH139_RELATIONS_COURT_UNBOUND_RESPONSE_ROWS")
+            response_count = response_valid_count = 0
+        else:
+            response_path = _bound_path(dict(response_parent), "TRUTH139_RELATIONS_COURT_RESPONSE_PARENT_DRIFT")
+            if not response_names.issubset(files):
+                raise ValueError("TRUTH139_RELATIONS_COURT_RESPONSE_ROWS_MISSING")
+            with np.load(response_path, allow_pickle=False) as parent:
+                for name in sorted(response_names):
+                    _assert_equal(
+                        "RESPONSE_" + name.upper(),
+                        np.asarray(z[name]),
+                        np.asarray(parent[name]),
+                    )
+                response_count = int(len(np.asarray(parent["response_pair_face"])))
+                response_valid_count = int(np.count_nonzero(np.asarray(parent["response_pair_valid"], dtype=np.bool_)))
+
+    payload = {
+        "schema": SCHEMA,
+        "date": "2026-10-29",
+        "status": "PASS_RELATIONS_TEACHER_REBINDS_EXACT_PARENT_TRUTH",
+        "subject_id": subject_id,
+        "relations_teacher_identity_sha256": str(manifest.get("relations_teacher_identity_sha256") or ""),
+        "validated_counts": {
+            "local_surface_adjacency": int(len(expected_adjacent)),
+            "same_canonical_surface_across_views": int(len(same_rows)),
+            "front_behind_on_shared_camera_ray": int(len(behind_rows)),
+            "response_pair_total": int(response_count),
+            "response_pair_valid": int(response_valid_count),
+        },
+        "policy": {
+            "mesh_adjacency_recomputed_independently": True,
+            "graphics_relation_rows_recomputed_from_bound_parent_arrays": True,
+            "visibility_physics_owned_by_separate_cpu_ray_oracle_court": True,
+            "response_rows_exact_hash_bound_copy_only": True,
+            "semantic_part_material_owner_or_bone_name_evidence_used": False,
+            "global_draw_order_assumed": False,
+            "training_performed": False,
+        },
+        "qualification_minted": False,
+    }
+    payload["court_identity_sha256"] = _digest(payload)
+    return payload
+
+
+def main() -> None:
+    parser = argparse.ArgumentParser(description=__doc__)
+    parser.add_argument("--relations-teacher-manifest", required=True, type=Path)
+    parser.add_argument("--output", required=True, type=Path)
+    args = parser.parse_args()
+    manifest = json.loads(args.relations_teacher_manifest.resolve().read_text())
+    result = validate(manifest)
+    output = args.output.resolve()
+    output.parent.mkdir(parents=True, exist_ok=True)
+    staged = output.with_suffix(output.suffix + ".partial")
+    staged.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
+    staged.replace(output)
+    print(json.dumps({
+        "status": result["status"],
+        "subject_id": result["subject_id"],
+        "validated_counts": result["validated_counts"],
+        "court_identity_sha256": result["court_identity_sha256"],
+    }, sort_keys=True), flush=True)
+
+
+if __name__ == "__main__":
+    main()
