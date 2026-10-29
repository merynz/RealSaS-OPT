#!/usr/bin/env python3
"""Build primitive, architecture-neutral Relations truth for Truth139 IRIS/TESSA.

Truth is rebound only from exact admitted mesh topology and the hash-bound dense Graphics teacher.
Optional response rows are copied from already-measured Stage47 evidence; they never become rig or
skin authority. Missing rows are missing truth, never negatives.
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

SCHEMA = "RealSaS.Truth139RelationsTeacher.v1"
GRAPHICS_SCHEMA = "RealSaS.Truth139DenseGraphicsTeacher.v1"
GRAPHICS_STATUS = "PASS_DENSE_GRAPHICS_TEACHER_CANDIDATE__INDEPENDENT_COURTS_REQUIRED"
CONTRACT = "canonical/TRUTH139_RELATIONS_TEACHER_V1.json"
VISIBLE = 1
OCCLUDED = 2


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


def exact_face_edge_adjacency(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return all unordered face pairs sharing an exact undirected mesh edge."""
    f = np.asarray(faces, dtype=np.int64)
    if f.ndim != 2 or f.shape[1:] != (3,) or len(f) == 0 or np.any(f < 0):
        raise ValueError("TRUTH139_RELATIONS_FACE_LAYOUT")
    edge_to_faces: dict[tuple[int, int], set[int]] = {}
    for face_index, tri in enumerate(f.tolist()):
        if len(set(map(int, tri))) != 3:
            raise ValueError("TRUTH139_RELATIONS_DEGENERATE_FACE_INDEX")
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge_to_faces.setdefault(tuple(sorted((int(a), int(b)))), set()).add(face_index)
    pairs, edges = [], []
    for edge in sorted(edge_to_faces):
        for a, b in itertools.combinations(sorted(edge_to_faces[edge]), 2):
            pairs.append((a, b))
            edges.append(edge)
    return np.asarray(pairs, dtype=np.int32).reshape(-1, 2), np.asarray(edges, dtype=np.int32).reshape(-1, 2)


def _load_dense(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        required = {"triangle_id", "barycentric_uv", "surface_xyz_source_world", "surface_xyz_canonical"}
        if not required.issubset(z.files):
            raise ValueError("TRUTH139_RELATIONS_DENSE_FIELDS_MISSING")
        tri = np.asarray(z["triangle_id"], dtype=np.int32).reshape(-1)
        bary = np.asarray(z["barycentric_uv"], dtype=np.float32)
        source = np.asarray(z["surface_xyz_source_world"], dtype=np.float32)
        canonical = np.asarray(z["surface_xyz_canonical"], dtype=np.float32)
    n = len(tri)
    if bary.shape != (n, 2) or source.shape != (n, 3) or canonical.shape != (n, 3):
        raise ValueError("TRUTH139_RELATIONS_DENSE_LAYOUT")
    if not np.isfinite(bary).all() or not np.isfinite(source).all() or not np.isfinite(canonical).all():
        raise ValueError("TRUTH139_RELATIONS_DENSE_NONFINITE")
    return {"triangle": tri, "bary": bary, "source": source, "canonical": canonical}


def _load_response(path: Path, face_count: int) -> dict[str, np.ndarray]:
    names = ("response_pair_face", "response_pair_coherence", "response_pair_jump", "response_pair_valid")
    with np.load(path, allow_pickle=False) as z:
        if not set(names).issubset(z.files):
            raise ValueError("TRUTH139_RELATIONS_RESPONSE_FIELDS_MISSING")
        pair = np.asarray(z[names[0]], dtype=np.int32)
        coherence = np.asarray(z[names[1]], dtype=np.float32).reshape(-1)
        jump = np.asarray(z[names[2]], dtype=np.float32).reshape(-1)
        valid = np.asarray(z[names[3]], dtype=np.bool_).reshape(-1)
    if pair.ndim != 2 or pair.shape[1:] != (2,) or not (len(pair) == len(coherence) == len(jump) == len(valid)):
        raise ValueError("TRUTH139_RELATIONS_RESPONSE_LAYOUT")
    if len(pair) and (np.any(pair < 0) or np.any(pair >= face_count)):
        raise ValueError("TRUTH139_RELATIONS_RESPONSE_FACE_RANGE")
    if not np.isfinite(coherence).all() or not np.isfinite(jump).all():
        raise ValueError("TRUTH139_RELATIONS_RESPONSE_NONFINITE")
    if np.any((coherence[valid] < -1.000001) | (coherence[valid] > 1.000001)) or np.any(jump[valid] < -1.0e-8):
        raise ValueError("TRUTH139_RELATIONS_RESPONSE_RANGE")
    return {"pair": pair, "coherence": coherence, "jump": jump, "valid": valid}


def build(*, dense_graphics_teacher_manifest: Path, output_dir: Path, response_arrays: Path | None = None) -> dict:
    manifest_path = dense_graphics_teacher_manifest.resolve()
    output_dir = output_dir.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError("TRUTH139_RELATIONS_GRAPHICS_MANIFEST_MISSING")
    graphics = json.loads(manifest_path.read_text())
    if graphics.get("schema") != GRAPHICS_SCHEMA or graphics.get("status") != GRAPHICS_STATUS:
        raise ValueError("TRUTH139_RELATIONS_GRAPHICS_PARENT_NOT_CANDIDATE")
    subject_id = str(graphics.get("subject_id") or "")
    if not subject_id:
        raise ValueError("TRUTH139_RELATIONS_SUBJECT_ID")

    parents = dict(graphics.get("parents") or {})
    geometry_path = _bound_path(dict(parents.get("geometry_npz") or {}), "TRUTH139_RELATIONS_GEOMETRY_DRIFT")
    visibility_ref = dict(graphics.get("visibility_occlusion") or {})
    visibility_path = _bound_path({"path": visibility_ref.get("path"), "sha256": visibility_ref.get("sha256")}, "TRUTH139_RELATIONS_VISIBILITY_DRIFT")
    surface_tolerance = float(visibility_ref.get("surface_tolerance", 0.0))
    if not np.isfinite(surface_tolerance) or surface_tolerance <= 0:
        raise ValueError("TRUTH139_RELATIONS_SURFACE_TOLERANCE")

    with np.load(geometry_path, allow_pickle=False) as z:
        faces = np.asarray(z["faces"], dtype=np.int64)
    adjacency_pair, adjacency_edge = exact_face_edge_adjacency(faces)

    dense_meta = sorted(list(graphics.get("dense_views") or ()), key=lambda row: int(row.get("view_index", -1)))
    if len(dense_meta) != 8 or [int(row.get("view_index", -1)) for row in dense_meta] != list(range(8)):
        raise ValueError("TRUTH139_RELATIONS_DENSE_VIEW_COUNT")
    dense = [
        _load_dense(_bound_path({"path": row.get("teacher_path"), "sha256": row.get("teacher_sha256")}, "TRUTH139_RELATIONS_DENSE_VIEW_DRIFT"))
        for row in dense_meta
    ]

    with np.load(visibility_path, allow_pickle=False) as z:
        required = {
            "source_view_index", "source_surface_row", "source_triangle_id", "source_barycentric_uv",
            "target_visibility_class", "target_front_row", "target_depth_gap",
            "cross_view_same_surface_match", "cross_view_match_row", "cross_view_match_surface_error",
        }
        if not required.issubset(z.files):
            raise ValueError("TRUTH139_RELATIONS_VISIBILITY_FIELDS_MISSING")
        sv = np.asarray(z["source_view_index"], dtype=np.int16).reshape(-1)
        sr = np.asarray(z["source_surface_row"], dtype=np.int32).reshape(-1)
        stri = np.asarray(z["source_triangle_id"], dtype=np.int32).reshape(-1)
        sbary = np.asarray(z["source_barycentric_uv"], dtype=np.float32)
        classes = np.asarray(z["target_visibility_class"], dtype=np.uint8)
        front = np.asarray(z["target_front_row"], dtype=np.int32)
        gap = np.asarray(z["target_depth_gap"], dtype=np.float32)
        same = np.asarray(z["cross_view_same_surface_match"], dtype=np.bool_)
        match = np.asarray(z["cross_view_match_row"], dtype=np.int32)
        error = np.asarray(z["cross_view_match_surface_error"], dtype=np.float32)
    n = len(sv)
    if not (sr.shape == stri.shape == (n,) and sbary.shape == (n, 2) and classes.shape == front.shape == gap.shape == same.shape == match.shape == error.shape == (n, 8)):
        raise ValueError("TRUTH139_RELATIONS_VISIBILITY_LAYOUT")

    same_rows, behind_rows = [], []
    for sample in range(n):
        source_view, source_row = int(sv[sample]), int(sr[sample])
        if source_view < 0 or source_view >= 8 or source_row < 0 or source_row >= len(dense[source_view]["triangle"]):
            raise ValueError("TRUTH139_RELATIONS_SOURCE_RANGE")
        if int(stri[sample]) != int(dense[source_view]["triangle"][source_row]) or not np.allclose(sbary[sample], dense[source_view]["bary"][source_row], atol=1.0e-6, rtol=0.0):
            raise ValueError("TRUTH139_RELATIONS_SOURCE_BINDING_DRIFT")
        for target_view in range(8):
            if bool(same[sample, target_view]):
                target_row = int(match[sample, target_view])
                e = float(error[sample, target_view])
                if int(classes[sample, target_view]) != VISIBLE or target_row < 0 or target_row >= len(dense[target_view]["triangle"]) or not np.isfinite(e) or e > surface_tolerance:
                    raise ValueError("TRUTH139_RELATIONS_SAME_SURFACE_PARENT_CONTRADICTION")
                same_rows.append((source_view, sample, source_row, target_view, target_row, e))
            if int(classes[sample, target_view]) == OCCLUDED:
                front_row = int(front[sample, target_view])
                depth_gap = float(gap[sample, target_view])
                if front_row < 0 or front_row >= len(dense[target_view]["triangle"]) or not np.isfinite(depth_gap) or depth_gap <= 0:
                    raise ValueError("TRUTH139_RELATIONS_BEHIND_PARENT_CONTRADICTION")
                behind_rows.append((source_view, sample, source_row, target_view, front_row, depth_gap))

    same_arr = np.asarray(same_rows, dtype=np.float64).reshape(-1, 6)
    behind_arr = np.asarray(behind_rows, dtype=np.float64).reshape(-1, 6)
    same_sample = same_arr[:, 1].astype(np.int64) if len(same_arr) else np.empty(0, dtype=np.int64)
    same_tv = same_arr[:, 3].astype(np.int64) if len(same_arr) else np.empty(0, dtype=np.int64)
    same_tr = same_arr[:, 4].astype(np.int64) if len(same_arr) else np.empty(0, dtype=np.int64)
    behind_sample = behind_arr[:, 1].astype(np.int64) if len(behind_arr) else np.empty(0, dtype=np.int64)
    behind_tv = behind_arr[:, 3].astype(np.int64) if len(behind_arr) else np.empty(0, dtype=np.int64)
    behind_fr = behind_arr[:, 4].astype(np.int64) if len(behind_arr) else np.empty(0, dtype=np.int64)

    response = None
    response_parent = None
    if response_arrays is not None:
        response_path = response_arrays.resolve()
        if not response_path.is_file():
            raise FileNotFoundError("TRUTH139_RELATIONS_RESPONSE_PARENT_MISSING")
        response = _load_response(response_path, len(faces))
        response_parent = {"path": str(response_path), "sha256": sha256(response_path)}

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "RELATIONS_TEACHER.npz"
    arrays = {
        "adjacent_face_pair": adjacency_pair,
        "adjacent_shared_edge_vertex": adjacency_edge,
        "same_source_view_index": same_arr[:, 0].astype(np.int16),
        "same_source_sample_index": same_arr[:, 1].astype(np.int32),
        "same_source_surface_row": same_arr[:, 2].astype(np.int32),
        "same_target_view_index": same_arr[:, 3].astype(np.int16),
        "same_target_surface_row": same_arr[:, 4].astype(np.int32),
        "same_source_triangle_id": stri[same_sample].astype(np.int32),
        "same_source_barycentric_uv": sbary[same_sample].astype(np.float32),
        "same_target_triangle_id": np.asarray([dense[tv]["triangle"][tr] for tv, tr in zip(same_tv, same_tr)], dtype=np.int32),
        "same_target_barycentric_uv": np.asarray([dense[tv]["bary"][tr] for tv, tr in zip(same_tv, same_tr)], dtype=np.float32).reshape(-1, 2),
        "same_canonical_surface_error": same_arr[:, 5].astype(np.float32),
        "behind_source_view_index": behind_arr[:, 0].astype(np.int16),
        "behind_source_sample_index": behind_arr[:, 1].astype(np.int32),
        "behind_source_surface_row": behind_arr[:, 2].astype(np.int32),
        "behind_target_view_index": behind_arr[:, 3].astype(np.int16),
        "front_target_surface_row": behind_arr[:, 4].astype(np.int32),
        "behind_triangle_id": stri[behind_sample].astype(np.int32),
        "behind_barycentric_uv": sbary[behind_sample].astype(np.float32),
        "front_triangle_id": np.asarray([dense[tv]["triangle"][fr] for tv, fr in zip(behind_tv, behind_fr)], dtype=np.int32),
        "front_barycentric_uv": np.asarray([dense[tv]["bary"][fr] for tv, fr in zip(behind_tv, behind_fr)], dtype=np.float32).reshape(-1, 2),
        "front_behind_depth_gap": behind_arr[:, 5].astype(np.float32),
    }
    if response is not None:
        arrays.update(
            response_pair_face=response["pair"], response_pair_coherence=response["coherence"],
            response_pair_jump=response["jump"], response_pair_valid=response["valid"],
        )
    _write_npz(out, **arrays)

    payload = {
        "schema": SCHEMA,
        "date": "2026-10-29",
        "status": "PASS_RELATIONS_TEACHER_CANDIDATE__INDEPENDENT_COURT_REQUIRED",
        "subject_id": subject_id,
        "contract": CONTRACT,
        "parents": {
            "dense_graphics_teacher_manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
            "graphics_teacher_identity_sha256": str(graphics.get("graphics_teacher_identity_sha256") or ""),
            "geometry_npz": {"path": str(geometry_path), "sha256": sha256(geometry_path)},
            "visibility_occlusion_teacher": {"path": str(visibility_path), "sha256": sha256(visibility_path)},
            "stage47_response_arrays": response_parent,
        },
        "relations_npz": {"path": str(out), "sha256": sha256(out)},
        "counts": {
            "local_surface_adjacency": int(len(adjacency_pair)),
            "same_canonical_surface_across_views": int(len(same_arr)),
            "front_behind_on_shared_camera_ray": int(len(behind_arr)),
            "response_pair_total": 0 if response is None else int(len(response["pair"])),
            "response_pair_valid": 0 if response is None else int(np.count_nonzero(response["valid"])),
        },
        "policy": {
            "semantic_part_taxonomy_used": False,
            "owner_material_or_bone_name_targets_used": False,
            "global_total_draw_order_claimed": False,
            "absence_is_negative_label": False,
            "relations_are_axis_or_mira_authority": False,
            "training_performed": False,
        },
        "qualification_minted": False,
    }
    payload["relations_teacher_identity_sha256"] = _digest(payload)
    manifest_out = output_dir / "RELATIONS_TEACHER_MANIFEST.json"
    manifest_out.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": payload["status"], "subject_id": subject_id, "counts": payload["counts"], "relations_teacher_identity_sha256": payload["relations_teacher_identity_sha256"]}, sort_keys=True), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-graphics-teacher-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--response-arrays", type=Path)
    args = parser.parse_args()
    build(dense_graphics_teacher_manifest=args.dense_graphics_teacher_manifest, output_dir=args.output_dir, response_arrays=args.response_arrays)


if __name__ == "__main__":
    main()
