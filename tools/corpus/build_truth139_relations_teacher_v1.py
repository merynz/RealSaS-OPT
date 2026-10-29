#!/usr/bin/env python3
"""Build architecture-neutral primitive relation truth for Truth139 IRIS/TESSA.

The producer only rebinds facts already owned by admitted geometry, dense Graphics truth and,
optionally, Stage47 measured response arrays.  It does not invent semantic parts, hidden owners,
a skeleton, skin weights or a global draw order.

Relation families:
* same canonical surface across source views;
* local front/behind on one measured camera ray;
* exact mesh-edge adjacency;
* optional measured response coherence/discontinuity on adjacent faces.

Absence of a row is never interpreted as a negative label.
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


def exact_face_edge_adjacency(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return every unordered face pair sharing one exact undirected edge.

    Non-manifold edges are not silently discarded: all incident face pairs are emitted and the
    common edge is retained, so a later projection can choose whether to consume them.
    """
    f = np.asarray(faces, dtype=np.int64)
    if f.ndim != 2 or f.shape[1:] != (3,) or len(f) == 0:
        raise ValueError("TRUTH139_RELATIONS_FACE_LAYOUT")
    if np.any(f < 0):
        raise ValueError("TRUTH139_RELATIONS_NEGATIVE_VERTEX_INDEX")
    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for face_index, tri in enumerate(f.tolist()):
        if len(set(map(int, tri))) != 3:
            raise ValueError("TRUTH139_RELATIONS_DEGENERATE_FACE_INDEX")
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge = tuple(sorted((int(a), int(b))))
            edge_to_faces.setdefault(edge, []).append(int(face_index))

    rows: list[tuple[int, int]] = []
    edges: list[tuple[int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for edge, owners in sorted(edge_to_faces.items()):
        unique = sorted(set(owners))
        for a, b in itertools.combinations(unique, 2):
            key = (a, b, edge[0], edge[1])
            if key in seen:
                continue
            seen.add(key)
            rows.append((a, b))
            edges.append(edge)
    return (
        np.asarray(rows, dtype=np.int32).reshape(-1, 2),
        np.asarray(edges, dtype=np.int32).reshape(-1, 2),
    )


def _load_dense_view(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        required = {
            "triangle_id",
            "barycentric_uv",
            "surface_xyz_source_world",
            "surface_xyz_canonical",
            "pixel_linear_index",
        }
        if not required.issubset(set(z.files)):
            raise ValueError("TRUTH139_RELATIONS_DENSE_VIEW_FIELDS_MISSING")
        value = {name: np.asarray(z[name]) for name in required}
    n = len(value["triangle_id"])
    if not (
        value["triangle_id"].shape == (n,)
        and value["barycentric_uv"].shape == (n, 2)
        and value["surface_xyz_source_world"].shape == (n, 3)
        and value["surface_xyz_canonical"].shape == (n, 3)
        and value["pixel_linear_index"].shape == (n,)
    ):
        raise ValueError("TRUTH139_RELATIONS_DENSE_VIEW_LAYOUT")
    if not (
        np.isfinite(value["barycentric_uv"]).all()
        and np.isfinite(value["surface_xyz_source_world"]).all()
        and np.isfinite(value["surface_xyz_canonical"]).all()
    ):
        raise ValueError("TRUTH139_RELATIONS_DENSE_VIEW_NONFINITE")
    return value


def _load_response(path: Path, face_count: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        required = {
            "response_pair_face",
            "response_pair_coherence",
            "response_pair_jump",
            "response_pair_valid",
        }
        if not required.issubset(set(z.files)):
            raise ValueError("TRUTH139_RELATIONS_RESPONSE_FIELDS_MISSING")
        pair = np.asarray(z["response_pair_face"], dtype=np.int64)
        coherence = np.asarray(z["response_pair_coherence"], dtype=np.float64).reshape(-1)
        jump = np.asarray(z["response_pair_jump"], dtype=np.float64).reshape(-1)
        valid = np.asarray(z["response_pair_valid"], dtype=np.bool_).reshape(-1)
    if pair.ndim != 2 or pair.shape[1:] != (2,) or not (
        len(pair) == len(coherence) == len(jump) == len(valid)
    ):
        raise ValueError("TRUTH139_RELATIONS_RESPONSE_LAYOUT")
    if len(pair) and (np.any(pair < 0) or np.any(pair >= face_count)):
        raise ValueError("TRUTH139_RELATIONS_RESPONSE_FACE_RANGE")
    if not np.isfinite(coherence).all() or not np.isfinite(jump).all():
        raise ValueError("TRUTH139_RELATIONS_RESPONSE_NONFINITE")
    if np.any(coherence[valid] < -1.000001) or np.any(coherence[valid] > 1.000001):
        raise ValueError("TRUTH139_RELATIONS_RESPONSE_COHERENCE_RANGE")
    if np.any(jump[valid] < -1.0e-8):
        raise ValueError("TRUTH139_RELATIONS_RESPONSE_JUMP_RANGE")
    return {
        "pair": pair.astype(np.int32),
        "coherence": coherence.astype(np.float32),
        "jump": jump.astype(np.float32),
        "valid": valid,
    }


def build(
    *,
    dense_graphics_teacher_manifest: Path,
    output_dir: Path,
    response_arrays: Path | None = None,
+) -> dict:
    manifest_path = dense_graphics_teacher_manifest.resolve()
    output_dir = output_dir.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError("TRUTH139_RELATIONS_GRAPHICS_MANIFEST_MISSING")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != GRAPHICS_SCHEMA or manifest.get("status") != GRAPHICS_STATUS:
        raise ValueError("TRUTH139_RELATIONS_GRAPHICS_PARENT_NOT_CANDIDATE")
    subject_id = str(manifest.get("subject_id") or "")
    if not subject_id:
        raise ValueError("TRUTH139_RELATIONS_SUBJECT_ID")

    parents = dict(manifest.get("parents") or {})
    geometry_path = _bound_path(
        dict(parents.get("geometry_npz") or {}),
        "TRUTH139_RELATIONS_GEOMETRY_DRIFT",
    )
    visibility_ref = dict(manifest.get("visibility_occlusion") or {})
    visibility_path = _bound_path(
        {"path": visibility_ref.get("path"), "sha256": visibility_ref.get("sha256")},
        "TRUTH139_RELATIONS_VISIBILITY_DRIFT",
    )
    surface_tolerance = float(visibility_ref.get("surface_tolerance", 0.0))
    if not np.isfinite(surface_tolerance) or surface_tolerance <= 0:
        raise ValueError("TRUTH139_RELATIONS_SURFACE_TOLERANCE")

    with np.load(geometry_path, allow_pickle=False) as z:
        faces = np.asarray(z["faces"], dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1:] != (3,) or len(faces) == 0:
        raise ValueError("TRUTH139_RELATIONS_GEOMETRY_FACE_LAYOUT")
    face_count = int(len(faces))
    adjacency_pair, adjacency_edge = exact_face_edge_adjacency(faces)

    dense_meta = sorted(
        list(manifest.get("dense_views") or ()), key=lambda row: int(row.get("view_index", -1))
    )
    if len(dense_meta) != 8 or [int(row.get("view_index", -1)) for row in dense_meta] != list(range(8)):
        raise ValueError("TRUTH139_RELATIONS_DENSE_VIEW_COUNT")
    dense: list[dict[str, np.ndarray]] = []
    for row in dense_meta:
        path = _bound_path(
            {"path": row.get("teacher_path"), "sha256": row.get("teacher_sha256")},
            "TRUTH139_RELATIONS_DENSE_VIEW_DRIFT",
        )
        dense.append(_load_dense_view(path))

    with np.load(visibility_path, allow_pickle=False) as z:
        required = {
            "source_view_index",
            "source_surface_row",
            "source_triangle_id",
            "source_barycentric_uv",
            "target_visibility_class",
            "target_front_row",
            "target_depth_gap",
            "cross_view_same_surface_match",
            "cross_view_match_row",
            "cross_view_match_surface_error",
        }
        if not required.issubset(set(z.files)):
            raise ValueError("TRUTH139_RELATIONS_VISIBILITY_FIELDS_MISSING")
        source_view = np.asarray(z["source_view_index"], dtype=np.int64).reshape(-1)
        source_surface_row = np.asarray(z["source_surface_row"], dtype=np.int64).reshape(-1)
        source_triangle = np.asarray(z["source_triangle_id"], dtype=np.int64).reshape(-1)
        source_bary = np.asarray(z["source_barycentric_uv"], dtype=np.float32)
        visibility = np.asarray(z["target_visibility_class"], dtype=np.uint8)
        front_row = np.asarray(z["target_front_row"], dtype=np.int64)
        depth_gap = np.asarray(z["target_depth_gap"], dtype=np.float32)
        same_match = np.asarray(z["cross_view_same_surface_match"], dtype=np.bool_)
        match_row = np.asarray(z["cross_view_match_row"], dtype=np.int64)
        match_error = np.asarray(z["cross_view_match_surface_error"], dtype=np.float32)

    sample_count = len(source_view)
    if not (
        source_surface_row.shape == source_triangle.shape == (sample_count,)
        and source_bary.shape == (sample_count, 2)
        and visibility.shape == front_row.shape == depth_gap.shape == same_match.shape == match_row.shape == match_error.shape == (sample_count, 8)
    ):
        raise ValueError("TRUTH139_RELATIONS_VISIBILITY_LAYOUT")
    if np.any(source_view < 0) or np.any(source_view >= 8):
        raise ValueError("TRUTH139_RELATIONS_SOURCE_VIEW_RANGE")

    same_source_view: list[int] = []
    same_sample: list[int] = []
    same_source_row: list[int] = []
    same_target_view: list[int] = []
    same_target_row: list[int] = []
    same_source_triangle: list[int] = []
    same_source_bary: list[np.ndarray] = []
    same_target_triangle: list[int] = []
    same_target_bary: list[np.ndarray] = []
    same_error: list[float] = []

    behind_source_view: list[int] = []
    behind_sample: list[int] = []
    behind_source_row: list[int] = []
    behind_target_view: list[int] = []
    behind_front_row: list[int] = []
    behind_triangle: list[int] = []
    behind_bary: list[np.ndarray] = []
    front_triangle: list[int] = []
    front_bary: list[np.ndarray] = []
    behind_depth_gap: list[float] = []

    for sample in range(sample_count):
        sv = int(source_view[sample])
        sr = int(source_surface_row[sample])
        if sr < 0 or sr >= len(dense[sv]["triangle_id"]):
            raise ValueError("TRUTH139_RELATIONS_SOURCE_ROW_RANGE")
        if int(source_triangle[sample]) != int(dense[sv]["triangle_id"][sr]):
            raise ValueError("TRUTH139_RELATIONS_SOURCE_TRIANGLE_BINDING_DRIFT")
        if not np.allclose(source_bary[sample], dense[sv]["barycentric_uv"][sr], atol=1.0e-6, rtol=0.0):
            raise ValueError("TRUTH139_RELATIONS_SOURCE_BARY_BINDING_DRIFT")

        for tv in range(8):
            cls = int(visibility[sample, tv])
            mr = int(match_row[sample, tv])
            if bool(same_match[sample, tv]):
                if cls != VISIBLE or mr < 0 or mr >= len(dense[tv]["triangle_id"]):
                    raise ValueError("TRUTH139_RELATIONS_SAME_SURFACE_PARENT_CONTRADICTION")
                err = float(match_error[sample, tv])
                if not np.isfinite(err) or err > surface_tolerance:
                    raise ValueError("TRUTH139_RELATIONS_SAME_SURFACE_ERROR")
                same_source_view.append(sv)
                same_sample.append(sample)
                same_source_row.append(sr)
                same_target_view.append(tv)
                same_target_row.append(mr)
                same_source_triangle.append(int(source_triangle[sample]))
                same_source_bary.append(source_bary[sample].copy())
                same_target_triangle.append(int(dense[tv]["triangle_id"][mr]))
                same_target_bary.append(dense[tv]["barycentric_uv"][mr].astype(np.float32, copy=True))
                same_error.append(err)

            if cls == OCCLUDED:
                fr = int(front_row[sample, tv])
                gap = float(depth_gap[sample, tv])
                if fr < 0 or fr >= len(dense[tv]["triangle_id"]):
                    raise ValueError("TRUTH139_RELATIONS_FRONT_ROW_RANGE")
                if not np.isfinite(gap) or gap <= 0:
                    raise ValueError("TRUTH139_RELATIONS_OCCLUSION_GAP_NONPOSITIVE")
                behind_source_view.append(sv)
                behind_sample.append(sample)
                behind_source_row.append(sr)
                behind_target_view.append(tv)
                behind_front_row.append(fr)
                behind_triangle.append(int(source_triangle[sample]))
                behind_bary.append(source_bary[sample].copy())
                front_triangle.append(int(dense[tv]["triangle_id"][fr]))
                front_bary.append(dense[tv]["barycentric_uv"][fr].astype(np.float32, copy=True))
                behind_depth_gap.append(gap)

    def bary_array(rows: list[np.ndarray]) -> np.ndarray:
        return np.asarray(rows, dtype=np.float32).reshape(-1, 2)

    response = None
    response_parent = None
    if response_arrays is not None:
        response_path = response_arrays.resolve()
        if not response_path.is_file():
            raise FileNotFoundError("TRUTH139_RELATIONS_RESPONSE_PARENT_MISSING")
        response = _load_response(response_path, face_count)
        response_parent = {"path": str(response_path), "sha256": sha256(response_path)}

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "RELATIONS_TEACHER.npz"
    arrays = {
        "adjacent_face_pair": adjacency_pair,
        "adjacent_shared_edge_vertex": adjacency_edge,
        "same_source_view_index": np.asarray(same_source_view, dtype=np.int16),
        "same_source_sample_index": np.asarray(same_sample, dtype=np.int32),
        "same_source_surface_row": np.asarray(same_source_row, dtype=np.int32),
        "same_target_view_index": np.asarray(same_target_view, dtype=np.int16),
        "same_target_surface_row": np.asarray(same_target_row, dtype=np.int32),
        "same_source_triangle_id": np.asarray(same_source_triangle, dtype=np.int32),
        "same_source_barycentric_uv": bary_array(same_source_bary),
        "same_target_triangle_id": np.asarray(same_target_triangle, dtype=np.int32),
        "same_target_barycentric_uv": bary_array(same_target_bary),
        "same_canonical_surface_error": np.asarray(same_error, dtype=np.float32),
        "behind_source_view_index": np.asarray(behind_source_view, dtype=np.int16),
        "behind_source_sample_index": np.asarray(behind_sample, dtype=np.int32),
        "behind_source_surface_row": np.asarray(behind_source_row, dtype=np.int32),
        "behind_target_view_index": np.asarray(behind_target_view, dtype=np.int16),
        "front_target_surface_row": np.asarray(behind_front_row, dtype=np.int32),
        "behind_triangle_id": np.asarray(behind_triangle, dtype=np.int32),
        "behind_barycentric_uv": bary_array(behind_bary),
        "front_triangle_id": np.asarray(front_triangle, dtype=np.int32),
        "front_barycentric_uv": bary_array(front_bary),
        "front_behind_depth_gap": np.asarray(behind_depth_gap, dtype=np.float32),
    }
    if response is not None:
        arrays.update(
            response_pair_face=response["pair"],
            response_pair_coherence=response["coherence"],
            response_pair_jump=response["jump"],
            response_pair_valid=response["valid"],
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
            "graphics_teacher_identity_sha256": str(manifest.get("graphics_teacher_identity_sha256") or ""),
            "geometry_npz": {"path": str(geometry_path), "sha256": sha256(geometry_path)},
            "visibility_occlusion_teacher": {"path": str(visibility_path), "sha256": sha256(visibility_path)},
            "stage47_response_arrays": response_parent,
        },
        "relations_npz": {"path": str(out), "sha256": sha256(out)},
        "counts": {
            "local_surface_adjacency": int(len(adjacency_pair)),
            "same_canonical_surface_across_views": int(len(same_sample)),
            "front_behind_on_shared_camera_ray": int(len(behind_sample)),
            "response_pair_total": 0 if response is None else int(len(response["pair"])),
            "response_pair_valid": 0 if response is None else int(np.count_nonzero(response["valid"])),
        },
        "policy": {
            "semantic_part_taxonomy_used": False,
            "owner_material_or_bone_name_targets_used": False,
            "global_total_draw_order_claimed": False,
            "absence_is_negative_label": False,
            "response_relations_are_optional_measured_evidence": True,
            "relations_are_axis_or_mira_authority": False,
            "training_performed": False,
        },
        "qualification_minted": False,
    }
    payload["relations_teacher_identity_sha256"] = _digest(payload)
    manifest_out = output_dir / "RELATIONS_TEACHER_MANIFEST.json"
    manifest_out.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "subject_id": subject_id,
                "counts": payload["counts"],
                "relations_teacher_identity_sha256": payload["relations_teacher_identity_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-graphics-teacher-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--response-arrays", type=Path)
    args = parser.parse_args()
    build(
        dense_graphics_teacher_manifest=args.dense_graphics_teacher_manifest,
        output_dir=args.output_dir,
        response_arrays=args.response_arrays,
    )


if __name__ == "__main__":
    main()
