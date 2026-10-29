#!/usr/bin/env python3
"""Validate Truth139 primitive Relations V2 by exact rebinding to canonical V2 parents."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from tools.corpus.build_truth139_dense_graphics_gbuffer_v1 import sha256

SCHEMA = "RealSaS.Truth139RelationsTeacherCourt.v2"
PARENT_SCHEMA = "RealSaS.Truth139RelationsTeacher.v1"
PARENT_STATUS = "PASS_RELATIONS_TEACHER_CANDIDATE__INDEPENDENT_COURT_REQUIRED"
PARENT_REVISION = "V2_DENSE_GBUFFER_VISIBILITY_V2_REBIND"
GRAPHICS_SCHEMA = "RealSaS.Truth139DenseGraphicsGBuffer.v1"
VIS_SCHEMA = "RealSaS.Truth139VisibilityOcclusionTeacher.v1"
PASS_STATUS = "PASS_RELATIONS_V2_REBINDS_CANONICAL_DENSE_AND_VISIBILITY_PARENT_TRUTH"


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _bound_path(ref: Mapping[str, object], code: str) -> Path:
    path = Path(str(ref.get("path") or "")).resolve()
    expected = str(ref.get("sha256") or "")
    if not path.is_file() or len(expected) != 64 or sha256(path) != expected:
        raise ValueError(code)
    return path


def _adjacency(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f = np.asarray(faces, dtype=np.int64)
    if f.ndim != 2 or f.shape[1:] != (3,) or len(f) == 0:
        raise ValueError("TRUTH139_RELATIONS_V2_COURT_FACE_LAYOUT")
    edge_owners: dict[tuple[int, int], set[int]] = {}
    for fi, tri in enumerate(f.tolist()):
        if len(set(map(int, tri))) != 3:
            raise ValueError("TRUTH139_RELATIONS_V2_COURT_DEGENERATE_FACE")
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge_owners.setdefault(tuple(sorted((int(a), int(b)))), set()).add(int(fi))
    pairs, edges = [], []
    for edge in sorted(edge_owners):
        for a, b in itertools.combinations(sorted(edge_owners[edge]), 2):
            pairs.append((a, b))
            edges.append(edge)
    return np.asarray(pairs, dtype=np.int32).reshape(-1, 2), np.asarray(edges, dtype=np.int32).reshape(-1, 2)


def _dense(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        pixel = np.asarray(z["pixel_linear_index"], dtype=np.int32).reshape(-1)
        tri = np.asarray(z["triangle_id"], dtype=np.int32).reshape(-1)
        bary = np.asarray(z["barycentric_uv"], dtype=np.float32)
        canonical = np.asarray(z["surface_xyz_canonical"], dtype=np.float32)
    if bary.shape != (len(pixel), 2) or tri.shape != (len(pixel),) or canonical.shape != (len(pixel), 3):
        raise ValueError("TRUTH139_RELATIONS_V2_COURT_DENSE_LAYOUT")
    return {"pixel": pixel, "triangle": tri, "bary": bary, "canonical": canonical}


def _lookup(pixel: np.ndarray, resolution: int) -> np.ndarray:
    out = np.full(resolution * resolution, -1, dtype=np.int32)
    if np.any(pixel < 0) or np.any(pixel >= len(out)) or len(np.unique(pixel)) != len(pixel):
        raise ValueError("TRUTH139_RELATIONS_V2_COURT_DENSE_PIXEL")
    out[pixel] = np.arange(len(pixel), dtype=np.int32)
    return out


def _equal(code: str, got: np.ndarray, expected: np.ndarray) -> None:
    if got.shape != expected.shape or not np.array_equal(got, expected):
        raise ValueError("TRUTH139_RELATIONS_V2_COURT_" + code + "_DRIFT")


def _close(code: str, got: np.ndarray, expected: np.ndarray, atol: float = 1.0e-6) -> None:
    if got.shape != expected.shape or not np.allclose(got, expected, atol=atol, rtol=0.0, equal_nan=True):
        raise ValueError("TRUTH139_RELATIONS_V2_COURT_" + code + "_DRIFT")


def validate(manifest: dict) -> dict:
    if manifest.get("schema") != PARENT_SCHEMA or manifest.get("status") != PARENT_STATUS:
        raise ValueError("TRUTH139_RELATIONS_V2_COURT_PARENT")
    if manifest.get("producer_revision") != PARENT_REVISION:
        raise ValueError("TRUTH139_RELATIONS_V2_COURT_REVISION")
    subject_id = str(manifest.get("subject_id") or "")
    parents = dict(manifest.get("parents") or {})

    graphics_path = _bound_path(dict(parents.get("dense_graphics_gbuffer_manifest") or {}), "TRUTH139_RELATIONS_V2_COURT_GRAPHICS_DRIFT")
    visibility_path = _bound_path(dict(parents.get("visibility_occlusion_manifest") or {}), "TRUTH139_RELATIONS_V2_COURT_VIS_DRIFT")
    geometry_path = _bound_path(dict(parents.get("geometry_npz") or {}), "TRUTH139_RELATIONS_V2_COURT_GEOMETRY_DRIFT")
    relations_path = _bound_path(dict(manifest.get("relations_npz") or {}), "TRUTH139_RELATIONS_V2_COURT_RELATIONS_DRIFT")
    dense_court_path = _bound_path(dict(parents.get("dense_graphics_geometry_court") or {}), "TRUTH139_RELATIONS_V2_COURT_DENSE_COURT_DRIFT")
    vis_court_path = _bound_path(dict(parents.get("visibility_occlusion_court") or {}), "TRUTH139_RELATIONS_V2_COURT_VIS_COURT_DRIFT")

    graphics = json.loads(graphics_path.read_text())
    visibility = json.loads(visibility_path.read_text())
    dense_court = json.loads(dense_court_path.read_text())
    vis_court = json.loads(vis_court_path.read_text())
    if graphics.get("schema") != GRAPHICS_SCHEMA or visibility.get("schema") != VIS_SCHEMA:
        raise ValueError("TRUTH139_RELATIONS_V2_COURT_PARENT_SCHEMA")
    if any(str(x.get("subject_id") or "") != subject_id for x in (graphics, visibility, dense_court, vis_court)):
        raise ValueError("TRUTH139_RELATIONS_V2_COURT_SUBJECT")
    if str(dense_court.get("status") or "") != "PASS_INDEPENDENT_CPU_RAY_ORACLE_AGREES_WITH_DENSE_GPU_GBUFFER":
        raise ValueError("TRUTH139_RELATIONS_V2_COURT_DENSE_NOT_PASS")
    if str(vis_court.get("status") or "") != "PASS_INDEPENDENT_CPU_RAY_ORACLE_AGREES_WITH_VISIBILITY_OCCLUSION_TEACHER":
        raise ValueError("TRUTH139_RELATIONS_V2_COURT_VIS_NOT_PASS")

    with np.load(geometry_path, allow_pickle=False) as z:
        faces = np.asarray(z["faces"], dtype=np.int64)
    expected_adj, expected_edges = _adjacency(faces)

    views = sorted(list(graphics.get("views") or ()), key=lambda r: int(r.get("view_index", -1)))
    if len(views) != 8:
        raise ValueError("TRUTH139_RELATIONS_V2_COURT_VIEW_COUNT")
    dense = [_dense(_bound_path({"path": r.get("gbuffer_path"), "sha256": r.get("gbuffer_sha256")}, "TRUTH139_RELATIONS_V2_COURT_DENSE_VIEW_DRIFT")) for r in views]
    camera_path = _bound_path(dict(graphics.get("camera_bundle") or {}), "TRUTH139_RELATIONS_V2_COURT_CAMERA_DRIFT")
    cameras = json.loads(camera_path.read_text()).get("cameras") or []
    resolution = int(cameras[0].get("resolution", 0)) if len(cameras) == 8 else 0
    if resolution <= 0 or any(int(c.get("resolution", 0)) != resolution for c in cameras):
        raise ValueError("TRUTH139_RELATIONS_V2_COURT_RESOLUTION")
    lookups = [_lookup(v["pixel"], resolution) for v in dense]

    teacher_npz = _bound_path(dict(visibility.get("teacher_npz") or {}), "TRUTH139_RELATIONS_V2_COURT_VIS_TEACHER_DRIFT")
    with np.load(teacher_npz, allow_pickle=False) as z:
        sv = np.asarray(z["source_view_index"], dtype=np.int16).reshape(-1)
        sr = np.asarray(z["source_surface_row"], dtype=np.int32).reshape(-1)
        cls = np.asarray(z["target_visibility_class"], dtype=np.uint8)
        pix = np.asarray(z["target_projected_pixel_linear_index"], dtype=np.int32)
        front_tri = np.asarray(z["target_front_triangle_id"], dtype=np.int32)
        gap = np.asarray(z["target_depth_gap"], dtype=np.float32)
        same = np.asarray(z["cross_view_same_surface_match"], dtype=np.bool_)
    n = len(sv)
    if not (sr.shape == (n,) and cls.shape == pix.shape == front_tri.shape == gap.shape == same.shape == (n, 8)):
        raise ValueError("TRUTH139_RELATIONS_V2_COURT_VIS_LAYOUT")

    codebook = {str(name): int(code) for code, name in (visibility.get("visibility_class_codebook") or {}).items()}
    visible = codebook.get("VISIBLE_SAME_SURFACE")
    occluded = codebook.get("OCCLUDED_BY_FRONT_SURFACE")
    if visible is None or occluded is None:
        raise ValueError("TRUTH139_RELATIONS_V2_COURT_CODEBOOK")

    same_rows, behind_rows = [], []
    source_tri = np.empty(n, dtype=np.int32)
    source_bary = np.empty((n, 2), dtype=np.float32)
    for i in range(n):
        sview, srow = int(sv[i]), int(sr[i])
        if sview < 0 or sview >= 8 or srow < 0 or srow >= len(dense[sview]["triangle"]):
            raise ValueError("TRUTH139_RELATIONS_V2_COURT_SOURCE_RANGE")
        source_tri[i] = dense[sview]["triangle"][srow]
        source_bary[i] = dense[sview]["bary"][srow]
        for tv in range(8):
            if bool(same[i, tv]):
                if int(cls[i, tv]) != int(visible):
                    raise ValueError("TRUTH139_RELATIONS_V2_COURT_SAME_CLASS")
                p = int(pix[i, tv])
                tr = int(lookups[tv][p]) if 0 <= p < resolution * resolution else -1
                if tr < 0 or int(dense[tv]["triangle"][tr]) != int(front_tri[i, tv]):
                    raise ValueError("TRUTH139_RELATIONS_V2_COURT_SAME_BINDING")
                err = float(np.linalg.norm(dense[tv]["canonical"][tr].astype(np.float64) - dense[sview]["canonical"][srow].astype(np.float64)))
                same_rows.append((sview, i, srow, tv, tr, err))
            if int(cls[i, tv]) == int(occluded):
                p = int(pix[i, tv])
                fr = int(lookups[tv][p]) if 0 <= p < resolution * resolution else -1
                dg = float(gap[i, tv])
                if fr < 0 or not np.isfinite(dg) or dg <= 0 or int(dense[tv]["triangle"][fr]) != int(front_tri[i, tv]):
                    raise ValueError("TRUTH139_RELATIONS_V2_COURT_BEHIND_BINDING")
                behind_rows.append((sview, i, srow, tv, fr, dg))

    same_expected = np.asarray(same_rows, dtype=np.float64).reshape(-1, 6)
    behind_expected = np.asarray(behind_rows, dtype=np.float64).reshape(-1, 6)

    with np.load(relations_path, allow_pickle=False) as z:
        files = set(z.files)
        _equal("ADJ", np.asarray(z["adjacent_face_pair"], dtype=np.int32), expected_adj)
        _equal("ADJ_EDGE", np.asarray(z["adjacent_shared_edge_vertex"], dtype=np.int32), expected_edges)
        sc = len(same_expected)
        _equal("SAME_SV", np.asarray(z["same_source_view_index"], dtype=np.int16), same_expected[:, 0].astype(np.int16))
        _equal("SAME_SAMPLE", np.asarray(z["same_source_sample_index"], dtype=np.int32), same_expected[:, 1].astype(np.int32))
        _equal("SAME_SR", np.asarray(z["same_source_surface_row"], dtype=np.int32), same_expected[:, 2].astype(np.int32))
        _equal("SAME_TV", np.asarray(z["same_target_view_index"], dtype=np.int16), same_expected[:, 3].astype(np.int16))
        _equal("SAME_TR", np.asarray(z["same_target_surface_row"], dtype=np.int32), same_expected[:, 4].astype(np.int32))
        same_samples = same_expected[:, 1].astype(np.int64) if sc else np.empty(0, dtype=np.int64)
        same_tv = same_expected[:, 3].astype(np.int64) if sc else np.empty(0, dtype=np.int64)
        same_tr = same_expected[:, 4].astype(np.int64) if sc else np.empty(0, dtype=np.int64)
        _equal("SAME_SOURCE_TRI", np.asarray(z["same_source_triangle_id"], dtype=np.int32), source_tri[same_samples])
        _close("SAME_SOURCE_BARY", np.asarray(z["same_source_barycentric_uv"], dtype=np.float32), source_bary[same_samples])
        _equal("SAME_TARGET_TRI", np.asarray(z["same_target_triangle_id"], dtype=np.int32), np.asarray([dense[tv]["triangle"][tr] for tv, tr in zip(same_tv, same_tr)], dtype=np.int32))
        _close("SAME_TARGET_BARY", np.asarray(z["same_target_barycentric_uv"], dtype=np.float32), np.asarray([dense[tv]["bary"][tr] for tv, tr in zip(same_tv, same_tr)], dtype=np.float32).reshape(-1, 2))
        _close("SAME_ERR", np.asarray(z["same_canonical_surface_error"], dtype=np.float32), same_expected[:, 5].astype(np.float32), atol=2.5e-5)

        bc = len(behind_expected)
        _equal("BEHIND_SV", np.asarray(z["behind_source_view_index"], dtype=np.int16), behind_expected[:, 0].astype(np.int16))
        _equal("BEHIND_SAMPLE", np.asarray(z["behind_source_sample_index"], dtype=np.int32), behind_expected[:, 1].astype(np.int32))
        _equal("BEHIND_SR", np.asarray(z["behind_source_surface_row"], dtype=np.int32), behind_expected[:, 2].astype(np.int32))
        _equal("BEHIND_TV", np.asarray(z["behind_target_view_index"], dtype=np.int16), behind_expected[:, 3].astype(np.int16))
        _equal("BEHIND_FR", np.asarray(z["front_target_surface_row"], dtype=np.int32), behind_expected[:, 4].astype(np.int32))
        behind_samples = behind_expected[:, 1].astype(np.int64) if bc else np.empty(0, dtype=np.int64)
        behind_tv = behind_expected[:, 3].astype(np.int64) if bc else np.empty(0, dtype=np.int64)
        behind_fr = behind_expected[:, 4].astype(np.int64) if bc else np.empty(0, dtype=np.int64)
        _equal("BEHIND_TRI", np.asarray(z["behind_triangle_id"], dtype=np.int32), source_tri[behind_samples])
        _close("BEHIND_BARY", np.asarray(z["behind_barycentric_uv"], dtype=np.float32), source_bary[behind_samples])
        _equal("FRONT_TRI", np.asarray(z["front_triangle_id"], dtype=np.int32), np.asarray([dense[tv]["triangle"][fr] for tv, fr in zip(behind_tv, behind_fr)], dtype=np.int32))
        _close("FRONT_BARY", np.asarray(z["front_barycentric_uv"], dtype=np.float32), np.asarray([dense[tv]["bary"][fr] for tv, fr in zip(behind_tv, behind_fr)], dtype=np.float32).reshape(-1, 2))
        _close("GAP", np.asarray(z["front_behind_depth_gap"], dtype=np.float32), behind_expected[:, 5].astype(np.float32))

        response_names = {"response_pair_face", "response_pair_coherence", "response_pair_jump", "response_pair_valid"}
        response_parent = parents.get("stage47_response_arrays")
        if response_parent is None:
            if files.intersection(response_names):
                raise ValueError("TRUTH139_RELATIONS_V2_COURT_UNBOUND_RESPONSE")
            response_count = response_valid = 0
        else:
            rp = _bound_path(dict(response_parent), "TRUTH139_RELATIONS_V2_COURT_RESPONSE_DRIFT")
            if not response_names.issubset(files):
                raise ValueError("TRUTH139_RELATIONS_V2_COURT_RESPONSE_MISSING")
            with np.load(rp, allow_pickle=False) as p:
                for name in response_names:
                    _equal("RESPONSE_" + name.upper(), np.asarray(z[name]), np.asarray(p[name]))
                response_count = int(len(np.asarray(p["response_pair_face"])))
                response_valid = int(np.count_nonzero(np.asarray(p["response_pair_valid"], dtype=np.bool_)))

    payload = {"schema": SCHEMA, "date": "2026-10-29", "status": PASS_STATUS, "subject_id": subject_id, "relations_teacher_identity_sha256": str(manifest.get("relations_teacher_identity_sha256") or ""), "producer_revision": str(manifest.get("producer_revision") or ""), "validated_counts": {"local_surface_adjacency": int(len(expected_adj)), "same_canonical_surface_across_views": int(len(same_rows)), "front_behind_on_shared_camera_ray": int(len(behind_rows)), "response_pair_total": int(response_count), "response_pair_valid": int(response_valid)}, "policy": {"mesh_adjacency_recomputed": True, "canonical_dense_pixel_rebinding_recomputed": True, "visibility_physics_owned_by_separate_cpu_ray_oracle_court": True, "response_rows_exact_hash_bound_copy_only": True, "legacy_dense_graphics_teacher_authority_used": False, "training_performed": False}, "qualification_minted": False}
    payload["court_identity_sha256"] = _digest(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relations-teacher-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = validate(json.loads(args.relations_teacher_manifest.resolve().read_text()))
    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    staged = out.with_suffix(out.suffix + ".partial")
    staged.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    staged.replace(out)
    print(json.dumps({"status": result["status"], "subject_id": result["subject_id"], "validated_counts": result["validated_counts"], "court_identity_sha256": result["court_identity_sha256"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
