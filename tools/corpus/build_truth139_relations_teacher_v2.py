#!/usr/bin/env python3
"""Build primitive Truth139 Relations from canonical Dense G-buffer + Visibility/Occlusion V2.

This is a parent-schema rebind of the existing primitive Relations truth. It consumes the
court-validated canonical V2 Graphics parents directly; it never recreates the legacy aggregate
DenseGraphicsTeacher.v1 authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from tools.corpus.build_truth139_dense_graphics_gbuffer_v1 import sha256
from tools.corpus.build_truth139_relations_teacher_v1 import exact_face_edge_adjacency

SCHEMA = "RealSaS.Truth139RelationsTeacher.v1"
STATUS = "PASS_RELATIONS_TEACHER_CANDIDATE__INDEPENDENT_COURT_REQUIRED"
PRODUCER_REVISION = "V2_DENSE_GBUFFER_VISIBILITY_V2_REBIND"
GRAPHICS_SCHEMA = "RealSaS.Truth139DenseGraphicsGBuffer.v1"
GRAPHICS_STATUS = "PASS_DENSE_GRAPHICS_FRONT_SURFACE__VISIBILITY_OCCLUSION_COURT_PENDING"
DENSE_COURT_SCHEMA = "RealSaS.Truth139DenseGraphicsGeometryCourt.v1"
DENSE_COURT_STATUS = "PASS_INDEPENDENT_CPU_RAY_ORACLE_AGREES_WITH_DENSE_GPU_GBUFFER"
VIS_SCHEMA = "RealSaS.Truth139VisibilityOcclusionTeacher.v1"
VIS_STATUS = "PASS_VISIBILITY_OCCLUSION_TEACHER__INDEPENDENT_CPU_ORACLE_COURT_PENDING"
VIS_COURT_SCHEMA = "RealSaS.Truth139VisibilityOcclusionCourt.v1"
VIS_COURT_STATUS = "PASS_INDEPENDENT_CPU_RAY_ORACLE_AGREES_WITH_VISIBILITY_OCCLUSION_TEACHER"
CONTRACT = "canonical/TRUTH139_RELATIONS_TEACHER_V1.json"


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _bound_path(ref: Mapping[str, object], code: str) -> Path:
    path = Path(str(ref.get("path") or "")).resolve()
    expected = str(ref.get("sha256") or "")
    if not path.is_file() or len(expected) != 64 or sha256(path) != expected:
        raise ValueError(code)
    return path


def _write_npz(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(path.suffix + ".partial.npz")
    np.savez_compressed(staged, **arrays)
    staged.replace(path)


def _load_dense(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        required = {"pixel_linear_index", "triangle_id", "barycentric_uv", "surface_xyz_source_world", "surface_xyz_canonical"}
        if not required.issubset(z.files):
            raise ValueError("TRUTH139_RELATIONS_V2_DENSE_FIELDS")
        pix = np.asarray(z["pixel_linear_index"], dtype=np.int32).reshape(-1)
        tri = np.asarray(z["triangle_id"], dtype=np.int32).reshape(-1)
        bary = np.asarray(z["barycentric_uv"], dtype=np.float32)
        source = np.asarray(z["surface_xyz_source_world"], dtype=np.float32)
        canonical = np.asarray(z["surface_xyz_canonical"], dtype=np.float32)
    n = len(pix)
    if tri.shape != (n,) or bary.shape != (n, 2) or source.shape != (n, 3) or canonical.shape != (n, 3):
        raise ValueError("TRUTH139_RELATIONS_V2_DENSE_LAYOUT")
    if len(np.unique(pix)) != n or np.any(pix < 0):
        raise ValueError("TRUTH139_RELATIONS_V2_DENSE_PIXEL_UNIQUE")
    if not np.isfinite(bary).all() or not np.isfinite(source).all() or not np.isfinite(canonical).all():
        raise ValueError("TRUTH139_RELATIONS_V2_DENSE_NONFINITE")
    return {"pixel": pix, "triangle": tri, "bary": bary, "source": source, "canonical": canonical}


def _row_lookup(pixel: np.ndarray, resolution: int) -> np.ndarray:
    out = np.full(resolution * resolution, -1, dtype=np.int32)
    if np.any(pixel >= len(out)):
        raise ValueError("TRUTH139_RELATIONS_V2_DENSE_PIXEL_RANGE")
    out[pixel] = np.arange(len(pixel), dtype=np.int32)
    return out


def _load_response(path: Path, face_count: int) -> dict[str, np.ndarray]:
    names = ("response_pair_face", "response_pair_coherence", "response_pair_jump", "response_pair_valid")
    with np.load(path, allow_pickle=False) as z:
        if not set(names).issubset(z.files):
            raise ValueError("TRUTH139_RELATIONS_V2_RESPONSE_FIELDS")
        pair = np.asarray(z[names[0]], dtype=np.int32)
        coherence = np.asarray(z[names[1]], dtype=np.float32).reshape(-1)
        jump = np.asarray(z[names[2]], dtype=np.float32).reshape(-1)
        valid = np.asarray(z[names[3]], dtype=np.bool_).reshape(-1)
    if pair.ndim != 2 or pair.shape[1:] != (2,) or not (len(pair) == len(coherence) == len(jump) == len(valid)):
        raise ValueError("TRUTH139_RELATIONS_V2_RESPONSE_LAYOUT")
    if len(pair) and (np.any(pair < 0) or np.any(pair >= face_count)):
        raise ValueError("TRUTH139_RELATIONS_V2_RESPONSE_FACE_RANGE")
    return {"pair": pair, "coherence": coherence, "jump": jump, "valid": valid}


def _validate_parents(graphics_manifest: Path, dense_court: Path, visibility_manifest: Path, visibility_court: Path) -> tuple[dict, dict, dict, dict]:
    gp, dp, vp, cp = graphics_manifest.resolve(), dense_court.resolve(), visibility_manifest.resolve(), visibility_court.resolve()
    for path, code in ((gp, "GRAPHICS"), (dp, "DENSE_COURT"), (vp, "VISIBILITY"), (cp, "VIS_COURT")):
        if not path.is_file():
            raise FileNotFoundError("TRUTH139_RELATIONS_V2_" + code + "_MISSING")
    graphics, dcourt = json.loads(gp.read_text()), json.loads(dp.read_text())
    visibility, vcourt = json.loads(vp.read_text()), json.loads(cp.read_text())
    if graphics.get("schema") != GRAPHICS_SCHEMA or graphics.get("status") != GRAPHICS_STATUS:
        raise ValueError("TRUTH139_RELATIONS_V2_GRAPHICS_PARENT")
    if dcourt.get("schema") != DENSE_COURT_SCHEMA or dcourt.get("status") != DENSE_COURT_STATUS:
        raise ValueError("TRUTH139_RELATIONS_V2_DENSE_COURT")
    if visibility.get("schema") != VIS_SCHEMA or visibility.get("status") != VIS_STATUS:
        raise ValueError("TRUTH139_RELATIONS_V2_VIS_PARENT")
    if vcourt.get("schema") != VIS_COURT_SCHEMA or vcourt.get("status") != VIS_COURT_STATUS:
        raise ValueError("TRUTH139_RELATIONS_V2_VIS_COURT")
    sid = str(graphics.get("subject_id") or "")
    if not sid or any(str(x.get("subject_id") or "") != sid for x in (dcourt, visibility, vcourt)):
        raise ValueError("TRUTH139_RELATIONS_V2_SUBJECT_BINDING")
    gid = str(graphics.get("graphics_identity_sha256") or "")
    if str(dcourt.get("dense_graphics_identity_sha256") or "") != gid:
        raise ValueError("TRUTH139_RELATIONS_V2_DENSE_COURT_IDENTITY")
    vref = dict(visibility.get("graphics_manifest") or {})
    if str(vref.get("sha256") or "") != sha256(gp):
        raise ValueError("TRUTH139_RELATIONS_V2_VIS_GRAPHICS_BINDING")
    if str(vcourt.get("teacher_identity_sha256") or "") != str(visibility.get("teacher_identity_sha256") or ""):
        raise ValueError("TRUTH139_RELATIONS_V2_VIS_COURT_IDENTITY")
    return graphics, dcourt, visibility, vcourt


def build(*, dense_graphics_manifest: Path, dense_graphics_court: Path, visibility_manifest: Path, visibility_court: Path, output_dir: Path, response_arrays: Path | None = None) -> dict:
    graphics, dcourt, visibility, vcourt = _validate_parents(dense_graphics_manifest, dense_graphics_court, visibility_manifest, visibility_court)
    subject_id = str(graphics["subject_id"])
    output_dir = output_dir.resolve()
    geometry_path = _bound_path(dict(graphics.get("geometry_npz") or {}), "TRUTH139_RELATIONS_V2_GEOMETRY_DRIFT")
    with np.load(geometry_path, allow_pickle=False) as z:
        faces = np.asarray(z["faces"], dtype=np.int64)
    adjacent_pair, adjacent_edge = exact_face_edge_adjacency(faces)

    dense_meta = sorted(list(graphics.get("views") or ()), key=lambda row: int(row.get("view_index", -1)))
    if len(dense_meta) != 8 or [int(r.get("view_index", -1)) for r in dense_meta] != list(range(8)):
        raise ValueError("TRUTH139_RELATIONS_V2_DENSE_VIEW_COUNT")
    dense = [_load_dense(_bound_path({"path": row.get("gbuffer_path"), "sha256": row.get("gbuffer_sha256")}, "TRUTH139_RELATIONS_V2_DENSE_VIEW_DRIFT")) for row in dense_meta]
    camera_path = _bound_path(dict(graphics.get("camera_bundle") or {}), "TRUTH139_RELATIONS_V2_CAMERA_DRIFT")
    cameras = json.loads(camera_path.read_text()).get("cameras") or []
    if len(cameras) != 8:
        raise ValueError("TRUTH139_RELATIONS_V2_CAMERA_COUNT")
    resolution = int(cameras[0].get("resolution", 0))
    if resolution <= 0 or any(int(c.get("resolution", 0)) != resolution for c in cameras):
        raise ValueError("TRUTH139_RELATIONS_V2_RESOLUTION")
    lookup = [_row_lookup(v["pixel"], resolution) for v in dense]

    teacher_path = _bound_path(dict(visibility.get("teacher_npz") or {}), "TRUTH139_RELATIONS_V2_VISIBILITY_DRIFT")
    with np.load(teacher_path, allow_pickle=False) as z:
        required = {"source_view_index", "source_surface_row", "surface_xyz_source_world", "surface_xyz_canonical", "target_visibility_class", "target_projected_pixel_linear_index", "target_front_triangle_id", "target_depth_gap", "cross_view_same_surface_match"}
        if not required.issubset(z.files):
            raise ValueError("TRUTH139_RELATIONS_V2_VISIBILITY_FIELDS")
        sv = np.asarray(z["source_view_index"], dtype=np.int16).reshape(-1)
        sr = np.asarray(z["source_surface_row"], dtype=np.int32).reshape(-1)
        sxyz = np.asarray(z["surface_xyz_source_world"], dtype=np.float32)
        scanon = np.asarray(z["surface_xyz_canonical"], dtype=np.float32)
        cls = np.asarray(z["target_visibility_class"], dtype=np.uint8)
        pix = np.asarray(z["target_projected_pixel_linear_index"], dtype=np.int32)
        front_tri = np.asarray(z["target_front_triangle_id"], dtype=np.int32)
        gap = np.asarray(z["target_depth_gap"], dtype=np.float32)
        same = np.asarray(z["cross_view_same_surface_match"], dtype=np.bool_)
    n = len(sv)
    if not (sr.shape == (n,) and sxyz.shape == scanon.shape == (n, 3) and cls.shape == pix.shape == front_tri.shape == gap.shape == same.shape == (n, 8)):
        raise ValueError("TRUTH139_RELATIONS_V2_VISIBILITY_LAYOUT")

    codebook = {str(name): int(code) for code, name in (visibility.get("visibility_class_codebook") or {}).items()}
    visible_code = codebook.get("VISIBLE_SAME_SURFACE")
    occluded_code = codebook.get("OCCLUDED_BY_FRONT_SURFACE")
    if visible_code is None or occluded_code is None:
        raise ValueError("TRUTH139_RELATIONS_V2_VISIBILITY_CODEBOOK")

    same_rows, behind_rows = [], []
    for i in range(n):
        source_view, source_row = int(sv[i]), int(sr[i])
        if source_view < 0 or source_view >= 8 or source_row < 0 or source_row >= len(dense[source_view]["triangle"]):
            raise ValueError("TRUTH139_RELATIONS_V2_SOURCE_RANGE")
        if not np.allclose(sxyz[i], dense[source_view]["source"][source_row], atol=2.5e-5, rtol=0.0):
            raise ValueError("TRUTH139_RELATIONS_V2_SOURCE_POINT_DRIFT")
        if not np.allclose(scanon[i], dense[source_view]["canonical"][source_row], atol=2.5e-5, rtol=0.0):
            raise ValueError("TRUTH139_RELATIONS_V2_SOURCE_CANONICAL_DRIFT")
        for tv in range(8):
            if bool(same[i, tv]):
                if int(cls[i, tv]) != int(visible_code):
                    raise ValueError("TRUTH139_RELATIONS_V2_SAME_CLASS")
                p = int(pix[i, tv])
                if p < 0 or p >= resolution * resolution:
                    raise ValueError("TRUTH139_RELATIONS_V2_SAME_PIXEL")
                tr = int(lookup[tv][p])
                if tr < 0 or int(dense[tv]["triangle"][tr]) != int(front_tri[i, tv]):
                    raise ValueError("TRUTH139_RELATIONS_V2_SAME_FRONT_TRI")
                err = float(np.linalg.norm(dense[tv]["canonical"][tr].astype(np.float64) - scanon[i].astype(np.float64)))
                same_rows.append((source_view, i, source_row, tv, tr, err))
            if int(cls[i, tv]) == int(occluded_code):
                p = int(pix[i, tv])
                if p < 0 or p >= resolution * resolution:
                    raise ValueError("TRUTH139_RELATIONS_V2_BEHIND_PIXEL")
                fr = int(lookup[tv][p])
                dg = float(gap[i, tv])
                if fr < 0 or not np.isfinite(dg) or dg <= 0 or int(dense[tv]["triangle"][fr]) != int(front_tri[i, tv]):
                    raise ValueError("TRUTH139_RELATIONS_V2_BEHIND_PARENT")
                behind_rows.append((source_view, i, source_row, tv, fr, dg))

    same_arr = np.asarray(same_rows, dtype=np.float64).reshape(-1, 6)
    behind_arr = np.asarray(behind_rows, dtype=np.float64).reshape(-1, 6)
    same_sample = same_arr[:, 1].astype(np.int64) if len(same_arr) else np.empty(0, dtype=np.int64)
    same_tv = same_arr[:, 3].astype(np.int64) if len(same_arr) else np.empty(0, dtype=np.int64)
    same_tr = same_arr[:, 4].astype(np.int64) if len(same_arr) else np.empty(0, dtype=np.int64)
    behind_sample = behind_arr[:, 1].astype(np.int64) if len(behind_arr) else np.empty(0, dtype=np.int64)
    behind_tv = behind_arr[:, 3].astype(np.int64) if len(behind_arr) else np.empty(0, dtype=np.int64)
    behind_fr = behind_arr[:, 4].astype(np.int64) if len(behind_arr) else np.empty(0, dtype=np.int64)
    source_tri = np.asarray([dense[int(sv[i])]["triangle"][int(sr[i])] for i in range(n)], dtype=np.int32)
    source_bary = np.asarray([dense[int(sv[i])]["bary"][int(sr[i])] for i in range(n)], dtype=np.float32)

    response = response_parent = None
    if response_arrays is not None:
        response_path = response_arrays.resolve()
        if not response_path.is_file():
            raise FileNotFoundError("TRUTH139_RELATIONS_V2_RESPONSE_PARENT_MISSING")
        response = _load_response(response_path, len(faces))
        response_parent = {"path": str(response_path), "sha256": sha256(response_path)}

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "RELATIONS_TEACHER.npz"
    arrays = {
        "adjacent_face_pair": adjacent_pair,
        "adjacent_shared_edge_vertex": adjacent_edge,
        "same_source_view_index": same_arr[:, 0].astype(np.int16),
        "same_source_sample_index": same_arr[:, 1].astype(np.int32),
        "same_source_surface_row": same_arr[:, 2].astype(np.int32),
        "same_target_view_index": same_arr[:, 3].astype(np.int16),
        "same_target_surface_row": same_arr[:, 4].astype(np.int32),
        "same_source_triangle_id": source_tri[same_sample].astype(np.int32),
        "same_source_barycentric_uv": source_bary[same_sample].astype(np.float32),
        "same_target_triangle_id": np.asarray([dense[tv]["triangle"][tr] for tv, tr in zip(same_tv, same_tr)], dtype=np.int32),
        "same_target_barycentric_uv": np.asarray([dense[tv]["bary"][tr] for tv, tr in zip(same_tv, same_tr)], dtype=np.float32).reshape(-1, 2),
        "same_canonical_surface_error": same_arr[:, 5].astype(np.float32),
        "behind_source_view_index": behind_arr[:, 0].astype(np.int16),
        "behind_source_sample_index": behind_arr[:, 1].astype(np.int32),
        "behind_source_surface_row": behind_arr[:, 2].astype(np.int32),
        "behind_target_view_index": behind_arr[:, 3].astype(np.int16),
        "front_target_surface_row": behind_arr[:, 4].astype(np.int32),
        "behind_triangle_id": source_tri[behind_sample].astype(np.int32),
        "behind_barycentric_uv": source_bary[behind_sample].astype(np.float32),
        "front_triangle_id": np.asarray([dense[tv]["triangle"][fr] for tv, fr in zip(behind_tv, behind_fr)], dtype=np.int32),
        "front_barycentric_uv": np.asarray([dense[tv]["bary"][fr] for tv, fr in zip(behind_tv, behind_fr)], dtype=np.float32).reshape(-1, 2),
        "front_behind_depth_gap": behind_arr[:, 5].astype(np.float32),
    }
    if response is not None:
        arrays.update(response_pair_face=response["pair"], response_pair_coherence=response["coherence"], response_pair_jump=response["jump"], response_pair_valid=response["valid"])
    _write_npz(out, **arrays)

    payload = {
        "schema": SCHEMA,
        "date": "2026-10-29",
        "status": STATUS,
        "producer_revision": PRODUCER_REVISION,
        "subject_id": subject_id,
        "contract": CONTRACT,
        "parents": {
            "dense_graphics_gbuffer_manifest": {"path": str(Path(dense_graphics_manifest).resolve()), "sha256": sha256(Path(dense_graphics_manifest).resolve())},
            "dense_graphics_identity_sha256": str(graphics.get("graphics_identity_sha256") or ""),
            "dense_graphics_geometry_court": {"path": str(Path(dense_graphics_court).resolve()), "sha256": sha256(Path(dense_graphics_court).resolve())},
            "visibility_occlusion_manifest": {"path": str(Path(visibility_manifest).resolve()), "sha256": sha256(Path(visibility_manifest).resolve())},
            "visibility_occlusion_teacher_identity_sha256": str(visibility.get("teacher_identity_sha256") or ""),
            "visibility_occlusion_court": {"path": str(Path(visibility_court).resolve()), "sha256": sha256(Path(visibility_court).resolve())},
            "geometry_npz": {"path": str(geometry_path), "sha256": sha256(geometry_path)},
            "stage47_response_arrays": response_parent,
        },
        "relations_npz": {"path": str(out), "sha256": sha256(out)},
        "counts": {"local_surface_adjacency": int(len(adjacent_pair)), "same_canonical_surface_across_views": int(len(same_rows)), "front_behind_on_shared_camera_ray": int(len(behind_rows)), "response_pair_total": int(len(response["pair"])) if response is not None else 0, "response_pair_valid": int(np.count_nonzero(response["valid"])) if response is not None else 0},
        "policy": {"legacy_dense_graphics_teacher_authority_used": False, "dense_graphics_and_visibility_parents_must_have_independent_court_pass": True, "same_surface_target_row_recovered_from_canonical_dense_pixel_address": True, "front_behind_target_row_recovered_from_canonical_dense_pixel_address": True, "exact_mesh_edge_adjacency_used": True, "response_coherence_optional_measured_evidence_only": True, "semantic_part_material_owner_or_bone_name_evidence_used": False, "training_performed": False},
        "qualification_minted": False,
    }
    payload["relations_teacher_identity_sha256"] = _digest(payload)
    manifest_out = output_dir / "RELATIONS_TEACHER_MANIFEST.json"
    manifest_out.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-graphics-manifest", required=True, type=Path)
    parser.add_argument("--dense-graphics-court", required=True, type=Path)
    parser.add_argument("--visibility-manifest", required=True, type=Path)
    parser.add_argument("--visibility-court", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--response-arrays", type=Path)
    args = parser.parse_args()
    result = build(dense_graphics_manifest=args.dense_graphics_manifest, dense_graphics_court=args.dense_graphics_court, visibility_manifest=args.visibility_manifest, visibility_court=args.visibility_court, output_dir=args.output_dir, response_arrays=args.response_arrays)
    print(json.dumps({"status": result["status"], "producer_revision": result["producer_revision"], "subject_id": result["subject_id"], "counts": result["counts"], "relations_teacher_identity_sha256": result["relations_teacher_identity_sha256"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
