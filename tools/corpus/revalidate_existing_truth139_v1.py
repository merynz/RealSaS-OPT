#!/usr/bin/env python3
"""Recheck immutable historical geometry/RGB bytes; never regenerate/admit truth."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import platform
import sys
import numpy as np
import torch
import torchvision
import trimesh

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.corpus.validate_truth139_v32_exact_geometry_v1 import validate as geometry_court
from tools.corpus.validate_truth139_v32_rgb_patch_cache_v1 import validate as rgb_court
from tools.training.validate_truth139_v32_mechanical_tessa_teacher_v1 import validate as teacher_court


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bound(ref):
    p = Path(ref["path"])
    if sha(p) != ref["sha256"]:
        raise ValueError("EXISTING_TRUTH_BOUND_BYTES_DRIFT:" + str(p))
    return p


def max_error(a, b, limit, label):
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(label + "_LAYOUT")
    error = float(np.max(np.abs(a - b))) if a.size else 0.0
    if error > limit:
        raise ValueError(label + ":" + str(error))
    return error


def independent_geometry_payloads(manifest):
    # Reconstruct before the cache's float32 rounding, especially for thin faces.
    with np.load(bound(manifest["parents"]["geometry_npz"]), allow_pickle=False) as z:
        source_v, f = np.asarray(z["rest_vertices_source_world"], np.float64), np.asarray(z["faces"], np.int64)
    snapshot = json.loads(bound(manifest["parents"]["source_scene_snapshot"]).read_text())
    matrix = np.asarray(snapshot["source_world_to_canonical"], np.float64)
    v = source_v @ matrix[:3, :3].T + matrix[:3, 3]
    tri = v[f]
    source_tri = source_v[f]
    normal = np.cross(source_tri[:, 1] - source_tri[:, 0], source_tri[:, 2] - source_tri[:, 0])
    normal = normal @ np.linalg.inv(matrix[:3, :3])
    lengths = np.linalg.norm(normal, axis=1)
    normal /= np.maximum(lengths[:, None], 1e-30)
    with np.load(bound(manifest["outputs"]["surface"]), allow_pickle=False) as z:
        fi, bary = z["triangle_id"], np.asarray(z["barycentric"], np.float64)
        if np.any(fi < 0) or np.any(fi >= len(f)) or bary.shape != (len(fi), 3) or np.any(bary < -1e-7):
            raise ValueError("EXISTING_TRUTH_SURFACE_ADDRESS")
        max_error(bary.sum(1), np.ones(len(fi)), 1e-6, "EXISTING_TRUTH_BARYCENTRIC_SUM")
        surface_error = max_error(np.sum(tri[fi] * bary[..., None], axis=1), z["surface_xyz_canonical"], 4e-5, "EXISTING_TRUTH_ALL_SURFACE_ROWS")
        mask = np.asarray(z["normal_valid"], bool)
        normal_error = max_error(normal[fi[mask]], z["normal_canonical"][mask], 4e-5, "EXISTING_TRUTH_ALL_NORMAL_ROWS")
        surface_count, normal_count = len(fi), int(np.count_nonzero(mask))
    with np.load(bound(manifest["outputs"]["distance_sign"]), allow_pickle=False) as z:
        q, d, closest = np.asarray(z["query_xyz_canonical"], np.float64), z["unsigned_distance"], z["closest_face"]
        sign_count = int(np.count_nonzero(z["sign_valid"]))
        if closest.shape != (len(q),) or np.any(closest < 0) or np.any(closest >= len(f)):
            raise ValueError("EXISTING_TRUTH_DISTANCE_ADDRESS")
    addressed = np.linalg.norm(q - trimesh.triangles.closest_point(tri[closest], q), axis=1)
    distance_error = max_error(addressed, d, 5e-5, "EXISTING_TRUTH_ALL_ADDRESSED_DISTANCES")
    seed = int(hashlib.sha256((manifest["subject_id"] + "|independent-nearest").encode()).hexdigest()[:16], 16)
    idx = np.random.default_rng(seed).choice(len(q), min(96, len(q)), replace=False)
    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    _, nearest, _ = trimesh.proximity.closest_point(mesh, q[idx])
    nearest_error = max_error(nearest, d[idx], 5e-5, "EXISTING_TRUTH_INDEPENDENT_NEAREST")
    return {"surface_rows": surface_count, "oriented_normal_rows": normal_count,
            "addressed_distance_rows": len(q), "independent_nearest_rows": len(idx), "valid_sign_rows": sign_count,
            "max_surface_error": surface_error, "max_normal_error": normal_error,
            "max_addressed_distance_error": distance_error, "max_independent_nearest_error": nearest_error,
            "normal_semantics": "SOURCE_FACE_ORIENTATION__NOT_SOLID_OUTWARD_AUTHORITY"}


def oriented_triangle_multiset(vertices, faces):
    """Exact positions and winding; invariant only to row/vertex addresses."""
    triangles = np.asarray(vertices, np.float64)[faces].copy()
    triangles[triangles == 0] = 0  # signed zero is not a geometric difference
    return Counter(min(t.tobytes(), np.roll(t, 1, axis=0).tobytes(), np.roll(t, 2, axis=0).tobytes()) for t in triangles)


def imported_array_comparison(manifest, inspection_root):
    sid = manifest["subject_id"]
    path = inspection_root / sid / "IMPORTED_TEACHER_SOURCE_INSPECTION_V1.json"
    fresh = json.loads(path.read_text())
    if fresh["source"]["sha256"] != manifest["parents"]["source_sha256"] or sha(fresh["source"]["path"]) != fresh["source"]["sha256"]:
        raise ValueError("EXISTING_TRUTH_RAW_SOURCE_IDENTITY_DRIFT")
    with np.load(bound(manifest["parents"]["geometry_npz"]), allow_pickle=False) as z:
        old_v, old_f, old_owner = (z[k] for k in ("rest_vertices_source_world", "faces", "face_owner_privileged"))
    with np.load(bound(fresh["outputs"]["rest_geometry"]), allow_pickle=False) as z:
        new_v, new_f, new_owner = (z[k] for k in ("rest_vertices_source_world", "faces", "face_owner_privileged"))
    same_layout = old_v.shape == new_v.shape and old_f.shape == new_f.shape
    triangles_exact = oriented_triangle_multiset(old_v, old_f) == oriented_triangle_multiset(new_v, new_f)
    return {"fresh_import_report": {"path": str(path.resolve()), "sha256": sha(path)},
            "raw_source_sha256_matches": True, "same_vertex_face_shapes": same_layout,
            "faces_exact": bool(np.array_equal(old_f, new_f)), "vertices_exact": bool(np.array_equal(old_v, new_v)),
            "owner_addresses_exact": bool(np.array_equal(old_owner, new_owner)),
            "oriented_triangle_position_multisets_exact": triangles_exact,
            "array_order_difference_is_not_itself_surface_drift": True,
            "max_absolute_source_world_vertex_difference": float(np.max(np.abs(old_v.astype(np.float64) - new_v))) if same_layout else None,
            "fresh_import_runtime": fresh["runtime"], "admitted_import_replay_qualification_minted": False}


def revalidate_subject(sid, refs, inspection_root):
    roots = {k: bound(v) for k, v in refs.items() if isinstance(v, dict) and "path" in v}
    gm, rgb, teacher = (json.loads(roots[k].read_text()) for k in ("geometry", "rgb", "tessa_teacher"))
    for m in (gm, rgb, teacher):
        if m["subject_id"] != sid:
            raise ValueError("EXISTING_TRUTH_SUBJECT_BINDING")
    try:
        teacher_result = {"status": "PASS_CURRENT_TEACHER_COURT", "court": teacher_court(teacher)}
    except ValueError as exc:
        teacher_result = {"status": "BLOCKED_CURRENT_TEACHER_COURT", "reason": str(exc)}
    return {"schema": "RealSaS.ExistingTruthNumericRevalidation.v1", "subject_id": sid,
            "root_ref_count": len(roots), "geometry_court": geometry_court(gm), "rgb_court": rgb_court(rgb),
            "independent_geometry_payloads": independent_geometry_payloads(gm),
            "source_import_comparison": imported_array_comparison(gm, inspection_root),
            "teacher_selected_kind": teacher["selected"]["kind"],
            "teacher_measured_all_kinds_contract": teacher["selected"].get("candidate_evaluation_contract") == "MEASURED_ALL_KINDS_NO_FALLBACK_OVERRIDE_V1",
            "teacher_current_contract_court": teacher_result,
            "input_sha256": {str(p): sha(p) for p in roots.values()}, "truth_regenerated": False,
            "training_authorized": False, "qualification_minted": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--readiness", type=Path, required=True)
    p.add_argument("--inspection-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(1)
    roster = json.loads(a.readiness.read_text())["subjects"]
    if not roster:
        raise ValueError("EXISTING_TRUTH_EMPTY_ROSTER")
    a.output_root.mkdir(parents=True, exist_ok=True)
    def run(item):
        sid, refs = item
        try:
            result = revalidate_subject(sid, refs, a.inspection_root)
            (a.output_root / (sid + ".json")).write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
            return result, None
        except Exception as exc:
            return None, {"subject_id": sid, "error_type": type(exc).__name__, "reason": str(exc)}
    results, failures = [], []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for result, failure in pool.map(run, sorted(roster.items())):
            if failure:
                failures.append(failure)
            else:
                results.append(result)
            if (len(results) + len(failures)) % 10 == 0:
                print(json.dumps({"complete": len(results), "failed": len(failures), "total": len(roster)}), flush=True)
    receipt = {"schema": "RealSaS.ExistingEvidenceRevalidationReceipt.v1", "subject_count": len(roster),
               "numeric_pass_count": len(results), "failures": failures,
               "root_refs_verified": sum(r["root_ref_count"] for r in results),
               "geometry_court_pass_count": len(results), "rgb_court_pass_count": len(results),
               "valid_sign_rows": sum(r["independent_geometry_payloads"]["valid_sign_rows"] for r in results),
               "selected_teacher_kinds": dict(Counter(r["teacher_selected_kind"] for r in results)),
               "teacher_current_court_outcomes": dict(Counter(r["teacher_current_contract_court"].get("reason", r["teacher_current_contract_court"]["status"]) for r in results)),
               "exact_fresh_import_vertex_and_face_arrays": sum(r["source_import_comparison"]["vertices_exact"] and r["source_import_comparison"]["faces_exact"] for r in results),
               "exact_fresh_import_oriented_triangle_geometry": sum(r["source_import_comparison"]["oriented_triangle_position_multisets_exact"] for r in results),
               "readiness": {"path": str(a.readiness.resolve()), "sha256": sha(a.readiness)},
               "runtime": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                           "torchvision": torchvision.__version__, "trimesh": trimesh.__version__},
               "producer_source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (
                   Path(__file__).resolve(), ROOT / "tools/corpus/validate_truth139_v32_exact_geometry_v1.py",
                   ROOT / "tools/corpus/build_truth139_v32_exact_geometry_v1.py", ROOT / "tools/corpus/validate_truth139_v32_rgb_patch_cache_v1.py",
                   ROOT / "tools/training/validate_truth139_v32_mechanical_tessa_teacher_v1.py", ROOT / "tools/training/truth139_v32_spatial_index_v1.py")},
               "truth_regenerated": False, "training_authorized": False, "qualification_minted": False}
    dest = a.output_root / "EXISTING_EVIDENCE_REVALIDATION_RECEIPT_V1.json"
    dest.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    print(json.dumps(receipt, sort_keys=True), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
