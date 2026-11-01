#!/usr/bin/env python3
"""Independent geometry/solid-angle court for source-union SDF candidates.

No Boolean producer is imported. Source union membership, actual boundary
distance, signs and embedding are checked separately; camera/deformation/Registry
admission remain downstream requirements.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import numpy as np
import pymeshlab
import trimesh

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.corpus.structural_geometry_audit_v1 import audit_structural_geometry_v1
from tools.corpus.validate_component_geometry_authority_v1 import (
    direct_distance, solid_angle_inside, independent_quotient, face_geometry,
    independent_sign_components, independent_component_validity,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bound(ref):
    path = Path(ref["path"])
    if sha(path) != ref["sha256"]:
        raise ValueError("SOURCE_UNION_COURT_BOUND_BYTES_DRIFT")
    return path


def no_intersections(vertices, faces):
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=np.asarray(vertices, np.float64), face_matrix=np.asarray(faces, np.int32)))
    ms.compute_selection_by_self_intersections_per_face()
    if np.any(ms.current_mesh().face_selection_array()):
        raise ValueError("SOURCE_UNION_COURT_INTERSECTING_TRIANGLES")


def validate(manifest_path):
    manifest_path = Path(manifest_path)
    m = json.loads(manifest_path.read_text())
    if (m["schema"] != "RealSaS.SourceUnionSolidCandidate.v1" or
        m["domain"] != "BOOLEAN_UNION_OF_CLOSED_EMBEDDED_POSITIVE_VOLUME_SOURCE_COMPONENT_SOLIDS" or
        m["coordinate_frame"] != "SOURCE_WORLD__CAMERA_NORMALIZATION_BINDING_REQUIRED_FOR_TRAINING" or
        m["sign_convention"] != "NEGATIVE_INSIDE__POSITIVE_OUTSIDE" or
        m["distance_boundary"] != "BOOLEAN_UNION_BOUNDARY__INTERNAL_SOURCE_FACES_REMOVED"):
        raise ValueError("SOURCE_UNION_COURT_CONTRACT")
    for ref in m["parents"].values(): bound(ref)
    with np.load(bound(m["parents"]["source_geometry"]), allow_pickle=False) as z:
        source_v, source_f, owners = (z[k] for k in ("rest_vertices_source_world", "faces", "face_owner_privileged"))
    _, qv, qf, _ = independent_quotient(source_v, source_f, owners)
    _, _, face_valid = face_geometry(qv, qf)
    components = independent_sign_components(qf, face_valid)
    component_validity = independent_component_validity(qf, components, face_valid)
    if not (np.all(face_valid) and np.all(components >= 0) and all(component_validity.values())):
        raise ValueError("SOURCE_UNION_COURT_SOURCE_COMPONENTS_NOT_CLOSED")
    with np.load(bound(m["output"]), allow_pickle=False) as z:
        v, f = z["vertices_source_world"], z["faces"]
        source_face = z["source_face_id"]
        q, d, closest = z["query_xyz_source_world"], z["unsigned_distance_source_world"], z["closest_boundary_face"]
        signed_idx, signed, occ = z["signed_query_index"], z["signed_distance_source_world"], z["occupancy"]
    if not np.isfinite(v).all() or not np.isfinite(q).all() or not np.isfinite(d).all() or np.any(d < 0):
        raise ValueError("SOURCE_UNION_COURT_FINITE_ARRAYS")
    if f.ndim != 2 or f.shape[1] != 3 or np.any(f < 0) or np.any(f >= len(v)) or closest.shape != (len(q),) or np.any(closest < 0) or np.any(closest >= len(f)):
        raise ValueError("SOURCE_UNION_COURT_INDICES")
    if source_face.shape != (len(f),) or np.any(source_face < 0) or np.any(source_face >= len(source_f)):
        raise ValueError("SOURCE_UNION_COURT_SOURCE_FACE_ADDRESSES")
    structural = audit_structural_geometry_v1(v, f, np.zeros(len(f), np.int32), subject_id=m["subject_id"])
    for key in ("boundary_edge_count", "nonmanifold_edge_count", "orientation_conflict_edge_count", "raw_zero_area_face_count", "quotient_collapsed_face_count", "quotient_zero_area_face_count"):
        if structural["counts"][key]:
            raise ValueError("SOURCE_UNION_COURT_BOUNDARY_" + key)
    if structural["diagnostics"]["point_only_contact_detected"]:
        raise ValueError("SOURCE_UNION_COURT_POINT_CONTACT")
    no_intersections(v, f)
    scale = float(np.linalg.norm(v.max(0) - v.min(0)))
    # Boolean source-face provenance must rebind numerically; IDs alone are not evidence.
    raw_tri = source_v[source_f[source_face]]
    child_tri = v[f]
    max_source_error = 0.0
    for corner in range(3):
        cp = trimesh.triangles.closest_point(raw_tri, child_tri[:, corner])
        max_source_error = max(max_source_error, float(np.max(np.linalg.norm(cp - child_tri[:, corner], axis=1))))
    if max_source_error > scale * 1e-8:
        raise ValueError("SOURCE_UNION_COURT_BOUNDARY_NOT_ON_BOUND_SOURCE_FACE")
    tri = child_tri - v.mean(0)
    volume = float(np.sum(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2]))) / 6)
    if volume <= 0 or abs(volume - m["volume_source_world"]) > max(volume * 1e-7, scale ** 3 * 1e-12):
        raise ValueError("SOURCE_UNION_COURT_VOLUME")
    subset = np.linspace(0, len(q) - 1, min(256, len(q)), dtype=np.int64)
    expected_inside = np.zeros(len(subset), bool)
    for cid in sorted(component_validity):
        ids = np.flatnonzero(components == cid)
        sf = qf[ids]
        used, inverse = np.unique(sf, return_inverse=True)
        sv, sf = qv[used], inverse.reshape(-1, 3)
        no_intersections(sv, sf)
        st = sv[sf] - sv.mean(0)
        if np.sum(np.einsum("ij,ij->i", st[:, 0], np.cross(st[:, 1], st[:, 2]))) <= 0:
            raise ValueError("SOURCE_UNION_COURT_SOURCE_VOLUME_SIGN")
        expected_inside |= solid_angle_inside(q[subset], sv, sf)
    if not np.array_equal(expected_inside, solid_angle_inside(q[subset], v, f)):
        raise ValueError("SOURCE_UNION_COURT_RAW_UNION_MEMBERSHIP")
    if (signed_idx.ndim != 1 or not len(signed_idx) or len(np.unique(signed_idx)) != len(signed_idx) or
        np.any(signed_idx < 0) or np.any(signed_idx >= len(q)) or signed.shape != signed_idx.shape or occ.shape != signed_idx.shape):
        raise ValueError("SOURCE_UNION_COURT_COMPACT_SIGN_LAYOUT")
    inside = solid_angle_inside(q[signed_idx], v, f)
    if not np.array_equal(inside, occ) or not np.any(inside) or np.all(inside):
        raise ValueError("SOURCE_UNION_COURT_ALL_SIGN_ROWS")
    if not np.allclose(signed, d[signed_idx] * np.where(inside, -1., 1.), atol=scale * 1e-8, rtol=0):
        raise ValueError("SOURCE_UNION_COURT_SIGNED_DISTANCE_MAGNITUDE")
    cp = trimesh.triangles.closest_point(v[f[closest]], q)
    addressed_error = float(np.max(np.abs(np.linalg.norm(q - cp, axis=1) - d)))
    subset = np.linspace(0, len(q) - 1, min(96, len(q)), dtype=np.int64)
    independently_nearest = direct_distance(q[subset], v, f, np.arange(len(f)))
    nearest_error = float(np.max(np.abs(independently_nearest - d[subset])))
    if max(addressed_error, nearest_error) > scale * 1e-8:
        raise ValueError("SOURCE_UNION_COURT_ACTUAL_BOUNDARY_DISTANCE")
    return {"schema": "RealSaS.SourceUnionSolidCandidateCourt.v1", "status": "PASS_SOURCE_UNION_BOUNDARY_AND_GLOBAL_SDF_NUMERICS__CAMERA_AND_ADMISSION_SEPARATE",
        "subject_id": m["subject_id"], "candidate": {"path": str(manifest_path.resolve()), "sha256": sha(manifest_path)},
        "validated": {"all_signed_query_rows": len(signed_idx), "raw_union_membership_queries": len(expected_inside),
                      "all_addressed_distance_rows": len(q), "independent_nearest_distance_rows": len(subset),
                      "max_source_face_error_over_scale": max_source_error / scale,
                      "max_addressed_distance_error_over_scale": addressed_error / scale,
                      "max_independent_nearest_error_over_scale": nearest_error / scale},
        "boundary_structural_audit": structural, "pymeshlab_version": version("pymeshlab"),
        "producer_source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (
            Path(__file__).resolve(), ROOT / "tools/corpus/validate_component_geometry_authority_v1.py",
            ROOT / "tools/corpus/structural_geometry_audit_v1.py")},
        "deformation_qualified": False, "camera_normalization_bound": False, "training_authorized": False, "qualification_minted": False}


def run_one(row):
    if row["status"] == "BLOCKED": return row
    try:
        p = bound(row["manifest"])
        result = validate(p)
        out = p.with_name("SOURCE_UNION_SOLID_CANDIDATE_COURT_V1.json")
        out.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
        return {"subject_id": row["subject_id"], "status": result["status"], "court": {"path": str(out), "sha256": sha(out)}}
    except Exception as exc:
        return {"subject_id": row["subject_id"], "status": "BLOCKED", "error_type": type(exc).__name__, "reason": str(exc)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    rows = json.loads(a.batch.read_text())["subjects"]
    results = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for result in pool.map(run_one, rows):
            results.append(result)
            print(json.dumps({"done": len(results), "total": len(rows), "last": result}), flush=True)
    receipt = {"schema": "RealSaS.SourceUnionSolidCandidateCourtBatch.v1", "subject_count": len(rows), "subjects": results,
        "candidate_batch": {"path": str(a.batch.resolve()), "sha256": sha(a.batch)}, "producer_sha256": sha(__file__), "training_authorized": False}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
