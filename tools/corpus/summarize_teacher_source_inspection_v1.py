#!/usr/bin/env python3
"""Independent structural census of imported source bytes, without admission."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.iris.truth.component_geometry_authority_v1 import build_component_geometry_authority_v1
from tools.corpus.structural_geometry_audit_v1 import audit_structural_geometry_v1


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measure(row, report_path):
    report_path = Path(report_path)
    report = json.loads(report_path.read_text())
    if report["subject_id"] != row["subject_id"] or report["source"] != row["source"]:
        raise ValueError("SOURCE_INSPECTION_BINDING_DRIFT")
    if sha(row["source"]["path"]) != row["source"]["sha256"]:
        raise ValueError("SOURCE_INSPECTION_RAW_SOURCE_DRIFT")
    for ref in report["outputs"].values():
        if sha(ref["path"]) != ref["sha256"]:
            raise ValueError("SOURCE_INSPECTION_PAYLOAD_DRIFT")
    for path, expected in report["producer_source_sha256"].items():
        if sha(ROOT / path) != expected:
            raise ValueError("SOURCE_INSPECTION_PRODUCER_DRIFT")
    with np.load(report["outputs"]["rest_geometry"]["path"], allow_pickle=False) as z:
        v, f, owner = (z[k] for k in ("rest_vertices_source_world", "faces", "face_owner_privileged"))
        audit = audit_structural_geometry_v1(v, f, owner, subject_id=row["subject_id"])
        a = build_component_geometry_authority_v1(v, f, owner)
        wi, wj, w = (z[k] for k in ("weight_vertex", "weight_joint", "weight_value"))
        if (wi.shape != wj.shape or wi.shape != w.shape or wi.ndim != 1 or
            not np.isfinite(w).all() or np.any(w <= 0) or np.any(wi < 0) or np.any(wi >= len(v)) or
            np.any(wj < 0) or np.any(wj >= len(report["controls"]))):
            raise ValueError("SOURCE_INSPECTION_WEIGHT_LAYOUT")
        mass = np.bincount(z["weight_vertex"], weights=z["weight_value"], minlength=len(v))
        referenced = np.unique(f)
        weight = {"joint_count": len(report["controls"]),
                  "weighted_vertex_count": int(np.count_nonzero(mass > 0)),
                  "unweighted_referenced_vertex_count": int(np.count_nonzero(mass[referenced] == 0)),
                  "referenced_vertex_count": len(referenced),
                  "maximum_weight_sum_error_on_weighted_vertices": float(np.max(np.abs(mass[mass > 0] - 1))) if np.any(mass > 0) else None}
    all_closed = bool(np.all(a.face_valid) and np.all(a.sign_component_id >= 0) and
                      len(a.sign_component_valid) and np.all(a.sign_component_valid))
    routes = {
        "source_surface_and_unsigned_distance": "CANDIDATE__OBSERVATION_AND_NUMERIC_COURTS_REQUIRED",
        "global_signed_distance": "CLOSED_COMPONENTS__EMBEDDING_AND_SOLID_BOUNDARY_COURT_REQUIRED" if all_closed else "SOURCE_NOT_ALL_CLOSED__EXPLICIT_QUALIFIED_SOLID_TARGET_REQUIRED",
        "rig_skin": "AUTHORED_JOINTS_AND_WEIGHTS__EQUIVALENCE_COURTS_REQUIRED" if weight["joint_count"] and weight["weighted_vertex_count"] else "NO_AUTHORED_SKIN_TARGET_IN_THIS_IMPORT",
        "node_motion": "AUTHORED_NODE_CONTROLS__RESPONSE_MEASUREMENT_REQUIRED" if not weight["joint_count"] else "JOINT_CONTROLS_PRESENT",
        "occlusion_and_relations": "MULTI_OWNER_CANDIDATE__VIEW_AND_INTERVENTION_EVIDENCE_REQUIRED" if audit["counts"]["owner_count"] > 1 else "VIEW_AND_INTERVENTION_EVIDENCE_REQUIRED",
        "tessa_mesh_proposal": "MEASURED_CANDIDATE_AND_DEFORMATION_COURTS_REQUIRED__NO_SOURCE_FALLBACK_OVERRIDE"}
    result = {"schema": "RealSaS.SourceStructureAndFamilyRoutes.v1", "subject_id": row["subject_id"],
              "origin": row["origin"], "source_id": row["source_id"], "source": row["source"],
              "import_report": {"path": str(report_path.resolve()), "sha256": sha(report_path)},
              "runtime": report["runtime"], "structural_audit": audit, "skin_measurements": weight,
              "component_diagnostics": list(a.sign_components), "all_faces_in_closed_oriented_vertex_manifold_components": all_closed,
              "family_routes": routes, "training_authorized": False, "global_sign_authorized": False,
              "modular_semantics_inferred_from_mesh_count": False,
              "producer_sha256": sha(__file__),
              "producer_source_sha256": {p: sha(ROOT / p) for p in (
                  "models/iris/truth/component_geometry_authority_v1.py",
                  "tools/corpus/structural_geometry_audit_v1.py")}}
    dest = report_path.parent / "SOURCE_STRUCTURE_AND_FAMILY_ROUTES_V1.json"
    dest.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return {"subject_id": row["subject_id"], "origin": row["origin"], "source_id": row["source_id"],
            "report": {"path": str(dest.resolve()), "sha256": sha(dest)}, "runtime": report["runtime"],
            "counts": audit["counts"], "skin_measurements": weight,
            "all_faces_in_closed_oriented_vertex_manifold_components": all_closed,
            "family_routes": routes}


def run_one(args):
    row, path = args
    try:
        return {"ok": True, "row": measure(row, path)}
    except Exception as exc:
        return {"ok": False, "subject_id": row["subject_id"], "error_type": type(exc).__name__, "reason": str(exc)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--inspection-root", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    candidates = json.loads(a.candidates.read_text())
    rows, failures = [], []
    jobs = [(r, a.inspection_root / r["subject_id"] / "IMPORTED_TEACHER_SOURCE_INSPECTION_V1.json") for r in candidates["subjects"]]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for result in pool.map(run_one, jobs):
            if result["ok"]:
                rows.append(result["row"])
            else:
                failures.append(result)
            if (len(rows) + len(failures)) % 25 == 0:
                print(json.dumps({"complete": len(rows), "failures": len(failures), "total": len(jobs)}), flush=True)
    summary = {"schema": "RealSaS.ExpandedTeacherSourceCensus.v1", "status": "SOURCE_MEASUREMENT_COMPLETE__FAMILY_ADMISSION_PENDING" if not failures else "INCOMPLETE",
               "candidate_manifest": {"path": str(a.candidates.resolve()), "sha256": sha(a.candidates)},
               "candidate_count": len(jobs), "measured_count": len(rows), "failures": failures,
               "counts": {"base_rows": sum(r["origin"] == "ESTABLISHED_BASE" for r in rows),
                          "expansion_candidates": sum(r["origin"] == "EXPANSION" for r in rows),
                          "multi_mesh_owner_candidates": sum(r["counts"]["owner_count"] > 1 for r in rows),
                          "authored_joints_and_weights": sum(bool(r["skin_measurements"]["joint_count"] and r["skin_measurements"]["weighted_vertex_count"]) for r in rows),
                          "without_authored_joints": sum(r["skin_measurements"]["joint_count"] == 0 for r in rows),
                          "all_faces_in_closed_oriented_vertex_manifold_components": sum(r["all_faces_in_closed_oriented_vertex_manifold_components"] for r in rows)},
               "source_counts": dict(sorted(Counter(r["source_id"] for r in rows).items())),
               "subjects": rows, "final_iris_tessa_training_n": None, "training_authorized": False,
               "split_created": False, "solid_embedding_and_assembly_boundary_assessed": False,
               "producer_sha256": sha(__file__)}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "subjects"}, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
