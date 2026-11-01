#!/usr/bin/env python3
"""Check whether a rest-position quotient would conflate different skin weights."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.sparse import coo_matrix

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.corpus.structural_geometry_audit_v1 import owner_preserving_exact_quotient


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def probe(report_path):
    report = json.loads(Path(report_path).read_text())
    ref = report["outputs"]["rest_geometry"]
    if sha(ref["path"]) != ref["sha256"]:
        raise ValueError("COINCIDENCE_WEIGHT_PAYLOAD_DRIFT")
    with np.load(ref["path"], allow_pickle=False) as z:
        v, f, owner = (z[k] for k in ("rest_vertices_source_world", "faces", "face_owner_privileged"))
        q = owner_preserving_exact_quotient(v, f, owner)
        w = coo_matrix((z["weight_value"], (z["weight_vertex"], z["weight_joint"])), shape=(len(v), len(report["controls"]))).tocsr()
    duplicates = [g for g in q.raw_members if len(g) > 1]
    conflicts, max_error = [], 0.0
    for group in duplicates:
        a = w[list(group)].toarray()
        mass = a.sum(1)
        weighted = mass > 0
        a[weighted] /= mass[weighted, None]
        error = float(np.max(np.abs(a - a[:1]))) if a.size else 0.0
        max_error = max(max_error, error)
        # Unweighted and weighted vertices must not collapse into one skin row.
        if error > 1e-6 or (np.any(weighted) and not np.all(weighted)):
            conflicts.append({"raw_vertex_ids": list(group), "maximum_normalized_weight_difference": error})
    return {"subject_id": report["subject_id"], "import_report": {"path": str(Path(report_path).resolve()), "sha256": sha(report_path)},
            "same_owner_exact_position_duplicate_group_count": len(duplicates), "weight_conflict_group_count": len(conflicts),
            "maximum_normalized_weight_difference": max_error, "conflicts": conflicts,
            "threshold": 1e-6, "full_deformation_equivalence_qualified": False, "training_authorized": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--inspection-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    candidates = json.loads(a.candidates.read_text())["subjects"]
    paths = [a.inspection_root / r["subject_id"] / "IMPORTED_TEACHER_SOURCE_INSPECTION_V1.json" for r in candidates]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        rows = list(pool.map(probe, paths))
    result = {"schema": "RealSaS.SourceCoincidenceWeightProbe.v1", "subject_count": len(rows),
              "subjects_with_duplicate_positions": sum(r["same_owner_exact_position_duplicate_group_count"] > 0 for r in rows),
              "subjects_with_weight_conflicts": sum(r["weight_conflict_group_count"] > 0 for r in rows),
              "subjects": rows, "producer_sha256": sha(__file__),
              "quotient_producer_sha256": sha(ROOT / "tools/corpus/structural_geometry_audit_v1.py"),
              "training_authorized": False, "mechanical_weld_authority_minted": False}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "subjects"}, sort_keys=True))


if __name__ == "__main__":
    main()
