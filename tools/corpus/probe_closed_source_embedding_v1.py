#!/usr/bin/env python3
"""Independent MeshLab intersection probe of topologically closed candidates.

This is a diagnostic; absence of detected intersections does not qualify a solid
assembly, nesting policy, global SDF target, motion or training admission.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.iris.truth.component_geometry_authority_v1 import build_component_geometry_authority_v1


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def intersecting_faces(v, f):
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=np.asarray(v, np.float64), face_matrix=np.asarray(f, np.int32)))
    ms.compute_selection_by_self_intersections_per_face()
    return np.asarray(ms.current_mesh().face_selection_array(), bool).copy()


def probe(row):
    ref = row["report"]
    if sha(ref["path"]) != ref["sha256"]:
        raise ValueError("CLOSED_SOURCE_PROBE_REPORT_DRIFT")
    route = json.loads(Path(ref["path"]).read_text())
    parent = route["import_report"]
    if sha(parent["path"]) != parent["sha256"]:
        raise ValueError("CLOSED_SOURCE_PROBE_IMPORT_DRIFT")
    imported = json.loads(Path(parent["path"]).read_text())
    geometry = imported["outputs"]["rest_geometry"]
    if sha(geometry["path"]) != geometry["sha256"]:
        raise ValueError("CLOSED_SOURCE_PROBE_PAYLOAD_DRIFT")
    with np.load(geometry["path"], allow_pickle=False) as z:
        a = build_component_geometry_authority_v1(z["rest_vertices_source_world"], z["faces"], z["face_owner_privileged"])
    if not (np.all(a.face_valid) and np.all(a.sign_component_id >= 0) and np.all(a.sign_component_valid)):
        raise ValueError("CLOSED_SOURCE_PROBE_CANDIDATE_NO_LONGER_CLOSED")
    selected = intersecting_faces(a.topology_vertices, a.topology_faces)
    components = []
    for cid in range(len(a.sign_component_valid)):
        ids = np.flatnonzero(a.sign_component_id == cid)
        f = a.topology_faces[ids]
        used, inverse = np.unique(f, return_inverse=True)
        v = a.topology_vertices[used]
        local = inverse.reshape(-1, 3)
        self_selected = intersecting_faces(v, local)
        tri = v[local] - v.mean(axis=0)
        volume = float(np.sum(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2]))) / 6)
        components.append({"component_id": cid, "face_count": len(f),
            "component_self_intersecting_face_count": int(np.count_nonzero(self_selected)),
            "scene_intersecting_face_count": int(np.count_nonzero(selected[ids])),
            "signed_volume_source_world": volume})
    return {"schema": "RealSaS.ClosedSourceEmbeddingProbe.v1", "subject_id": row["subject_id"],
            "origin": row["origin"], "structural_report": ref,
            "scene_intersecting_face_count": int(np.count_nonzero(selected)),
            "components_with_self_intersection": sum(x["component_self_intersecting_face_count"] > 0 for x in components),
            "components": components, "pymeshlab_version": version("pymeshlab"),
            "filter": "compute_selection_by_self_intersections_per_face",
            "solid_assembly_and_nesting_qualified": False, "global_sign_authorized": False,
            "training_authorized": False, "producer_sha256": sha(__file__)}


def run_one(row):
    try:
        return probe(row), None
    except Exception as exc:
        return None, {"subject_id": row["subject_id"], "error_type": type(exc).__name__, "reason": str(exc)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--census", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    candidates = [r for r in json.loads(a.census.read_text())["subjects"] if r["all_faces_in_closed_oriented_vertex_manifold_components"]]
    rows, failures = [], []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for result, failure in pool.map(run_one, candidates):
            if failure: failures.append(failure)
            else: rows.append(result)
    report = {"schema": "RealSaS.ClosedSourceEmbeddingProbeSummary.v1", "candidate_count": len(candidates),
              "measured_count": len(rows), "failures": failures,
              "scenes_with_intersecting_faces": sum(r["scene_intersecting_face_count"] > 0 for r in rows),
              "scenes_with_component_self_intersection": sum(r["components_with_self_intersection"] > 0 for r in rows),
              "subjects": rows, "global_sign_authorized": False, "training_authorized": False,
              "census": {"path": str(a.census.resolve()), "sha256": sha(a.census)}, "producer_sha256": sha(__file__)}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "subjects"}, sort_keys=True))
    if failures: raise SystemExit(1)


if __name__ == "__main__":
    main()
