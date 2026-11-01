#!/usr/bin/env python3
"""Inspect exact source bytes and preserve the historical truth artifact index.

This is a read-only preparation tool, not a materializer or admission court. It
creates no replacement truth and never imports historical training authorization.
Every source, including the established base, receives the same structural pass.
The existing inventory/courts remain the owners of full mounted-artifact checks.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from urllib.parse import unquote, urlparse

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.corpus.gltf_bind_scene_v1 import SourceGLTF
from tools.corpus.structural_geometry_audit_v1 import audit_structural_geometry_v1

POLICY = ROOT / "canonical/CORPUS_TEACHER_INSPECTION_AND_REUSE_POLICY_V1.json"
FAMILY_INDEX = {
    "SURFACE_GEOMETRY": ("geometry", "geometry_court"),
    "SURFACE_TOPOLOGY": ("tessa_teacher", "tessa_court"),
    "SIGNED_DISTANCE_OCCUPANCY": ("geometry", "geometry_court"),
    "ORIENTED_NORMALS": ("geometry", "geometry_court"),
    "APPEARANCE_CAA": ("rgb", "rgb_court"),
    "VISIBILITY_OCCLUSION": ("stage48",),
    "RAY_COMPOSITING": ("stage48",),
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    partial.replace(path)


def checked_source(root, row):
    path = (Path(root) / row["corpus_relative_path"]).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("TEACHER_SOURCE_PATH_ESCAPE")
    expected = row["expected_sha256"]
    if row.get("valid") is not True or row.get("actual_sha256") != expected or sha256(path) != expected:
        raise ValueError("TEACHER_SOURCE_IDENTITY_DRIFT")
    return path


def inspect_source(path, subject_id):
    source = SourceGLTF(path)
    # Resources are part of source identity. Never treat .gltf hash alone as a
    # complete bundle identity when buffers/textures are external.
    resources = {}
    for section in ("buffers", "images"):
        for row in source.doc.get(section, []):
            uri = row.get("uri", "")
            if not uri or uri.startswith("data:"):
                continue
            if urlparse(uri).scheme or urlparse(uri).netloc:
                raise ValueError("TEACHER_EXTERNAL_RESOURCE_SCHEME")
            resource = (path.parent / unquote(uri)).resolve()
            if not resource.is_relative_to(path.parent.resolve()):
                raise ValueError("TEACHER_RESOURCE_PATH_ESCAPE")
            resources[uri] = sha256(resource)
    geometry = source.geometry()
    vertices = np.asarray(geometry["vertices"], np.float64)
    faces = np.asarray(geometry["faces"], np.int64)
    vertex_owner = np.concatenate([
        np.full(mesh["vertex_count"], mesh["node_index"], np.int32)
        for mesh in geometry["meshes"]
    ])
    face_owner = vertex_owner[faces[:, 0]]
    if np.any(vertex_owner[faces] != face_owner[:, None]):
        raise ValueError("TEACHER_RAW_FACE_OWNER_CROSSING")
    audit = audit_structural_geometry_v1(vertices, faces, face_owner, subject_id=subject_id)
    nodes = source.doc.get("nodes", [])
    profile = {
        "source_sha256": sha256(path),
        "external_resource_sha256": resources,
        "geometry_reader_scope": geometry["scope"],
        "mesh_node_count": len({m["node_index"] for m in geometry["meshes"]}),
        "mesh_primitive_count": len(geometry["meshes"]),
        "joint_count": geometry["bone_count"],
        "authored_animation_count": len(geometry["native_motion_names"]),
        "mesh_node_parents": [
            {"node_index": i, "parent_index": source.parents.get(i), "skin_index": node.get("skin")}
            for i, node in enumerate(nodes) if "mesh" in node
        ],
        "active_scene_equivalence": "PENDING_IMPORTER_REST_COURT",
        "solid_assembly_semantics": "PENDING_EXPLICIT_ENTITY_INSPECTION",
        "global_sign_authority": "NOT_ESTABLISHED_BY_STRUCTURAL_COUNTS",
    }
    result = {
        "schema": "RealSaS.SourceTeacherStructuralInspection.v1",
        "subject_id": subject_id,
        "scope": "RAW_BIND_STRUCTURAL__IMPORTER_REPLAY_AND_SOLID_SEMANTICS_PENDING",
        "structural_audit": audit,
        "counts": audit["counts"],
        "raw_source_profile": profile,
        "training_authorized": False,
    }
    result["source_inspection_identity_sha256"] = digest(result)
    return result


def plan(source_manifest_path, readiness_path, source_root, output_root, *, inspect=True):
    source_manifest_path, readiness_path = Path(source_manifest_path), Path(readiness_path)
    cache = json.loads(source_manifest_path.read_text())
    readiness = json.loads(readiness_path.read_text())
    policy = json.loads(POLICY.read_text())
    rows = cache["subjects"]
    by_id = {row["subject_id"]: row for row in rows}
    if not rows or len(by_id) != len(rows) or cache.get("subject_count") != len(rows):
        raise ValueError("TEACHER_SOURCE_MANIFEST_CARDINALITY")
    old = readiness["subjects"]
    if readiness.get("subject_count") != len(old) or set(old) != set(by_id):
        raise ValueError("TEACHER_BASE_INVENTORY_SUBJECT_SET")
    output_root = Path(output_root)
    planned, failures = [], {}
    measurement_counts = Counter()
    for subject_id, row in sorted(by_id.items()):
        try:
            path = checked_source(source_root, row)
            inspection_ref = None
            if inspect:
                audit = inspect_source(path, subject_id)
                audit_path = output_root / "subjects" / subject_id / "SOURCE_TEACHER_INSPECTION_V1.json"
                write_json(audit_path, audit)
                inspection_ref = {"path": str(audit_path.resolve()), "sha256": sha256(audit_path)}
                counts = audit["counts"]
                measurement_counts["inspected_subjects"] += 1
                measurement_counts["multi_owner_subjects"] += counts["owner_count"] > 1
                measurement_counts["open_boundary_subjects"] += counts["boundary_edge_count"] > 0
                measurement_counts["nonmanifold_edge_subjects"] += counts["nonmanifold_edge_count"] > 0
                measurement_counts["orientation_conflict_subjects"] += counts["orientation_conflict_edge_count"] > 0
            planned.append({
                "subject_id": subject_id,
                "source": {"path": str(path), "sha256": row["expected_sha256"]},
                "source_inspection": inspection_ref,
                "historical_artifact_index": old[subject_id],
                "historical_artifact_bytes_verified_by_this_tool": False,
                "families": {
                    family: {
                        **scope,
                        "historical_refs": {k: old[subject_id][k] for k in FAMILY_INDEX.get(family, ()) if k in old[subject_id]},
                        "reuse_certified": False,
                        "missing_index_is_not_proof_of_missing_truth": True,
                    }
                    for family, scope in policy["families"].items()
                },
            })
        except Exception as exc:
            failures[subject_id] = {"error_type": type(exc).__name__, "reason": str(exc)}
    result = {
        "schema": "RealSaS.TeacherTruthRevalidationPlan.v1",
        "status": ("RAW_SOURCE_STRUCTURAL_INSPECTED__FAMILY_REVALIDATION_PENDING" if inspect else "SOURCE_BYTES_VERIFIED__STRUCTURAL_INSPECTION_PENDING") if not failures else "SOURCE_INSPECTION_INCOMPLETE",
        "parents": {
            "source_manifest": {"path": str(source_manifest_path.resolve()), "sha256": sha256(source_manifest_path)},
            "historical_readiness_index": {"path": str(readiness_path.resolve()), "sha256": sha256(readiness_path)},
            "inspection_and_reuse_policy": {"path": str(POLICY.resolve()), "sha256": sha256(POLICY)},
        },
        "producer_source_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                Path(__file__).resolve(),
                ROOT / "tools/corpus/gltf_bind_scene_v1.py",
                ROOT / "tools/corpus/structural_geometry_audit_v1.py",
                ROOT / "tools/observation/source_scene_contract_v2.py",
            )
        },
        "runtime": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__},
        "subject_count": len(rows),
        "verified_source_count": len(planned),
        "measurement_counts": dict(measurement_counts),
        "subjects": planned,
        "failures": failures,
        "authority": {
            "training_authorized": False,
            "historical_split_inherited": False,
            "final_admitted_subject_count": None,
            "source_mutated": False,
            "truth_artifacts_regenerated": False,
            "attempt_created": False,
        },
    }
    result["plan_identity_sha256"] = digest(result)
    write_json(output_root / "TEACHER_TRUTH_REVALIDATION_PLAN_V1.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--prior-readiness", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--inventory-only", action="store_true", help="Verify source bytes without structural measurement")
    args = parser.parse_args()
    result = plan(args.source_manifest, args.prior_readiness, args.source_root, args.output_root, inspect=not args.inventory_only)
    print(json.dumps({key: result[key] for key in ("status", "subject_count", "verified_source_count", "measurement_counts", "failures")}, sort_keys=True))
    if result["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
