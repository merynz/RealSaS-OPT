#!/usr/bin/env python3
"""Original-source Semantic TruthBuilder teacher group probe (candidate only).

Run inside pinned Blender, with the exact original Corpus370 source, admission
receipt, and Stage50 source-rest geometry. Replays the SAME rest importer and
triangulation as the existing palette/native intervention producers.

This producer returns source-connected teacher annotation units and weak
StdGEN-style semantic name hints; it never labels an item as training-ready,
renders a false semantic pass, or mutates the original source file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.iris.truth.semantic_source_groups_v1 import build_semantic_source_groups
from tools.observation.blender_controlled_8view_v2 import import_rest
from tools.iris_vnext.blender_authored_teacher_export_v1 import _mesh_objects
from tools.iris_vnext.blender_authored_teacher_export_v3 import _extract_snapshot

SEMANTIC_POLICY = ROOT / "canonical/REALSAS_SEMANTIC_PRIOR_POLICY_V1.json"
HINT_POLICY = ROOT / "canonical/REALSAS_SEMANTIC_SOURCE_HINT_POLICY_V1.json"


def _sha(path: Path) -> str:
    with path.open("rb") as fp:
        return hashlib.file_digest(fp, "sha256").hexdigest()


def _require(value: bool, code: str) -> None:
    if not value:
        raise RuntimeError("CORPUS370_SEMANTIC_BUILDER_" + code)


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    _require(isinstance(value, dict), "JSON_OBJECT_REQUIRED")
    return value


def run(*, subject_id: str, source: Path, expected_source_sha: str,
        admission: Path, geometry: Path, expected_geometry_sha: str,
        output: Path, semantic_policy: Path = SEMANTIC_POLICY,
        hint_policy: Path = HINT_POLICY) -> dict:
    source = source.resolve()
    admission = admission.resolve()
    geometry = geometry.resolve()
    for path in (source, admission, geometry, semantic_policy, hint_policy):
        _require(path.is_file(), "MISSING_REQUIRED_INPUT:" + path.name)
    _require(_sha(source) == expected_source_sha, "SOURCE_BYTES_CHANGED")
    _require(_sha(geometry) == expected_geometry_sha, "GEOMETRY_BYTES_CHANGED")
    a = _load_json(admission)
    _require(a.get("schema") == "RealSaS.IrisBindObservationAdmission.v1"
             and a.get("status") == "PASS_IRIS_BIND_OBSERVATIONS"
             and a.get("subject_id") == subject_id
             and a.get("source_sha256") == expected_source_sha,
             "SOURCE_OBSERVATION_ADMISSION")
    _require(a.get("full_native_motion_truth_admitted") is False,
             "OBSERVATION_NOT_MOTION_AUTHORITY")
    with np.load(geometry, allow_pickle=False) as npz:
        _require({"rest_vertices_source_world", "faces", "face_owner_privileged"} <= set(npz.files),
                 "SOURCE_GEOMETRY_PAYLOAD")
        expected_v = np.asarray(npz["rest_vertices_source_world"], dtype=np.float64)
        expected_f = np.asarray(npz["faces"], dtype=np.int64)
        expected_owner = np.asarray(npz["face_owner_privileged"], dtype=np.int64)
    with tempfile.TemporaryDirectory(prefix="realsas_semantic_teacher_") as directory:
        import_receipt = import_rest(source, Path(directory))
        snapshot = _extract_snapshot(_mesh_objects(), freeze_triangulation=True)
        observed_v = np.asarray(snapshot["vertices"], dtype=np.float64)
        observed_f = np.asarray(snapshot["faces"], dtype=np.int64)
        observed_owner = np.asarray(snapshot["face_owner"], dtype=np.int64)
        _require(expected_v.shape == observed_v.shape, "VERTEX_COUNT_MISMATCH")
        _require(np.array_equal(expected_f, observed_f), "SOURCE_FACE_ORDER_MISMATCH")
        _require(np.array_equal(expected_owner, observed_owner), "SOURCE_OWNER_ORDER_MISMATCH")
        scale = max(float(np.ptp(expected_v, axis=0).max()), 1.0e-6)
        error = float(np.max(np.abs(expected_v - observed_v))) / scale
        _require(np.isfinite(error) and error <= 1.0e-5, "REST_VERTEX_PARITY")
        result = build_semantic_source_groups(
            subject_id=subject_id, source_sha256=expected_source_sha, snapshot=snapshot,
            hint_policy=_load_json(hint_policy),
            semantic_prior_policy=_load_json(semantic_policy),
        )
    result.update({
        "source_rest_geometry_sha256": expected_geometry_sha,
        "source_observation_admission_sha256": _sha(admission),
        "source_geometry_relative_vertex_error": error,
        "source_rest_replay_operation": import_receipt.get("operation"),
        "source_rest_face_order_verified": True,
        "source_rest_owner_binding_verified": True,
        "source_rest_8_view_semantic_raster_rendered": False,
        "semantic_region_reviewed": False,
        "semantic_family_independent_court_passed": False,
        "training_authorized": False,
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return result


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--expected-geometry-sha256", required=True)
    parser.add_argument("--semantic-policy", type=Path, default=SEMANTIC_POLICY)
    parser.add_argument("--hint-policy", type=Path, default=HINT_POLICY)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(
        subject_id=args.subject_id, source=args.source,
        expected_source_sha=args.expected_source_sha256,
        admission=args.admission, geometry=args.geometry,
        expected_geometry_sha=args.expected_geometry_sha256,
        output=args.output, semantic_policy=args.semantic_policy,
        hint_policy=args.hint_policy,
    )
    print(json.dumps({"status": result["status"], "subject_id": result["subject_id"],
                      "group_count": result["group_count"],
                      "training_authorized": result["training_authorized"]}, sort_keys=True))


if __name__ == "__main__":
    main()
