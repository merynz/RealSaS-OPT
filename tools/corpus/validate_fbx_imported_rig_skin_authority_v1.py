#!/usr/bin/env python3
from __future__ import annotations

"""Court pinned-Blender FBX rig/skin decoded source semantics.

The court compares the existing animation-free source inspection with a separately
implemented Blender replay. This proves deterministic importer-decoded hierarchy,
rest matrices and authored vertex-group weights for the exact FBX bytes. It does
not claim an independent raw-FBX parser or convert missing/invalid weights to zero.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

INSPECTION_SCHEMA = "RealSaS.ImportedTeacherSourceInspection.v1"
REPLAY_SCHEMA = "RealSaS.FbxRigSkinImporterReplay.v1"
REPLAY_STATUS = "PASS_FBX_RIG_SKIN_IMPORTER_REPLAY_CANDIDATE__INDEPENDENT_COURT_REQUIRED"
SCHEMA = "RealSaS.FbxImportedRigSkinAuthorityCourt.v1"
STATUS = "PASS_FBX_IMPORTED_RIG_SKIN_DECODE_EQUIVALENCE__FINAL_ADMISSION_SEPARATE"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def load(path: Path) -> dict:
    value = json.loads(path.read_text())
    require(isinstance(value, dict), "FBX_RIG_SKIN_COURT_JSON_OBJECT")
    return value


def bound(ref: dict, code: str) -> Path:
    path = Path(str((ref or {}).get("path") or "")).resolve()
    expected = str((ref or {}).get("sha256") or "")
    require(path.is_file() and len(expected) == 64 and sha256(path) == expected, code)
    return path


def control_key(row: dict) -> tuple[str, str]:
    return str(row.get("armature") or ""), str(row.get("name") or "")


def canonical_controls(rows: list[dict]) -> dict[tuple[str, str], dict]:
    result = {}
    for row in rows:
        key = control_key(row)
        require(all(key) and key not in result, "FBX_RIG_SKIN_COURT_CONTROL_SET")
        matrix = np.asarray(row.get("rest_matrix_world"), dtype=np.float64)
        require(matrix.shape == (4, 4) and np.isfinite(matrix).all(), "FBX_RIG_SKIN_COURT_CONTROL_MATRIX")
        require(np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-10, rtol=0), "FBX_RIG_SKIN_COURT_CONTROL_AFFINE")
        require(np.linalg.matrix_rank(matrix) == 4, "FBX_RIG_SKIN_COURT_CONTROL_SINGULAR")
        result[key] = {
            "parent": row.get("parent"),
            "use_deform": bool(row.get("use_deform")),
            "matrix": matrix,
        }
    return result


def inspection_weights(inspection: dict, arrays_path: Path) -> tuple[dict, int]:
    controls = list(inspection.get("controls") or [])
    meshes = list(inspection.get("meshes") or [])
    require(controls and meshes, "FBX_RIG_SKIN_COURT_INSPECTION_INVENTORY")
    offsets = np.asarray([0, *np.cumsum([int(row["vertex_count"]) for row in meshes])], dtype=np.int64)
    with np.load(arrays_path, allow_pickle=False) as z:
        required = {"weight_vertex", "weight_joint", "weight_value"}
        require(required.issubset(z.files), "FBX_RIG_SKIN_COURT_INSPECTION_ARRAYS")
        vertex = np.asarray(z["weight_vertex"], dtype=np.int64)
        joint = np.asarray(z["weight_joint"], dtype=np.int64)
        value = np.asarray(z["weight_value"], dtype=np.float64)
    require(vertex.shape == joint.shape == value.shape, "FBX_RIG_SKIN_COURT_INSPECTION_WEIGHT_LAYOUT")
    require(np.isfinite(value).all() and np.all(value > 0), "FBX_RIG_SKIN_COURT_INSPECTION_WEIGHT_VALUE")
    result = {}
    for global_vertex, control_index, weight in zip(vertex, joint, value):
        require(0 <= global_vertex < offsets[-1], "FBX_RIG_SKIN_COURT_INSPECTION_VERTEX_RANGE")
        require(0 <= control_index < len(controls), "FBX_RIG_SKIN_COURT_INSPECTION_CONTROL_RANGE")
        mesh_index = int(np.searchsorted(offsets, int(global_vertex), side="right") - 1)
        local_vertex = int(global_vertex - offsets[mesh_index])
        control = controls[int(control_index)]
        key = (
            str(meshes[mesh_index]["name"]),
            local_vertex,
            str(control.get("armature") or ""),
            str(control.get("name") or ""),
        )
        require(key not in result, "FBX_RIG_SKIN_COURT_INSPECTION_WEIGHT_DUPLICATE")
        result[key] = float(weight)
    return result, int(offsets[-1])


def replay_weights(replay: dict, arrays_path: Path) -> dict:
    controls = list(replay.get("controls") or [])
    meshes = list(replay.get("meshes") or [])
    with np.load(arrays_path, allow_pickle=False) as z:
        required = {"weight_mesh_index", "weight_vertex_index", "weight_control_index", "weight_value"}
        require(required.issubset(z.files), "FBX_RIG_SKIN_COURT_REPLAY_ARRAYS")
        mesh_index = np.asarray(z["weight_mesh_index"], dtype=np.int64)
        vertex_index = np.asarray(z["weight_vertex_index"], dtype=np.int64)
        control_index = np.asarray(z["weight_control_index"], dtype=np.int64)
        value = np.asarray(z["weight_value"], dtype=np.float64)
    require(mesh_index.shape == vertex_index.shape == control_index.shape == value.shape, "FBX_RIG_SKIN_COURT_REPLAY_WEIGHT_LAYOUT")
    require(np.isfinite(value).all() and np.all(value > 0), "FBX_RIG_SKIN_COURT_REPLAY_WEIGHT_VALUE")
    result = {}
    for mi, vi, ci, weight in zip(mesh_index, vertex_index, control_index, value):
        require(0 <= mi < len(meshes), "FBX_RIG_SKIN_COURT_REPLAY_MESH_RANGE")
        require(0 <= ci < len(controls), "FBX_RIG_SKIN_COURT_REPLAY_CONTROL_RANGE")
        mesh = meshes[int(mi)]
        require(0 <= vi < int(mesh["vertex_count"]), "FBX_RIG_SKIN_COURT_REPLAY_VERTEX_RANGE")
        control = controls[int(ci)]
        key = (str(mesh["name"]), int(vi), str(control["armature"]), str(control["name"]))
        require(key not in result, "FBX_RIG_SKIN_COURT_REPLAY_WEIGHT_DUPLICATE")
        result[key] = float(weight)
    return result


def validate(inspection_path: Path, replay_path: Path) -> dict:
    inspection_path = inspection_path.resolve()
    replay_path = replay_path.resolve()
    inspection = load(inspection_path)
    replay = load(replay_path)
    require(inspection.get("schema") == INSPECTION_SCHEMA, "FBX_RIG_SKIN_COURT_INSPECTION_SCHEMA")
    require(replay.get("schema") == REPLAY_SCHEMA and replay.get("status") == REPLAY_STATUS, "FBX_RIG_SKIN_COURT_REPLAY_SCHEMA")
    subject_id = str(inspection.get("subject_id") or "")
    require(subject_id and replay.get("subject_id") == subject_id, "FBX_RIG_SKIN_COURT_SUBJECT")

    source_a = bound(dict(inspection.get("source") or {}), "FBX_RIG_SKIN_COURT_INSPECTION_SOURCE")
    source_b = bound(dict(replay.get("source") or {}), "FBX_RIG_SKIN_COURT_REPLAY_SOURCE")
    require(source_a == source_b and source_a.suffix.lower() == ".fbx", "FBX_RIG_SKIN_COURT_SOURCE_IDENTITY")
    source_sha = sha256(source_a)
    require((inspection.get("source") or {}).get("sha256") == source_sha, "FBX_RIG_SKIN_COURT_SOURCE_SHA")

    import_a = dict(inspection.get("import") or {})
    import_b = dict(replay.get("import") or {})
    expected_options = {"automatic_bone_orientation": False, "use_anim": False}
    require(import_a.get("operation") == "FBX_IMPORT_WITHOUT_ANIMATION", "FBX_RIG_SKIN_COURT_INSPECTION_IMPORT")
    require(import_b.get("operation") == "FBX_IMPORT_WITHOUT_ANIMATION", "FBX_RIG_SKIN_COURT_REPLAY_IMPORT")
    require(dict(import_a.get("importer_options") or {}) == expected_options, "FBX_RIG_SKIN_COURT_INSPECTION_OPTIONS")
    require(dict(import_b.get("importer_options") or {}) == expected_options, "FBX_RIG_SKIN_COURT_REPLAY_OPTIONS")
    blender_a = str((inspection.get("runtime") or {}).get("blender") or "")
    blender_b = str(replay.get("blender_version") or "")
    require(blender_a and blender_a == blender_b, "FBX_RIG_SKIN_COURT_BLENDER_RUNTIME_DRIFT")

    controls_a = canonical_controls(list(inspection.get("controls") or []))
    controls_b = canonical_controls(list(replay.get("controls") or []))
    require(set(controls_a) == set(controls_b), "FBX_RIG_SKIN_COURT_CONTROL_IDENTITY")
    max_matrix_error = 0.0
    for key in controls_a:
        a, b = controls_a[key], controls_b[key]
        require(a["parent"] == b["parent"] and a["use_deform"] == b["use_deform"], "FBX_RIG_SKIN_COURT_CONTROL_HIERARCHY")
        error = float(np.max(np.abs(a["matrix"] - b["matrix"])))
        max_matrix_error = max(max_matrix_error, error)
        require(error <= 1e-8, "FBX_RIG_SKIN_COURT_CONTROL_MATRIX_DRIFT")

    meshes_a = {str(row["name"]): row for row in inspection.get("meshes") or []}
    meshes_b = {str(row["name"]): row for row in replay.get("meshes") or []}
    require(len(meshes_a) == len(inspection.get("meshes") or []) and len(meshes_b) == len(replay.get("meshes") or []), "FBX_RIG_SKIN_COURT_MESH_NAMES")
    require(set(meshes_a) == set(meshes_b), "FBX_RIG_SKIN_COURT_MESH_SET")
    skinned_vertex_count = 0
    for name in meshes_a:
        a = meshes_a[name]
        b = meshes_b[name]
        targets = list(a.get("armature_targets") or [])
        require(len(targets) <= 1, "FBX_RIG_SKIN_COURT_INSPECTION_MULTI_ARMATURE")
        target = targets[0] if targets else None
        require(target == b.get("armature_target"), "FBX_RIG_SKIN_COURT_MESH_ARMATURE")
        if target is not None:
            require(int(a["vertex_count"]) == int(b["vertex_count"]), "FBX_RIG_SKIN_COURT_SKINNED_VERTEX_COUNT")
            skinned_vertex_count += int(a["vertex_count"])
    require(skinned_vertex_count > 0, "FBX_RIG_SKIN_COURT_NO_SKINNED_VERTICES")

    inspection_arrays = bound(dict((inspection.get("outputs") or {}).get("rest_geometry") or {}), "FBX_RIG_SKIN_COURT_INSPECTION_ARRAY_DRIFT")
    replay_arrays = bound(dict(replay.get("arrays") or {}), "FBX_RIG_SKIN_COURT_REPLAY_ARRAY_DRIFT")
    weights_a, _ = inspection_weights(inspection, inspection_arrays)
    weights_b = replay_weights(replay, replay_arrays)
    require(set(weights_a) == set(weights_b), "FBX_RIG_SKIN_COURT_WEIGHT_ADDRESS_DRIFT")
    max_weight_error = 0.0
    for key, value in weights_a.items():
        error = abs(value - weights_b[key])
        max_weight_error = max(max_weight_error, error)
        require(error <= 2e-6, "FBX_RIG_SKIN_COURT_WEIGHT_VALUE_DRIFT")

    by_vertex: dict[tuple[str, int], float] = {}
    for mesh_name, vertex, _armature, _bone in weights_a:
        by_vertex[(mesh_name, vertex)] = by_vertex.get((mesh_name, vertex), 0.0) + weights_a[(mesh_name, vertex, _armature, _bone)]
    valid_weight_vertices = {
        key for key, total in by_vertex.items()
        if abs(total - 1.0) <= 2e-4
    }
    authored_weight_vertices = set(by_vertex)
    coverage = len(valid_weight_vertices) / skinned_vertex_count
    authored_coverage = len(authored_weight_vertices) / skinned_vertex_count
    maximum_sum_error = max((abs(total - 1.0) for total in by_vertex.values()), default=0.0)
    if coverage == 1.0:
        skin_outcome = "QUALIFIED_CORE_CANDIDATE"
    elif coverage > 0.0:
        skin_outcome = "QUALIFIED_MASKED_CANDIDATE"
    else:
        skin_outcome = "UNAVAILABLE_CANDIDATE"

    return {
        "schema": SCHEMA,
        "status": STATUS,
        "subject_id": subject_id,
        "source": {"path": str(source_a), "sha256": source_sha},
        "authority_scope": "PINNED_BLENDER_FBX_DECODED_SOURCE_SEMANTICS__NOT_RAW_FBX_PARSER_PROOF",
        "blender_version": blender_a,
        "counts": {
            "control_count": len(controls_a),
            "mesh_count": len(meshes_a),
            "skinned_vertex_count": skinned_vertex_count,
            "authored_weight_vertex_count": len(authored_weight_vertices),
            "simplex_valid_weight_vertex_count": len(valid_weight_vertices),
            "weight_row_count": len(weights_a),
        },
        "equivalence": {
            "control_identity_and_parent_exact": True,
            "maximum_rest_matrix_abs_error": max_matrix_error,
            "weight_addresses_exact": True,
            "maximum_weight_abs_error": max_weight_error,
            "semantic_skin_transfer_used": False,
            "nearest_neighbor_skin_transfer_used": False,
            "raw_fbx_parser_authority_claimed": False,
        },
        "skin_quality": {
            "authored_weight_coverage_fraction": authored_coverage,
            "simplex_valid_weight_coverage_fraction": coverage,
            "maximum_authored_weight_sum_error": maximum_sum_error,
            "invalid_or_missing_vertices_are_masked_not_zero_filled": True,
        },
        "family_candidate_outcomes": {
            "RIG_HIERARCHY": "QUALIFIED_CORE_CANDIDATE",
            "SKIN_WEIGHTS": skin_outcome,
        },
        "parents": {
            "inspection": {"path": str(inspection_path), "sha256": sha256(inspection_path)},
            "replay": {"path": str(replay_path), "sha256": sha256(replay_path)},
        },
        "authority": {
            "independent_replay_comparison_passed": True,
            "final_family_authority_minted": False,
            "training_authorized": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspection", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.inspection, args.replay)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "subject_id": result["subject_id"], "family_candidate_outcomes": result["family_candidate_outcomes"]}, sort_keys=True))


if __name__ == "__main__":
    main()
