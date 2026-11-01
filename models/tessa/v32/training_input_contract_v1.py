from __future__ import annotations

"""TESSA V3.2 static Corpus370 materialization contract.

The notebook may materialize the qualified mechanical teacher and its canonical
coordinate binding. It may NOT materialize or synthesize the student conditioning
Dense3D Z / predicted geometry G, because those are post-IRIS-training dependencies.
This validator therefore separates "static corpus shard ready" from "TESSA training
ready" explicitly.
"""

from pathlib import Path
from typing import Any

import numpy as np

from models.materialization_contract_v1 import (
    bound_file,
    load_json,
    load_npz,
    require,
    require_shape,
    validate_family_status,
    validate_subject_identity,
)

SCHEMA = "RealSaS.TESSAV32MaterializedSubject.v1"
TEACHER_TARGET_SCHEMA = "RealSaS.TESSAV32MechanicalTeacherTargets.v1"
TEACHER_MANIFEST_SCHEMA = "RealSaS.CorpusMechanicalTESSATeacher.v5"
TEACHER_MANIFEST_STATUS = "PASS_CORPUS_MECHANICAL_TESSA_TEACHER_V5_CANDIDATE__INDEPENDENT_COURT_REQUIRED"
TEACHER_COURT_SCHEMA = "RealSaS.CorpusMechanicalTESSATeacherCourt.v5"
TEACHER_COURT_STATUS = "PASS_CORPUS_MECHANICAL_TESSA_V5_CONSUMER_COURT__CAUSAL_RESPONSE_AUTHORITY_SEPARATE"
CANONICAL_BINDING_SCHEMA = "RealSaS.Corpus370CanonicalizationBinding.v1"

REQUIRED_PASS = (
    "SOURCE_IMPORT",
    "MECHANICAL_SURFACE_TOPOLOGY",
    "TESSA_TEACHER",
)


def _validate_authority(manifest: dict[str, Any]) -> None:
    authority = dict(manifest.get("teacher_authority") or {})
    parent_ref = dict(authority.get("teacher_manifest") or {})
    court_ref = dict(authority.get("teacher_court") or {})
    parent_path = bound_file(parent_ref, expected_schema=TEACHER_MANIFEST_SCHEMA, json_required=True)
    court_path = bound_file(court_ref, expected_schema=TEACHER_COURT_SCHEMA, json_required=True)
    parent = load_json(parent_path)
    court = load_json(court_path)
    require(parent.get("status") == TEACHER_MANIFEST_STATUS, "TESSA_V32_TEACHER_PARENT_STATUS")
    require(court.get("status") == TEACHER_COURT_STATUS, "TESSA_V32_TEACHER_COURT_STATUS")
    require(parent.get("subject_id") == manifest.get("subject_id") == court.get("subject_id"), "TESSA_V32_TEACHER_SUBJECT_BINDING")
    source_topology = dict(parent.get("source_topology") or {})
    require(source_topology.get("sdf_sign_topology_consumed") is False, "TESSA_V32_SDF_TOPOLOGY_CONTAMINATION")
    require(source_topology.get("appearance_addressing_consumed") is False, "TESSA_V32_APPEARANCE_ADDRESS_CONTAMINATION")
    require(source_topology.get("render_vertex_identity_used_as_topology") is False, "TESSA_V32_RENDER_VERTEX_TOPOLOGY_CONTAMINATION")
    require((court.get("authority") or {}).get("tessa_mechanical_surface_consumer_court_passed") is True, "TESSA_V32_TEACHER_COURT_NOT_PASS")


def _validate_canonical_binding(manifest: dict[str, Any]) -> None:
    ref = dict(manifest.get("canonicalization") or {})
    path = bound_file(ref, expected_schema=CANONICAL_BINDING_SCHEMA, json_required=True)
    payload = load_json(path)
    require(payload.get("subject_id") == manifest.get("subject_id"), "TESSA_V32_CANONICAL_BINDING_SUBJECT")
    require(payload.get("source_sha256") == manifest.get("source_sha256"), "TESSA_V32_CANONICAL_BINDING_SOURCE")
    matrix = np.asarray(payload.get("source_world_to_canonical"), dtype=np.float64)
    require(matrix.shape == (4, 4) and bool(np.isfinite(matrix).all()), "TESSA_V32_CANONICAL_BINDING_MATRIX")
    require(abs(float(np.linalg.det(matrix[:3, :3]))) > 1.0e-12, "TESSA_V32_CANONICAL_BINDING_SINGULAR")
    require(payload.get("camera_observation_authority_passed") is True, "TESSA_V32_CANONICAL_BINDING_CAMERA_AUTHORITY")
    require(payload.get("training_authorized") is False, "TESSA_V32_CANONICAL_BINDING_PREAUTHORIZED")


def _validate_teacher_targets(manifest: dict[str, Any]) -> dict[str, int]:
    _, arrays = load_npz(dict(manifest.get("teacher_targets") or {}), expected_schema=TEACHER_TARGET_SCHEMA)
    required = {"vertices_canonical", "faces", "source_triangle_id", "source_barycentric"}
    require(required.issubset(arrays), "TESSA_V32_TEACHER_TARGET_ARRAYS")
    vertices = np.asarray(arrays["vertices_canonical"], dtype=np.float64)
    faces = np.asarray(arrays["faces"], dtype=np.int64)
    source_triangle = np.asarray(arrays["source_triangle_id"], dtype=np.int64)
    barycentric = np.asarray(arrays["source_barycentric"], dtype=np.float64)
    require_shape(vertices, ndim=2, last_dim=3, code="TESSA_V32_TEACHER_VERTICES")
    require_shape(faces, ndim=2, last_dim=3, code="TESSA_V32_TEACHER_FACES")
    require(len(vertices) > 0 and len(faces) > 0, "TESSA_V32_TEACHER_EMPTY")
    require(bool(np.all(faces >= 0)) and int(faces.max(initial=-1)) < len(vertices), "TESSA_V32_TEACHER_FACE_RANGE")
    require(source_triangle.shape == (len(vertices),), "TESSA_V32_SOURCE_TRIANGLE_SHAPE")
    require(barycentric.shape == (len(vertices), 3), "TESSA_V32_SOURCE_BARYCENTRIC_SHAPE")
    require(bool(np.all(source_triangle >= 0)), "TESSA_V32_SOURCE_TRIANGLE_NEGATIVE")
    require(bool(np.all(barycentric >= -1.0e-6)), "TESSA_V32_SOURCE_BARYCENTRIC_NEGATIVE")
    require(bool(np.all(np.abs(barycentric.sum(axis=1) - 1.0) <= 1.0e-5)), "TESSA_V32_SOURCE_BARYCENTRIC_SIMPLEX")
    return {"teacher_vertex_count": int(len(vertices)), "teacher_face_count": int(len(faces))}


def validate_manifest(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    manifest = load_json(path)
    require(manifest.get("schema") == SCHEMA, "TESSA_V32_MATERIALIZED_SCHEMA")
    validate_subject_identity(manifest)
    statuses = validate_family_status(manifest, required_pass=REQUIRED_PASS)
    _validate_authority(manifest)
    _validate_canonical_binding(manifest)
    counts = _validate_teacher_targets(manifest)

    # Static truth must not impersonate future IRIS-produced conditioning.
    require(manifest.get("dense3d_z") in {None, False}, "TESSA_V32_STATIC_DENSE3D_Z_FORBIDDEN")
    require(manifest.get("predicted_geometry_g") in {None, False}, "TESSA_V32_STATIC_PREDICTED_G_FORBIDDEN")
    deps = dict(manifest.get("post_materialization_dependencies") or {})
    require(deps.get("frozen_iris_shared_scene_release_required") is True, "TESSA_V32_FROZEN_IRIS_DEPENDENCY")
    require(deps.get("qualified_geometry_g_from_frozen_iris_required") is True, "TESSA_V32_FROZEN_G_DEPENDENCY")
    require(deps.get("teacher_mesh_is_student_conditioning") is False, "TESSA_V32_TEACHER_CONDITIONING_FORBIDDEN")

    authority = dict(manifest.get("authority") or {})
    require(authority.get("matrix_cell_status_copied_not_recomputed") is True, "TESSA_V32_MATRIX_AUTHORITY_BINDING")
    require(authority.get("split_membership_copied_not_recomputed") is True, "TESSA_V32_SPLIT_AUTHORITY_BINDING")
    require(authority.get("sdf_sign_topology_consumed") is False, "TESSA_V32_SDF_SIGN_TOPOLOGY_FORBIDDEN")
    require(authority.get("appearance_address_consumed_as_topology") is False, "TESSA_V32_APPEARANCE_TOPOLOGY_FORBIDDEN")

    return {
        "schema": SCHEMA,
        "subject_id": manifest["subject_id"],
        "split": manifest["split"],
        "family_status": statuses,
        **counts,
        "static_corpus_shard_ready": True,
        "tessa_training_ready": False,
        "training_blockers": [
            "FROZEN_IRIS_SHARED_SCENE_RELEASE_REQUIRED",
            "QUALIFIED_GEOMETRY_G_FROM_FROZEN_IRIS_REQUIRED",
        ],
        "training_authorized": False,
    }
