from __future__ import annotations

"""Generic source/import/rest-scene admission binding for V2 Stage03/04.

This module validates exact external evidence. It does not create source truth,
repair a scene, infer attachments, or grant downstream model/product authority.
"""

import json
import math
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    load_file_ref,
    sha256_file,
)

PROFILE = "SOURCE_SCENE_BIND_V2"
ADMISSION_MODE = "CONTROLLED_FULL_SUBJECT_8VIEW_V2_SOURCE_SCENE"
COURT_SCHEMA = "RealSaS.SourceBindObservationCourt.v1"
CANDIDATE_SCHEMA = "RealSaS.Controlled8ViewSourceRestCandidate.v2"
SNAPSHOT_SCHEMA = "RealSaS.ImportedSourceSceneSnapshot.v2"
CAMERA_SCHEMA = "RealSaS.CameraProjectionBundle.v1"


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise QualificationError(code)


def _matrix(value, code: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    _require(matrix.shape == (4, 4), code)
    _require(bool(np.isfinite(matrix).all()), code)
    _require(bool(np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-10, rtol=0)), code)
    _require(np.linalg.matrix_rank(matrix[:3, :3]) == 3, code)
    return matrix


def _validate_snapshot(snapshot: dict, tolerance: float) -> dict:
    _require(snapshot.get("schema") == SNAPSHOT_SCHEMA, "SOURCE_SCENE_SNAPSHOT_SCHEMA")
    _require(math.isfinite(tolerance) and tolerance > 0, "SOURCE_SCENE_TOLERANCE")
    nodes = list(snapshot.get("nodes") or ())
    _require(bool(nodes), "SOURCE_SCENE_EMPTY")
    index = {str(row.get("node_id")): row for row in nodes}
    _require(len(index) == len(nodes) and "" not in index, "SOURCE_SCENE_NODE_IDENTITY")
    norm = _matrix(snapshot.get("source_world_to_canonical"), "SOURCE_SCENE_NORMALIZATION")
    inverse = _matrix(snapshot.get("canonical_to_source_world"), "SOURCE_SCENE_NORMALIZATION_INVERSE")
    _require(np.allclose(norm @ inverse, np.eye(4), atol=tolerance, rtol=0),
             "SOURCE_SCENE_NORMALIZATION_INVERSE")
    gram = norm[:3, :3].T @ norm[:3, :3]
    _require(np.allclose(gram / np.trace(gram), np.eye(3) / 3, atol=tolerance, rtol=0),
             "SOURCE_SCENE_NONUNIFORM_NORMALIZATION")
    _require(np.linalg.slogdet(norm[:3, :3])[0] > 0, "SOURCE_SCENE_REFLECTED_NORMALIZATION")
    maximum_residual = 0.0
    attachment_relations = 0
    for row in nodes:
        node_id = str(row["node_id"])
        world = _matrix(row.get("node_to_source_world"), "SOURCE_SCENE_WORLD_MATRIX")
        local = _matrix(row.get("effective_node_to_parent"), "SOURCE_SCENE_LOCAL_MATRIX")
        parent_id = row.get("parent_node_id")
        seen = {node_id}
        current = parent_id
        while current is not None:
            _require(current in index, "SOURCE_SCENE_PARENT_MISSING")
            _require(current not in seen, "SOURCE_SCENE_PARENT_CYCLE")
            seen.add(current)
            current = index[current].get("parent_node_id")
        parent = np.eye(4) if parent_id is None else _matrix(
            index[parent_id].get("node_to_source_world"), "SOURCE_SCENE_PARENT_WORLD_MATRIX"
        )
        residual = float(np.max(np.abs(world - parent @ local)))
        maximum_residual = max(maximum_residual, residual)
        _require(residual <= tolerance * max(1.0, float(np.max(np.abs(world)))),
                 "SOURCE_SCENE_PARENT_WORLD_DRIFT")
        canonical = _matrix(row.get("node_to_canonical"), "SOURCE_SCENE_CANONICAL_MATRIX")
        _require(np.allclose(canonical, norm @ world, atol=tolerance, rtol=tolerance),
                 "SOURCE_SCENE_CANONICAL_BINDING_DRIFT")
        _require(int(row.get("active_action_count", 0)) == 0, "SOURCE_SCENE_REST_HAS_LIVE_ACTION")
        _require(int(row.get("active_nla_track_count", 0)) == 0, "SOURCE_SCENE_REST_HAS_LIVE_NLA")
        _require(int(row.get("driver_count", 0)) == 0, "SOURCE_SCENE_REST_HAS_LIVE_DRIVERS")
        if parent_id is not None or row.get("parent_bone"):
            attachment_relations += 1
    return {
        "node_count": len(nodes),
        "bone_count": len(snapshot.get("bones") or ()),
        "attachment_relation_count": attachment_relations,
        "max_parent_world_residual": maximum_residual,
    }


def _sealed_hashes(source_payload: dict) -> tuple[dict[str, str], set[str]]:
    roles = {}
    hashes = set()
    for row in source_payload.get("files") or ():
        role = str(row.get("role") or "")
        digest = str(row.get("sha256") or "")
        _require(role and len(digest) == 64 and role not in roles, "SOURCE_SCENE_STAGE1_ROLE_INVALID")
        roles[role] = digest
        hashes.add(digest)
    return roles, hashes


def _ref_sha(ref: dict) -> str:
    digest = str(ref.get("sha256") or "")
    _require(len(digest) == 64, "SOURCE_SCENE_REF_SHA_REQUIRED")
    return digest


def _observation_hashes(ctx: dict) -> tuple[dict[int, str], dict[int, str], str]:
    observation = dict(ctx["run_manifest"].get("observation") or {})
    rasters = list(observation.get("source_rasters") or ())
    masks = list(observation.get("source_foreground_masks") or ())
    _require(len(rasters) == 8 and len(masks) == 8, "SOURCE_SCENE_EXACT_8_OBSERVATIONS_REQUIRED")
    raster_hashes = {int(row["view_index"]): _ref_sha(dict(row.get("image") or {})) for row in rasters}
    mask_hashes = {int(row["view_index"]): _ref_sha(dict(row.get("mask") or {})) for row in masks}
    _require(set(raster_hashes) == set(range(8)) and set(mask_hashes) == set(range(8)),
             "SOURCE_SCENE_VIEW_INDEX_SET")
    camera_ref = dict(observation.get("camera_bundle") or {})
    camera_sha = _ref_sha(camera_ref)
    return raster_hashes, mask_hashes, camera_sha


def validate_source_scene_admission(ctx: dict, source_payload: dict) -> dict | None:
    cfg = dict(ctx["run_manifest"].get("source_audit") or {})
    profile = str(cfg.get("profile") or "")
    if not profile:
        return None
    _require(profile == PROFILE, "SOURCE_SCENE_PROFILE_UNKNOWN")
    subject_id = str(ctx["run_manifest"].get("subject_id") or "")
    _require(subject_id, "SOURCE_SCENE_SUBJECT_ID_REQUIRED")
    primary_role = str(cfg.get("primary_source_role") or "")
    _require(primary_role, "SOURCE_SCENE_PRIMARY_SOURCE_ROLE_REQUIRED")
    roles, sealed_hashes = _sealed_hashes(source_payload)
    _require(primary_role in roles, "SOURCE_SCENE_PRIMARY_SOURCE_NOT_STAGE1_SEALED")
    primary_sha = roles[primary_role]

    candidate_ref = dict(cfg.get("source_rest_candidate") or {})
    snapshot_ref = dict(cfg.get("source_scene_snapshot") or {})
    court_ref = dict(cfg.get("source_bind_court") or {})
    candidate_sha, snapshot_sha, court_sha = map(_ref_sha, (candidate_ref, snapshot_ref, court_ref))
    for digest in (candidate_sha, snapshot_sha, court_sha):
        _require(digest in sealed_hashes, "SOURCE_SCENE_EVIDENCE_NOT_STAGE1_SEALED")

    candidate = load_file_ref(candidate_ref, expected_schema=CANDIDATE_SCHEMA)
    snapshot = load_file_ref(snapshot_ref, expected_schema=SNAPSHOT_SCHEMA)
    court = load_file_ref(court_ref, expected_schema=COURT_SCHEMA)
    _require(str(candidate.get("subject_id")) == subject_id, "SOURCE_SCENE_CANDIDATE_SUBJECT_DRIFT")
    _require(str(candidate.get("source_sha256")) == primary_sha, "SOURCE_SCENE_CANDIDATE_SOURCE_DRIFT")
    _require(str(snapshot.get("source_sha256")) == primary_sha, "SOURCE_SCENE_SNAPSHOT_SOURCE_DRIFT")
    _require(str(court.get("subject_id")) == subject_id, "SOURCE_SCENE_COURT_SUBJECT_DRIFT")
    _require(str(court.get("source_sha256")) == primary_sha, "SOURCE_SCENE_COURT_SOURCE_DRIFT")
    _require(str((candidate.get("source_scene_snapshot") or {}).get("sha256")) == snapshot_sha,
             "SOURCE_SCENE_SNAPSHOT_BINDING_DRIFT")
    _require(str(court.get("candidate_receipt_sha256")) == candidate_sha,
             "SOURCE_SCENE_COURT_CANDIDATE_DRIFT")
    _require(str(court.get("scene_snapshot_sha256")) == snapshot_sha,
             "SOURCE_SCENE_COURT_SNAPSHOT_DRIFT")

    _require(candidate.get("raw_source_mutated", candidate.get("import", {}).get("raw_source_mutated")) is False,
             "SOURCE_SCENE_RAW_SOURCE_MUTATED")
    _require(candidate.get("pose_address") == "SOURCE_BIND_REST", "SOURCE_SCENE_REST_POSE_ADDRESS")
    _require(candidate.get("native_frame0_is_rest_assumed") is False, "SOURCE_SCENE_FRAME0_REST_ASSUMPTION")
    _require(candidate.get("training_admission_claimed") is False and candidate.get("product_pass_claimed") is False,
             "SOURCE_SCENE_CANDIDATE_UNAUTHORIZED_CLAIM")
    _require((candidate.get("internal_consistency") or {}).get("animation_free_rest") is True,
             "SOURCE_SCENE_CANDIDATE_NOT_ANIMATION_FREE")
    _require((candidate.get("internal_consistency") or {}).get("normalization_uniform") is True,
             "SOURCE_SCENE_CANDIDATE_NORMALIZATION_UNQUALIFIED")

    tolerance = float(cfg.get("matrix_tolerance", 1e-6))
    snapshot_stats = _validate_snapshot(snapshot, tolerance)
    raster_hashes, mask_hashes, camera_sha = _observation_hashes(ctx)
    _require(camera_sha in sealed_hashes, "SOURCE_SCENE_CAMERA_NOT_STAGE1_SEALED")
    camera = load_file_ref(dict(ctx["run_manifest"]["observation"]["camera_bundle"]), expected_schema=CAMERA_SCHEMA)
    _require(str((candidate.get("camera_bundle") or {}).get("sha256")) == camera_sha,
             "SOURCE_SCENE_CAMERA_BINDING_DRIFT")
    _require(camera.get("per_view_recenter") is False and camera.get("per_view_pose_change") is False,
             "SOURCE_SCENE_PER_VIEW_CAMERA_OR_POSE_CHANGE")
    _require(len(camera.get("cameras") or ()) == 8, "SOURCE_SCENE_CAMERA_COUNT")
    _require(math.isfinite(float(camera.get("common_half_extent", 0))) and float(camera.get("common_half_extent", 0)) > 0,
             "SOURCE_SCENE_CAMERA_COMMON_EXTENT")
    center = camera.get("common_center_xyz") or ()
    _require(len(center) == 3 and all(math.isfinite(float(x)) for x in center), "SOURCE_SCENE_CAMERA_COMMON_CENTER")

    candidate_views = list(candidate.get("views") or ())
    _require(len(candidate_views) == 8, "SOURCE_SCENE_CANDIDATE_VIEW_COUNT")
    by_view = {int(row.get("view_index", -1)): row for row in candidate_views}
    _require(set(by_view) == set(range(8)), "SOURCE_SCENE_CANDIDATE_VIEW_INDEX_SET")
    for vi in range(8):
        view = by_view[vi]
        _require(str((view.get("raster") or {}).get("sha256")) == raster_hashes[vi],
                 "SOURCE_SCENE_RASTER_BINDING_DRIFT")
        _require(str((view.get("foreground_mask") or {}).get("sha256")) == mask_hashes[vi],
                 "SOURCE_SCENE_MASK_BINDING_DRIFT")
        _require(raster_hashes[vi] in sealed_hashes and mask_hashes[vi] in sealed_hashes,
                 "SOURCE_SCENE_OBSERVATION_NOT_STAGE1_SEALED")

    _require(court.get("status") == "PASS", "SOURCE_SCENE_BIND_COURT_NOT_PASS")
    _require(court.get("independent_raw_parser") is True, "SOURCE_SCENE_BIND_COURT_NOT_INDEPENDENT")
    _require(court.get("source_bind_parity") is True, "SOURCE_SCENE_BIND_PARITY_NOT_PROVEN")
    relative_vertex_error = float(court.get("relative_vertex_error", math.inf))
    maximum_mask_gap_px = float(court.get("maximum_mask_gap_px", math.inf))
    max_vertex_error = float(cfg.get("max_relative_vertex_error", 1e-5))
    max_mask_gap = float(cfg.get("max_mask_gap_px", 2.0))
    _require(0 <= relative_vertex_error <= max_vertex_error, "SOURCE_SCENE_VERTEX_PARITY_FAIL")
    _require(0 <= maximum_mask_gap_px <= max_mask_gap, "SOURCE_SCENE_MASK_PARITY_FAIL")

    return {
        "schema": "RealSaS.SourceSceneAdmissionBinding.v2",
        "profile": PROFILE,
        "subject_id": subject_id,
        "primary_source_sha256": primary_sha,
        "candidate_receipt_sha256": candidate_sha,
        "scene_snapshot_sha256": snapshot_sha,
        "bind_court_sha256": court_sha,
        "camera_bundle_sha256": camera_sha,
        "node_count": snapshot_stats["node_count"],
        "bone_count": snapshot_stats["bone_count"],
        "attachment_relation_count": snapshot_stats["attachment_relation_count"],
        "max_parent_world_residual": snapshot_stats["max_parent_world_residual"],
        "relative_vertex_error": relative_vertex_error,
        "maximum_mask_gap_px": maximum_mask_gap_px,
        "thresholds": {
            "matrix_tolerance": tolerance,
            "max_relative_vertex_error": max_vertex_error,
            "max_mask_gap_px": max_mask_gap,
        },
        "scope": "SOURCE_IMPORT_BIND_REST_SCENE_AND_CONTROLLED_OBSERVATION_ONLY",
        "native_motion_qualified": False,
        "appearance_qualified": False,
        "product_pass_claimed": False,
    }
