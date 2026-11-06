#!/usr/bin/env python3
"""Source GLB UV -> original teacher canonical-face address alignment v0.

Research-only. No artist drawable ownership, inference authority, source claim
admission or Graphics production promotion. Intentionally abstains if original
triangle identity cannot be established from *two independently archived* inputs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tools.corpus.compile_corpus370_original_source_uv_edit_plan_v2 import (
    glb_document, accessor,
)

SCHEMA = "RealSaS.SourceFaceUVAlignment.v0"
NO_AUTHORITY = dict(
    graphics_owner_authorized=False,
    materialization_authorized=False,
    iris_training_authorized=False,
    tessa_training_authorized=False,
    training_authorized=False,
)


def sha256(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def source_arrays(glb):
    document, binary = glb_document(glb)
    primitives = [p for mesh in document.get("meshes", ())
                  for p in mesh.get("primitives", ())]
    # A single primitive permits an exact indexed teacher face witness. A
    # multi-primitive source requires a separate qualified vertex offset map.
    if len(primitives) != 1:
        return None, "MULTIPRIMITIVE_VERTEX_MAP_UNQUALIFIED"
    p = primitives[0]
    att = p.get("attributes", {})
    if p.get("mode", 4) != 4 or not all(
            k in att for k in ("POSITION", "TEXCOORD_0")) or "indices" not in p:
        return None, "GLB_TRIANGLE_TEXTURE_ATTRIBUTES_UNAVAILABLE"
    positions = accessor(document, binary, att["POSITION"])
    uv = accessor(document, binary, att["TEXCOORD_0"])
    indices = accessor(document, binary, p["indices"]).reshape(-1)
    if indices.size % 3 or positions.shape != (len(uv), 3) or uv.shape[1] != 2:
        return None, "GLB_VERTEX_UV_TOPOLOGY_INVALID"
    faces = indices.reshape(-1, 3).astype(np.int64)
    if np.any(faces < 0) or np.any(faces >= len(positions)):
        return None, "GLB_FACE_INDEX_OOB"
    if not np.all(np.isfinite(uv)) or np.any((uv < 0) | (uv > 1)):
        return None, "GLB_FACE_UV_ADDRESS_INVALID"
    return (positions.astype(np.float64), faces, uv[faces].astype(np.float64)), None


def _edge_distances(positions, faces):
    vertices = positions[faces]
    return np.stack([
        np.linalg.norm(vertices[:, 0] - vertices[:, 1], axis=1),
        np.linalg.norm(vertices[:, 1] - vertices[:, 2], axis=1),
        np.linalg.norm(vertices[:, 2] - vertices[:, 0], axis=1),
    ], axis=-1)


def audit_alignment(*, glb, teacher, expected_glb_sha, expected_teacher_sha,
                    tolerance=1e-5):
    if sha256(glb) != expected_glb_sha or sha256(teacher) != expected_teacher_sha:
        raise ValueError("UV_ALIGNMENT_PARENT_BYTE_SHA_DRIFT")
    result = {
        "schema": SCHEMA,
        "original_source_glb_sha256": expected_glb_sha,
        "original_source_teacher_sha256": expected_teacher_sha,
        "source_artist_owner": "UNKNOWN",
        "student_inference_qualified": False,
        "source_to_canonical_binding": "UNKNOWN",
        "reason": "",
        "source_face_count": None,
        "teacher_face_count": None,
        "source_uv_face_array_sha256": None,
        "teacher_pose_count": None,
        "maximum_normalized_edge_length_residual": None,
        **NO_AUTHORITY,
    }
    arrays, error = source_arrays(glb)
    if error:
        result["reason"] = error
        return result, None
    source_pos, source_faces, face_uv = arrays
    with np.load(teacher, allow_pickle=False) as rows:
        if not all(k in rows.files for k in (
                "faces", "rest_vertices_source_world",
                "posed_vertices_source_world")):
            result["reason"] = "TEACHER_ORIGINAL_GEOMETRY_MISSING"
            return result, None
        teacher_faces = np.asarray(rows["faces"], dtype=np.int64)
        rest = np.asarray(rows["rest_vertices_source_world"], dtype=np.float64)
        poses = np.asarray(rows["posed_vertices_source_world"], dtype=np.float64)
    result["source_face_count"] = int(len(source_faces))
    result["teacher_face_count"] = int(len(teacher_faces))
    result["teacher_pose_count"] = int(len(poses))
    if source_faces.shape != teacher_faces.shape or not np.array_equal(
            source_faces, teacher_faces):
        result["reason"] = "ORIGINAL_FACE_INDEX_ORDER_NOT_VERIFIED"
        return result, None
    if source_pos.shape != rest.shape or poses.ndim != 3 or poses.shape[1:] != rest.shape:
        result["reason"] = "ORIGINAL_SOURCE_VERTEX_IDENTITY_NOT_VERIFIED"
        return result, None
    if not np.isfinite(rest).all() or not np.isfinite(poses).all():
        result["reason"] = "TEACHER_POSITIONS_NONFINITE"
        return result, None
    # Distances are invariant to rigid global transforms. Permit only *one*
    # global positive uniform scale; per-vertex/nonrigid retargeting is not an
    # identity witness and must go through an independently courted binding.
    original_edges = _edge_distances(source_pos, source_faces)
    teacher_edges = _edge_distances(rest, teacher_faces)
    lengths = original_edges.ravel()
    if np.any(lengths <= 1e-12):
        result["reason"] = "ORIGINAL_SOURCE_DEGENERATE_TRIANGLE"
        return result, None
    scale = float(np.sum(lengths * teacher_edges.ravel()) / np.sum(lengths ** 2))
    if not np.isfinite(scale) or scale <= 0:
        result["reason"] = "SOURCE_TO_TEACHER_GLOBAL_SCALE_UNDETERMINED"
        return result, None
    residual = float(np.max(np.abs(teacher_edges - scale * original_edges)
                            / np.maximum(scale * original_edges, 1e-8)))
    result["maximum_normalized_edge_length_residual"] = residual
    if residual > tolerance:
        result["reason"] = "SOURCE_TO_TEACHER_GEOMETRIC_IDENTITY_UNDETERMINED"
        return result, None
    result["source_uv_face_array_sha256"] = hashlib.sha256(
        np.ascontiguousarray(face_uv, dtype="<f8").tobytes()).hexdigest()
    result["source_to_canonical_binding"] = "INDEX_AND_GLOBAL_REST_EDGE_METRIC_VERIFIED"
    result["reason"] = "INDEX_ALIGNED_SOURCE_TRIANGLE_UV__NOT_PUPPET_PARITY"
    return result, face_uv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", type=Path, required=True)
    ap.add_argument("--teacher", type=Path, required=True)
    ap.add_argument("--expected-glb-sha", required=True)
    ap.add_argument("--expected-teacher-sha", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--private-face-uv", type=Path, required=True)
    args = ap.parse_args()
    result, uv = audit_alignment(
        glb=args.glb, teacher=args.teacher,
        expected_glb_sha=args.expected_glb_sha,
        expected_teacher_sha=args.expected_teacher_sha)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if uv is not None:
        args.private_face_uv.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.private_face_uv, face_uv=uv)
    print(json.dumps({"status": result["source_to_canonical_binding"],
                      "reason": result["reason"],
                      "face_count": result["source_face_count"],
                      "owner": "UNKNOWN",
                      "training_authorized": False}, sort_keys=True))


if __name__ == "__main__":
    main()
