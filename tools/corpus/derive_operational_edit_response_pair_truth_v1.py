#!/usr/bin/env python3
"""Extract source-verified, morphology-neutral edit-response relation targets.

This does NOT identify artist drawables. It derives a measurable pair relation:
two source surface loci share/differ in their FULL observed Boolean response
signature under the fixed, independently measured source intervention suite.

Input: *already independently courted* original source UV/pose anchor evidence.
The operation identity is privileged teacher metadata and MUST NOT enter
student inference features. Source face IDs/barycentrics select training queries,
not scene inputs. No new source pixels, renderer or ownership claims.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

SCHEMA = "RealSaS.OperationalEditResponsePairTruthCandidate.v1"
SOURCE_PARENT = "RealSaS.OriginalCatPhysicalPoseSubfaceUVAnchorTransport.v1"
SOURCE_STATUS = "SOURCE_NATIVE_POSE_SUBFACE_UV_ANCHOR_TRANSPORT_PRODUCED__INDEPENDENT_COURT_REQUIRED"
COURT_STATUS = "PASS_ORIGINAL_CAT_POSE0_POSE14_SAME_FACE_UV_ANCHOR_PHYSICAL_TRANSPORT__OWNER_UNKNOWN"
CANDIDATE_STATUS = "PASS_QUALIFIED_ORIGINAL_SOURCE_EDIT_RESPONSE_PAIR_FACTS__NOT_IRIS_TRAINING"

def require(ok: bool, why: str) -> None:
    if not ok:
        raise ValueError("OPERATIONAL_EDIT_PAIR_" + why)

def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def _finite_number(x) -> bool:
    import math
    return type(x) in (int, float) and math.isfinite(float(x))

def _locus(row: dict, *, teacher=None) -> dict:
    bary = row["pose0_center_source_barycentric"]
    require(isinstance(bary, list) and len(bary) == 3 and
            all(_finite_number(x) and -1e-8 <= x <= 1 + 1e-8 for x in bary)
            and abs(sum(bary) - 1) <= 1e-7,
            "INVALID_SOURCE_BARYCENTRIC")
    p0 = row["pose0_pixel_xy"]
    p14 = row["pose14_qualified_pixel_xy"]
    pt14 = row["pose14_transported_barycentric_screen_xy"]
    require(all(isinstance(z, list) and len(z) == 2 for z in (p0, p14, pt14))
            and all(type(x) is int and 0 <= x < 192 for z in (p0, p14) for x in z)
            and all(_finite_number(x) for x in pt14)
            and int(pt14[0]) == p14[0] and int(pt14[1]) == p14[1],
            "INVALID_SOURCE_POSE_ADDRESS")
    require(row["source_pose0_and_pose14_independent_physical_positive"] is True
            and row["same_source_face_UV_anchor"] is True
            and row["original_artist_drawable_owner"] == "UNKNOWN"
            and row["exact_Cycles_sample_ray_UV_correspondence"] ==
                "UNKNOWN_NOT_ASSERTED",
            "SOURCE_PROOF_OR_UNKNOWN_SCOPE_VIOLATED")
    locus = {
        "original_face_id_TEACHER_PRIVATE": row["original_face_id"],
        "source_barycentric_TEACHER_PRIVATE": bary,
        "source_view_index": row["original_camera_view"],
        "pose0_pixel_xy_TEACHER_PRIVATE": p0,
        "pose14_qualified_pixel_xy_TEACHER_PRIVATE": p14,
    }
    if teacher is not None:
        # Original teacher geometry produces supervised query addresses only,
        # not any source-edit operation ID in student features.
        positions, faces = teacher
        import numpy as np
        fi = row["original_face_id"]
        require(0 <= fi < len(faces), "SOURCE_FACE_OUT_OF_RANGE")
        ids = faces[fi]
        for pose in (0, 14):
            vertices = np.asarray(positions[pose, ids], dtype=np.float64)
            point = np.asarray(bary, dtype=np.float64) @ vertices
            require(np.isfinite(point).all(), "NONFINITE_TEACHER_QUERY")
            locus[f"source_world_xyz_pose{pose}_TEACHER_PRIVATE"] = point.tolist()
    return locus

def derive_pairs(anchors: list[dict], *, num_operations: int, teacher=None) -> list[dict]:
    """One positive per certified (face,view,op), one negative per distinct op pair.

    Both endpoints reside on the SAME original triangle at DISTINCT barycentric
    loci. This deliberately exercises a case the V3 *face-pair* truth schema
    cannot express. Positive means same measured edit signature, NOT owner SAME.
    """
    require(type(num_operations) is int and 2 <= num_operations <= 64,
            "CONTROL_CARDINALITY")
    groups = defaultdict(lambda: defaultdict(list))
    seen = set()
    for a in anchors:
        face = a.get("original_face_id")
        view = a.get("original_camera_view")
        op = a.get("original_uv_edit_index")
        require(all(type(v) is int and v >= 0 for v in (face, view, op))
                and op < num_operations, "SOURCE_HANDLE_INVALID")
        locus = _locus(a)
        key = (view, face, tuple(locus["pose0_pixel_xy_TEACHER_PRIVATE"]))
        require(key not in seen, "AMBIGUOUS_SOURCE_PIXEL_OPERATION")
        seen.add(key)
        groups[(view, face)][op].append(a)
    rows = []
    def add(a, b, value):
        la, lb = _locus(a, teacher=teacher), _locus(b, teacher=teacher)
        require(la["source_barycentric_TEACHER_PRIVATE"] != lb["source_barycentric_TEACHER_PRIVATE"],
                "IDENTICAL_SOURCE_LOCI")
        ops = (a["original_uv_edit_index"], b["original_uv_edit_index"])
        require((ops[0] == ops[1]) == bool(value),
                "EDIT_SIGNATURE_LABEL_INCONSISTENT")
        # The teacher *observed* full signatures through original BASE/SINGLE arms
        # with both independently checked Cycles offsets in BOTH authored poses.
        rows.append({
            "source_view_index": a["original_camera_view"],
            "source_pose_ordinals": [0, 14],
            "locus_a_TEACHER_PRIVATE": la,
            "locus_b_TEACHER_PRIVATE": lb,
            "observed_operation_indices_TEACHER_ONLY": list(ops),
            "measured_edit_signature_agreement": float(value),
            "validity_source_physical_both_poses": True,
            "evidence_kind": "SAME_SOURCE_FACE_TWO_DISTINCT_UV_LOCI_3_CONTROL_FULL_SIGNATURE",
            "same_drawable_owner": "UNKNOWN",
            "attachment_relation": "UNKNOWN",
            "proposed_compiler_cut": "UNDECIDED",
        })
    for (view, face) in sorted(groups):
        by_op = groups[(view, face)]
        for op in sorted(by_op):
            choices = sorted(by_op[op],
                             key=lambda r: (r["pose0_pixel_xy"][1], r["pose0_pixel_xy"][0]))
            if len(choices) > 1:
                add(choices[0], choices[-1], 1)
        ops = sorted(by_op)
        for i, aop in enumerate(ops):
            for bop in ops[i + 1:]:
                a = min(by_op[aop], key=lambda r: tuple(reversed(r["pose0_pixel_xy"])))
                b = min(by_op[bop], key=lambda r: tuple(reversed(r["pose0_pixel_xy"])))
                add(a, b, 0)
    return rows

def run(parent_path: Path, court_path: Path, out_path: Path,
        parent_sha: str, court_sha: str, num_operations: int,
        teacher_path: Path, teacher_sha: str) -> dict:
    require(digest(parent_path) == parent_sha, "PARENT_SHA_DRIFT")
    require(digest(court_path) == court_sha, "INDEPENDENT_COURT_SHA_DRIFT")
    require(digest(teacher_path) == teacher_sha and
            teacher_sha == "c2792c8efb36c4ccdf9178c6b933351b50364bbd28707e9365a3c115db00d603",
            "TEACHER_POSED_GEOMETRY_SHA_DRIFT")
    import numpy as np
    with np.load(teacher_path, allow_pickle=False) as raw_teacher:
        positions = np.asarray(raw_teacher["posed_vertices_source_world"], dtype=np.float64)
        faces = np.asarray(raw_teacher["faces"], dtype=np.int64)
    require(positions.shape == (32, 1204, 3) and faces.shape == (684, 3)
            and np.all((faces >= 0) & (faces < positions.shape[1]))
            and np.isfinite(positions).all(), "ORIGINAL_TEACHER_MESH_INVALID")
    source = json.loads(parent_path.read_text())
    court = json.loads(court_path.read_text())
    require(source.get("schema") == SOURCE_PARENT
            and source.get("status") == SOURCE_STATUS
            and court.get("status") == COURT_STATUS
            and source.get("original_source_sha256") ==
                court.get("original_source_sha256")
            and source.get("original_source_physical_64_cycles_frames_reused") is True
            and source.get("original_pose_ordinals") == [0, 14]
            and source.get("original_camera_views") == [0, 2, 4, 6]
            and source.get("same_cycles_sample_offsets") == [0, 16]
            and source.get("source_original_artist_layer_ownership") == "UNKNOWN"
            and court.get("original_source_artist_drawable_owner") == "UNKNOWN"
            and court.get("original_same_barycentric_UV_anchor_physical_transport_witnesses") ==
                len(source["original_source_transported_uv_anchors_PRIVATE"]),
            "SOURCE_CERTIFICATION_MISMATCH")
    for name in ("graphics_owner_authorized", "iris_training_authorized",
                 "training_authorized", "materialization_authorized"):
        require(source.get(name) is False and court.get(name) is False,
                "FORGED_PARENT_AUTHORITY:" + name)
    require(source["original_parent_bytes_sha256"]["teacher.npz"] == teacher_sha,
            "ORIGINAL_SOURCE_TEACHER_PARENT_DRIFT")
    pairs = derive_pairs(source["original_source_transported_uv_anchors_PRIVATE"],
                         num_operations=num_operations, teacher=(positions, faces))
    positives = sum(r["measured_edit_signature_agreement"] == 1.0 for r in pairs)
    negatives = len(pairs) - positives
    require(positives > 0 and negatives > 0, "MISSING_POSITIVE_OR_NEGATIVE_SOURCE_SUPPORT")
    result = {
        "schema": SCHEMA, "status": CANDIDATE_STATUS,
        "original_source_sha256": source["original_source_sha256"],
        "independent_source_parent_sha256": parent_sha,
        "independent_source_court_sha256": court_sha,
        "source_teacher_posed_geometry_sha256": teacher_sha,
        "query_xyz_coordinate_frame": "ORIGINAL_SOURCE_WORLD__CANONICAL_MODEL_NORMALIZATION_NOT_BOUND",
        "source_teacher_query_point_per_locus_both_poses": True,
        "measured_source_intervention_count": num_operations,
        "source_pose_ordinals": [0, 14], "original_camera_views": [0, 2, 4, 6],
        "pair_address_domain": "SAME_ORIGINAL_FACE_DISTINCT_BARYCENTRIC_SURFACE_LOCI",
        "relation_semantics": "EQUALITY_OF_OBSERVED_BOOLEAN_EDIT_EFFECT_SIGNATURE_IN_DECLARED_FINITE_CONTROL_SUITE",
        "source_evidence_valid": True,
        "iris_model_head_trained": False,
        "iris_training_authorized": False,
        "student_input_contains_teacher_op_id": False,
        "student_input_contains_teacher_face_id": False,
        "student_input_binding_qualified": False,
        "unconditioned_intrinsic_static_relation_label_valid": False,
        "student_intervention_conditioning_required": True,
        "declared_intervention_grammar": "SOURCE_ARBITRARY_UV_RECTANGLE_RGB_INVERT__THREE_FIXED_CONTROLS",
        "artist_drawable_owner_identified": False,
        "compiler_discrete_partition_decided": False,
        "appearance_parity_across_all_capabilities": False,
        "counts": {
            "source_qualified_transported_anchors": len(source["original_source_transported_uv_anchors_PRIVATE"]),
            "source_measured_same_signature_pairs": positives,
            "source_measured_different_signature_pairs": negatives,
            "source_pair_rows_total": len(pairs),
            "source_distinct_faces_with_pair_targets": len({r["locus_a_TEACHER_PRIVATE"]["original_face_id_TEACHER_PRIVATE"] for r in pairs}),
            "source_distinct_faces_with_different_signature": len({r["locus_a_TEACHER_PRIVATE"]["original_face_id_TEACHER_PRIVATE"] for r in pairs if r["measured_edit_signature_agreement"] == 0}),
        },
        "source_pair_training_CANDIDATES_PRIVATE": pairs,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": CANDIDATE_STATUS, "counts": result["counts"],
                      "iris_training_authorized": False}, sort_keys=True), flush=True)
    return result

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-parent", type=Path, required=True)
    ap.add_argument("--independent-court", type=Path, required=True)
    ap.add_argument("--source-parent-sha256", required=True)
    ap.add_argument("--independent-court-sha256", required=True)
    ap.add_argument("--operations", type=int, default=3)
    ap.add_argument("--teacher", type=Path, required=True)
    ap.add_argument("--teacher-sha256", required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    run(a.source_parent, a.independent_court, a.output,
        a.source_parent_sha256, a.independent_court_sha256, a.operations,
        a.teacher, a.teacher_sha256)
