#!/usr/bin/env python3
"""Independent *derived-target* court, separate from the original physical court.

Reopens the immutable 43MB source evidence and its original independent source
court by SHA. Reconstructs expected same-face pair identities independently,
checks EVERY pair and both classes, and refuses any trainer/owner promotion.
The parent source physical first-hit, UV and Cycles verification belongs to the
already-PASS separate original court, not this script.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

STATUS = "PASS_SOURCE_PHYSICAL_EDIT_SIGNATURE_PAIRS__IRIS_CONSUMER_PENDING"
PARENT_SHA = "974c1a2a4cdbd51f2d799cc29a8d3b4ef3bbb36ba6442a3cc79c8f541d44ca6b"
COURT_SHA = "3ab8cbedd37d31e3fed6ddb554fab6fe5c80294076b7aaf94897b713e4bdad8c"

def require(ok, error):
    if not ok:
        raise ValueError("INDEPENDENT_EDIT_SIGNATURE_PAIR_" + error)

def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def address(a):
    return (a["original_camera_view"], a["original_face_id"],
            a["original_uv_edit_index"], tuple(a["pose0_pixel_xy"]))

def expected_from_original(anchors, num_ops):
    groups = defaultdict(dict)
    for a in anchors:
        view, face, op, xy = address(a)
        require(0 <= op < num_ops, "UNKNOWN_INTERVENTION")
        require(xy not in groups[(view, face)].get(op, {}), "DUPLICATE_SOURCE_ADDRESS")
        groups[(view, face)].setdefault(op, {})[xy] = a
    expected = Counter()
    for (view, face), operations in groups.items():
        representatives = {}
        for op, pixels in sorted(operations.items()):
            ordered = sorted(pixels, key=lambda xy: (xy[1], xy[0]))
            representatives[op] = ordered[0]
            if len(ordered) > 1:
                k = (view, face, op, ordered[0], op, ordered[-1], 1)
                expected[k] += 1
        for index, op1 in enumerate(sorted(representatives)):
            for op2 in sorted(representatives)[index+1:]:
                expected[(view, face, op1, representatives[op1],
                          op2, representatives[op2], 0)] += 1
    return expected

def audit(source_path, source_court_path, target_path, output_path, teacher_path):
    require(sha(source_path) == PARENT_SHA, "ORIGINAL_PHYSICAL_SOURCE_PARENT_SHA")
    require(sha(source_court_path) == COURT_SHA, "ORIGINAL_PHYSICAL_SOURCE_COURT_SHA")
    teacher_sha = "c2792c8efb36c4ccdf9178c6b933351b50364bbd28707e9365a3c115db00d603"
    require(sha(teacher_path) == teacher_sha, "ORIGINAL_TEACHER_SHA")
    import numpy as np
    with np.load(teacher_path, allow_pickle=False) as teacher:
        posed = np.asarray(teacher["posed_vertices_source_world"], dtype=np.float64)
        faces = np.asarray(teacher["faces"], dtype=np.int64)
    require(posed.shape == (32, 1204, 3) and faces.shape == (684, 3),
            "ORIGINAL_TEACHER_SHAPE")
    source = json.loads(Path(source_path).read_text())
    court = json.loads(Path(source_court_path).read_text())
    target = json.loads(Path(target_path).read_text())
    require(court.get("status") ==
            "PASS_ORIGINAL_CAT_POSE0_POSE14_SAME_FACE_UV_ANCHOR_PHYSICAL_TRANSPORT__OWNER_UNKNOWN",
            "SOURCE_PHYSICAL_COURT_NOT_PASS")
    require(target.get("schema") == "RealSaS.OperationalEditResponsePairTruthCandidate.v1"
            and target.get("status") ==
                "PASS_QUALIFIED_ORIGINAL_SOURCE_EDIT_RESPONSE_PAIR_FACTS__NOT_IRIS_TRAINING"
            and target.get("original_source_sha256") ==
                source.get("original_source_sha256") ==
                court.get("original_source_sha256")
            and target.get("independent_source_parent_sha256") == PARENT_SHA
            and target.get("independent_source_court_sha256") == COURT_SHA
            and target.get("source_pose_ordinals") == [0, 14]
            and target.get("original_camera_views") == [0, 2, 4, 6]
            and target.get("source_evidence_valid") is True
            and target.get("unconditioned_intrinsic_static_relation_label_valid") is False
            and target.get("student_intervention_conditioning_required") is True
            and target.get("declared_intervention_grammar") ==
                "SOURCE_ARBITRARY_UV_RECTANGLE_RGB_INVERT__THREE_FIXED_CONTROLS"
            and target.get("source_teacher_posed_geometry_sha256") == teacher_sha
            and target.get("source_teacher_query_point_per_locus_both_poses") is True
            and target.get("query_xyz_coordinate_frame") ==
                "ORIGINAL_SOURCE_WORLD__CANONICAL_MODEL_NORMALIZATION_NOT_BOUND",
            "DERIVED_SOURCE_PARENT_IDENTITY")
    for key in ("iris_training_authorized", "iris_model_head_trained",
                "student_input_contains_teacher_op_id",
                "student_input_contains_teacher_face_id",
                "student_input_binding_qualified",
                "artist_drawable_owner_identified",
                "compiler_discrete_partition_decided",
                "appearance_parity_across_all_capabilities"):
        require(target.get(key) is False, "INVALID_PROMOTION:" + key)
    num_ops = target.get("measured_source_intervention_count")
    require(type(num_ops) is int and 2 <= num_ops <= 64, "INTERVENTION_COUNT")
    anchors = source["original_source_transported_uv_anchors_PRIVATE"]
    expected = expected_from_original(anchors, num_ops)
    lookups = {address(a): a for a in anchors}
    observed = Counter()
    for p in target["source_pair_training_CANDIDATES_PRIVATE"]:
        a, b = p["locus_a_TEACHER_PRIVATE"], p["locus_b_TEACHER_PRIVATE"]
        ops = p["observed_operation_indices_TEACHER_ONLY"]
        view = p["source_view_index"]
        face = a["original_face_id_TEACHER_PRIVATE"]
        require(face == b["original_face_id_TEACHER_PRIVATE"]
                and view == a["source_view_index"] == b["source_view_index"],
                "NOT_A_SAME_FACE_VIEW_PAIR")
        key_a = (view, face, ops[0], tuple(a["pose0_pixel_xy_TEACHER_PRIVATE"]))
        key_b = (view, face, ops[1], tuple(b["pose0_pixel_xy_TEACHER_PRIVATE"]))
        require(key_a in lookups and key_b in lookups and key_a != key_b,
                "SOURCE_PAIR_NOT_CERTIFIED")
        pa, pb = lookups[key_a], lookups[key_b]
        for locus, original in ((a, pa), (b, pb)):
            require(locus["source_barycentric_TEACHER_PRIVATE"] ==
                    original["pose0_center_source_barycentric"]
                    and locus["pose14_qualified_pixel_xy_TEACHER_PRIVATE"] ==
                        original["pose14_qualified_pixel_xy"]
                    and original["source_pose0_and_pose14_independent_physical_positive"] is True
                    and original["original_artist_drawable_owner"] == "UNKNOWN",
                    "SOURCE_BARYCENTRIC_OR_PHYSICAL_EVIDENCE_INVALID")
            # Independent exact 3D point reconstruction: original face vertex
            # identities and each original native authored pose, not screen
            # backprojection or a student-inferred source position.
            ids = faces[face]
            bary = original["pose0_center_source_barycentric"]
            for pose in (0, 14):
                point = [
                    sum(float(bary[k]) * float(posed[pose, ids[k], axis])
                        for k in range(3))
                    for axis in range(3)
                ]
                declared = locus[f"source_world_xyz_pose{pose}_TEACHER_PRIVATE"]
                require(len(declared) == 3 and
                        max(abs(float(x) - float(y)) for x, y in zip(declared, point)) <= 1e-10,
                        "SOURCE_WORLD_POINT_INDEPENDENT_GEOMETRY_BINDING")
        label = p["measured_edit_signature_agreement"]
        require(label in (0.0, 1.0) and label == float(ops[0] == ops[1])
                and p["validity_source_physical_both_poses"] is True
                and p["same_drawable_owner"] == "UNKNOWN"
                and p["attachment_relation"] == "UNKNOWN"
                and p["proposed_compiler_cut"] == "UNDECIDED",
                "NOT_A_SOURCE_OPERATION_SIGNATURE_FACT")
        observed[(view, face, ops[0], key_a[-1], ops[1], key_b[-1], int(label))] += 1
    require(observed == expected, "PAIR_COMPLETENESS_OR_FORGED_SOURCE_LABEL")
    pos = sum(n for k, n in observed.items() if k[-1] == 1)
    neg = sum(n for k, n in observed.items() if k[-1] == 0)
    require(pos > 0 and neg > 0, "MISSING_POSITIVE_OR_NEGATIVE")
    faces_all = {k[1] for k in observed}
    faces_neg = {k[1] for k in observed if k[-1] == 0}
    c = target["counts"]
    require(c == {
        "source_qualified_transported_anchors": len(anchors),
        "source_measured_same_signature_pairs": pos,
        "source_measured_different_signature_pairs": neg,
        "source_pair_rows_total": pos + neg,
        "source_distinct_faces_with_pair_targets": len(faces_all),
        "source_distinct_faces_with_different_signature": len(faces_neg),
    }, "TARGET_COUNTS_FORGED")
    result = {
        "schema": "RealSaS.IndependentOperationalEditSignaturePairCourt.v1",
        "status": STATUS,
        "original_source_sha256": source["original_source_sha256"],
        "source_parent_sha256": PARENT_SHA,
        "source_court_sha256": COURT_SHA,
        "source_teacher_posed_geometry_sha256": teacher_sha,
        "source_world_points_recomputed_both_poses": True,
        "canonical_normalization_bound": False,
        "derived_target_sha256": sha(target_path),
        "counts": c,
        "same_face_different_UV_loci_checked": pos + neg,
        "teacher_edit_id_is_never_student_feature": True,
        "original_source_physically_observed": True,
        "physical_source_recomputed_by_this_court": False,
        "iris_training_authorized": False,
        "compiler_partition_authorized": False,
        "artist_drawable_owner": "UNKNOWN",
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": STATUS, "counts": c,
                      "training": False}, sort_keys=True), flush=True)
    return result

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-parent", type=Path, required=True)
    ap.add_argument("--source-court", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--teacher", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    audit(a.source_parent, a.source_court, a.candidate, a.output, a.teacher)
