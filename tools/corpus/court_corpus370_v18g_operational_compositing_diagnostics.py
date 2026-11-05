#!/usr/bin/env python3
"""Independent, immutable-read V18g operational alpha compositing diagnosis.

Tests whether the *physically rendered* candidate group and its physical
complement can reconstruct the original source RGBA under either standard
premultiplied Porter-Duff OVER ordering. Reuses exact V18f render assets;
makes ZERO new renders and ZERO SAME/DIF artist/graphics ownership labels.

This diagnostic is a falsification instrument, not an artist layer teacher:
trivial K=1 partitions, invisible/degenerate complement, or absent
order-discriminating pixels cannot qualify a layer representation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.corpus.compile_corpus370_v18a_reactive_staining import THRESHOLDS
from tools.corpus.court_corpus370_v18f_full_physical_edit_benchmark import (
    RENDER as PARENT_RENDER, SCHEMA as PARENT_COURT, PASS as PARENT_PASS,
    expected_arms, TARGET_INDICES, VIEWS, TOL,
)

SCHEMA = "RealSaS.Corpus370IndependentOriginalPhysicalCompositingDiagnostics.v18g"
STATUS = "ORIGINAL_SOURCE_PHYSICAL_LAYER_COMPLEMENT_ALPHA_COMPOSING_DIAGNOSED__NO_OWNER_TRUTH"
RGBA_MEMBER = "original_stain_editor_rgba"
MEASURED_POSE = 2
REQUIRED_V18F_RUN = "42114401233-1"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def validate_rgba(array):
    x = np.asarray(array, np.float32)
    require(x.ndim == 3 and x.shape == (192, 192, 4)
            and np.isfinite(x).all()
            and np.min(x[..., 3]) >= -1e-5
            and np.max(x[..., 3]) <= 1.00001,
            "V18G_SOURCE_RGBA_NOT_VALID")
    return x


def packed_rgba(array):
    """Convert source RGBA assumed straight RGB to premultiplied RGB+alpha."""
    x = validate_rgba(array)
    return np.concatenate((x[..., :3] * x[..., 3:], x[..., 3:]), axis=-1)


def over_premul(top, bottom):
    top = validate_rgba(top)
    bottom = validate_rgba(bottom)
    ta = np.clip(top[..., 3:], 0., 1.)
    bp = packed_rgba(bottom)
    tp = packed_rgba(top)
    return tp + bp * (1. - ta)


def maximum_premul_rgba_residual(lhs, rhs_packed):
    ref = packed_rgba(lhs)
    require(rhs_packed.shape == ref.shape and np.isfinite(rhs_packed).all(),
            "V18G_COMPOSITE_INVALID")
    return np.max(np.abs(ref - rhs_packed), axis=-1)


def metrics_from_residual(residual):
    require(residual.shape == (192, 192) and np.isfinite(residual).all(),
            "V18G_RESIDUAL_INVALID")
    return {
        "p99_premultiplied_rgba_error": float(np.percentile(residual, 99)),
        "max_premultiplied_rgba_error": float(residual.max()),
        "pixel_count_above_original_tolerance": int(np.count_nonzero(residual > TOL)),
    }


def compositor_probe(original, isolated_group, removed_group):
    original = validate_rgba(original)
    isolated_group = validate_rgba(isolated_group)
    removed_group = validate_rgba(removed_group)
    # Image is source of truth; each arm has been physically rendered, not
    # partition-reconstructed or hallucinated. Same camera, same native pose.
    group_over_remaining = over_premul(isolated_group, removed_group)
    remaining_over_group = over_premul(removed_group, isolated_group)
    forward = maximum_premul_rgba_residual(original, group_over_remaining)
    backward = maximum_premul_rgba_residual(original, remaining_over_group)
    oracle_best = np.minimum(forward, backward)
    alpha_group = isolated_group[..., 3]
    alpha_remain = removed_group[..., 3]
    overlap = (alpha_group > TOL) & (alpha_remain > TOL)
    effect_orders = np.max(np.abs(group_over_remaining - remaining_over_group), axis=-1)
    informative = overlap & (effect_orders > TOL)
    g_only_ok = informative & (forward <= TOL) & (backward > TOL)
    r_only_ok = informative & (backward <= TOL) & (forward > TOL)
    both_bad = informative & (forward > TOL) & (backward > TOL)
    both_ok = informative & (forward <= TOL) & (backward <= TOL)
    return {
        "group_alone_alpha_visible_pixels": int(np.count_nonzero(alpha_group > TOL)),
        "remainder_alone_alpha_visible_pixels": int(np.count_nonzero(alpha_remain > TOL)),
        "both_layers_alpha_overlap_pixels": int(np.count_nonzero(overlap)),
        "order_discriminating_overlap_pixels": int(np.count_nonzero(informative)),
        "group_over_remainder": metrics_from_residual(forward),
        "remainder_over_group": metrics_from_residual(backward),
        "per_pixel_order_oracle": metrics_from_residual(oracle_best),
        "informative_pixels_support_group_over_only": int(np.count_nonzero(g_only_ok)),
        "informative_pixels_support_remainder_over_only": int(np.count_nonzero(r_only_ok)),
        "informative_pixels_both_orders_fail": int(np.count_nonzero(both_bad)),
        "informative_pixels_both_orders_pass": int(np.count_nonzero(both_ok)),
        "draw_order_observable":
            bool(informative.any()),
        "draw_order_one_fixed_global_direction_possible":
            bool(informative.any() and
                 (np.all(forward[informative] <= TOL) or
                  np.all(backward[informative] <= TOL))),
        "draw_order_is_original_artist_label": False,
    }


def diagnose(render_json, rendered_npz, independent_v18f_court,
             v18a_npz, v18a_independent_court, expected_v18f_court_sha):
    require(expected_v18f_court_sha and len(expected_v18f_court_sha) == 64,
            "V18G_EXPLICIT_SEALED_COURT_SHA_REQUIRED")
    require(sha(independent_v18f_court) == expected_v18f_court_sha,
            "V18G_SEALED_V18F_INDEPENDENT_COURT_SHA_DRIFT")
    manifest = load(render_json)
    court = load(independent_v18f_court)
    stain = load(v18a_independent_court)
    require(manifest.get("schema") == PARENT_RENDER
            and manifest.get("status") ==
            "V18F_REAL_ORIGINAL_FULL_PHYSICAL_FACE_EDITS_RENDERED__INDEPENDENT_COURT_REQUIRED"
            and court.get("schema") == PARENT_COURT
            and court.get("status") == PARENT_PASS
            and court.get("source_sha256") == manifest.get("original_source_sha256")
            and court.get("real_v18b_source_editor_manifest_sha256") == sha(render_json)
            and court.get("real_v18b_source_editor_rgba_npz_sha256") == sha(rendered_npz)
            and court.get("parent_v18a_independent_stain_court_sha256") ==
                sha(v18a_independent_court)
            and manifest.get("source_reactive_stain_npz_sha256") == sha(v18a_npz)
            and court.get("physical_render_face_counts_courted") is True
            and court.get("source_frozen_faces", 0) > 1
            and court.get("original_camera_views") == list(VIEWS)
            and court.get("heldout_original_source_authored_pose_ordinal") == MEASURED_POSE
            and manifest.get("source_edit_arms") == expected_arms()
            and court.get("actual_real_blender_source_render_arms_per_camera") == 52
            and court.get("actual_heldout_source_edit_cases") == 72
            and manifest.get("physical_face_visibility_operator") ==
               "ACTUAL_TRIANGLE_REMOVAL_AND_OBJECT_RENDER_HIDE"
            and stain.get("status") ==
               "ORIGINAL_NATIVE_SOURCE_STAIN_GRAPH_REPRODUCED__ARTIST_EDIT_TRUTH_UNKNOWN"
            and stain.get("original_source_sha256") == manifest.get("original_source_sha256")
            and all(court.get(key) is False for key in (
                "graphics_owner_authorized", "iris_training_authorized",
                "tessa_training_authorized", "materialization_authorized",
                "training_authorized"))
            and all(manifest.get(key) is False for key in (
                "graphics_owner_authorized", "iris_training_authorized",
                "tessa_training_authorized", "materialization_authorized",
                "training_authorized")),
            "V18G_PARENT_RENDER_AND_INDEPENDENT_SOURCE_COURT_INVALID")
    with np.load(rendered_npz, allow_pickle=False) as x:
        require(x.files == [RGBA_MEMBER], "V18G_PRIVATE_RENDER_NPZ_MEMBERS")
        rgba = np.asarray(x[RGBA_MEMBER], np.float32)
    with np.load(v18a_npz, allow_pickle=False) as x:
        require(set(x.files) == {"faces", "stain_channels", "graph_edges",
                "selected_source_control_axes_teacher_only"} |
                {"labels_" + str(t).replace(".", "p") for t in THRESHOLDS},
                "V18G_V18A_SOURCE_FACE_ARRAY_MEMBERS")
        labels = {t: np.asarray(x["labels_" + str(t).replace(".", "p")],
                                np.int32) for t in THRESHOLDS}
    require(rgba.shape == (4, 52, 192, 192, 4)
            and np.isfinite(rgba).all()
            and np.min(rgba[..., 3]) >= -1e-5
            and np.max(rgba[..., 3]) <= 1.00001,
            "V18G_RENDER_DATA_SHAPE_OR_ALPHA_INVALID")
    from tools.corpus.court_corpus370_v18f_full_physical_edit_benchmark import (
        load as load_json
    )
    # Get parent target face IDs from already independently courted,
    # immutable V18f rows, rather than inferring them from edited pixels.
    parent_rows = court["source_edit_consequence_rows_teacher_only"]
    indexed = {
        (x["source_camera_view"], x["precommitted_v11_heldout_target_ordinal"],
         float(x["candidate_threshold"])): x
        for x in parent_rows
    }
    require(len(indexed) == 72 and all(len(x) == court["source_frozen_faces"]
            for x in labels.values()), "V18G_PRIVATE_SOURCE_FACE_BINDING")
    arms = {v: k for k, v in enumerate(expected_arms())}
    rows = []
    unique = {}
    for vi, view in enumerate(VIEWS):
        original = rgba[vi, arms["SOUP_FULL"]]
        empty = rgba[vi, arms["SOUP_EMPTY"]]
        require(not np.any(empty[..., 3] > TOL),
                "V18G_ORIGINAL_PHYSICAL_EMPTY_NOT_EMPTY")
        for ti, ordinal in enumerate(TARGET_INDICES):
            for t in THRESHOLDS:
                key = str(t).replace(".", "p")
                parent = indexed[(view, ordinal, t)]
                face = int(parent["source_target_face_teacher_only"])
                group = int(labels[t][face])
                size = int(np.count_nonzero(labels[t] == group))
                require(size == parent["group_candidate_face_count"],
                        "V18G_V18A_SOURCE_GROUP_CHANGED")
                iso = rgba[vi, arms[f"STAIN_ISOLATE_{key}_T{ti:02d}"]]
                rem = rgba[vi, arms[f"STAIN_REMOVE_{key}_T{ti:02d}"]]
                tested = compositor_probe(original, iso, rem)
                entry = {
                    "view": int(view),
                    "v11_target_ordinal": int(ordinal),
                    "frozen_threshold": float(t),
                    "anonymous_source_group_number_private": group,
                    "group_source_face_count": size,
                    "source_group_count": int(np.unique(labels[t]).size),
                    "compositor": tested,
                    "not_authorized_drawable": True,
                }
                rows.append(entry)
                ukey = (float(t), int(view), group)
                if ukey not in unique:
                    unique[ukey] = entry
                else:
                    require(unique[ukey]["compositor"] == tested
                            and unique[ukey]["group_source_face_count"] == size,
                            "V18G_NONDETERMINISTIC_REPEATED_GROUP_RENDER")
    result_summary = {}
    for t in THRESHOLDS:
        entries = [v for k, v in unique.items() if k[0] == float(t)]
        informative = [v for v in entries if v["compositor"]
                       ["order_discriminating_overlap_pixels"] > 0]
        nonvacuous = [v for v in entries if
                      v["compositor"]["group_alone_alpha_visible_pixels"] > 0
                      and v["compositor"]["remainder_alone_alpha_visible_pixels"] > 0]
        result_summary[str(t)] = {
            "original_source_group_count": int(np.unique(labels[t]).size),
            "unique_groups_examined": len({k[2] for k in unique if k[0] == float(t)}),
            "unique_group_view_cases": len(entries),
            "both_complements_visible_group_view_cases": len(nonvacuous),
            "order_discriminating_group_view_cases": len(informative),
            "order_discriminating_overlap_pixel_instances": int(sum(
                v["compositor"]["order_discriminating_overlap_pixels"]
                for v in entries)),
            "per_pixel_order_oracle_failed_pixel_instances": int(sum(
                v["compositor"]["per_pixel_order_oracle"]
                  ["pixel_count_above_original_tolerance"]
                for v in entries)),
            "group_over_remainder_failed_pixel_instances": int(sum(
                v["compositor"]["group_over_remainder"]
                  ["pixel_count_above_original_tolerance"]
                for v in entries)),
            "remainder_over_group_failed_pixel_instances": int(sum(
                v["compositor"]["remainder_over_group"]
                  ["pixel_count_above_original_tolerance"]
                for v in entries)),
            "neither_order_reconstructs_count_of_informative_pixels": int(sum(
                v["compositor"]["informative_pixels_both_orders_fail"]
                for v in entries)),
            "evidence_promotion": "NONE__COMPOSITING_CONSEQUENCE_DIAGNOSTIC_ONLY",
        }
    return {
        "schema": SCHEMA, "status": STATUS,
        "source_subject_private": manifest["subject_id"],
        "original_source_sha256": manifest["original_source_sha256"],
        "original_v18f_run": REQUIRED_V18F_RUN,
        "exact_v18f_render_manifest_sha256": sha(render_json),
        "exact_v18f_rgba_npz_sha256": sha(rendered_npz),
        "exact_v18f_independent_court_sha256": sha(independent_v18f_court),
        "exact_v18a_independent_source_court_sha256": sha(v18a_independent_court),
        "exact_v18a_frozen_group_npz_sha256": sha(v18a_npz),
        "no_new_blender_renders": True,
        "no_new_group_optimizer_fitted": True,
        "premultiplication_convention": "RGBA_RGB_ASSUMED_STRAIGHT_LINEAR__MULTIPLIED_BY_ALPHA_FOR_EVALUATION",
        "per_original_source_target_view_threshold": rows,
        "per_threshold_unique_group_view_summary": result_summary,
        "artist_drawable_identity": "UNKNOWN",
        "operational_drawable_identity": "UNKNOWN",
        "invisible_complement_means_success": False,
        "unobserved_draw_order_is_negative_evidence": False,
        "graphics_owner_authorized": False,
        "iris_training_authorized": False,
        "tessa_training_authorized": False,
        "materialization_authorized": False,
        "training_authorized": False,
    }


def main():
    p = argparse.ArgumentParser()
    for name in ("render_manifest", "render_npz", "parent_court",
                 "v18a_npz", "v18a_court", "expected_parent_court_sha256",
                 "output"):
        p.add_argument("--" + name.replace("_", "-"), required=True,
                       type=str if name == "expected_parent_court_sha256"
                       else Path)
    a = p.parse_args()
    d = diagnose(a.render_manifest, a.render_npz, a.parent_court,
                 a.v18a_npz, a.v18a_court, a.expected_parent_court_sha256)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(d, sort_keys=True, indent=2,
                                   allow_nan=False) + "\n")
    print(json.dumps({
        "status": d["status"],
        "original_run": d["original_v18f_run"],
        "per_threshold": d["per_threshold_unique_group_view_summary"],
        "training_authorized": False,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
