#!/usr/bin/env python3
"""Read-only, independent V18b archival render-parity and observability diagnosis.

Never rerenders, never trains, never creates artist DRAWABLE labels. Reopens the
exact sealed V18b editor NPZ + V14 original baseline and disaggregates the
original FAIL-closed combined parity gate, without relaxing its tolerance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.corpus.court_corpus370_v18b_reactive_stain_editor_edits import (
    RENDER, TOL, VIEWS, TARGET_INDICES, expected_arms, rgba_effect
)

SCHEMA = "RealSaS.Corpus370V18bArchivedRenderParityDiagnosis.v1"
STATUS = "ARCHIVED_RENDER_PARITY_AND_PROBE_VISIBILITY_DIAGNOSED__NO_AUTHORITY"
IMAGE_MEMBER = "original_stain_editor_rgba"
V14_MEMBER = "original_native_actuator_rgba"


def require(ok, code):
    if not ok:
        raise ValueError(code)


def sha(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def pair_metrics(left, right):
    """Independently expose raw, premultiplied RGB and alpha mismatches."""
    a = np.asarray(left, dtype=np.float32)
    b = np.asarray(right, dtype=np.float32)
    require(a.shape == b.shape and a.ndim == 3 and a.shape[-1] == 4
            and a.shape[:2] == (192, 192) and np.isfinite(a).all()
            and np.isfinite(b).all(), "V18B_DIAGNOSTIC_INVALID_RGBA_PAIR")
    raw = np.max(np.abs(a - b), axis=-1)
    alpha = np.abs(a[..., 3] - b[..., 3])
    premul = np.max(np.abs(a[..., :3] * a[..., 3:]
                                - b[..., :3] * b[..., 3:]), axis=-1)
    raw_p99 = float(np.percentile(raw, 99))
    visible = (a[..., 3] > TOL) | (b[..., 3] > TOL)
    return {
        "raw_max_channel_p99": raw_p99,
        "raw_max_channel_max": float(raw.max()),
        "raw_max_channel_over_tolerance_pixel_count": int((raw > TOL).sum()),
        "alpha_abs_p99": float(np.percentile(alpha, 99)),
        "alpha_abs_over_tolerance_pixel_count": int((alpha > TOL).sum()),
        "premultiplied_rgb_p99": float(np.percentile(premul, 99)),
        "premultiplied_rgb_over_tolerance_pixel_count": int((premul > TOL).sum()),
        "raw_difference_over_tolerance_but_no_visible_premul_or_alpha_pixel_count": int(
            ((raw > TOL) & (premul <= TOL) & (alpha <= TOL)).sum()
        ),
        "visible_support_pixel_count": int(visible.sum()),
        "pixel_difference_at_original_threshold_pass": raw_p99 <= TOL,
        "effect_mask_empty": not bool(rgba_effect(a, b).any()),
    }


def diagnose(manifest_path, edited_path, v14_path, v14_job_path):
    manifest = json.loads(Path(manifest_path).read_text())
    parent_job = json.loads(Path(v14_job_path).read_text())
    require(parent_job.get("schema") ==
            "RealSaS.Corpus370SourceNativeMechanicalRasterJointJob.v14"
            and parent_job.get("subject_id") == manifest.get("subject_id")
            and parent_job.get("source", {}).get("sha256") ==
                manifest.get("original_source_sha256")
            and parent_job.get("training_authorized") is False
            and parent_job.get("graphics_owner_authorized") is False
            and manifest.get("source_original_v14_job_sha256") == sha(v14_job_path),
            "V18B_DIAGNOSTIC_V14_JOB_PARENT_INVALID")
    require(manifest.get("schema") == RENDER
            and manifest.get("status") ==
            "V18B_REAL_ORIGINAL_SOURCE_STAIN_GROUP_EDITS_RENDERED__INDEPENDENT_COURT_REQUIRED"
            and manifest.get("source_edit_arms") == expected_arms()
            and manifest.get("heldout_source_camera_views") == list(VIEWS)
            and manifest.get("heldout_original_authored_pose_ordinal") == 2
            and manifest.get("new_source_actual_edited_rgba_npz_sha256") == sha(edited_path)
            and manifest.get("original_v14_RGBA_sha256") == sha(v14_path)
            and manifest.get("source_vs_original_v14_RGBA_reference_source_parity_verified") is True
            and manifest.get("artist_original_source_part_names_as_labels") is False
            and manifest.get("source_editor_edit_independence_proven") is False
            and manifest.get("graphics_owner_authorized") is False
            and manifest.get("training_authorized") is False,
            "V18B_DIAGNOSTIC_PARENT_SHA_OR_ARTIST_AUTHORITY")
    with np.load(edited_path, allow_pickle=False) as data:
        require(data.files == [IMAGE_MEMBER], "V18B_DIAGNOSTIC_EDITOR_MEMBERS")
        images = np.asarray(data[IMAGE_MEMBER], np.float32)
    with np.load(v14_path, allow_pickle=False) as data:
        require(data.files == [V14_MEMBER], "V18B_DIAGNOSTIC_V14_MEMBERS")
        original = np.asarray(data[V14_MEMBER], np.float32)
    # V14 arm 0 is statically ORIGINAL_BASE in arms_for_plan. The validated
    # source court references the same original parent for pose ordinal 2.
    require(images.shape == (4, 52, 192, 192, 4)
            and original.ndim == 6
            and original.shape[:2] == (3, 4)
            and original.shape[2] >= 5
            and original.shape[3:] == (192, 192, 4)
            and np.isfinite(images).all() and np.isfinite(original).all()
            and np.all((images[..., 3] >= -1e-5) & (images[..., 3] <= 1.00001)),
            "V18B_DIAGNOSTIC_ARCHIVE_SHAPE_OR_NUMERIC")
    arms = {name: i for i, name in enumerate(expected_arms())}
    per_view = []
    summary = {key: 0 for key in (
        "original_v14_parity_pass_views",
        "source_soup_parity_pass_views",
        "soup_noop_parity_pass_views",
        "empty_soup_alpha_pass_views",
        "observable_single_face_proxy_cases",
        "unobservable_single_face_proxy_cases",
        "observable_single_face_probe_count",
    )}
    seen_targets = set()
    for vi, view in enumerate(VIEWS):
        base = images[vi, arms["SOURCE_ORIGINAL"]]
        soup = images[vi, arms["SOUP_FULL"]]
        noop = images[vi, arms["SOUP_NOOP"]]
        empty = images[vi, arms["SOUP_EMPTY"]]
        cases = {
            "source_vs_frozen_v14_original": pair_metrics(base, original[2, vi, 0]),
            "original_source_vs_full_triangle_soup": pair_metrics(base, soup),
            "full_triangle_soup_vs_noop": pair_metrics(soup, noop),
            "empty_soup_alpha_above_tolerance_pixel_count": int((empty[..., 3] > TOL).sum()),
            "empty_soup_alpha_max": float(empty[..., 3].max()),
        }
        verdicts = {
            "original_v14_parity_pass":
                cases["source_vs_frozen_v14_original"]["pixel_difference_at_original_threshold_pass"],
            "source_soup_parity_pass":
                cases["original_source_vs_full_triangle_soup"]["pixel_difference_at_original_threshold_pass"],
            "soup_noop_parity_pass":
                cases["full_triangle_soup_vs_noop"]["effect_mask_empty"],
            "empty_soup_alpha_pass":
                cases["empty_soup_alpha_above_tolerance_pixel_count"] == 0,
        }
        for label,passed in verdicts.items():
            summary[{
                "original_v14_parity_pass": "original_v14_parity_pass_views",
                "source_soup_parity_pass": "source_soup_parity_pass_views",
                "soup_noop_parity_pass": "soup_noop_parity_pass_views",
                "empty_soup_alpha_pass": "empty_soup_alpha_pass_views",
            }[label]] += int(passed)
        references = []
        for ti in range(len(TARGET_INDICES)):
            removed = images[vi, arms[f"SINGLE_FACE_REMOVE_T{ti:02d}"]]
            isolated = images[vi, arms[f"SINGLE_FACE_ISOLATE_T{ti:02d}"]]
            remove_count = int(rgba_effect(soup, removed).sum())
            isolate_count = int(rgba_effect(empty, isolated).sum())
            observable = bool(remove_count or isolate_count)
            references.append({
                "precommitted_v11_target_ordinal": TARGET_INDICES[ti],
                "remove_effect_pixel_count": remove_count,
                "isolate_effect_pixel_count": isolate_count,
                "reference_observable": observable,
                "artist_drawable_identity": "UNKNOWN",
            })
            summary["observable_single_face_proxy_cases"] += int(observable)
            summary["unobservable_single_face_proxy_cases"] += int(not observable)
            if observable:
                seen_targets.add(ti)
        per_view.append({
            "original_source_camera_view": int(view),
            "parity_metrics": cases,
            "original_court_verdicts": verdicts,
            "precommitted_single_source_face_proxy_observability": references,
        })
    summary["observable_single_face_probe_count"] = len(seen_targets)
    summary["per_view_count"] = len(per_view)
    summary["source_soup_parity_fail_views"] = 4 - summary["source_soup_parity_pass_views"]
    summary["all_four_original_parity_gates_pass"] = all(
        summary[x] == 4 for x in (
            "original_v14_parity_pass_views",
            "source_soup_parity_pass_views",
            "soup_noop_parity_pass_views",
            "empty_soup_alpha_pass_views",
        )
    )
    summary["has_observable_original_precommitted_target"] = (
        summary["observable_single_face_proxy_cases"] > 0
    )
    return {
        "schema": SCHEMA, "status": STATUS,
        "source_subject_id_private": manifest["subject_id"],
        "original_source_sha256": manifest["original_source_sha256"],
        "source_manifest_sha256": sha(manifest_path),
        "original_editor_rgba_npz_sha256": sha(edited_path),
        "frozen_v14_original_rgba_npz_sha256": sha(v14_path),
        "original_v18b_workflow_run": "42100552844-1",
        "original_heldout_pose_ordinal": 2,
        "frozen_original_court_tolerance": TOL,
        "original_source_camera_views": list(VIEWS),
        "per_view": per_view,
        "aggregate": summary,
        "new_renders_performed": 0,
        "source_pixels_exposed_in_report": False,
        "original_artist_drawable_same_different_labels": "UNKNOWN",
        "source_editability_qualified": False,
        "graphics_owner_authorized": False,
        "iris_training_authorized": False,
        "tessa_training_authorized": False,
        "training_authorized": False,
        "materialization_authorized": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--editor-npz", type=Path, required=True)
    parser.add_argument("--v14-npz", type=Path, required=True)
    parser.add_argument("--v14-job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(args.manifest, args.editor_npz, args.v14_npz, args.v14_job)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True,
                                       allow_nan=False) + "\n")
    print(json.dumps({
        "status": result["status"],
        "aggregate": result["aggregate"],
        "view_breakdown": [
            {
                "view": row["original_source_camera_view"],
                "source_vs_v14_p99": row["parity_metrics"]
                    ["source_vs_frozen_v14_original"]["raw_max_channel_p99"],
                "source_vs_soup_p99": row["parity_metrics"]
                    ["original_source_vs_full_triangle_soup"]["raw_max_channel_p99"],
                "source_vs_soup_alpha_difference_pixels": row["parity_metrics"]
                    ["original_source_vs_full_triangle_soup"]
                    ["alpha_abs_over_tolerance_pixel_count"],
                "source_vs_soup_premul_rgb_difference_pixels": row["parity_metrics"]
                    ["original_source_vs_full_triangle_soup"]
                    ["premultiplied_rgb_over_tolerance_pixel_count"],
                "soup_noop_effect_empty": row["parity_metrics"]
                    ["full_triangle_soup_vs_noop"]["effect_mask_empty"],
            } for row in result["per_view"]
        ],
        "training_authorized": False
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
