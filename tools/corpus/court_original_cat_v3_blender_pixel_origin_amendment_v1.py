#!/usr/bin/env python3
"""A1 independent Cat V3 original EXR pixel-buffer origin amendment court.

Runs ORIGINAL source comparison first and proves exact archived V0 failure,
then changes only source original scanline indexing from Blender bottom-first
Image.pixels buffer to CAA top-first image-array convention, for ALL poses,
views, samples and edit arms. No renderer changes, no target registration
adjustment, no alpha/UV/material/tolerance changes. Preserves UNKNOWN K4.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.corpus.court_original_cat_v3_puppet_reference_parity_v0 import (
    compare_original_v3_source_against_puppet,
    SOURCE_SAMPLES, SOURCE_MANIFEST, INDEPENDENT_SOURCE_COURT,
    EXPECTED_SOURCE_SHAPE, EXPECTED_PREDICTION_SHAPE,
    SOURCE, TEACHER, TOL, NO_AUTHORITY,
)

ORIGINAL_REFERENCE_PREDICTIONS = (
    "55675886ae92c3b122647b1749290675dc8fbc13d1c617c9340ec42fe3e702d8"
)
ORIGINAL_PHYSICAL_COURT_V0 = (
    "90759dd7f431b89ccc697824a9f9485fa2cdef15360a89746d0978dbd1ae50b4"
)
SCHEMA = "RealSaS.IndependentCatV3PhysicalOriginAmendmentCourt.v1"
ORIGIN_CONTRACT = (
    "BLENDER_IMAGE_PIXELS_BOTTOM_LEFT_FLAT_BUFFER__"
    "REVERSE_SOURCE_HEIGHT_AXIS_TO_CAA_TOP_LEFT"
)


def require(ok, name):
    if not ok:
        raise ValueError("CAT_V3_SOURCE_ORIGIN_AMENDMENT_" + name)


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def source_bottom_first_to_top_first(source):
    value = np.asarray(source)
    require(value.ndim == 7 and value.shape[-1] == 4,
            "ORIGINAL_SOURCE_ARRAY_SHAPE_INVALID")
    # Source array axes: pose, camera, Cycles sample, arm, y, x, RGBA.
    # Image.pixels foreach_get is bottom-first; CAA y=0 is TOP row.
    # The algorithm cannot inspect target pixels, choose per-pose alignment,
    # warp image coordinates, or retune any geometry/appearance parameters.
    return np.ascontiguousarray(value[..., ::-1, :, :])


def corrected_origin_recount(*, predictions, physical_source, diagnostics,
                             archived_v0):
    historical = compare_original_v3_source_against_puppet(
        predictions, physical_source, diagnostics, tolerance=TOL)
    old = archived_v0["physical_comparison"]
    require(
        historical["total_source_sample_operation_pixel_mismatches"] ==
        old["total_source_sample_operation_pixel_mismatches"] == 1004587 and
        historical["maximum_premultiplied_rgba_error"] ==
        old["maximum_premultiplied_rgba_error"] == 1.0 and
        historical["reference_layer_overflow_pixels_across_frames"] ==
        old["reference_layer_overflow_pixels_across_frames"] == 38928 and
        historical["status"] == old["status"] ==
        "VIOLATED_WITH_UNQUALIFIED_VISIBILITY_SUPPORT",
        "ARCHIVED_UNCORRECTED_V0_NOT_REPRODUCED"
    )
    normalized = source_bottom_first_to_top_first(physical_source)
    corrected = compare_original_v3_source_against_puppet(
        predictions, normalized, diagnostics, tolerance=TOL)
    require(corrected["reference_layer_overflow_pixels_across_frames"] ==
            historical["reference_layer_overflow_pixels_across_frames"] and
            corrected["reference_exact_depth_ambiguity_pixels_across_frames"] ==
            historical["reference_exact_depth_ambiguity_pixels_across_frames"],
            "PIXEL_ORIGIN_CORRECTION_CHANGED_GEOMETRY_GATES")
    return historical, corrected


def independently_court(root, output):
    root, output = Path(root), Path(output)
    original_samples = root / "PRIVATE_CAT_NATIVE_POSE_UV_EDIT_CYCLES_SAMPLES_V3.npz"
    original_manifest = root / "PRIVATE_CAT_NATIVE_POSE_UV_EDIT_CYCLES_V3.json"
    original_court = root / "PRIVATE_INDEPENDENT_CAT_POSE_UV_TRANSPORT_V3.json"
    predicted = root / "PRIVATE_CAT_V3_PUPPET_REFERENCE_PREDICTIONS_V0.npz"
    prediction_manifest = root / "PRIVATE_CAT_V3_PUPPET_REFERENCE_PREDICTIONS_V0.json"
    archived_report = root / "PRIVATE_INDEPENDENT_CAT_V3_PUPPET_REFERENCE_PHYSICAL_PARITY_V0.json"
    for p, expected in (
        (original_samples, SOURCE_SAMPLES),
        (original_manifest, SOURCE_MANIFEST),
        (original_court, INDEPENDENT_SOURCE_COURT),
        (predicted, ORIGINAL_REFERENCE_PREDICTIONS),
        (archived_report, ORIGINAL_PHYSICAL_COURT_V0),
    ):
        require(digest(p) == expected, "SOURCE_PARENT_SHA:" + p.name)
    original_court_doc = json.loads(original_court.read_text())
    require(original_court_doc.get("status") ==
            "PASS_ORIGINAL_CAT_POSE_UV_EDIT_PHYSICAL_SOURCE__FIXED_SCREEN_CLASS_VIOLATED",
            "SOURCE_ORIGINAL_INDEPENDENT_COURT_UNQUALIFIED")
    prediction_doc = json.loads(prediction_manifest.read_text())
    require(
        prediction_doc.get("source_original_glb_sha256") == SOURCE and
        prediction_doc.get("source_original_native_teacher_sha256") == TEACHER and
        prediction_doc.get("reference_prediction_sha256") ==
        ORIGINAL_REFERENCE_PREDICTIONS and
        prediction_doc.get("source_target_pixels_loaded_in_prediction") is False and
        prediction_doc.get("source_physical_parity") == "UNKNOWN" and
        prediction_doc.get("candidate_id") == "addressed_aggregate" and
        prediction_doc.get("reference_prediction_shape") ==
        list(EXPECTED_PREDICTION_SHAPE),
        "PREDICTION_MANIFEST_UNQUALIFIED"
    )
    archived_doc = json.loads(archived_report.read_text())
    require(archived_doc.get("source_original_render_samples_sha256") ==
            SOURCE_SAMPLES and archived_doc.get("reference_prediction_artifact_sha256") ==
            ORIGINAL_REFERENCE_PREDICTIONS, "ARCHIVED_V0_SOURCE_DRIFT")
    with np.load(original_samples, allow_pickle=False) as f:
        s = np.asarray(f["source_native_pose_uv_rgba"], dtype=np.float32)
    with np.load(predicted, allow_pickle=False) as f:
        p = np.asarray(f["puppet_reference_premultiplied_linear_rgba"], dtype=np.float32)
    require(s.shape == EXPECTED_SOURCE_SHAPE and
            p.shape == EXPECTED_PREDICTION_SHAPE,
            "FROZEN_SOURCE_RENDER_SCOPE_INVALID")
    before, after = corrected_origin_recount(
        predictions=p, physical_source=s,
        diagnostics=prediction_doc["frame_diagnostics"],
        archived_v0=archived_doc,
    )
    report = {
        "schema": SCHEMA,
        "historical_original_v0_court_sha256": ORIGINAL_PHYSICAL_COURT_V0,
        "original_unmodified_source_v3_npz_sha256": SOURCE_SAMPLES,
        "original_unmodified_prediction_npz_sha256": ORIGINAL_REFERENCE_PREDICTIONS,
        "original_native_teacher_sha256": TEACHER,
        "correction_scope": "POST_HOC_SOURCE_IMAGE_BUFFER_ORIGIN_ONLY",
        "correction_source_axis": -3,
        "source_pixel_buffer_contract": ORIGIN_CONTRACT,
        "fixed_original_pm_tolerance": TOL,
        "reference_puppet_rerendered": False,
        "original_source_pixels_modified_on_disk": False,
        "learned_alignment_or_warp": False,
        "original_v0_recomputed": {
            "status": before["status"],
            "total_mismatches": before[
                "total_source_sample_operation_pixel_mismatches"],
            "max_error": before["maximum_premultiplied_rgba_error"],
            "visibility_overflow": before[
                "reference_layer_overflow_pixels_across_frames"],
        },
        "corrected_source_readback_comparison": after,
        "artist_drawable_owner": "UNKNOWN",
        "source_is_teacher_only": True,
        "student_one_image_inference_qualified": False,
        "gpu_runtime_measurement": "UNMEASURED_COST",
        **NO_AUTHORITY,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True,
                                 allow_nan=False) + "\n")
    print(json.dumps({
        "status": after["status"],
        "source_buffer_origin": ORIGIN_CONTRACT,
        "old_mismatched_source_pixel_samples": before[
            "total_source_sample_operation_pixel_mismatches"],
        "corrected_mismatched_source_pixel_samples": after[
            "total_source_sample_operation_pixel_mismatches"],
        "corrected_max_channel_error": after[
            "maximum_premultiplied_rgba_error"],
        "unknown_K4_overflow_pixels_unchanged": after[
            "reference_layer_overflow_pixels_across_frames"],
        "training_authorized": False,
    }, sort_keys=True), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    independently_court(a.root, a.output)


if __name__ == "__main__":
    main()
