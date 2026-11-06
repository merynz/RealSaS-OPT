#!/usr/bin/env python3
"""Independent Cat original V3 Cycles vs compiled CAA reference puppet court.

The puppet compiler never imports this module or reads source target renders.
Court handles full-screen linear premultiplied RGBA and explicit UNKNOWN support.
No style/face/owner truth inferred. No training/materialization authority minted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

SOURCE_SAMPLES = "b2bb4fa9fc0b777bb8418aebc729123616b774b55eeb215ba17651e7f3448e0d"
SOURCE_MANIFEST = "89dcb4d516e5579e766b19bd2d25eb31cbfaaa64f27ce34632405d99ce6874bc"
INDEPENDENT_SOURCE_COURT = "0171e9ec836d7b3c44ec64abcc59f712ebd33a87c2ad4edf6188976dc08e272c"
SOURCE = "1e73aaf7d497dd7d2946badf92d189d2333f163e66dced91feb446bc9bdee850"
TEACHER = "c2792c8efb36c4ccdf9178c6b933351b50364bbd28707e9365a3c115db00d603"
POSES = (0, 14)
VIEWS = (0, 2, 4, 6)
OFFSETS = (0, 16)
ARMS = ("BASE", "SINGLE_0", "SINGLE_1", "SINGLE_2")
TOL = 0.025
EXPECTED_SOURCE_SHAPE = (2, 4, 2, 4, 192, 192, 4)
EXPECTED_PREDICTION_SHAPE = (2, 4, 4, 192, 192, 4)
NO_AUTHORITY = {
    "original_artist_drawable_owner_authorized": False,
    "graphics_training_authorized": False,
    "iris_training_authorized": False,
    "tessa_training_authorized": False,
    "materialization_authorized": False,
}


def require(ok, reason):
    if not ok:
        raise ValueError("INDEPENDENT_CAT_REFERENCE_PUPPET_" + reason)


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def compare_original_v3_source_against_puppet(predicted, source, diagnostics,
                                              tolerance=TOL):
    """Pure independently computed physical comparison, never a compile input."""
    p = np.asarray(predicted, dtype=np.float64)
    s = np.asarray(source, dtype=np.float64)
    require(p.ndim == 6 and p.shape[0:3] == (2, 4, 4)
            and p.shape[-1] == 4, "PREDICTION_ARRAY_INVALID")
    require(s.shape == (2, 4, 2, 4, *p.shape[-3:-1], 4),
            "SOURCE_ARRAY_INVALID")
    require(len(diagnostics) == 32
            and np.isfinite(p).all() and np.isfinite(s).all()
            and np.min(s[..., 3]) >= -1e-5
            and np.max(s[..., 3]) <= 1.00001,
            "FINITE_SOURCE_OR_DIAGNOSTICS_INVALID")
    require(tolerance == TOL, "PRECOMMITTED_TOLERANCE_DRIFT")
    # Original physical Cycles EXRs were recorded as straight linear RGBA;
    # exactly the V3 independent court's conversion, no curve adjustment.
    physical_pm = np.concatenate(
        [s[..., :3] * s[..., 3:4], s[..., 3:4]], axis=-1)
    pm_pred = p[:, :, None, :, :, :, :]
    error = np.max(np.abs(pm_pred - physical_pm), axis=-1)
    bad = error > tolerance
    rows = []
    for pi, pose in enumerate(POSES):
        for vi, view in enumerate(VIEWS):
            for oi, offset in enumerate(OFFSETS):
                for ai, arm in enumerate(ARMS):
                    e = error[pi, vi, oi, ai]
                    fails = bad[pi, vi, oi, ai]
                    first = np.argwhere(fails)
                    witness = None
                    if len(first):
                        y, x = [int(z) for z in first[0]]
                        witness = {
                            "source_pose_ordinal": pose, "view_index": view,
                            "source_cycles_sample_offset": offset,
                            "source_uv_arm": arm, "pixel_yx": [y, x],
                            "reference_premultiplied_rgba":
                                p[pi, vi, ai, y, x].tolist(),
                            "original_source_premultiplied_rgba":
                                physical_pm[pi, vi, oi, ai, y, x].tolist(),
                            "max_channel_error": float(e[y, x]),
                        }
                    rows.append({
                        "pose_ordinal": pose, "view_index": view,
                        "cycles_sample_offset": offset, "source_arm": arm,
                        "mismatched_sample_pixels": int(np.count_nonzero(fails)),
                        "max_channel_error": float(np.max(e)),
                        "first_private_source_counterexample": witness,
                    })
    overflow = sum(int(d["layer_overflow_pixels"]) for d in diagnostics)
    ties = sum(int(d["exact_depth_ambiguity_pixels"]) for d in diagnostics)
    for index, d in enumerate(diagnostics):
        pi = index // (len(VIEWS) * len(ARMS))
        vi = (index // len(ARMS)) % len(VIEWS)
        ai = index % len(ARMS)
        require(d["pose_ordinal"] == POSES[pi]
                and d["view_index"] == VIEWS[vi]
                and d["source_arm_index"] == ai, "FRAME_DIAGNOSTICS_ORDER_DRIFT")
    mismatches = int(np.count_nonzero(bad))
    status = (
        "VIOLATED_WITH_UNQUALIFIED_VISIBILITY_SUPPORT"
        if mismatches and (overflow or ties) else
        "VIOLATED_PHYSICAL_SOURCE_PARITY" if mismatches else
        "UNKNOWN_UNQUALIFIED_VISIBILITY_SUPPORT" if overflow or ties else
        "SATISFIED_ORIGINAL_CAT_FINITE_REFERENCE_ONLY"
    )
    return {
        "status": status,
        "physical_original_full_source_render_comparisons": 64,
        "predicted_reference_frames": 32,
        "maximum_premultiplied_rgba_error": float(np.max(error)),
        "total_source_sample_operation_pixel_mismatches": mismatches,
        "source_sample_operation_pixels_compared": int(bad.size),
        "reference_layer_overflow_pixels_across_frames": overflow,
        "reference_exact_depth_ambiguity_pixels_across_frames": ties,
        "operation_breakdown": rows,
        "candidate_compiled_with_original_target_pixels": False,
        "physical_target_rgb_interpretation": "ORIGINAL_EXR_STRAIGHT_LINEAR_TO_PM",
        "premultiplied_rgba_tolerance": TOL,
        "artist_drawable_owner": "UNKNOWN",
        "full_puppet_and_unseen_training_qualification": "UNKNOWN",
        **NO_AUTHORITY,
    }


def independently_court(root, out):
    root = Path(root)
    out = Path(out)
    source_file = root / "PRIVATE_CAT_NATIVE_POSE_UV_EDIT_CYCLES_SAMPLES_V3.npz"
    source_manifest_file = root / "PRIVATE_CAT_NATIVE_POSE_UV_EDIT_CYCLES_V3.json"
    source_court_file = root / "PRIVATE_INDEPENDENT_CAT_POSE_UV_TRANSPORT_V3.json"
    prediction_file = root / "PRIVATE_CAT_V3_PUPPET_REFERENCE_PREDICTIONS_V0.npz"
    prediction_manifest_file = root / "PRIVATE_CAT_V3_PUPPET_REFERENCE_PREDICTIONS_V0.json"
    require(digest(source_file) == SOURCE_SAMPLES
            and digest(source_manifest_file) == SOURCE_MANIFEST
            and digest(source_court_file) == INDEPENDENT_SOURCE_COURT,
            "ORIGINAL_PHYSICAL_SOURCE_LINEAGE_DRIFT")
    source_court = json.loads(source_court_file.read_text())
    require(source_court.get("status") ==
            "PASS_ORIGINAL_CAT_POSE_UV_EDIT_PHYSICAL_SOURCE__FIXED_SCREEN_CLASS_VIOLATED"
            and source_court.get("source_original_blender_physical_sample_sha256")
            == SOURCE_SAMPLES and
            source_court.get("actual_source_original_full_physical_renders") == 64
            and source_court.get("source_missing_evidence_obligations") == [],
            "ORIGINAL_INDEPENDENT_SOURCE_COURT_NOT_QUALIFIED")
    manifest = json.loads(prediction_manifest_file.read_text())
    require(manifest.get("source_original_glb_sha256") == SOURCE
            and manifest.get("source_original_native_teacher_sha256") == TEACHER
            and manifest.get("source_target_pixels_loaded_in_prediction") is False
            and manifest.get("source_original_artist_ownership") == "UNKNOWN"
            and manifest.get("candidate_id") == "addressed_aggregate"
            and manifest.get("source_physical_parity") == "UNKNOWN"
            and manifest.get("reference_renderer") ==
               "REAL_SAS_CAA_V2_DEPTH_K4__2X2_COVERAGE"
            and all(manifest.get(f) is False for f in (
                "training_authorized", "materialization_authorized",
                "iris_training_authorized", "tessa_training_authorized",
                "graphics_owner_authorized"))
            and manifest.get("reference_prediction_sha256") == digest(prediction_file),
            "REFERENCE_PREDICTION_MANIFEST_DRIFT")
    require(manifest.get("reference_prediction_shape") ==
            list(EXPECTED_PREDICTION_SHAPE), "REFERENCE_RENDER_SCOPE_DRIFT")
    with np.load(source_file, allow_pickle=False) as f:
        source = np.asarray(f["source_native_pose_uv_rgba"], dtype=np.float32)
    with np.load(prediction_file, allow_pickle=False) as f:
        predictions = np.asarray(
            f["puppet_reference_premultiplied_linear_rgba"], dtype=np.float32)
    require(source.shape == EXPECTED_SOURCE_SHAPE
            and predictions.shape == EXPECTED_PREDICTION_SHAPE,
            "ORIGINAL_PHYSICAL_COMPARISON_SHAPE_DRIFT")
    measurement = compare_original_v3_source_against_puppet(
        predictions, source, manifest["frame_diagnostics"])
    result = {
        "schema": "RealSaS.IndependentCatV3ReferencePuppetPhysicalCourt.v0",
        "source_original_render_samples_sha256": SOURCE_SAMPLES,
        "source_original_independent_court_sha256": INDEPENDENT_SOURCE_COURT,
        "reference_prediction_artifact_sha256": digest(prediction_file),
        "source_original_cat_native_pose_scope": list(POSES),
        "source_original_cat_camera_scope": list(VIEWS),
        "source_original_cat_sample_offsets": list(OFFSETS),
        "original_source_uv_arm_names": list(ARMS),
        "original_artist_owner": "UNKNOWN",
        "source_only_teacher_mode": True,
        "student_one_image_inference_qualified": False,
        "runtime_gpu_benchmark": "UNMEASURED_COST",
        "physical_comparison": measurement,
        **NO_AUTHORITY,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True,
                              allow_nan=False) + "\n")
    print(json.dumps({
        "status": measurement["status"],
        "physical_source_comparisons": measurement[
            "physical_original_full_source_render_comparisons"],
        "mismatched_sample_operation_pixels": measurement[
            "total_source_sample_operation_pixel_mismatches"],
        "max_channel_error": measurement["maximum_premultiplied_rgba_error"],
        "unknown_layer_overflow_pixels": measurement[
            "reference_layer_overflow_pixels_across_frames"],
        "unknown_exact_depth_tie_pixels": measurement[
            "reference_exact_depth_ambiguity_pixels_across_frames"],
        "training_authorized": False,
    }, sort_keys=True), flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    independently_court(args.root, args.output)


if __name__ == "__main__":
    main()
