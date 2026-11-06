#!/usr/bin/env python3
"""Original-source edit-consequence Möbius truth V1: full Boolean factorial.

Mathematical identity (not a learned segmentation):
  Y(S) = sum_{T subset S} interaction(T)
  interaction(S) = sum_{T subset S} (-1)^(|S|-|T|) Y(T).

All Y(S) MUST be *independently observed* original-source physical renders
with the same source/pose/camera/Cycles sample/pixel/linear-RGBA protocol.
Missing arms = UNKNOWN, never substitute predictions. Effect coefficients
are operation-set causal interactions, not drawable/layer ownership.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = "RealSaS.SourcePhysicalInterventionMobiusTruth.v1"
CAT_SOURCE = "1e73aaf7d497dd7d2946badf92d189d2333f163e66dced91feb446bc9bdee850"
CAT_SOURCE_ARRAY = "d422b7d7d082fba5deac5dd1239b8c6aec078f4d635a7bd5b1f10e514100b114"
CAT_SOURCE_MANIFEST = "8a56e47a09f5c1216de2c97e6f13d5e04385f7fe7d2e580d0085b2ab69dc2da1"
CAT_INDEPENDENT_PHYSICAL_COURT = "90b85b9c1a963b23f7c3037ca443975bf2d5c5d4123568d8aa8e54557e433c08"
CAT_SOURCE_SAMPLE_SHAPE = (4, 8, 8, 192, 192, 4)
ARMS = ("BASE", "SINGLE_0", "SINGLE_1", "SINGLE_2",
        "PAIR_01", "PAIR_02", "PAIR_12", "JOINT_012")
MASK_TO_ORIGINAL_ARM_INDEX = (0, 1, 2, 4, 3, 5, 6, 7)
MEASURED_TOL = 0.025
NO_AUTHORITY = {
    "graphics_owner_authorized": False,
    "artist_drawable_ownership": "UNKNOWN",
    "same_drawable_training_pairs_qualified": 0,
    "different_drawable_training_pairs_qualified": 0,
    "iris_training_authorized": False,
    "tessa_training_authorized": False,
    "training_authorized": False,
    "materialization_authorized": False,
}


def require(ok, code):
    if not ok:
        raise ValueError("ORIGINAL_INTERVENTION_MOBIUS_" + code)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def source_pm(physical_straight_rgba):
    a = np.asarray(physical_straight_rgba, dtype=np.float64)
    require(a.ndim >= 2 and a.shape[-1] == 4 and
            np.isfinite(a).all() and
            np.min(a[..., 3]) >= -1e-6 and np.max(a[..., 3]) <= 1.000001,
            "ORIGINAL_SOURCE_LINEAR_RGBA_INVALID")
    out = a.copy()
    out[..., :3] *= out[..., 3:4]
    return out


def mobius_transform(full_mask_order_values, *, control_count):
    """Unique factorial interaction coefficients via in-place Möbius transform.

    First axis: mask 0,1,...,2^m-1. Every physical operation combination is
    measured. All other axes are arbitrary but must be EXACTLY aligned.
    """
    y = np.asarray(full_mask_order_values, dtype=np.float64)
    require(1 <= control_count <= 8 and y.ndim >= 2 and
            y.shape[0] == 1 << control_count and np.isfinite(y).all(),
            "INCOMPLETE_OR_INVALID_FULL_FACTORIAL")
    z = y.copy()
    for bit in range(control_count):
        for mask in range(1 << control_count):
            if mask & (1 << bit):
                z[mask] -= z[mask ^ (1 << bit)]
    return z


def inverse_mobius(interactions, *, control_count):
    """Exact zeta transform: sums causal coefficients of all subsets."""
    z = np.asarray(interactions, dtype=np.float64)
    require(1 <= control_count <= 8 and z.ndim >= 2 and
            z.shape[0] == 1 << control_count and np.isfinite(z).all(),
            "INVALID_INTERVENTION_COEFFICIENTS")
    y = z.copy()
    for bit in range(control_count):
        for mask in range(1 << control_count):
            if mask & (1 << bit):
                y[mask] += y[mask ^ (1 << bit)]
    return y


def measure_original_source_mobius(physical_source_rgba):
    """Full original Cat 3-control × 8-arm scope, no source packet labels."""
    raw = np.asarray(physical_source_rgba, dtype=np.float32)
    require(raw.shape == CAT_SOURCE_SAMPLE_SHAPE and np.isfinite(raw).all(),
            "SOURCE_PHYSICAL_SAMPLE_SCOPE_DRIFT")
    pm = source_pm(raw)
    # Original physical archive: [VIEW,SAMPLE,ARM,Y,X,RGBA].
    # Boolean mask order differs from ORIGINAL arm order for PAIR_01 and
    # SINGLE_2. This explicit fixed permutation is part of source provenance.
    y = np.take(pm, MASK_TO_ORIGINAL_ARM_INDEX, axis=2)
    y = np.moveaxis(y, 2, 0)  # [MASK, VIEW, SAMPLE, Y, X, RGBA]
    effects = mobius_transform(y, control_count=3)
    rebuilt = inverse_mobius(effects, control_count=3)
    error = float(np.max(np.abs(rebuilt - y)))
    require(error <= 1e-12, "MOBIUS_INVERSION_INTERNAL_ERROR")

    per_mask = []
    for mask in range(8):
        magnitude = np.max(np.abs(effects[mask]), axis=-1)
        per_mask.append({
            "bitmask": mask,
            "source_operation_subset_cardinality": int(mask.bit_count()),
            "source_operation_indices": [i for i in range(3) if mask & (1 << i)],
            "source_sample_operation_pixels_above_original_tolerance":int(
                np.count_nonzero(magnitude > MEASURED_TOL)),
            "source_max_abs_linear_pm_rgba_effect":float(np.max(magnitude)),
            "source_max_abs_alpha_effect":float(np.max(
                np.abs(effects[mask, ..., 3]))) if mask != 0 else None,
            "measured_original_source_not_predicted": True,
            "artist_drawable_owner":"UNKNOWN",
        })
    single_effect_masks = [
        np.max(np.abs(effects[1 << i]), axis=-1) > MEASURED_TOL
        for i in range(3)
    ]
    overlap = {
        f"{i}_{j}":int(np.count_nonzero(single_effect_masks[i] &
                                       single_effect_masks[j]))
        for i,j in ((0,1),(0,2),(1,2))
    }
    return {
        "schema": SCHEMA,
        "status": "ORIGINAL_SOURCE_FINITE_EDIT_INTERACTIONS_IDENTIFIED__ARTIST_OWNER_UNKNOWN",
        "original_source_sha256": CAT_SOURCE,
        "original_physical_sample_array_sha256": CAT_SOURCE_ARRAY,
        "intervention_boolean_control_count": 3,
        "original_source_physical_arm_order": list(ARMS),
        "mask_to_original_arm_index": list(MASK_TO_ORIGINAL_ARM_INDEX),
        "original_source_scope":{
            "poses":"ORIGINAL_BIND_REST_ONLY",
            "views":[1,3,5,7],
            "cycles_sample_offsets":[0,4,8,12,16,20,24,28],
            "resolution":[192,192],
            "linear_color":"ORIGINAL_STRAIGHT_TO_PREMULTIPLIED_LINEAR_RGBA",
            "physical_sample_frames":256,
            "original_source_uv_edits":"THREE_DISJOINT_64x64_RGB_INVERT_ATLAS_TILES",
        },
        "identity":"FINITE_BOOLEAN_MOBIUS_TRANSFORM_AND_EXACT_ZETA_INVERSE",
        "inverse_max_abs_error_float64": error,
        "tolerance_from_original_independent_physical_source_court": MEASURED_TOL,
        "coefficients_by_exact_edit_subset": per_mask,
        "single_edit_image_sample_support_overlap_at_source_tolerance": overlap,
        "triple_interaction_is_original_artist_triple_drawable": False,
        "source_operation_interactions_do_not_imply_drawable_owner":True,
        "original_native_pose14_edit_interactions":"UNKNOWN_NOT_MEASURED_IN_THIS_SOURCE",
        "source_physical_ray_to_original_face_edit_effect":"UNKNOWN_NOT_COURTED",
        **NO_AUTHORITY,
    }


def produce(root: Path, output: Path):
    root = Path(root)
    inputs = (
        ("original-physical-samples.npz", CAT_SOURCE_ARRAY),
        ("original-physical-manifest.json", CAT_SOURCE_MANIFEST),
        ("original-physical-independent-court.json", CAT_INDEPENDENT_PHYSICAL_COURT),
    )
    for name, expected in inputs:
        require(sha(root / name) == expected, "IMMUTABLE_PHYSICAL_PARENT_SHA:" + name)
    manifest = json.loads((root / "original-physical-manifest.json").read_text())
    court = json.loads((root / "original-physical-independent-court.json").read_text())
    require(
        manifest.get("schema") ==
            "RealSaS.Corpus370OriginalCatSourceTextureSameCyclesEditEvidence.v2" and
        court.get("status") ==
            "PASS_ORIGINAL_CAT_LINKED_UV_EDIT_CONSEQUENCE_SUPERPOSITION__OWNER_UNKNOWN" and
        court.get("failed_obligations") == [] and
        manifest.get("source_original_sample_npz_sha256") == CAT_SOURCE_ARRAY and
        court.get("original_source_blender_sample_npz_sha256") == CAT_SOURCE_ARRAY and
        manifest.get("original_source_sha256") == CAT_SOURCE and
        court.get("source_original_glb_sha256") == CAT_SOURCE and
        manifest.get("camera_indices") == [1,3,5,7] and
        manifest.get("cycles_subset_indices") == [0,4,8,12,16,20,24,28] and
        manifest.get("original_source_uv_texture_arms") == list(ARMS) and
        manifest.get("original_source_rest_only") is True and
        manifest.get("all_texture_edit_sample_frames_independent_full_source_renders") is True and
        court.get("source_sampler_tolerance_rgba_premultiplied_max") == MEASURED_TOL and
        court.get("source_atlas_region_is_art_drawable") is False and
        all(court.get(k) is False for k in (
            "graphics_owner_authorized", "iris_training_authorized",
            "materialization_authorized", "training_authorized")),
        "ACTUAL_PHYSICAL_SOURCE_COURT_NOT_ADMITTED")
    with np.load(root / "original-physical-samples.npz", allow_pickle=False) as arr:
        source=np.asarray(arr["source_uv_edit_cycles_sample_rgba"],dtype=np.float32)
    result=measure_original_source_mobius(source)
    require(
        [result["coefficients_by_exact_edit_subset"][m][
            "source_sample_operation_pixels_above_original_tolerance"] for m in (1,2,4)]
            == court["source_single_edit_observable_samples_per_operation"] ==
               [290979,264554,6048] and
        result["single_edit_image_sample_support_overlap_at_source_tolerance"] ==
           {k:sum(int(v["pairwise_single_edit_image_support_overlaps"][k])
                  for v in court["independent_court_per_camera"])
            for k in ("0_1","0_2","1_2")},
        "INDEPENDENT_SOURCE_SINGLE_EFFECT_COURT_DISAGREES")
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True,
                                 allow_nan=False) + "\n")
    print(json.dumps({
        "status":result["status"],
        "max_binary_interaction":[result["coefficients_by_exact_edit_subset"][m][
            "source_max_abs_linear_pm_rgba_effect"] for m in (3,5,6)],
        "triple_interaction_max":result["coefficients_by_exact_edit_subset"][7][
            "source_max_abs_linear_pm_rgba_effect"],
        "original_source_semantic_owner":"UNKNOWN",
        "training_authorized":False,
    },sort_keys=True),flush=True)
    return result


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    produce(a.root,a.output)
