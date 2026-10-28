#!/usr/bin/env python3
"""Stable, fail-closed IRIS V3.1 LOFO input preparation.

V1 builds exact frozen observation caches and binds the legacy F* geometry shards
plus Stage48 targets. V2 adds two requirements before training:

1. runtime cache-hit state is excluded from scientific identity; and
2. a target-projection observability/leakage court proves that the model forward
   can see only admitted observation features, cameras and typed query addresses.

Stage48 itself remains an oracle candidate and does not authorize student
supervision. This downstream court authorizes only the exact V3.1 projection:
- mechanics: mechanism-conditional Gaussian mean/log-variance supervision;
- appearance: deterministic RGBA only on single-contributor observed rays;
- pair attenuation: mechanism-conditional bounded sigmoid/BCE evidence.

No semantic part, owner, material, control, bone, attachment or grip identifier is
permitted in the student target packet or forward-input contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.training.prepare_iris_v31_lofo_inputs_v1 import atomic_json, prepare as prepare_v1
from models.training_runtime.durable_state_v1 import file_sha256


TARGET_FIELDS = {
    "mechanical_xyz",
    "mechanical_target",
    "mechanical_valid",
    "appearance_xyz",
    "appearance_camera_index",
    "appearance_rgba",
    "appearance_valid",
    "pair_xyz_a",
    "pair_xyz_b",
    "pair_camera_index",
    "pair_target",
    "pair_valid",
}
FORBIDDEN_TARGET_TOKENS = (
    "owner",
    "material",
    "bone",
    "control",
    "attachment",
    "grip",
    "semantic",
    "class",
    "cape",
    "hair",
    "cloth",
    "weapon",
)


def _require(value, code):
    if not value:
        raise ValueError(code)


def _digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _target_census(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        fields = set(z.files)
        _require(fields == TARGET_FIELDS, "IRIS_V31_PROJECTION_TARGET_FIELDS")
        _require(
            not any(token in name.lower() for name in fields for token in FORBIDDEN_TARGET_TOKENS),
            "IRIS_V31_PROJECTION_PRIVILEGED_TARGET_FIELD",
        )
        values = {name: np.asarray(z[name]) for name in z.files}
    for name, value in values.items():
        _require(value.dtype != object, "IRIS_V31_PROJECTION_OBJECT_ARRAY:" + name)
        if np.issubdtype(value.dtype, np.floating):
            _require(np.isfinite(value).all(), "IRIS_V31_PROJECTION_NONFINITE:" + name)
    for name in ("mechanical_valid", "appearance_valid", "pair_valid"):
        _require(values[name].dtype == np.bool_, "IRIS_V31_PROJECTION_VALIDITY_DTYPE:" + name)
    mechanical_valid = values["mechanical_valid"]
    appearance_valid = values["appearance_valid"]
    pair_valid = values["pair_valid"]
    _require(np.count_nonzero(mechanical_valid) > 0, "IRIS_V31_PROJECTION_EMPTY_MECHANICS")
    _require(np.count_nonzero(appearance_valid) > 0, "IRIS_V31_PROJECTION_EMPTY_DIRECT_APPEARANCE")
    _require(values["mechanical_target"].shape == (len(mechanical_valid), 2), "IRIS_V31_PROJECTION_MECHANICAL_LAYOUT")
    _require(values["appearance_rgba"].shape == (len(appearance_valid), 4), "IRIS_V31_PROJECTION_APPEARANCE_LAYOUT")
    _require(values["pair_target"].shape == (len(pair_valid), 1), "IRIS_V31_PROJECTION_PAIR_LAYOUT")
    if len(pair_valid):
        valid_pair_targets = values["pair_target"][pair_valid]
        _require(
            np.all((valid_pair_targets >= 0.0) & (valid_pair_targets <= 1.0)),
            "IRIS_V31_PROJECTION_PAIR_TARGET_RANGE",
        )
    return {
        "mechanical_rows": int(len(mechanical_valid)),
        "mechanical_valid": int(np.count_nonzero(mechanical_valid)),
        "appearance_rows": int(len(appearance_valid)),
        "appearance_valid": int(np.count_nonzero(appearance_valid)),
        "pair_rows": int(len(pair_valid)),
        "pair_valid": int(np.count_nonzero(pair_valid)),
    }


def _projection_qualification(manifest: dict, presentation_bundle: dict) -> dict:
    _require(manifest.get("teacher_inputs_in_forward") is False, "IRIS_V31_PROJECTION_TEACHER_FORWARD_FORBIDDEN")
    _require(manifest.get("semantic_part_taxonomy_used") is False, "IRIS_V31_PROJECTION_SEMANTIC_TAXONOMY_FORBIDDEN")
    _require(presentation_bundle.get("semantic_part_taxonomy_used") is False, "IRIS_V31_PROJECTION_BUNDLE_SEMANTICS_FORBIDDEN")
    _require(presentation_bundle.get("knight_is_training_subject") is False, "IRIS_V31_PROJECTION_KNIGHT_TRAINING_FORBIDDEN")
    _require(
        presentation_bundle.get("bundle_builder_revision") == "V2_OBJECTIVE_EXACT_OBSERVABILITY_COUNTS",
        "IRIS_V31_PROJECTION_OBJECTIVE_EXACT_BUNDLE_REQUIRED",
    )

    rows = []
    pair_subjects = 0
    total_pair_valid = 0
    for subject in manifest.get("subjects") or ():
        path = Path(subject["presentation_targets"]["path"]).resolve()
        _require(path.is_file(), "IRIS_V31_PROJECTION_TARGET_MISSING:" + subject["subject_id"])
        _require(file_sha256(path) == subject["presentation_targets"]["sha256"], "IRIS_V31_PROJECTION_TARGET_DRIFT:" + subject["subject_id"])
        census = _target_census(path)
        if census["pair_valid"] > 0:
            pair_subjects += 1
            total_pair_valid += census["pair_valid"]
        rows.append({"subject_id": subject["subject_id"], **census})

    _require(len(rows) == 8, "IRIS_V31_PROJECTION_EIGHT_SUBJECTS_REQUIRED")
    # At least two distinct subjects must carry conditional pair evidence. Then
    # every 7/1 fold retains at least one pair-evidence training subject even when
    # one of those two is held out. Never advertise an occlusion objective that is
    # silently absent from an entire fold.
    _require(pair_subjects >= 2 and total_pair_valid > 0, "IRIS_V31_PROJECTION_CROSS_FOLD_PAIR_EVIDENCE_INSUFFICIENT")
    qualification = {
        "schema": "RealSaS.IRISV31StudentProjectionQualification.v1",
        "status": "PASS_EXACT_STUDENT_PROJECTION__TRAINING_AUTHORIZED",
        "forward_input_contract": [
            "FROZEN_DINO_FINAL384",
            "FROZEN_DINO_LAYER2_384",
            "RAW_ALPHA_PATCH196",
            "CAMERA13",
            "NORMALIZED_QUERY_XYZ",
            "NORMALIZED_PAIR_QUERY_XYZ",
        ],
        "teacher_inputs_in_forward": False,
        "privileged_ids_in_target_packet": False,
        "semantic_part_taxonomy_used": False,
        "families": {
            "mechanics": {
                "observability": "MECHANISM_CONDITIONAL",
                "objective": "GAUSSIAN_NLL__MEAN_PLUS_LOG_VARIANCE",
                "student_supervision": "CONDITIONAL_DISTRIBUTION_AUTHORIZED",
                "source_scale_normalized": True,
            },
            "appearance": {
                "observability": "OBSERVATION_CONDITIONED",
                "objective": "RGBA_SMOOTH_L1",
                "student_supervision": "DETERMINISTIC_ONLY_ON_EXACTLY_ONE_MEASURABLE_RAY_CONTRIBUTOR",
                "overlap_factored_teacher_colour_is_masked": True,
            },
            "occlusion_attenuation": {
                "observability": "MECHANISM_CONDITIONAL",
                "objective": "SIGMOID_BCE__BOUNDED_ATTENUATION_PARAMETER",
                "student_supervision": "CONDITIONAL_EVIDENCE_ONLY__NOT_DRAW_ORDER_AUTHORITY",
                "requires_both_ray_endpoints_contribute": True,
            },
        },
        "geometric_front_order_student_objective": False,
        "compiler_oracle_retains_draw_order_authority": True,
        "pair_evidence_subject_count": pair_subjects,
        "pair_evidence_valid_rows": total_pair_valid,
        "cross_fold_pair_evidence_minimum_subjects": 2,
        "subjects": rows,
        "heldout_selection_blind": True,
        "fit8_pass_claimed": False,
        "product_pass_claimed": False,
        "genericity_claimed": False,
    }
    qualification["witness_sha256"] = _digest(qualification)
    return qualification


def prepare(**kwargs):
    manifest = prepare_v1(**kwargs)
    for row in manifest.get("subjects") or ():
        cache = row.get("frozen_feature_cache") or {}
        cache.pop("cache_hit", None)
    manifest["runtime_cache_hit_excluded_from_scientific_identity"] = True

    bundle_path = Path(kwargs["presentation_bundle"]).resolve()
    _require(bundle_path.is_file(), "IRIS_V31_PROJECTION_BUNDLE_MISSING")
    presentation_bundle = json.loads(bundle_path.read_text())
    qualification = _projection_qualification(manifest, presentation_bundle)
    manifest["student_projection_qualification"] = qualification
    manifest["student_training_authorized"] = True
    manifest["training_authority_scope"] = "EXACT_V31_RESEARCH_LOFO_INPUT_PROJECTION_ONLY"
    manifest["status"] = "LOFO_INPUTS_SEALED__TRAINING_AUTHORIZED_BY_EXACT_PROJECTION_COURT"

    output = Path(kwargs["output"]).resolve()
    path = output / "IRIS_V31_LOFO_INPUT_MANIFEST_V1.json"
    atomic_json(path, manifest)
    print(
        json.dumps(
            {
                "manifest": str(path),
                "sha256": file_sha256(path),
                "cache_hit_in_scientific_identity": False,
                "student_training_authorized": True,
                "projection_witness_sha256": qualification["witness_sha256"],
                "pair_evidence_subject_count": qualification["pair_evidence_subject_count"],
                "pair_evidence_valid_rows": qualification["pair_evidence_valid_rows"],
                "folds": len(manifest.get("folds") or ()),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "input-root", "legacy-manifest", "legacy-shard-root", "presentation-bundle",
        "presentation-root", "source-dir", "weight-path", "cache-root", "output",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--view-batch-size", type=int, default=8)
    args = parser.parse_args()
    prepare(
        input_root=args.input_root,
        legacy_manifest=args.legacy_manifest,
        legacy_shard_root=args.legacy_shard_root,
        presentation_bundle=args.presentation_bundle,
        presentation_root=args.presentation_root,
        source_dir=args.source_dir,
        weight_path=args.weight_path,
        cache_root=args.cache_root,
        output=args.output,
        device=args.device,
        view_batch_size=args.view_batch_size,
    )
