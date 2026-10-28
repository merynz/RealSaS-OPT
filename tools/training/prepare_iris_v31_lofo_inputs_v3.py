#!/usr/bin/env python3
"""IRIS V3.1 LOFO input preparation V3 with exact family observability.

V3 preserves the V2 leakage/observability court but treats a completely
unobservable Stage48 pair family as an explicit ABSTAIN instead of inventing
negative targets or blocking independently observable geometry, mechanics and
appearance supervision. Partial cross-fold pair evidence remains fail-closed:
if pair rows exist, at least two distinct FIT8 subjects must carry them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from models.training_runtime.durable_state_v1 import file_sha256
from tools.training.prepare_iris_v31_lofo_inputs_v1 import atomic_json, prepare as prepare_v1
from tools.training.prepare_iris_v31_lofo_inputs_v2 import (
    _digest,
    _git_head,
    _require,
    _target_census,
)


def _projection_qualification_v3(manifest: dict, presentation_bundle: dict, source_code_sha: str) -> dict:
    _require(manifest.get("teacher_inputs_in_forward") is False, "IRIS_V31_PROJECTION_TEACHER_FORWARD_FORBIDDEN")
    _require(manifest.get("semantic_part_taxonomy_used") is False, "IRIS_V31_PROJECTION_SEMANTIC_TAXONOMY_FORBIDDEN")
    _require(presentation_bundle.get("semantic_part_taxonomy_used") is False, "IRIS_V31_PROJECTION_BUNDLE_SEMANTICS_FORBIDDEN")
    _require(presentation_bundle.get("knight_is_training_subject") is False, "IRIS_V31_PROJECTION_KNIGHT_TRAINING_FORBIDDEN")
    _require(
        presentation_bundle.get("bundle_builder_revision") == "V4_OBJECTIVE_EXACT_PAIR_OBSERVABILITY_ADMISSION"
        and presentation_bundle.get("subject_binding_revision") == "V3_PLATFORM_UUID_TO_LOGICAL_KEY_FROM_EXACT_STAGE48_PAYLOAD",
        "IRIS_V31_PROJECTION_OBJECTIVE_EXACT_BUNDLE_REQUIRED",
    )
    _require(
        presentation_bundle.get("source_code_sha") == source_code_sha,
        "IRIS_V31_PROJECTION_STAGE48_SOURCE_CODE_DRIFT",
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
    bundle_pair_authorized = presentation_bundle.get("pair_student_supervision_authorized")
    bundle_pair_subjects = int(presentation_bundle.get("pair_evidence_subject_count", -1))
    bundle_pair_rows = int(presentation_bundle.get("pair_evidence_valid_rows", -1))
    _require(
        bundle_pair_subjects == pair_subjects and bundle_pair_rows == total_pair_valid,
        "IRIS_V31_PROJECTION_PAIR_CENSUS_DRIFT",
    )

    if pair_subjects >= 2 and total_pair_valid > 0:
        _require(bundle_pair_authorized is True, "IRIS_V31_PROJECTION_PAIR_BUNDLE_AUTHORITY_DRIFT")
        pair_training_authorized = True
        pair_observability = "MECHANISM_CONDITIONAL__CROSS_FOLD_EVIDENCE"
        pair_supervision = "CONDITIONAL_EVIDENCE_ONLY__NOT_DRAW_ORDER_AUTHORITY"
        forward_inputs = [
            "FROZEN_DINO_FINAL384", "FROZEN_DINO_LAYER2_384", "RAW_ALPHA_PATCH196",
            "CAMERA13", "NORMALIZED_QUERY_XYZ", "NORMALIZED_PAIR_QUERY_XYZ",
        ]
    elif pair_subjects == 0 and total_pair_valid == 0:
        _require(bundle_pair_authorized is False, "IRIS_V31_PROJECTION_PAIR_BUNDLE_AUTHORITY_DRIFT")
        _require(
            presentation_bundle.get("pair_observability")
            == "UNAVAILABLE__ZERO_JOINTLY_CONTRIBUTING_SAME_RAY_ROWS__ABSTAIN",
            "IRIS_V31_PROJECTION_PAIR_ABSTAIN_BINDING",
        )
        pair_training_authorized = False
        pair_observability = "UNAVAILABLE__OBSERVATION_CONDITIONED_EVIDENCE_EMPTY"
        pair_supervision = "ABSTAIN__NO_PAIR_GRADIENTS_OR_VALIDATION_METRIC"
        forward_inputs = [
            "FROZEN_DINO_FINAL384", "FROZEN_DINO_LAYER2_384", "RAW_ALPHA_PATCH196",
            "CAMERA13", "NORMALIZED_QUERY_XYZ",
        ]
    else:
        raise ValueError("IRIS_V31_PROJECTION_CROSS_FOLD_PAIR_EVIDENCE_INSUFFICIENT")

    qualification = {
        "schema": "RealSaS.IRISV31StudentProjectionQualification.v1",
        "status": "PASS_EXACT_STUDENT_PROJECTION__TRAINING_AUTHORIZED",
        "source_code_sha": source_code_sha,
        "forward_input_contract": forward_inputs,
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
                "observability": pair_observability,
                "objective": "SIGMOID_BCE__BOUNDED_ATTENUATION_PARAMETER" if pair_training_authorized else "ABSTAIN",
                "student_supervision": pair_supervision,
                "requires_both_ray_endpoints_contribute": True,
            },
        },
        "geometric_front_order_student_objective": False,
        "compiler_oracle_retains_draw_order_authority": True,
        "pair_training_authorized": pair_training_authorized,
        "pair_evidence_subject_count": pair_subjects,
        "pair_evidence_valid_rows": total_pair_valid,
        "cross_fold_pair_evidence_minimum_subjects": 2,
        "missing_pair_truth_becomes_negative": False,
        "subjects": rows,
        "heldout_selection_blind": True,
        "fit8_pass_claimed": False,
        "product_pass_claimed": False,
        "genericity_claimed": False,
    }
    qualification["witness_sha256"] = _digest(qualification)
    return qualification


def prepare(**kwargs):
    source_code_sha = _git_head()
    manifest = prepare_v1(**kwargs)
    for row in manifest.get("subjects") or ():
        cache = row.get("frozen_feature_cache") or {}
        cache.pop("cache_hit", None)
    manifest["runtime_cache_hit_excluded_from_scientific_identity"] = True

    bundle_path = Path(kwargs["presentation_bundle"]).resolve()
    _require(bundle_path.is_file(), "IRIS_V31_PROJECTION_BUNDLE_MISSING")
    presentation_bundle = json.loads(bundle_path.read_text())
    qualification = _projection_qualification_v3(manifest, presentation_bundle, source_code_sha)
    manifest["source_code_sha"] = source_code_sha
    manifest["student_projection_qualification"] = qualification
    manifest["student_training_authorized"] = True
    manifest["training_authority_scope"] = "EXACT_V31_RESEARCH_LOFO_INPUT_PROJECTION__PAIR_FAMILY_MAY_ABSTAIN"
    manifest["status"] = "LOFO_INPUTS_SEALED__TRAINING_AUTHORIZED_BY_EXACT_PROJECTION_COURT"

    output = Path(kwargs["output"]).resolve()
    path = output / "IRIS_V31_LOFO_INPUT_MANIFEST_V1.json"
    atomic_json(path, manifest)
    print(
        json.dumps(
            {
                "manifest": str(path),
                "sha256": file_sha256(path),
                "source_code_sha": source_code_sha,
                "cache_hit_in_scientific_identity": False,
                "student_training_authorized": True,
                "projection_witness_sha256": qualification["witness_sha256"],
                "pair_training_authorized": qualification["pair_training_authorized"],
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
