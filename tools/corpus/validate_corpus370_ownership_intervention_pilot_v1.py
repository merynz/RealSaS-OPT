#!/usr/bin/env python3
from __future__ import annotations

"""Fail-closed preregistration gate for Corpus370 ownership truth interventions.

This validates experimental intent and supervision boundaries only. It never
reads source observations, executes Blender, admits truth, or authorizes training.
The actual results need independent source/geometry/graphics/mechanics courts.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "RealSaS.Corpus370OwnershipInterventionPilot.v1"
VALID_STATUS = "PREREGISTERED_NOT_EXECUTED"
REQUIRED_BASELINE = {"FULL", "STANDALONE", "REMOVAL"}
REQUIRED_PALETTE = {
    "NOOP_ORIGINAL",
    "SINGLE_LOCUS_RECOLOR",
    "SPLIT_ONE_DOMAIN_DIFFERENT_COLORS",
    "COLLISION_DIFFERENT_DOMAINS_SAME_COLOR",
}
REQUIRED_MECHANICS = {
    "ZERO_CONTROL", "PARENT_TO_CHILD", "CHILD_TO_PARENT", "COMMON_ROOT_SHAM",
}
REQUIRED_LABEL_STATES = {"POSITIVE", "NEGATIVE", "NOT_APPLICABLE", "ABSTAIN"}
REQUIRED_TARGETS = {
    "SAME_DRAWABLE", "DIFFERENT_DRAWABLE", "CUT_VS_ATTACHMENT",
    "MECHANICAL_OWNER", "RELATIVE_DEFORMATION", "VIEW_SPECIFIC_OCCLUSION",
    "ALPHA_COMPOSITING", "SOURCE_LOCUS_CANONICAL_JOIN",
}
REQUIRED_PRESERVE = {
    "source_mesh", "source_uv", "source_material_alpha", "camera", "time",
    "rig_and_skin_identity", "source_art_bytes",
}
REQUIRED_TEACHER_ONLY = {
    "source_triangle_id", "source_barycentric", "teacher_material_name",
    "teacher_semantic_name", "source_bone_id", "source_skin_id",
    "source_drawable_id", "teacher_motion_control",
}
SELECTION_GATES = (
    "require_exact_source_sha256",
    "require_current_independent_observation_court",
    "preregister_before_measuring",
    "morphology_strata_required",
    "global_lineage_cluster_disjoint_holdout",
    "all_view_and_palette_variants_share_lineage",
)
MEASUREMENT_GATES = (
    "require_randomized_deterministic_palette",
    "require_noop_source_parity",
    "require_reversible_and_replay_court",
    "require_full_stack_alpha_aware_rays",
)
TARGET_GATES = (
    "independent_court_per_measured_positive",
    "independent_court_per_measured_negative",
    "camera_view_time_conditioning",
    "exact_source_locus_and_canonical_binding",
    "separate_graphical_and_mechanical_ownership",
    "source_art_fidelity",
    "stage48_zero_pair_warning_not_overridden",
)
COMPARISON_GATES = (
    "same_source_and_compute_budget",
    "measure_usable_positive_negative_and_abstain_counts",
    "measure_cross_family_intersection",
    "measure_oracle_consumer_consequence",
    "select_best_arm_using_sealed_train_val_only",
)
PROHIBITION_GATES = (
    "material_or_semantic_name_as_student_label",
    "color_similarity_as_owner_label",
    "source_ids_as_student_input",
    "unverified_positive_or_negative_labels",
    "skipping_missing_subjects_silently",
    "relaxing_existing_geometry_or_two_pixel_thresholds",
    "minting_model_release",
    "starting_optimizer_from_this_spec",
)


def require(ok: bool, code: str) -> None:
    if not ok:
        raise ValueError(code)


def _set_exact(value: Any, expected: set[str], code: str) -> None:
    require(isinstance(value, list), code + "_LIST")
    require(all(isinstance(item, str) for item in value), code + "_TYPE")
    require(len(value) == len(set(value)), code + "_DUPLICATE")
    require(set(value) == expected, code + "_MISSING_OR_EXTRA")


def _must_true(doc: dict[str, Any], fields: tuple[str, ...], code: str) -> None:
    require(isinstance(doc, dict), code + "_OBJECT")
    for field in fields:
        require(doc.get(field) is True, code + "_" + field.upper())


def validate_contract(spec: dict[str, Any]) -> dict[str, Any]:
    require(isinstance(spec, dict), "OWNERSHIP_PILOT_OBJECT")
    require(spec.get("schema") == SCHEMA, "OWNERSHIP_PILOT_SCHEMA")
    require(spec.get("phase") == VALID_STATUS, "OWNERSHIP_PILOT_NOT_PREREGISTERED")
    require(spec.get("scope") == "SMALL_MORPHOLOGY_STRATIFIED_PILOT__NO_FAMILY_AUTHORITY", "OWNERSHIP_PILOT_SCOPE")
    require(spec.get("no_training_authority") is True, "OWNERSHIP_PILOT_TRAINING_FORBIDDEN")
    require(spec.get("source_immutable") is True, "OWNERSHIP_PILOT_SOURCE_MUTATION")
    selection = spec.get("selection")
    _must_true(selection, SELECTION_GATES, "OWNERSHIP_PILOT_SELECTION")
    require(type(selection.get("min_subjects")) is int and type(selection.get("max_subjects")) is int, "OWNERSHIP_PILOT_SUBJECT_INTEGER")
    require(3 <= selection["min_subjects"] <= selection["max_subjects"] <= 8, "OWNERSHIP_PILOT_BOUNDED_SUBJECTS")
    measurement = spec.get("measurement")
    require(isinstance(measurement, dict), "OWNERSHIP_PILOT_MEASUREMENT_OBJECT")
    _set_exact(measurement.get("preserve"), REQUIRED_PRESERVE, "OWNERSHIP_PILOT_SOURCE_PRESERVATION")
    require(measurement.get("source_address") == "source_triangle_id+barycentric+source_sha256", "OWNERSHIP_PILOT_SOURCE_ADDRESS")
    _set_exact(measurement.get("baseline_conditions"), REQUIRED_BASELINE, "OWNERSHIP_PILOT_BASELINE")
    _set_exact(measurement.get("palette_conditions"), REQUIRED_PALETTE, "OWNERSHIP_PILOT_PALETTE")
    _set_exact(measurement.get("mechanical_conditions"), REQUIRED_MECHANICS, "OWNERSHIP_PILOT_MECHANICS")
    _must_true(measurement, MEASUREMENT_GATES, "OWNERSHIP_PILOT_MEASUREMENT")
    require(type(measurement.get("palette_seed")) is int and measurement["palette_seed"] >= 0, "OWNERSHIP_PILOT_SEED")
    require(measurement.get("missing_mechanics") == "MASKED_NOT_ZERO", "OWNERSHIP_PILOT_MISSING_MECHANICS")
    require(measurement.get("unknown_secondary_physics") == "ABSTAIN_NOT_STIFFNESS_LABEL", "OWNERSHIP_PILOT_UNOBSERVABLE_PHYSICS")

    arms = spec.get("student_arms")
    require(isinstance(arms, list) and len(arms) == 2, "OWNERSHIP_PILOT_TWO_ARMS")
    expected_arms = {
        "UNMODIFIED_RGB_TARGET_ONLY": ("SOURCE_RGB_ORIGINAL", "QUALIFIED_INTERVENTION_MEASUREMENTS_ONLY"),
        "RECOLORED_OBSERVATION_ABLATION": ("RECOLORED_SOURCE_RGBA", "SAME_QUALIFIED_INTERVENTION_MEASUREMENTS_ONLY"),
    }
    require({a.get("name") for a in arms if isinstance(a, dict)} == set(expected_arms), "OWNERSHIP_PILOT_ARM_IDENTITIES")
    for arm in arms:
        require(isinstance(arm, dict) and arm.get("name") in expected_arms, "OWNERSHIP_PILOT_ARM_OBJECT")
        observation, teacher = expected_arms[arm["name"]]
        require(arm.get("observation") == observation, "OWNERSHIP_PILOT_ARM_OBSERVATION")
        require(arm.get("teacher_targets") == teacher, "OWNERSHIP_PILOT_ARM_TEACHER")
        require(arm.get("same_lineage_split") is True, "OWNERSHIP_PILOT_ARM_LINEAGE")

    student = spec.get("student_visible_fields")
    teacher = spec.get("teacher_only_fields")
    require(isinstance(student, list) and all(isinstance(x, str) for x in student), "OWNERSHIP_PILOT_STUDENT_FIELDS")
    _set_exact(teacher, REQUIRED_TEACHER_ONLY, "OWNERSHIP_PILOT_TEACHER_ONLY")
    require(not (set(student) & set(teacher)), "OWNERSHIP_PILOT_TEACHER_LEAKAGE")
    require("source_rgb" in student and "source_alpha" in student and "camera" in student, "OWNERSHIP_PILOT_STUDENT_OBSERVATION")
    require(spec.get("forbid_semantic_name_as_owner_label") is True, "OWNERSHIP_PILOT_SEMANTIC_AS_OWNER")

    _set_exact(spec.get("require_joint_targets"), REQUIRED_TARGETS, "OWNERSHIP_PILOT_TARGETS")
    _set_exact(spec.get("required_label_states"), REQUIRED_LABEL_STATES, "OWNERSHIP_PILOT_LABEL_STATES")
    _must_true(spec.get("target_requirements"), TARGET_GATES, "OWNERSHIP_PILOT_TARGET")
    comparison = spec.get("comparison")
    _must_true(comparison, COMPARISON_GATES, "OWNERSHIP_PILOT_COMPARISON")
    require(comparison.get("baseline") == "EXISTING_FULL_STANDALONE_REMOVAL", "OWNERSHIP_PILOT_COMPARISON_BASELINE")
    require(comparison.get("if_no_measured_gain") == "KEEP_RECOLOR_AS_TEACHER_ORACLE_ONLY", "OWNERSHIP_PILOT_NO_GAIN")
    _must_true(spec.get("prohibited"), PROHIBITION_GATES, "OWNERSHIP_PILOT_PROHIBITIONS")

    return {
        "schema": "RealSaS.Corpus370OwnershipInterventionPilotContractReceipt.v1",
        "status": "PASS_PREREGISTRATION_CONTRACT_ONLY__NO_MEASURED_SUBJECTS",
        "training_authorized": False,
        "family_authority_minted": False,
        "pilot_executed": False,
        "measured_subject_count": 0,
        "subject_count_range": [selection["min_subjects"], selection["max_subjects"]],
        "required_joint_targets": sorted(REQUIRED_TARGETS),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec", type=Path,
        default=Path(__file__).resolve().parents[2] / "canonical" / "CORPUS370_OWNERSHIP_INTERVENTION_PILOT_V1.json",
    )
    args = parser.parse_args()
    raw = args.spec.read_bytes()
    result = validate_contract(json.loads(raw))
    result["spec_sha256"] = hashlib.sha256(raw).hexdigest()
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
