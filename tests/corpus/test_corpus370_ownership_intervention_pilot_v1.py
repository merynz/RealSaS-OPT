from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from tools.corpus.validate_corpus370_ownership_intervention_pilot_v1 import (
    validate_contract,
)

SPEC_PATH = (
    Path(__file__).resolve().parents[2]
    / "canonical"
    / "CORPUS370_OWNERSHIP_INTERVENTION_PILOT_V1.json"
)


class OwnershipInterventionPreregisrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = json.loads(SPEC_PATH.read_text())

    def reject(self, code: str) -> None:
        with self.assertRaisesRegex(ValueError, code):
            validate_contract(self.spec)

    def test_valid_spec_does_not_mint_truth_or_training(self) -> None:
        receipt = validate_contract(self.spec)
        self.assertEqual(
            receipt["status"],
            "PASS_PREREGISTRATION_CONTRACT_ONLY__NO_MEASURED_SUBJECTS",
        )
        self.assertFalse(receipt["training_authorized"])
        self.assertFalse(receipt["family_authority_minted"])
        self.assertFalse(receipt["pilot_executed"])
        self.assertEqual(receipt["measured_subject_count"], 0)

    def test_rejects_training_promotion(self) -> None:
        self.spec["no_training_authority"] = False
        self.reject("OWNERSHIP_PILOT_TRAINING_FORBIDDEN")

    def test_rejects_semantic_name_as_owner_label(self) -> None:
        self.spec["forbid_semantic_name_as_owner_label"] = False
        self.reject("OWNERSHIP_PILOT_SEMANTIC_AS_OWNER")

    def test_rejects_student_teacher_name_leakage(self) -> None:
        self.spec["student_visible_fields"].append("teacher_material_name")
        self.reject("OWNERSHIP_PILOT_TEACHER_LEAKAGE")

    def test_rejects_unregistered_semantic_student_observation(self) -> None:
        self.spec["student_visible_fields"].append("semantic_body_name")
        self.reject("OWNERSHIP_PILOT_STUDENT_OBSERVATIONS_MISSING_OR_EXTRA")

    def test_rejects_unregistered_source_identifier_alias(self) -> None:
        self.spec["student_visible_fields"].append("mesh_component_id")
        self.reject("OWNERSHIP_PILOT_STUDENT_OBSERVATIONS_MISSING_OR_EXTRA")

    def test_rejects_duplicate_student_observation(self) -> None:
        self.spec["student_visible_fields"].append("source_rgb")
        self.reject("OWNERSHIP_PILOT_STUDENT_OBSERVATIONS_DUPLICATE")

    def test_rejects_missing_student_observation(self) -> None:
        self.spec["student_visible_fields"].remove("source_alpha")
        self.reject("OWNERSHIP_PILOT_STUDENT_OBSERVATIONS_MISSING_OR_EXTRA")

    def test_rejects_palette_collision_control_removed(self) -> None:
        self.spec["measurement"]["palette_conditions"].remove(
            "COLLISION_DIFFERENT_DOMAINS_SAME_COLOR"
        )
        self.reject("OWNERSHIP_PILOT_PALETTE_MISSING_OR_EXTRA")

    def test_rejects_palette_split_control_removed(self) -> None:
        self.spec["measurement"]["palette_conditions"].remove(
            "SPLIT_ONE_DOMAIN_DIFFERENT_COLORS"
        )
        self.reject("OWNERSHIP_PILOT_PALETTE_MISSING_OR_EXTRA")

    def test_rejects_fake_negative_supervision(self) -> None:
        self.spec["target_requirements"][
            "independent_court_per_measured_negative"
        ] = False
        self.reject("OWNERSHIP_PILOT_TARGET_INDEPENDENT_COURT_PER_MEASURED_NEGATIVE")

    def test_rejects_unknown_filled_as_zero(self) -> None:
        self.spec["measurement"]["missing_mechanics"] = "ZERO"
        self.reject("OWNERSHIP_PILOT_MISSING_MECHANICS")

    def test_rejects_cross_lineage_arm_split(self) -> None:
        self.spec["student_arms"][1]["same_lineage_split"] = False
        self.reject("OWNERSHIP_PILOT_ARM_LINEAGE")

    def test_rejects_unaudited_target_omission(self) -> None:
        self.spec["require_joint_targets"].remove("SOURCE_LOCUS_CANONICAL_JOIN")
        self.reject("OWNERSHIP_PILOT_TARGETS_MISSING_OR_EXTRA")

    def test_rejects_broad_unreviewed_corpus_scan(self) -> None:
        self.spec["selection"]["max_subjects"] = 352
        self.reject("OWNERSHIP_PILOT_BOUNDED_SUBJECTS")

    def test_rejects_relaxed_thresholds(self) -> None:
        self.spec["prohibited"]["relaxing_existing_geometry_or_two_pixel_thresholds"] = False
        self.reject("OWNERSHIP_PILOT_PROHIBITIONS_RELAXING_EXISTING_GEOMETRY_OR_TWO_PIXEL_THRESHOLDS")

    def test_rejects_fake_motion_physics(self) -> None:
        self.spec["measurement"]["unknown_secondary_physics"] = "KNOWN_STIFFNESS"
        self.reject("OWNERSHIP_PILOT_UNOBSERVABLE_PHYSICS")

    def test_rejects_nonreversible_recolor(self) -> None:
        self.spec["measurement"]["require_reversible_and_replay_court"] = False
        self.reject("OWNERSHIP_PILOT_MEASUREMENT_REQUIRE_REVERSIBLE_AND_REPLAY_COURT")

    def test_rejects_original_art_mutation(self) -> None:
        self.spec["source_immutable"] = False
        self.reject("OWNERSHIP_PILOT_SOURCE_MUTATION")


if __name__ == "__main__":
    unittest.main()
