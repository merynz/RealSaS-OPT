from __future__ import annotations

import json
from pathlib import Path
import unittest

from tools.iris_vnext.train_fit8_tessa_v3_structural_evidence_ab_v1 import (
    ARM_A,
    ARM_B,
    ARMS,
    learning_rate_at_step,
    validate_preregistration,
)


ROOT = Path(__file__).resolve().parents[2]
PREREG = ROOT / "canonical" / "STRUCTURAL_EVIDENCE_AB_PREREGISTRATION_V2.json"
METRICS = ROOT / "canonical" / "STRUCTURAL_EVIDENCE_AB_COURT_METRIC_DEFINITIONS_V1.json"
TRAINER = ROOT / "tools" / "iris_vnext" / "train_fit8_tessa_v3_structural_evidence_ab_v1.py"
RESUMABLE_TRAINER = ROOT / "tools" / "iris_vnext" / "train_fit8_tessa_v3_structural_evidence_ab_v2.py"
PREFLIGHT = ROOT / "tools" / "iris_vnext" / "preflight_structural_evidence_ab_a100_v1.py"
FREE_RUN = ROOT / "tools" / "iris_vnext" / "evaluate_structural_evidence_ab_free_running_v1.py"
TOPOLOGY_COURT = ROOT / "tools" / "iris_vnext" / "court_structural_evidence_ab_topology_surface_v1.py"
PRESENTATION_COURT = ROOT / "tools" / "iris_vnext" / "court_structural_evidence_ab_static_presentation_v1.py"
DYNAMIC_COURT = ROOT / "tools" / "iris_vnext" / "court_structural_evidence_ab_dynamic_motion_v1.py"


class StructuralEvidenceABExecutionV1Tests(unittest.TestCase):
    def setUp(self):
        self.prereg = json.loads(PREREG.read_text(encoding="utf-8"))
        self.protocol = validate_preregistration(self.prereg)

    def test_arm_order_and_locked_horizon(self):
        self.assertEqual(ARMS, (ARM_A, ARM_B))
        self.assertEqual(ARM_A, "A_NULL_PRESENTATION")
        self.assertEqual(ARM_B, "B_ROLE_FREE_PRESENTATION")
        self.assertEqual(self.protocol["optimizer_steps_per_arm"], 2400)
        self.assertEqual(self.protocol["seed_set"], [26091991])

    def test_learning_rate_matches_locked_warmup_cosine_contract(self):
        peak = float(self.protocol["optimizer"]["learning_rate"])
        floor = float(self.protocol["minimum_learning_rate"])
        self.assertAlmostEqual(learning_rate_at_step(1, self.protocol), peak / 120.0, places=14)
        self.assertAlmostEqual(learning_rate_at_step(120, self.protocol), peak, places=14)
        self.assertAlmostEqual(learning_rate_at_step(2400, self.protocol), floor, places=14)

    def test_teacher_all_hit_truth_is_forbidden_as_inference_conditioning(self):
        authority = self.prereg["evidence_input_authority"]
        self.assertIs(authority["teacher_all_hit_evidence_allowed_as_inference_conditioning"], False)
        source = TRAINER.read_text(encoding="utf-8")
        self.assertIn('materialized_v2_truth_in_conditioning', source)
        self.assertNotIn('--all-hit', source)
        self.assertNotIn('--teacher-mesh', source)
        self.assertNotIn('--teacher-topology', source)

    def test_execution_sources_are_fail_closed_before_result_courts(self):
        ordinary = (TRAINER, RESUMABLE_TRAINER, PREFLIGHT, FREE_RUN)
        for path in ordinary:
            source = path.read_text(encoding="utf-8")
            self.assertIn('scientific_pass_claimed', source, path.name)
            self.assertNotIn('"scientific_pass_claimed": True', source, path.name)
        for path in (TOPOLOGY_COURT, PRESENTATION_COURT):
            source = path.read_text(encoding="utf-8")
            self.assertIn('"scientific_ab_pass_claimed": False', source, path.name)
            self.assertNotIn('"scientific_ab_pass_claimed": True', source, path.name)
        preflight = PREFLIGHT.read_text(encoding="utf-8")
        self.assertIn('optimizer_steps_executed": 0', preflight)
        self.assertNotIn('\n    optimizer.step(', preflight)
        self.assertNotIn('\n        optimizer.step(', preflight)

    def test_resumable_trainer_restores_exact_model_optimizer_rng_lineage(self):
        source = RESUMABLE_TRAINER.read_text(encoding="utf-8")
        self.assertIn('_restore_rng', source)
        self.assertIn('optimizer.load_state_dict', source)
        self.assertIn('model.load_state_dict', source)
        self.assertIn('initial_state_sha256', source)
        self.assertIn('RESUME_LINEAGE_DRIFT', source)
        self.assertIn('DETERMINISTIC_CHECKPOINT_RESUME_WITH_EXACT_RNG_MODEL_OPTIMIZER_LINEAGE', source)
        self.assertNotIn('--all-hit', source)
        self.assertNotIn('--teacher-topology', source)

    def test_free_running_uses_both_arms_and_no_sampling(self):
        source = FREE_RUN.read_text(encoding="utf-8")
        self.assertIn('for arm in ARMS', source)
        self.assertIn('sampling=False', source)
        self.assertIn('teacher_prefix_used', source)

    def test_topology_surface_court_is_not_final_scientific_decision(self):
        source = TOPOLOGY_COURT.read_text(encoding="utf-8")
        self.assertIn('topology_surface_no_regression_pass', source)
        self.assertIn('presentation_primary_court_complete', source)
        self.assertIn('motion_primary_court_complete', source)
        self.assertIn('"scientific_ab_pass_claimed": False', source)

    def test_static_presentation_court_is_evaluation_only_and_leaves_pose_flip_pending(self):
        source = PRESENTATION_COURT.read_text(encoding="utf-8")
        for metric in (
            "GLOBAL_VISIBILITY_ACCURACY",
            "LOCAL_PARTIAL_ORDER_ACCURACY",
            "FRAGMENT_IDENTITY_STABILITY",
            "SOURCE_ART_SUPPORT_PRESERVATION",
        ):
            self.assertIn(metric, source)
        self.assertIn('POSE_DEPENDENT_ORDER_FLIP_ACCURACY', source)
        self.assertIn('"v2_all_hit_truth_used_as_inference_conditioning": False', source)
        self.assertIn('"presentation_primary_final_pass_claimed": False', source)
        self.assertIn('"scientific_ab_pass_claimed": False', source)

    def test_dynamic_court_is_teacher_evaluation_only_and_uses_frozen_g3_bounds(self):
        source = DYNAMIC_COURT.read_text(encoding="utf-8")
        for metric in (
            "POSE_DEPENDENT_ORDER_FLIP_ACCURACY",
            "OWNER_FRAGMENT_STABILITY_OVER_MOTION",
            "GAP_TEAR_RATE",
            "ATTACHMENT_PERSISTENCE_WHERE_AUTHORITATIVE",
            "G3_DYNAMIC_QUALIFICATION",
        ):
            self.assertIn(metric, source)
        self.assertIn('teacher_rig_skin_motion_used_in_inference_forward', source)
        self.assertIn('INDEPENDENT_EVALUATION_DEFORMATION_FIELD_ONLY', source)
        self.assertIn('QUALIFIED_MESH_PRODUCT_POLICY_V2_20260919.json', source)
        self.assertIn('canonical_m_authorized', source)
        definitions = json.loads(METRICS.read_text(encoding="utf-8"))
        self.assertEqual(definitions["status"], "FROZEN_BEFORE_AB_OPTIMIZER_EXECUTION")
        self.assertIs(definitions["authority_boundary"]["teacher_or_all_hit_truth_allowed_as_inference_conditioning"], False)
        self.assertIs(definitions["aggregation"]["post_result_definition_or_threshold_relaxation_allowed"], False)
        self.assertEqual(definitions["presentation_dynamic"]["ray_grid"]["grid_size"], 32)


if __name__ == "__main__":
    unittest.main()
