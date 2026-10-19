from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class SurfA1RepositoryContractTests(unittest.TestCase):
    def test_experiment_ledger_has_one_active_lane_and_parked_relation_v10(self):
        ledger = json.loads((ROOT / "canonical/IRIS_RESEARCH_EXPERIMENT_LEDGER_V1.json").read_text())
        self.assertEqual(ledger["active_lane"], "SURFACE_TESSA")
        self.assertIn("RELATION_V10", ledger["parked_lanes"])
        rows = {row["id"]: row for row in ledger["experiments"]}
        self.assertEqual(rows["SURF-A0-RAWFIELD"]["status"], "FALSIFIED")
        self.assertEqual(rows["SURF-C0-NATIVE-R24"]["status"], "MEASURED_DIAGNOSTIC")
        self.assertEqual(rows["SURF-A1-FROZEN-H"]["next_gate"], "SOURCE_CONTRACT_AND_ZERO_STEP_A100_PREFLIGHT")
        self.assertEqual(rows["SURF-A2-HIRES"]["status"], "BLOCKED_ON_SURF_A1")
        self.assertEqual(rows["TESSA-V2-B1"]["status"], "BLOCKED_ON_SURF_A2_READINESS")

    def test_a1_prereg_seals_v8_and_forbids_teacher_forward(self):
        p = json.loads((ROOT / "canonical/SURF_A1_FROZEN_H_PREREGISTRATION_V1.json").read_text())
        self.assertFalse(p["model_authority"]["scene_core_trainable"])
        self.assertFalse(p["new_surface_decoder"]["teacher_topology_as_input"])
        self.assertFalse(p["new_surface_decoder"]["teacher_rig_skin_as_input"])
        self.assertFalse(p["new_surface_decoder"]["uncalibrated_uncertainty_allowed"])
        self.assertTrue(p["runtime_admission"]["zero_step_native_r24_surface_parity_required"])
        self.assertTrue(p["runtime_admission"]["zero_step_native_r24_faces_exact_required"])
        self.assertEqual(p["runtime_admission"]["maximum_zero_step_native_r24_vertex_difference_iris"], 2e-6)
        self.assertTrue(p["runtime_admission"]["all_64_raster_parity_views_required"])
        self.assertFalse(p["runtime_admission"]["reduced_capacity_or_resolution_fallback_allowed"])
        self.assertEqual(p["optimizer"]["subject_policy"], "FIT8_SHARED_ARCHITECTURE__DETERMINISTIC_SUBJECT_ROUND_ROBIN__NO_SUBJECT_SPECIFIC_PARAMETERS")
        self.assertFalse(p["interpretation"]["a1_pass_authorizes_tessa_training"])

    def test_single_consent_notebook_contract(self):
        ux = json.loads((ROOT / "canonical/COLAB_RESEARCH_WORKER_UX_V1.json").read_text())
        ui = ux["interaction_contract"]
        self.assertEqual(ui["drive_user_consent_count"], 1)
        self.assertEqual(ui["drive_mount_calls_allowed"], 1)
        self.assertFalse(ui["additional_google_oauth_ui_allowed"])
        self.assertFalse(ui["auth_authenticate_user_allowed"])
        self.assertFalse(ui["rclone_interactive_config_allowed"])
        self.assertEqual(ui["intended_operator_action"], "OPEN_NOTEBOOK__RUN_ALL__APPROVE_DRIVE_ONCE__OBSERVE_LOGS")

    def test_teacher_binding_is_hash_authority_not_mount_path_authority(self):
        b = json.loads((ROOT / "canonical/SURF_A1_FROZEN_H_INPUT_BINDING_V1.json").read_text())
        self.assertFalse(b["teacher_truth_allowed_in_inference_forward"])
        self.assertEqual(len(b["authored_teachers"]["subjects"]), 8)
        self.assertEqual(b["authored_teachers"]["batch_receipt"]["sha256"], "37b203f9c16e1a5dddbab1709ebb3806e486d66d1ed5b48a42817bc0044a7843")
        self.assertTrue(b["transport_policy"]["sha256_is_authority"])
        self.assertTrue(b["transport_policy"]["mounted_path_is_not_authority"])
        self.assertTrue(b["transport_policy"]["colab_drive_mount_only"])
        self.assertTrue(b["transport_policy"]["second_google_oauth_forbidden"])

    def test_tessa_prereg_cannot_consume_historical_raw_surface_or_a1(self):
        p = json.loads((ROOT / "canonical/TESSA_V2_FIT8_PREREGISTRATION_V1.json").read_text())
        upstream = p["upstream_surface_authority"]
        self.assertEqual(upstream["required_experiment"], "SURF-A2-HIRES")
        self.assertFalse(upstream["surf_a0_raw_field_sparse_allowed"])
        self.assertFalse(upstream["surf_c0_native_r24_allowed_as_training_surface"])
        self.assertFalse(upstream["surf_a1_frozen_h_allowed_as_final_direct_arm"])
        self.assertEqual(p["conditioning"]["surface"], "QUALIFIED_SURF_A2_HIGH_RESOLUTION_DIRECT_WITNESS")
        self.assertFalse(p["runtime_admission"]["historical_raw_t512_t1024_surface_allowed"])

    def test_a1_runtime_tools_exist_and_are_separated_by_authority(self):
        paths = [
            "tools/iris_vnext/build_fit8_surface_raster_truth_v1.py",
            "tools/iris_vnext/preflight_surf_a1_a100_v1.py",
            "tools/iris_vnext/train_fit8_surf_a1_frozen_h_v1.py",
            "tools/iris_vnext/audit_fit8_surf_a1_representation_v1.py",
            "models/iris/vnext/surface_rescue_v1.py",
            "models/iris/vnext/surface_rescue_raster_v1.py",
            "models/iris/vnext/surface_rescue_training_v1.py",
        ]
        for rel in paths:
            self.assertTrue((ROOT / rel).is_file(), rel)
        preflight = (ROOT / paths[1]).read_text()
        trainer = (ROOT / paths[2]).read_text()
        court = (ROOT / paths[3]).read_text()
        self.assertIn('"optimizer_steps_executed": 0', preflight)
        self.assertIn("PASS_SURF_A1_A100_PREFLIGHT__TRAINING_AUTHORIZED", preflight)
        self.assertIn("COMPLETED_SURF_A1_2400_STEPS__AWAITING_INDEPENDENT_COURT", trainer)
        self.assertIn("SCIENTIFIC_PASS_SURF_A1_FROZEN_H_REPRESENTATION_RESCUE", court)
        self.assertNotIn("optimizer.step()", preflight)
        self.assertNotIn("optimizer.step()", court)


if __name__ == "__main__":
    unittest.main()
