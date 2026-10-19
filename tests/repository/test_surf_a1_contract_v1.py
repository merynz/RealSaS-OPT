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

    def test_a1_prereg_seals_v8_and_forbids_teacher_forward(self):
        p = json.loads((ROOT / "canonical/SURF_A1_FROZEN_H_PREREGISTRATION_V1.json").read_text())
        self.assertFalse(p["model_authority"]["scene_core_trainable"])
        self.assertFalse(p["new_surface_decoder"]["teacher_topology_as_input"])
        self.assertFalse(p["new_surface_decoder"]["teacher_rig_skin_as_input"])
        self.assertFalse(p["new_surface_decoder"]["uncalibrated_uncertainty_allowed"])
        self.assertTrue(p["runtime_admission"]["zero_step_native_r24_surface_parity_required"])
        self.assertFalse(p["runtime_admission"]["reduced_capacity_or_resolution_fallback_allowed"])
        self.assertFalse(p["interpretation"]["a1_pass_authorizes_tessa_training"])

    def test_single_consent_notebook_contract(self):
        ux = json.loads((ROOT / "canonical/COLAB_RESEARCH_WORKER_UX_V1.json").read_text())
        ui = ux["interaction_contract"]
        self.assertEqual(ui["drive_user_consent_count"], 1)
        self.assertEqual(ui["drive_mount_calls_allowed"], 1)
        self.assertFalse(ui["additional_google_oauth_ui_allowed"])
        self.assertFalse(ui["auth_authenticate_user_allowed"])
        self.assertFalse(ui["rclone_interactive_config_allowed"])

    def test_teacher_binding_is_hash_authority_not_mount_path_authority(self):
        b = json.loads((ROOT / "canonical/SURF_A1_FROZEN_H_INPUT_BINDING_V1.json").read_text())
        self.assertFalse(b["teacher_truth_allowed_in_inference_forward"])
        self.assertEqual(len(b["authored_teachers"]["subjects"]), 8)
        self.assertEqual(b["authored_teachers"]["batch_receipt"]["sha256"], "37b203f9c16e1a5dddbab1709ebb3806e486d66d1ed5b48a42817bc0044a7843")
        self.assertTrue(b["transport_policy"]["sha256_is_authority"])
        self.assertFalse(b["transport_policy"]["mounted_path_is_not_authority"] is False)


if __name__ == "__main__":
    unittest.main()
