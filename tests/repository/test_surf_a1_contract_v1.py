from __future__ import annotations

import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]

class SurfA1RepositoryContractTests(unittest.TestCase):
    def test_experiment_ledger_preserves_a1_result_and_current_program(self):
        ledger = json.loads((ROOT / "canonical/IRIS_RESEARCH_EXPERIMENT_LEDGER_V1.json").read_text())
        self.assertEqual(ledger["active_lane"], "SURFACE_TESSA")
        self.assertIn(ledger["active_experiment"], ("H-GEO-P2-SCENE-FIELD", "H-GEO-P3-EXPLICIT-HIRES-CARRIER", "H-GEO-P3-EFREQ", "H-GEO-P3-SCALE-XFER"))
        self.assertIn("RELATION_V10", ledger["parked_lanes"])
        rows = {row["id"]: row for row in ledger["experiments"]}
        self.assertEqual(rows["SURF-A0-RAWFIELD"]["status"], "FALSIFIED")
        self.assertEqual(rows["SURF-C0-NATIVE-R24"]["status"], "MEASURED_DIAGNOSTIC")
        self.assertEqual(rows["SURF-A1-FROZEN-H"]["status"], "SCIENTIFIC_FAIL")
        self.assertEqual(rows["SURF-A1-FROZEN-H"]["next_gate"], "H-REP-P0")
        self.assertFalse(rows["SURF-A1-FROZEN-H"]["scientific_pass_claimed"])
        self.assertEqual(rows["SURF-A1-FROZEN-H"]["independent_court"]["status"], "SCIENTIFIC_FAIL_SURF_A1_FROZEN_H_REPRESENTATION_RESCUE")
        self.assertEqual(rows["H-GEO-P1"]["status"], "SCIENTIFIC_FAIL")
        p2 = rows["H-GEO-P2-SCENE-FIELD"]
        self.assertIn(p2["status"], ("SOURCE_PREPARATION", "SCIENTIFIC_FAIL"))
        if ledger["active_experiment"] in ("H-GEO-P3-EXPLICIT-HIRES-CARRIER", "H-GEO-P3-EFREQ", "H-GEO-P3-SCALE-XFER"):
            self.assertEqual(p2["status"], "SCIENTIFIC_FAIL")
        if ledger["active_experiment"] == "H-GEO-P3-EXPLICIT-HIRES-CARRIER":
            self.assertEqual(rows["H-GEO-P3-EXPLICIT-HIRES-CARRIER"]["status"], "SOURCE_PREPARATION")
        if ledger["active_experiment"] == "H-GEO-P3-EFREQ":
            self.assertEqual(rows["H-GEO-P3-EXPLICIT-HIRES-CARRIER"]["status"], "SCIENTIFIC_FAIL")
            self.assertEqual(rows["H-GEO-P3-EFREQ"]["status"], "SOURCE_PREPARATION")
            self.assertEqual(rows["H-GEO-P3-EFREQ"]["optimizer_steps_authorized"], 0)
            self.assertEqual(rows["H-GEO-P3-EFREQ"]["probe_optimizer_steps_authorized"], 0)
        if ledger["active_experiment"] == "H-GEO-P3-SCALE-XFER":
            self.assertEqual(rows["H-GEO-P3-EXPLICIT-HIRES-CARRIER"]["status"], "SCIENTIFIC_FAIL")
            self.assertEqual(rows["H-GEO-P3-EFREQ"]["status"], "COMPLETED_UNRESOLVED")
            self.assertEqual(rows["H-GEO-P3-EFREQ"]["optimizer_steps_executed"], 0)
            sx = rows["H-GEO-P3-SCALE-XFER"]
            self.assertEqual(sx["status"], "SOURCE_PREPARATION")
            self.assertEqual(sx["optimizer_steps_authorized"], 0)
            self.assertFalse(sx["p4_training_authorized"])
        self.assertTrue(rows["SURF-A2-HIRES"]["status"].startswith("BLOCKED_"))
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
