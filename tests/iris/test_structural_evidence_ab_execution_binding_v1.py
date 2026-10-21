from __future__ import annotations

import json
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
BINDING_PATH = ROOT / "canonical" / "STRUCTURAL_EVIDENCE_AB_EXECUTION_BINDING_V1.json"
LEDGER_PATH = ROOT / "canonical" / "IRIS_RESEARCH_EXPERIMENT_LEDGER_V4.json"


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


class StructuralEvidenceABExecutionBindingV1Tests(unittest.TestCase):
    def setUp(self):
        self.binding = json.loads(BINDING_PATH.read_text(encoding="utf-8"))
        self.ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))

    def test_exact_execution_commit_and_bound_blobs_exist(self):
        self.assertEqual(self.binding["schema"], "RealSaS.StructuralEvidenceABExecutionBinding.v1")
        source = self.binding["execution_source"]
        execution_sha = source["merge_sha"]
        self.assertEqual(execution_sha, "ebfe80fc9d0c43cc8075d3d38bdbc9231357b969")
        self.assertEqual(source["pull_request"], 139)
        git("cat-file", "-e", execution_sha + "^{commit}")

        authority = self.binding["locked_authority"]
        checks = (
            ("preregistration_path", "preregistration_git_blob_sha"),
            ("court_metric_definitions_path", "court_metric_definitions_git_blob_sha"),
            ("evidence_admission_path", "evidence_admission_git_blob_sha"),
        )
        for path_key, sha_key in checks:
            observed = git("rev-parse", f"{execution_sha}:{authority[path_key]}")
            self.assertEqual(observed, authority[sha_key], authority[path_key])

    def test_admitted_v2_hashes_match_repository_authority(self):
        admission = json.loads((ROOT / self.binding["locked_authority"]["evidence_admission_path"]).read_text(encoding="utf-8"))
        authority = self.binding["locked_authority"]
        self.assertEqual(admission["status"], "ADMITTED_TIE_SAFE_HASHED_EVIDENCE_ONLY__NO_SCIENTIFIC_PASS")
        self.assertEqual(admission["drive"]["manifest"]["sha256"], authority["admitted_v2_manifest_sha256"])
        self.assertEqual(admission["drive"]["receipt"]["sha256"], authority["admitted_v2_receipt_sha256"])
        self.assertEqual(admission["payload_totals"]["oriented_pair_rows"], authority["admitted_v2_oriented_pair_rows"])
        self.assertIs(admission["execution_facts"]["scientific_pass_claimed"], False)

    def test_preregistration_protocol_matches_binding(self):
        prereg = json.loads((ROOT / self.binding["locked_authority"]["preregistration_path"]).read_text(encoding="utf-8"))
        program = self.binding["execution_program"]
        protocol = prereg["training_protocol"]
        self.assertEqual(prereg["status"], "LOCKED_PREOPTIMIZER_SOURCE_CONTRACT__USER_HANDOFF_REQUIRED")
        self.assertEqual(protocol["seed_set"], program["seed_set"])
        self.assertEqual(protocol["optimizer_steps_per_arm"], program["optimizer_steps_per_arm"])
        self.assertEqual(program["arm_order"], [prereg["arm_a"]["id"], prereg["arm_b"]["id"]])
        self.assertIs(prereg["evidence_input_authority"]["teacher_all_hit_evidence_allowed_as_inference_conditioning"], False)

    def test_notebook_is_build_only_until_user_handoff(self):
        notebook = self.binding["notebook_contract"]
        facts = self.binding["execution_facts_at_binding_time"]
        self.assertIs(notebook["build_authorized"], True)
        self.assertIs(notebook["execution_authorized"], False)
        self.assertIs(notebook["optimizer_execution_authorized"], False)
        self.assertIs(notebook["requires_user_handoff_before_execution"], True)
        self.assertIs(facts["scientific_training_executed"], False)
        self.assertEqual(facts["optimizer_steps_executed"], 0)
        self.assertIs(facts["scientific_ab_court_executed"], False)
        self.assertIs(facts["scientific_pass_or_fail_observed"], False)

    def test_v4_ledger_points_to_exact_same_boundary(self):
        self.assertEqual(self.ledger["schema"], "RealSaS.IRISResearchExperimentLedger.v4")
        self.assertEqual(self.ledger["ab_execution_source"]["merge_sha"], self.binding["execution_source"]["merge_sha"])
        self.assertEqual(self.ledger["ab_execution_source"]["execution_binding"], "canonical/STRUCTURAL_EVIDENCE_AB_EXECUTION_BINDING_V1.json")
        boundary = self.ledger["next_execution_boundary"]
        self.assertIs(boundary["requires_new_run_all_notebook"], True)
        self.assertIs(boundary["notebook_build_authorized"], True)
        self.assertIs(boundary["notebook_execution_authorized"], False)
        self.assertIs(boundary["requires_user_handoff_before_execution"], True)


if __name__ == "__main__":
    unittest.main()
