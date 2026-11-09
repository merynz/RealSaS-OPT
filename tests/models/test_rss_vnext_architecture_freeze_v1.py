"""Contract-only regression for frozen RSS.vNext design; no model qualification."""
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "canonical/RSS_VNEXT_UNIFIED_CHARACTER_ARCHITECTURE_FREEZE_V1_20261109.json"
DOCUMENT = ROOT / "docs/research/RSS_VNEXT_UNIFIED_CHARACTER_ARCHITECTURE_FREEZE_20261109.md"


class RSSvNextArchitectureFreezeV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.freeze = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_four_experts_and_one_carrier(self):
        f = self.freeze
        self.assertEqual(f["architecture_id"], "RSS.vNext.UnifiedCharacterCompiler.v1")
        self.assertEqual([e["name"] for e in f["experts"]], ["IRIS", "TESSA", "AXIS", "MIRA"])
        self.assertEqual(f["shared_carrier"]["name"], "SharedCharacterCarrierIR.vNext")
        self.assertIn("IMMUTABLE_SOURCE_EVIDENCE", f["shared_carrier"]["lifecycle"])
        self.assertIn("TEACHER", f["shared_carrier"]["truth_firewall"])

    def test_non_overlapping_core_authorities(self):
        f = self.freeze
        scopes = {e["name"]: set(e["owns"]) for e in f["experts"]}
        for left, right in [("IRIS", "TESSA"), ("TESSA", "AXIS"),
                            ("AXIS", "MIRA"), ("MIRA", "IRIS")]:
            self.assertFalse(scopes[left] & scopes[right],
                             f"Unversioned same-owner output in {left}/{right}")
        self.assertIn("ANIMATION_READY_STRUCTURAL_MESH_PROPOSAL", scopes["TESSA"])
        self.assertIn("EDITABLE_RIGPROGRAM", scopes["AXIS"])
        self.assertIn("SKINTOKENS_CODEC_AND_BINDING", scopes["MIRA"])

    def test_gradient_and_discrete_boundaries(self):
        f = self.freeze
        self.assertIn("NOT_DIRECTLY_DIFFERENTIABLE", f["tessa"]["discrete_gradient_policy"])
        self.assertIn("MESH_CONNECTIVITY", f["gradient_policy"]["hard_decisions"])
        self.assertIn("NO_SILENT_MESH_MUTATION_AFTER_RIG_OR_SKIN_BINDING",
                      f["runtime_invariants"])
        self.assertIn("PRE_REPAIR_AND_POST_REPAIR_QUALITY_SEPARATE",
                      f["runtime_invariants"])

    def test_no_unearned_runtime_or_training_claims(self):
        status = self.freeze["readiness"]
        self.assertTrue(status["architecture_design_frozen"])
        self.assertFalse(status["implementation_complete"])
        self.assertFalse(status["trained_joint_model"])
        self.assertFalse(status["generic_animation_ready_unseen_proven"])
        self.assertFalse(status["full_control_unseen_proven"])
        self.assertFalse(status["product_deploy_authorized"])
        self.assertFalse(status["truth_foundry_mutated"])

    def test_agent_entry_and_live_ledger_are_bound_to_freeze(self):
        ledger = self.freeze["work_ledger"]
        self.assertEqual(ledger["hub_issue"], 399)
        self.assertEqual(ledger["catalog"], "canonical/RSS_VNEXT_WORK_LEDGER_INDEX_V1.json")
        self.assertTrue(ledger["task_completion_must_not_imply_model_training_or_product_admission"])
        self.assertTrue((ROOT / ledger["catalog"]).is_file())
        self.assertTrue((ROOT / ledger["agent_protocol"]).is_file())
        agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        self.assertIn("RSS.vNext mandatory agent entry", agents)
        self.assertIn("rss_vnext_status.py", agents)

    def test_document_exists_and_matches_design(self):
        self.assertEqual(
            self.freeze["normative_document"], str(DOCUMENT.relative_to(ROOT)))
        content = DOCUMENT.read_text(encoding="utf-8")
        for key in ("SharedCharacterCarrierIR.vNext", "IRIS", "TESSA",
                    "AXIS", "MIRA", "FlexiCubes", "RigProgramIR.vNext"):
            self.assertIn(key, content)


if __name__ == "__main__":
    unittest.main()
