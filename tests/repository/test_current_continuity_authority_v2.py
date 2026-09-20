from __future__ import annotations

import json
from pathlib import Path
import unittest

from compiler.realsas_compiler_services.orchestrator.mainline import validate_ledger


ROOT = Path(__file__).resolve().parents[2]
CURRENT_CONTEXT = "canonical/CONTEXT_STATE_V2.json"
CURRENT_REGISTRY = "canonical/EXPERIMENT_REGISTRY_V3.json"
CURRENT_JOURNAL = "canonical/SCIENTIFIC_JOURNAL_V2_20260909.jsonl"
HISTORICAL_REGISTRY = "canonical/EXPERIMENT_REGISTRY_V2.json"
HISTORICAL_JOURNAL = "canonical/SCIENTIFIC_JOURNAL_V1.jsonl"
ACTIVE_RUN = "canonical/ACTIVE_RUN_V2.json"
PIPELINE_PLAN = "canonical/MAINLINE_EXECUTION_PLAN_V2.json"
PRODUCT_AUTHORITY = "canonical/QUALIFIED_MESH_PRESENTATION_AUTHORITY_V1_20260918.md"
ARCHITECTURE_V2 = "canonical/REALSAS_CANONICAL_ARCHITECTURE_V2_20260920.json"
CAA_V1 = "canonical/COMPLETE_APPEARANCE_AUTHORITY_V1_20260920.json"


class CurrentContinuityAuthorityV2(unittest.TestCase):
    def test_machine_pointers_agree(self) -> None:
        context = json.loads((ROOT / CURRENT_CONTEXT).read_text(encoding="utf-8"))
        authority = json.loads((ROOT / "canonical/AUTHORITY_MAP_V1.json").read_text(encoding="utf-8"))
        self.assertEqual(authority["context_state"], CURRENT_CONTEXT)
        self.assertEqual(context["authority"]["experiment_registry"], CURRENT_REGISTRY)
        self.assertEqual(context["authority"]["scientific_journal"], CURRENT_JOURNAL)
        self.assertEqual(context["authority"]["historical_experiment_registry"], HISTORICAL_REGISTRY)
        self.assertEqual(context["authority"]["historical_scientific_journal"], HISTORICAL_JOURNAL)
        self.assertEqual(context["authority"]["active_run_ledger"], ACTIVE_RUN)
        self.assertEqual(authority["experiment_registry"], CURRENT_REGISTRY)
        self.assertEqual(authority["scientific_journal"], CURRENT_JOURNAL)
        self.assertEqual(authority["historical_experiment_registry"], HISTORICAL_REGISTRY)
        self.assertEqual(authority["historical_scientific_journal"], HISTORICAL_JOURNAL)
        self.assertEqual(authority["active_run_ledger"], ACTIVE_RUN)
        self.assertEqual(authority["pipeline_plan"], PIPELINE_PLAN)
        for rel in (CURRENT_CONTEXT, CURRENT_REGISTRY, CURRENT_JOURNAL, HISTORICAL_REGISTRY,
                    HISTORICAL_JOURNAL, ACTIVE_RUN, PIPELINE_PLAN, PRODUCT_AUTHORITY, ARCHITECTURE_V2, CAA_V1):
            self.assertTrue((ROOT / rel).is_file(), rel)

    def test_agent_contract_names_current_before_historical(self) -> None:
        text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        for token in (CURRENT_REGISTRY, CURRENT_JOURNAL, HISTORICAL_REGISTRY,
                      HISTORICAL_JOURNAL, ACTIVE_RUN, PIPELINE_PLAN, PRODUCT_AUTHORITY):
            self.assertIn(token, text)
        self.assertLess(text.index(CURRENT_JOURNAL), text.index(HISTORICAL_JOURNAL))
        self.assertLess(text.index(CURRENT_REGISTRY), text.index(HISTORICAL_REGISTRY))
        self.assertIn("historical", text.lower())

    def test_authority_map_rehydration_binds_current_execution(self) -> None:
        authority = json.loads((ROOT / "canonical/AUTHORITY_MAP_V1.json").read_text(encoding="utf-8"))
        required = set(authority["required_files"])
        order = authority["rehydration_order"]
        for rel in (CURRENT_CONTEXT, CURRENT_REGISTRY, CURRENT_JOURNAL, ACTIVE_RUN, PIPELINE_PLAN, PRODUCT_AUTHORITY):
            self.assertIn(rel, required)
        for rel in (ACTIVE_RUN, PIPELINE_PLAN, "CURRENT_STATE.md", "canonical/AUTHORITY_MAP_V1.json", PRODUCT_AUTHORITY):
            self.assertIn(rel, order)

    def test_plan_and_active_run_hash_contract_agree(self) -> None:
        plan = json.loads((ROOT / PIPELINE_PLAN).read_text(encoding="utf-8"))
        ledger = json.loads((ROOT / ACTIVE_RUN).read_text(encoding="utf-8"))
        validate_ledger(plan, ledger)
        self.assertEqual(plan["stage_count"], 46)
        self.assertEqual(plan["stages"][17]["id"], "18_CANONICAL_MESH_ADDRESSING_BUILD")
        self.assertEqual(plan["stages"][23]["id"], "24_COMPLETE_APPEARANCE_QUALIFIED")
        self.assertEqual(plan["stages"][45]["id"], "46_PRODUCT_CLOSURE_SEAL")

    def test_current_self_hosted_workflows_are_main_bound(self) -> None:
        for rel in (
            ".github/workflows/current_mainline_self_hosted_ci.yml",
            ".github/workflows/model_mainline_source_gate.yml",
            ".github/workflows/live_authority_map.yml",
        ):
            text = (ROOT / rel).read_text(encoding="utf-8")
            self.assertIn("self-hosted", text)
            self.assertIn("main", text)

    def test_active_experiment_records_satisfy_live_map_contract(self) -> None:
        authority = json.loads((ROOT / "canonical/AUTHORITY_MAP_V1.json").read_text(encoding="utf-8"))
        active = authority.get("active_experiments", [])
        self.assertEqual([x["id"] for x in active], ["SUBJECT2_KNIGHT_FULL_CLOSURE"])
        for exp in active:
            for key in ("id", "branch", "status", "question"):
                self.assertIsInstance(exp[key], str)
                self.assertTrue(exp[key].strip())
            self.assertIsInstance(exp["does_not_prove"], list)
            self.assertTrue(exp["does_not_prove"])


if __name__ == "__main__":
    unittest.main()
