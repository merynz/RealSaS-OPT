from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
CURRENT_REGISTRY = "canonical/EXPERIMENT_REGISTRY_V2.json"
CURRENT_JOURNAL = "canonical/SCIENTIFIC_JOURNAL_V2_20260909.jsonl"
HISTORICAL_REGISTRY = "canonical/EXPERIMENT_REGISTRY_V1.json"
HISTORICAL_JOURNAL = "canonical/SCIENTIFIC_JOURNAL_V1.jsonl"
GEPPETTO_EVIDENCE = "canonical/GEPPETTO_FIT1_EVIDENCE_MANIFEST_V1.json"


class CurrentContinuityAuthorityV2(unittest.TestCase):
    def test_machine_pointers_agree(self) -> None:
        context = json.loads((ROOT / "canonical/CONTEXT_STATE_V1.json").read_text(encoding="utf-8"))
        authority = json.loads((ROOT / "canonical/AUTHORITY_MAP_V1.json").read_text(encoding="utf-8"))

        self.assertEqual(context["authority"]["experiment_registry"], CURRENT_REGISTRY)
        self.assertEqual(context["authority"]["scientific_journal"], CURRENT_JOURNAL)
        self.assertEqual(context["authority"]["historical_experiment_registry"], HISTORICAL_REGISTRY)
        self.assertEqual(context["authority"]["historical_scientific_journal"], HISTORICAL_JOURNAL)

        self.assertEqual(authority["experiment_registry"], CURRENT_REGISTRY)
        self.assertEqual(authority["scientific_journal"], CURRENT_JOURNAL)
        self.assertEqual(authority["historical_registry"], HISTORICAL_REGISTRY)
        self.assertEqual(authority["historical_journal"], HISTORICAL_JOURNAL)
        self.assertEqual(authority["geppetto_fit1_evidence_manifest"], GEPPETTO_EVIDENCE)

        for rel in (CURRENT_REGISTRY, CURRENT_JOURNAL, HISTORICAL_REGISTRY, HISTORICAL_JOURNAL, GEPPETTO_EVIDENCE):
            self.assertTrue((ROOT / rel).is_file(), rel)

    def test_agent_contract_names_current_before_historical(self) -> None:
        text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        for token in (CURRENT_REGISTRY, CURRENT_JOURNAL, HISTORICAL_REGISTRY, HISTORICAL_JOURNAL):
            self.assertIn(token, text)
        self.assertLess(text.index(CURRENT_JOURNAL), text.index(HISTORICAL_JOURNAL))
        self.assertLess(text.index(CURRENT_REGISTRY), text.index(HISTORICAL_REGISTRY))
        self.assertIn("remain historical", text)

    def test_context_generators_index_current_v2_sources(self) -> None:
        audit = (ROOT / "tools/audit_context_coverage.py").read_text(encoding="utf-8")
        catalog = (ROOT / "tools/build_knowledge_artifact_catalog.py").read_text(encoding="utf-8")
        renderer = (ROOT / "tools/render_rehydration_packet.py").read_text(encoding="utf-8")

        for text in (audit, catalog):
            self.assertIn(CURRENT_REGISTRY, text)
            self.assertIn(CURRENT_JOURNAL, text)
            self.assertIn(HISTORICAL_REGISTRY, text)
            self.assertIn(HISTORICAL_JOURNAL, text)
        self.assertIn("GEPPETTO_FIT1_EVIDENCE_MANIFEST_V1.json", renderer)
        self.assertIn('context.get("authority", {})', renderer)

    def test_live_context_workflow_watches_v2_and_is_self_hosted(self) -> None:
        workflow = (ROOT / ".github/workflows/live_authority_map.yml").read_text(encoding="utf-8")
        self.assertIn(CURRENT_REGISTRY, workflow)
        self.assertIn(CURRENT_JOURNAL, workflow)
        self.assertIn(GEPPETTO_EVIDENCE, workflow)
        self.assertIn("runs-on: [self-hosted, linux, x64, realsas]", workflow)
        self.assertNotIn("runs-on: ubuntu-latest", workflow)

    def test_live_context_validates_before_generated_commit(self) -> None:
        workflow = (ROOT / ".github/workflows/live_authority_map.yml").read_text(encoding="utf-8")
        validate = workflow.index("- name: Enforce live-context validity")
        publish = workflow.index("- name: Publish generated views to job summary")
        commit = workflow.index("- name: Commit refreshed generated views")
        self.assertLess(validate, publish)
        self.assertLess(validate, commit)

    def test_current_source_gate_is_self_hosted(self) -> None:
        workflow = (ROOT / ".github/workflows/model_mainline_source_gate.yml").read_text(encoding="utf-8")
        self.assertIn("runs-on: [self-hosted, linux, x64, realsas]", workflow)
        self.assertNotIn("runs-on: ubuntu-latest", workflow)
        self.assertIn("test_geppetto_fit1_evidence_manifest_v1.py", workflow)

    def test_authority_map_rehydration_binds_current_evidence(self) -> None:
        authority = json.loads((ROOT / "canonical/AUTHORITY_MAP_V1.json").read_text(encoding="utf-8"))
        required = set(authority["required_files"])
        order = authority["rehydration_order"]
        self.assertIn(GEPPETTO_EVIDENCE, required)
        self.assertIn(CURRENT_REGISTRY, required)
        self.assertIn(CURRENT_JOURNAL, required)
        self.assertIn(GEPPETTO_EVIDENCE, order)
        self.assertIn(CURRENT_REGISTRY, order)
        self.assertIn(CURRENT_JOURNAL, order)

    def test_active_experiment_records_satisfy_live_map_contract(self) -> None:
        authority = json.loads((ROOT / "canonical/AUTHORITY_MAP_V1.json").read_text(encoding="utf-8"))
        for exp in authority.get("active_experiments", []):
            with self.subTest(experiment=exp.get("id", "<missing-id>")):
                for key in ("id", "branch", "status", "question"):
                    self.assertIn(key, exp)
                    self.assertIsInstance(exp[key], str)
                    self.assertTrue(exp[key].strip())
                self.assertIn("does_not_prove", exp)
                self.assertIsInstance(exp["does_not_prove"], list)
                self.assertTrue(exp["does_not_prove"])
                for claim in exp["does_not_prove"]:
                    self.assertIsInstance(claim, str)
                    self.assertTrue(claim.strip())


if __name__ == "__main__":
    unittest.main()
