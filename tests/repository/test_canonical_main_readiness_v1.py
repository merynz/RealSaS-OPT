from __future__ import annotations

from pathlib import Path
import ast
import json
import unittest

ROOT = Path(__file__).resolve().parents[2]
MAINLINE_ROOTS = (ROOT / "models", ROOT / "compiler", ROOT / "runtime")


class CanonicalMainReadinessV1(unittest.TestCase):
    def test_living_system_homes_exist(self) -> None:
        required = (
            "models/iris/v2/model_v2.py",
            "models/geppetto/v2/geppetto_candidate_v2.py",
            "models/skin_field_codec/v1/skin_field_codec_v1.py",
            "models/arachne/v2/arachne_candidate_v2.py",
            "compiler/realsas_compiler_core/substrate/surface.py",
            "compiler/realsas_compiler_core/substrate/local_geometry.py",
            "compiler/realsas_compiler_core/substrate/iris_v2.py",
            "compiler/realsas_compiler_core/mesh/mwb2.py",
            "compiler/realsas_compiler_core/mesh/mwb2_skin.py",
            "compiler/realsas_compiler_core/mesh/mesh_binding.py",
            "compiler/realsas_compiler_core/directional_binding.py",
            "compiler/realsas_compiler_services/proof/directional_motion_provider.py",
            "compiler/realsas_compiler_services/proof/directional_motion_evaluator.py",
            "compiler/realsas_compiler_services/export/current_v4_runtime_v2.py",
            "compiler/realsas_compiler_services/export/runtime_v2.py",
            "runtime/realsas_cpp/CMakeLists.txt",
            "runtime/reference_v4/consumer.py",
            "SYSTEM_INDEX.md",
            "RESTORATION_STATE.md",
        )
        missing = [path for path in required if not (ROOT / path).is_file()]
        self.assertEqual(missing, [], "missing canonical mainline files:\n" + "\n".join(missing))

    def test_mainline_has_no_experiment_or_historical_runtime_imports(self) -> None:
        violations: list[str] = []
        forbidden_prefixes = ("experiments", "historical")
        for source_root in MAINLINE_ROOTS:
            if not source_root.exists():
                continue
            for path in source_root.rglob("*.py"):
                text = path.read_text(encoding="utf-8")
                try:
                    tree = ast.parse(text, filename=str(path))
                except SyntaxError as exc:
                    violations.append(f"syntax:{path.relative_to(ROOT)}:{exc}")
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if any(alias.name == p or alias.name.startswith(p + ".") for p in forbidden_prefixes):
                                violations.append(f"{path.relative_to(ROOT)}:{alias.name}")
                    elif isinstance(node, ast.ImportFrom):
                        module = node.module or ""
                        if any(module == p or module.startswith(p + ".") for p in forbidden_prefixes):
                            violations.append(f"{path.relative_to(ROOT)}:{module}")
                    elif isinstance(node, ast.Call):
                        fn = node.func
                        dynamic_import = (
                            isinstance(fn, ast.Name) and fn.id == "__import__"
                        ) or (
                            isinstance(fn, ast.Name) and fn.id == "import_module"
                        ) or (
                            isinstance(fn, ast.Attribute) and fn.attr == "import_module"
                        )
                        if dynamic_import and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                            name = node.args[0].value
                            if any(name == p or name.startswith(p + ".") for p in forbidden_prefixes):
                                violations.append(f"{path.relative_to(ROOT)}:dynamic:{name}")
        self.assertEqual(violations, [], "mainline dependency firewall violations:\n" + "\n".join(violations))

    def test_retracted_directional_shortcut_is_not_current_source(self) -> None:
        forbidden = (
            "compiler/realsas_compiler_services/proof/motion_probe.py",
            "compiler/realsas_compiler_services/proof/motion_probe_geometry.py",
            "compiler/realsas_compiler_services/export/current_v4_projection.py",
            "tests/compiler/test_authored_motion_probe_v1.py",
        )
        present = [path for path in forbidden if (ROOT / path).exists()]
        self.assertEqual(present, [], "retracted source still present:\n" + "\n".join(present))

    def test_current_authority_records_are_present(self) -> None:
        required = (
            "canonical/AUTHORED_MOTION_PROOF_RETRACTION_V1_20260903.json",
            "canonical/P0_DIRECTIONAL_BINDING_RUNTIME_INTERLOCK_CLOSURE_V1_20260904.json",
            "canonical/HISTORICAL_NUMERICS_SOURCE_DIFF_DISPOSITION_V1_20260904.json",
            "canonical/REPAIR_EXECUTION_AUTHORITY_DISPOSITION_V1_20260904.json",
            "canonical/CANONICAL_MAIN_BEFORE_FIT_GATE_V1_20260904.json",
            "canonical/COMPILER_RUNTIME_PROMOTION_SOURCE_SEAL_V1_20260903.json",
            "canonical/RESTORATION_CLOSURE_VERDICT_V1_20260904.json",
        )
        missing = [path for path in required if not (ROOT / path).is_file()]
        self.assertEqual(missing, [], "missing canonical authority records:\n" + "\n".join(missing))

    def test_fit_is_blocked_until_canonical_main_post_merge_integrity(self) -> None:
        gate = json.loads((ROOT / "canonical/CANONICAL_MAIN_BEFORE_FIT_GATE_V1_20260904.json").read_text(encoding="utf-8"))
        self.assertFalse(bool(gate["fit_authorized_now"]))
        self.assertFalse(bool(gate["fit_branch_may_precede_canonical_main"]))
        self.assertFalse(bool(gate["fit_may_run_on_restoration_branch"]))
        self.assertTrue(bool(gate["first_fit_must_record_canonical_main_base_commit"]))
        order = tuple(gate["required_order"])
        self.assertLess(order.index("PROMOTE_RESTORATION_TREE_TO_CANONICAL_MAIN"), order.index("ONLY_THEN_AUTHORIZE_REAL_FAMILY_FIT"))
        self.assertLess(order.index("POST_MERGE_MAIN_REPOSITORY_INTEGRITY_CHECK"), order.index("ONLY_THEN_AUTHORIZE_REAL_FAMILY_FIT"))

    def test_state_and_index_are_not_stale_on_closed_decisions(self) -> None:
        state = (ROOT / "RESTORATION_STATE.md").read_text(encoding="utf-8")
        index = (ROOT / "SYSTEM_INDEX.md").read_text(encoding="utf-8")
        for text in (state, index):
            self.assertNotIn("CURRENT_DIRECTIONAL_JOINT_VIEW_BINDING_MISSING", text)
            self.assertIn("CANONICAL_MAIN_BEFORE_FIT_GATE_V1_20260904.json", text)
            self.assertIn("RESTORATION_CLOSURE_VERDICT_V1_20260904.json", text)
        self.assertIn("Real-family fit:** `NOT AUTHORIZED`", state)
        self.assertIn("Full behavioral + complete-E2E restoration closure | **DONE / PASS**", state)
        self.assertIn("Native current-source interlock: **PASS", index)
        self.assertIn("Restoration-wide source/regression/E2E/native closure: **PASS", index)
        self.assertIn("Canonical `main` promotion", index)

    def test_closure_workflow_is_manual_only(self) -> None:
        workflow = (ROOT / ".github/workflows/restoration_closure_manual.yml").read_text(encoding="utf-8")
        self.assertIn("workflow_dispatch", workflow)
        self.assertNotIn("\n  push:", workflow)
        self.assertNotIn("\n  pull_request:", workflow)


if __name__ == "__main__":
    unittest.main()
