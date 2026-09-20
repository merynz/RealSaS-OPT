from __future__ import annotations

from pathlib import Path
import ast
import importlib
import json
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
MAINLINE_ROOTS = (ROOT / "models", ROOT / "compiler", ROOT / "runtime")
FROZEN_GEPPETTO_COMPAT_PATH = Path(
    "models/geppetto/reference_strength_v1/geppetto_reference_strength_candidate_v1.py"
)
FROZEN_GEPPETTO_COMPAT_MODULE = (
    "experiments.geppetto_reference_strength_fullstack_v1."
    "rigging_surface_tensorization_v1"
)
PROMOTED_GEPPETTO_TENSORIZATION = (
    ROOT / "models/geppetto/reference_strength_v1/rigging_surface_tensorization_v1.py"
)


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
            "compiler/realsas_compiler_core/product_authority_v1.py",
            "compiler/realsas_compiler_core/surface_addressing_v1.py",
            "compiler/realsas_compiler_core/complete_appearance_authority_v1.py",
            "compiler/realsas_compiler_core/output_presentation_v1.py",
            "compiler/realsas_compiler_services/orchestrator/mainline.py",
            "canonical/MAINLINE_EXECUTION_PLAN_V2.json",
            "canonical/ACTIVE_RUN_V2.json",
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
                rel = path.relative_to(ROOT)
                text = path.read_text(encoding="utf-8")
                try:
                    tree = ast.parse(text, filename=str(path))
                except SyntaxError as exc:
                    violations.append(f"syntax:{rel}:{exc}")
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if any(alias.name == p or alias.name.startswith(p + ".") for p in forbidden_prefixes):
                                violations.append(f"{rel}:{alias.name}")
                    elif isinstance(node, ast.ImportFrom):
                        module = node.module or ""
                        if any(module == p or module.startswith(p + ".") for p in forbidden_prefixes):
                            allowed_frozen_geppetto_seam = (
                                rel == FROZEN_GEPPETTO_COMPAT_PATH
                                and module == FROZEN_GEPPETTO_COMPAT_MODULE
                            )
                            if not allowed_frozen_geppetto_seam:
                                violations.append(f"{rel}:{module}")
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
                                violations.append(f"{rel}:dynamic:{name}")
        self.assertEqual(violations, [], "mainline dependency firewall violations:\n" + "\n".join(violations))

        promoted = importlib.import_module(
            "models.geppetto.reference_strength_v1.rigging_surface_tensorization_v1"
        )
        historical_binding = sys.modules.get(FROZEN_GEPPETTO_COMPAT_MODULE)
        self.assertIsNotNone(historical_binding, "frozen Geppetto compatibility alias was not installed")
        self.assertIs(
            historical_binding,
            promoted,
            "historical Geppetto tensorization name must resolve to the promoted models module object",
        )
        self.assertEqual(
            Path(promoted.__file__).resolve(),
            PROMOTED_GEPPETTO_TENSORIZATION.resolve(),
            "compatibility seam resolved outside the promoted semantic home",
        )

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

    def test_historical_prefit_gate_is_preserved_but_not_current_authority(self) -> None:
        gate = json.loads((ROOT / "canonical/CANONICAL_MAIN_BEFORE_FIT_GATE_V1_20260904.json").read_text(encoding="utf-8"))
        restoration = (ROOT / "RESTORATION_STATE.md").read_text(encoding="utf-8")
        current = (ROOT / "CURRENT_STATE.md").read_text(encoding="utf-8")
        self.assertFalse(bool(gate["fit_authorized_now"]))
        self.assertIn("SUPERSEDED FOR CONTINUATION", restoration)
        self.assertIn("SUBJECT2_KNIGHT_FULL_CLOSURE", current)
        self.assertIn("SurfaceAddressingIR", current)
        self.assertIn("Complete Appearance Authority", current)

    def test_state_and_index_point_to_current_product_authorities(self) -> None:
        restoration = (ROOT / "RESTORATION_STATE.md").read_text(encoding="utf-8")
        state = (ROOT / "CURRENT_STATE.md").read_text(encoding="utf-8")
        index = (ROOT / "SYSTEM_INDEX.md").read_text(encoding="utf-8")
        self.assertIn("SUPERSEDED FOR CONTINUATION", restoration)
        self.assertIn("SurfaceAddressingIR", state)
        self.assertIn("Complete Appearance Authority", state)
        self.assertIn("46-stage V2", state)
        self.assertIn("Product mesh", index)
        self.assertIn("Complete Appearance", index)
        self.assertIn("surface_addressing_v1.py", index)

    def test_closure_workflow_is_manual_only(self) -> None:
        workflow = (ROOT / ".github/workflows/restoration_closure_manual.yml").read_text(encoding="utf-8")
        self.assertIn("workflow_dispatch", workflow)
        self.assertNotIn("\n  push:", workflow)
        self.assertNotIn("\n  pull_request:", workflow)


if __name__ == "__main__":
    unittest.main()
