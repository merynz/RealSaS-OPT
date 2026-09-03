from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments" / "iris_reprojection_v2_20260831"
TARGET = ROOT / "models" / "iris" / "v2"

PROMOTED = (
    "__init__.py",
    "checkpoint_v2.py",
    "depth_output_head_v2.py",
    "dino_token_parity_v2.py",
    "dinov2_foundation_v2.py",
    "eval_v2.py",
    "evidence_field_v2.py",
    "foundation_adapter_v2.py",
    "iris_apparatus_v2.py",
    "local_refinement_v2.py",
    "model_v2.py",
    "observation_contract_v2.py",
    "observation_evidence_emitter_v2.py",
    "q_descriptor_sampler_v2.py",
    "q_domain_v2.py",
    "q_evidence_encoder_v2.py",
    "q_spatial_graph_v2.py",
    "ray_modes_v2.py",
    "train_v2.py",
    "world_regularizer_v2.py",
)

MAINLINE_PYTHON_ROOTS = (
    ROOT / "models",
    ROOT / "compiler" / "realsas_compiler_core",
    ROOT / "compiler" / "realsas_compiler_services",
    ROOT / "runtime" / "reference_v4",
)


def _experiment_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "experiments" or alias.name.startswith("experiments."):
                    bad.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "experiments" or module.startswith("experiments."):
                bad.append(module)
    return bad


class IrisMainlineSourceV1(unittest.TestCase):
    def test_promoted_iris_v2_is_byte_identical_to_audited_source(self) -> None:
        for name in PROMOTED:
            source = SOURCE / name
            target = TARGET / name
            self.assertTrue(source.is_file(), source)
            self.assertTrue(target.is_file(), target)
            self.assertEqual(source.read_bytes(), target.read_bytes(), name)

    def test_compiler_owned_persistence_adapter_not_promoted_as_model(self) -> None:
        self.assertTrue((SOURCE / "persistence_adapter_v2.py").is_file())
        self.assertFalse((TARGET / "persistence_adapter_v2.py").exists())

    def test_gate_harness_not_promoted_as_model(self) -> None:
        for name in (
            "gate0_geometry_v1.py",
            "gate0_real_corpus_v1.py",
            "gate0_real_corpus_v1_1_hashfix.py",
            "run_gate0_synthetic_preflight_v1.py",
        ):
            self.assertTrue((SOURCE / name).is_file())
            self.assertFalse((TARGET / name).exists())

    def test_current_iris_model_symbol_is_present(self) -> None:
        source = (TARGET / "model_v2.py").read_text(encoding="utf-8")
        self.assertIn("class IrisReprojectionV2(nn.Module):", source)
        self.assertIn("The only learned geometric output is forward depth/support/uncertainty", source)

    def test_mainline_python_does_not_import_dated_experiments(self) -> None:
        violations: list[str] = []
        for root in MAINLINE_PYTHON_ROOTS:
            if not root.exists():
                continue
            for path in root.rglob("*.py"):
                bad = _experiment_imports(path)
                if bad:
                    violations.append(f"{path.relative_to(ROOT)} -> {bad}")
        self.assertEqual(violations, [], "\n".join(violations))


if __name__ == "__main__":
    unittest.main()
