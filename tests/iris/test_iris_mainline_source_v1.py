from __future__ import annotations

import ast
import importlib
from pathlib import Path
import sys
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
    "resource_contract_v2.py",
    "train_v2.py",
    "world_regularizer_v2.py",
)

MAINLINE_PYTHON_ROOTS = (
    ROOT / "models",
    ROOT / "compiler" / "realsas_compiler_core",
    ROOT / "compiler" / "realsas_compiler_services",
    ROOT / "runtime" / "reference_v4",
)

# One compatibility seam is deliberately admitted so the scientifically frozen
# Geppetto candidate can remain byte-identical to the sealed FIT1 source. The
# package __init__ must bind this historical name to the byte-identical promoted
# models/ tensorization module before the candidate imports it. No other dated
# experiments.* import is admitted anywhere in the current mainline.
GEPPETTO_FROZEN_CANDIDATE = Path(
    "models/geppetto/reference_strength_v1/geppetto_reference_strength_candidate_v1.py"
)
GEPPETTO_HISTORICAL_TENSORIZATION = (
    "experiments.geppetto_reference_strength_fullstack_v1."
    "rigging_surface_tensorization_v1"
)
GEPPETTO_PROMOTED_TENSORIZATION = (
    "models.geppetto.reference_strength_v1.rigging_surface_tensorization_v1"
)
APPROVED_FROZEN_IMPORT_SEAMS = {
    (GEPPETTO_FROZEN_CANDIDATE.as_posix(), GEPPETTO_HISTORICAL_TENSORIZATION),
}


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

    def test_mainline_python_has_no_unapproved_dated_experiment_imports(self) -> None:
        violations: list[str] = []
        observed_approved: set[tuple[str, str]] = set()
        for root in MAINLINE_PYTHON_ROOTS:
            if not root.exists():
                continue
            for path in root.rglob("*.py"):
                rel = path.relative_to(ROOT).as_posix()
                for module in _experiment_imports(path):
                    seam = (rel, module)
                    if seam in APPROVED_FROZEN_IMPORT_SEAMS:
                        observed_approved.add(seam)
                    else:
                        violations.append(f"{rel} -> {module}")
        self.assertEqual(violations, [], "\n".join(violations))
        self.assertEqual(observed_approved, APPROVED_FROZEN_IMPORT_SEAMS)

    def test_geppetto_frozen_import_seam_resolves_to_promoted_module(self) -> None:
        package = importlib.import_module("models.geppetto.reference_strength_v1")
        del package  # import side effect is the contract under test
        promoted = importlib.import_module(GEPPETTO_PROMOTED_TENSORIZATION)
        historical = sys.modules.get(GEPPETTO_HISTORICAL_TENSORIZATION)
        self.assertIs(historical, promoted)
        self.assertTrue(Path(promoted.__file__).resolve().is_relative_to(ROOT / "models"))

        candidate = importlib.import_module(
            "models.geppetto.reference_strength_v1.geppetto_reference_strength_candidate_v1"
        )
        self.assertIs(candidate.RiggingSurfaceTensorV1, promoted.RiggingSurfaceTensorV1)


if __name__ == "__main__":
    unittest.main()
