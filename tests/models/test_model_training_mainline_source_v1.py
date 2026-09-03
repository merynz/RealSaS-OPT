from __future__ import annotations

from pathlib import Path
import ast
import unittest

ROOT = Path(__file__).resolve().parents[2]
R6 = ROOT / "experiments" / "geppetto_arachne_r6_20260901"


class ModelTrainingMainlineSourceV1(unittest.TestCase):
    def test_geppetto_v2_training_source_is_promoted(self) -> None:
        target = ROOT / "models" / "geppetto" / "v2"
        for name in (
            "training_targets_v2.py",
            "geppetto_loss_v2.py",
            "geppetto_train_v2.py",
            "geppetto_eval_v2.py",
        ):
            self.assertEqual((R6 / name).read_bytes(), (target / name).read_bytes(), name)
        compat = (target / "training_targets_v1.py").read_text(encoding="utf-8")
        self.assertIn("class GeppettoTeacherTargetV1", compat)
        self.assertNotIn("conditioning_v1", compat)
        self.assertNotIn("experiments.", compat)

    def test_codec_a0_training_and_eval_are_byte_identical(self) -> None:
        target = ROOT / "models" / "skin_field_codec" / "v1"
        for name in (
            "codec_deformation_loss_v1.py",
            "train_codec_r6_a0_v1.py",
            "eval_codec_r6_a0_v1.py",
        ):
            self.assertEqual((R6 / name).read_bytes(), (target / name).read_bytes(), name)

    def test_arachne_a1_training_eval_and_base_tail_are_byte_identical(self) -> None:
        target = ROOT / "models" / "arachne" / "v2"
        for name in (
            "arachne_tail_objective_v1.py",
            "train_arachne_r6_a1_v1.py",
            "eval_arachne_r6_a1_v1.py",
        ):
            self.assertEqual((R6 / name).read_bytes(), (target / name).read_bytes(), name)

    def test_arachne_training_bridges_point_to_codec_mainline(self) -> None:
        target = ROOT / "models" / "arachne" / "v2"
        expected = {
            "skin_field_codec_v1.py": "models.skin_field_codec.v1.skin_field_codec_v1",
            "codec_deformation_loss_v1.py": "models.skin_field_codec.v1.codec_deformation_loss_v1",
            "train_codec_r6_a0_v1.py": "models.skin_field_codec.v1.train_codec_r6_a0_v1",
        }
        for name, module in expected.items():
            source = (target / name).read_text(encoding="utf-8")
            self.assertIn(module, source, name)
            self.assertNotIn("experiments.", source, name)

    def test_conditional_tail_remediation_is_not_mainline_base_training(self) -> None:
        self.assertTrue((R6 / "arachne_tail_remediation_v1.py").is_file())
        self.assertFalse((ROOT / "models" / "arachne" / "v2" / "arachne_tail_remediation_v1.py").exists())

    def test_promoted_training_files_do_not_import_experiments(self) -> None:
        violations = []
        for root in (
            ROOT / "models" / "geppetto" / "v2",
            ROOT / "models" / "skin_field_codec" / "v1",
            ROOT / "models" / "arachne" / "v2",
        ):
            for path in root.glob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                for node in ast.walk(tree):
                    module = ""
                    if isinstance(node, ast.ImportFrom):
                        module = node.module or ""
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name == "experiments" or alias.name.startswith("experiments."):
                                violations.append(f"{path.relative_to(ROOT)}:{alias.name}")
                    if module == "experiments" or module.startswith("experiments."):
                        violations.append(f"{path.relative_to(ROOT)}:{module}")
        self.assertEqual(violations, [], "\n".join(violations))


if __name__ == "__main__":
    unittest.main()
