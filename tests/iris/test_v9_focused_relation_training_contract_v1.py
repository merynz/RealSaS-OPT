from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
STRICT = ROOT / "models/iris/vnext/cohort_v9_fp32.py"
BASE = ROOT / "models/iris/vnext/cohort_v9.py"
CLI = ROOT / "tools/iris_vnext/run_v9_focused_relation.py"


def _function_source(path: Path, name: str) -> str:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    lines = text.splitlines()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    raise AssertionError(f"missing function {name} in {path}")


class V9FocusedRelationTrainingContractTests(unittest.TestCase):
    def test_official_numeric_authority_is_fp32_deterministic_and_nonfused(self) -> None:
        text = STRICT.read_text(encoding="utf-8")
        self.assertIn("FP32_TF32_OFF_DETERMINISTIC_CUDA_NONFUSED_ADAMW_V1", text)
        self.assertIn("torch.use_deterministic_algorithms(True)", text)
        self.assertIn("torch.set_float32_matmul_precision(\"highest\")", text)
        self.assertIn("torch.backends.cuda.matmul.allow_tf32 = False", text)
        self.assertIn("torch.backends.cudnn.allow_tf32 = False", text)
        self.assertIn("fused=False", text)
        self.assertIn("foreach=False", text)
        self.assertNotIn("torch.autocast", text)
        self.assertNotIn("bfloat16", text)

    def test_official_cli_imports_only_strict_execution_authority(self) -> None:
        text = CLI.read_text(encoding="utf-8")
        self.assertIn("from models.iris.vnext.cohort_v9_fp32 import", text)
        self.assertNotIn("from models.iris.vnext.cohort_v9 import IRISVNextFocusedRelationTrainerV9", text)
        self.assertIn("AUTHORIZE_IRIS_V9_FOCUSED_RELATION_OPTIMIZER_STEPS", text)

    def test_preflight_cannot_execute_optimizer_step(self) -> None:
        source = _function_source(CLI, "preflight")
        self.assertNotIn("one_step(", source)
        self.assertIn('trainer.step != 0', source)
        self.assertIn('"training_steps_executed": 0', source)
        self.assertIn("STOP_FOR_RESEARCHER_REVIEW_BEFORE_OPTIMIZER_STEP_1", source)

    def test_train_action_is_the_only_optimizer_step_entry(self) -> None:
        source = _function_source(CLI, "train")
        self.assertIn("args.authorize_optimizer_steps != TRAIN_AUTHORIZATION", source)
        self.assertIn("trainer.one_step()", source)
        self.assertIn("args.maximum_steps > trainer.policy.maximum_optimizer_steps", source)
        main_source = _function_source(CLI, "main")
        self.assertIn("VNEXT_V9_PREFLIGHT_MUST_NOT_ACCEPT_TRAINING_FLAGS", main_source)

    def test_base_contract_keeps_scene_and_static_relation_authority_frozen(self) -> None:
        text = BASE.read_text(encoding="utf-8")
        self.assertIn("RELATION_VIEW_AND_VIEW_PAIR_ONLY", text)
        self.assertIn("SEALED__NO_AUTHORITY_TO_REWRITE", text)
        self.assertIn("VNEXT_V9_STATIC_RELATION_MUST_ABSTAIN", text)
        self.assertIn("VNEXT_V9_PAIR_ORIENTATION_POINT_DRIFT", text)
        self.assertIn("orientation_grouped", text)


if __name__ == "__main__":
    unittest.main()
