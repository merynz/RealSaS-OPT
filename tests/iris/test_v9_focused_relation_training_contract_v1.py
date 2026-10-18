from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
STRICT = ROOT / "models/iris/vnext/cohort_v9_fp32.py"
BALANCED = ROOT / "models/iris/vnext/cohort_v9_balanced_fp32.py"
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
    def test_numeric_base_is_fp32_deterministic_and_nonfused(self) -> None:
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

    def test_official_cli_imports_only_balanced_execution_authority(self) -> None:
        text = CLI.read_text(encoding="utf-8")
        self.assertIn("from models.iris.vnext.cohort_v9_balanced_fp32 import", text)
        self.assertIn("IRISVNextFocusedRelationTrainerV9BalancedFP32", text)
        self.assertNotIn("from models.iris.vnext.cohort_v9_fp32 import", text)
        self.assertNotIn("from models.iris.vnext.cohort_v9 import IRISVNextFocusedRelationTrainerV9", text)
        self.assertIn("AUTHORIZE_IRIS_V9_FOCUSED_RELATION_OPTIMIZER_STEPS", text)

    def test_balanced_policy_groups_independent_semantics_and_excludes_occlusion_gradient(self) -> None:
        text = BALANCED.read_text(encoding="utf-8")
        self.assertIn("TRAIN_SPLIT_PER_VIEW_CLASS_BALANCED_GROUPED_BCE", text)
        self.assertIn("EQUAL_THREE_GROUP_MEAN", text)
        self.assertIn("TRAIN_SPLIT_PER_VIEW_INVERSE_CLASS_FREQUENCY", text)
        self.assertIn('V9_METRIC_ONLY_FAMILIES = ("occlusion_probability",)', text)
        self.assertIn("V9_OPTIMIZED_FAMILIES = V9_ORDERING_FAMILIES + V9_BOUNDARY_FAMILIES", text)
        self.assertIn("VNEXT_V9_OPTIMIZED_FAMILY_VIEW_BINARY_SUPPORT_MISSING", text)
        self.assertIn("VNEXT_V9_OCCLUSION_POLICY_MUST_BE_REVIEWED_AFTER_NEGATIVE_AUTHORITY", text)
        official_loss = _function_source(BALANCED, "_official_loss")
        self.assertNotIn("occlusion_probability", official_loss)
        one_step = _function_source(BALANCED, "one_step")
        self.assertIn("self._official_loss", one_step)
        self.assertNotIn("graphics_relation_loss_v2", one_step)

    def test_balanced_authority_refuses_old_nonzero_checkpoint_policy(self) -> None:
        text = BALANCED.read_text(encoding="utf-8")
        self.assertIn("VNEXT_V9_BALANCED_POLICY_REQUIRES_FRESH_STEP_ZERO", text)
        self.assertIn("VNEXT_V9_BALANCED_POLICY_CHECKPOINT_ROOT_MUST_BE_FRESH", text)
        self.assertIn('"supervision_execution"', text)

    def test_preflight_cannot_execute_optimizer_step(self) -> None:
        source = _function_source(CLI, "preflight")
        self.assertNotIn("one_step(", source)
        self.assertIn('trainer.step != 0', source)
        self.assertIn('"training_steps_executed": 0', source)
        self.assertIn("STOP_FOR_RESEARCHER_REVIEW_BEFORE_OPTIMIZER_STEP_1", source)
        self.assertIn("V9_SUPERVISION_POLICY", source)

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