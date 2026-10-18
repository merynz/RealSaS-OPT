from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
STRICT = ROOT / "models/iris/vnext/cohort_v9_fp32.py"
BALANCED = ROOT / "models/iris/vnext/cohort_v9_balanced_fp32.py"
BASE = ROOT / "models/iris/vnext/cohort_v9.py"
CLI = ROOT / "tools/iris_vnext/run_v9_focused_relation.py"
SUPPORT = ROOT / "tools/iris_vnext/audit_v9_relation_split_support_v1.py"
PREREG = ROOT / "tools/iris_vnext/build_v9_focused_relation_preregistration_v1.py"
READINESS = ROOT / "tools/iris_vnext/audit_fit8_pretraining_readiness_v1.py"
COURT = ROOT / "tools/iris_vnext/audit_v9_scientific_court_v1.py"


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
        self.assertIn('p.add_argument("--support-receipt", required=True)', text)

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

    def test_balanced_authority_uses_disposable_parent_bootstrap_and_safe_resume(self) -> None:
        text = BALANCED.read_text(encoding="utf-8")
        self.assertIn(".balanced_bootstrap", text)
        self.assertIn("VNEXT_V9_BALANCED_BOOTSTRAP_STEP_DRIFT", text)
        self.assertIn("self._restore(previous)", text)
        self.assertIn("relation_split_support_receipt_sha256", text)
        self.assertIn("VNEXT_V9_SUPPORT_RECEIPT_TRAIN_COUNT_DRIFT", text)

    def test_split_support_court_reads_sidecars_and_requires_each_view_binary_support(self) -> None:
        text = SUPPORT.read_text(encoding="utf-8")
        self.assertIn("PASS__SPLIT_SUPPORT_QUALIFIED__NO_TRAINING", text)
        self.assertIn("RELATION_SIDECAR_SHA_DRIFT", text)
        self.assertIn("OPTIMIZED_FAMILY_VIEW_BINARY_SUPPORT_MISSING", text)
        self.assertIn("METRIC_ONLY_NEGATIVE_AUTHORITY_CHANGED__REVIEW_POLICY", text)
        self.assertIn("DIAGNOSTIC_ONLY__NO_PER_CELL_BINARY_PASS_REQUIREMENT", text)

    def test_prereg_is_fixed_schedule_and_never_peeks_mid_training(self) -> None:
        text = PREREG.read_text(encoding="utf-8")
        self.assertIn("RELATION_LR = 1.0e-4", text)
        self.assertIn("WEIGHT_DECAY = 0.01", text)
        self.assertIn("MAXIMUM_STEPS = 1600", text)
        self.assertIn("EVALUATION_INTERVAL = 1600", text)
        self.assertIn("CHECKPOINT_INTERVAL = 100", text)
        self.assertIn("BATCH_PAIR_GROUPS_PER_SUBJECT = 512", text)
        self.assertIn("FIXED_SCHEDULE__STEP0_BASELINE_AND_FINAL_HELDOUT_ONLY", text)
        self.assertIn('"thresholds_locked_before_training": True', text)
        self.assertIn("relation_split_support_receipt_sha256", text)
        self.assertIn("boundary_causality_v1_manifest_sha256", text)

    def test_readiness_requires_boundary_support_and_complete_trainer_policy(self) -> None:
        text = READINESS.read_text(encoding="utf-8")
        self.assertIn("--boundary-manifest", text)
        self.assertIn("--support-receipt", text)
        self.assertIn("projected_alpha_boundary_probability", text)
        self.assertIn("projected_appearance_boundary_probability", text)
        self.assertIn("PREREGISTRATION_BETAS_INVALID", text)
        self.assertIn("PREREGISTRATION_SCHEDULE_INCOMPLETE", text)
        self.assertIn("PREREGISTRATION_HELDOUT_PEEK_FORBIDDEN__EVAL_ONLY_AT_FINAL_STEP", text)

    def test_scientific_court_uses_locked_thresholds_and_zero_step_baseline(self) -> None:
        text = COURT.read_text(encoding="utf-8")
        self.assertIn("SCIENTIFIC_PASS", text)
        self.assertIn("balanced_accuracy_delta_from_step0", text)
        self.assertIn("balanced_bce_delta_from_step0", text)
        self.assertIn("MIN_PER_VIEW_AGGREGATE", text)
        self.assertIn("RUN_BINDING_DRIFT_FROM_ZERO_STEP", text)
        self.assertNotIn("threshold = 0.5", text)

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