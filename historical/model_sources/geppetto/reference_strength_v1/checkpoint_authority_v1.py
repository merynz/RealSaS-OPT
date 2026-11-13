from __future__ import annotations

"""Hash authority for the sealed Mage FIT1 Geppetto checkpoint/result."""

FROZEN_SOURCE_COMMIT = "f7be46f0a97df62a793ebf91b22297c894854f39"
SEAL_COMMIT = "ae0af0cd39dd2468a012ba21890a4fed2da7c4c9"
TARGET_CONTENT_SHA256 = "0b5a25c877116de60b710b7bb2a7848f30988e1622e2eda8084cad21c8ca23c9"
QUALIFIED_SKELETON_IR_SHA256 = "48754ad703c596ec9d332c6f733f1dd31e74d016ef15f3ce451263a724493992"
FINAL_CHECKPOINT_SHA256 = "b75f991564b64cfcec9b50b006544380ee482362a8439775bb505002349cbc30"
FINAL_RESULT_JSON_SHA256 = "728f5b5fe9e98865dd38c907ef19a741c57606f0e15557a40095d81144dc2045"
SIGNED_ZERO_SURFACE_SHA256 = "987f7d18ce202454c4ea5101225bfaed54aeb4638cba1077e70efc15f2038e9b"
UPSTREAM_IRIS_CHECKPOINT_SHA256 = "766f43cefd98925ada804853bafff93bb2352e23ba4a4e77e38174ae9e6b83a2"
CLOSURE_STEP = 14080
TERMINAL_STREAK_CHECKS = 48
TERMINAL_STABILITY_OPTIMIZER_STEPS = 3072
QUALIFIED_CONTROL_COUNT = 22
DIFFUSION_EVAL_SEEDS = (11, 23, 47, 89)
GENERALIZATION_CLAIM = False


def sealed_geppetto_fit1_authority() -> dict[str, object]:
    return {
        "frozen_source_commit": FROZEN_SOURCE_COMMIT,
        "seal_commit": SEAL_COMMIT,
        "target_content_sha256": TARGET_CONTENT_SHA256,
        "qualified_skeleton_ir_sha256": QUALIFIED_SKELETON_IR_SHA256,
        "final_checkpoint_sha256": FINAL_CHECKPOINT_SHA256,
        "final_result_json_sha256": FINAL_RESULT_JSON_SHA256,
        "signed_zero_surface_sha256": SIGNED_ZERO_SURFACE_SHA256,
        "upstream_iris_checkpoint_sha256": UPSTREAM_IRIS_CHECKPOINT_SHA256,
        "closure_step": CLOSURE_STEP,
        "terminal_streak_checks": TERMINAL_STREAK_CHECKS,
        "terminal_stability_optimizer_steps": TERMINAL_STABILITY_OPTIMIZER_STEPS,
        "qualified_control_count": QUALIFIED_CONTROL_COUNT,
        "diffusion_eval_seeds": DIFFUSION_EVAL_SEEDS,
        "generalization_claim": GENERALIZATION_CLAIM,
    }
