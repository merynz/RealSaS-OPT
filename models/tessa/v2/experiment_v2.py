from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .config_v2 import TESSAStructuralizerConfigV2
from .global_attention_v2 import TESSAGlobalHistoryPolicyV2
from .objective_v2 import TESSAObjectiveWeightsV2
from .training_corpus_v2 import TESSATrainingWindowPolicyV2


PREREG_SCHEMA = "RealSaS.TESSAV2Fit8Preregistration.v1"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_preregistration_v1(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    value = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("TESSA_V2_PREREG_JSON_OBJECT_REQUIRED")
    validate_preregistration_v1(value)
    return value


def validate_preregistration_v1(value: dict[str, Any]) -> None:
    if value.get("schema") != PREREG_SCHEMA:
        raise ValueError("TESSA_V2_PREREG_SCHEMA_INVALID")
    if value.get("scope") != "FIT8_SHARED_ARCHITECTURE_FIT__NOT_UNSEEN_GENERALIZATION__NOT_PRODUCT_PROMOTION":
        raise ValueError("TESSA_V2_PREREG_SCOPE_DRIFT")

    cfg = TESSAStructuralizerConfigV2(); cfg.validate()
    arch = value.get("architecture") or {}
    exact_arch = {
        "architecture_id": "RealSaS.TESSA.Structuralizer.GlobalHistory.v2",
        "d_model": cfg.d_model,
        "n_heads": cfg.n_heads,
        "decoder_layers": cfg.decoder_layers,
        "feedforward_dim": cfg.feedforward_dim,
        "coordinate_bins": cfg.coordinate_bins,
        "scene_memory_tokens": cfg.scene_memory_tokens,
        "surface_memory_tokens": cfg.surface_memory_tokens,
        "total_conditioning_tokens": cfg.total_conditioning_tokens,
        "local_attention_window": cfg.local_attention_window,
        "global_history_block_size": TESSAGlobalHistoryPolicyV2().block_size,
        "global_history_max_summary_tokens": TESSAGlobalHistoryPolicyV2().max_summary_tokens,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "dropout": cfg.dropout,
        "reduced_capacity_fallback_allowed": False,
    }
    if arch != exact_arch:
        raise ValueError("TESSA_V2_PREREG_ARCHITECTURE_DRIFT")

    corpus = value.get("training_corpus") or {}
    window = TESSATrainingWindowPolicyV2(); window.validate(cfg)
    if (
        int(corpus.get("witness_count_per_subject", -1)) != 8192
        or int(corpus.get("context_tokens", -1)) != window.context_tokens
        or int(corpus.get("score_tokens", -1)) != window.score_tokens
        or int(corpus.get("stride_tokens", -1)) != window.stride_tokens
        or corpus.get("global_sequence_reconstructed_from_sealed_v4_windows") is not True
        or corpus.get("overlap_equality_required") is not True
    ):
        raise ValueError("TESSA_V2_PREREG_CORPUS_DRIFT")

    weights = TESSAObjectiveWeightsV2()
    objective = value.get("objective") or {}
    if (
        float(objective.get("token_cross_entropy_weight", -1)) != weights.token_cross_entropy
        or float(objective.get("coordinate_metric_weight", -1)) != weights.coordinate_metric
        or float(objective.get("structural_special_token_weight", -1)) != weights.structural_special_token
        or objective.get("teacher_forced_loss_is_scientific_pass") is not False
    ):
        raise ValueError("TESSA_V2_PREREG_OBJECTIVE_DRIFT")

    optimizer = value.get("optimizer") or {}
    exact_optimizer = {
        "name": "AdamW",
        "learning_rate": 1.0e-4,
        "betas": [0.9, 0.95],
        "epsilon": 1.0e-8,
        "weight_decay": 0.01,
        "gradient_clip_norm": 1.0,
        "precision": "BF16_AUTOCAST__FP32_PARAMETERS_AND_OPTIMIZER_STATE",
        "optimizer_steps": 2400,
        "warmup_steps": 120,
        "minimum_learning_rate": 1.0e-5,
        "schedule": "LINEAR_WARMUP_THEN_COSINE",
        "checkpoint_interval_steps": 200,
        "telemetry_interval_steps": 10,
        "seed": 26091991,
        "subject_schedule": "ROUND_ROBIN_8_SUBJECTS__300_OPTIMIZER_STEPS_EACH",
        "window_schedule": "DETERMINISTIC_CYCLIC_PER_SUBJECT",
        "gradient_accumulation_steps": 1,
    }
    if optimizer != exact_optimizer:
        raise ValueError("TESSA_V2_PREREG_OPTIMIZER_DRIFT")

    runtime = value.get("runtime_admission") or {}
    if (
        runtime.get("accelerator") != "NVIDIA_A100_CLASS"
        or int(runtime.get("minimum_cuda_memory_gib", -1)) != 39
        or runtime.get("full_t512_t1024_direct_surface_required") is not True
        or runtime.get("low_resolution_surface_fallback_allowed") is not False
        or runtime.get("full_reference_capacity_tessa_forward_backward_preflight_required") is not True
        or runtime.get("preflight_optimizer_step_allowed") is not False
    ):
        raise ValueError("TESSA_V2_PREREG_RUNTIME_DRIFT")

    free = value.get("free_running") or {}
    if (
        free.get("sampling") is not False
        or int(free.get("maximum_new_tokens", -1)) != 131072
        or free.get("teacher_prefix_allowed") is not False
        or free.get("grammar_mask_required") is not True
        or free.get("eos_required") is not True
        or free.get("incremental_decoder_required") is not True
        or free.get("incremental_full_prefix_logit_parity_required") is not True
        or free.get("final_step_checkpoint_evaluated") is not True
    ):
        raise ValueError("TESSA_V2_PREREG_FREE_RUNNING_DRIFT")

    interpretation = value.get("interpretation") or {}
    if (
        interpretation.get("tessa_mandatory_authorized_by_this_experiment") is not False
        or interpretation.get("canonical_m_authorized_by_this_experiment") is not False
        or interpretation.get("unseen_generalization_claimed") is not False
    ):
        raise ValueError("TESSA_V2_PREREG_AUTHORITY_ESCALATION")


def learning_rate_at_step_v1(step: int, prereg: dict[str, Any]) -> float:
    """LR for the optimizer step numbered from one through the locked horizon."""
    opt = prereg["optimizer"]
    total = int(opt["optimizer_steps"])
    warmup = int(opt["warmup_steps"])
    peak = float(opt["learning_rate"])
    floor = float(opt["minimum_learning_rate"])
    if step <= 0 or step > total:
        raise ValueError("TESSA_V2_LR_STEP_OUT_OF_RANGE")
    if step <= warmup:
        return peak * float(step) / float(warmup)
    progress = float(step - warmup) / float(max(1, total - warmup))
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * progress))
