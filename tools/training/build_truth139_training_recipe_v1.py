#!/usr/bin/env python3
"""Build the split-bound Truth139 training recipe without opening unseen subject bytes.

The scientific comparison preserves the historical 40^3 x 160 dense carrier and preserves the
historical per-subject exposure horizon: 1000 geometry-foundation exposures followed by 5000
protected evidence-adapter exposures.  One global optimizer update still represents the equal
mean over the entire TRAIN cohort; physical execution remains one subject at a time, so the larger
cohort changes subject diversity rather than Adam-update semantics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


SPLIT_SCHEMA = "RealSaS.LineageSafeSubjectSplit.v1"
RECIPE_SCHEMA = "RealSaS.IRISV31Truth139TrainingRecipe.v1"


def build_recipe(split: dict) -> dict:
    if split.get("schema") != SPLIT_SCHEMA:
        raise ValueError("TRUTH139_RECIPE_SPLIT_SCHEMA")
    train_subjects = list(map(str, split.get("train_subjects") or ()))
    unseen_subjects = list(map(str, split.get("unseen_subjects") or ()))
    if not train_subjects or not unseen_subjects or set(train_subjects) & set(unseen_subjects):
        raise ValueError("TRUTH139_RECIPE_SPLIT_INVALID")
    split_identity = str(split.get("split_identity_sha256") or "")
    if len(split_identity) != 64:
        raise ValueError("TRUTH139_RECIPE_SPLIT_IDENTITY")
    train_count = len(train_subjects)

    return {
        "schema": RECIPE_SCHEMA,
        "date": "2026-10-29",
        "status": "PREREGISTERED_FROM_SEALED_COMPONENT_SPLIT__TRAINING_NOT_STARTED",
        "split_identity_sha256": split_identity,
        "train_subject_count": train_count,
        "unseen_subject_count_commitment_only": len(unseen_subjects),
        "selection_rule": "FIXED_FINAL_STAGE_ENDPOINT__NO_CHECKPOINT_SHOPPING",
        "architecture_id": "RealSaS.IRIS.DenseObservationEvidence.v3_1.truth139_v1",
        "model_config": {
            "carrier_resolution": 40,
            "carrier_channels": 160,
            "mechanical_channels": [
                "nonrigid_response_mean",
                "nonrigid_response_log_scale",
                "response_energy_mean",
                "response_energy_log_scale",
            ],
            "pair_channels": ["occlusion_attenuation_logit"],
            "query_width": 896,
            "query_layers": 3,
            "fourier_bands": 4,
            "private_bottleneck": 320,
            "evidence_adapter_bottleneck": 320,
            "lift_chunk": 16384,
            "delta_bound": 0.03,
        },
        "geometry": {
            "primary_authority": "DIRECT_FSTAR",
            "near_zero_band": 0.03,
            "sign_weight": 1.0,
            "normal_weight": 0.01,
            "normal_weight_rationale": (
                "COSINE_NORMAL_LOSS_IS_ORDER_ONE_WHILE_DIRECT_FSTAR_AND_WRONG_SIGN_DEPTH_ARE_"
                "ORDER_1E_MINUS_2__0P01_PREVENTS_NORMAL_AUXILIARY_FROM_REPLACING_FSTAR_AUTHORITY"
            ),
            "normal_target": "VISIBLE_SOURCE_SURFACE_WORLD_NORMAL__TEACHER_ONLY",
            "normal_forward_input": False,
            "depth_forward_input": False,
            "semantic_part_taxonomy_used": False,
        },
        "mechanics": {
            "target_transform": "LOG1P_NONNEGATIVE_MAGNITUDES",
            "objective": "FIXED_SCALE_STUDENT_T_LOCATION_ONLY",
            "degrees_of_freedom": 3.0,
            "minimum_fixed_scale": 0.001,
            "fixed_scale_estimator": "TRAIN_ONLY_CHANNELWISE_1P4826_MAD_WITH_FLOOR",
            "learned_per_query_variance": False,
            "historical_log_scale_head_channels_used_by_loss": False,
            "shortcut_courts_required": [
                "TRAIN_MARGINAL_PRIOR_BASELINE",
                "DETERMINISTIC_SHUFFLED_SCENE_BASELINE",
            ],
        },
        "appearance": {
            "objective": "UNCHANGED_SOURCE_EQUIVALENT_RGBA",
            "architecture_change": "NONE_BEYOND_EXISTING_PRIVATE_GP_ADAPTER",
        },
        "schedule": [
            {
                "id": "GEOMETRY_FOUNDATION",
                "epochs": 1000,
                "subject_exposures_per_subject": 1000,
                "trainable_group": "FOUNDATION_ONLY",
                "families": ["DIRECT_FSTAR", "NEAR_ZERO_SIGN", "VISIBLE_SURFACE_NORMAL"],
                "optimizer": {
                    "kind": "ADAMW",
                    "learning_rate": 0.00008,
                    "minimum_learning_rate": 0.000004,
                    "weight_decay": 0.0,
                    "gradient_clip_norm": 1.0,
                },
                "loss_weights": {"mechanics": 0.0, "appearance": 0.0},
            },
            {
                "id": "EVIDENCE_ADAPTER_ACQUISITION",
                "epochs": 5000,
                "subject_exposures_per_subject": 5000,
                "trainable_group": "EVIDENCE_ADAPTERS_ONLY",
                "families": ["FIXED_SCALE_MECHANICS_LOCATION", "SOURCE_EQUIVALENT_RGBA"],
                "optimizer": {
                    "kind": "ADAMW",
                    "learning_rate": 0.00004,
                    "minimum_learning_rate": 0.000002,
                    "weight_decay": 0.0,
                    "gradient_clip_norm": 1.0,
                },
                "loss_weights": {"mechanics": 1.0, "appearance": 1.0},
            },
        ],
        "optimizer": {
            "subjects_per_optimizer_step": train_count,
            "logical_update_semantics": "MEAN_GRADIENT_OVER_ALL_TRAIN_SUBJECTS_PER_GLOBAL_STEP",
            "physical_microbatch_semantics": "ONE_SUBJECT_AT_A_TIME__ACCUMULATE_THEN_SINGLE_STEP",
            "cohort_size_changes_adam_update_count": False,
            "per_subject_exposure_horizon_matches_7_SUBJECT_BASELINE": True,
        },
        "optimizer_update_semantics": "MEAN_GRADIENT_OVER_ALL_TRAIN_SUBJECTS_PER_GLOBAL_STEP",
        "sampling": {
            "geometry_queries_per_subject_exposure": 8192,
            "geometry_mix": {"replay": 0.5, "near": 0.25, "straddle": 0.25},
            "normal_queries_per_subject_exposure": 128,
            "mechanical_queries_per_subject_exposure": 1024,
            "appearance_queries_per_subject_exposure": 128,
            "presentation_train_fraction": 0.8,
            "seed": 261029139,
            "field_huber_beta": 0.01,
            "field_clamp": 0.25,
        },
        "runtime": {
            "float32_matmul_precision": "highest",
            "tf32": False,
            "cudnn_deterministic": True,
            "bf16_autocast": True,
        },
        "checkpoint": {
            "interval_epochs": 25,
            "durable_resume_required": True,
            "stage_transition_uses_fresh_optimizer": True,
            "foundation_bytes_sealed_before_evidence_optimizer_exists": True,
        },
        "firewall": {
            "train_manifest_may_contain_unseen_ids": False,
            "train_manifest_may_contain_unseen_file_refs": False,
            "unseen_used_for_gradient": False,
            "unseen_used_for_checkpoint_selection": False,
            "unseen_campaign_starts_only_after_final_train_checkpoint_sealed": True,
        },
        "claims": {
            "training_started": False,
            "unseen_opened": False,
            "truth139_pass": False,
            "generalization": False,
            "product": False,
        },
    }


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(path.suffix + ".partial")
    staged.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    staged.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    split = json.loads(Path(args.split).resolve().read_text())
    recipe = build_recipe(split)
    output = Path(args.output).resolve()
    _write(output, recipe)
    print(json.dumps({
        "output": str(output),
        "train_subject_count": recipe["train_subject_count"],
        "foundation_epochs": recipe["schedule"][0]["epochs"],
        "evidence_epochs": recipe["schedule"][1]["epochs"],
        "subjects_per_optimizer_step": recipe["optimizer"]["subjects_per_optimizer_step"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
