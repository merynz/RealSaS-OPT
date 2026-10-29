#!/usr/bin/env python3
"""Build the fixed Truth139 TESSA training budget from sealed TRAIN topology coverage.

TESSA is a downstream topology consequence court, not the primary 7->Truth139 IRIS sample-
complexity comparison.  Its budget is therefore derived from the actual teacher sequence lengths:
one deterministic topology window per subject per epoch, at least four complete passes over the
largest subject's window set, with a 32-epoch floor.  This preserves equal subject exposure while
ensuring no complex source topology receives only a partial target pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


TEACHER_MANIFEST_SCHEMA = "RealSaS.Truth139TESSATeacherTrainManifest.v1"
CONDITIONING_INDEX_SCHEMA = "RealSaS.Truth139TESSAConditioningIndex.v1"
RECIPE_SCHEMA = "RealSaS.Truth139TESSATrainingRecipe.v1"


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_recipe(
    *,
    teacher_manifest: dict,
    conditioning_index: dict,
    target_tokens_per_window: int = 2048,
    coverage_cycles: int = 4,
    minimum_epochs: int = 32,
) -> dict:
    if teacher_manifest.get("schema") != TEACHER_MANIFEST_SCHEMA:
        raise ValueError("TRUTH139_TESSA_RECIPE_TEACHER_SCHEMA")
    if conditioning_index.get("schema") != CONDITIONING_INDEX_SCHEMA or conditioning_index.get("scope") != "TRAIN":
        raise ValueError("TRUTH139_TESSA_RECIPE_CONDITIONING_SCHEMA")
    split = str(teacher_manifest.get("split_identity_sha256") or "")
    if split != str(conditioning_index.get("split_identity_sha256") or ""):
        raise ValueError("TRUTH139_TESSA_RECIPE_SPLIT_DRIFT")
    if teacher_manifest.get("contains_unseen_subject_ids") is not False or teacher_manifest.get("contains_unseen_file_refs") is not False:
        raise ValueError("TRUTH139_TESSA_RECIPE_TEACHER_UNSEEN_FIREWALL")
    if conditioning_index.get("teacher_topology_used") is not False:
        raise ValueError("TRUTH139_TESSA_RECIPE_CONDITIONING_TEACHER_LEAKAGE")
    if min(int(target_tokens_per_window), int(coverage_cycles), int(minimum_epochs)) <= 0:
        raise ValueError("TRUTH139_TESSA_RECIPE_BUDGET_INVALID")

    teacher_rows = {str(row["subject_id"]): row for row in teacher_manifest.get("subjects") or ()}
    conditioning_rows = {str(row["subject_id"]): row for row in conditioning_index.get("subjects") or ()}
    if not teacher_rows or set(teacher_rows) != set(conditioning_rows):
        raise ValueError("TRUTH139_TESSA_RECIPE_SUBJECT_ROSTER_DRIFT")

    windows_by_subject = {}
    for subject, row in teacher_rows.items():
        ref = dict(row.get("teacher_receipt") or {})
        path = Path(str(ref.get("path") or "")).resolve()
        if not path.is_file() or _hash(path) != str(ref.get("sha256") or ""):
            raise ValueError("TRUTH139_TESSA_RECIPE_RECEIPT_DRIFT:" + subject)
        receipt = json.loads(path.read_text())
        if (
            receipt.get("schema") != "RealSaS.Truth139TESSATeacherMesh.v1"
            or receipt.get("subject_id") != subject
            or receipt.get("teacher_topology_target_only") is not True
            or receipt.get("student_conditioning_allowed") is not False
            or receipt.get("representability_court") != "PASS_FULL_SOURCE_TOPOLOGY_SEQUENCE"
        ):
            raise ValueError("TRUTH139_TESSA_RECIPE_RECEIPT_AUTHORITY:" + subject)
        token_count = int(receipt.get("token_count", 0))
        if token_count < 2:
            raise ValueError("TRUTH139_TESSA_RECIPE_TOKEN_COUNT:" + subject)
        windows_by_subject[subject] = int(math.ceil((token_count - 1) / int(target_tokens_per_window)))

    max_windows = max(windows_by_subject.values())
    epochs = max(int(minimum_epochs), int(coverage_cycles) * int(max_windows))
    train_count = len(teacher_rows)
    return {
        "schema": RECIPE_SCHEMA,
        "date": "2026-10-29",
        "status": "SEALED_TRAIN_TOPOLOGY_COVERAGE_BOUND__TRAINING_NOT_STARTED",
        "split_identity_sha256": split,
        "teacher_manifest_identity_sha256": teacher_manifest["manifest_identity_sha256"],
        "conditioning_index_identity_sha256": conditioning_index["index_identity_sha256"],
        "iris_checkpoint_sha256": conditioning_index["iris_checkpoint_sha256"],
        "iris_foundation_identity": conditioning_index["iris_foundation_identity"],
        "architecture_id": "RealSaS.TESSA.DualStreamStructuralizer.v3_1",
        "attention_semantics": "ROLLING_CAUSAL_SELF_INCLUSIVE_V31",
        "memory_semantics": "V2_DUAL_STREAM_TYPED_MEMORY",
        "presentation_stream_present": False,
        "scene_input_dim": 163,
        "surface_input_dim": 906,
        "model_config": {
            "d_model": 1024,
            "n_heads": 16,
            "decoder_layers": 24,
            "mlp_ratio": 4,
            "dropout": 0.0,
            "coordinate_bins": 1024,
            "max_faces": 32768,
            "max_vertices": 32768,
            "scene_memory_tokens": 512,
            "surface_memory_tokens": 512,
            "memory_source_width": 512,
            "memory_refinement_layers": 4,
            "local_attention_window": 4096,
            "query_chunk_size": 256,
            "gradient_checkpointing": True
        },
        "teacher_windows": {
            "target_tokens_per_window": int(target_tokens_per_window),
            "windows_by_subject": windows_by_subject,
            "maximum_windows_per_subject": int(max_windows),
            "selection_per_epoch": "WINDOW_INDEX_EQUALS_EPOCH_MOD_SUBJECT_WINDOW_COUNT",
            "each_subject_exposures_per_epoch": 1,
            "coverage_cycles_of_largest_subject": int(coverage_cycles),
            "minimum_epochs": int(minimum_epochs),
            "epochs": int(epochs),
            "subject_exposure_is_equal": True
        },
        "objective": {
            "token_cross_entropy": 1.0,
            "coordinate_metric": 0.5,
            "structural_special_token": 0.25
        },
        "optimizer": {
            "kind": "ADAMW",
            "learning_rate": 0.0001,
            "minimum_learning_rate": 0.00001,
            "weight_decay": 0.01,
            "gradient_clip_norm": 1.0,
            "subjects_per_optimizer_step": min(4, train_count),
            "physical_microbatch_subjects": 1,
            "scheduler": "COSINE_ANNEAL_OVER_FIXED_TOTAL_OPTIMIZER_STEPS"
        },
        "runtime": {
            "bf16_autocast": True,
            "float32_matmul_precision": "highest",
            "tf32": False,
            "cudnn_deterministic": True,
            "attention_backend": "SDPA"
        },
        "checkpoint": {
            "interval_epochs": 4,
            "selection_rule": "FIXED_FINAL_ENDPOINT__NO_CHECKPOINT_SHOPPING",
            "durable_resume_required": True
        },
        "firewall": {
            "iris_parameters_trainable": False,
            "conditioning_cache_contains_teacher_topology": False,
            "teacher_topology_is_target_only": True,
            "unseen_subjects_present": False,
            "unseen_used_for_gradient": False,
            "unseen_used_for_checkpoint_selection": False
        },
        "claims": {
            "training_started": False,
            "tessa_pass": False,
            "unseen": False,
            "compiler_m": False,
            "product": False
        }
    }


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(path.suffix + ".partial")
    staged.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    staged.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-manifest", required=True)
    parser.add_argument("--conditioning-index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-tokens-per-window", type=int, default=2048)
    parser.add_argument("--coverage-cycles", type=int, default=4)
    parser.add_argument("--minimum-epochs", type=int, default=32)
    args = parser.parse_args()
    teacher = json.loads(Path(args.teacher_manifest).resolve().read_text())
    conditioning = json.loads(Path(args.conditioning_index).resolve().read_text())
    recipe = build_recipe(
        teacher_manifest=teacher,
        conditioning_index=conditioning,
        target_tokens_per_window=args.target_tokens_per_window,
        coverage_cycles=args.coverage_cycles,
        minimum_epochs=args.minimum_epochs,
    )
    output = Path(args.output).resolve()
    _write(output, recipe)
    print(json.dumps({
        "output": str(output),
        "train_subject_count": len(recipe["teacher_windows"]["windows_by_subject"]),
        "epochs": recipe["teacher_windows"]["epochs"],
        "maximum_windows_per_subject": recipe["teacher_windows"]["maximum_windows_per_subject"],
        "architecture_id": recipe["architecture_id"]
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
