from __future__ import annotations

"""Build the immutable artifact-bound preregistration for focused IRIS V9.

Policy choices are fixed in source before optimizer step 1.  The focused schedule
inherits the historical V8 R1 evidence-head scale (1e-4 LR, 0.01 weight decay,
1600 updates) while keeping scene_core sealed.  Held-out evidence is evaluated
only at step zero and at the fixed schedule end, so it cannot become an adaptive
checkpoint-selection signal.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from models.iris.vnext.cohort_v9 import V9_HOLDOUT_FRACTION, V9_SPLIT_POLICY, V9_SPLIT_SEED
from models.iris.vnext.cohort_v9_balanced_fp32 import V9_SUPERVISION_POLICY


SCHEMA = "RealSaS.IRISVNextFocusedRelationPreregistration.v1"
RELATION_LR = 1.0e-4
WEIGHT_DECAY = 0.01
BETAS = [0.9, 0.999]
MAXIMUM_STEPS = 1600
EVALUATION_INTERVAL = 1600
CHECKPOINT_INTERVAL = 100
BATCH_PAIR_GROUPS_PER_SUBJECT = 512


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("VNEXT_V9_PREREG_JSON_OBJECT_REQUIRED")
    return value


def _sha40(value: str) -> str:
    out = str(value).strip().lower()
    if len(out) != 40 or any(c not in "0123456789abcdef" for c in out):
        raise ValueError("VNEXT_V9_PREREG_SOURCE_SHA_INVALID")
    return out


def build_preregistration(
    *,
    expected_source_sha: str,
    materialization_receipt_path: Path,
    relation_manifest_path: Path,
    boundary_manifest_path: Path,
    migration_receipt_path: Path,
    support_receipt_path: Path,
) -> dict[str, Any]:
    source = _sha40(expected_source_sha)
    materialization = _load(materialization_receipt_path)
    relation = _load(relation_manifest_path)
    boundary = _load(boundary_manifest_path)
    migration = _load(migration_receipt_path)
    support = _load(support_receipt_path)

    relation_sha = _sha(relation_manifest_path)
    boundary_sha = _sha(boundary_manifest_path)
    migration_sha = _sha(migration_receipt_path)
    materialization_sha = _sha(materialization_receipt_path)
    support_sha = _sha(support_receipt_path)

    if materialization.get("status") != "MATERIALIZED_AND_BYTE_VERIFIED__TRAINING_NOT_STARTED":
        raise ValueError("VNEXT_V9_PREREG_MATERIALIZATION_NOT_READY")
    if materialization.get("source_commit_sha") != source:
        raise ValueError("VNEXT_V9_PREREG_MATERIALIZATION_SOURCE_DRIFT")
    if materialization.get("candidate_roundtrip_verified") is not True:
        raise ValueError("VNEXT_V9_PREREG_ROUNDTRIP_REQUIRED")
    if materialization.get("relation_truth_v2_manifest_sha256") != relation_sha:
        raise ValueError("VNEXT_V9_PREREG_RELATION_BINDING_DRIFT")
    if materialization.get("boundary_causality_v1_manifest_sha256") != boundary_sha:
        raise ValueError("VNEXT_V9_PREREG_BOUNDARY_BINDING_DRIFT")
    if relation.get("schema") != "RealSaS.IRISVNextFit8RelationTruthManifest.v2":
        raise ValueError("VNEXT_V9_PREREG_RELATION_SCHEMA_INVALID")
    if boundary.get("schema") != "RealSaS.IRISVNextFit8BoundaryCausalityManifest.v1":
        raise ValueError("VNEXT_V9_PREREG_BOUNDARY_SCHEMA_INVALID")
    if boundary.get("relation_truth_v2_manifest_sha256") != relation_sha:
        raise ValueError("VNEXT_V9_PREREG_BOUNDARY_RELATION_DRIFT")
    if migration.get("schema") != "RealSaS.IRISVNextV8RelationHeadToV2ArtifactReceipt.v1" or migration.get("status") != "PASS":
        raise ValueError("VNEXT_V9_PREREG_MIGRATION_NOT_PASS")
    if int(migration.get("training_steps_performed", -1)) != 0:
        raise ValueError("VNEXT_V9_PREREG_MIGRATION_ALREADY_TRAINED")
    if support.get("schema") != "RealSaS.IRISVNextV9RelationSplitSupport.v1":
        raise ValueError("VNEXT_V9_PREREG_SUPPORT_SCHEMA_INVALID")
    if support.get("status") != "PASS__SPLIT_SUPPORT_QUALIFIED__NO_TRAINING" or support.get("blockers") not in ([], None):
        raise ValueError("VNEXT_V9_PREREG_SUPPORT_NOT_PASS")
    if support.get("source_commit_sha") != source:
        raise ValueError("VNEXT_V9_PREREG_SUPPORT_SOURCE_DRIFT")
    if support.get("relation_truth_v2_manifest_sha256") != relation_sha:
        raise ValueError("VNEXT_V9_PREREG_SUPPORT_RELATION_DRIFT")
    if support.get("supervision_policy") != V9_SUPERVISION_POLICY:
        raise ValueError("VNEXT_V9_PREREG_SUPPORT_POLICY_DRIFT")

    return {
        "schema": SCHEMA,
        "status": "LOCKED__ARTIFACT_BOUND__TRAINING_NOT_STARTED",
        "purpose": "Focused post-V8 camera-conditioned relation/boundary specialist optimization with scene_core sealed",
        "exact_artifact_bindings": {
            "source_commit_sha": source,
            "materialization_receipt_sha256": materialization_sha,
            "migration_receipt_sha256": migration_sha,
            "relation_truth_v2_manifest_sha256": relation_sha,
            "boundary_causality_v1_manifest_sha256": boundary_sha,
            "relation_split_support_receipt_sha256": support_sha,
        },
        "supervision_policy": V9_SUPERVISION_POLICY,
        "held_out": {
            "split_policy": V9_SPLIT_POLICY,
            "split_seed": V9_SPLIT_SEED,
            "holdout_fraction": V9_HOLDOUT_FRACTION,
            "no_training_example_leakage": True,
            "selection_policy": "FIXED_SCHEDULE__STEP0_BASELINE_AND_FINAL_HELDOUT_ONLY",
            "subject_view_cells": "DIAGNOSTIC_ONLY__COURTS_USE_GLOBAL_AND_PER_VIEW_AGGREGATES",
        },
        "optimizer": {
            "relation_head_learning_rate": RELATION_LR,
            "scene_core_learning_rate": 0.0,
            "weight_decay": WEIGHT_DECAY,
            "betas": BETAS,
            "trainable_scope": "RELATION_VIEW_AND_VIEW_PAIR_ONLY",
        },
        "schedule": {
            "maximum_optimizer_steps": MAXIMUM_STEPS,
            "evaluation_interval": EVALUATION_INTERVAL,
            "checkpoint_interval": CHECKPOINT_INTERVAL,
            "batch_pair_groups_per_subject": BATCH_PAIR_GROUPS_PER_SUBJECT,
            "stage_extension_policy": "NO_EXTENSION_WITHOUT_SEPARATE_PREREGISTERED_AMENDMENT",
        },
        "held_out_gates": [
            {"family": "projected_alpha_boundary_probability", "metric": "balanced_accuracy", "scope": "GLOBAL", "direction": "MIN", "threshold": 0.65},
            {"family": "projected_alpha_boundary_probability", "metric": "balanced_bce_mean", "scope": "GLOBAL", "direction": "MAX", "threshold": 0.62},
            {"family": "projected_alpha_boundary_probability", "metric": "balanced_accuracy", "scope": "MIN_PER_VIEW_AGGREGATE", "direction": "MIN", "threshold": 0.55},
            {"family": "projected_appearance_boundary_probability", "metric": "balanced_accuracy", "scope": "GLOBAL", "direction": "MIN", "threshold": 0.65},
            {"family": "projected_appearance_boundary_probability", "metric": "balanced_bce_mean", "scope": "GLOBAL", "direction": "MAX", "threshold": 0.62},
            {"family": "projected_appearance_boundary_probability", "metric": "balanced_accuracy", "scope": "MIN_PER_VIEW_AGGREGATE", "direction": "MIN", "threshold": 0.55}
        ],
        "non_regression_gates": [
            {"family": "front_probability", "metric": "balanced_accuracy_delta_from_step0", "direction": "MIN", "threshold": -0.02},
            {"family": "visibility_probability_a", "metric": "balanced_accuracy_delta_from_step0", "direction": "MIN", "threshold": -0.02},
            {"family": "visibility_probability_b", "metric": "balanced_accuracy_delta_from_step0", "direction": "MIN", "threshold": -0.02},
            {"family": "front_probability", "metric": "balanced_bce_delta_from_step0", "direction": "MAX", "threshold": 0.03},
            {"family": "visibility_probability_a", "metric": "balanced_bce_delta_from_step0", "direction": "MAX", "threshold": 0.03},
            {"family": "visibility_probability_b", "metric": "balanced_bce_delta_from_step0", "direction": "MAX", "threshold": 0.03},
            {"family": "FROZEN_PARAMETER_DOMAINS", "metric": "all_hashes_exact", "direction": "MIN", "threshold": 1.0},
            {"family": "STATIC_RELATION_ABSTENTION", "metric": "valid_target_count", "direction": "MAX", "threshold": 0.0},
            {"family": "occlusion_probability", "metric": "binary_court_eligible", "direction": "MAX", "threshold": 0.0}
        ],
        "court_notes": {
            "directional_front_visibility": "BINARY_SUPPORT__PRESERVE_ZERO_STEP_AUTHORITY",
            "projected_boundaries": "BINARY_SUPPORT__CLASS_BALANCED_OPTIMIZATION__ABSOLUTE_AND_PER_VIEW_GATES",
            "occlusion_probability": "POSITIVE_ONLY__METRIC_ONLY__NO_GRADIENT__NO_BINARY_PASS_COURT",
            "static_same_cut_attachment": "ABSTAIN__NO_GRADIENT",
            "threshold_rule": "LOCKED_BEFORE_TRAINING__NO_POST_HOC_RELAXATION",
            "heldout_peeking": "FORBIDDEN_BETWEEN_STEP0_AND_FIXED_STEP1600"
        },
        "policy_lineage": {
            "historical_v8_relation_stage_steps": 1600,
            "historical_v8_evidence_learning_rate": 1.0e-4,
            "historical_v8_weight_decay": 0.01,
            "v9_change": "ONLY_VIEW_RELATION_BRANCH__BALANCED_GROUPED_SUPERVISION__SCENE_CORE_AND_STATIC_PATH_SEALED"
        },
        "scene_core_reopening_authorized": False,
        "thresholds_locked_before_training": True,
        "scientific_fit_started": False,
        "scientific_fit_pass_claimed": False
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--expected-source-sha", required=True)
    p.add_argument("--materialization-receipt", type=Path, required=True)
    p.add_argument("--relation-manifest", type=Path, required=True)
    p.add_argument("--boundary-manifest", type=Path, required=True)
    p.add_argument("--migration-receipt", type=Path, required=True)
    p.add_argument("--support-receipt", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    value = build_preregistration(
        expected_source_sha=args.expected_source_sha,
        materialization_receipt_path=args.materialization_receipt.resolve(),
        relation_manifest_path=args.relation_manifest.resolve(),
        boundary_manifest_path=args.boundary_manifest.resolve(),
        migration_receipt_path=args.migration_receipt.resolve(),
        support_receipt_path=args.support_receipt.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(
        "VNEXT_V9_PREREGISTRATION",
        f"status={value['status']}",
        f"maximum_steps={value['schedule']['maximum_optimizer_steps']}",
        "training_steps=0",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())