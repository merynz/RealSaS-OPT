from __future__ import annotations

"""Fail-closed pre-training readiness audit for focused post-V8 relation V9.

This tool does not train. It verifies exact materialized truth, Boundary
Causality V1, the deterministic TRAIN/HELD-OUT split-support court, the V8->V2
zero-step relation migration, and an artifact-bound preregistration before any
optimizer step may be considered.

The Python API accepts missing boundary/support paths only for compatibility with
older diagnostic callers; missing evidence is converted into explicit blockers.
The CLI requires both paths, and no missing-evidence call can authorize training.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from models.iris.vnext.cohort_v9 import V9_HOLDOUT_FRACTION, V9_SPLIT_POLICY, V9_SPLIT_SEED
from models.iris.vnext.cohort_v9_balanced_fp32 import V9_SUPERVISION_POLICY
from models.iris.vnext.relation_support_v2 import qualify_relation_manifest_support_v2

SCHEMA = "RealSaS.IRISVNextFocusedRelationPretrainingReadiness.v1"
PREREG_SCHEMA = "RealSaS.IRISVNextFocusedRelationPreregistration.v1"
SUPPORT_SCHEMA = "RealSaS.IRISVNextV9RelationSplitSupport.v1"
BOUNDARY_SCHEMA = "RealSaS.IRISVNextFit8BoundaryCausalityManifest.v1"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"VNEXT_PRETRAINING_READINESS_JSON_OBJECT_REQUIRED:{path}")
    return value


def _sha40(value: str) -> str:
    out = value.strip().lower()
    if len(out) != 40 or any(c not in "0123456789abcdef" for c in out):
        raise ValueError("VNEXT_PRETRAINING_READINESS_SOURCE_SHA_INVALID")
    return out


def _validate_preregistration(
    prereg: dict[str, Any],
    *,
    source_sha: str,
    materialization_sha: str,
    migration_sha: str,
    relation_manifest_sha: str,
    boundary_manifest_sha: str,
    support_receipt_sha: str,
) -> list[str]:
    blockers: list[str] = []
    if prereg.get("schema") != PREREG_SCHEMA:
        blockers.append("PREREGISTRATION_SCHEMA_INVALID")
        return blockers
    exact = prereg.get("exact_artifact_bindings") or {}
    expected = {
        "source_commit_sha": source_sha,
        "materialization_receipt_sha256": materialization_sha,
        "migration_receipt_sha256": migration_sha,
        "relation_truth_v2_manifest_sha256": relation_manifest_sha,
        "boundary_causality_v1_manifest_sha256": boundary_manifest_sha,
        "relation_split_support_receipt_sha256": support_receipt_sha,
    }
    for key, value in expected.items():
        if str(exact.get(key) or "").lower() != value.lower():
            blockers.append("PREREGISTRATION_ARTIFACT_BINDING_DRIFT:" + key)

    if prereg.get("supervision_policy") != V9_SUPERVISION_POLICY:
        blockers.append("PREREGISTRATION_SUPERVISION_POLICY_DRIFT")

    optimizer = prereg.get("optimizer") or {}
    if float(optimizer.get("scene_core_learning_rate", -1.0)) != 0.0:
        blockers.append("PREREGISTRATION_SCENE_CORE_MUST_BE_ZERO")
    try:
        relation_lr = float(optimizer["relation_head_learning_rate"])
        weight_decay = float(optimizer["weight_decay"])
        betas = optimizer["betas"]
    except (KeyError, TypeError, ValueError):
        blockers.append("PREREGISTRATION_OPTIMIZER_INCOMPLETE")
    else:
        if relation_lr <= 0.0:
            blockers.append("PREREGISTRATION_RELATION_LR_MUST_BE_POSITIVE")
        if weight_decay < 0.0:
            blockers.append("PREREGISTRATION_WEIGHT_DECAY_INVALID")
        if not isinstance(betas, list) or len(betas) != 2:
            blockers.append("PREREGISTRATION_BETAS_INVALID")
        else:
            try:
                beta_values = [float(x) for x in betas]
            except (TypeError, ValueError):
                blockers.append("PREREGISTRATION_BETAS_INVALID")
            else:
                if any(x < 0.0 or x >= 1.0 for x in beta_values):
                    blockers.append("PREREGISTRATION_BETAS_INVALID")

    schedule = prereg.get("schedule") or {}
    try:
        max_steps = int(schedule["maximum_optimizer_steps"])
        evaluation_interval = int(schedule["evaluation_interval"])
        checkpoint_interval = int(schedule["checkpoint_interval"])
        batch_pairs = int(schedule["batch_pair_groups_per_subject"])
    except (KeyError, TypeError, ValueError):
        blockers.append("PREREGISTRATION_SCHEDULE_INCOMPLETE")
    else:
        if min(max_steps, evaluation_interval, checkpoint_interval, batch_pairs) <= 0:
            blockers.append("PREREGISTRATION_SCHEDULE_NONPOSITIVE")
        if evaluation_interval > max_steps:
            blockers.append("PREREGISTRATION_EVAL_INTERVAL_EXCEEDS_SCHEDULE")
        if evaluation_interval != max_steps:
            blockers.append("PREREGISTRATION_HELDOUT_PEEK_FORBIDDEN__EVAL_ONLY_AT_FINAL_STEP")

    held_out = prereg.get("held_out") or {}
    if held_out.get("split_policy") != V9_SPLIT_POLICY:
        blockers.append("PREREGISTRATION_HELD_OUT_SPLIT_POLICY_DRIFT")
    if int(held_out.get("split_seed", -1)) != V9_SPLIT_SEED:
        blockers.append("PREREGISTRATION_HELD_OUT_SPLIT_SEED_DRIFT")
    try:
        fraction = float(held_out.get("holdout_fraction", -1.0))
    except (TypeError, ValueError):
        blockers.append("PREREGISTRATION_HELD_OUT_FRACTION_INVALID")
    else:
        if abs(fraction - V9_HOLDOUT_FRACTION) > 1.0e-12:
            blockers.append("PREREGISTRATION_HELD_OUT_FRACTION_DRIFT")
    if held_out.get("no_training_example_leakage") is not True:
        blockers.append("PREREGISTRATION_HELD_OUT_LEAKAGE_GUARD_REQUIRED")
    if held_out.get("selection_policy") != "FIXED_SCHEDULE__STEP0_BASELINE_AND_FINAL_HELDOUT_ONLY":
        blockers.append("PREREGISTRATION_HELD_OUT_SELECTION_POLICY_INVALID")

    gates = prereg.get("held_out_gates")
    if not isinstance(gates, list) or not gates:
        blockers.append("PREREGISTRATION_HELD_OUT_GATES_MISSING")
    else:
        for index, gate in enumerate(gates):
            if not isinstance(gate, dict):
                blockers.append(f"PREREGISTRATION_GATE_INVALID:{index}")
                continue
            for key in ("family", "metric", "direction", "threshold"):
                if key not in gate:
                    blockers.append(f"PREREGISTRATION_GATE_FIELD_MISSING:{index}:{key}")
            if gate.get("direction") not in {"MAX", "MIN"}:
                blockers.append(f"PREREGISTRATION_GATE_DIRECTION_INVALID:{index}")

    nonreg = prereg.get("non_regression_gates")
    if not isinstance(nonreg, list) or not nonreg:
        blockers.append("PREREGISTRATION_NON_REGRESSION_GATES_MISSING")
    if prereg.get("thresholds_locked_before_training") is not True:
        blockers.append("PREREGISTRATION_THRESHOLDS_NOT_LOCKED")
    if prereg.get("scientific_fit_started") is not False:
        blockers.append("PREREGISTRATION_ALREADY_CLAIMS_TRAINING_STARTED")
    return blockers


def audit_readiness(
    *,
    expected_source_sha: str,
    materialization_receipt_path: Path,
    relation_manifest_path: Path,
    migration_receipt_path: Path,
    preregistration_path: Path | None,
    boundary_manifest_path: Path | None = None,
    support_receipt_path: Path | None = None,
) -> dict[str, Any]:
    source = _sha40(expected_source_sha)
    materialization = _load(materialization_receipt_path)
    relation = _load(relation_manifest_path)
    migration = _load(migration_receipt_path)
    blockers: list[str] = []

    if materialization.get("schema") != "RealSaS.IRISVNextFit8PretrainingMaterializationReceipt.v1":
        blockers.append("MATERIALIZATION_SCHEMA_INVALID")
    if materialization.get("source_commit_sha") != source:
        blockers.append("MATERIALIZATION_SOURCE_DRIFT")
    if materialization.get("candidate_roundtrip_verified") is not True:
        blockers.append("CANDIDATE_ROUNDTRIP_NOT_VERIFIED")
    if materialization.get("truth_materialized") is not True:
        blockers.append("TRUTH_NOT_MATERIALIZED")
    if materialization.get("scientific_fit_started") is not False:
        blockers.append("MATERIALIZATION_ALREADY_CLAIMS_TRAINING_STARTED")

    relation_sha = _sha(relation_manifest_path)
    if relation.get("schema") != "RealSaS.IRISVNextFit8RelationTruthManifest.v2":
        blockers.append("RELATION_MANIFEST_SCHEMA_INVALID")
    if materialization.get("relation_truth_v2_manifest_sha256") != relation_sha:
        blockers.append("RELATION_MANIFEST_BINDING_DRIFT")

    boundary_sha: str | None = None
    boundary: dict[str, Any] = {}
    if boundary_manifest_path is None:
        blockers.append("BOUNDARY_MANIFEST_REQUIRED")
    else:
        boundary_manifest_path = boundary_manifest_path.resolve()
        boundary = _load(boundary_manifest_path)
        boundary_sha = _sha(boundary_manifest_path)
        if boundary.get("schema") != BOUNDARY_SCHEMA:
            blockers.append("BOUNDARY_MANIFEST_SCHEMA_INVALID")
        if materialization.get("boundary_causality_v1_manifest_sha256") != boundary_sha:
            blockers.append("BOUNDARY_MANIFEST_BINDING_DRIFT")
        if boundary.get("relation_truth_v2_manifest_sha256") != relation_sha:
            blockers.append("BOUNDARY_TO_RELATION_BINDING_DRIFT")
        if boundary.get("scientific_fit_started") is not False:
            blockers.append("BOUNDARY_ALREADY_CLAIMS_TRAINING_STARTED")

    support_report: dict[str, Any] = {}
    try:
        support = qualify_relation_manifest_support_v2(relation)
    except (TypeError, ValueError) as exc:
        blockers.append("RELATION_SUPPORT_AUDIT_INVALID:" + str(exc))
    else:
        for name, row in support.items():
            support_report[name] = {
                "status": row.status.value,
                "valid": row.valid,
                "positive": row.positive,
                "negative": row.negative,
                "binary_court_eligible": row.court_eligible,
            }
        for name in (
            "front_probability",
            "visibility_probability_a",
            "visibility_probability_b",
            "projected_alpha_boundary_probability",
            "projected_appearance_boundary_probability",
        ):
            if not support[name].court_eligible:
                blockers.append("OPTIMIZED_BINARY_SUPPORT_REQUIRED:" + name)
        for name in (
            "same_drawable_affinity",
            "cut_probability",
            "attachment_affinity",
        ):
            if support[name].valid != 0:
                blockers.append("STATIC_RELATION_SHOULD_ABSTAIN:" + name)
        if support["occlusion_probability"].court_eligible:
            blockers.append("OCCLUSION_BINARY_COURT_UNEXPECTEDLY_ELIGIBLE")

    support_sha: str | None = None
    support_receipt: dict[str, Any] = {}
    if support_receipt_path is None:
        blockers.append("SPLIT_SUPPORT_RECEIPT_REQUIRED")
    else:
        support_receipt_path = support_receipt_path.resolve()
        support_receipt = _load(support_receipt_path)
        support_sha = _sha(support_receipt_path)
        if support_receipt.get("schema") != SUPPORT_SCHEMA:
            blockers.append("SPLIT_SUPPORT_RECEIPT_SCHEMA_INVALID")
        if support_receipt.get("status") != "PASS__SPLIT_SUPPORT_QUALIFIED__NO_TRAINING":
            blockers.append("SPLIT_SUPPORT_RECEIPT_NOT_PASS")
        if support_receipt.get("source_commit_sha") != source:
            blockers.append("SPLIT_SUPPORT_SOURCE_DRIFT")
        if support_receipt.get("relation_truth_v2_manifest_sha256") != relation_sha:
            blockers.append("SPLIT_SUPPORT_RELATION_BINDING_DRIFT")
        if support_receipt.get("supervision_policy") != V9_SUPERVISION_POLICY:
            blockers.append("SPLIT_SUPPORT_SUPERVISION_POLICY_DRIFT")
        if support_receipt.get("blockers") not in ([], None):
            blockers.append("SPLIT_SUPPORT_HAS_BLOCKERS")
        split = support_receipt.get("split") or {}
        try:
            split_fraction = float(split.get("holdout_fraction", -1.0))
        except (TypeError, ValueError):
            blockers.append("SPLIT_SUPPORT_POLICY_DRIFT")
        else:
            if (
                split.get("policy") != V9_SPLIT_POLICY
                or int(split.get("seed", -1)) != V9_SPLIT_SEED
                or abs(split_fraction - V9_HOLDOUT_FRACTION) > 1.0e-12
                or split.get("orientation_grouped") is not True
            ):
                blockers.append("SPLIT_SUPPORT_POLICY_DRIFT")

    migration_sha = _sha(migration_receipt_path)
    if migration.get("schema") != "RealSaS.IRISVNextV8RelationHeadToV2ArtifactReceipt.v1":
        blockers.append("MIGRATION_SCHEMA_INVALID")
    if migration.get("status") != "PASS":
        blockers.append("MIGRATION_NOT_PASS")
    if int(migration.get("training_steps_performed", -1)) != 0:
        blockers.append("MIGRATION_PERFORMED_TRAINING")
    if migration.get("source_v8", {}).get("source_commit_sha") != "5083c9915533ec8ec6530c7d229c635887abfea7":
        blockers.append("MIGRATION_V8_SOURCE_DRIFT")
    if migration.get("target", {}).get("new_projected_boundaries_camera_conditioned") is not True:
        blockers.append("MIGRATION_PROJECTED_BOUNDARY_CONDITIONING_MISSING")

    materialization_sha = _sha(materialization_receipt_path)
    prereg_sha = None
    if preregistration_path is None:
        blockers.append("ARTIFACT_BOUND_PREREGISTRATION_REQUIRED")
    elif boundary_sha is None or support_sha is None:
        blockers.append("PREREGISTRATION_CANNOT_VALIDATE_WITHOUT_BOUNDARY_AND_SUPPORT")
    else:
        prereg = _load(preregistration_path)
        prereg_sha = _sha(preregistration_path)
        blockers.extend(
            _validate_preregistration(
                prereg,
                source_sha=source,
                materialization_sha=materialization_sha,
                migration_sha=migration_sha,
                relation_manifest_sha=relation_sha,
                boundary_manifest_sha=boundary_sha,
                support_receipt_sha=support_sha,
            )
        )

    status = (
        "READY_FOR_FOCUSED_TRAINING_ENTRY__NO_STEP_EXECUTED"
        if not blockers
        else "BLOCKED_BEFORE_TRAINING"
    )
    return {
        "schema": SCHEMA,
        "status": status,
        "source_commit_sha": source,
        "materialization_receipt_sha256": materialization_sha,
        "relation_truth_v2_manifest_sha256": relation_sha,
        "boundary_causality_v1_manifest_sha256": boundary_sha,
        "relation_split_support_receipt_sha256": support_sha,
        "migration_receipt_sha256": migration_sha,
        "preregistration_sha256": prereg_sha,
        "relation_support": support_report,
        "blockers": blockers,
        "scene_core_required_learning_rate": 0.0,
        "training_steps_executed": 0,
        "scientific_fit_started": False,
        "scientific_fit_pass_claimed": False,
        "training_entry_authorized": not blockers,
        "execution_note": "READINESS_ONLY__A_SEPARATE_EXPLICIT_TRAINING_COMMAND_IS_REQUIRED",
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--expected-source-sha", required=True)
    p.add_argument("--materialization-receipt", type=Path, required=True)
    p.add_argument("--relation-manifest", type=Path, required=True)
    p.add_argument("--boundary-manifest", type=Path, required=True)
    p.add_argument("--migration-receipt", type=Path, required=True)
    p.add_argument("--support-receipt", type=Path, required=True)
    p.add_argument("--preregistration", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--require-ready", action="store_true")
    args = p.parse_args()
    report = audit_readiness(
        expected_source_sha=args.expected_source_sha,
        materialization_receipt_path=args.materialization_receipt.resolve(),
        relation_manifest_path=args.relation_manifest.resolve(),
        boundary_manifest_path=args.boundary_manifest.resolve(),
        migration_receipt_path=args.migration_receipt.resolve(),
        support_receipt_path=args.support_receipt.resolve(),
        preregistration_path=(args.preregistration.resolve() if args.preregistration is not None else None),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(
        "VNEXT_FOCUSED_RELATION_PRETRAINING_READINESS",
        f"status={report['status']}",
        f"blockers={len(report['blockers'])}",
        "training_steps=0",
    )
    if args.require_ready and not report["training_entry_authorized"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())