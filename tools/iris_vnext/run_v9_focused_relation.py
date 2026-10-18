from __future__ import annotations

"""Fail-closed CLI for the official deterministic IRIS V9 relation experiment.

`preflight` is zero-step only.  `train` is a separate explicit action and cannot
be reached without an authorization phrase.  The A100 notebook delivered to the
researcher stops after preflight and never invokes the train action automatically.
"""

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

from models.iris.vnext.cohort_v9_fp32 import (
    IRISVNextFocusedRelationTrainerV9FP32,
    V9_NUMERIC_POLICY,
)


TRAIN_AUTHORIZATION = "AUTHORIZE_IRIS_V9_FOCUSED_RELATION_OPTIMIZER_STEPS"


def _git_head(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip().lower()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _trainer(args: argparse.Namespace) -> IRISVNextFocusedRelationTrainerV9FP32:
    repo_root = Path(args.repo_root).resolve()
    expected = args.expected_source_sha.lower()
    actual = _git_head(repo_root)
    if actual != expected:
        raise RuntimeError(f"VNEXT_V9_CHECKOUT_SHA_DRIFT:{actual}:{expected}")
    return IRISVNextFocusedRelationTrainerV9FP32(
        prepared_root=Path(args.prepared_root),
        prepared_manifest_path=Path(args.prepared_manifest),
        relation_root=Path(args.relation_root),
        relation_manifest_path=Path(args.relation_manifest),
        v8_checkpoint_path=Path(args.v8_checkpoint),
        migration_artifact_path=Path(args.migration_artifact),
        migration_receipt_path=Path(args.migration_receipt),
        materialization_receipt_path=Path(args.materialization_receipt),
        preregistration_path=Path(args.preregistration),
        readiness_receipt_path=Path(args.readiness_receipt),
        checkpoint_root=Path(args.checkpoint_root),
        expected_source_sha=expected,
        device=args.device,
        repo_root=repo_root,
    )


def preflight(args: argparse.Namespace) -> int:
    trainer = _trainer(args)
    if trainer.step != 0:
        raise RuntimeError("VNEXT_V9_PREFLIGHT_REQUIRES_FRESH_STEP_ZERO")
    heldout = trainer.evaluate_heldout()
    if trainer.step != 0:
        raise RuntimeError("VNEXT_V9_PREFLIGHT_EXECUTED_OPTIMIZER_STEP")
    result = {
        "schema": "RealSaS.IRISVNextFocusedRelationZeroStepPreflight.v1",
        "status": "PASS__ZERO_STEP_ONLY__TRAINING_NOT_STARTED",
        "source_commit_sha": args.expected_source_sha.lower(),
        "numeric_policy": V9_NUMERIC_POLICY,
        "binding_sha256": trainer.binding_hash,
        "heldout": heldout,
        "training_steps_executed": 0,
        "scientific_fit_started": False,
        "scientific_fit_pass_claimed": False,
        "next_action": "STOP_FOR_RESEARCHER_REVIEW_BEFORE_OPTIMIZER_STEP_1",
    }
    _write_json(Path(args.output), result)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


def train(args: argparse.Namespace) -> int:
    if args.authorize_optimizer_steps != TRAIN_AUTHORIZATION:
        raise RuntimeError("VNEXT_V9_OPTIMIZER_STEP_AUTHORIZATION_MISSING")
    if args.maximum_steps is None or args.maximum_steps <= 0:
        raise ValueError("VNEXT_V9_MAXIMUM_STEPS_REQUIRED")
    trainer = _trainer(args)
    if args.maximum_steps > trainer.policy.maximum_optimizer_steps:
        raise ValueError("VNEXT_V9_REQUESTED_STEPS_EXCEED_PREREGISTRATION")

    log_path = Path(args.output)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    while trainer.step < args.maximum_steps:
        row = trainer.one_step()
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        if trainer.step % trainer.policy.evaluation_interval == 0:
            heldout = trainer.evaluate_heldout()
            _write_json(
                log_path.with_name(f"heldout_step_{trainer.step:08d}.json"), heldout
            )
    trainer.save()
    final = {
        "schema": "RealSaS.IRISVNextFocusedRelationRunState.v1",
        "status": "SCHEDULE_STOP_REACHED",
        "source_commit_sha": args.expected_source_sha.lower(),
        "numeric_policy": V9_NUMERIC_POLICY,
        "binding_sha256": trainer.binding_hash,
        "training_steps_executed": trainer.step,
        "scientific_fit_pass_claimed": False,
    }
    _write_json(log_path.with_name("V9_RUN_STATE.json"), final)
    print(json.dumps(final, sort_keys=True, allow_nan=False))
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("preflight", "train"))
    p.add_argument("--repo-root", required=True)
    p.add_argument("--expected-source-sha", required=True)
    p.add_argument("--prepared-root", required=True)
    p.add_argument("--prepared-manifest", required=True)
    p.add_argument("--relation-root", required=True)
    p.add_argument("--relation-manifest", required=True)
    p.add_argument("--v8-checkpoint", required=True)
    p.add_argument("--migration-artifact", required=True)
    p.add_argument("--migration-receipt", required=True)
    p.add_argument("--materialization-receipt", required=True)
    p.add_argument("--preregistration", required=True)
    p.add_argument("--readiness-receipt", required=True)
    p.add_argument("--checkpoint-root", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", required=True)
    p.add_argument("--maximum-steps", type=int)
    p.add_argument("--authorize-optimizer-steps")
    return p


def main() -> int:
    args = parser().parse_args()
    if args.mode == "preflight":
        if args.authorize_optimizer_steps is not None or args.maximum_steps is not None:
            raise RuntimeError("VNEXT_V9_PREFLIGHT_MUST_NOT_ACCEPT_TRAINING_FLAGS")
        return preflight(args)
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
