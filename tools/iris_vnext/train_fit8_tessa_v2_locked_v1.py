#!/usr/bin/env python3
from __future__ import annotations

"""Locked 2400-step Fit8 TESSA V2 trainer.

The trainer consumes only a PASS pretraining-admission receipt plus the exact
materialized corpus bound by that receipt. Training telemetry is diagnostic and
cannot mint scientific PASS. Final judgment is reserved for strict free-running
A/B courts.
"""

import argparse
import json
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any

import numpy as np
import torch

from models.iris.vnext.cohort_v1 import SUBJECT_IDS
from models.iris.vnext.structure_global_v3 import IRISVNextGlobalStructuralizationBridgeV3
from models.tessa.v2.experiment_v2 import (
    learning_rate_at_step_v1,
    load_preregistration_v1,
    sha256_file,
)
from models.tessa.v2.objective_v2 import TESSAObjectiveWeightsV2
from tools.iris_vnext.materialize_fit8_direct_surface_v1 import git_head
from tools.iris_vnext.qualify_tessa_v2_pretraining_admission_v1 import PASS_STATUS
from tools.iris_vnext.preflight_tessa_v2_a100_v1 import batch_from_subject, load_subject

SCHEMA = "RealSaS.TESSAV2LockedTrainingRun.v1"
CHECKPOINT_SCHEMA = "RealSaS.TESSAV2LockedCheckpoint.v1"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("TESSA_V2_TRAIN_JSON_OBJECT_REQUIRED")
    return value


def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".staged", delete=False) as handle:
        staged = Path(handle.name)
    try:
        torch.save(value, staged)
        os.replace(staged, path)
    finally:
        staged.unlink(missing_ok=True)


def atomic_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(path.suffix + ".staged")
    staged.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(staged, path)


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def _subject_window_index(step: int, subject_window_count: int) -> tuple[int, int]:
    if step <= 0 or subject_window_count <= 0:
        raise ValueError("TESSA_V2_TRAIN_SCHEDULE_INPUT_INVALID")
    subject_index = (step - 1) % len(SUBJECT_IDS)
    subject_visit = (step - 1) // len(SUBJECT_IDS)
    return subject_index, subject_visit % subject_window_count


def _checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    source_sha: str,
    prereg_sha: str,
    corpus_sha: str,
    readiness_sha: str,
    telemetry_sha: str | None,
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "completed_step": int(step),
        "execution_source_sha": source_sha,
        "preregistration_sha256": prereg_sha,
        "corpus_manifest_sha256": corpus_sha,
        "pretraining_admission_sha256": readiness_sha,
        "telemetry_sha256_before_checkpoint": telemetry_sha,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng_state": _rng_state(),
        "scientific_pass_claimed": False,
        "tessa_mandatory": False,
        "canonical_m_authorized": False,
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    source_sha = str(args.expected_source_sha).lower()
    if git_head(repo_root) != source_sha:
        raise RuntimeError("TESSA_V2_TRAIN_CHECKOUT_SHA_DRIFT")
    if not torch.cuda.is_available():
        raise RuntimeError("TESSA_V2_TRAIN_CUDA_REQUIRED")
    props = torch.cuda.get_device_properties(0)
    if "A100" not in str(props.name).upper():
        raise RuntimeError("TESSA_V2_TRAIN_A100_REQUIRED:" + str(props.name))
    device = torch.device("cuda")

    prereg_path = Path(args.preregistration).resolve()
    prereg = load_preregistration_v1(prereg_path)
    prereg_sha = sha256_file(prereg_path)
    corpus_root = Path(args.corpus_root).resolve()
    corpus_path = corpus_root / "TESSA_V2_CORPUS_MANIFEST.json"
    corpus = load_json(corpus_path)
    corpus_sha = sha256_file(corpus_path)
    readiness_path = Path(args.pretraining_admission).resolve()
    readiness = load_json(readiness_path)
    readiness_sha = sha256_file(readiness_path)
    if readiness.get("status") != PASS_STATUS or readiness.get("optimizer_step_1_authorized") is not True:
        raise RuntimeError("TESSA_V2_TRAIN_PRETRAINING_ADMISSION_NOT_PASS")
    if readiness.get("preregistration_sha256") != prereg_sha:
        raise RuntimeError("TESSA_V2_TRAIN_PREREGISTRATION_BINDING_DRIFT")
    if readiness.get("corpus_manifest_sha256") != corpus_sha:
        raise RuntimeError("TESSA_V2_TRAIN_CORPUS_BINDING_DRIFT")
    if corpus.get("schema") != "RealSaS.Fit8TESSAV2TrainingCorpus.v1" or tuple(corpus.get("subject_order") or ()) != SUBJECT_IDS:
        raise ValueError("TESSA_V2_TRAIN_CORPUS_INVALID")

    rows = list(corpus["subjects"])
    subjects: list[dict[str, Any]] = []
    for subject_id, row in zip(SUBJECT_IDS, rows, strict=True):
        if row.get("subject_id") != subject_id:
            raise ValueError("TESSA_V2_TRAIN_SUBJECT_ORDER_DRIFT")
        subjects.append(load_subject(corpus_root / str(row["file"]), str(row["sha256"])))

    opt_cfg = prereg["optimizer"]
    seed = int(opt_cfg["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = IRISVNextGlobalStructuralizationBridgeV3().to(device=device, dtype=torch.float32)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(opt_cfg["learning_rate"]),
        betas=tuple(map(float, opt_cfg["betas"])),
        eps=float(opt_cfg["epsilon"]),
        weight_decay=float(opt_cfg["weight_decay"]),
    )
    total_steps = int(opt_cfg["optimizer_steps"])
    if total_steps != 2400:
        raise RuntimeError("TESSA_V2_TRAIN_HORIZON_DRIFT")
    weights = TESSAObjectiveWeightsV2()

    output_root = Path(args.output_root).resolve(); output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_root / "checkpoints"; checkpoint_dir.mkdir(parents=True, exist_ok=True)
    telemetry_path = output_root / "telemetry.jsonl"
    start_step = 1
    resume_path = Path(args.resume).resolve() if args.resume else None
    if resume_path:
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        if payload.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError("TESSA_V2_TRAIN_RESUME_SCHEMA_INVALID")
        exact = {
            "execution_source_sha": source_sha,
            "preregistration_sha256": prereg_sha,
            "corpus_manifest_sha256": corpus_sha,
            "pretraining_admission_sha256": readiness_sha,
        }
        for key, expected in exact.items():
            if payload.get(key) != expected:
                raise RuntimeError("TESSA_V2_TRAIN_RESUME_LINEAGE_DRIFT:" + key)
        completed = int(payload.get("completed_step", -1))
        if completed < 0 or completed >= total_steps:
            raise ValueError("TESSA_V2_TRAIN_RESUME_STEP_INVALID")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        _restore_rng(payload["rng_state"])
        start_step = completed + 1

    started = time.time()
    for step in range(start_step, total_steps + 1):
        subject_index = (step - 1) % len(SUBJECT_IDS)
        subject = subjects[subject_index]
        window_count = int(subject["topology_input_ids"].shape[0])
        _, window_index = _subject_window_index(step, window_count)
        lr = learning_rate_at_step_v1(step, prereg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        scene, xyz, normal, h, mechanical, inputs, labels, offsets = batch_from_subject(subject, window_index, device)

        optimizer.zero_grad(set_to_none=True)
        step_started = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(
                scene_planes=scene,
                witness_xyz=xyz,
                witness_normal=normal,
                witness_h=h,
                witness_mechanical=mechanical,
                input_ids=inputs,
                labels=labels,
                sequence_position_offset=offsets,
                objective_weights=weights,
            )
            loss = output.loss
        if loss is None or not bool(torch.isfinite(loss.detach()).item()):
            raise RuntimeError(f"TESSA_V2_TRAIN_NONFINITE_LOSS:step={step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, float(opt_cfg["gradient_clip_norm"]))
        if not bool(torch.isfinite(torch.as_tensor(grad_norm)).item()):
            raise RuntimeError(f"TESSA_V2_TRAIN_NONFINITE_GRADIENT:step={step}")
        optimizer.step()
        torch.cuda.synchronize()

        if step % int(opt_cfg["telemetry_interval_steps"]) == 0 or step in (start_step, total_steps):
            row = {
                "step": step,
                "subject_id": SUBJECT_IDS[subject_index],
                "window_index": window_index,
                "learning_rate": lr,
                "loss": float(loss.detach().float().cpu()),
                "token_cross_entropy": float(output.objective.token_cross_entropy.detach().float().cpu()),
                "coordinate_metric": float(output.objective.coordinate_metric.detach().float().cpu()),
                "structural_special_token": float(output.objective.structural_special_token.detach().float().cpu()),
                "gradient_norm": float(torch.as_tensor(grad_norm).detach().float().cpu()),
                "step_seconds": float(time.perf_counter() - step_started),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "scientific_pass_claimed": False,
            }
            with telemetry_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")

        interval = int(opt_cfg["checkpoint_interval_steps"])
        if step % interval == 0 or step == total_steps:
            telemetry_sha = sha256_file(telemetry_path) if telemetry_path.exists() else None
            payload = _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                step=step,
                source_sha=source_sha,
                prereg_sha=prereg_sha,
                corpus_sha=corpus_sha,
                readiness_sha=readiness_sha,
                telemetry_sha=telemetry_sha,
            )
            checkpoint = checkpoint_dir / f"step_{step:08d}.pt"
            atomic_torch_save(payload, checkpoint)
            atomic_json({
                "schema": "RealSaS.TESSAV2CheckpointReceipt.v1",
                "step": step,
                "file": checkpoint.name,
                "sha256": sha256_file(checkpoint),
                "preregistration_sha256": prereg_sha,
                "corpus_manifest_sha256": corpus_sha,
                "pretraining_admission_sha256": readiness_sha,
                "scientific_pass_claimed": False,
            }, checkpoint.with_suffix(".json"))

        del scene, xyz, normal, h, mechanical, inputs, labels, offsets, output, loss

    final_checkpoint = checkpoint_dir / f"step_{total_steps:08d}.pt"
    result = {
        "schema": SCHEMA,
        "status": "TRAINING_COMPLETED__FREE_RUNNING_COURT_REQUIRED",
        "execution_source_sha": source_sha,
        "preregistration_sha256": prereg_sha,
        "corpus_manifest_sha256": corpus_sha,
        "pretraining_admission_sha256": readiness_sha,
        "completed_optimizer_steps": total_steps,
        "subject_steps": {subject: total_steps // len(SUBJECT_IDS) for subject in SUBJECT_IDS},
        "final_checkpoint": final_checkpoint.name,
        "final_checkpoint_sha256": sha256_file(final_checkpoint),
        "telemetry_sha256": sha256_file(telemetry_path),
        "elapsed_wall_seconds": float(time.time() - started),
        "teacher_forced_loss_is_scientific_pass": False,
        "scientific_pass_claimed": False,
        "tessa_mandatory": False,
        "canonical_m_authorized": False,
        "next_action": "STRICT_FREE_RUNNING_ALL8_PARTIAL_AB_COURT",
    }
    atomic_json(result, output_root / "TESSA_V2_TRAINING_RECEIPT.json")
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", required=True)
    p.add_argument("--expected-source-sha", required=True)
    p.add_argument("--preregistration", default="canonical/TESSA_V2_FIT8_PREREGISTRATION_V1.json")
    p.add_argument("--corpus-root", required=True)
    p.add_argument("--pretraining-admission", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--resume")
    result = train(p.parse_args())
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
