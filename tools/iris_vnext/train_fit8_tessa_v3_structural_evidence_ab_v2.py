#!/usr/bin/env python3
from __future__ import annotations

"""Resumable locked Fit8 Structural-Evidence A/B trainer for TESSA V3.

V2 is an execution-hardening wrapper over the frozen V1 scientific contract.
It changes no architecture, optimizer, schedule, seed, conditioning or court
semantics.  It adds deterministic checkpoint restart so a long A100 run can be
re-entered without discarding completed optimizer work.

The inference firewall remains inherited from V1: materialized V2 all-hit truth,
teacher geometry/topology and named semantics are never accepted as A/B model
conditioning.  Checkpoints are lineage-bound to source, preregistration, corpus,
presentation manifest and the one shared initial state used by both arms.
"""

import argparse
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch

from models.iris.vnext.cohort_v1 import SUBJECT_IDS
from models.iris.vnext.structure_v4 import IRISVNextStructuralizationBridgeV4
from models.tessa.v2.experiment_v2 import sha256_file
from models.tessa.v2.objective_v2 import TESSAObjectiveWeightsV2
from tools.iris_vnext.materialize_fit8_direct_surface_v1 import git_head
from tools.iris_vnext.preflight_tessa_v2_a100_v1 import batch_from_subject, load_subject
from tools.iris_vnext.train_fit8_tessa_v3_structural_evidence_ab_v1 import (
    ARM_A,
    ARM_B,
    ARMS,
    CHECKPOINT_SCHEMA,
    PRESENTATION_MANIFEST_SCHEMA,
    RUN_SCHEMA,
    _checkpoint_payload,
    _load_presentation_subject,
    _subject_window_index,
    _typed_presentation,
    atomic_json,
    atomic_torch_save,
    learning_rate_at_step,
    load_json,
    validate_preregistration,
)


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def _latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    rows = sorted(checkpoint_dir.glob("step_*.pt")) if checkpoint_dir.is_dir() else []
    return rows[-1] if rows else None


def _validate_resume_payload(
    payload: dict[str, Any],
    *,
    arm_id: str,
    source_sha: str,
    prereg_sha: str,
    corpus_sha: str,
    presentation_sha: str,
    initial_state_sha: str,
    total_steps: int,
) -> int:
    if payload.get("schema") != CHECKPOINT_SCHEMA or payload.get("arm_id") != arm_id:
        raise ValueError("STRUCTURAL_EVIDENCE_AB_RESUME_CHECKPOINT_SCHEMA_OR_ARM_INVALID")
    exact = {
        "execution_source_sha": source_sha,
        "preregistration_sha256": prereg_sha,
        "corpus_manifest_sha256": corpus_sha,
        "presentation_manifest_sha256": presentation_sha,
        "initial_state_sha256": initial_state_sha,
    }
    for key, expected in exact.items():
        if payload.get(key) != expected:
            raise RuntimeError("STRUCTURAL_EVIDENCE_AB_RESUME_LINEAGE_DRIFT:" + key)
    completed = int(payload.get("completed_step", -1))
    if completed <= 0 or completed >= total_steps:
        raise ValueError("STRUCTURAL_EVIDENCE_AB_RESUME_STEP_INVALID")
    return completed


def _completed_receipt_if_valid(
    *,
    arm_root: Path,
    arm_id: str,
    total_steps: int,
    prereg_sha: str,
    corpus_sha: str,
    presentation_sha: str,
    initial_state_sha: str,
) -> dict[str, Any] | None:
    path = arm_root / "TRAINING_RECEIPT.json"
    if not path.is_file():
        return None
    receipt = load_json(path)
    exact = {
        "schema": "RealSaS.StructuralEvidenceABArmTrainingReceipt.v1",
        "arm_id": arm_id,
        "completed_optimizer_steps": total_steps,
        "preregistration_sha256": prereg_sha,
        "corpus_manifest_sha256": corpus_sha,
        "presentation_manifest_sha256": presentation_sha,
        "initial_state_sha256": initial_state_sha,
    }
    for key, expected in exact.items():
        if receipt.get(key) != expected:
            raise RuntimeError("STRUCTURAL_EVIDENCE_AB_COMPLETED_ARM_RECEIPT_DRIFT:" + key)
    checkpoint = arm_root / str(receipt["final_checkpoint"])
    if not checkpoint.is_file() or sha256_file(checkpoint) != receipt.get("final_checkpoint_sha256"):
        raise RuntimeError("STRUCTURAL_EVIDENCE_AB_COMPLETED_ARM_CHECKPOINT_DRIFT")
    return receipt


def _train_arm_resumable(
    *,
    arm_id: str,
    protocol: dict[str, Any],
    subjects: list[dict[str, Any]],
    presentations: list[dict[str, Any]],
    initial_state_path: Path,
    initial_state_sha: str,
    output_root: Path,
    device: torch.device,
    source_sha: str,
    prereg_sha: str,
    corpus_sha: str,
    presentation_sha: str,
) -> dict[str, Any]:
    total_steps = int(protocol["optimizer_steps_per_arm"])
    arm_root = output_root / arm_id
    arm_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = arm_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    telemetry_path = arm_root / "telemetry.jsonl"

    completed_receipt = _completed_receipt_if_valid(
        arm_root=arm_root,
        arm_id=arm_id,
        total_steps=total_steps,
        prereg_sha=prereg_sha,
        corpus_sha=corpus_sha,
        presentation_sha=presentation_sha,
        initial_state_sha=initial_state_sha,
    )
    if completed_receipt is not None:
        return completed_receipt

    seed = int(protocol["seed_set"][0])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = IRISVNextStructuralizationBridgeV4().to(device=device, dtype=torch.float32)
    model.load_state_dict(torch.load(initial_state_path, map_location="cpu", weights_only=True), strict=True)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt_cfg = protocol["optimizer"]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(opt_cfg["learning_rate"]),
        betas=tuple(map(float, opt_cfg["betas"])),
        eps=float(opt_cfg["epsilon"]),
        weight_decay=float(opt_cfg["weight_decay"]),
    )
    start_step = 1
    resume_checkpoint = _latest_checkpoint(checkpoint_dir)
    resumed_from = None
    if resume_checkpoint is not None:
        payload = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        completed = _validate_resume_payload(
            payload,
            arm_id=arm_id,
            source_sha=source_sha,
            prereg_sha=prereg_sha,
            corpus_sha=corpus_sha,
            presentation_sha=presentation_sha,
            initial_state_sha=initial_state_sha,
            total_steps=total_steps,
        )
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        _restore_rng(payload["rng_state"])
        start_step = completed + 1
        resumed_from = resume_checkpoint.name
    elif telemetry_path.exists() and telemetry_path.stat().st_size:
        raise RuntimeError("STRUCTURAL_EVIDENCE_AB_TELEMETRY_WITHOUT_RESUMABLE_CHECKPOINT")

    weights = TESSAObjectiveWeightsV2()
    started = time.time()
    for step in range(start_step, total_steps + 1):
        subject_index = (step - 1) % len(SUBJECT_IDS)
        subject = subjects[subject_index]
        window_count = int(subject["topology_input_ids"].shape[0])
        _, window_index = _subject_window_index(step, window_count)
        lr = learning_rate_at_step(step, protocol)
        for group in optimizer.param_groups:
            group["lr"] = lr
        scene, xyz, normal, h, mechanical, inputs, labels, offsets = batch_from_subject(subject, window_index, device)
        presentation = None if arm_id == ARM_A else _typed_presentation(presentations[subject_index], device)

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
                presentation=presentation,
                sequence_position_offset=offsets,
                objective_weights=weights,
            )
            loss = output.loss
        if loss is None or not bool(torch.isfinite(loss.detach()).item()):
            raise RuntimeError(f"STRUCTURAL_EVIDENCE_AB_NONFINITE_LOSS:{arm_id}:step={step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, float(opt_cfg["gradient_clip_norm"]))
        if not bool(torch.isfinite(torch.as_tensor(grad_norm)).item()):
            raise RuntimeError(f"STRUCTURAL_EVIDENCE_AB_NONFINITE_GRADIENT:{arm_id}:step={step}")
        optimizer.step()
        torch.cuda.synchronize()

        if step % int(protocol["telemetry_interval_steps"]) == 0 or step in (start_step, total_steps):
            row = {
                "arm_id": arm_id,
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

        interval = int(protocol["checkpoint_interval_steps"])
        if step % interval == 0 or step == total_steps:
            checkpoint = checkpoint_dir / f"step_{step:08d}.pt"
            atomic_torch_save(
                _checkpoint_payload(
                    arm_id=arm_id,
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    source_sha=source_sha,
                    prereg_sha=prereg_sha,
                    corpus_sha=corpus_sha,
                    presentation_sha=presentation_sha,
                    initial_state_sha=initial_state_sha,
                ),
                checkpoint,
            )
            atomic_json(
                {
                    "schema": "RealSaS.StructuralEvidenceABCheckpointReceipt.v1",
                    "arm_id": arm_id,
                    "step": step,
                    "file": checkpoint.name,
                    "sha256": sha256_file(checkpoint),
                    "initial_state_sha256": initial_state_sha,
                    "scientific_pass_claimed": False,
                },
                checkpoint.with_suffix(".json"),
            )

        del scene, xyz, normal, h, mechanical, inputs, labels, offsets, output, loss, presentation

    final_checkpoint = checkpoint_dir / f"step_{total_steps:08d}.pt"
    receipt = {
        "schema": "RealSaS.StructuralEvidenceABArmTrainingReceipt.v1",
        "status": "TRAINING_COMPLETED__FREE_RUNNING_COURT_REQUIRED",
        "arm_id": arm_id,
        "execution_source_sha": source_sha,
        "preregistration_sha256": prereg_sha,
        "corpus_manifest_sha256": corpus_sha,
        "presentation_manifest_sha256": presentation_sha,
        "initial_state_sha256": initial_state_sha,
        "completed_optimizer_steps": total_steps,
        "subject_steps": {subject: total_steps // len(SUBJECT_IDS) for subject in SUBJECT_IDS},
        "final_checkpoint": final_checkpoint.relative_to(arm_root).as_posix(),
        "final_checkpoint_sha256": sha256_file(final_checkpoint),
        "telemetry_sha256": sha256_file(telemetry_path),
        "resumed_from_checkpoint": resumed_from,
        "elapsed_wall_seconds_this_invocation": float(time.time() - started),
        "teacher_forced_loss_is_scientific_pass": False,
        "scientific_pass_claimed": False,
        "canonical_m_authorized": False,
    }
    atomic_json(receipt, arm_root / "TRAINING_RECEIPT.json")
    del model, optimizer, trainable
    torch.cuda.empty_cache()
    return receipt


def train(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    source_sha = str(args.expected_source_sha).strip().lower()
    if git_head(repo_root) != source_sha:
        raise RuntimeError("STRUCTURAL_EVIDENCE_AB_CHECKOUT_SHA_DRIFT")
    if not torch.cuda.is_available():
        raise RuntimeError("STRUCTURAL_EVIDENCE_AB_CUDA_REQUIRED")
    props = torch.cuda.get_device_properties(0)
    if "A100" not in str(props.name).upper():
        raise RuntimeError("STRUCTURAL_EVIDENCE_AB_A100_REQUIRED:" + str(props.name))
    device = torch.device("cuda")

    prereg_path = Path(args.preregistration).resolve()
    prereg = load_json(prereg_path)
    protocol = validate_preregistration(prereg)
    prereg_sha = sha256_file(prereg_path)

    corpus_root = Path(args.corpus_root).resolve()
    corpus_path = corpus_root / "TESSA_V2_CORPUS_MANIFEST.json"
    corpus = load_json(corpus_path)
    corpus_sha = sha256_file(corpus_path)
    if corpus.get("schema") != "RealSaS.Fit8TESSAV2TrainingCorpus.v1":
        raise ValueError("STRUCTURAL_EVIDENCE_AB_CORPUS_SCHEMA_INVALID")
    rows = list(corpus.get("subjects") or ())
    if tuple(row.get("subject_id") for row in rows) != SUBJECT_IDS:
        raise ValueError("STRUCTURAL_EVIDENCE_AB_CORPUS_SUBJECT_ORDER_DRIFT")
    subjects = [load_subject(corpus_root / str(row["file"]), str(row["sha256"])) for row in rows]

    presentation_root = Path(args.presentation_root).resolve()
    presentation_manifest_path = presentation_root / "DIRECT_IRIS_PRESENTATION_MEMORY_V1_MANIFEST.json"
    presentation_manifest = load_json(presentation_manifest_path)
    presentation_sha = sha256_file(presentation_manifest_path)
    if presentation_manifest.get("schema") != PRESENTATION_MANIFEST_SCHEMA:
        raise ValueError("STRUCTURAL_EVIDENCE_AB_PRESENTATION_MANIFEST_SCHEMA_INVALID")
    for forbidden in (
        "teacher_geometry_in_conditioning",
        "teacher_topology_in_conditioning",
        "materialized_v2_truth_in_conditioning",
    ):
        if presentation_manifest.get(forbidden) is not False:
            raise ValueError("STRUCTURAL_EVIDENCE_AB_PRESENTATION_MANIFEST_FIREWALL_BROKEN:" + forbidden)
    prows = list(presentation_manifest.get("subjects") or ())
    if tuple(row.get("subject_id") for row in prows) != SUBJECT_IDS:
        raise ValueError("STRUCTURAL_EVIDENCE_AB_PRESENTATION_SUBJECT_ORDER_DRIFT")
    presentations = [
        _load_presentation_subject(presentation_root / str(row["file"]), str(row["sha256"]), subject_id)
        for subject_id, row in zip(SUBJECT_IDS, prows, strict=True)
    ]

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    initial_state_path = output_root / "INITIAL_MODEL_STATE.pt"
    seed = int(protocol["seed_set"][0])
    if initial_state_path.is_file():
        initial_state_sha = sha256_file(initial_state_path)
    else:
        if any((output_root / arm).exists() for arm in ARMS):
            raise RuntimeError("STRUCTURAL_EVIDENCE_AB_INITIAL_STATE_MISSING_WITH_PARTIAL_ARM_OUTPUT")
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        initial = IRISVNextStructuralizationBridgeV4().to(device="cpu", dtype=torch.float32)
        atomic_torch_save(initial.state_dict(), initial_state_path)
        initial_state_sha = sha256_file(initial_state_path)
        del initial

    arm_receipts = []
    for arm_id in ARMS:
        arm_receipts.append(
            _train_arm_resumable(
                arm_id=arm_id,
                protocol=protocol,
                subjects=subjects,
                presentations=presentations,
                initial_state_path=initial_state_path,
                initial_state_sha=initial_state_sha,
                output_root=output_root,
                device=device,
                source_sha=source_sha,
                prereg_sha=prereg_sha,
                corpus_sha=corpus_sha,
                presentation_sha=presentation_sha,
            )
        )
    if arm_receipts[0]["initial_state_sha256"] != arm_receipts[1]["initial_state_sha256"]:
        raise AssertionError("STRUCTURAL_EVIDENCE_AB_INITIALIZATION_MISMATCH")
    result = {
        "schema": RUN_SCHEMA,
        "status": "BOTH_ARMS_TRAINED__STRICT_FREE_RUNNING_AND_INDEPENDENT_COURT_REQUIRED",
        "execution_source_sha": source_sha,
        "preregistration_sha256": prereg_sha,
        "corpus_manifest_sha256": corpus_sha,
        "presentation_manifest_sha256": presentation_sha,
        "initial_state_sha256": initial_state_sha,
        "arm_order": list(ARMS),
        "arms": arm_receipts,
        "optimizer_steps_per_arm": int(protocol["optimizer_steps_per_arm"]),
        "total_optimizer_steps": 2 * int(protocol["optimizer_steps_per_arm"]),
        "same_initialization_between_arms": True,
        "restart_semantics": "DETERMINISTIC_CHECKPOINT_RESUME_WITH_EXACT_RNG_MODEL_OPTIMIZER_LINEAGE",
        "teacher_or_all_hit_v2_truth_used_as_inference_conditioning": False,
        "scientific_pass_claimed": False,
        "canonical_m_authorized": False,
        "next_action": "STRICT_FREE_RUNNING_BOTH_ARMS_THEN_PREREGISTERED_INDEPENDENT_AB_COURT",
    }
    atomic_json(result, output_root / "STRUCTURAL_EVIDENCE_AB_TRAINING_RECEIPT.json")
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", required=True)
    p.add_argument("--expected-source-sha", required=True)
    p.add_argument("--preregistration", default="canonical/STRUCTURAL_EVIDENCE_AB_PREREGISTRATION_V2.json")
    p.add_argument("--corpus-root", required=True)
    p.add_argument("--presentation-root", required=True)
    p.add_argument("--output-root", required=True)
    result = train(p.parse_args())
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
