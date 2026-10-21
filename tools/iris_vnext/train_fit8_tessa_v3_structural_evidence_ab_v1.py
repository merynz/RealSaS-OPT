#!/usr/bin/env python3
from __future__ import annotations

"""Locked Fit8 structural-evidence A/B trainer for TESSA V3.

A and B use the same TESSA V3 parameterization, exact initial weights, optimizer,
subject/window schedule and update budget.  The only conditioning difference is:
A -> null presentation evidence; B -> inference-admissible role-free presentation
memory built from model-produced surface anchors + admitted source views.

This trainer never accepts the materialized V2 all-hit/teacher truth as inference
conditioning.  That corpus remains supervision / independent-evaluation authority.
Training telemetry cannot mint scientific PASS; final judgement belongs to the
preregistered free-running independent court.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any

import numpy as np
import torch

from models.iris.vnext.cohort_v1 import SUBJECT_IDS
from models.iris.vnext.presentation_evidence_v1 import RoleFreePresentationEvidenceV1
from models.iris.vnext.structure_v4 import IRISVNextStructuralizationBridgeV4
from models.tessa.v2.experiment_v2 import sha256_file
from models.tessa.v2.objective_v2 import TESSAObjectiveWeightsV2
from tools.iris_vnext.materialize_fit8_direct_surface_v1 import git_head
from tools.iris_vnext.preflight_tessa_v2_a100_v1 import batch_from_subject, load_subject

PREREG_SCHEMA = "RealSaS.StructuralEvidenceABPreregistration.v2"
PRESENTATION_MANIFEST_SCHEMA = "RealSaS.DirectIRISPresentationMemoryManifest.v1"
PRESENTATION_SUBJECT_SCHEMA = "RealSaS.DirectIRISPresentationMemorySubject.v1"
RUN_SCHEMA = "RealSaS.StructuralEvidenceABTrainingRun.v1"
CHECKPOINT_SCHEMA = "RealSaS.StructuralEvidenceABCheckpoint.v1"
ARM_A = "A_NULL_PRESENTATION"
ARM_B = "B_ROLE_FREE_PRESENTATION"
ARMS = (ARM_A, ARM_B)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("STRUCTURAL_EVIDENCE_AB_JSON_OBJECT_REQUIRED")
    return value


def atomic_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(path.suffix + ".staged")
    staged.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(staged, path)


def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".staged", delete=False) as handle:
        staged = Path(handle.name)
    try:
        torch.save(value, staged)
        os.replace(staged, path)
    finally:
        staged.unlink(missing_ok=True)


def validate_preregistration(prereg: dict[str, Any]) -> dict[str, Any]:
    if prereg.get("schema") != PREREG_SCHEMA:
        raise ValueError("STRUCTURAL_EVIDENCE_AB_PREREG_SCHEMA_INVALID")
    if prereg.get("status") != "LOCKED_PREOPTIMIZER_SOURCE_CONTRACT__USER_HANDOFF_REQUIRED":
        raise ValueError("STRUCTURAL_EVIDENCE_AB_PREREG_STATUS_INVALID")
    common = prereg.get("common_controls") or {}
    required_true = (
        "same_tessa_v3_parameterization",
        "same_initialization_between_arms",
        "same_optimizer_and_schedule",
        "same_update_budget",
        "same_batch_and_sampling_policy",
        "same_checkpoint_schedule",
    )
    if any(common.get(key) is not True for key in required_true):
        raise ValueError("STRUCTURAL_EVIDENCE_AB_MATCHED_CONTROL_DRIFT")
    evidence = prereg.get("evidence_input_authority") or {}
    if evidence.get("teacher_all_hit_evidence_allowed_as_inference_conditioning") is not False:
        raise ValueError("STRUCTURAL_EVIDENCE_AB_TEACHER_EVIDENCE_FIREWALL_BROKEN")
    protocol = prereg.get("training_protocol") or {}
    exact = {
        "seed_set": [26091991],
        "seed_count": 1,
        "optimizer_steps_per_arm": 2400,
        "warmup_steps": 120,
        "minimum_learning_rate": 1.0e-5,
        "schedule": "LINEAR_WARMUP_THEN_COSINE",
        "checkpoint_interval_steps": 200,
        "telemetry_interval_steps": 10,
        "subject_schedule": "ROUND_ROBIN_8_SUBJECTS__300_OPTIMIZER_STEPS_EACH",
        "window_schedule": "DETERMINISTIC_CYCLIC_PER_SUBJECT",
        "gradient_accumulation_steps": 1,
    }
    for key, expected in exact.items():
        if protocol.get(key) != expected:
            raise ValueError("STRUCTURAL_EVIDENCE_AB_PROTOCOL_DRIFT:" + key)
    opt = protocol.get("optimizer") or {}
    if (
        opt.get("name") != "AdamW"
        or float(opt.get("learning_rate", -1.0)) != 1.0e-4
        or list(opt.get("betas") or ()) != [0.9, 0.95]
        or float(opt.get("epsilon", -1.0)) != 1.0e-8
        or float(opt.get("weight_decay", -1.0)) != 0.01
        or float(opt.get("gradient_clip_norm", -1.0)) != 1.0
    ):
        raise ValueError("STRUCTURAL_EVIDENCE_AB_OPTIMIZER_DRIFT")
    return protocol


def learning_rate_at_step(step: int, protocol: dict[str, Any]) -> float:
    total = int(protocol["optimizer_steps_per_arm"])
    warmup = int(protocol["warmup_steps"])
    peak = float(protocol["optimizer"]["learning_rate"])
    floor = float(protocol["minimum_learning_rate"])
    if step <= 0 or step > total:
        raise ValueError("STRUCTURAL_EVIDENCE_AB_LR_STEP_OUT_OF_RANGE")
    if step <= warmup:
        return peak * float(step) / float(warmup)
    progress = float(step - warmup) / float(max(1, total - warmup))
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * progress))


def _subject_window_index(step: int, count: int) -> tuple[int, int]:
    if step <= 0 or count <= 0:
        raise ValueError("STRUCTURAL_EVIDENCE_AB_SCHEDULE_INPUT_INVALID")
    subject_index = (step - 1) % len(SUBJECT_IDS)
    subject_visit = (step - 1) // len(SUBJECT_IDS)
    return subject_index, subject_visit % count


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def _load_presentation_subject(path: Path, expected_sha: str, subject_id: str) -> dict[str, Any]:
    if sha256_file(path) != expected_sha:
        raise ValueError("STRUCTURAL_EVIDENCE_AB_PRESENTATION_SUBJECT_SHA_DRIFT")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value.get("schema") != PRESENTATION_SUBJECT_SCHEMA or value.get("subject_id") != subject_id:
        raise ValueError("STRUCTURAL_EVIDENCE_AB_PRESENTATION_SUBJECT_INVALID")
    for forbidden in (
        "teacher_geometry_in_conditioning",
        "teacher_topology_in_conditioning",
        "materialized_v2_truth_in_conditioning",
    ):
        if value.get(forbidden) is not False:
            raise ValueError("STRUCTURAL_EVIDENCE_AB_PRESENTATION_FIREWALL_BROKEN:" + forbidden)
    return value


def _typed_presentation(payload: dict[str, Any], device: torch.device) -> RoleFreePresentationEvidenceV1:
    values = payload["values"]
    valid = payload["valid_mask"]
    kind = payload["kind_id"]
    if not isinstance(values, torch.Tensor) or not isinstance(valid, torch.Tensor) or not isinstance(kind, torch.Tensor):
        raise TypeError("STRUCTURAL_EVIDENCE_AB_PRESENTATION_TENSOR_REQUIRED")
    return RoleFreePresentationEvidenceV1(
        values=values.to(device=device, dtype=torch.float32)[None],
        valid_mask=valid.to(device=device, dtype=torch.bool)[None],
        kind_id=kind.to(device=device, dtype=torch.long)[None],
    )


def _checkpoint_payload(*, arm_id: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                        step: int, source_sha: str, prereg_sha: str, corpus_sha: str,
                        presentation_sha: str, initial_state_sha: str) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "arm_id": arm_id,
        "completed_step": int(step),
        "execution_source_sha": source_sha,
        "preregistration_sha256": prereg_sha,
        "corpus_manifest_sha256": corpus_sha,
        "presentation_manifest_sha256": presentation_sha,
        "initial_state_sha256": initial_state_sha,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng_state": _rng_state(),
        "scientific_pass_claimed": False,
        "canonical_m_authorized": False,
    }


def _train_arm(*, arm_id: str, protocol: dict[str, Any], subjects: list[dict[str, Any]],
               presentations: list[dict[str, Any]], initial_state_path: Path,
               initial_state_sha: str, output_root: Path, device: torch.device,
               source_sha: str, prereg_sha: str, corpus_sha: str,
               presentation_sha: str) -> dict[str, Any]:
    seed = int(protocol["seed_set"][0])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
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
    total_steps = int(protocol["optimizer_steps_per_arm"])
    arm_root = output_root / arm_id
    arm_root.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = arm_root / "checkpoints"; checkpoint_dir.mkdir()
    telemetry_path = arm_root / "telemetry.jsonl"
    weights = TESSAObjectiveWeightsV2()
    started = time.time()

    for step in range(1, total_steps + 1):
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
        optimizer.step(); torch.cuda.synchronize()

        if step % int(protocol["telemetry_interval_steps"]) == 0 or step in (1, total_steps):
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
                    arm_id=arm_id, model=model, optimizer=optimizer, step=step,
                    source_sha=source_sha, prereg_sha=prereg_sha, corpus_sha=corpus_sha,
                    presentation_sha=presentation_sha, initial_state_sha=initial_state_sha,
                ),
                checkpoint,
            )
            atomic_json({
                "schema": "RealSaS.StructuralEvidenceABCheckpointReceipt.v1",
                "arm_id": arm_id,
                "step": step,
                "file": checkpoint.name,
                "sha256": sha256_file(checkpoint),
                "initial_state_sha256": initial_state_sha,
                "scientific_pass_claimed": False,
            }, checkpoint.with_suffix(".json"))

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
        "elapsed_wall_seconds": float(time.time() - started),
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
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError("STRUCTURAL_EVIDENCE_AB_OUTPUT_MUST_BE_EMPTY")
    output_root.mkdir(parents=True, exist_ok=True)

    seed = int(protocol["seed_set"][0])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    initial = IRISVNextStructuralizationBridgeV4().to(device="cpu", dtype=torch.float32)
    initial_state_path = output_root / "INITIAL_MODEL_STATE.pt"
    atomic_torch_save(initial.state_dict(), initial_state_path)
    initial_state_sha = sha256_file(initial_state_path)
    del initial

    arm_receipts = []
    for arm_id in ARMS:
        arm_receipts.append(_train_arm(
            arm_id=arm_id, protocol=protocol, subjects=subjects, presentations=presentations,
            initial_state_path=initial_state_path, initial_state_sha=initial_state_sha,
            output_root=output_root, device=device, source_sha=source_sha,
            prereg_sha=prereg_sha, corpus_sha=corpus_sha, presentation_sha=presentation_sha,
        ))

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
