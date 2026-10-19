#!/usr/bin/env python3
from __future__ import annotations

"""Train exact equal-capacity H-REP-P0 controls over sealed V8 features."""

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any

import torch

from models.iris.vnext.cohort_v1 import SUBJECT_IDS
from models.iris.vnext.h_rep_probe_v1 import (
    PROBE_ARMS_V1,
    HRepresentationProbeV1,
    build_probe_input_v1,
    h_rep_probe_objective_v1,
)
from tools.iris_vnext.h_rep_probe_authority_v1 import (
    load_json,
    require_a100_runtime_v1,
    require_exact_clean_source_v1,
    sha256_file,
)
from tools.iris_vnext.h_rep_probe_runtime_v1 import (
    cache_frozen_v8_probe_features_v1,
    load_h_rep_binding_v1,
)

SCHEMA = "RealSaS.HRepP0TrainingResult.v1"
CHECKPOINT_SCHEMA = "RealSaS.HRepP0Checkpoint.v1"
TOTAL_STEPS = 1200
BATCH_SIZE = 2048


def _atomic_torch(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def _atomic_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _lr(step: int) -> float:
    warm = 60
    if step < warm:
        return 2.0e-4 * float(step + 1) / float(warm)
    t = (step - warm) / max(1, TOTAL_STEPS - warm - 1)
    return 2.0e-5 + 0.5 * (2.0e-4 - 2.0e-5) * (1.0 + math.cos(math.pi * t))


def train(args: argparse.Namespace) -> dict[str, Any]:
    repo = Path(args.repo_root).resolve()
    source_sha = str(args.expected_source_sha).lower()
    require_exact_clean_source_v1(repo, source_sha)

    binding_path = Path(args.binding).resolve()
    prereg_path = Path(args.preregistration).resolve()
    dataset_receipt = Path(args.dataset_receipt).resolve()
    binding = load_h_rep_binding_v1(binding_path)
    prereg = load_json(prereg_path)
    if prereg.get("schema") != "RealSaS.HRepP0Preregistration.v1":
        raise ValueError("H_REP_P0_PREREG_SCHEMA_INVALID")
    if prereg.get("input_binding") != "canonical/H_REP_P0_INPUT_BINDING_V1.json":
        raise ValueError("H_REP_P0_PREREG_BINDING_PATH_DRIFT")
    runtime = prereg.get("runtime_admission") or {}
    if runtime.get("accelerator") != "NVIDIA_A100_CLASS":
        raise ValueError("H_REP_P0_ACCELERATOR_PREREG_DRIFT")
    minimum_memory_gib = float(runtime.get("minimum_cuda_memory_gib", -1))
    if minimum_memory_gib != 39.0:
        raise ValueError("H_REP_P0_MEMORY_PREREG_DRIFT")
    if runtime.get("optimizer_step_allowed_before_all_authority_gates") is not False:
        raise ValueError("H_REP_P0_OPTIMIZER_ADMISSION_PREREG_DRIFT")
    gpu = require_a100_runtime_v1(minimum_memory_gib)

    binding_sha = sha256_file(binding_path)
    prereg_sha = sha256_file(prereg_path)
    dataset_sha = sha256_file(dataset_receipt)
    dataset_meta = load_json(dataset_receipt)
    if dataset_meta.get("input_binding_sha256") != binding_sha:
        raise ValueError("H_REP_P0_TRAIN_DATASET_BINDING_SHA_DRIFT")

    device = torch.device("cuda")
    torch.manual_seed(26101990)
    torch.cuda.manual_seed_all(26101990)

    cache, scene_hash = cache_frozen_v8_probe_features_v1(
        v8_checkpoint=Path(args.v8_checkpoint),
        prepared_root=Path(args.prepared_root),
        prepared_manifest=Path(args.prepared_manifest),
        dataset_root=Path(args.dataset_root),
        dataset_receipt=dataset_receipt,
        binding=binding,
        binding_sha256=binding_sha,
        split="train",
        device=device,
        query_chunk=int(args.query_chunk),
    )

    models = {arm: HRepresentationProbeV1(arm).to(device) for arm in PROBE_ARMS_V1}
    parameter_counts = {arm: models[arm].trainable_parameter_count() for arm in PROBE_ARMS_V1}
    if len(set(parameter_counts.values())) != 1:
        raise RuntimeError("H_REP_P0_ARM_PARAMETER_COUNT_DRIFT")
    optimizers = {
        arm: torch.optim.AdamW(
            models[arm].parameters(), lr=2.0e-4, betas=(0.9, 0.95), weight_decay=0.01
        )
        for arm in PROBE_ARMS_V1
    }
    print("H_REP_P0_A100_AUTHORITY", json.dumps(gpu, sort_keys=True), flush=True)
    print("H_REP_P0_EQUAL_CAPACITY", json.dumps(parameter_counts, sort_keys=True), flush=True)

    output = Path(args.output_dir).resolve()
    mirror = Path(args.checkpoint_mirror_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    mirror.mkdir(parents=True, exist_ok=True)
    local = output / "checkpoint_latest.pt"
    remote = mirror / "checkpoint_latest.pt"
    completed = 0
    if not local.is_file() and remote.is_file():
        shutil.copy2(remote, local)

    authority = {
        "execution_source_sha": source_sha,
        "binding_sha256": binding_sha,
        "preregistration_sha256": prereg_sha,
        "dataset_receipt_sha256": dataset_sha,
        "v8_scene_core_sha256": scene_hash,
    }
    if local.is_file():
        checkpoint = torch.load(local, map_location="cpu", weights_only=True)
        if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
            raise RuntimeError("H_REP_P0_RESUME_SCHEMA_DRIFT")
        for key, value in authority.items():
            if checkpoint.get(key) != value:
                raise RuntimeError("H_REP_P0_RESUME_AUTHORITY_DRIFT:" + key)
        if checkpoint.get("parameter_counts") != parameter_counts:
            raise RuntimeError("H_REP_P0_RESUME_PARAMETER_COUNT_DRIFT")
        for arm in PROBE_ARMS_V1:
            models[arm].load_state_dict(checkpoint["models"][arm], strict=True)
            optimizers[arm].load_state_dict(checkpoint["optimizers"][arm])
        completed = int(checkpoint["completed_steps"])
        if completed < 0 or completed > TOTAL_STEPS:
            raise RuntimeError("H_REP_P0_RESUME_STEP_INVALID")
        print("H_REP_P0_RESUME", completed, flush=True)

    started = time.perf_counter()
    generators = {
        subject: torch.Generator(device="cpu").manual_seed(26104000 + index)
        for index, subject in enumerate(SUBJECT_IDS)
    }
    for step in range(completed, TOTAL_STEPS):
        subject = SUBJECT_IDS[step % len(SUBJECT_IDS)]
        item = cache[subject]
        n = len(item["xyz_iris"])
        index = torch.randint(
            0, n, (min(BATCH_SIZE, n),), generator=generators[subject]
        )
        batch = {
            key: value[index].to(device=device)
            for key, value in item.items()
            if key != "shuffle_index"
        }
        shuffled = item["H"][item["shuffle_index"][index]].to(device=device)
        lr = _lr(step)
        logs: dict[str, float] = {}
        for arm in PROBE_ARMS_V1:
            for group in optimizers[arm].param_groups:
                group["lr"] = lr
            optimizers[arm].zero_grad(set_to_none=True)
            x = build_probe_input_v1(
                arm,
                h=batch["H"],
                xyz_iris=batch["xyz_iris"],
                raw_field=batch["raw_field"],
                shuffled_h=shuffled if arm == "SHUFFLED_H_XYZ" else None,
            )
            prediction = models[arm](x)
            losses = h_rep_probe_objective_v1(
                prediction,
                distance_iris=batch["distance_iris"],
                offset_to_reference_iris=batch["offset_to_reference_iris"],
                normal_unit=batch["normal_unit"],
                normal_valid=batch["normal_valid"].bool(),
                near_label=batch["near_label"].bool(),
            )
            if not bool(torch.isfinite(losses["total"]).item()):
                raise RuntimeError("H_REP_P0_NONFINITE_LOSS:" + arm)
            losses["total"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(models[arm].parameters(), 1.0)
            if not bool(torch.isfinite(torch.as_tensor(grad_norm)).item()):
                raise RuntimeError("H_REP_P0_NONFINITE_GRAD:" + arm)
            optimizers[arm].step()
            logs[arm] = float(losses["total"].detach().item())

        done = step + 1
        if done % 20 == 0 or done in (1, TOTAL_STEPS):
            print(
                "H_REP_P0_STEP",
                f"step={done}/{TOTAL_STEPS}",
                f"subject={subject}",
                f"lr={lr:.8g}",
                *(f"{arm}={logs[arm]:.6f}" for arm in PROBE_ARMS_V1),
                flush=True,
            )
        if done % 200 == 0 or done == TOTAL_STEPS:
            checkpoint = {
                "schema": CHECKPOINT_SCHEMA,
                "completed_steps": done,
                **authority,
                "parameter_counts": parameter_counts,
                "models": {arm: models[arm].state_dict() for arm in PROBE_ARMS_V1},
                "optimizers": {
                    arm: optimizers[arm].state_dict() for arm in PROBE_ARMS_V1
                },
            }
            _atomic_torch(checkpoint, local)
            tmp = remote.with_suffix(".pt.tmp")
            shutil.copy2(local, tmp)
            os.replace(tmp, remote)
            pointer = {
                "schema": "RealSaS.HRepP0CheckpointPointer.v1",
                "completed_steps": done,
                "checkpoint_sha256": sha256_file(remote),
                **authority,
            }
            _atomic_json(pointer, mirror / "LATEST.json")
            print(
                "H_REP_P0_CHECKPOINT", done, pointer["checkpoint_sha256"][:16], flush=True
            )

    result = {
        "schema": SCHEMA,
        "status": "COMPLETED_H_REP_P0_EQUAL_CAPACITY_PROBES__AWAITING_INDEPENDENT_COURT",
        **authority,
        "gpu": gpu,
        "optimizer_steps_per_arm": TOTAL_STEPS,
        "arms": list(PROBE_ARMS_V1),
        "parameter_counts": parameter_counts,
        "checkpoint_file": local.name,
        "checkpoint_sha256": sha256_file(local),
        "elapsed_seconds": time.perf_counter() - started,
        "teacher_truth_in_probe_input": False,
        "mesh_extraction_invoked": False,
        "tessa_invoked": False,
        "scientific_pass_claimed": False,
        "next_action": "RUN_INDEPENDENT_H_REP_P0_HELDOUT_COURT",
    }
    _atomic_json(result, output / "H_REP_P0_TRAINING_RESULT.json")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument("--binding", default="canonical/H_REP_P0_INPUT_BINDING_V1.json")
    parser.add_argument(
        "--preregistration", default="canonical/H_REP_P0_PREREGISTRATION_V1.json"
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--dataset-receipt", required=True)
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--prepared-manifest", required=True)
    parser.add_argument("--v8-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-mirror-dir", required=True)
    parser.add_argument("--query-chunk", type=int, default=8192)
    train(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
