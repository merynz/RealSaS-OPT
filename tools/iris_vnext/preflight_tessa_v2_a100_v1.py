#!/usr/bin/env python3
from __future__ import annotations

"""Full-reference TESSA V2 A100 training-resource preflight.

Consumes one real materialized corpus window, executes one BF16 forward/backward,
and allocates AdamW state without changing a parameter. This is engineering
admission only: optimizer_steps_executed remains zero and no scientific quality
claim is minted.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

import torch

from models.iris.vnext.structure_global_v3 import IRISVNextGlobalStructuralizationBridgeV3
from models.tessa.v2.experiment_v2 import load_preregistration_v1, sha256_file
from models.tessa.v2.objective_v2 import TESSAObjectiveWeightsV2


SCHEMA = "RealSaS.TESSAV2A100TrainingPreflight.v1"


def host_memory_bytes() -> int:
    try:
        return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        return 0


def load_subject(path: Path, expected_sha: str) -> dict[str, Any]:
    if sha256_file(path) != expected_sha:
        raise ValueError("TESSA_V2_PREFLIGHT_SUBJECT_SHA_DRIFT")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value.get("schema") != "RealSaS.TESSAV2TrainingSubject.v1":
        raise ValueError("TESSA_V2_PREFLIGHT_SUBJECT_SCHEMA_INVALID")
    if value.get("teacher_topology_in_conditioning") is not False:
        raise ValueError("TESSA_V2_PREFLIGHT_TEACHER_CONDITIONING_FORBIDDEN")
    if value.get("teacher_rig_skin_in_conditioning") is not False:
        raise ValueError("TESSA_V2_PREFLIGHT_RIG_SKIN_CONDITIONING_FORBIDDEN")
    if value.get("contact_persistence_valid") is not False:
        raise ValueError("TESSA_V2_PREFLIGHT_CONTACT_MUST_ABSTAIN")
    return value


def batch_from_subject(subject: dict[str, Any], window_index: int, device: torch.device):
    def one(name: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        value = subject[name]
        if not isinstance(value, torch.Tensor):
            raise ValueError("TESSA_V2_PREFLIGHT_TENSOR_MISSING:" + name)
        return value.to(device=device, dtype=dtype, non_blocking=False)

    scene = one("scene_planes")[None]
    xyz = one("witness_xyz")[None]
    normal = one("witness_normal")[None]
    h = one("witness_h")[None]
    mechanical = {
        "stretch_major": one("mechanical_stretch_major")[None],
        "stretch_minor": one("mechanical_stretch_minor")[None],
        "tangent_direction_2theta": one("mechanical_tangent_direction_2theta")[None],
        "bend_demand": one("mechanical_bend_demand")[None],
        "rigidity_probability": one("mechanical_rigidity_probability")[None],
    }
    inputs = one("topology_input_ids", torch.long)[window_index : window_index + 1]
    labels = one("topology_labels", torch.long)[window_index : window_index + 1]
    offsets = one("topology_window_offsets", torch.long)[window_index : window_index + 1]
    return scene, xyz, normal, h, mechanical, inputs, labels, offsets


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    prereg_path = Path(args.preregistration).resolve()
    prereg = load_preregistration_v1(prereg_path)
    if not torch.cuda.is_available():
        raise RuntimeError("TESSA_V2_PREFLIGHT_CUDA_REQUIRED")
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    cc = torch.cuda.get_device_capability(device)
    if "A100" not in str(props.name).upper():
        raise RuntimeError("TESSA_V2_PREFLIGHT_A100_REQUIRED:" + str(props.name))

    corpus_root = Path(args.corpus_root).resolve()
    manifest_path = corpus_root / "TESSA_V2_CORPUS_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "RealSaS.Fit8TESSAV2TrainingCorpus.v1":
        raise ValueError("TESSA_V2_PREFLIGHT_CORPUS_SCHEMA_INVALID")
    rows = list(manifest.get("subjects") or ())
    if len(rows) != 8:
        raise ValueError("TESSA_V2_PREFLIGHT_CORPUS_SUBJECT_COUNT")
    # Use the longest available first-subject window: real 4608-context workload.
    row = rows[0]
    subject = load_subject(corpus_root / str(row["file"]), str(row["sha256"]))
    window_count = int(subject["topology_input_ids"].shape[0])
    active = (subject["topology_input_ids"] != 1031).sum(dim=1)
    window_index = int(torch.argmax(active).item())
    if int(active[window_index].item()) <= 4096:
        raise RuntimeError("TESSA_V2_PREFLIGHT_GLOBAL_HISTORY_WINDOW_NOT_EXERCISED")

    torch.manual_seed(int(prereg["optimizer"]["seed"]))
    torch.cuda.manual_seed_all(int(prereg["optimizer"]["seed"]))
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    model = IRISVNextGlobalStructuralizationBridgeV3().to(device=device, dtype=torch.float32)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(prereg["optimizer"]["learning_rate"]),
        betas=tuple(map(float, prereg["optimizer"]["betas"])),
        eps=float(prereg["optimizer"]["epsilon"]),
        weight_decay=float(prereg["optimizer"]["weight_decay"]),
    )
    scene, xyz, normal, h, mechanical, inputs, labels, offsets = batch_from_subject(
        subject, window_index, device
    )
    weights = TESSAObjectiveWeightsV2()

    started = time.perf_counter()
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
        raise RuntimeError("TESSA_V2_PREFLIGHT_NONFINITE_LOSS")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, float(prereg["optimizer"]["gradient_clip_norm"]))
    if not bool(torch.isfinite(torch.as_tensor(grad_norm)).item()):
        raise RuntimeError("TESSA_V2_PREFLIGHT_NONFINITE_GRADIENT")

    # Adam state is materialized to measure realistic persistent training memory,
    # but no optimizer.step() occurs and no parameter is changed.
    for parameter in trainable:
        if parameter.grad is None:
            continue
        state = optimizer.state[parameter]
        state["step"] = torch.zeros((), dtype=torch.float32, device=parameter.device)
        state["exp_avg"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
        state["exp_avg_sq"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    out_dir = Path(args.output_dir).resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(out_dir)
    runtime = prereg["runtime_admission"]
    blockers: list[str] = []
    if int(props.total_memory) < int(runtime["minimum_cuda_memory_gib"]) * 1024**3:
        blockers.append("CUDA_MEMORY_BELOW_PREREG")
    if host_memory_bytes() < int(runtime["minimum_host_memory_gib"]) * 1024**3:
        blockers.append("HOST_MEMORY_BELOW_PREREG")
    if int(disk.free) < int(runtime["minimum_free_disk_gib"]) * 1024**3:
        blockers.append("FREE_DISK_BELOW_PREREG")
    if int(cc[0]) < int(runtime["compute_capability_major_minimum"]):
        blockers.append("COMPUTE_CAPABILITY_BELOW_PREREG")

    result = {
        "schema": SCHEMA,
        "status": "PASS__ENGINEERING_ONLY__ZERO_OPTIMIZER_STEPS" if not blockers else "BLOCKED__ENGINEERING_RESOURCES",
        "preregistration_sha256": sha256_file(prereg_path),
        "corpus_manifest_sha256": sha256_file(manifest_path),
        "probe_subject": row["subject_id"],
        "probe_window_index": window_index,
        "probe_active_context_tokens": int(active[window_index].item()),
        "loss": float(loss.detach().float().cpu()),
        "token_cross_entropy": float(output.objective.token_cross_entropy.detach().float().cpu()),
        "coordinate_metric": float(output.objective.coordinate_metric.detach().float().cpu()),
        "structural_special_token": float(output.objective.structural_special_token.detach().float().cpu()),
        "gradient_norm_after_backward_before_update": float(torch.as_tensor(grad_norm).detach().float().cpu()),
        "accelerator": {
            "name": props.name,
            "total_memory_bytes": int(props.total_memory),
            "compute_capability": [int(cc[0]), int(cc[1])],
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
        "host_memory_bytes": host_memory_bytes(),
        "free_disk_bytes": int(disk.free),
        "forward_backward_and_adam_state_seconds": float(elapsed),
        "optimizer_steps_executed": 0,
        "parameters_updated": false if False else False,
        "scientific_quality_evaluated": False,
        "scientific_pass_claimed": False,
        "training_authorized": not blockers,
        "blockers": blockers,
        "next_action": "RUN_LOCKED_TESSA_V2_TRAINER" if not blockers else "RESOLVE_ENGINEERING_BLOCKERS",
    }
    path = out_dir / "TESSA_V2_A100_PREFLIGHT.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preregistration", default="canonical/TESSA_V2_FIT8_PREREGISTRATION_V1.json")
    p.add_argument("--corpus-root", required=True)
    p.add_argument("--output-dir", required=True)
    result = preflight(p.parse_args())
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["status"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
