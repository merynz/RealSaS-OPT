#!/usr/bin/env python3
from __future__ import annotations

"""Full-reference zero-optimizer A100 preflight for Structural-Evidence A/B.

Executes one real BF16 forward/backward for each matched TESSA V3 arm and
materializes AdamW state without calling optimizer.step().  It is engineering
admission only and cannot mint scientific PASS.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

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
    _load_presentation_subject,
    _typed_presentation,
    load_json,
    validate_preregistration,
)

SCHEMA = "RealSaS.StructuralEvidenceABA100Preflight.v1"


def host_memory_bytes() -> int:
    try:
        return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        return 0


def _adam_state_without_step(optimizer: torch.optim.Optimizer) -> None:
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            state = optimizer.state[parameter]
            state["step"] = torch.zeros((), dtype=torch.float32, device=parameter.device)
            state["exp_avg"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
            state["exp_avg_sq"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)


def _probe_arm(*, arm_id: str, subject: dict[str, Any], presentation_payload: dict[str, Any],
               window_index: int, protocol: dict[str, Any], initial_state: dict[str, torch.Tensor],
               device: torch.device) -> dict[str, Any]:
    seed = int(protocol["seed_set"][0])
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model = IRISVNextStructuralizationBridgeV4().to(device=device, dtype=torch.float32)
    model.load_state_dict(initial_state, strict=True)
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
    scene, xyz, normal, h, mechanical, inputs, labels, offsets = batch_from_subject(subject, window_index, device)
    presentation = None if arm_id == ARM_A else _typed_presentation(presentation_payload, device)
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
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
            objective_weights=TESSAObjectiveWeightsV2(),
        )
        loss = output.loss
    if loss is None or not bool(torch.isfinite(loss.detach()).item()):
        raise RuntimeError("STRUCTURAL_EVIDENCE_AB_PREFLIGHT_NONFINITE_LOSS:" + arm_id)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, float(opt_cfg["gradient_clip_norm"]))
    if not bool(torch.isfinite(torch.as_tensor(grad_norm)).item()):
        raise RuntimeError("STRUCTURAL_EVIDENCE_AB_PREFLIGHT_NONFINITE_GRADIENT:" + arm_id)
    _adam_state_without_step(optimizer)
    torch.cuda.synchronize()
    result = {
        "arm_id": arm_id,
        "loss": float(loss.detach().float().cpu()),
        "token_cross_entropy": float(output.objective.token_cross_entropy.detach().float().cpu()),
        "coordinate_metric": float(output.objective.coordinate_metric.detach().float().cpu()),
        "structural_special_token": float(output.objective.structural_special_token.detach().float().cpu()),
        "gradient_norm_after_backward_before_update": float(torch.as_tensor(grad_norm).detach().float().cpu()),
        "conditioning_memory_tokens": int(output.conditioning_memory.shape[1]),
        "forward_backward_adam_state_seconds": float(time.perf_counter() - started),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "optimizer_steps_executed": 0,
        "parameters_updated": False,
    }
    del model, optimizer, trainable, scene, xyz, normal, h, mechanical, inputs, labels, offsets, presentation, output, loss
    torch.cuda.empty_cache()
    return result


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    expected_source = str(args.expected_source_sha).lower()
    if git_head(repo_root) != expected_source:
        raise RuntimeError("STRUCTURAL_EVIDENCE_AB_PREFLIGHT_CHECKOUT_SHA_DRIFT")
    if not torch.cuda.is_available():
        raise RuntimeError("STRUCTURAL_EVIDENCE_AB_PREFLIGHT_CUDA_REQUIRED")
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    cc = torch.cuda.get_device_capability(device)
    if "A100" not in str(props.name).upper():
        raise RuntimeError("STRUCTURAL_EVIDENCE_AB_PREFLIGHT_A100_REQUIRED:" + str(props.name))

    prereg_path = Path(args.preregistration).resolve()
    prereg = load_json(prereg_path)
    protocol = validate_preregistration(prereg)

    corpus_root = Path(args.corpus_root).resolve()
    corpus_manifest_path = corpus_root / "TESSA_V2_CORPUS_MANIFEST.json"
    corpus_manifest = load_json(corpus_manifest_path)
    rows = list(corpus_manifest.get("subjects") or ())
    if corpus_manifest.get("schema") != "RealSaS.Fit8TESSAV2TrainingCorpus.v1" or tuple(row.get("subject_id") for row in rows) != SUBJECT_IDS:
        raise ValueError("STRUCTURAL_EVIDENCE_AB_PREFLIGHT_CORPUS_INVALID")
    row = rows[0]
    subject = load_subject(corpus_root / str(row["file"]), str(row["sha256"]))
    active = subject["topology_input_ids"].ne(1031).sum(dim=1)
    window_index = int(torch.argmax(active).item())

    presentation_root = Path(args.presentation_root).resolve()
    presentation_manifest_path = presentation_root / "DIRECT_IRIS_PRESENTATION_MEMORY_V1_MANIFEST.json"
    presentation_manifest = load_json(presentation_manifest_path)
    prows = list(presentation_manifest.get("subjects") or ())
    if presentation_manifest.get("schema") != "RealSaS.DirectIRISPresentationMemoryManifest.v1" or tuple(r.get("subject_id") for r in prows) != SUBJECT_IDS:
        raise ValueError("STRUCTURAL_EVIDENCE_AB_PREFLIGHT_PRESENTATION_MANIFEST_INVALID")
    presentation_payload = _load_presentation_subject(
        presentation_root / str(prows[0]["file"]), str(prows[0]["sha256"]), SUBJECT_IDS[0]
    )

    seed = int(protocol["seed_set"][0])
    torch.manual_seed(seed)
    initial_model = IRISVNextStructuralizationBridgeV4().to(device="cpu", dtype=torch.float32)
    initial_state = {k: v.detach().clone() for k, v in initial_model.state_dict().items()}
    del initial_model

    probes = [
        _probe_arm(arm_id=arm, subject=subject, presentation_payload=presentation_payload,
                   window_index=window_index, protocol=protocol, initial_state=initial_state, device=device)
        for arm in (ARM_A, ARM_B)
    ]
    if any(int(row["conditioning_memory_tokens"]) != 1024 for row in probes):
        raise RuntimeError("STRUCTURAL_EVIDENCE_AB_PREFLIGHT_MEMORY_BUDGET_DRIFT")

    output_dir = Path(args.output_dir).resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(output_dir)
    blockers: list[str] = []
    if int(props.total_memory) < 39 * 1024**3:
        blockers.append("CUDA_MEMORY_BELOW_LOCKED_PRECEDENT")
    if host_memory_bytes() < 24 * 1024**3:
        blockers.append("HOST_MEMORY_BELOW_LOCKED_PRECEDENT")
    if int(disk.free) < 12 * 1024**3:
        blockers.append("FREE_DISK_BELOW_LOCKED_PRECEDENT")
    if int(cc[0]) < 8:
        blockers.append("COMPUTE_CAPABILITY_BELOW_LOCKED_PRECEDENT")

    result = {
        "schema": SCHEMA,
        "status": "PASS__ENGINEERING_ONLY__ZERO_OPTIMIZER_STEPS" if not blockers else "BLOCKED__ENGINEERING_RESOURCES",
        "execution_source_sha": expected_source,
        "preregistration_sha256": sha256_file(prereg_path),
        "corpus_manifest_sha256": sha256_file(corpus_manifest_path),
        "presentation_manifest_sha256": sha256_file(presentation_manifest_path),
        "probe_subject": SUBJECT_IDS[0],
        "probe_window_index": window_index,
        "probe_active_context_tokens": int(active[window_index].item()),
        "arms": probes,
        "matched_conditioning_token_budget": 1024,
        "accelerator": {"name": props.name, "total_memory_bytes": int(props.total_memory), "compute_capability": [int(cc[0]), int(cc[1])]},
        "host_memory_bytes": host_memory_bytes(),
        "free_disk_bytes": int(disk.free),
        "optimizer_steps_executed": 0,
        "parameters_updated": False,
        "scientific_quality_evaluated": False,
        "scientific_pass_claimed": False,
        "training_authorized_by_engineering_preflight": not blockers,
        "blockers": blockers,
        "next_action": "USER_AUTHORIZED_LOCKED_AB_TRAINING" if not blockers else "RESOLVE_ENGINEERING_BLOCKERS",
    }
    (output_dir / "STRUCTURAL_EVIDENCE_AB_A100_PREFLIGHT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", required=True)
    p.add_argument("--expected-source-sha", required=True)
    p.add_argument("--preregistration", default="canonical/STRUCTURAL_EVIDENCE_AB_PREREGISTRATION_V2.json")
    p.add_argument("--corpus-root", required=True)
    p.add_argument("--presentation-root", required=True)
    p.add_argument("--output-dir", required=True)
    result = preflight(p.parse_args())
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["status"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
