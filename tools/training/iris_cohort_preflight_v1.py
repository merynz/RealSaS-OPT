"""Real source/model/optimizer prefix; honest CPU or on-A100 runtime evidence."""
from __future__ import annotations
import copy
import gc
import json
from pathlib import Path

import torch

from models.iris.training.cohort_v1 import (
    SharedIrisTrainer,
    source_identity,
    atomic_json,
)

ROOT = Path(__file__).resolve().parents[2]


def _preflight_stages(policy):
    # Exercise direct -> C -> low-LR C transitions and optimizer restarts without
    # pretending this short prefix reproduces the production horizon.
    base = {row["id"]: row for row in policy["training_stages"]}
    result = []
    for stage_id, steps in (
        ("DIRECT_FSTAR_COLD", 1),
        ("C_OBJECTIVE", 1),
        ("C_LONG_HORIZON", 2),
    ):
        row = copy.deepcopy(base[stage_id])
        row.update(steps=steps, evaluation_interval=1, checkpoint_interval=1)
        result.append(row)
    return result


def preflight(data_root, policy, checkpoint_root, output_root, device="cuda"):
    data_root = Path(data_root)
    output_root = Path(output_root)
    manifest = json.loads((data_root / "COHORT_MANIFEST.json").read_text())
    p = copy.deepcopy(policy)
    p.update(
        steps=4,
        log_interval=1,
        training_stages=_preflight_stages(policy),
        scope="RUNTIME_PREFLIGHT__NOT_PRODUCTION_FIT",
    )
    if str(device) == "cpu":
        p.update(
            queries_per_step=512,
            silhouette_rays_per_step=32,
            engineering_eval_query_limit=2048,
        )
    source = source_identity(ROOT)
    trainer = SharedIrisTrainer(
        data_root, manifest, p, source, checkpoint_root, output_root, device
    )
    if trainer.step == 4:
        old = output_root / "RUNTIME_PREFLIGHT.json"
        if old.exists():
            receipt = json.loads(old.read_text())
            if receipt["binding_sha256"] == trainer.binding_hash:
                print("[PREFLIGHT] exact completed runtime prefix reused", flush=True)
                return receipt
        raise RuntimeError("PREFLIGHT_COMPLETE_WITHOUT_BOUND_RECEIPT")

    if trainer.step < 2:
        trainer.run(stop_after=2)
    if trainer.step == 2 and trainer.counts != [2] * 8:
        raise RuntimeError("PREFLIGHT_NOT_ALL_EIGHT_TRAINED_EVERY_GLOBAL_STEP")
    expected = {k: v.detach().cpu().clone() for k, v in trainer.model.state_dict().items()}
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    resumed = SharedIrisTrainer(
        data_root, manifest, p, source, checkpoint_root, output_root, device
    )
    for key, value in expected.items():
        if not torch.equal(value, resumed.model.state_dict()[key].cpu()):
            raise RuntimeError("PREFLIGHT_RESTORE_WEIGHTS_DRIFT")
    result = resumed.run()
    if resumed.counts != [4] * 8 or resumed.step != 4:
        raise RuntimeError("PREFLIGHT_RESUME_PROGRESS_DRIFT")
    if len(result.get("training_stage_receipts") or []) != 3:
        raise RuntimeError("PREFLIGHT_STAGE_TRANSITION_DRIFT")

    receipt = {
        "schema": "RealSaS.SharedIrisRuntimePreflight.v2",
        "status": "PASS",
        "device": str(device),
        "gpu_name": (
            torch.cuda.get_device_name() if str(device).startswith("cuda") else None
        ),
        "a100_executed": str(device).startswith("cuda")
        and "A100" in torch.cuda.get_device_name(),
        "production_architecture": p["model_config"],
        "trainable_parameters": sum(v.numel() for v in resumed.model.parameters()),
        "actual_optimizer_steps": 4,
        "subject_exposures": result["subject_exposures"],
        "total_subject_exposures": result["total_subject_exposures"],
        "optimizer_update_semantics": p["optimizer_update_semantics"],
        "weights_exactly_restored": True,
        "all_eight_evaluated": True,
        "all_three_training_stages_exercised": True,
        "checkpoint_payload_sha256": json.loads(
            (Path(checkpoint_root) / "LATEST.json").read_text()
        )["sha256"],
        "binding_sha256": resumed.binding_hash,
        "cpu_exact_resume_test": "tests/iris/test_shared_cohort_v1.py",
        "fit8_pass_claimed": False,
        "go_qualification_claimed": False,
    }
    atomic_json(output_root / "RUNTIME_PREFLIGHT.json", receipt)
    del resumed
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return receipt
