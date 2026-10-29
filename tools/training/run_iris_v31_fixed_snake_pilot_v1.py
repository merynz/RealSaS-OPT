#!/usr/bin/env python3
"""Run the fixed-Snake A100 capacity pilot for IRIS V3.1."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys
import time

import numpy as np
import torch

from models.iris.training.cohort_v1 import file_hash
from models.iris.v31.fixed_snake_capacity_pilot_v1 import (
    ARCHITECTURE_ID,
    FIXED_FOLD_ID,
    FIXED_HELDOUT,
    TRAINER_REVISION,
    IrisV31FixedSnakeCapacityPilotTrainer,
    run_fixed_snake_pilot,
)
from models.training_runtime.durable_state_v1 import file_sha256

ROOT = Path(__file__).resolve().parents[2]


def _canonical_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _runtime() -> dict:
    gpu = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu = {
            "name": props.name,
            "total_memory_bytes": int(props.total_memory),
            "compute_capability": [int(props.major), int(props.minor)],
        }
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu0": gpu,
        "sys_executable": sys.executable,
    }


def _fmt(value) -> str:
    return "NA" if value is None else f"{float(value):.6g}"


def _worst(rows: list[dict], getter) -> float | None:
    values = [getter(row) for row in rows]
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return max(values) if values else None


def _install_gate_telemetry(output: Path) -> None:
    original = IrisV31FixedSnakeCapacityPilotTrainer._train_validation

    def wrapped(self):
        validation = original(self)
        rows = list(validation["subjects"])
        geometry_sign = _worst(rows, lambda row: row["geometry"].get("sign_disagreement_fraction"))
        geometry_mae = _worst(rows, lambda row: row["geometry"].get("mae"))
        geometry_p95 = _worst(rows, lambda row: row["geometry"].get("p95_abs"))
        geometry_near = _worst(rows, lambda row: row["geometry"].get("near_zero_mae"))
        mechanics = _worst(rows, lambda row: row.get("presentation_losses", {}).get("mechanics"))
        rgba = _worst(rows, lambda row: row.get("presentation_losses", {}).get("appearance"))
        worst_row = max(rows, key=lambda row: float(row["score"]))
        lr = float(self.optimizer.param_groups[0]["lr"])
        peak_gib = float(torch.cuda.max_memory_allocated(self.device) / (2 ** 30))
        telemetry = {
            "schema": "RealSaS.IRISV31FixedSnakeCapacityPilotTelemetry.v1",
            "fold_id": FIXED_FOLD_ID,
            "heldout_subject": FIXED_HELDOUT,
            "step": int(validation["step"]),
            "stage": self._stage(max(0, self.step - 1))["id"],
            "validation_rows_fixed_across_checkpoints": True,
            "worst_selector_subject": worst_row["subject_id"],
            "worst_subject_score": float(validation["worst_subject_score"]),
            "mean_subject_score": float(validation["mean_subject_score"]),
            "geometry": {
                "worst_sign_disagreement_fraction": geometry_sign,
                "worst_mae": geometry_mae,
                "worst_p95_abs": geometry_p95,
                "worst_near_zero_mae": geometry_near,
            },
            "presentation": {
                "mechanics_log1p_gaussian_nll": mechanics,
                "worst_rgba_smoothl1": rgba,
                "pair_family": "ABSTAIN",
            },
            "learning_rate": lr,
            "peak_cuda_allocated_gib": peak_gib,
            "heldout_visible": False,
            "lofo_expansion_authorized": False,
            "fit8_pass_claimed": False,
        }
        path = output / "LATEST_TRAIN_GATE_TELEMETRY.json"
        temp = path.with_suffix(path.suffix + ".partial")
        temp.write_text(json.dumps(telemetry, sort_keys=True, indent=2, allow_nan=False) + "\n")
        temp.replace(path)
        print(
            "[GATE_PILOT] "
            f"step={validation['step']} stage={telemetry['stage']} "
            f"worst_subject={worst_row['subject_id']} "
            f"sign_err={_fmt(geometry_sign)} mae={_fmt(geometry_mae)} "
            f"p95_abs={_fmt(geometry_p95)} near0_mae={_fmt(geometry_near)} "
            f"mech_log1p_nll={_fmt(mechanics)} rgba_l1={_fmt(rgba)} "
            f"lr={lr:.3e} peak_vram_gib={peak_gib:.2f} heldout=SEALED",
            flush=True,
        )
        return validation

    IrisV31FixedSnakeCapacityPilotTrainer._train_validation = wrapped


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input-manifest", "recipe", "checkpoint-root", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cuda")
    parsed = parser.parse_args(args)

    input_path = Path(parsed.input_manifest).resolve()
    recipe_path = Path(parsed.recipe).resolve()
    output = Path(parsed.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(input_path.read_text())
    recipe = json.loads(recipe_path.read_text())

    fold = [row for row in manifest.get("folds") or () if row.get("fold_id") == FIXED_FOLD_ID]
    if len(fold) != 1 or fold[0].get("heldout_subject") != FIXED_HELDOUT:
        raise SystemExit("IRIS_CAPACITY_PILOT_FIXED_FOLD_BINDING")
    if recipe.get("trainer_revision") != TRAINER_REVISION:
        raise SystemExit("IRIS_CAPACITY_PILOT_TRAINER_REVISION")

    source_files = [
        ROOT / "models/iris/v31/model_v31.py",
        ROOT / "models/iris/v31/presentation_objective_v1.py",
        ROOT / "models/iris/v31/lofo_trainer_v1.py",
        ROOT / "models/iris/v31/lofo_trainer_v2.py",
        ROOT / "models/iris/v31/lofo_trainer_v3.py",
        ROOT / "models/iris/v31/fstar_teacher_v31.py",
        ROOT / "models/iris/v31/fixed_snake_capacity_pilot_v1.py",
        ROOT / "models/iris/training/apparatus_v1.py",
        ROOT / "models/iris/v5/source_signed_field_v5.py",
        Path(__file__).resolve(),
    ]
    source_hashes = {str(p.relative_to(ROOT)): file_sha256(p) for p in source_files}
    binding = {
        "schema": "RealSaS.IRISV31FixedSnakeCapacityPilotExecutionBinding.v1",
        "architecture_id": ARCHITECTURE_ID,
        "input_manifest_sha256": file_sha256(input_path),
        "recipe_sha256": file_sha256(recipe_path),
        "fold": fold[0],
        "source_hashes": source_hashes,
    }
    preregistration_binding_hash = _canonical_hash(binding)

    _install_gate_telemetry(output)
    started = time.monotonic()
    result = run_fixed_snake_pilot(
        input_manifest_path=input_path,
        recipe_path=recipe_path,
        checkpoint_root=parsed.checkpoint_root,
        output_root=output,
        device=parsed.device,
    )
    if result.get("status") != "PASS_EXECUTED_FIXED_SNAKE_CAPACITY_PILOT__LOFO_EXPANSION_SEPARATE":
        raise SystemExit("IRIS_CAPACITY_PILOT_RESULT_NOT_PASS")

    checkpoint_path = output / "selected_checkpoint.pt"
    result_path = output / "result.json"
    held = result["heldout"]
    print(
        "[GATE_HELDOUT_PILOT] "
        f"subject={FIXED_HELDOUT} selected_step={result['selected_step']} "
        f"sign_err={_fmt(held['geometry'].get('sign_disagreement_fraction'))} "
        f"mae={_fmt(held['geometry'].get('mae'))} "
        f"p95_abs={_fmt(held['geometry'].get('p95_abs'))} "
        f"near0_mae={_fmt(held['geometry'].get('near_zero_mae'))} "
        f"mech_log1p_nll={_fmt((held.get('presentation_losses') or {}).get('mechanics'))} "
        f"rgba_l1={_fmt((held.get('presentation_losses') or {}).get('appearance'))} "
        "lofo_expansion=NOT_AUTHORIZED",
        flush=True,
    )

    receipt = {
        "schema": "RealSaS.IRISV31FixedSnakeCapacityPilotExecutionReceipt.v1",
        "lane": "IRIS",
        "status": "PASS",
        "architecture_id": ARCHITECTURE_ID,
        "trainer_revision": TRAINER_REVISION,
        "fold_id": FIXED_FOLD_ID,
        "train_subjects": fold[0]["train_subjects"],
        "heldout_subject": FIXED_HELDOUT,
        "preregistration_binding_hash": preregistration_binding_hash,
        "source_hashes": source_hashes,
        "upstream_bindings": {
            "lofo_input_manifest": file_sha256(input_path),
            "pilot_recipe": file_sha256(recipe_path),
        },
        "checkpoint_sha256": file_hash(checkpoint_path),
        "result_sha256": file_hash(result_path),
        "teacher_inference_inputs_used": False,
        "heldout_used_for_gradient": False,
        "heldout_used_for_checkpoint_selection": False,
        "mechanics_target_transform": recipe["mechanics_target_transform"],
        "pair_training_authorized": False,
        "lofo_expansion_authorized": False,
        "fit8_pass_claimed": False,
        "product_pass_claimed": False,
        "runtime_environment": _runtime(),
        "wall_seconds": time.monotonic() - started,
    }
    receipt_path = output / "execution_receipt.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps({
            "checkpoint_sha256": receipt["checkpoint_sha256"],
            "result_sha256": receipt["result_sha256"],
            "receipt_sha256": file_hash(receipt_path),
        }, sort_keys=True),
        flush=True,
    )
    return receipt


if __name__ == "__main__":
    main()
