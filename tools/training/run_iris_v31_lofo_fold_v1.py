#!/usr/bin/env python3
"""External GPU worker for one exact IRIS V3.1 FIT8 LOFO fold.

The worker emits the same exact-hash receipt vocabulary used by model-fit external
jobs, but this is a research-fold receipt rather than product model promotion.
It also publishes gate-focused telemetry without changing optimizer semantics.
"""
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

from models.iris.v31.lofo_trainer_v3 import IrisV31LofoTrainerV3, TRAINER_REVISION_V3, run_fold_v3
from models.training_runtime.durable_state_v1 import file_sha256

ROOT = Path(__file__).resolve().parents[2]
ARCHITECTURE_ID = "RealSaS.IRIS.DenseObservationEvidence.v3_1"


def _canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _runtime() -> dict:
    gpu = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu = {"name": props.name, "total_memory_bytes": int(props.total_memory), "compute_capability": [int(props.major), int(props.minor)]}
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
    values = [float(value) for value in values if value is not None and np.isfinite(value)]
    return max(values) if values else None


def _install_gate_telemetry(output: Path) -> None:
    """Wrap fixed train-subject validation with read-only scientific telemetry."""
    original = IrisV31LofoTrainerV3._train_validation

    def wrapped(self):
        validation = original(self)
        rows = list(validation["subjects"])
        geometry_sign = _worst(rows, lambda row: row["geometry"].get("sign_disagreement_fraction"))
        geometry_mae = _worst(rows, lambda row: row["geometry"].get("mae"))
        geometry_p95 = _worst(rows, lambda row: row["geometry"].get("p95_abs"))
        geometry_near = _worst(rows, lambda row: row["geometry"].get("near_zero_mae"))
        mechanics = _worst(rows, lambda row: row.get("presentation_losses", {}).get("mechanics"))
        rgba = _worst(rows, lambda row: row.get("presentation_losses", {}).get("appearance"))
        occlusion = _worst(rows, lambda row: row.get("presentation_losses", {}).get("occlusion_attenuation"))
        worst_row = max(rows, key=lambda row: float(row["score"]))
        lr = float(self.optimizer.param_groups[0]["lr"])
        peak_gib = float(torch.cuda.max_memory_allocated(self.device) / (2 ** 30))
        telemetry = {
            "schema": "RealSaS.IRISV31LofoGateTelemetry.v1",
            "fold_id": self.fold["fold_id"],
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
                "direction": "LOWER_IS_BETTER__SIGN_CLEAN_IS_HISTORICAL_SELECTOR_PRIORITY",
                "numeric_v31_pass_threshold": None,
            },
            "presentation": {
                "worst_mechanics_nll": mechanics,
                "worst_rgba_smoothl1": rgba,
                "worst_occlusion_bce": occlusion,
                "occlusion_role": "ABSTAIN_OR_MEASURED_PER_EXACT_PROJECTION_QUALIFICATION",
                "direction": "LOWER_IS_BETTER",
                "numeric_v31_pass_threshold": None,
            },
            "learning_rate": lr,
            "peak_cuda_allocated_gib": peak_gib,
            "heldout_visible": False,
            "fit8_pass_claimed": False,
        }
        path = output / "LATEST_TRAIN_GATE_TELEMETRY.json"
        temp = path.with_suffix(path.suffix + ".partial")
        temp.write_text(json.dumps(telemetry, sort_keys=True, indent=2, allow_nan=False) + "\n")
        temp.replace(path)
        print(
            "[GATE_TRAIN] "
            f"fold={self.fold['fold_id']} step={validation['step']} stage={telemetry['stage']} "
            f"worst_subject={worst_row['subject_id']} "
            f"sign_err={_fmt(geometry_sign)} mae={_fmt(geometry_mae)} "
            f"p95_abs={_fmt(geometry_p95)} near0_mae={_fmt(geometry_near)} "
            f"mech_nll={_fmt(mechanics)} rgba_l1={_fmt(rgba)} occ_bce={_fmt(occlusion)} "
            f"lr={lr:.3e} peak_vram_gib={peak_gib:.2f} "
            "validation=fixed presentation_threshold=UNSET",
            flush=True,
        )
        return validation

    IrisV31LofoTrainerV3._train_validation = wrapped


def _print_heldout_gate(result: dict) -> None:
    held = result["heldout"]
    geometry = held["geometry"]
    presentation = held.get("presentation_losses") or {}
    print(
        "[GATE_HELDOUT] "
        f"fold={result['fold_id']} subject={result['heldout_subject']} selected_step={result['selected_step']} "
        f"sign_err={_fmt(geometry.get('sign_disagreement_fraction'))} "
        f"mae={_fmt(geometry.get('mae'))} p95_abs={_fmt(geometry.get('p95_abs'))} "
        f"near0_mae={_fmt(geometry.get('near_zero_mae'))} "
        f"mech_nll={_fmt(presentation.get('mechanics'))} "
        f"rgba_l1={_fmt(presentation.get('appearance'))} "
        f"occ_bce={_fmt(presentation.get('occlusion_attenuation'))} "
        "presentation_threshold=UNSET scientific_qualification=SEPARATE",
        flush=True,
    )


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input-manifest", "recipe", "fold-id", "checkpoint-root", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cuda")
    parsed = parser.parse_args(args)
    input_path = Path(parsed.input_manifest).resolve()
    recipe_path = Path(parsed.recipe).resolve()
    output = Path(parsed.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(input_path.read_text())
    recipe = json.loads(recipe_path.read_text())
    folds = [row for row in manifest.get("folds") or () if row.get("fold_id") == parsed.fold_id]
    if len(folds) != 1:
        raise SystemExit("IRIS_V31_LOFO_WORKER_FOLD_BINDING")
    if recipe.get("trainer_revision") != TRAINER_REVISION_V3:
        raise SystemExit("IRIS_V31_LOFO_WORKER_TRAINER_REVISION")
    fold = folds[0]
    source_files = [
        ROOT / "models/iris/v31/model_v31.py",
        ROOT / "models/iris/v31/presentation_objective_v1.py",
        ROOT / "models/iris/v31/lofo_trainer_v1.py",
        ROOT / "models/iris/v31/lofo_trainer_v2.py",
        ROOT / "models/iris/v31/lofo_trainer_v3.py",
        Path(__file__).resolve(),
    ]
    source_hashes = {str(path.relative_to(ROOT)): file_sha256(path) for path in source_files}
    binding = {
        "schema": "RealSaS.IRISV31LofoExecutionBinding.v1",
        "architecture_id": ARCHITECTURE_ID,
        "input_manifest_sha256": file_sha256(input_path),
        "recipe_sha256": file_sha256(recipe_path),
        "fold": fold,
        "source_hashes": source_hashes,
    }
    preregistration_binding_hash = _canonical_hash(binding)
    _install_gate_telemetry(output)
    started = time.monotonic()
    result = run_fold_v3(
        input_manifest_path=input_path,
        recipe_path=recipe_path,
        fold_id=parsed.fold_id,
        checkpoint_root=parsed.checkpoint_root,
        output_root=output,
        device=parsed.device,
    )
    result_path = output / "result.json"
    checkpoint_path = output / "selected_checkpoint.pt"
    if result.get("status") != "PASS_EXECUTED_LOFO_FOLD__SCIENTIFIC_QUALIFICATION_SEPARATE":
        raise SystemExit("IRIS_V31_LOFO_WORKER_RESULT_NOT_PASS")
    _print_heldout_gate(result)
    receipt = {
        "schema": "RealSaS.IRISV31LofoExecutionReceipt.v1",
        "lane": "IRIS",
        "status": "PASS",
        "executor_kind": "EXTERNAL_GPU_JOB_V1",
        "architecture_id": ARCHITECTURE_ID,
        "trainer_revision": recipe["trainer_revision"],
        "fold_id": parsed.fold_id,
        "train_subjects": fold["train_subjects"],
        "heldout_subject": fold["heldout_subject"],
        "preregistration_binding_hash": preregistration_binding_hash,
        "model_source_sha256": source_hashes["models/iris/v31/model_v31.py"],
        "runtime_source_sha256": source_hashes["models/iris/v31/lofo_trainer_v3.py"],
        "upstream_bindings": {
            "lofo_input_manifest": file_sha256(input_path),
            "lofo_recipe": file_sha256(recipe_path),
        },
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "result_sha256": file_sha256(result_path),
        "teacher_inference_inputs_used": False,
        "heldout_used_for_gradient": False,
        "heldout_used_for_checkpoint_selection": False,
        "validation_rows_fixed_across_checkpoints": True,
        "semantic_part_taxonomy_used": False,
        "pair_training_authorized": bool((manifest.get("student_projection_qualification") or {}).get("pair_training_authorized")),
        "fit8_pass_claimed": False,
        "product_pass_claimed": False,
        "runtime_environment": _runtime(),
        "wall_seconds": time.monotonic() - started,
    }
    receipt_path = output / "execution_receipt.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"fold_id": parsed.fold_id, "checkpoint_sha256": receipt["checkpoint_sha256"], "result_sha256": receipt["result_sha256"], "receipt_sha256": file_sha256(receipt_path)}, sort_keys=True), flush=True)
    return receipt


if __name__ == "__main__":
    main()
