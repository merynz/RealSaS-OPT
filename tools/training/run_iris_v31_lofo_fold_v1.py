#!/usr/bin/env python3
"""External GPU worker for one exact IRIS V3.1 FIT8 LOFO fold.

The worker emits the same exact-hash receipt vocabulary used by model-fit external
jobs, but this is a research-fold receipt rather than product model promotion.
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

from models.iris.v31.lofo_trainer_v1 import run_fold
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
    fold = folds[0]
    source_files = [
        ROOT / "models/iris/v31/model_v31.py",
        ROOT / "models/iris/v31/presentation_objective_v1.py",
        ROOT / "models/iris/v31/lofo_trainer_v1.py",
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
    started = time.monotonic()
    result = run_fold(
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
    receipt = {
        "schema": "RealSaS.IRISV31LofoExecutionReceipt.v1",
        "lane": "IRIS",
        "status": "PASS",
        "executor_kind": "EXTERNAL_GPU_JOB_V1",
        "architecture_id": ARCHITECTURE_ID,
        "fold_id": parsed.fold_id,
        "train_subjects": fold["train_subjects"],
        "heldout_subject": fold["heldout_subject"],
        "preregistration_binding_hash": preregistration_binding_hash,
        "model_source_sha256": source_hashes["models/iris/v31/model_v31.py"],
        "runtime_source_sha256": source_hashes["models/iris/v31/lofo_trainer_v1.py"],
        "upstream_bindings": {
            "lofo_input_manifest": file_sha256(input_path),
            "lofo_recipe": file_sha256(recipe_path),
        },
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "result_sha256": file_sha256(result_path),
        "teacher_inference_inputs_used": False,
        "heldout_used_for_gradient": False,
        "heldout_used_for_checkpoint_selection": False,
        "semantic_part_taxonomy_used": False,
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
