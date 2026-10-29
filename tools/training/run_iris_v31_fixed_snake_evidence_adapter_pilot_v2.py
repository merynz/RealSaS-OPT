#!/usr/bin/env python3
"""External GPU worker for the protected fixed-Snake evidence-adapter pilot."""
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

from models.iris.v31.fixed_snake_evidence_adapter_pilot_v2 import (
    ARCHITECTURE_ID_V2,
    FIXED_FOLD_ID,
    TRAINER_REVISION_V2,
    run_fixed_snake_evidence_adapter_pilot_v2,
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

    if recipe.get("trainer_revision") != TRAINER_REVISION_V2:
        raise SystemExit("IRIS_EA_V2_WORKER_TRAINER_REVISION")
    folds = [row for row in manifest.get("folds") or () if row.get("fold_id") == FIXED_FOLD_ID]
    if len(folds) != 1:
        raise SystemExit("IRIS_EA_V2_WORKER_FIXED_FOLD_BINDING")

    source_files = [
        ROOT / "models/iris/v31/model_v31.py",
        ROOT / "models/iris/v31/evidence_adapter_model_v2.py",
        ROOT / "models/iris/v31/fixed_snake_evidence_adapter_pilot_v2.py",
        ROOT / "models/iris/v31/fstar_teacher_v31.py",
        ROOT / "models/iris/v31/lofo_trainer_v1.py",
        ROOT / "models/iris/v31/lofo_trainer_v2.py",
        ROOT / "models/iris/v31/lofo_trainer_v3.py",
        ROOT / "models/iris/v31/presentation_objective_v1.py",
        Path(__file__).resolve(),
    ]
    source_hashes = {
        str(path.relative_to(ROOT)): file_sha256(path)
        for path in source_files
    }
    binding = {
        "schema": "RealSaS.IRISV31FixedSnakeEvidenceAdapterExecutionBinding.v2",
        "architecture_id": ARCHITECTURE_ID_V2,
        "input_manifest_sha256": file_sha256(input_path),
        "recipe_sha256": file_sha256(recipe_path),
        "fold": folds[0],
        "source_hashes": source_hashes,
    }
    binding_hash = _canonical_hash(binding)

    started = time.monotonic()
    result = run_fixed_snake_evidence_adapter_pilot_v2(
        input_manifest_path=input_path,
        recipe_path=recipe_path,
        checkpoint_root=parsed.checkpoint_root,
        output_root=output,
        device=parsed.device,
    )
    if result.get("status") != "PASS_EXECUTED_FIXED_SNAKE_EVIDENCE_ADAPTER_PILOT__SCIENTIFIC_QUALIFICATION_SEPARATE":
        raise SystemExit("IRIS_EA_V2_WORKER_RESULT_NOT_PASS")

    result_path = output / "result.json"
    checkpoint_path = output / "selected_checkpoint.pt"
    receipt = {
        "schema": "RealSaS.IRISV31FixedSnakeEvidenceAdapterExecutionReceipt.v2",
        "lane": "IRIS",
        "status": "PASS",
        "executor_kind": "EXTERNAL_GPU_JOB_V1",
        "architecture_id": ARCHITECTURE_ID_V2,
        "trainer_revision": TRAINER_REVISION_V2,
        "fold_id": FIXED_FOLD_ID,
        "preregistration_binding_hash": binding_hash,
        "source_hashes": source_hashes,
        "input_manifest_sha256": file_sha256(input_path),
        "recipe_sha256": file_sha256(recipe_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "result_sha256": file_sha256(result_path),
        "foundation_identity": result["foundation_identity"],
        "foundation_frozen_during_evidence": result["foundation_frozen_during_evidence"],
        "heldout_used_for_gradient": False,
        "heldout_used_for_checkpoint_selection": False,
        "selection_rule": result["selection_rule"],
        "semantic_part_taxonomy_used": False,
        "shared_foundation_jointly_finetuned_with_evidence": False,
        "fit8_pass_claimed": False,
        "product_pass_claimed": False,
        "runtime_environment": _runtime(),
        "wall_seconds": time.monotonic() - started,
    }
    receipt_path = output / "execution_receipt.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "fold_id": FIXED_FOLD_ID,
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "result_sha256": receipt["result_sha256"],
        "receipt_sha256": file_sha256(receipt_path),
        "foundation_identity": receipt["foundation_identity"],
    }, sort_keys=True), flush=True)
    return receipt


if __name__ == "__main__":
    main()
