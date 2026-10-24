from __future__ import annotations

"""Lightweight engine boundary for product.input.

Heavy Torch/DINO/scientific imports are intentionally deferred until Input is
actually executed so generic Platform workers remain independent of this
replaceable scientific implementation.
"""

import os
from pathlib import Path
from time import perf_counter
from typing import Any, Callable


INPUT_EXECUTOR_CONTRACT_V1 = "RealSaS.ProductInputInference.v1"
SOURCE_OBSERVATION_ARTIFACT_TYPE_V1 = "RealSaS.SourceObservation"
SHAPE_INPUT_ARTIFACT_TYPE_V1 = "RealSaS.ShapeInput"
INPUT_OUTPUT_AUTHORITY_V1 = "VERIFIED_SOURCE_DERIVED_SHAPE_INPUT"


def decode_source_observation_v1(payload: bytes):
    from compiler.realsas_compiler_services.platform_worker.input_capability_v1 import (
        decode_source_observation_v1 as implementation,
    )

    return implementation(payload)


def load_product_dino_v1(**kwargs):
    from compiler.realsas_compiler_services.platform_worker.input_capability_v1 import (
        load_product_dino_v1 as implementation,
    )

    return implementation(**kwargs)


def run_input_inference_v1(*args, **kwargs):
    from compiler.realsas_compiler_services.platform_worker.input_capability_v1 import (
        run_input_inference_v1 as implementation,
    )

    return implementation(*args, **kwargs)


def _one_source_input(request: dict[str, Any]) -> dict[str, Any]:
    matches = [
        dict(ref)
        for ref in request.get("input_artifacts") or ()
        if str(ref.get("artifact_type") or "") == SOURCE_OBSERVATION_ARTIFACT_TYPE_V1
        and str(ref.get("schema_version") or "") == "v1"
    ]
    if len(matches) != 1:
        raise RuntimeError(f"PRODUCT_INPUT_V1_SOURCE_ARTIFACT_CARDINALITY:{len(matches)}")
    return matches[0]


def _device_from_environment():
    import torch

    raw = os.environ.get("REALSAS_INFERENCE_DEVICE", "cpu").strip() or "cpu"
    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PRODUCT_INPUT_V1_CUDA_REQUESTED_BUT_UNAVAILABLE")
    return device


def _dino_paths_from_deployment_root() -> tuple[Path, Path]:
    raw = os.environ.get("REALSAS_DEPLOY_ROOT", "").strip()
    if not raw:
        raise RuntimeError("PRODUCT_INPUT_V1_DEPLOY_ROOT_REQUIRED")
    root = Path(raw).expanduser().resolve()
    foundation = root / "foundations" / "dinov2"
    return foundation / "source", foundation / "dinov2_vits14_pretrain.pth"


def execute_input_capability_v1(
    request: dict[str, Any],
    *,
    read_object: Callable[[dict[str, Any]], bytes],
    put_bytes: Callable[[bytes], dict[str, Any]],
) -> dict[str, Any]:
    metadata = dict(request.get("capability_metadata") or {})
    if metadata.get("executor_contract") != INPUT_EXECUTOR_CONTRACT_V1:
        raise RuntimeError("PRODUCT_INPUT_V1_EXECUTOR_CONTRACT_DRIFT")
    if str(request.get("kind") or "") != "MODULE":
        raise RuntimeError("PRODUCT_INPUT_V1_CAPABILITY_KIND_DRIFT")
    execution_id = str(request.get("execution_id") or "")
    if not execution_id:
        raise RuntimeError("PRODUCT_INPUT_V1_EXECUTION_ID_REQUIRED")

    source_ref = _one_source_input(request)
    source = decode_source_observation_v1(read_object(source_ref))
    device = _device_from_environment()
    dino_source, dino_weights = _dino_paths_from_deployment_root()

    load_started = perf_counter()
    dino_model, loaded_authority = load_product_dino_v1(
        source_dir=dino_source,
        weight_path=dino_weights,
        device=device,
    )
    load_seconds = perf_counter() - load_started
    result = run_input_inference_v1(
        source,
        dino_model=dino_model,
        loaded_authority=loaded_authority,
        device=device,
    )
    sealed = put_bytes(result.payload)
    diagnostics = dict(result.diagnostics)
    diagnostics["dino_load_seconds"] = float(load_seconds)
    diagnostics["source_content_sha256"] = str(source_ref.get("content_sha256") or "")
    diagnostics["inference_device"] = str(device)

    return {
        "capability_id": str(request.get("capability_id") or ""),
        "status": "PASS",
        "outputs": [
            {
                "role": "shape_input",
                "artifact_type": SHAPE_INPUT_ARTIFACT_TYPE_V1,
                "schema_version": "v1",
                "storage_key": sealed["storage_key"],
                "content_sha256": sealed["content_sha256"],
                "size_bytes": sealed["size_bytes"],
                "authority_class": INPUT_OUTPUT_AUTHORITY_V1,
            }
        ],
        "diagnostics": diagnostics,
        "failure": None,
    }
