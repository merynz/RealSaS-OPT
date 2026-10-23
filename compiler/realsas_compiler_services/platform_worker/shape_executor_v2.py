from __future__ import annotations

"""Capability-local executor for ProductShapeInference.v2."""

import os
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from compiler.realsas_compiler_services.platform_worker.model_inputs import materialize_model_releases


SHAPE_EXECUTOR_CONTRACT_V2 = "RealSaS.ProductShapeInference.v2"
SHAPE_INPUT_ARTIFACT_TYPE_V1 = "RealSaS.ShapeInput"
SHAPE_OUTPUT_ARTIFACT_TYPE_V2 = "RealSaS.ShapeProposal"


def _one_shape_input(request: dict[str, Any]) -> dict[str, Any]:
    matches = [
        dict(ref)
        for ref in request.get("input_artifacts") or ()
        if str(ref.get("artifact_type") or "") == SHAPE_INPUT_ARTIFACT_TYPE_V1
        and str(ref.get("schema_version") or "") == "v1"
    ]
    if len(matches) != 1:
        raise RuntimeError(f"SHAPE_V2_INPUT_ARTIFACT_CARDINALITY:{len(matches)}")
    return matches[0]


def _device_from_environment():
    import torch

    raw = os.environ.get("REALSAS_INFERENCE_DEVICE", "cpu").strip() or "cpu"
    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SHAPE_V2_CUDA_REQUESTED_BUT_UNAVAILABLE")
    return device


def execute_shape_capability_v2(
    request: dict[str, Any],
    *,
    read_object: Callable[[dict[str, Any]], bytes],
    put_bytes: Callable[[bytes], dict[str, Any]],
) -> dict[str, Any]:
    metadata = dict(request.get("capability_metadata") or {})
    if metadata.get("executor_contract") != SHAPE_EXECUTOR_CONTRACT_V2:
        raise RuntimeError("SHAPE_V2_EXECUTOR_CONTRACT_DRIFT")
    if str(request.get("kind") or "") != "MODULE":
        raise RuntimeError("SHAPE_V2_CAPABILITY_KIND_DRIFT")
    execution_id = str(request.get("execution_id") or "")
    if not execution_id:
        raise RuntimeError("SHAPE_V2_EXECUTION_ID_REQUIRED")

    from compiler.realsas_compiler_services.platform_worker.scene_input_contract_v1 import decode_shape_input_v1
    from compiler.realsas_compiler_services.platform_worker.shape_capability_v2 import run_shape_inference_v2
    from models.iris.vnext.flexicubes_v1 import PinnedFlexiCubesV1, flexicubes_root_from_environment_v1

    manifest: dict[str, Any] = {"run_id": "capability-" + execution_id}
    materialize_started = perf_counter()
    materialize_model_releases(
        manifest=manifest,
        models=request.get("model_releases") or (),
        read_object=read_object,
    )
    materialize_seconds = perf_counter() - materialize_started

    source_ref = _one_shape_input(request)
    inputs = decode_shape_input_v1(read_object(source_ref))
    device = _device_from_environment()
    root: Path = flexicubes_root_from_environment_v1()
    extractor = PinnedFlexiCubesV1(root, device=device)
    maximum_new_tokens = int(metadata.get("maximum_new_tokens") or 131072)
    result = run_shape_inference_v2(
        manifest,
        inputs,
        extractor=extractor,
        device=device,
        maximum_new_tokens=maximum_new_tokens,
    )
    sealed = put_bytes(result.payload)
    diagnostics = dict(result.diagnostics)
    diagnostics["model_materialization_seconds"] = float(materialize_seconds)
    diagnostics["input_content_sha256"] = str(source_ref.get("content_sha256") or "")
    diagnostics["inference_device"] = str(device)
    diagnostics["maximum_new_tokens"] = maximum_new_tokens
    return {
        "capability_id": str(request.get("capability_id") or ""),
        "status": "PASS",
        "outputs": [
            {
                "role": "shape_proposal",
                "artifact_type": SHAPE_OUTPUT_ARTIFACT_TYPE_V2,
                "schema_version": "v2",
                "storage_key": sealed["storage_key"],
                "content_sha256": sealed["content_sha256"],
                "size_bytes": sealed["size_bytes"],
                "authority_class": "LEARNED_SHAPE_PROPOSAL_ONLY",
            }
        ],
        "diagnostics": diagnostics,
        "failure": None,
    }
