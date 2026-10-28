from __future__ import annotations

"""Versioned numerical-environment identity for frozen V3.1 feature bytes.

This identity is cache/input lineage only. It is not scientific qualification,
ModelRelease authority, or a statement that two different environments are
numerically inequivalent.
"""

import hashlib
import json
import platform
import sys
from typing import Any

import torch


SCHEMA_V1 = "RealSaS.IRISNumericalEnvironment.v1"


def numerical_environment_v1(device: torch.device | str) -> dict[str, Any]:
    device = torch.device(device)
    try:
        import torchvision

        torchvision_version = str(torchvision.__version__)
    except Exception as exc:
        raise RuntimeError("IRIS_V31_TORCHVISION_ENVIRONMENT_REQUIRED") from exc

    cuda_runtime = str(torch.version.cuda) if torch.version.cuda is not None else None
    cudnn_version = int(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None
    resolved_index = None
    device_name = None
    device_capability = None
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("IRIS_V31_NUMERICAL_ENVIRONMENT_CUDA_UNAVAILABLE")
        resolved_index = int(torch.cuda.current_device() if device.index is None else device.index)
        device_name = str(torch.cuda.get_device_name(resolved_index))
        capability = torch.cuda.get_device_capability(resolved_index)
        device_capability = [int(capability[0]), int(capability[1])]
    elif device.type != "cpu":
        raise RuntimeError("IRIS_V31_NUMERICAL_ENVIRONMENT_DEVICE_UNSUPPORTED")

    torch_build = str(torch.__config__.show()).encode("utf-8")
    payload = {
        "schema": SCHEMA_V1,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch_version": str(torch.__version__),
        "torchvision_version": torchvision_version,
        "torch_build_config_sha256": hashlib.sha256(torch_build).hexdigest(),
        "device_type": device.type,
        "device_index": resolved_index,
        "device_name": device_name,
        "device_capability": device_capability,
        "cuda_runtime_version": cuda_runtime,
        "cudnn_version": cudnn_version,
        "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "default_dtype": str(torch.get_default_dtype()),
        "byteorder": str(sys.byteorder),
        "authority_class": "NUMERICAL_LINEAGE_ONLY",
    }
    return payload


def numerical_environment_sha256_v1(device: torch.device | str) -> str:
    payload = json.dumps(
        numerical_environment_v1(device),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
