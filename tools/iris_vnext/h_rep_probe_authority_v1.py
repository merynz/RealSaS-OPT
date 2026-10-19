from __future__ import annotations

"""Extractor-free authority helpers for H-REP-P0.

This module intentionally depends only on the sealed IRIS V8 model/contract and
prepared source tensors. It must not import direct-surface materializers,
FlexiCubes, sparse tetra decoders, SURF-A1 rescue modules, TESSA, or Compiler
repair code.
"""

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import torch

from models.iris.vnext.cohort_v9 import (
    V8_BINDING_SHA256,
    V8_CHECKPOINT_SHA256,
    V8_GLOBAL_STEP,
)
from models.iris.vnext.contracts_v1 import IRISVNextConfigV1
from models.iris.vnext.model_v1 import IRISVNextV1

SOURCE_KEYS = ("image_tokens", "token_xy", "camera_features")
EXPECTED_SOURCE_RANKS = {"image_tokens": 4, "token_xy": 4, "camera_features": 3}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"H_REP_P0_JSON_OBJECT_REQUIRED:{path}")
    return value


def git_head(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip().lower()


def require_exact_clean_source_v1(repo_root: Path, expected_sha: str) -> str:
    repo_root = repo_root.resolve()
    expected = str(expected_sha).lower()
    head = git_head(repo_root)
    if head != expected:
        raise RuntimeError(f"H_REP_P0_SOURCE_SHA_DRIFT:{head}:{expected}")
    dirty = subprocess.check_output(
        ["git", "-C", str(repo_root), "status", "--porcelain"], text=True
    )
    if dirty.strip():
        raise RuntimeError("H_REP_P0_SOURCE_REPO_DIRTY")
    return head


def require_a100_runtime_v1(minimum_memory_gib: float = 39.0) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("H_REP_P0_A100_CUDA_REQUIRED")
    properties = torch.cuda.get_device_properties(0)
    memory_gib = float(properties.total_memory) / (1024.0 ** 3)
    if "A100" not in str(properties.name).upper():
        raise RuntimeError(f"H_REP_P0_EXACT_A100_CLASS_REQUIRED:{properties.name}")
    if memory_gib < float(minimum_memory_gib):
        raise RuntimeError(
            f"H_REP_P0_MINIMUM_CUDA_MEMORY_GIB_REQUIRED:{memory_gib}:{minimum_memory_gib}"
        )
    return {
        "name": str(properties.name),
        "total_memory_gib": memory_gib,
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
    }


def module_state_sha256_v1(module: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        cpu = tensor.detach().cpu().contiguous()
        h.update(name.encode("utf-8"))
        h.update(str(cpu.dtype).encode("utf-8"))
        h.update(json.dumps(list(cpu.shape), separators=(",", ":")).encode("utf-8"))
        h.update(cpu.numpy().tobytes(order="C"))
    return h.hexdigest()


def load_exact_v8_model(checkpoint_path: Path, device: torch.device) -> IRISVNextV1:
    checkpoint_path = checkpoint_path.resolve()
    if sha256_file(checkpoint_path) != V8_CHECKPOINT_SHA256:
        raise ValueError("H_REP_P0_V8_CHECKPOINT_SHA_DRIFT")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if int(payload.get("step", -1)) != V8_GLOBAL_STEP:
        raise ValueError("H_REP_P0_V8_STEP_DRIFT")
    if payload.get("binding_sha256") != V8_BINDING_SHA256:
        raise ValueError("H_REP_P0_V8_BINDING_SHA_DRIFT")
    binding = payload.get("binding") or {}
    if binding.get("schema") != "RealSaS.IRISVNextFit8Binding.v10":
        raise ValueError("H_REP_P0_V8_BINDING_SCHEMA_INVALID")
    scene_authority = str(
        (binding.get("post_m1_parameter_authority") or {}).get("scene_core", "")
    )
    if not scene_authority.startswith("SEALED_LR_ZERO"):
        raise ValueError("H_REP_P0_V8_SCENE_CORE_NOT_SEALED")

    model = IRISVNextV1(IRISVNextConfigV1())
    model.load_state_dict(payload["model"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    model.to(device=device, dtype=torch.float32)
    return model


def load_subject_source(
    shard_path: Path, expected_sha256: str, device: torch.device
) -> dict[str, torch.Tensor]:
    if sha256_file(shard_path) != str(expected_sha256).lower():
        raise ValueError("H_REP_P0_PREPARED_SHARD_SHA_DRIFT:" + shard_path.name)
    payload = torch.load(shard_path, map_location="cpu", weights_only=True)
    source: dict[str, torch.Tensor] = {}
    for key in SOURCE_KEYS:
        tensor = payload.get(key)
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("H_REP_P0_SOURCE_FIELD_MISSING:" + key)
        expected_rank = EXPECTED_SOURCE_RANKS[key]
        if tensor.ndim == expected_rank - 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim != expected_rank or tensor.shape[0] != 1:
            raise ValueError("H_REP_P0_SOURCE_SHAPE_INVALID:" + key)
        if not torch.is_floating_point(tensor) or not bool(
            torch.isfinite(tensor).all().item()
        ):
            raise ValueError("H_REP_P0_SOURCE_TENSOR_INVALID:" + key)
        source[key] = tensor.to(device=device, dtype=torch.float32).contiguous()
    if source["image_tokens"].shape[:2] != (1, 8):
        raise ValueError("H_REP_P0_VIEW_COUNT_INVALID:image_tokens")
    if source["token_xy"].shape[:2] != (1, 8):
        raise ValueError("H_REP_P0_VIEW_COUNT_INVALID:token_xy")
    if source["camera_features"].shape[:2] != (1, 8):
        raise ValueError("H_REP_P0_VIEW_COUNT_INVALID:camera_features")
    return source
