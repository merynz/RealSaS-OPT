from __future__ import annotations

"""Self-contained deployable checkpoint contract for AXIS V5.4.1.

The historical FIT1 lineage was a three-part reconstruction:
Geppetto base checkpoint + AXIS V5.3 causal/tree delta + V5.4.1 coordinate delta.
That layout is valid research provenance but is not a valid Product ModelRelease
runtime contract because it introduces hidden checkpoint dependencies.

This module seals that historical lineage once into a single immutable runtime
checkpoint after legacy diffusion has been retired. Product inference consumes
only the resulting full retired-model state.
"""

from hashlib import sha256
import os
from pathlib import Path
from typing import Any


AXIS_V541_DEPLOYABLE_CHECKPOINT_SCHEMA = "RealSaS.AXISBFSDiscreteXYZV541.DeployableCheckpoint.v1"
AXIS_V541_HISTORICAL_COMPOSITE_SCHEMA = "RealSaS.AXIS41DiscreteXYZComposite.v5_4_1"
AXIS_V53_PROGRESS_SCHEMA = "RealSaS.AXIS41TreeGrammarProgress.v5_3"


def _sha256_file(path: Path) -> str:
    h = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _require_hex_sha(value: Any, code: str) -> str:
    raw = str(value or "")
    if len(raw) != 64 or raw.lower() != raw or any(ch not in "0123456789abcdef" for ch in raw):
        raise RuntimeError(code)
    return raw


def _load_torch(path: Path, *, weights_only: bool):
    import torch

    return torch.load(path, map_location="cpu", weights_only=weights_only)


def seal_axis_v541_deployable_checkpoint_v1(
    *,
    base_checkpoint_path: Path,
    v53_progress_path: Path,
    v541_composite_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Collapse the sealed historical FIT lineage into one runtime checkpoint.

    This is an offline migration/sealing operation, not product inference. The
    returned checkpoint contains the complete state of the runtime-retired AXIS
    model and no optimizer, teacher target, live alias or historical diffusion
    parameters.
    """

    import torch

    from models.axis.axis_bfs_discrete_xyz_v541 import (
        AXISBFSDiscreteXYZV541,
        AXIS_V541_ARCHITECTURE_ID,
    )

    base = Path(base_checkpoint_path).resolve()
    v53 = Path(v53_progress_path).resolve()
    v541 = Path(v541_composite_path).resolve()
    out = Path(output_path).resolve()
    for path in (base, v53, v541):
        if not path.is_file():
            raise RuntimeError("AXIS_V541_DEPLOYABLE_SOURCE_MISSING:" + path.name)

    base_sha = _sha256_file(base)
    v53_sha = _sha256_file(v53)
    v541_sha = _sha256_file(v541)

    base_payload = _load_torch(base, weights_only=False)
    v53_payload = _load_torch(v53, weights_only=False)
    v541_payload = _load_torch(v541, weights_only=False)
    if not isinstance(base_payload, dict) or not isinstance(v53_payload, dict) or not isinstance(v541_payload, dict):
        raise RuntimeError("AXIS_V541_DEPLOYABLE_SOURCE_PAYLOAD_INVALID")
    if str(v53_payload.get("schema") or "") != AXIS_V53_PROGRESS_SCHEMA:
        raise RuntimeError("AXIS_V541_V53_SCHEMA_DRIFT")
    if str(v541_payload.get("schema") or "") != AXIS_V541_HISTORICAL_COMPOSITE_SCHEMA:
        raise RuntimeError("AXIS_V541_COMPOSITE_SCHEMA_DRIFT")
    if str(v541_payload.get("architecture") or "") != AXIS_V541_ARCHITECTURE_ID:
        raise RuntimeError("AXIS_V541_COMPOSITE_ARCHITECTURE_DRIFT")
    if _require_hex_sha(v53_payload.get("base_checkpoint_sha256"), "AXIS_V541_V53_BASE_SHA_INVALID") != base_sha:
        raise RuntimeError("AXIS_V541_V53_BASE_SHA_DRIFT")
    if _require_hex_sha(v541_payload.get("base_checkpoint_sha256"), "AXIS_V541_COMPOSITE_BASE_SHA_INVALID") != base_sha:
        raise RuntimeError("AXIS_V541_COMPOSITE_BASE_SHA_DRIFT")
    if _require_hex_sha(v541_payload.get("v53_progress_sha256"), "AXIS_V541_COMPOSITE_V53_SHA_INVALID") != v53_sha:
        raise RuntimeError("AXIS_V541_COMPOSITE_V53_SHA_DRIFT")
    if v541_payload.get("hard_causal_parent_authority") is not True:
        raise RuntimeError("AXIS_V541_HARD_PARENT_AUTHORITY_REQUIRED")
    if v541_payload.get("compiler_tree_reselection") is not False:
        raise RuntimeError("AXIS_V541_COMPILER_RESELECTION_FORBIDDEN")
    if str(v541_payload.get("geometry_authority") or "") != "ORDERED_256_BIN_XYZ_ARGMAX":
        raise RuntimeError("AXIS_V541_GEOMETRY_AUTHORITY_DRIFT")
    if v541_payload.get("diffusion_runtime_authority") is not False:
        raise RuntimeError("AXIS_V541_DIFFUSION_AUTHORITY_FORBIDDEN")

    model = AXISBFSDiscreteXYZV541().cpu().float()
    base_state = base_payload.get("model")
    if not isinstance(base_state, dict):
        raise RuntimeError("AXIS_V541_BASE_MODEL_STATE_REQUIRED")
    missing, unexpected = model.load_state_dict(base_state, strict=False)
    if unexpected:
        raise RuntimeError("AXIS_V541_BASE_UNEXPECTED_STATE:" + ",".join(sorted(map(str, unexpected))))
    expected_missing = {name for name in model.state_dict() if name.startswith("coord_token_delta.")}
    if set(map(str, missing)) != expected_missing:
        raise RuntimeError("AXIS_V541_BASE_MISSING_STATE_DRIFT")
    if str(base_payload.get("config_hash") or "") != str(model.config.config_hash):
        raise RuntimeError("AXIS_V541_BASE_CONFIG_HASH_DRIFT")

    state = model.state_dict()
    v53_delta = v53_payload.get("trainable_delta")
    if not isinstance(v53_delta, dict) or not v53_delta:
        raise RuntimeError("AXIS_V541_V53_DELTA_REQUIRED")
    for name, value in v53_delta.items():
        key = str(name)
        if key not in state or key.startswith("coord_token_delta."):
            raise RuntimeError("AXIS_V541_V53_DELTA_KEY_INVALID:" + key)
        state[key] = value
    model.load_state_dict(state, strict=True)

    v541_delta = v541_payload.get("trainable_model_delta")
    if not isinstance(v541_delta, dict) or not v541_delta:
        raise RuntimeError("AXIS_V541_COORD_DELTA_REQUIRED")
    state = model.state_dict()
    expected_coord = {name for name in state if name.startswith("coord_token_delta.")}
    if set(map(str, v541_delta)) != expected_coord:
        raise RuntimeError("AXIS_V541_COORD_DELTA_KEY_DRIFT")
    for name, value in v541_delta.items():
        state[str(name)] = value
    model.load_state_dict(state, strict=True)

    # Product runtime must not even carry the historical diffusion parameters.
    model.retire_legacy_diffusion_runtime()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    runtime_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    if any(name.startswith("diffusion.") for name in runtime_state):
        raise RuntimeError("AXIS_V541_DEPLOYABLE_DIFFUSION_STATE_FORBIDDEN")

    payload = {
        "schema": AXIS_V541_DEPLOYABLE_CHECKPOINT_SCHEMA,
        "architecture_id": AXIS_V541_ARCHITECTURE_ID,
        "config_hash": str(model.config.config_hash),
        "model_state_dict": runtime_state,
        "runtime_diffusion_retired": True,
        "legacy_diffusion_parameters_present": False,
        "hard_causal_parent_authority": True,
        "compiler_role": "VALIDATE_AND_MATERIALIZE",
        "compiler_parent_reselection_allowed": False,
        "geometry_authority": "ORDERED_256_BIN_XYZ_ARGMAX",
        "coord_bins_per_axis": 256,
        "full_runtime_state": True,
        "optimizer_state": None,
        "teacher_runtime_authority": False,
        "historical_fit_provenance": {
            "base_checkpoint_sha256": base_sha,
            "v53_progress_sha256": v53_sha,
            "v541_composite_sha256": v541_sha,
            "historical_three_part_runtime_dependency": False,
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, out)
    return {
        "schema": AXIS_V541_DEPLOYABLE_CHECKPOINT_SCHEMA,
        "architecture_id": AXIS_V541_ARCHITECTURE_ID,
        "checkpoint_content_sha256": _sha256_file(out),
        "checkpoint_size_bytes": out.stat().st_size,
        "historical_fit_provenance": dict(payload["historical_fit_provenance"]),
    }


def load_axis_v541_deployable_checkpoint_v1(path: Path, *, device):
    """Load exactly one self-contained ModelRelease checkpoint, fail closed."""

    from models.axis.axis_bfs_discrete_xyz_v541 import (
        AXISBFSDiscreteXYZV541,
        AXIS_V541_ARCHITECTURE_ID,
    )

    payload = _load_torch(Path(path), weights_only=True)
    if not isinstance(payload, dict):
        raise RuntimeError("AXIS_V541_DEPLOYABLE_PAYLOAD_INVALID")
    if str(payload.get("schema") or "") != AXIS_V541_DEPLOYABLE_CHECKPOINT_SCHEMA:
        raise RuntimeError("AXIS_V541_DEPLOYABLE_SCHEMA_DRIFT")
    if str(payload.get("architecture_id") or "") != AXIS_V541_ARCHITECTURE_ID:
        raise RuntimeError("AXIS_V541_DEPLOYABLE_ARCHITECTURE_DRIFT")
    if payload.get("runtime_diffusion_retired") is not True:
        raise RuntimeError("AXIS_V541_DEPLOYABLE_DIFFUSION_NOT_RETIRED")
    if payload.get("legacy_diffusion_parameters_present") is not False:
        raise RuntimeError("AXIS_V541_DEPLOYABLE_LEGACY_DIFFUSION_PRESENT")
    if payload.get("hard_causal_parent_authority") is not True:
        raise RuntimeError("AXIS_V541_DEPLOYABLE_HARD_PARENT_AUTHORITY_DRIFT")
    if payload.get("compiler_parent_reselection_allowed") is not False:
        raise RuntimeError("AXIS_V541_DEPLOYABLE_RESELECTION_DRIFT")
    if str(payload.get("geometry_authority") or "") != "ORDERED_256_BIN_XYZ_ARGMAX":
        raise RuntimeError("AXIS_V541_DEPLOYABLE_GEOMETRY_AUTHORITY_DRIFT")
    if int(payload.get("coord_bins_per_axis") or 0) != 256:
        raise RuntimeError("AXIS_V541_DEPLOYABLE_COORD_BIN_DRIFT")
    state = payload.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise RuntimeError("AXIS_V541_DEPLOYABLE_MODEL_STATE_REQUIRED")
    if any(str(name).startswith("diffusion.") for name in state):
        raise RuntimeError("AXIS_V541_DEPLOYABLE_DIFFUSION_STATE_FORBIDDEN")

    model = AXISBFSDiscreteXYZV541()
    model.retire_legacy_diffusion_runtime()
    if str(payload.get("config_hash") or "") != str(model.config.config_hash):
        raise RuntimeError("AXIS_V541_DEPLOYABLE_CONFIG_HASH_DRIFT")
    model.load_state_dict(state, strict=True)
    model = model.to(device).float().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload


__all__ = [
    "AXIS_V541_DEPLOYABLE_CHECKPOINT_SCHEMA",
    "load_axis_v541_deployable_checkpoint_v1",
    "seal_axis_v541_deployable_checkpoint_v1",
]
