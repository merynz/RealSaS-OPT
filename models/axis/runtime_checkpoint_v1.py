from __future__ import annotations

"""Self-contained runtime packaging for AXIS V5.4.1.

The sealed FIT1 V5.4.1 composite is a scientific delta receipt, not a standalone
runtime checkpoint: it references a historical base model and V5.3 progress.
This module verifies those immutable relationships once and packs the exact
inference state after legacy diffusion is retired. Product execution accepts only
this runtime package; it never reconstructs model state from live historical
paths or notebook conventions.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

AXIS_RUNTIME_CHECKPOINT_SCHEMA_V1 = "RealSaS.AXIS.RuntimeCheckpoint.v1"
AXIS_V541_COMPOSITE_SCHEMA = "RealSaS.AXIS41DiscreteXYZComposite.v5_4_1"
AXIS_V53_PROGRESS_SCHEMA = "RealSaS.AXIS41TreeGrammarProgress.v5_3"
AXIS_V541_ARCHITECTURE_ID = (
    "RealSaS.AXIS41.BFSHardParent.OrderedDiscreteXYZ256.FrozenCausal.v5_4_1"
)


@dataclass(frozen=True)
class AxisRuntimePackageReceiptV1:
    output_path: str
    content_sha256: str
    base_checkpoint_sha256: str
    v53_progress_sha256: str
    v541_composite_sha256: str
    architecture_id: str = AXIS_V541_ARCHITECTURE_ID
    schema: str = "RealSaS.AXIS.RuntimePackageReceipt.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "output_path": self.output_path,
            "content_sha256": self.content_sha256,
            "base_checkpoint_sha256": self.base_checkpoint_sha256,
            "v53_progress_sha256": self.v53_progress_sha256,
            "v541_composite_sha256": self.v541_composite_sha256,
            "architecture_id": self.architecture_id,
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(code)
    return value


def _load_torch_file(path: Path) -> Mapping[str, Any]:
    import torch

    if not path.is_file():
        raise RuntimeError(f"AXIS_RUNTIME_PACKAGE_INPUT_MISSING:{path}")
    value = torch.load(path, map_location="cpu", weights_only=False)
    return _mapping(value, "AXIS_RUNTIME_PACKAGE_INPUT_NOT_MAPPING")


def _terminal_eval_pass(value: Mapping[str, Any]) -> bool:
    terminal = value.get("terminal_eval")
    if not isinstance(terminal, Mapping):
        return False
    if terminal.get("pass") is True:
        return True
    report = terminal.get("report")
    return isinstance(report, Mapping) and report.get("pass") is True


def build_axis_runtime_checkpoint_v1(
    *,
    base_checkpoint_path: str | Path,
    v53_progress_path: str | Path,
    v541_composite_path: str | Path,
) -> dict[str, Any]:
    """Build one complete inference checkpoint from the sealed FIT1 components.

    This is deterministic packaging, not training or scientific promotion.  The
    returned state contains no live diffusion module parameters because runtime
    diffusion is explicitly forbidden by the promoted AXIS contract.
    """

    import torch

    from models.axis.axis_bfs_discrete_xyz_v541 import AXISBFSDiscreteXYZV541
    from models.geppetto.reference_strength_v1.geppetto_reference_strength_no_learned_slot_v1 import (
        assert_no_learned_view_slot_identity_v1,
    )

    base_path = Path(base_checkpoint_path).resolve()
    v53_path = Path(v53_progress_path).resolve()
    v541_path = Path(v541_composite_path).resolve()
    base_sha = _sha256_file(base_path)
    v53_sha = _sha256_file(v53_path)
    v541_sha = _sha256_file(v541_path)

    base = _load_torch_file(base_path)
    v53 = _load_torch_file(v53_path)
    composite = _load_torch_file(v541_path)

    if str(v53.get("schema") or "") != AXIS_V53_PROGRESS_SCHEMA:
        raise RuntimeError("AXIS_RUNTIME_PACKAGE_V53_SCHEMA_DRIFT")
    if str(composite.get("schema") or "") != AXIS_V541_COMPOSITE_SCHEMA:
        raise RuntimeError("AXIS_RUNTIME_PACKAGE_V541_SCHEMA_DRIFT")
    if str(composite.get("architecture") or "") != AXIS_V541_ARCHITECTURE_ID:
        raise RuntimeError("AXIS_RUNTIME_PACKAGE_ARCHITECTURE_DRIFT")
    if str(v53.get("base_checkpoint_sha256") or "") != base_sha:
        raise RuntimeError("AXIS_RUNTIME_PACKAGE_V53_BASE_SHA_DRIFT")
    if str(composite.get("base_checkpoint_sha256") or "") != base_sha:
        raise RuntimeError("AXIS_RUNTIME_PACKAGE_V541_BASE_SHA_DRIFT")
    if str(composite.get("v53_progress_sha256") or "") != v53_sha:
        raise RuntimeError("AXIS_RUNTIME_PACKAGE_V541_V53_SHA_DRIFT")
    if composite.get("hard_causal_parent_authority") is not True:
        raise RuntimeError("AXIS_RUNTIME_PACKAGE_PARENT_AUTHORITY_DRIFT")
    if composite.get("compiler_tree_reselection") is not False:
        raise RuntimeError("AXIS_RUNTIME_PACKAGE_COMPILER_RESELECTION_FORBIDDEN")
    if str(composite.get("geometry_authority") or "") != "ORDERED_256_BIN_XYZ_ARGMAX":
        raise RuntimeError("AXIS_RUNTIME_PACKAGE_GEOMETRY_AUTHORITY_DRIFT")
    if int(composite.get("coord_bins_per_axis") or 0) != 256:
        raise RuntimeError("AXIS_RUNTIME_PACKAGE_COORD_BIN_DRIFT")
    if composite.get("diffusion_runtime_authority") is not False:
        raise RuntimeError("AXIS_RUNTIME_PACKAGE_DIFFUSION_AUTHORITY_FORBIDDEN")
    if int(composite.get("diffusion_invocation_count") or 0) != 0:
        raise RuntimeError("AXIS_RUNTIME_PACKAGE_DIFFUSION_WAS_INVOKED")
    if not _terminal_eval_pass(composite):
        raise RuntimeError("AXIS_RUNTIME_PACKAGE_TERMINAL_EVAL_NOT_PASS")

    base_state = _mapping(base.get("model"), "AXIS_RUNTIME_PACKAGE_BASE_MODEL_MISSING")
    v53_delta = _mapping(v53.get("trainable_delta"), "AXIS_RUNTIME_PACKAGE_V53_DELTA_MISSING")
    v541_delta = _mapping(
        composite.get("trainable_model_delta"),
        "AXIS_RUNTIME_PACKAGE_V541_DELTA_MISSING",
    )

    model = AXISBFSDiscreteXYZV541()
    assert_no_learned_view_slot_identity_v1(model)
    missing, unexpected = model.load_state_dict(dict(base_state), strict=False)
    if unexpected:
        raise RuntimeError(f"AXIS_RUNTIME_PACKAGE_BASE_UNEXPECTED_KEYS:{sorted(unexpected)}")
    expected_missing = {
        key for key in model.state_dict() if key.startswith("coord_token_delta.")
    }
    if set(missing) != expected_missing:
        raise RuntimeError(
            "AXIS_RUNTIME_PACKAGE_BASE_MISSING_KEYS_DRIFT:"
            + repr(sorted(set(missing) ^ expected_missing))
        )
    base_config_hash = str(base.get("config_hash") or "")
    if not base_config_hash or model.config.config_hash != base_config_hash:
        raise RuntimeError("AXIS_RUNTIME_PACKAGE_CONFIG_HASH_DRIFT")

    state = model.state_dict()
    for key, tensor in v53_delta.items():
        name = str(key)
        if name not in state or name.startswith("coord_token_delta."):
            raise RuntimeError(f"AXIS_RUNTIME_PACKAGE_V53_DELTA_KEY_INVALID:{name}")
        state[name] = tensor
    for key, tensor in v541_delta.items():
        name = str(key)
        if name not in state or not name.startswith("coord_token_delta."):
            raise RuntimeError(f"AXIS_RUNTIME_PACKAGE_V541_DELTA_KEY_INVALID:{name}")
        state[name] = tensor
    model.load_state_dict(state, strict=True)
    model.retire_legacy_diffusion_runtime()
    model.eval()
    runtime_state = {
        str(key): value.detach().cpu()
        for key, value in model.state_dict().items()
    }
    if any(key.startswith("diffusion.") for key in runtime_state):
        raise RuntimeError("AXIS_RUNTIME_PACKAGE_DIFFUSION_STATE_PRESENT")

    return {
        "schema": AXIS_RUNTIME_CHECKPOINT_SCHEMA_V1,
        "architecture_id": AXIS_V541_ARCHITECTURE_ID,
        "config_hash": model.config.config_hash,
        "state_dict": runtime_state,
        "source_components": {
            "base_checkpoint_sha256": base_sha,
            "v53_progress_sha256": v53_sha,
            "v541_composite_sha256": v541_sha,
        },
        "runtime_rules": {
            "self_contained_inference_state": True,
            "hard_causal_parent_authority": True,
            "compiler_parent_reselection_allowed": False,
            "geometry_authority": "ORDERED_256_BIN_XYZ_ARGMAX",
            "coord_bins_per_axis": 256,
            "diffusion_runtime_authority": False,
            "legacy_diffusion_state_packaged": False,
            "teacher_mapping_runtime_authority": False,
            "source_rig_runtime_authority": False,
            "source_skin_runtime_authority": False,
        },
        "packaging_authority": "RUNTIME_PACKAGING_ONLY__NO_NEW_SCIENTIFIC_PASS",
    }


def save_axis_runtime_checkpoint_v1(
    *,
    output_path: str | Path,
    base_checkpoint_path: str | Path,
    v53_progress_path: str | Path,
    v541_composite_path: str | Path,
) -> AxisRuntimePackageReceiptV1:
    import torch

    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_axis_runtime_checkpoint_v1(
        base_checkpoint_path=base_checkpoint_path,
        v53_progress_path=v53_progress_path,
        v541_composite_path=v541_composite_path,
    )
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    return AxisRuntimePackageReceiptV1(
        output_path=str(output),
        content_sha256=_sha256_file(output),
        base_checkpoint_sha256=_sha256_file(Path(base_checkpoint_path)),
        v53_progress_sha256=_sha256_file(Path(v53_progress_path)),
        v541_composite_sha256=_sha256_file(Path(v541_composite_path)),
    )


def load_axis_runtime_model_v1(
    checkpoint_path: str | Path,
    *,
    device: str | Any = "cpu",
):
    """Load only the self-contained runtime package accepted by product.rig."""

    import torch

    from models.axis.axis_bfs_discrete_xyz_v541 import AXISBFSDiscreteXYZV541
    from models.geppetto.reference_strength_v1.geppetto_reference_strength_no_learned_slot_v1 import (
        assert_no_learned_view_slot_identity_v1,
    )

    path = Path(checkpoint_path).resolve()
    payload = _load_torch_file(path)
    if str(payload.get("schema") or "") != AXIS_RUNTIME_CHECKPOINT_SCHEMA_V1:
        raise RuntimeError("AXIS_RUNTIME_CHECKPOINT_SCHEMA_DRIFT")
    if str(payload.get("architecture_id") or "") != AXIS_V541_ARCHITECTURE_ID:
        raise RuntimeError("AXIS_RUNTIME_CHECKPOINT_ARCHITECTURE_DRIFT")
    rules = _mapping(payload.get("runtime_rules"), "AXIS_RUNTIME_CHECKPOINT_RULES_MISSING")
    required_rules = {
        "self_contained_inference_state": True,
        "hard_causal_parent_authority": True,
        "compiler_parent_reselection_allowed": False,
        "geometry_authority": "ORDERED_256_BIN_XYZ_ARGMAX",
        "coord_bins_per_axis": 256,
        "diffusion_runtime_authority": False,
        "legacy_diffusion_state_packaged": False,
        "teacher_mapping_runtime_authority": False,
        "source_rig_runtime_authority": False,
        "source_skin_runtime_authority": False,
    }
    for key, expected in required_rules.items():
        if rules.get(key) != expected:
            raise RuntimeError(f"AXIS_RUNTIME_CHECKPOINT_RULE_DRIFT:{key}")
    state = _mapping(payload.get("state_dict"), "AXIS_RUNTIME_CHECKPOINT_STATE_MISSING")
    if any(str(key).startswith("diffusion.") for key in state):
        raise RuntimeError("AXIS_RUNTIME_CHECKPOINT_DIFFUSION_STATE_FORBIDDEN")

    model = AXISBFSDiscreteXYZV541().retire_legacy_diffusion_runtime()
    assert_no_learned_view_slot_identity_v1(model)
    model.load_state_dict(dict(state), strict=True)
    if str(payload.get("config_hash") or "") != model.config.config_hash:
        raise RuntimeError("AXIS_RUNTIME_CHECKPOINT_CONFIG_HASH_DRIFT")
    model.to(torch.device(device))
    model.eval()
    return model, dict(payload)


__all__ = [
    "AXIS_RUNTIME_CHECKPOINT_SCHEMA_V1",
    "AXIS_V541_ARCHITECTURE_ID",
    "AxisRuntimePackageReceiptV1",
    "build_axis_runtime_checkpoint_v1",
    "load_axis_runtime_model_v1",
    "save_axis_runtime_checkpoint_v1",
]
