from __future__ import annotations

"""Self-contained runtime packaging for carrier-native MIRA V5.5.

The historical V5.5 composite contains the trained direct-carrier decoder and
binds it to an exact frozen Arachne V6 backbone. Product execution must not
reconstruct that pair from Drive/notebook paths. This module verifies the
sealed component relationship once and emits one immutable inference package.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping


MIRA_RUNTIME_CHECKPOINT_SCHEMA_V1 = "RealSaS.MIRA.RuntimeCheckpoint.v1"
MIRA_RUNTIME_PACKAGE_RECEIPT_SCHEMA_V1 = "RealSaS.MIRA.RuntimePackageReceipt.v1"
MIRA_V55_COMPOSITE_SCHEMA = "RealSaS.MIRAMDirectComposite.v5_5"
MIRA_V55_ARCHITECTURE_ID = "RealSaS.MIRA.DirectCarrierSimplex.Axis541.v5_5"

MIRA_V6_MODEL_SCHEMA = "RealSaS.KnightArachneV6DemoClosure.v1.Model.v1"
MIRA_V6_ARCHITECTURE_ID = "RealSaS.Arachne.A1.RawSurfaceK4AttentionDirectSimplex.v6"
MIRA_V6_BACKBONE_ARCHITECTURE_ID = "RealSaS.Arachne.A1.RichQualifiedBidirectional.v4"
MIRA_V6_BACKBONE_PARAMETER_COUNT = 138_053_153

MIRA_V55_CANONICAL_BASE_V6_SHA256 = (
    "bfd2bd24f55bdce2dc36b0ae6a7afe01b8381c680de545c66a11a3d707fe73ed"
)
MIRA_V55_CANONICAL_COMPOSITE_SHA256 = (
    "490cd7791c94c0e15e3fa6479a6701bd7bec21e03dff76b69ada638d5bf30a40"
)
MIRA_V55_HISTORICAL_RUNNER_SHA256 = (
    "39ff4f7a2051f0d9c1251a5999f472b9926e83deacbedcedc8e92eea424dce01"
)


@dataclass(frozen=True)
class MiraRuntimePackageReceiptV1:
    output_path: str
    content_sha256: str
    base_v6_checkpoint_sha256: str
    v55_composite_sha256: str
    architecture_id: str = MIRA_V55_ARCHITECTURE_ID
    schema: str = MIRA_RUNTIME_PACKAGE_RECEIPT_SCHEMA_V1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "output_path": self.output_path,
            "content_sha256": self.content_sha256,
            "base_v6_checkpoint_sha256": self.base_v6_checkpoint_sha256,
            "v55_composite_sha256": self.v55_composite_sha256,
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


def _load_torch_mapping(path: Path) -> Mapping[str, Any]:
    import torch

    if not path.is_file():
        raise RuntimeError(f"MIRA_RUNTIME_PACKAGE_INPUT_MISSING:{path}")
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - old torch compatibility only
        value = torch.load(path, map_location="cpu")
    return _mapping(value, "MIRA_RUNTIME_PACKAGE_INPUT_NOT_MAPPING")


def _state_parameter_count(state: Mapping[str, Any]) -> int:
    total = 0
    for key, value in state.items():
        if not hasattr(value, "numel"):
            raise RuntimeError(f"MIRA_RUNTIME_STATE_NON_TENSOR:{key}")
        total += int(value.numel())
    return total


def _validate_sha(value: str, code: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise RuntimeError(code)
    return normalized


def build_mira_runtime_checkpoint_v1(
    *,
    base_v6_checkpoint_path: str | Path,
    v55_composite_path: str | Path,
    expected_base_v6_sha256: str,
    expected_v55_composite_sha256: str,
) -> dict[str, Any]:
    """Pack exact V6 backbone + exact V5.5 decoder as one inference checkpoint."""

    from models.mira.direct_readout_v55 import MIRA_V55_DECODER_PARAMETER_COUNT

    base_path = Path(base_v6_checkpoint_path).resolve()
    composite_path = Path(v55_composite_path).resolve()
    base_sha = _sha256_file(base_path)
    composite_sha = _sha256_file(composite_path)
    if base_sha != _validate_sha(
        expected_base_v6_sha256, "MIRA_RUNTIME_EXPECTED_BASE_SHA_INVALID"
    ):
        raise RuntimeError("MIRA_RUNTIME_BASE_V6_SHA_DRIFT")
    if composite_sha != _validate_sha(
        expected_v55_composite_sha256, "MIRA_RUNTIME_EXPECTED_COMPOSITE_SHA_INVALID"
    ):
        raise RuntimeError("MIRA_RUNTIME_V55_COMPOSITE_SHA_DRIFT")

    base = _load_torch_mapping(base_path)
    composite = _load_torch_mapping(composite_path)

    if str(base.get("schema") or "") != MIRA_V6_MODEL_SCHEMA:
        raise RuntimeError("MIRA_RUNTIME_BASE_V6_SCHEMA_DRIFT")
    if str(base.get("architecture_id") or "") != MIRA_V6_ARCHITECTURE_ID:
        raise RuntimeError("MIRA_RUNTIME_BASE_V6_ARCHITECTURE_DRIFT")
    if str(base.get("backbone_architecture_id") or "") != MIRA_V6_BACKBONE_ARCHITECTURE_ID:
        raise RuntimeError("MIRA_RUNTIME_BACKBONE_ARCHITECTURE_DRIFT")
    if base.get("teacher_predictor_input_used") is not False:
        raise RuntimeError("MIRA_RUNTIME_TEACHER_PREDICTOR_INPUT_FORBIDDEN")

    if str(composite.get("schema") or "") != MIRA_V55_COMPOSITE_SCHEMA:
        raise RuntimeError("MIRA_RUNTIME_V55_COMPOSITE_SCHEMA_DRIFT")
    if str(composite.get("base_v6_checkpoint_sha256") or "").lower() != base_sha:
        raise RuntimeError("MIRA_RUNTIME_COMPOSITE_BASE_V6_SHA_DRIFT")
    if str(composite.get("frozen_backbone_sha256") or "") != "embedded-in-base-v6-checkpoint":
        raise RuntimeError("MIRA_RUNTIME_FROZEN_BACKBONE_BINDING_DRIFT")

    terminal = _mapping(
        composite.get("terminal_evaluation"),
        "MIRA_RUNTIME_TERMINAL_EVALUATION_MISSING",
    )
    if terminal.get("full_gate") is not True:
        raise RuntimeError("MIRA_RUNTIME_TERMINAL_EVALUATION_NOT_PASS")
    if str(terminal.get("output_domain") or "") not in {
        "TESSA_COMPILER_M",
        "MECHANICAL_CARRIER_M",
    }:
        raise RuntimeError("MIRA_RUNTIME_TERMINAL_OUTPUT_DOMAIN_DRIFT")
    if terminal.get("finite") is not True or terminal.get("nonnegative") is not True:
        raise RuntimeError("MIRA_RUNTIME_TERMINAL_NUMERIC_GATE_DRIFT")

    backbone_state = _mapping(
        base.get("backbone"),
        "MIRA_RUNTIME_BASE_BACKBONE_STATE_MISSING",
    )
    decoder_state = _mapping(
        composite.get("decoder"),
        "MIRA_RUNTIME_V55_DECODER_STATE_MISSING",
    )
    if _state_parameter_count(backbone_state) != MIRA_V6_BACKBONE_PARAMETER_COUNT:
        raise RuntimeError("MIRA_RUNTIME_BACKBONE_PARAMETER_COUNT_DRIFT")
    if _state_parameter_count(decoder_state) != MIRA_V55_DECODER_PARAMETER_COUNT:
        raise RuntimeError("MIRA_RUNTIME_DECODER_PARAMETER_COUNT_DRIFT")

    return {
        "schema": MIRA_RUNTIME_CHECKPOINT_SCHEMA_V1,
        "architecture_id": MIRA_V55_ARCHITECTURE_ID,
        "backbone_architecture_id": MIRA_V6_BACKBONE_ARCHITECTURE_ID,
        "backbone_state_dict": {
            str(key): value.detach().cpu() for key, value in backbone_state.items()
        },
        "decoder_state_dict": {
            str(key): value.detach().cpu() for key, value in decoder_state.items()
        },
        "source_components": {
            "base_v6_checkpoint_sha256": base_sha,
            "v55_composite_sha256": composite_sha,
            "historical_v6_runner_sha256": MIRA_V55_HISTORICAL_RUNNER_SHA256,
            "carrier_candidate_sha256": str(composite.get("carrier_candidate_sha256") or ""),
            "carrier_support_sha256": str(composite.get("carrier_support_sha256") or ""),
            "carrier_basis_sha256": str(composite.get("carrier_basis_sha256") or ""),
            "axis541_skeleton_sha256": str(composite.get("axis541_skeleton_sha256") or ""),
            "skeleton_lineage_hash": str(composite.get("skeleton_lineage_hash") or ""),
        },
        "runtime_rules": {
            "self_contained_inference_state": True,
            "backbone_frozen": True,
            "decoder_frozen_for_product_inference": True,
            "fit_allowed": False,
            "optimizer_state_packaged": False,
            "teacher_predictor_input_used": False,
            "source_skin_runtime_authority": False,
            "gsa_skin_field_authority_minted": False,
            "semantic_skin_transfer_performed": False,
            "output_domain": "MECHANICAL_CARRIER_M",
            "canonical_skin_authority": "COMPILER_ONLY",
        },
        "historical_terminal_evaluation": dict(terminal),
        "packaging_authority": "RUNTIME_PACKAGING_ONLY__NO_NEW_SCIENTIFIC_PASS",
    }


def save_mira_runtime_checkpoint_v1(
    *,
    output_path: str | Path,
    base_v6_checkpoint_path: str | Path,
    v55_composite_path: str | Path,
    expected_base_v6_sha256: str,
    expected_v55_composite_sha256: str,
) -> MiraRuntimePackageReceiptV1:
    import torch

    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_mira_runtime_checkpoint_v1(
        base_v6_checkpoint_path=base_v6_checkpoint_path,
        v55_composite_path=v55_composite_path,
        expected_base_v6_sha256=expected_base_v6_sha256,
        expected_v55_composite_sha256=expected_v55_composite_sha256,
    )
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    return MiraRuntimePackageReceiptV1(
        output_path=str(output),
        content_sha256=_sha256_file(output),
        base_v6_checkpoint_sha256=_sha256_file(Path(base_v6_checkpoint_path)),
        v55_composite_sha256=_sha256_file(Path(v55_composite_path)),
    )


def load_mira_runtime_model_v1(
    checkpoint_path: str | Path,
    *,
    device: str | Any = "cpu",
):
    """Load only the self-contained runtime package accepted by product.skin."""

    import torch

    from models.arachne.v4.arachne_candidate_v4 import ArachneA1ConfigV4, ArachneA1V4
    from models.mira.direct_readout_v55 import make_mira_v55_readout

    payload = _load_torch_mapping(Path(checkpoint_path).resolve())
    if str(payload.get("schema") or "") != MIRA_RUNTIME_CHECKPOINT_SCHEMA_V1:
        raise RuntimeError("MIRA_RUNTIME_CHECKPOINT_SCHEMA_DRIFT")
    if str(payload.get("architecture_id") or "") != MIRA_V55_ARCHITECTURE_ID:
        raise RuntimeError("MIRA_RUNTIME_CHECKPOINT_ARCHITECTURE_DRIFT")
    if str(payload.get("backbone_architecture_id") or "") != MIRA_V6_BACKBONE_ARCHITECTURE_ID:
        raise RuntimeError("MIRA_RUNTIME_CHECKPOINT_BACKBONE_ARCHITECTURE_DRIFT")

    rules = _mapping(payload.get("runtime_rules"), "MIRA_RUNTIME_CHECKPOINT_RULES_MISSING")
    required = {
        "self_contained_inference_state": True,
        "backbone_frozen": True,
        "decoder_frozen_for_product_inference": True,
        "fit_allowed": False,
        "optimizer_state_packaged": False,
        "teacher_predictor_input_used": False,
        "source_skin_runtime_authority": False,
        "gsa_skin_field_authority_minted": False,
        "semantic_skin_transfer_performed": False,
        "output_domain": "MECHANICAL_CARRIER_M",
        "canonical_skin_authority": "COMPILER_ONLY",
    }
    for key, expected in required.items():
        if rules.get(key) != expected:
            raise RuntimeError(f"MIRA_RUNTIME_CHECKPOINT_RULE_DRIFT:{key}")

    config = ArachneA1ConfigV4()
    backbone = ArachneA1V4(config).cpu().float()
    backbone.load_state_dict(
        dict(
            _mapping(
                payload.get("backbone_state_dict"),
                "MIRA_RUNTIME_BACKBONE_STATE_MISSING",
            )
        ),
        strict=True,
    )
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    backbone.eval()

    decoder = make_mira_v55_readout(
        surface_dim=int(config.model_dim),
        token_dim=int(config.latent_channels),
        pair_dim=int(config.pair_geometry_dim),
    ).cpu().float()
    decoder.load_state_dict(
        dict(
            _mapping(
                payload.get("decoder_state_dict"),
                "MIRA_RUNTIME_DECODER_STATE_MISSING",
            )
        ),
        strict=True,
    )
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    decoder.eval()

    torch_device = torch.device(device)
    backbone.to(torch_device)
    decoder.to(torch_device)
    return backbone, decoder, dict(payload)


__all__ = [
    "MIRA_RUNTIME_CHECKPOINT_SCHEMA_V1",
    "MIRA_RUNTIME_PACKAGE_RECEIPT_SCHEMA_V1",
    "MIRA_V55_ARCHITECTURE_ID",
    "MIRA_V55_CANONICAL_BASE_V6_SHA256",
    "MIRA_V55_CANONICAL_COMPOSITE_SHA256",
    "MIRA_V55_COMPOSITE_SCHEMA",
    "MIRA_V55_HISTORICAL_RUNNER_SHA256",
    "MiraRuntimePackageReceiptV1",
    "build_mira_runtime_checkpoint_v1",
    "load_mira_runtime_model_v1",
    "save_mira_runtime_checkpoint_v1",
]
