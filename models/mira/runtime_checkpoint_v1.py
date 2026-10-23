from __future__ import annotations

"""Self-contained runtime packaging for the sealed FIT1 MIRA V5.5 model.

The V5.5 experiment froze the V6/V4 rich backbone and refit only the direct
carrier-native decoder on the exact Knight mechanical carrier. Runtime packaging
copies the frozen backbone state from the sealed V6 checkpoint and the terminal
V5.5 decoder state from the V5.5 composite. Teacher arrays, optimizer state and
source skin never enter the runtime package.

This FIT1 release is intentionally carrier/rig-bound. It is a regression
baseline, not a generalization claim. A future generic MIRA ModelRelease may
replace this implementation without changing the stable product.skin role.
"""

import hashlib
from pathlib import Path
from typing import Any, Mapping

from compiler.realsas_compiler_core.types import QualificationError
from models.mira.carrier_native_v55 import MIRA_V55_ARCHITECTURE_ID

MIRA_RUNTIME_SCHEMA_V1 = "RealSaS.MIRARuntimeCheckpoint.v1"
MIRA_V6_SOURCE_ARCHITECTURE_ID = "RealSaS.Arachne.A1.RawSurfaceK4AttentionDirectSimplex.v6"
MIRA_BACKBONE_ARCHITECTURE_ID = "RealSaS.Arachne.A1.RichQualifiedBidirectional.v4"
MIRA_V6_CHECKPOINT_SCHEMA = "RealSaS.KnightArachneV6DemoClosure.v1.Model.v1"
MIRA_V55_COMPOSITE_SCHEMA = "RealSaS.MIRAMDirectComposite.v5_5"
EXPECTED_V6_SOURCE_SHA256 = "39ff4f7a2051f0d9c1251a5999f472b9926e83deacbedcedc8e92eea424dce01"
EXPECTED_FIT1_V6_CHECKPOINT_SHA256 = "bfd2bd24f55bdce2dc36b0ae6a7afe01b8381c680de545c66a11a3d707fe73ed"
EXPECTED_FIT1_CARRIER_BASIS_SHA256 = "dbac301b9c47fa7f31a024ef0612008d8315b590d7a2dd1552a75f75c7ac9ff2"
EXPECTED_FIT1_AXIS_SKELETON_SHA256 = "c5a3c7f5675ba84233c8d7eb4ccab076e69c105462f86673dc259e30e2ca6ca5"
EXPECTED_FIT1_W_M_SHA256 = "880dd73b2f3a21196c229156a3271fa95e7261def3f50a3dfe9f1c49ca2bb27b"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _require_torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - runtime environment guard
        raise QualificationError("MIRA_RUNTIME_TORCH_REQUIRED") from exc
    return torch


def _state_dict_cpu(value: Mapping[str, Any]) -> dict[str, Any]:
    torch = _require_torch()
    out: dict[str, Any] = {}
    for key, tensor in value.items():
        if not isinstance(tensor, torch.Tensor):
            raise QualificationError("MIRA_RUNTIME_STATE_DICT_NON_TENSOR")
        out[str(key)] = tensor.detach().cpu().contiguous()
    if not out:
        raise QualificationError("MIRA_RUNTIME_STATE_DICT_EMPTY")
    return out


def build_mira_runtime_checkpoint_v1(
    *,
    v6_checkpoint_path: Path,
    v55_composite_path: Path,
    expected_v55_composite_sha256: str | None = None,
) -> dict[str, Any]:
    torch = _require_torch()
    v6_sha = _sha256_file(v6_checkpoint_path)
    composite_sha = _sha256_file(v55_composite_path)
    if v6_sha != EXPECTED_FIT1_V6_CHECKPOINT_SHA256:
        raise QualificationError("MIRA_RUNTIME_V6_CHECKPOINT_SHA_DRIFT")
    if expected_v55_composite_sha256 is not None and composite_sha != str(expected_v55_composite_sha256):
        raise QualificationError("MIRA_RUNTIME_V55_COMPOSITE_SHA_DRIFT")

    v6 = torch.load(v6_checkpoint_path, map_location="cpu", weights_only=False)
    composite = torch.load(v55_composite_path, map_location="cpu", weights_only=False)
    if str(v6.get("schema") or "") != MIRA_V6_CHECKPOINT_SCHEMA:
        raise QualificationError("MIRA_RUNTIME_V6_SCHEMA_DRIFT")
    if str(v6.get("architecture_id") or "") != MIRA_V6_SOURCE_ARCHITECTURE_ID:
        raise QualificationError("MIRA_RUNTIME_V6_ARCHITECTURE_DRIFT")
    if str(v6.get("backbone_architecture_id") or "") != MIRA_BACKBONE_ARCHITECTURE_ID:
        raise QualificationError("MIRA_RUNTIME_BACKBONE_ARCHITECTURE_DRIFT")
    if v6.get("teacher_predictor_input_used") is not False:
        raise QualificationError("MIRA_RUNTIME_TEACHER_PREDICTOR_FORBIDDEN")
    if str(composite.get("schema") or "") != MIRA_V55_COMPOSITE_SCHEMA:
        raise QualificationError("MIRA_RUNTIME_V55_COMPOSITE_SCHEMA_DRIFT")
    if str(composite.get("base_v6_checkpoint_sha256") or "") != v6_sha:
        raise QualificationError("MIRA_RUNTIME_V55_BASE_BINDING_DRIFT")
    if str(composite.get("carrier_basis_sha256") or "") != EXPECTED_FIT1_CARRIER_BASIS_SHA256:
        raise QualificationError("MIRA_RUNTIME_TRAINED_CARRIER_DRIFT")
    if str(composite.get("axis541_skeleton_sha256") or "") != EXPECTED_FIT1_AXIS_SKELETON_SHA256:
        raise QualificationError("MIRA_RUNTIME_TRAINED_RIG_FILE_DRIFT")
    skeleton_lineage = str(composite.get("skeleton_lineage_hash") or "").strip()
    if not skeleton_lineage:
        raise QualificationError("MIRA_RUNTIME_TRAINED_RIG_LINEAGE_MISSING")
    terminal = dict(composite.get("terminal_evaluation") or {})
    if terminal.get("full_gate") is not True:
        raise QualificationError("MIRA_RUNTIME_TERMINAL_EVALUATION_NOT_PASS")

    package = {
        "schema": MIRA_RUNTIME_SCHEMA_V1,
        "architecture_id": MIRA_V55_ARCHITECTURE_ID,
        "source_architecture_id": MIRA_V6_SOURCE_ARCHITECTURE_ID,
        "backbone_architecture_id": MIRA_BACKBONE_ARCHITECTURE_ID,
        "v6_source_sha256": EXPECTED_V6_SOURCE_SHA256,
        "base_v6_checkpoint_sha256": v6_sha,
        "v55_composite_sha256": composite_sha,
        "trained_carrier_basis_sha256": EXPECTED_FIT1_CARRIER_BASIS_SHA256,
        "trained_axis_skeleton_file_sha256": EXPECTED_FIT1_AXIS_SKELETON_SHA256,
        "trained_skeleton_lineage_hash": skeleton_lineage,
        "sealed_fit1_w_m_sha256": EXPECTED_FIT1_W_M_SHA256,
        "backbone": _state_dict_cpu(v6["backbone"]),
        "decoder": _state_dict_cpu(composite["decoder"]),
        "terminal_evaluation": terminal,
        "runtime_rules": {
            "fit_allowed": False,
            "optimizer_state_present": False,
            "teacher_arrays_present": False,
            "teacher_predictor_input_allowed": False,
            "source_skin_runtime_authority": False,
            "semantic_skin_transfer_allowed": False,
            "gsa_skin_field_authority_allowed": False,
            "gsa_role": "EVIDENCE_SUBSTRATE_ONLY",
            "output_domain": "CANONICAL_MESH_CARRIER",
            "direct_simplex_readout": True,
            "mesh_mutation_invalidates_prediction": True,
            "rig_mutation_invalidates_prediction": True,
            "generalization_claimed": False,
            "fit1_carrier_bound": True,
            "fit1_rig_bound": True,
        },
        "packaging_authority": "RUNTIME_PACKAGING_ONLY__NO_NEW_SCIENTIFIC_PASS",
    }
    return package


def validate_mira_runtime_checkpoint_v1(payload: Mapping[str, Any]) -> None:
    if str(payload.get("schema") or "") != MIRA_RUNTIME_SCHEMA_V1:
        raise QualificationError("MIRA_RUNTIME_SCHEMA_DRIFT")
    if str(payload.get("architecture_id") or "") != MIRA_V55_ARCHITECTURE_ID:
        raise QualificationError("MIRA_RUNTIME_ARCHITECTURE_DRIFT")
    if str(payload.get("source_architecture_id") or "") != MIRA_V6_SOURCE_ARCHITECTURE_ID:
        raise QualificationError("MIRA_RUNTIME_SOURCE_ARCHITECTURE_DRIFT")
    if str(payload.get("backbone_architecture_id") or "") != MIRA_BACKBONE_ARCHITECTURE_ID:
        raise QualificationError("MIRA_RUNTIME_BACKBONE_ARCHITECTURE_DRIFT")
    if str(payload.get("v6_source_sha256") or "") != EXPECTED_V6_SOURCE_SHA256:
        raise QualificationError("MIRA_RUNTIME_SOURCE_SHA_DRIFT")
    if str(payload.get("trained_carrier_basis_sha256") or "") != EXPECTED_FIT1_CARRIER_BASIS_SHA256:
        raise QualificationError("MIRA_RUNTIME_TRAINED_CARRIER_DRIFT")
    if str(payload.get("trained_axis_skeleton_file_sha256") or "") != EXPECTED_FIT1_AXIS_SKELETON_SHA256:
        raise QualificationError("MIRA_RUNTIME_TRAINED_RIG_FILE_DRIFT")
    if not str(payload.get("trained_skeleton_lineage_hash") or "").strip():
        raise QualificationError("MIRA_RUNTIME_TRAINED_RIG_LINEAGE_MISSING")
    rules = dict(payload.get("runtime_rules") or {})
    required = {
        "fit_allowed": False,
        "optimizer_state_present": False,
        "teacher_arrays_present": False,
        "teacher_predictor_input_allowed": False,
        "source_skin_runtime_authority": False,
        "semantic_skin_transfer_allowed": False,
        "gsa_skin_field_authority_allowed": False,
        "gsa_role": "EVIDENCE_SUBSTRATE_ONLY",
        "output_domain": "CANONICAL_MESH_CARRIER",
        "direct_simplex_readout": True,
        "mesh_mutation_invalidates_prediction": True,
        "rig_mutation_invalidates_prediction": True,
        "generalization_claimed": False,
        "fit1_carrier_bound": True,
        "fit1_rig_bound": True,
    }
    for key, expected in required.items():
        if rules.get(key) != expected:
            raise QualificationError(f"MIRA_RUNTIME_RULE_DRIFT:{key}")
    _state_dict_cpu(payload.get("backbone") or {})
    _state_dict_cpu(payload.get("decoder") or {})


def save_mira_runtime_checkpoint_v1(payload: Mapping[str, Any], path: Path) -> None:
    validate_mira_runtime_checkpoint_v1(payload)
    torch = _require_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(payload), path)


def load_mira_runtime_checkpoint_v1(path: Path) -> dict[str, Any]:
    torch = _require_torch()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise QualificationError("MIRA_RUNTIME_PAYLOAD_INVALID")
    validate_mira_runtime_checkpoint_v1(payload)
    return payload


__all__ = [
    "EXPECTED_FIT1_AXIS_SKELETON_SHA256",
    "EXPECTED_FIT1_CARRIER_BASIS_SHA256",
    "EXPECTED_FIT1_V6_CHECKPOINT_SHA256",
    "EXPECTED_FIT1_W_M_SHA256",
    "MIRA_BACKBONE_ARCHITECTURE_ID",
    "MIRA_RUNTIME_SCHEMA_V1",
    "MIRA_V55_COMPOSITE_SCHEMA",
    "MIRA_V6_CHECKPOINT_SCHEMA",
    "MIRA_V6_SOURCE_ARCHITECTURE_ID",
    "build_mira_runtime_checkpoint_v1",
    "load_mira_runtime_checkpoint_v1",
    "save_mira_runtime_checkpoint_v1",
    "validate_mira_runtime_checkpoint_v1",
]
