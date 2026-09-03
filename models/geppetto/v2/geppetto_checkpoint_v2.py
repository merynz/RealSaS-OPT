from __future__ import annotations

from pathlib import Path
import hashlib
import json
from typing import Any

import torch

from .geppetto_candidate_v2 import GeppettoCandidateConfigV2
from .geppetto_conditioning_v2 import FEATURE_CONTRACT_V2

SCHEMA = "RealSaS.GeppettoCheckpoint.v2"


def _sha(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def checkpoint_payload_v2(model, *, extra_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = model.config
    if not isinstance(cfg, GeppettoCandidateConfigV2):
        raise TypeError("Geppetto V2 checkpoint requires GeppettoCandidateConfigV2")
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    metadata = dict(extra_metadata or {})
    authority = {
        "schema": SCHEMA,
        "architecture_id": cfg.architecture_id,
        "config_hash": cfg.config_hash,
        "feature_contract": FEATURE_CONTRACT_V2,
        "feature_contract_hash": _sha(FEATURE_CONTRACT_V2),
        "dynamic_cardinality": True,
        "resource_policy": cfg.resource_policy,
        "product_max_joint_count": None,
        "compiler_owns_canonical_ids_and_tree": True,
        "full_3d_reconstruction_authority": False,
        "metadata": metadata,
    }
    return {"authority": authority, "state_dict": state}


def save_geppetto_checkpoint_v2(path: str | Path, model, *, extra_metadata: dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = checkpoint_payload_v2(model, extra_metadata=extra_metadata)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_geppetto_checkpoint_v2(path: str | Path, model, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or "authority" not in payload or "state_dict" not in payload:
        raise ValueError("Geppetto checkpoint payload malformed")
    authority = dict(payload["authority"])
    cfg = model.config
    expected = {
        "schema": SCHEMA,
        "architecture_id": cfg.architecture_id,
        "config_hash": cfg.config_hash,
        "feature_contract_hash": _sha(FEATURE_CONTRACT_V2),
        "dynamic_cardinality": True,
        "resource_policy": cfg.resource_policy,
        "product_max_joint_count": None,
        "compiler_owns_canonical_ids_and_tree": True,
        "full_3d_reconstruction_authority": False,
    }
    for key, value in expected.items():
        if authority.get(key) != value:
            raise ValueError(f"Geppetto checkpoint authority mismatch:{key}")
    if tuple(authority.get("feature_contract", ())) != FEATURE_CONTRACT_V2:
        raise ValueError("Geppetto checkpoint feature contract mismatch")
    model.load_state_dict(payload["state_dict"], strict=True)
    return authority
