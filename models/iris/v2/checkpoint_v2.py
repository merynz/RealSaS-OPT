from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import torch


def _trainable_state(model):
    if hasattr(model, "trainable_state_dict"):
        return model.trainable_state_dict(), "TRAINABLE_LEARNER_ONLY"
    return model.state_dict(), "FULL_MODEL"


def _load_trainable_state(model, state, scope: str):
    if scope == "TRAINABLE_LEARNER_ONLY":
        learner = getattr(model, "learner", None)
        if learner is None:
            raise ValueError("checkpoint requires foundation-bound apparatus with learner")
        learner.load_state_dict(state)
        return
    if scope == "FULL_MODEL":
        model.load_state_dict(state)
        return
    raise ValueError(f"unknown checkpoint state scope:{scope}")


def save_checkpoint_v2(path, *, model, optimizer=None, step: int, config: dict, source_contract_hash: str) -> dict:
    path = Path(path)
    state, scope = _trainable_state(model)
    runtime_seal = getattr(model, "runtime_seal", None)
    runtime_seal_payload = dict(runtime_seal.__dict__) if runtime_seal is not None else None
    payload = {
        "schema": "RealSaS.IRISV2.Checkpoint.v2",
        "step": int(step),
        "config": config,
        "source_contract_hash": str(source_contract_hash),
        "state_scope": scope,
        "foundation_runtime_seal": runtime_seal_payload,
        "model": state,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
    }
    torch.save(payload, path)
    digest = sha256(path.read_bytes()).hexdigest()
    meta = {
        "path": str(path),
        "sha256": digest,
        "step": int(step),
        "source_contract_hash": str(source_contract_hash),
        "state_scope": scope,
        "foundation_runtime_seal_hash": getattr(runtime_seal, "seal_hash", None),
    }
    path.with_suffix(path.suffix + ".json").write_text(json.dumps(meta, sort_keys=True, indent=2), encoding="utf-8")
    return meta


def load_checkpoint_v2(path, *, model, optimizer=None, expected_source_contract_hash: str | None = None) -> dict:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("schema") not in {"RealSaS.IRISV2.Checkpoint.v1", "RealSaS.IRISV2.Checkpoint.v2"}:
        raise ValueError("checkpoint schema mismatch")
    if expected_source_contract_hash is not None and payload.get("source_contract_hash") != expected_source_contract_hash:
        raise ValueError("checkpoint source contract mismatch")
    if payload.get("schema") == "RealSaS.IRISV2.Checkpoint.v2":
        expected_seal = payload.get("foundation_runtime_seal")
        actual_seal = getattr(model, "runtime_seal", None)
        if expected_seal is not None:
            if actual_seal is None or dict(actual_seal.__dict__) != expected_seal:
                raise ValueError("foundation runtime seal mismatch")
        _load_trainable_state(model, payload["model"], payload.get("state_scope", "FULL_MODEL"))
    else:
        model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return payload
