from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import torch


def save_checkpoint_v2(path, *, model, optimizer=None, step: int, config: dict, source_contract_hash: str) -> dict:
    path = Path(path)
    payload = {
        "schema": "RealSaS.IRISV2.Checkpoint.v1",
        "step": int(step),
        "config": config,
        "source_contract_hash": str(source_contract_hash),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
    }
    torch.save(payload, path)
    digest = sha256(path.read_bytes()).hexdigest()
    meta = {"path": str(path), "sha256": digest, "step": int(step), "source_contract_hash": str(source_contract_hash)}
    path.with_suffix(path.suffix + ".json").write_text(json.dumps(meta, sort_keys=True, indent=2), encoding="utf-8")
    return meta


def load_checkpoint_v2(path, *, model, optimizer=None, expected_source_contract_hash: str | None = None) -> dict:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("schema") != "RealSaS.IRISV2.Checkpoint.v1":
        raise ValueError("checkpoint schema mismatch")
    if expected_source_contract_hash is not None and payload.get("source_contract_hash") != expected_source_contract_hash:
        raise ValueError("checkpoint source contract mismatch")
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return payload
