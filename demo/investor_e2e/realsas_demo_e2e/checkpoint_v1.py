from __future__ import annotations

from dataclasses import asdict, is_dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
import json

import torch

CHECKPOINT_SCHEMA = "RealSaS.SingleSpecimenDemoCheckpoint.v1"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def config_dict(config: Any) -> dict:
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, dict):
        return dict(config)
    raise TypeError("config must be a dataclass or dict")


def model_parameter_sha256(model: torch.nn.Module) -> str:
    h=sha256()
    for name,tensor in sorted(model.state_dict().items()):
        h.update(name.encode("utf-8"));h.update(b"\0")
        x=tensor.detach().cpu().contiguous()
        h.update(str(x.dtype).encode("ascii"));h.update(str(tuple(x.shape)).encode("ascii"));h.update(x.numpy().tobytes())
    return h.hexdigest()


def save_checkpoint(
    path: str | Path,
    *,
    component: str,
    model: torch.nn.Module,
    config: Any,
    training_observation_sha256: str,
    training_truth_sha256: str,
    fit_verdict: dict,
    source_revision: str,
) -> dict:
    if not training_observation_sha256 or not training_truth_sha256:
        raise ValueError("checkpoint requires observation/truth content hashes")
    metadata={
        "schema":CHECKPOINT_SCHEMA,
        "component":str(component),
        "config":config_dict(config),
        "training_observation_sha256":str(training_observation_sha256),
        "training_truth_sha256":str(training_truth_sha256),
        "fit_verdict":dict(fit_verdict),
        "source_revision":str(source_revision),
        "generalization_claim":False,
        "single_specimen_fit":True,
        "manual_output_injection":False,
    }
    metadata["metadata_sha256"]=sha256(_canonical_json(metadata)).hexdigest()
    payload={"metadata":metadata,"state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()}}
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    torch.save(payload,path)
    metadata["model_parameter_sha256"]=model_parameter_sha256(model)
    return metadata


def load_checkpoint(path: str | Path, model: torch.nn.Module, *, expected_component: str) -> dict:
    payload=torch.load(path,map_location="cpu",weights_only=False)
    metadata=dict(payload.get("metadata") or {})
    if metadata.get("schema")!=CHECKPOINT_SCHEMA: raise RuntimeError("CHECKPOINT_SCHEMA_MISMATCH")
    if metadata.get("component")!=expected_component: raise RuntimeError("CHECKPOINT_COMPONENT_MISMATCH")
    claimed=metadata.pop("metadata_sha256",None)
    actual=sha256(_canonical_json(metadata)).hexdigest()
    metadata["metadata_sha256"]=claimed
    if claimed!=actual: raise RuntimeError("CHECKPOINT_METADATA_SHA_MISMATCH")
    model.load_state_dict(payload["state_dict"],strict=True)
    return metadata
