"""Step-boundary model/optimizer/scheduler/RNG/sampler resume with byte proof.

Drive is a durable checkpoint mirror, never a per-step tensor store. LATEST is
published only after the immutable checkpoint copy has been read back and hashed.
An interrupted publication leaves the previous LATEST usable. CAS import and
Attempt qualification remain Platform responsibilities.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import uuid

import numpy as np
import torch

SCHEMA = "RealSaS.ExternalWorkerResume.v1"
REQUIRED_BINDINGS = {"source_sha", "corpus_sha256", "model_sha256", "recipe_sha256",
                     "execution_plan_sha256", "campaign_id", "attempt_id", "engine_release_id",
                     "precision_policy_sha256"}


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def _bindings(value):
    if (not isinstance(value, dict) or not REQUIRED_BINDINGS.issubset(value)
            or any(not isinstance(k, str) or not isinstance(v, str) or not v for k, v in value.items())):
        raise ValueError("RESUME_EXACT_BINDINGS_REQUIRED")
    return dict(value)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+"."+uuid.uuid4().hex+".partial")
    try:
        with temp.open("w") as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def capture_rng():
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(value):
    random.setstate(value["python"])
    n = value["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(value["torch"])
    if value["cuda"]:
        if not torch.cuda.is_available() or len(value["cuda"]) != torch.cuda.device_count():
            raise ValueError("RESUME_CUDA_RNG_DEVICE_MISMATCH")
        torch.cuda.set_rng_state_all(value["cuda"])


def step_generator(seed, global_step, stream, *, device="cpu"):
    if not isinstance(global_step, int) or global_step < 0 or not isinstance(stream, str) or not stream:
        raise ValueError("SAMPLER_ABSOLUTE_STEP_STREAM_REQUIRED")
    digest = hashlib.sha256(json.dumps([seed, global_step, stream], separators=(",", ":")).encode()).digest()
    return torch.Generator(device=device).manual_seed(int.from_bytes(digest[:8], "little") % (2**63-1))


def save_resume(*, local_dir, durable_dir, model, optimizer, scheduler, global_step, sampler_state, bindings, terminal=False, retain_local=2):
    bindings = _bindings(bindings)
    if not isinstance(global_step, int) or global_step < 0 or retain_local < 1:
        raise ValueError("RESUME_STEP_RETENTION_INVALID")
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError("RESUME_REQUIRES_COMMITTED_STEP_WITH_CLEARED_GRADIENTS")
    local, durable = Path(local_dir), Path(durable_dir)
    if local.resolve() == durable.resolve():
        raise ValueError("RESUME_LOCAL_DURABLE_MUST_DIFFER")
    local.mkdir(parents=True, exist_ok=True)
    durable.mkdir(parents=True, exist_ok=True)
    payload = {"schema": SCHEMA, "bindings": bindings, "global_step": global_step, "terminal": bool(terminal),
               "model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "scheduler": scheduler.state_dict() if scheduler else None,
               "rng": capture_rng(), "sampler": sampler_state}
    temp = local/(uuid.uuid4().hex+".partial")
    with temp.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    sha, size = file_sha256(temp), temp.stat().st_size
    name = f"step-{global_step:09d}-{sha}.pt"
    target = local/name
    os.replace(temp, target)
    remote = durable/name
    remote_temp = durable/(name+"."+uuid.uuid4().hex+".partial")
    try:
        if not remote.is_file() or file_sha256(remote) != sha:
            with target.open("rb") as src, remote_temp.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=8*1024*1024)
                dst.flush()
                os.fsync(dst.fileno())
            if remote_temp.stat().st_size != size or file_sha256(remote_temp) != sha:
                raise ValueError("RESUME_DURABLE_COPY_VERIFICATION_FAILED")
            os.replace(remote_temp, remote)
        receipt = {"schema": SCHEMA, "file": name, "sha256": sha, "size_bytes": size,
                   "bindings": bindings, "global_step": global_step, "terminal": bool(terminal)}
        atomic_json(durable/"LATEST.json", receipt)
        # Drop local snapshots only after durable byte verification/publication.
        older = sorted((p for p in local.glob("step-*.pt") if p != target), key=lambda p: p.stat().st_mtime_ns, reverse=True)
        for p in older[retain_local-1:]:
            p.unlink()
        return receipt
    finally:
        remote_temp.unlink(missing_ok=True)


def load_resume(*, durable_dir, model, optimizer, scheduler, bindings):
    bindings = _bindings(bindings)
    root = Path(durable_dir).resolve()
    receipt = json.loads((root/"LATEST.json").read_text())
    if receipt.get("schema") != SCHEMA or receipt.get("bindings") != bindings:
        raise ValueError("RESUME_BINDING_OR_SCHEMA_DRIFT")
    name = receipt.get("file")
    if not isinstance(name, str) or Path(name).name != name or not name.endswith(".pt"):
        raise ValueError("RESUME_CHECKPOINT_PATH_ESCAPE")
    path = root/name
    if (not path.is_file() or path.is_symlink() or path.stat().st_size != receipt["size_bytes"]
            or file_sha256(path) != receipt["sha256"]):
        raise ValueError("RESUME_CHECKPOINT_BYTES_MISSING_OR_CORRUPT")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if (state.get("schema") != SCHEMA or state.get("bindings") != bindings
            or state.get("global_step") != receipt["global_step"] or state.get("terminal") != receipt["terminal"]):
        raise ValueError("RESUME_PAYLOAD_RECEIPT_DRIFT")
    if (scheduler is None) != (state["scheduler"] is None):
        raise ValueError("RESUME_SCHEDULER_DRIFT")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    if scheduler:
        scheduler.load_state_dict(state["scheduler"])
    optimizer.zero_grad(set_to_none=True)
    restore_rng(state["rng"])
    return {"global_step": state["global_step"], "next_global_step": state["global_step"]+1,
            "sampler_state": state["sampler"], "terminal": state["terminal"], "receipt": receipt}
