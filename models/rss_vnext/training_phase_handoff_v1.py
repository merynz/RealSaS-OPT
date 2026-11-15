from __future__ import annotations

"""Explicit weights-only handoff between RSS.vNext training *domains*.

A normal trainer checkpoint is only a same-policy resume. Handoff intentionally
starts a NEW optimizer and a NEW global-step clock. It never imports moments,
scheduler state, a source admission registry, or sampler position. Registries
must be independently provided to the destination trainer.

The exact embedded RSS.vNext MIRA codec may also transfer into structural
training, but only as a full unified-model state. No second-model codec
checkpoint or unqualified external skin tokens are accepted. This is a transport contract, not proof of training quality.
"""

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from .training_optimizer_ownership_v1 import qualify_rss_vnext_optimizer_ownership_v1


SCHEMA = "RealSaS.RSSvNext.TrainingPhaseHandoff.v1"
TRANSITIONS = frozenset({
    ("SLRM", "STRUCTURAL"),
    ("MIRA_CODEC", "STRUCTURAL"),
    ("STRUCTURAL", "STRUCTURAL"),
})


def _hex_digest(value: str) -> bool:
    return type(value) is str and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def _model_state_digest(state: dict[str, Tensor]) -> str:
    """Digest actual tensor bits, including BF16, rather than metadata alone."""
    if type(state) is not dict or not state:
        raise ValueError("RSS_VNEXT_PHASE_HANDOFF_EMPTY_MODEL_STATE")
    digest = sha256()
    for name, value in sorted(state.items()):
        if type(name) is not str or not name or not isinstance(value, Tensor):
            raise ValueError("RSS_VNEXT_PHASE_HANDOFF_NON_TENSOR_STATE")
        cpu = value.detach().contiguous().cpu()
        header = json.dumps(
            [name, str(cpu.dtype), list(cpu.shape)],
            separators=(",", ":"), ensure_ascii=True,
        ).encode("utf8")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class RSSvNextTrainingPhaseHandoffReceiptV1:
    schema: str
    source_kind: str
    target_kind: str
    source_global_step: int
    dataset_manifest_sha256: str
    source_objective_policy_sha256: str
    source_optimizer_binding_sha256: str
    architecture_id: str
    architecture_digest: str
    model_state_sha256: str


def _identity(model: nn.Module, architecture_id: str, architecture_digest: str,
              dataset_manifest_sha256: str) -> None:
    if (
        not isinstance(model, nn.Module)
        or type(architecture_id) is not str or not architecture_id
        or not _hex_digest(architecture_digest)
        or not _hex_digest(dataset_manifest_sha256)
    ):
        raise ValueError("RSS_VNEXT_PHASE_HANDOFF_IDENTITY_INVALID")


def export_rss_vnext_training_phase_handoff_v1(
    path: str | Path,
    *,
    model: nn.Module,
    source_kind: str,
    target_kind: str,
    source_global_step: int,
    dataset_manifest_sha256: str,
    source_objective_policy_sha256: str,
    source_optimizer_binding_sha256: str,
    architecture_id: str,
    architecture_digest: str,
) -> RSSvNextTrainingPhaseHandoffReceiptV1:
    _identity(model, architecture_id, architecture_digest, dataset_manifest_sha256)
    if (source_kind, target_kind) not in TRANSITIONS:
        raise ValueError("RSS_VNEXT_PHASE_HANDOFF_TRANSITION_NOT_ALLOWED")
    if (
        type(source_global_step) is not int or source_global_step < 1
        or not _hex_digest(source_objective_policy_sha256)
        or not _hex_digest(source_optimizer_binding_sha256)
    ):
        raise ValueError("RSS_VNEXT_PHASE_HANDOFF_SOURCE_POLICY_INVALID")
    state = dict(model.state_dict())
    digest = _model_state_digest(state)
    receipt = RSSvNextTrainingPhaseHandoffReceiptV1(
        schema=SCHEMA,
        source_kind=source_kind,
        target_kind=target_kind,
        source_global_step=source_global_step,
        dataset_manifest_sha256=dataset_manifest_sha256,
        source_objective_policy_sha256=source_objective_policy_sha256,
        source_optimizer_binding_sha256=source_optimizer_binding_sha256,
        architecture_id=architecture_id,
        architecture_digest=architecture_digest,
        model_state_sha256=digest,
    )
    from dataclasses import asdict
    filename = Path(path)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=filename.parent, prefix="rss-phase-", suffix=".pt", delete=False,
    ) as file:
        temporary = Path(file.name)
    try:
        torch.save({"receipt": asdict(receipt), "model": state}, temporary)
        os.replace(temporary, filename)
    finally:
        temporary.unlink(missing_ok=True)
    return receipt


def import_rss_vnext_training_phase_handoff_v1(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    target_kind: str,
    expected_source_kind: str,
    expected_source_objective_policy_sha256: str,
    expected_source_optimizer_binding_sha256: str,
    dataset_manifest_sha256: str,
    architecture_id: str,
    architecture_digest: str,
) -> RSSvNextTrainingPhaseHandoffReceiptV1:
    _identity(model, architecture_id, architecture_digest, dataset_manifest_sha256)
    if (expected_source_kind, target_kind) not in TRANSITIONS:
        raise ValueError("RSS_VNEXT_PHASE_HANDOFF_TRANSITION_NOT_ALLOWED")
    if (
        not _hex_digest(expected_source_objective_policy_sha256)
        or not _hex_digest(expected_source_optimizer_binding_sha256)
    ):
        raise ValueError("RSS_VNEXT_PHASE_HANDOFF_EXPECTED_SOURCE_INVALID")
    # A phase transition cannot silently reuse optimizer moments or stale grads.
    qualify_rss_vnext_optimizer_ownership_v1(model, optimizer)
    if any(bool(group) for group in optimizer.state.values()):
        raise ValueError("RSS_VNEXT_PHASE_HANDOFF_DESTINATION_OPTIMIZER_NOT_FRESH")
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if type(payload) is not dict or type(payload.get("receipt")) is not dict:
        raise ValueError("RSS_VNEXT_PHASE_HANDOFF_PAYLOAD_INVALID")
    data = payload["receipt"]
    expected = {
        "schema": SCHEMA,
        "source_kind": expected_source_kind,
        "target_kind": target_kind,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "source_objective_policy_sha256": expected_source_objective_policy_sha256,
        "source_optimizer_binding_sha256": expected_source_optimizer_binding_sha256,
        "architecture_id": architecture_id,
        "architecture_digest": architecture_digest,
    }
    if any(data.get(key) != value for key, value in expected.items()):
        raise ValueError("RSS_VNEXT_PHASE_HANDOFF_PROVENANCE_DRIFT")
    if type(data.get("source_global_step")) is not int or data["source_global_step"] < 1:
        raise ValueError("RSS_VNEXT_PHASE_HANDOFF_SOURCE_PROGRESS_INVALID")
    state = payload.get("model")
    if type(state) is not dict or data.get("model_state_sha256") != _model_state_digest(state):
        raise ValueError("RSS_VNEXT_PHASE_HANDOFF_MODEL_DIGEST_DRIFT")
    current = model.state_dict()
    if (
        set(current) != set(state)
        or any(current[k].shape != v.shape or current[k].dtype != v.dtype
               for k, v in state.items())
    ):
        raise ValueError("RSS_VNEXT_PHASE_HANDOFF_MODEL_ABI_DRIFT")
    if any(not bool(torch.isfinite(v).all()) for v in state.values()
           if torch.is_floating_point(v) or torch.is_complex(v)):
        raise ValueError("RSS_VNEXT_PHASE_HANDOFF_NONFINITE_MODEL")
    # All admission/optimizer checks happen before changing a single weight.
    model.load_state_dict(state, strict=True)
    model.zero_grad(set_to_none=True)
    optimizer.zero_grad(set_to_none=True)
    return RSSvNextTrainingPhaseHandoffReceiptV1(**data)


__all__ = [
    "SCHEMA",
    "TRANSITIONS",
    "RSSvNextTrainingPhaseHandoffReceiptV1",
    "export_rss_vnext_training_phase_handoff_v1",
    "import_rss_vnext_training_phase_handoff_v1",
]
