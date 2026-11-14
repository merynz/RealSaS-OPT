from __future__ import annotations

"""Fail-closed trainable-parameter ↔ optimizer ownership for RSS.vNext stages.

Removing an active tensor from optimizer.param_groups does NOT freeze it:
backward still writes a gradient while optimizer.zero_grad() clears only
registered tensors, producing stale accumulation and misleading clipping.
A staged trainer must explicitly freeze every excluded student parameter.

This is model-identity/gradient-owner validation, not a choice of curriculum,
optimizer, learning-rate strategy, or a claim of training quality.
"""
from dataclasses import dataclass
from hashlib import sha256
import json

from torch import nn
from torch.optim import Optimizer


SCHEMA = "RealSaS.RSSvNext.TrainingOptimizerOwnership.v1"


@dataclass(frozen=True)
class RSSvNextOptimizerOwnershipReceiptV1:
    schema: str
    optimizer_class: str
    binding_sha256: str
    total_model_parameter_tensors: int
    trainable_model_parameter_tensors: int
    optimizer_registered_parameter_tensors: int
    optimizer_registered_frozen_parameter_tensors: int
    number_of_optimizer_groups: int


def qualify_rss_vnext_optimizer_ownership_v1(
    model: nn.Module, optimizer: Optimizer,
) -> RSSvNextOptimizerOwnershipReceiptV1:
    """Require every trainable model tensor in exactly one optimizer group.

    Frozen model parameters MAY remain in the optimizer for compatibility;
    their no-grad state is included in the binding fingerprint. No parameter
    foreign to the model may be registered.
    """
    if not isinstance(model, nn.Module) or not isinstance(optimizer, Optimizer):
        raise TypeError("RSS_VNEXT_TRAIN_OPTIMIZER_SCOPE_ABI_INVALID")
    named = list(model.named_parameters())
    if not named or not optimizer.param_groups:
        raise ValueError("RSS_VNEXT_TRAIN_OPTIMIZER_SCOPE_EMPTY")
    id_to_name = {id(parameter): name for name, parameter in named}
    if len(id_to_name) != len(named):
        raise RuntimeError("RSS_VNEXT_TRAIN_MODEL_PARAMETER_IDENTITY_DUPLICATE")
    trainable = {id(p) for _, p in named if p.requires_grad}
    if not trainable:
        raise ValueError("RSS_VNEXT_TRAIN_OPTIMIZER_NO_TRAINABLE_PARAMETERS")
    seen: set[int] = set()
    groups = []
    n_frozen_registered = 0
    for group in optimizer.param_groups:
        members = group.get("params")
        if type(members) not in (list, tuple) or not members:
            raise ValueError("RSS_VNEXT_TRAIN_OPTIMIZER_GROUP_EMPTY")
        member_names = []
        for parameter in members:
            key = id(parameter)
            name = id_to_name.get(key)
            if name is None:
                raise ValueError("RSS_VNEXT_TRAIN_OPTIMIZER_FOREIGN_PARAMETER")
            if key in seen:
                raise ValueError("RSS_VNEXT_TRAIN_OPTIMIZER_PARAMETER_DUPLICATE")
            seen.add(key)
            member_names.append(name)
            if key not in trainable:
                n_frozen_registered += 1
        groups.append(member_names)
    missing = trainable - seen
    if missing:
        names = sorted(id_to_name[id_] for id_ in missing)
        raise ValueError(
            "RSS_VNEXT_TRAIN_OPTIMIZER_UNOWNED_TRAINABLE_PARAMETER:"
            + ",".join(names[:5])
        )
    # The exact parameter-group membership is a training-stage identity.
    # Optimizer hyperparameters and moments live in optimizer.state_dict;
    # this receipt separately guarantees that optimizer.state_dict applies
    # to the same model parameter ownership, not an unrelated same-sized
    # group in a different stage.
    payload = {
        "schema": SCHEMA,
        "optimizer_class":type(optimizer).__module__+"."+type(optimizer).__qualname__,
        "groups": groups,
        "trainable_names":sorted(id_to_name[id_] for id_ in trainable),
        "model_architecture":type(model).__module__+"."+type(model).__qualname__,
    }
    digest = sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()
    return RSSvNextOptimizerOwnershipReceiptV1(
        schema=SCHEMA,
        optimizer_class=payload["optimizer_class"],
        binding_sha256=digest,
        total_model_parameter_tensors=len(named),
        trainable_model_parameter_tensors=len(trainable),
        optimizer_registered_parameter_tensors=len(seen),
        optimizer_registered_frozen_parameter_tensors=n_frozen_registered,
        number_of_optimizer_groups=len(groups),
    )


__all__ = [
    "SCHEMA", "RSSvNextOptimizerOwnershipReceiptV1",
    "qualify_rss_vnext_optimizer_ownership_v1",
]
