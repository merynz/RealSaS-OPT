from __future__ import annotations

"""Training-only loss contract for the reference-strength Geppetto FIT arm.

Teacher target values supervise outputs but never become recurrent inputs.
The module intentionally does not import legacy Geppetto losses or conditioning.
"""

from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn.functional as F

from experiments.geppetto_reference_strength_fullstack_v1.geppetto_reference_strength_candidate_v1 import (
    GeppettoReferenceStrengthRawOutputV1,
)
from experiments.geppetto_reference_strength_fullstack_v1.mechanical_core_target_v1 import (
    MechanicalCoreTargetV1,
)
from experiments.geppetto_reference_strength_fullstack_v1.rigging_surface_tensorization_v1 import (
    RiggingSurfaceTensorV1,
)


SCHEMA = "RealSaS.GeppettoReferenceStrengthLoss.v1"


@dataclass(frozen=True)
class GeppettoReferenceStrengthLossConfigV1:
    position_smooth_l1: float = 4.0
    position_nll: float = 0.25
    diffusion: float = 0.50
    stop: float = 1.0
    existence: float = 0.25
    root: float = 1.0
    internal_parent: float = 1.0
    final_parent: float = 1.5
    support_presence: float = 0.25
    support_index: float = 0.25
    mechanical_salience: float = 0.10
    support_target_k: int = 8

    def validate(self) -> None:
        values = (
            self.position_smooth_l1,
            self.position_nll,
            self.diffusion,
            self.stop,
            self.existence,
            self.root,
            self.internal_parent,
            self.final_parent,
            self.support_presence,
            self.support_index,
            self.mechanical_salience,
        )
        if any((not math.isfinite(float(x))) or float(x) < 0 for x in values):
            raise ValueError("loss weights must be finite and non-negative")
        if self.support_target_k < 1:
            raise ValueError("support_target_k must be positive")


@dataclass(frozen=True)
class PreparedReferenceStrengthTargetV1:
    positions_normalized: np.ndarray
    parent_indices: np.ndarray
    root_mask: np.ndarray
    nearest_surface_indices: np.ndarray
    schema_version: str = SCHEMA

    @property
    def count(self) -> int:
        return int(len(self.positions_normalized))


def prepare_reference_strength_target_v1(
    surface: RiggingSurfaceTensorV1,
    target: MechanicalCoreTargetV1,
    *,
    support_target_k: int = 8,
) -> PreparedReferenceStrengthTargetV1:
    if support_target_k < 1 or support_target_k > surface.node_count:
        raise ValueError("support_target_k outside surface cardinality")
    world = np.asarray(target.positions_world, dtype=np.float64)
    parents = np.asarray(target.parent_indices, dtype=np.int64)
    roots = np.asarray(target.root_mask, dtype=bool)
    if world.ndim != 2 or world.shape[1] != 3 or len(world) < 1:
        raise ValueError("mechanical target positions must be [J,3]")
    if parents.shape != (len(world),) or roots.shape != (len(world),):
        raise ValueError("mechanical target topology shape mismatch")
    if not np.isfinite(world).all():
        raise ValueError("mechanical target positions non-finite")
    center = np.asarray(surface.normalization_center, dtype=np.float64)
    scale = float(surface.normalization_scale)
    if center.shape != (3,) or not np.isfinite(center).all() or not math.isfinite(scale) or scale <= 0:
        raise ValueError("surface normalization contract invalid")
    pn = ((world - center[None, :]) / scale).astype(np.float32)
    sp = np.asarray(surface.positions_normalized, dtype=np.float64)
    dist2 = np.sum((pn[:, None, :].astype(np.float64) - sp[None, :, :]) ** 2, axis=-1)
    # Stable mergesort preserves deterministic surface-row order for exact ties.
    nearest = np.argsort(dist2, axis=1, kind="mergesort")[:, :support_target_k].astype(np.int64)
    return PreparedReferenceStrengthTargetV1(
        positions_normalized=pn,
        parent_indices=parents.copy(),
        root_mask=roots.copy(),
        nearest_surface_indices=nearest,
    )


def _parent_losses(
    out: GeppettoReferenceStrengthRawOutputV1,
    parents: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = out.coarse_positions_normalized.device
    zero = torch.zeros((), device=device, dtype=out.coarse_positions_normalized.dtype)
    internal_terms = []
    final_terms = []
    j = int(len(parents))
    for child in range(j):
        parent = int(parents[child].item())
        if parent < 0:
            continue
        if parent >= child:
            raise ValueError("teacher target must be parent-before-child serialized")
        ilogits = out.internal_parent_logits[0, child, :child]
        if ilogits.numel() != child:
            raise ValueError("internal parent logit shape mismatch")
        internal_terms.append(
            F.cross_entropy(ilogits[None], torch.tensor([parent], device=device))
        )
        flogits = out.all_pair_parent_logits[0, child, :j]
        final_terms.append(
            F.cross_entropy(flogits[None], torch.tensor([parent], device=device))
        )
    return (
        torch.stack(internal_terms).mean() if internal_terms else zero,
        torch.stack(final_terms).mean() if final_terms else zero,
    )


def reference_strength_loss_v1(
    out: GeppettoReferenceStrengthRawOutputV1,
    target: PreparedReferenceStrengthTargetV1,
    *,
    config: GeppettoReferenceStrengthLossConfigV1 = GeppettoReferenceStrengthLossConfigV1(),
) -> dict[str, torch.Tensor]:
    config.validate()
    device = out.coarse_positions_normalized.device
    dtype = out.coarse_positions_normalized.dtype
    j = target.count
    if out.coarse_positions_normalized.shape != (1, j, 3):
        raise ValueError("training output/target cardinality mismatch")
    if out.teacher_feedback_used:
        raise RuntimeError("teacher feedback entered reference-strength recurrence")
    if not out.teacher_target_used:
        raise ValueError("training loss requires teacher objective target")

    pos = torch.as_tensor(target.positions_normalized, device=device, dtype=dtype)
    parents = torch.as_tensor(target.parent_indices, device=device, dtype=torch.long)
    roots = torch.as_tensor(target.root_mask, device=device, dtype=dtype)
    nearest = torch.as_tensor(target.nearest_surface_indices, device=device, dtype=torch.long)

    err = out.coarse_positions_normalized[0] - pos
    position_smooth = F.smooth_l1_loss(out.coarse_positions_normalized[0], pos, beta=0.02)
    log_sigma = out.position_log_sigma[0]
    inv_var = torch.exp(-2.0 * log_sigma)
    position_nll = (0.5 * err.square() * inv_var + log_sigma).mean()

    stop_target = torch.zeros((j,), device=device, dtype=dtype)
    stop_target[-1] = 1.0
    stop = F.binary_cross_entropy_with_logits(out.stop_logits[0], stop_target)
    existence = F.binary_cross_entropy_with_logits(
        out.existence_logits[0], torch.ones((j,), device=device, dtype=dtype)
    )
    root = F.binary_cross_entropy_with_logits(out.root_logits[0], roots)
    internal_parent, final_parent = _parent_losses(out, parents)

    support_presence = F.binary_cross_entropy_with_logits(
        out.support_presence_logits[0], torch.ones((j,), device=device, dtype=dtype)
    )
    support_terms = []
    for child in range(j):
        logits = out.support_logits[0, child]
        logp = F.log_softmax(logits, dim=-1)
        # A mechanical locus may legitimately be interior. Teacher geometry is
        # used only to define a small nearest-surface target set; no source bone
        # identity or skin value is exposed to the learner.
        support_terms.append(-logp[nearest[child]].mean())
    support_index = torch.stack(support_terms).mean()

    # Every row in the projected target survived the mechanical-core rule and
    # is therefore positive mechanical salience for this FIT target. This does
    # not establish calibrated salience magnitude or generalization.
    salience = F.binary_cross_entropy_with_logits(
        out.salience_logits[0], torch.ones((j,), device=device, dtype=dtype)
    )
    diffusion = out.diffusion_loss
    if diffusion is None:
        raise ValueError("training output missing diffusion objective")

    pieces = {
        "position_smooth_l1": position_smooth,
        "position_nll": position_nll,
        "diffusion": diffusion,
        "stop": stop,
        "existence": existence,
        "root": root,
        "internal_parent": internal_parent,
        "final_parent": final_parent,
        "support_presence": support_presence,
        "support_index": support_index,
        "mechanical_salience": salience,
    }
    weights = {
        "position_smooth_l1": config.position_smooth_l1,
        "position_nll": config.position_nll,
        "diffusion": config.diffusion,
        "stop": config.stop,
        "existence": config.existence,
        "root": config.root,
        "internal_parent": config.internal_parent,
        "final_parent": config.final_parent,
        "support_presence": config.support_presence,
        "support_index": config.support_index,
        "mechanical_salience": config.mechanical_salience,
    }
    total = sum(pieces[name] * float(weights[name]) for name in pieces)
    pieces["total"] = total
    if not all(bool(torch.isfinite(v).all()) for v in pieces.values()):
        raise FloatingPointError("non-finite reference-strength loss")
    return pieces


__all__ = [
    "SCHEMA",
    "GeppettoReferenceStrengthLossConfigV1",
    "PreparedReferenceStrengthTargetV1",
    "prepare_reference_strength_target_v1",
    "reference_strength_loss_v1",
]
