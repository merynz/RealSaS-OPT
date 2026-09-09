from __future__ import annotations

"""Reference-strength Geppetto arm without learned absolute view-slot identity.

The product camera contract has eight canonical yaw views. That direction is
shipping-known information and must not be hidden from Geppetto, but a free
learned V0..V7 embedding is an avoidable shortcut. This arm replaces it with a
fixed Fourier code derived only from canonical yaw angle.
"""

from dataclasses import replace
import math

import torch

from experiments.geppetto_reference_strength_fullstack_v1.geppetto_reference_strength_candidate_v1 import (
    DirectRiggingSurfaceEncoderV1,
    GeppettoReferenceStrengthCandidateV1,
    GeppettoReferenceStrengthConfigV1,
)


ARCHITECTURE_ID = (
    "RealSaS.Geppetto.ReferenceStrength.DirectSurfaceCausalDiffusion."
    "DeterministicViewDirection.v1"
)
VIEW_CONTRACT = "CANONICAL_8_YAW__45_DEG_INCREMENT__NO_LEARNED_SLOT_ID"


def deterministic_canonical_view_code_v1(view_dim: int) -> torch.Tensor:
    """Return [8,view_dim] fixed Fourier features of canonical yaw direction."""
    if view_dim < 2:
        raise ValueError("view_dim must be >=2")
    yaw = torch.arange(8, dtype=torch.float32) * (2.0 * math.pi / 8.0)
    pairs = (view_dim + 1) // 2
    freq = torch.arange(1, pairs + 1, dtype=torch.float32)
    angle = yaw[:, None] * freq[None]
    code = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1).reshape(8, -1)
    code = code[:, :view_dim].contiguous()
    if code.shape != (8, view_dim) or not torch.isfinite(code).all():
        raise RuntimeError("deterministic view code construction failed")
    return code


class DeterministicViewRiggingSurfaceEncoderV1(DirectRiggingSurfaceEncoderV1):
    def __init__(self, cfg: GeppettoReferenceStrengthConfigV1):
        super().__init__(cfg)
        # Remove the free learned slot parameter created by the historical
        # base encoder, then install a non-trainable camera-direction buffer at
        # the same field name so the inherited forward path is unchanged.
        if "view_index" not in self._parameters:
            raise RuntimeError("base encoder view-index parameter contract drift")
        del self._parameters["view_index"]
        self.register_buffer(
            "view_index",
            deterministic_canonical_view_code_v1(cfg.view_dim),
            persistent=True,
        )


class GeppettoReferenceStrengthNoLearnedSlotV1(GeppettoReferenceStrengthCandidateV1):
    def __init__(
        self,
        config: GeppettoReferenceStrengthConfigV1 | None = None,
    ):
        cfg = config or replace(
            GeppettoReferenceStrengthConfigV1(),
            architecture_id=ARCHITECTURE_ID,
        )
        if cfg.architecture_id != ARCHITECTURE_ID:
            raise ValueError("no-slot candidate requires its frozen architecture_id")
        super().__init__(cfg)
        self.encoder = DeterministicViewRiggingSurfaceEncoderV1(cfg)

    @property
    def view_contract(self) -> str:
        return VIEW_CONTRACT


def assert_no_learned_view_slot_identity_v1(model: GeppettoReferenceStrengthNoLearnedSlotV1) -> None:
    bad = [name for name, _ in model.named_parameters() if name.endswith("view_index")]
    if bad:
        raise RuntimeError(f"learned absolute view-slot identity present:{bad}")
    code = dict(model.named_buffers()).get("encoder.view_index")
    if code is None or tuple(code.shape) != (8, model.config.view_dim):
        raise RuntimeError("deterministic camera-direction buffer missing")


__all__ = [
    "ARCHITECTURE_ID",
    "VIEW_CONTRACT",
    "deterministic_canonical_view_code_v1",
    "DeterministicViewRiggingSurfaceEncoderV1",
    "GeppettoReferenceStrengthNoLearnedSlotV1",
    "assert_no_learned_view_slot_identity_v1",
]
