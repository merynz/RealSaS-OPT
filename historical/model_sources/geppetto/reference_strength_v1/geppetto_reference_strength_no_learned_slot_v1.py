from __future__ import annotations

"""Current FIT1-frozen Geppetto entry point.

This file is the mainline import-rebound form of the scientifically frozen
`geppetto_reference_strength_no_learned_slot_v1.py` from source commit
`f7be46f0a97df62a793ebf91b22297c894854f39` / Git blob
`770909a9c2992ada0f3018badb2a42632a124b3b`.

The only semantic-home change is that the frozen base candidate is imported
from the current `models.geppetto.reference_strength_v1` package instead of the
dated experiment package. The deterministic camera-direction construction,
architecture id and no-learned-slot assertion are unchanged.
"""

from dataclasses import replace
import math

import torch

from .geppetto_reference_strength_candidate_v1 import (
    DirectRiggingSurfaceEncoderV1,
    GeppettoReferenceStrengthCandidateV1,
    GeppettoReferenceStrengthConfigV1,
)


ARCHITECTURE_ID = (
    "RealSaS.Geppetto.ReferenceStrength.DirectSurfaceCausalDiffusion."
    "DeterministicViewDirection.v1"
)
VIEW_CONTRACT = "CANONICAL_8_YAW__45_DEG_INCREMENT__NO_LEARNED_SLOT_ID"
FROZEN_SOURCE_COMMIT = "f7be46f0a97df62a793ebf91b22297c894854f39"
FROZEN_SOURCE_BLOB = "770909a9c2992ada0f3018badb2a42632a124b3b"


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
    "FROZEN_SOURCE_COMMIT",
    "FROZEN_SOURCE_BLOB",
    "deterministic_canonical_view_code_v1",
    "DeterministicViewRiggingSurfaceEncoderV1",
    "GeppettoReferenceStrengthNoLearnedSlotV1",
    "assert_no_learned_view_slot_identity_v1",
]
