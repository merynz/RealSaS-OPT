from __future__ import annotations

"""Geppetto joint-locus distribution interface.

D0 wraps the existing multimodal Gaussian locus formulation without owning or
renaming parameters. This preserves historical Geppetto checkpoint state-dict
keys exactly while creating a clean seam for D1 diffusion challengers.

This module is research-only until its parity contract passes.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class JointLocusDistributionV1:
    representative_position: torch.Tensor
    representative_log_sigma: torch.Tensor
    representative_mode_index: torch.Tensor
    modes_normalized: torch.Tensor
    mode_log_sigma: torch.Tensor
    mode_logits: torch.Tensor
    distribution_kind: str


class MultimodalGaussianLocusHeadV1:
    """Non-Module adapter over the existing registered linear heads.

    Deliberately not a torch.nn.Module: parameter ownership remains with the
    historical GeppettoCandidateV2 attributes (position, log_sigma,
    position_mode_logits), preserving checkpoint keys byte-for-byte.
    """

    distribution_kind = "MULTIMODAL_GAUSSIAN_V1"

    def __init__(self, *, position, log_sigma, mode_logits, position_modes: int, position_scale: float):
        self.position = position
        self.log_sigma = log_sigma
        self.mode_logits = mode_logits
        self.position_modes = int(position_modes)
        self.position_scale = float(position_scale)
        if self.position_modes < 2:
            raise ValueError("D0 locus head requires at least two modes")
        if self.position_scale <= 0:
            raise ValueError("D0 locus head position scale invalid")

    @staticmethod
    def map_representative(
        modes: torch.Tensor,
        mode_log_sigma: torch.Tensor,
        mode_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            modes.ndim != 3
            or mode_log_sigma.shape != modes.shape
            or mode_logits.shape != modes.shape[:2]
        ):
            raise ValueError("multimodal representative contract drift")
        B, M, C = modes.shape
        if C != 3 or M < 2:
            raise ValueError("expected at least two 3D locus modes")
        idx = torch.argsort(
            mode_logits, dim=-1, descending=True, stable=True
        )[:, 0]
        gather3 = idx[:, None, None].expand(B, 1, 3)
        pos = torch.gather(modes, 1, gather3).squeeze(1)
        log_sigma = torch.gather(mode_log_sigma, 1, gather3).squeeze(1)
        return pos, log_sigma, idx

    def __call__(self, h: torch.Tensor) -> JointLocusDistributionV1:
        B = h.shape[0]
        M = self.position_modes
        modes = (
            torch.tanh(self.position(h).reshape(B, M, 3))
            * self.position_scale
        )
        # Preserve the historical uncertainty-isolation contract exactly.
        mode_ls = self.log_sigma(h.detach()).reshape(B, M, 3).clamp(-8.0, 4.0)
        logits = self.mode_logits(h)
        pos, rep_ls, idx = self.map_representative(modes, mode_ls, logits)
        return JointLocusDistributionV1(
            representative_position=pos,
            representative_log_sigma=rep_ls,
            representative_mode_index=idx,
            modes_normalized=modes,
            mode_log_sigma=mode_ls,
            mode_logits=logits,
            distribution_kind=self.distribution_kind,
        )


__all__ = [
    "JointLocusDistributionV1",
    "MultimodalGaussianLocusHeadV1",
]
