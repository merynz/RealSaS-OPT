"""STDGEN-inspired protected evidence adapters over the IRIS V3.1 shared scene.

The geometry foundation and the evidence families share the same observation
carrier and continuous query state, but they do not share their final private
adapter path. This prevents a newly introduced mechanics objective from
rewriting the geometry-private state merely because both are local fields.

New evidence adapters are zero-initialized residuals: at construction they are
exact identities over the shared query state, analogous to safe LoRA/adaptor
initialization around a pretrained foundation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .model_v31 import IRISV31


@dataclass(frozen=True)
class IRISEvidenceAdapterConfigV2:
    carrier_resolution: int
    carrier_channels: int
    mechanical_channels: tuple[str, ...]
    pair_channels: tuple[str, ...]
    query_width: int
    query_layers: int
    fourier_bands: int
    private_bottleneck: int
    evidence_adapter_bottleneck: int
    lift_chunk: int
    delta_bound: float

    def validate(self) -> None:
        numeric = (
            self.carrier_resolution,
            self.carrier_channels,
            self.query_width,
            self.query_layers,
            self.fourier_bands,
            self.private_bottleneck,
            self.evidence_adapter_bottleneck,
            self.lift_chunk,
        )
        if min(numeric) <= 0:
            raise ValueError("IRIS_EVIDENCE_ADAPTER_V2_DIMENSION_INVALID")
        if self.query_layers != 3 or self.fourier_bands != 4:
            raise ValueError("IRIS_EVIDENCE_ADAPTER_V2_STRUCTURAL_DRIFT")
        if not np.isfinite(self.delta_bound) or self.delta_bound <= 0:
            raise ValueError("IRIS_EVIDENCE_ADAPTER_V2_DELTA_BOUND_INVALID")
        if not self.mechanical_channels or not self.pair_channels:
            raise ValueError("IRIS_EVIDENCE_ADAPTER_V2_CHANNELS_REQUIRED")


class ZeroInitPrivatePathV2(nn.Module):
    """Residual family adapter whose initial function is exactly identity."""

    def __init__(self, width: int, bottleneck: int):
        super().__init__()
        self.down = nn.Linear(width, bottleneck)
        self.up = nn.Linear(bottleneck, width)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return state + self.up(F.silu(self.down(state)))


class IRISV31EvidenceAdapterV2(IRISV31):
    """V3.1 shared scene with protected geometry/mechanics/appearance readouts."""

    architecture_id = "RealSaS.IRIS.DenseObservationEvidence.v3_1.evidence_adapter_v2"
    output_authority = "CONTINUOUS_EVIDENCE_ONLY"

    def __init__(self, config: IRISEvidenceAdapterConfigV2):
        # IRISV31 only relies on the common config fields and validate().
        super().__init__(config)
        width = config.query_width
        bottleneck = config.evidence_adapter_bottleneck
        # Geometry keeps the established gm path from the capacity pilot.
        # Mechanics and appearance get independent, identity-at-init adapters.
        self.mm = ZeroInitPrivatePathV2(width, bottleneck)
        self.gp = ZeroInitPrivatePathV2(width, bottleneck)

    def shared_query_state(self, carrier: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        if xyz.ndim != 3 or xyz.shape[-1] != 3 or xyz.shape[0] != carrier.shape[0]:
            raise ValueError("IRIS_EVIDENCE_ADAPTER_V2_XYZ_SHAPE")
        sampled = F.grid_sample(
            carrier,
            xyz[:, None, None],
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).squeeze(2).squeeze(2).transpose(1, 2)
        bands = xyz.new_tensor(
            [math.pi * 2**i for i in range(self.config.fourier_bands)]
        )
        angle = xyz[..., None] * bands
        fourier = torch.cat(
            (xyz, angle.sin().flatten(-2), angle.cos().flatten(-2)), dim=-1
        )
        return self.query(torch.cat((sampled, fourier), dim=-1))

    def query_states(self, carrier: torch.Tensor, xyz: torch.Tensor):
        state = self.shared_query_state(carrier, xyz)
        return self.gm(state), self.gp(state)

    def query_mechanics_state(self, carrier: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        return self.mm(self.shared_query_state(carrier, xyz))

    def continuous(self, carrier, xyz, cameras):
        state = self.shared_query_state(carrier, xyz)
        gg = self.gm(state)
        gm = self.mm(state)
        gp = self.gp(state)
        geometry = self.geometry(gg)
        appearance = self.appearance(torch.cat((gp, cameras), -1))
        return {
            "signed_field": geometry[..., 0],
            "bounded_delta": geometry[..., 1:].tanh() * self.config.delta_bound,
            "mechanics": self.mechanics(gm),
            "rgba": appearance[..., :4].sigmoid(),
            "boundary_logit": appearance[..., 4],
            "gp_state": gp,
        }

    def foundation_named_parameters(self):
        for prefix in ("fusion", "encoder", "query", "gm", "geometry"):
            module = getattr(self, prefix)
            for name, parameter in module.named_parameters():
                yield f"{prefix}.{name}", parameter

    def evidence_named_parameters(self):
        for prefix in ("mm", "mechanics", "gp", "appearance"):
            module = getattr(self, prefix)
            for name, parameter in module.named_parameters():
                yield f"{prefix}.{name}", parameter

    def frozen_unused_named_parameters(self):
        for name, parameter in self.relation.named_parameters():
            yield f"relation.{name}", parameter
