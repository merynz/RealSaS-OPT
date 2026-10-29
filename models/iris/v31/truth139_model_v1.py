from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .evidence_adapter_model_v2 import IRISEvidenceAdapterConfigV2, IRISV31EvidenceAdapterV2


class IRISV31Truth139V1(IRISV31EvidenceAdapterV2):
    """Truth139 model: unchanged dense carrier plus an additive geometry-normal readout.

    The carrier, shared query MLP, signed-field head, mechanics/appearance private adapters and
    relation path are inherited unchanged.  The only geometry-capacity addition is a world-space
    surface-normal evidence head over the established geometry-private state.  Normals are
    teacher-only targets; they never enter the forward observation packet.
    """

    architecture_id = "RealSaS.IRIS.DenseObservationEvidence.v3_1.truth139_v1"

    def __init__(self, config: IRISEvidenceAdapterConfigV2):
        super().__init__(config)
        self.surface_normal = nn.Linear(config.query_width, 3)

    def query_surface_normal(self, carrier: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        state = self.gm(self.shared_query_state(carrier, xyz))
        raw = self.surface_normal(state)
        return F.normalize(raw, dim=-1, eps=1.0e-8)

    def foundation_named_parameters(self):
        yield from super().foundation_named_parameters()
        for name, parameter in self.surface_normal.named_parameters():
            yield "surface_normal." + name, parameter

    def continuous(self, carrier, xyz, cameras):
        value = super().continuous(carrier, xyz, cameras)
        value["surface_normal"] = self.query_surface_normal(carrier, xyz)
        return value
