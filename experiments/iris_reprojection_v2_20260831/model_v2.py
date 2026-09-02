from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn

from .q_domain_v2 import RayHypothesisDomainV2
from .q_descriptor_sampler_v2 import NativeResolutionPyramidV2, QDescriptorSamplerV2
from .q_evidence_encoder_v2 import QEvidenceEncoderV2
from .evidence_field_v2 import EvidenceFieldV2, EvidenceFieldOutputV2
from .ray_modes_v2 import RayModesV2, extract_ray_modes_v2
from .local_refinement_v2 import LocalDepthRefinementV2
from .depth_output_head_v2 import DepthOutputV2, DepthSupportUncertaintyHeadV2


@dataclass
class IrisReprojectionOutputV2:
    field: EvidenceFieldOutputV2
    modes: RayModesV2
    refined_depth: torch.Tensor
    depth_output: DepthOutputV2


class IrisReprojectionV2(nn.Module):
    """Reprojection-centered depth evidence learner.

    The only learned geometric output is forward depth/support/uncertainty. q_points
    are analytic candidate coordinates generated outside the network from exact cameras.
    """
    def __init__(self, foundation_dims: tuple[int, ...], hidden_dim: int = 192, max_modes: int = 3):
        super().__init__()
        self.native = NativeResolutionPyramidV2()
        self.sampler = QDescriptorSamplerV2(foundation_dims, self.native)
        self.evidence = QEvidenceEncoderV2(self.sampler.descriptor_dim, hidden_dim=hidden_dim)
        self.field = EvidenceFieldV2(hidden_dim)
        self.refine = LocalDepthRefinementV2(hidden_dim)
        self.output = DepthSupportUncertaintyHeadV2(hidden_dim)
        self.max_modes = int(max_modes)

    def forward(self, images: torch.Tensor, frozen_foundation_maps: tuple[torch.Tensor, ...], domain: RayHypothesisDomainV2) -> IrisReprojectionOutputV2:
        sampled = self.sampler(images, frozen_foundation_maps, domain)
        encoded = self.evidence(sampled, domain)
        field = self.field(encoded, domain)
        modes = extract_ray_modes_v2(field.score_logits, sampled.valid_views.any(dim=-1), max_modes=self.max_modes)
        refined = self.refine(field.hidden, domain.depth_values, modes)
        out = self.output(field.hidden, modes)
        return IrisReprojectionOutputV2(field, modes, refined, out)
