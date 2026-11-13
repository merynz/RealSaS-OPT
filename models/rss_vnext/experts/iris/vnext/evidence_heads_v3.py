from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .contracts_v1 import IRISVNextConfigV1


class MechanicalDemandHeadV3(nn.Module):
    """Only explicitly supervised mechanical/support evidence.

    Historical V1 emitted a free `uncertainty` scalar without an objective.
    V3 removes it and adds authored-motion `contact_persistence_probability`,
    which is eligible only where an explicit validity mask exists.
    """

    architecture_id = "RealSaS.IRIS.MechanicalDemandHead.v3"
    output_semantics = "TRAINED_MECHANICAL_AND_PERSISTENT_CONTACT_EVIDENCE_ONLY"
    uncalibrated_uncertainty_output = False

    def __init__(self, cfg: IRISVNextConfigV1 = IRISVNextConfigV1()) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.LayerNorm(cfg.local_feature_dim),
            nn.Linear(cfg.local_feature_dim, cfg.local_feature_dim),
            nn.GELU(),
            nn.Linear(cfg.local_feature_dim, cfg.local_feature_dim),
            nn.GELU(),
            # minor, major_delta, tangent2, bend, rigidity, contact = 7
            nn.Linear(cfg.local_feature_dim, 7),
        )

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        if h.shape[-1] != self.cfg.local_feature_dim:
            raise ValueError("IRIS_MECHANICAL_V3_H_DIM_INVALID")
        raw = self.net(h)
        minor = F.softplus(raw[..., 0])
        major = minor + F.softplus(raw[..., 1])
        tangent = F.normalize(raw[..., 2:4], dim=-1, eps=1.0e-6)
        return {
            "stretch_major": major,
            "stretch_minor": minor,
            "tangent_direction_2theta": tangent,
            "bend_demand": F.softplus(raw[..., 4]),
            "rigidity_probability": torch.sigmoid(raw[..., 5]),
            "contact_persistence_probability": torch.sigmoid(raw[..., 6]),
        }


class AppearanceEvidenceHeadV3(nn.Module):
    """View-conditioned appearance outputs with a complete training contract.

    Historical V1 emitted an appearance descriptor and neural confidence with no
    direct objective.  V3 exposes only RGBA plus the two supervised boundary
    families.  A future descriptor/calibration channel must arrive with its own
    explicit objective and court.
    """

    architecture_id = "RealSaS.IRIS.AppearanceEvidenceHead.v3"
    output_semantics = "TRAINED_VIEW_APPEARANCE_AND_BOUNDARY_EVIDENCE_ONLY"
    neural_confidence_output = False
    unsupervised_descriptor_output = False

    def __init__(self, cfg: IRISVNextConfigV1 = IRISVNextConfigV1()) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.camera = nn.Sequential(
            nn.LayerNorm(cfg.camera_feature_dim),
            nn.Linear(cfg.camera_feature_dim, cfg.camera_embed_dim),
            nn.GELU(),
            nn.Linear(cfg.camera_embed_dim, cfg.camera_embed_dim),
        )
        width = cfg.local_feature_dim
        self.body = nn.Sequential(
            nn.LayerNorm(cfg.local_feature_dim + cfg.camera_embed_dim),
            nn.Linear(cfg.local_feature_dim + cfg.camera_embed_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
        )
        self.rgba = nn.Linear(width, 4)
        self.boundary = nn.Linear(width, 2)

    def forward(
        self,
        h: torch.Tensor,
        camera_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if h.ndim != 3 or h.shape[-1] != self.cfg.local_feature_dim:
            raise ValueError("IRIS_APPEARANCE_V3_H_SHAPE_INVALID")
        if camera_features.shape != (h.shape[0], self.cfg.camera_feature_dim):
            raise ValueError("IRIS_APPEARANCE_V3_CAMERA_SHAPE_INVALID")
        c = self.camera(camera_features)[:, None, :].expand(-1, h.shape[1], -1)
        z = self.body(torch.cat((h, c), dim=-1))
        b = torch.sigmoid(self.boundary(z))
        return {
            "rgba_candidate": torch.sigmoid(self.rgba(z)),
            "alpha_boundary_probability": b[..., 0],
            "appearance_boundary_probability": b[..., 1],
        }
