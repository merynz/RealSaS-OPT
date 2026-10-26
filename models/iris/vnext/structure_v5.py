from __future__ import annotations

"""Coordinate-coherent IRIS -> TESSA bridge candidate for apparatus audit.

V5 does not change TESSA authority or topology semantics. It only makes the
coordinate domain explicit at the IRIS/TESSA boundary and exposes the inverse
mapping required by typed endpoint binding.
"""

import torch

from .coordinate_contract_v1 import (
    tessa_xyz_to_iris_v1,
    triplane_cell_center_xy_tessa_v1,
)
from .presentation_evidence_v1 import (
    PRESENTATION_KIND_PAIR,
    RoleFreePresentationEvidenceV1,
)
from .structure_v4 import IRISVNextStructuralizationBridgeV4


class IRISVNextStructuralizationBridgeV5(IRISVNextStructuralizationBridgeV4):
    architecture_id = "RealSaS.IRIS.TESSAStructuralizationBridge.v5"
    coordinate_contract = "IRIS[-1,1]__TESSA[-0.5,0.5]__EXPLICIT_V1"
    endpoint_binding_authority = "DETERMINISTIC_ADDRESSABILITY_ONLY"

    def scene_source(self, scene_planes: torch.Tensor) -> torch.Tensor:
        if (
            scene_planes.ndim != 5
            or scene_planes.shape[1] != 3
            or scene_planes.shape[2] != self.iris_config.plane_channels
        ):
            raise ValueError("IRIS_VNEXT_STRUCTURE_V5_TRIPLANE_SHAPE_INVALID")
        if not torch.is_floating_point(scene_planes) or not bool(
            torch.isfinite(scene_planes).all()
        ):
            raise ValueError("IRIS_VNEXT_STRUCTURE_V5_TRIPLANE_NUMERIC_INVALID")

        b, planes, channels, height, width = scene_planes.shape
        values = scene_planes.permute(0, 1, 3, 4, 2).contiguous()
        xy = triplane_cell_center_xy_tessa_v1(
            height,
            width,
            device=scene_planes.device,
            dtype=scene_planes.dtype,
        )
        plane_id = torch.eye(3, device=scene_planes.device, dtype=scene_planes.dtype)[
            :, None, None, :
        ].expand(3, height, width, 3)
        pos = torch.cat(
            (xy[None].expand(3, -1, -1, -1), plane_id),
            dim=-1,
        ).unsqueeze(0).expand(b, -1, -1, -1, -1)
        return torch.cat((values, pos), dim=-1).reshape(
            b, planes * height * width, channels + 5
        )

    @staticmethod
    def presentation_endpoint_query_xyz_iris_v1(
        presentation: RoleFreePresentationEvidenceV1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ordered pair endpoints in the IRIS Q-domain.

        Returns `(xyz_a_iris, xyz_b_iris, pair_valid)`. Non-pair rows are zeroed
        and invalid so callers cannot accidentally query or train on fabricated
        endpoints.
        """

        values = presentation.values
        xyz_a_tessa = values[..., 0:3]
        xyz_b_tessa = values[..., 3:6]
        xyz_b_valid = values[..., 6] > 0.5
        pair_valid = (
            presentation.valid_mask.bool()
            & (presentation.kind_id == PRESENTATION_KIND_PAIR)
            & xyz_b_valid
        )
        xyz_a_iris = tessa_xyz_to_iris_v1(xyz_a_tessa)
        xyz_b_iris = tessa_xyz_to_iris_v1(xyz_b_tessa)
        zero_a = torch.zeros_like(xyz_a_iris)
        zero_b = torch.zeros_like(xyz_b_iris)
        xyz_a_iris = torch.where(pair_valid[..., None], xyz_a_iris, zero_a)
        xyz_b_iris = torch.where(pair_valid[..., None], xyz_b_iris, zero_b)
        return xyz_a_iris, xyz_b_iris, pair_valid
