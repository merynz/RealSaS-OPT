from __future__ import annotations

import torch

from models.tessa.v2 import (
    TESSAObjectiveWeightsV2,
    TESSAStructuralizerConfigV2,
    TESSAStructuralizerV2,
)
from .contracts_v1 import IRISVNextConfigV1


def _triplane_position_features_v3(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if height <= 0 or width <= 0:
        raise ValueError("IRIS_VNEXT_STRUCTURE_V3_SCENE_SPATIAL_SHAPE_INVALID")
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    xy = torch.stack((xx, yy), dim=-1)
    plane_id = torch.eye(3, device=device, dtype=dtype)[:, None, None, :].expand(
        3, height, width, 3
    )
    return torch.cat((xy[None].expand(3, -1, -1, -1), plane_id), dim=-1)


class IRISVNextStructuralizationBridgeV3(torch.nn.Module):
    """Clean IRIS evidence -> TESSA V2 structuralization bridge.

    The V8 structure memory consumed two uncalibrated channels:
    `witness_log_uncertainty` and mechanical `uncertainty`.  Neither had a
    training objective in the historical V8 trainer.  V3 intentionally forbids
    them.  Only supervised mechanical families and explicitly validity-typed
    contact persistence may condition the structuralizer.
    """

    architecture_id = "RealSaS.IRIS.TESSAStructuralizationBridge.v3"
    output_authority = "LEARNED_MESH_CANDIDATE_PROPOSAL_ONLY"
    uncalibrated_uncertainty_allowed = False

    def __init__(
        self,
        *,
        iris_config: IRISVNextConfigV1 = IRISVNextConfigV1(),
        tessa_config: TESSAStructuralizerConfigV2 = TESSAStructuralizerConfigV2(),
    ) -> None:
        super().__init__()
        iris_config.validate()
        tessa_config.validate()
        self.iris_config = iris_config
        self.tessa_config = tessa_config

        # scene: plane feature + explicit 2D coordinate + one-hot plane id.
        scene_input_dim = iris_config.plane_channels + 5
        # surface: xyz3 + normal3 + H + trained mechanics6 +
        # contact_persistence(value, validity)2.
        surface_input_dim = 3 + 3 + iris_config.local_feature_dim + 6 + 2
        self.structuralizer = TESSAStructuralizerV2(
            scene_input_dim=scene_input_dim,
            surface_input_dim=surface_input_dim,
            config=tessa_config,
        )

    def scene_source(self, scene_planes: torch.Tensor) -> torch.Tensor:
        if (
            scene_planes.ndim != 5
            or scene_planes.shape[1] != 3
            or scene_planes.shape[2] != self.iris_config.plane_channels
        ):
            raise ValueError("IRIS_VNEXT_STRUCTURE_V3_TRIPLANE_SHAPE_INVALID")
        if not torch.is_floating_point(scene_planes) or not bool(
            torch.isfinite(scene_planes).all()
        ):
            raise ValueError("IRIS_VNEXT_STRUCTURE_V3_TRIPLANE_NUMERIC_INVALID")
        b, p, c, h, w = scene_planes.shape
        values = scene_planes.permute(0, 1, 3, 4, 2).contiguous()
        pos = _triplane_position_features_v3(
            h,
            w,
            device=scene_planes.device,
            dtype=scene_planes.dtype,
        ).unsqueeze(0).expand(b, -1, -1, -1, -1)
        return torch.cat((values, pos), dim=-1).reshape(b, p * h * w, c + 5)

    def surface_source(
        self,
        *,
        witness_xyz: torch.Tensor,
        witness_normal: torch.Tensor,
        witness_h: torch.Tensor,
        witness_mechanical: dict[str, torch.Tensor],
        contact_persistence: torch.Tensor | None = None,
        contact_persistence_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if witness_xyz.ndim != 3 or witness_xyz.shape[-1] != 3:
            raise ValueError("IRIS_VNEXT_STRUCTURE_V3_XYZ_SHAPE_INVALID")
        if witness_normal.shape != witness_xyz.shape:
            raise ValueError("IRIS_VNEXT_STRUCTURE_V3_NORMAL_SHAPE_INVALID")
        if witness_h.shape != witness_xyz.shape[:2] + (
            self.iris_config.local_feature_dim,
        ):
            raise ValueError("IRIS_VNEXT_STRUCTURE_V3_H_SHAPE_INVALID")
        if witness_xyz.numel() and float(witness_xyz.detach().abs().max()) > 0.500001:
            raise ValueError("IRIS_VNEXT_STRUCTURE_V3_XYZ_DOMAIN_DRIFT")

        required = (
            "stretch_major",
            "stretch_minor",
            "tangent_direction_2theta",
            "bend_demand",
            "rigidity_probability",
        )
        missing = [name for name in required if name not in witness_mechanical]
        if missing:
            raise ValueError(
                "IRIS_VNEXT_STRUCTURE_V3_MECHANICAL_MISSING:" + ",".join(missing)
            )
        scalar_shape = witness_xyz.shape[:2]
        for name in (
            "stretch_major",
            "stretch_minor",
            "bend_demand",
            "rigidity_probability",
        ):
            if witness_mechanical[name].shape != scalar_shape:
                raise ValueError(
                    "IRIS_VNEXT_STRUCTURE_V3_MECHANICAL_SHAPE_INVALID:" + name
                )
        tangent = witness_mechanical["tangent_direction_2theta"]
        if tangent.shape != scalar_shape + (2,):
            raise ValueError("IRIS_VNEXT_STRUCTURE_V3_TANGENT_SHAPE_INVALID")

        # Unknown contact is represented as value=0 plus validity=0, never as a
        # fabricated negative.  If one contact tensor is supplied, both are required.
        if (contact_persistence is None) != (contact_persistence_valid is None):
            raise ValueError("IRIS_VNEXT_STRUCTURE_V3_CONTACT_VALUE_VALIDITY_PAIR_REQUIRED")
        if contact_persistence is None:
            contact_value = witness_xyz.new_zeros(scalar_shape)
            contact_valid = witness_xyz.new_zeros(scalar_shape)
        else:
            if (
                contact_persistence.shape != scalar_shape
                or contact_persistence_valid.shape != scalar_shape
            ):
                raise ValueError("IRIS_VNEXT_STRUCTURE_V3_CONTACT_SHAPE_INVALID")
            valid_bool = contact_persistence_valid.bool()
            contact_value = torch.where(
                valid_bool,
                contact_persistence.to(dtype=witness_xyz.dtype),
                torch.zeros_like(contact_persistence, dtype=witness_xyz.dtype),
            )
            contact_valid = valid_bool.to(dtype=witness_xyz.dtype)

        mechanics = torch.cat(
            (
                witness_mechanical["stretch_major"][..., None],
                witness_mechanical["stretch_minor"][..., None],
                tangent,
                witness_mechanical["bend_demand"][..., None],
                witness_mechanical["rigidity_probability"][..., None],
            ),
            dim=-1,
        )
        source = torch.cat(
            (
                witness_xyz,
                witness_normal,
                witness_h,
                mechanics,
                contact_value[..., None],
                contact_valid[..., None],
            ),
            dim=-1,
        )
        if not bool(torch.isfinite(source).all()):
            raise ValueError("IRIS_VNEXT_STRUCTURE_V3_SURFACE_SOURCE_NONFINITE")
        return source

    def forward(
        self,
        *,
        scene_planes: torch.Tensor,
        witness_xyz: torch.Tensor,
        witness_normal: torch.Tensor,
        witness_h: torch.Tensor,
        witness_mechanical: dict[str, torch.Tensor],
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        witness_valid_mask: torch.Tensor | None = None,
        contact_persistence: torch.Tensor | None = None,
        contact_persistence_valid: torch.Tensor | None = None,
        sequence_position_offset: int | torch.Tensor = 0,
        objective_weights: TESSAObjectiveWeightsV2 = TESSAObjectiveWeightsV2(),
    ):
        scene = self.scene_source(scene_planes)
        surface = self.surface_source(
            witness_xyz=witness_xyz,
            witness_normal=witness_normal,
            witness_h=witness_h,
            witness_mechanical=witness_mechanical,
            contact_persistence=contact_persistence,
            contact_persistence_valid=contact_persistence_valid,
        )
        return self.structuralizer(
            scene_source=scene,
            surface_source=surface,
            input_ids=input_ids,
            labels=labels,
            surface_valid_mask=witness_valid_mask,
            sequence_position_offset=sequence_position_offset,
            objective_weights=objective_weights,
        )
