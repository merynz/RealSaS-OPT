from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from models.tessa.v2 import (
    TESSAObjectiveWeightsV2,
    TESSAStructuralizerConfigV2,
    TESSAStructuralizerV2,
)
from .truth139_model_v1 import IRISV31Truth139V1


@dataclass(frozen=True)
class Truth139TESSABridgeConfigV1:
    scene_channels: int
    iris_query_width: int
    pooled_scene_resolution: int = 12

    def validate(self) -> None:
        if min(self.scene_channels, self.iris_query_width, self.pooled_scene_resolution) <= 0:
            raise ValueError("TRUTH139_TESSA_BRIDGE_CONFIG_INVALID")

    @property
    def scene_input_dim(self) -> int:
        return self.scene_channels + 3

    @property
    def surface_input_dim(self) -> int:
        return 3 + 3 + self.iris_query_width + 4


def dense_cell_center_xyz_v1(resolution: int, *, device, dtype) -> torch.Tensor:
    """Exact DenseObservationEncoderV31 cell-centre coordinates, scaled to TESSA [-.5,.5]."""
    if int(resolution) <= 0:
        raise ValueError("TRUTH139_TESSA_SCENE_RESOLUTION_INVALID")
    axis = ((torch.arange(resolution, device=device, dtype=dtype) + 0.5) * (2.0 / resolution) - 1.0) * 0.5
    z, y, x = torch.meshgrid(axis, axis, axis, indexing="ij")
    return torch.stack((x, y, z), dim=-1)


class Truth139TESSASourceBuilderV1(nn.Module):
    """Inference-equivalent typed sources for TESSA; contains no trainable parameters."""

    architecture_id = "RealSaS.IRISV31.Truth139.TESSASourceBuilder.v1"
    teacher_topology_used = False
    presentation_stream_fabricated = False

    def __init__(self, config: Truth139TESSABridgeConfigV1) -> None:
        super().__init__()
        config.validate()
        self.config = config

    def scene_source(self, carrier: torch.Tensor) -> torch.Tensor:
        if carrier.ndim != 5 or carrier.shape[1] != self.config.scene_channels:
            raise ValueError("TRUTH139_TESSA_SCENE_CARRIER_SHAPE_INVALID")
        if not torch.is_floating_point(carrier) or not bool(torch.isfinite(carrier).all()):
            raise ValueError("TRUTH139_TESSA_SCENE_CARRIER_NUMERIC_INVALID")
        r = int(self.config.pooled_scene_resolution)
        pooled = F.adaptive_avg_pool3d(carrier, output_size=(r, r, r))
        values = pooled.permute(0, 2, 3, 4, 1).contiguous()
        xyz = dense_cell_center_xyz_v1(r, device=carrier.device, dtype=carrier.dtype)
        xyz = xyz.unsqueeze(0).expand(carrier.shape[0], -1, -1, -1, -1)
        return torch.cat((values, xyz), dim=-1).reshape(
            carrier.shape[0], r**3, self.config.scene_input_dim
        )

    def surface_source(
        self,
        *,
        iris_model: IRISV31Truth139V1,
        carrier: torch.Tensor,
        witness_xyz_iris: torch.Tensor,
    ) -> torch.Tensor:
        if witness_xyz_iris.ndim != 3 or witness_xyz_iris.shape[-1] != 3:
            raise ValueError("TRUTH139_TESSA_WITNESS_XYZ_SHAPE_INVALID")
        if witness_xyz_iris.shape[0] != carrier.shape[0]:
            raise ValueError("TRUTH139_TESSA_WITNESS_BATCH_MISMATCH")
        if not torch.is_floating_point(witness_xyz_iris) or not bool(torch.isfinite(witness_xyz_iris).all()):
            raise ValueError("TRUTH139_TESSA_WITNESS_XYZ_NUMERIC_INVALID")
        if witness_xyz_iris.numel() and float(witness_xyz_iris.detach().abs().max()) > 1.000001:
            raise ValueError("TRUTH139_TESSA_WITNESS_IRIS_DOMAIN_DRIFT")

        xyz_tessa = witness_xyz_iris * 0.5
        state = iris_model.shared_query_state(carrier, witness_xyz_iris.to(carrier.dtype))
        normal = iris_model.query_surface_normal(carrier, witness_xyz_iris.to(carrier.dtype))
        gg = iris_model.gm(state)
        geometry = iris_model.geometry(gg)
        source = torch.cat(
            (
                xyz_tessa.to(state.dtype),
                normal.to(state.dtype),
                state,
                geometry.to(state.dtype),
            ),
            dim=-1,
        )
        if source.shape[-1] != self.config.surface_input_dim:
            raise AssertionError("TRUTH139_TESSA_SURFACE_DIM_DRIFT")
        if not bool(torch.isfinite(source).all()):
            raise ValueError("TRUTH139_TESSA_SURFACE_SOURCE_NONFINITE")
        return source


class Truth139DenseIRISTESSABridgeV1(nn.Module):
    """Dense IRIS evidence -> current TESSA V2 dual-stream topology proposal.

    This bridge deliberately does not invent the V3 presentation stream because the current
    Truth139/Stage48 input court abstains on incompletely observed pair/presentation families.
    Scene evidence comes directly from the dense IRIS carrier.  Surface evidence is generated
    from IRIS-predicted zero-surface witness locations and contains only learned/inference-time
    quantities: normalized position, learned normal, shared query state H, signed field and bounded
    local delta.  Teacher topology is used only as the autoregressive training target.
    """

    architecture_id = "RealSaS.IRISV31.Truth139.TESSABridge.v1"
    output_authority = "LEARNED_MESH_CANDIDATE_PROPOSAL_ONLY"
    compiler_mesh_authority = True
    teacher_topology_in_inference = False
    named_semantic_taxonomy_required = False
    presentation_stream_fabricated = False

    def __init__(
        self,
        *,
        bridge_config: Truth139TESSABridgeConfigV1,
        tessa_config: TESSAStructuralizerConfigV2 = TESSAStructuralizerConfigV2(),
    ) -> None:
        super().__init__()
        bridge_config.validate()
        tessa_config.validate()
        self.bridge_config = bridge_config
        self.tessa_config = tessa_config
        self.source_builder = Truth139TESSASourceBuilderV1(bridge_config)
        self.structuralizer = TESSAStructuralizerV2(
            scene_input_dim=bridge_config.scene_input_dim,
            surface_input_dim=bridge_config.surface_input_dim,
            config=tessa_config,
        )

    def scene_source(self, carrier: torch.Tensor) -> torch.Tensor:
        return self.source_builder.scene_source(carrier)

    def surface_source(
        self,
        *,
        iris_model: IRISV31Truth139V1,
        carrier: torch.Tensor,
        witness_xyz_iris: torch.Tensor,
    ) -> torch.Tensor:
        return self.source_builder.surface_source(
            iris_model=iris_model,
            carrier=carrier,
            witness_xyz_iris=witness_xyz_iris,
        )

    def forward(
        self,
        *,
        iris_model: IRISV31Truth139V1,
        carrier: torch.Tensor,
        witness_xyz_iris: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        witness_valid_mask: torch.Tensor | None = None,
        sequence_position_offset: int | torch.Tensor = 0,
        objective_weights: TESSAObjectiveWeightsV2 = TESSAObjectiveWeightsV2(),
    ):
        with torch.no_grad():
            scene = self.scene_source(carrier).detach()
            surface = self.surface_source(
                iris_model=iris_model,
                carrier=carrier,
                witness_xyz_iris=witness_xyz_iris,
            ).detach()
        return self.structuralizer(
            scene_source=scene,
            surface_source=surface,
            input_ids=input_ids,
            labels=labels,
            surface_valid_mask=witness_valid_mask,
            sequence_position_offset=sequence_position_offset,
            objective_weights=objective_weights,
        )
