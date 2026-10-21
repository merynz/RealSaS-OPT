from __future__ import annotations

import torch
from torch import nn

from models.tessa.v2 import TESSAGlobalHistoryPolicyV2, TESSAStructuralizerConfigV2
from .contracts_v1 import IRISVNextConfigV1
from .heads_v3 import GraphicsRelationHeadV3Config
from .model_v3 import IRISVNextEvidenceModelV3
from .structure_global_v3 import IRISVNextGlobalStructuralizationBridgeV3


class IRISTESSAStructuralizationStackV3(nn.Module):
    """Optional IRIS -> TESSA composition; Compiler remains external authority.

    This wrapper is convenience, not ownership: the IRIS evidence model and the
    TESSA structuralizer remain separately addressable/checkpointable so the
    direct-surface baseline can use the exact same IRIS state.
    """

    architecture_id = "RealSaS.IRIS_TESSA.StructuralizationStack.v3"
    canonical_mesh_authority = False

    def __init__(
        self,
        *,
        iris_config: IRISVNextConfigV1 = IRISVNextConfigV1(),
        relation_config: GraphicsRelationHeadV3Config = GraphicsRelationHeadV3Config(),
        tessa_config: TESSAStructuralizerConfigV2 = TESSAStructuralizerConfigV2(),
        history_policy: TESSAGlobalHistoryPolicyV2 = TESSAGlobalHistoryPolicyV2(),
    ) -> None:
        super().__init__()
        self.iris = IRISVNextEvidenceModelV3(iris_config, relation_config)
        self.tessa = IRISVNextGlobalStructuralizationBridgeV3(
            iris_config=iris_config,
            tessa_config=tessa_config,
            history_policy=history_policy,
        )

    def encode_scene(self, *args, **kwargs) -> torch.Tensor:
        return self.iris.encode_scene(*args, **kwargs)

    def structuralize(
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
    ):
        return self.tessa(
            scene_planes=scene_planes,
            witness_xyz=witness_xyz,
            witness_normal=witness_normal,
            witness_h=witness_h,
            witness_mechanical=witness_mechanical,
            input_ids=input_ids,
            labels=labels,
            witness_valid_mask=witness_valid_mask,
            contact_persistence=contact_persistence,
            contact_persistence_valid=contact_persistence_valid,
            sequence_position_offset=sequence_position_offset,
        )

    def parameter_groups(self) -> dict[str, tuple[nn.Parameter, ...]]:
        iris_groups = self.iris.parameter_groups()
        tessa = tuple(self.tessa.parameters())
        seen = {id(p) for parameters in iris_groups.values() for p in parameters}
        if any(id(p) in seen for p in tessa):
            raise RuntimeError("IRIS_TESSA_V3_PARAMETER_GROUP_OVERLAP")
        return {**iris_groups, "tessa_structuralizer": tessa}
