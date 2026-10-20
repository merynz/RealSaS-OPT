from __future__ import annotations

import torch

from models.tessa.v2 import TESSAObjectiveWeightsV2
from models.tessa.v3 import TESSAStructuralizerConfigV3, TESSAStructuralizerV3
from .contracts_v1 import IRISVNextConfigV1
from .presentation_evidence_v1 import (
    PRESENTATION_FEATURE_NAMES_V1,
    RoleFreePresentationEvidenceV1,
    null_presentation_evidence_v1,
)
from .structure_v3 import IRISVNextStructuralizationBridgeV3


class IRISVNextStructuralizationBridgeV4(IRISVNextStructuralizationBridgeV3):
    """IRIS evidence -> matched-capacity presentation-conditioned TESSA V3.

    V4 reuses V3 scene/surface source contracts verbatim.  The only new
    conditioning family is typed role-free presentation evidence.  Passing
    `presentation=None` constructs the future A-arm null source while keeping
    exactly the same TESSA V3 parameterization and 1024-token memory budget.

    This source path does not authorize training and does not make TESSA product
    authority.  Compiler remains the only component allowed to mint canonical M.
    """

    architecture_id = "RealSaS.IRIS.TESSAStructuralizationBridge.v4"
    output_authority = "LEARNED_MESH_CANDIDATE_PROPOSAL_ONLY"
    named_semantic_taxonomy_required = False
    teacher_topology_in_inference = False

    def __init__(
        self,
        *,
        iris_config: IRISVNextConfigV1 = IRISVNextConfigV1(),
        tessa_config: TESSAStructuralizerConfigV3 = TESSAStructuralizerConfigV3(),
    ) -> None:
        torch.nn.Module.__init__(self)
        iris_config.validate()
        tessa_config.validate()
        self.iris_config = iris_config
        self.tessa_config = tessa_config
        scene_input_dim = iris_config.plane_channels + 5
        surface_input_dim = 3 + 3 + iris_config.local_feature_dim + 6 + 2
        presentation_input_dim = len(PRESENTATION_FEATURE_NAMES_V1)
        self.structuralizer = TESSAStructuralizerV3(
            scene_input_dim=scene_input_dim,
            surface_input_dim=surface_input_dim,
            presentation_input_dim=presentation_input_dim,
            config=tessa_config,
        )

    def presentation_source(
        self,
        *,
        reference: torch.Tensor,
        presentation: RoleFreePresentationEvidenceV1 | None,
    ) -> RoleFreePresentationEvidenceV1:
        if presentation is None:
            return null_presentation_evidence_v1(
                batch_size=reference.shape[0],
                device=reference.device,
                dtype=reference.dtype,
            )
        if presentation.values.shape[0] != reference.shape[0]:
            raise ValueError("IRIS_VNEXT_STRUCTURE_V4_PRESENTATION_BATCH_MISMATCH")
        if presentation.values.device != reference.device:
            raise ValueError("IRIS_VNEXT_STRUCTURE_V4_PRESENTATION_DEVICE_MISMATCH")
        if presentation.values.dtype != reference.dtype:
            raise ValueError("IRIS_VNEXT_STRUCTURE_V4_PRESENTATION_DTYPE_MISMATCH")
        return presentation

    def forward(
        self,
        *,
        scene_planes: torch.Tensor,
        witness_xyz: torch.Tensor,
        witness_normal: torch.Tensor,
        witness_h: torch.Tensor,
        witness_mechanical: dict[str, torch.Tensor],
        input_ids: torch.Tensor,
        presentation: RoleFreePresentationEvidenceV1 | None = None,
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
        typed_presentation = self.presentation_source(
            reference=surface,
            presentation=presentation,
        )
        return self.structuralizer(
            scene_source=scene,
            surface_source=surface,
            presentation_source=typed_presentation.values,
            input_ids=input_ids,
            labels=labels,
            surface_valid_mask=witness_valid_mask,
            presentation_valid_mask=typed_presentation.valid_mask,
            sequence_position_offset=sequence_position_offset,
            objective_weights=objective_weights,
        )
