from __future__ import annotations

import torch
from torch import nn

from models.tessa.v2.model_v2 import (
    TESSAStructuralizerOutputV2,
    TESSAStructuralizerV2,
    _TypedMemoryStreamV2,
)
from models.tessa.v2.objective_v2 import TESSAObjectiveWeightsV2
from .config_v3 import TESSAStructuralizerConfigV3


class TESSATriStreamMemoryV3(nn.Module):
    """Matched scene + surface/mechanics + presentation structural memory."""

    architecture_id = "RealSaS.TESSA.TriStreamStructuralMemory.v3"

    def __init__(
        self,
        *,
        scene_input_dim: int,
        surface_input_dim: int,
        presentation_input_dim: int,
        config: TESSAStructuralizerConfigV3 = TESSAStructuralizerConfigV3(),
    ) -> None:
        super().__init__()
        config.validate()
        if min(scene_input_dim, surface_input_dim, presentation_input_dim) <= 0:
            raise ValueError("TESSA_V3_MEMORY_INPUT_DIM_INVALID")
        self.config = config
        self.scene = _TypedMemoryStreamV2(
            input_dim=int(scene_input_dim),
            token_count=config.scene_memory_tokens,
            type_index=0,
            config=config,
        )
        self.surface = _TypedMemoryStreamV2(
            input_dim=int(surface_input_dim),
            token_count=config.surface_memory_tokens,
            type_index=1,
            config=config,
        )
        self.presentation = _TypedMemoryStreamV2(
            input_dim=int(presentation_input_dim),
            token_count=config.presentation_memory_tokens,
            type_index=2,
            config=config,
        )
        self.fusion_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        *,
        scene_source: torch.Tensor,
        surface_source: torch.Tensor,
        presentation_source: torch.Tensor,
        scene_valid_mask: torch.Tensor | None = None,
        surface_valid_mask: torch.Tensor | None = None,
        presentation_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scene = self.scene(scene_source, scene_valid_mask)
        surface = self.surface(surface_source, surface_valid_mask)
        presentation = self.presentation(presentation_source, presentation_valid_mask)
        if not (scene.shape[0] == surface.shape[0] == presentation.shape[0]):
            raise ValueError("TESSA_V3_MEMORY_BATCH_MISMATCH")
        memory = torch.cat((scene, surface, presentation), dim=1)
        if memory.shape[1] != self.config.total_conditioning_tokens:
            raise AssertionError("TESSA_V3_MEMORY_TOKEN_COUNT_DRIFT")
        return self.fusion_norm(memory)


class TESSAStructuralizerV3(TESSAStructuralizerV2):
    """Presentation-conditioned topology proposal model.

    V3 deliberately keeps the V2 autoregressive decoder and vocabulary while
    replacing only the conditioning memory.  This keeps the first structural-
    evidence A/B focused on whether role-free presentation evidence helps.

    This module emits topology proposal logits only.  Compiler remains the only
    product authority allowed to mint canonical M.
    """

    architecture_id = "RealSaS.TESSA.Structuralizer.v3"
    output_authority = "LEARNED_MESH_CANDIDATE_PROPOSAL_ONLY"
    named_semantic_taxonomy_required = False
    teacher_topology_in_inference = False

    def __init__(
        self,
        *,
        scene_input_dim: int,
        surface_input_dim: int,
        presentation_input_dim: int,
        config: TESSAStructuralizerConfigV3 = TESSAStructuralizerConfigV3(),
    ) -> None:
        config.validate()
        # Build the proven V2 decoder/objective stack, then replace only memory.
        super().__init__(
            scene_input_dim=scene_input_dim,
            surface_input_dim=surface_input_dim,
            config=config,
        )
        self.config = config
        self.memory = TESSATriStreamMemoryV3(
            scene_input_dim=scene_input_dim,
            surface_input_dim=surface_input_dim,
            presentation_input_dim=presentation_input_dim,
            config=config,
        )

    def encode_memory(
        self,
        *,
        scene_source: torch.Tensor,
        surface_source: torch.Tensor,
        presentation_source: torch.Tensor,
        scene_valid_mask: torch.Tensor | None = None,
        surface_valid_mask: torch.Tensor | None = None,
        presentation_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.memory(
            scene_source=scene_source,
            surface_source=surface_source,
            presentation_source=presentation_source,
            scene_valid_mask=scene_valid_mask,
            surface_valid_mask=surface_valid_mask,
            presentation_valid_mask=presentation_valid_mask,
        )

    def forward(
        self,
        *,
        scene_source: torch.Tensor,
        surface_source: torch.Tensor,
        presentation_source: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        scene_valid_mask: torch.Tensor | None = None,
        surface_valid_mask: torch.Tensor | None = None,
        presentation_valid_mask: torch.Tensor | None = None,
        sequence_position_offset: int | torch.Tensor = 0,
        objective_weights: TESSAObjectiveWeightsV2 = TESSAObjectiveWeightsV2(),
    ) -> TESSAStructuralizerOutputV2:
        memory = self.encode_memory(
            scene_source=scene_source,
            surface_source=surface_source,
            presentation_source=presentation_source,
            scene_valid_mask=scene_valid_mask,
            surface_valid_mask=surface_valid_mask,
            presentation_valid_mask=presentation_valid_mask,
        )
        return self.decode_tokens(
            memory=memory,
            input_ids=input_ids,
            labels=labels,
            sequence_position_offset=sequence_position_offset,
            objective_weights=objective_weights,
        )
