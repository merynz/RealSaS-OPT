from __future__ import annotations

import torch
from torch import nn

from models.tessa.v1.model_v1 import TESSAConfigV1, TESSAOutputV1

from .dense_scene_memory_v1 import (
    IRISDenseTopologySceneMemoryConfigV1,
    IRISDenseTopologySceneMemoryV1,
)
from .head_v1 import TESSADecoderOnlyV1


class IRISDenseSceneTopologyHeadV1(nn.Module):
    """TESSA decoder over the same dense IRIS carrier used by the Truth139 geometry line.

    The head remains proposal-only.  It does not mint canonical mesh authority and does not
    consume the legacy GSA 17-D conditioning path.
    """

    architecture_id = "RealSaS.IRIS.DenseSceneTopologyHead.v1"
    input_authority = "IRIS_DENSE_SCENE_EVIDENCE_ONLY"
    output_authority = "LEARNED_TOPOLOGY_PROPOSAL_ONLY"
    legacy_gsa_17d_used = False

    def __init__(
        self,
        *,
        scene_channels: int,
        tessa_config: TESSAConfigV1 | None = None,
        memory_config: IRISDenseTopologySceneMemoryConfigV1 | None = None,
    ) -> None:
        super().__init__()
        tessa = tessa_config or TESSAConfigV1()
        tessa.scaling_policy()
        memory = memory_config or IRISDenseTopologySceneMemoryConfigV1(
            scene_channels=int(scene_channels),
            topology_width=tessa.d_model,
            memory_tokens=tessa.surface_latent_count,
            attention_heads=tessa.n_heads,
        )
        memory.validate()
        if memory.scene_channels != int(scene_channels):
            raise ValueError("IRIS_DENSE_TESSA_SCENE_CHANNEL_MISMATCH")
        if memory.topology_width != tessa.d_model:
            raise ValueError("IRIS_DENSE_TESSA_TOPOLOGY_WIDTH_MISMATCH")
        if memory.memory_tokens != tessa.surface_latent_count:
            raise ValueError("IRIS_DENSE_TESSA_MEMORY_TOKEN_POLICY_MISMATCH")
        if memory.attention_heads != tessa.n_heads:
            raise ValueError("IRIS_DENSE_TESSA_ATTENTION_HEAD_POLICY_MISMATCH")

        self.tessa_config = tessa
        self.memory_config = memory
        self.scene_memory = IRISDenseTopologySceneMemoryV1(memory)
        self.topology_decoder = TESSADecoderOnlyV1(tessa)

    def encode_scene_memory(self, carrier: torch.Tensor) -> torch.Tensor:
        return self.scene_memory(carrier)

    def forward(
        self,
        *,
        carrier: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        sequence_position_offset: int = 0,
    ) -> TESSAOutputV1:
        memory = self.encode_scene_memory(carrier)
        return self.topology_decoder(
            memory=memory,
            input_ids=input_ids,
            labels=labels,
            sequence_position_offset=sequence_position_offset,
        )
