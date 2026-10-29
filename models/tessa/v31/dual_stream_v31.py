"""V31 exact rolling-attention decoder over the V2 dual-stream structural memory.

Truth139 has qualified dense scene and geometry-surface evidence but intentionally abstains from
an incompletely observed presentation stream.  This variant keeps the V2 two-stream memory budget
while adopting V31's chunk-independent rolling causal attention semantics.  No historical V2/V3
checkpoint is compatible unless the attention semantics identity also matches.
"""
from __future__ import annotations

import torch
from torch import nn

from models.tessa.v2.config_v2 import TESSAStructuralizerConfigV2
from models.tessa.v2.model_v2 import TESSADualStreamMemoryV2, TESSAStructuralizerV2
from .attention_v31 import SEMANTICS_ID
from .model_v31 import TESSADecoderBlockV31


class TESSADualStreamStructuralizerV31(TESSAStructuralizerV2):
    architecture_id = "RealSaS.TESSA.DualStreamStructuralizer.v3_1"
    attention_semantics = SEMANTICS_ID
    output_authority = "LEARNED_MESH_CANDIDATE_PROPOSAL_ONLY"
    presentation_stream_present = False

    def __init__(
        self,
        *,
        scene_input_dim: int,
        surface_input_dim: int,
        config: TESSAStructuralizerConfigV2 = TESSAStructuralizerConfigV2(),
        backend: str = "SDPA",
    ) -> None:
        nn.Module.__init__(self)
        config.validate()
        if config.dropout != 0.0:
            raise ValueError("TESSA_DUAL_V31_DROPOUT_MUST_BE_ZERO")
        self.config = config
        self.memory = TESSADualStreamMemoryV2(
            scene_input_dim=scene_input_dim,
            surface_input_dim=surface_input_dim,
            config=config,
        )
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.local_attention_window, config.d_model)
        self.decoder = nn.ModuleList(
            [TESSADecoderBlockV31(config, backend=backend) for _ in range(config.decoder_layers)]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
