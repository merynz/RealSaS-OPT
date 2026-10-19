from __future__ import annotations

from torch import nn

from .config_v2 import TESSAStructuralizerConfigV2
from .global_attention_v2 import (
    HierarchicalCausalSelfAttentionV2,
    TESSAGlobalHistoryPolicyV2,
)
from .model_v2 import TESSADecoderBlockV2, TESSAStructuralizerV2


class TESSAGlobalDecoderBlockV2(TESSADecoderBlockV2):
    def __init__(
        self,
        config: TESSAStructuralizerConfigV2,
        history_policy: TESSAGlobalHistoryPolicyV2,
    ) -> None:
        super().__init__(config)
        self.self_attn = HierarchicalCausalSelfAttentionV2(
            config,
            history=history_policy,
        )


class TESSAStructuralizerGlobalV2(TESSAStructuralizerV2):
    """TESSA V2 with exact local topology context and causal global history."""

    architecture_id = "RealSaS.TESSA.Structuralizer.GlobalHistory.v2"

    def __init__(
        self,
        *,
        scene_input_dim: int,
        surface_input_dim: int,
        config: TESSAStructuralizerConfigV2 = TESSAStructuralizerConfigV2(),
        history_policy: TESSAGlobalHistoryPolicyV2 = TESSAGlobalHistoryPolicyV2(),
    ) -> None:
        history_policy.validate()
        super().__init__(
            scene_input_dim=scene_input_dim,
            surface_input_dim=surface_input_dim,
            config=config,
        )
        self.history_policy = history_policy
        self.decoder = nn.ModuleList(
            [
                TESSAGlobalDecoderBlockV2(config, history_policy)
                for _ in range(config.decoder_layers)
            ]
        )
