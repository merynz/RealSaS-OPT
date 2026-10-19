from __future__ import annotations

from models.tessa.v2 import (
    TESSAGlobalHistoryPolicyV2,
    TESSAStructuralizerConfigV2,
    TESSAStructuralizerGlobalV2,
)
from .contracts_v1 import IRISVNextConfigV1
from .structure_v3 import IRISVNextStructuralizationBridgeV3


class IRISVNextGlobalStructuralizationBridgeV3(IRISVNextStructuralizationBridgeV3):
    """Canonical V3 bridge using TESSA V2 causal global topology history."""

    architecture_id = "RealSaS.IRIS.TESSAGlobalStructuralizationBridge.v3"

    def __init__(
        self,
        *,
        iris_config: IRISVNextConfigV1 = IRISVNextConfigV1(),
        tessa_config: TESSAStructuralizerConfigV2 = TESSAStructuralizerConfigV2(),
        history_policy: TESSAGlobalHistoryPolicyV2 = TESSAGlobalHistoryPolicyV2(),
    ) -> None:
        super().__init__(iris_config=iris_config, tessa_config=tessa_config)
        scene_input_dim = iris_config.plane_channels + 5
        surface_input_dim = 3 + 3 + iris_config.local_feature_dim + 6 + 2
        self.structuralizer = TESSAStructuralizerGlobalV2(
            scene_input_dim=scene_input_dim,
            surface_input_dim=surface_input_dim,
            config=tessa_config,
            history_policy=history_policy,
        )
