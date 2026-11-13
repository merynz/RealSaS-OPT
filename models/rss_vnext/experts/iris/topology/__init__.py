"""IRIS scene-native topology head.

This package removes the canonical topology learner's dependence on the legacy
17-D GSA scene summary.  Compiler/GSA evidence remains independent product
qualification evidence; it is not deleted by this model-side bridge.
"""

from .scene_memory_v1 import (
    IRISTopologySceneMemoryConfigV1,
    IRISTopologySceneMemoryV1,
)
from .head_v1 import (
    IRISSceneTopologyHeadV1,
    TESSADecoderOnlyV1,
)

__all__ = [
    "IRISTopologySceneMemoryConfigV1",
    "IRISTopologySceneMemoryV1",
    "IRISSceneTopologyHeadV1",
    "TESSADecoderOnlyV1",
]
