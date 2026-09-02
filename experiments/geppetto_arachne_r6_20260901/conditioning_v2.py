from __future__ import annotations

# Canonical shared conditioning entrypoint. Geppetto V2 is implemented now; Arachne
# V2 symbols are added here only after the AR-05 source gate is implemented.
from .geppetto_conditioning_v2 import (
    FEATURE_CONTRACT_V2,
    GeometryNormalizationV2,
    GeppettoConditioningAdapterV2,
    GeppettoConditioningBatchV2,
)

__all__ = [
    "FEATURE_CONTRACT_V2",
    "GeometryNormalizationV2",
    "GeppettoConditioningAdapterV2",
    "GeppettoConditioningBatchV2",
]
