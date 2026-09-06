"""Canonical Geppetto product-conditioning entrypoint.

Product callers must import from this module rather than a historical versioned
adapter. The current product seam preserves the complete typed RiggingSurfaceIR
payload and view-indexed raster/support evidence before any learner-specific
compression.
"""

from models.geppetto.v3.conditioning_lossless_v3 import (
    AUTHORITY_REVOCATION_ID_V1,
    AUTHORITY_REVOCATION_REASON_V1,
    GeppettoLosslessConditioningBatchV3,
    GeppettoProductConditioningAdapterV3,
    LOSSLESS_EVIDENCE_CONTRACT_V1,
    PRODUCT_CONDITIONING_AUTHORITY_V3,
)

CURRENT_PRODUCT_CONDITIONING_ADAPTER = GeppettoProductConditioningAdapterV3


def build_product_conditioning(surfaces):
    return CURRENT_PRODUCT_CONDITIONING_ADAPTER()(surfaces)


__all__ = [
    "GeppettoLosslessConditioningBatchV3",
    "GeppettoProductConditioningAdapterV3",
    "CURRENT_PRODUCT_CONDITIONING_ADAPTER",
    "build_product_conditioning",
    "PRODUCT_CONDITIONING_AUTHORITY_V3",
    "LOSSLESS_EVIDENCE_CONTRACT_V1",
    "AUTHORITY_REVOCATION_ID_V1",
    "AUTHORITY_REVOCATION_REASON_V1",
]
