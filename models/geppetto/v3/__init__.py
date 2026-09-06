"""Geppetto V3 product conditioning authority.

The learned skeleton model remains V2 until a lossless-evidence consumer is proven.
This package owns only the product conditioning seam.
"""

from .conditioning_lossless_v3 import (
    AUTHORITY_REVOCATION_ID_V1,
    AUTHORITY_REVOCATION_REASON_V1,
    GeppettoLosslessConditioningBatchV3,
    GeppettoProductConditioningAdapterV3,
    LEGACY_SUMMARY_AUTHORITY_V2,
    LOSSLESS_EVIDENCE_CONTRACT_V1,
    PRODUCT_CONDITIONING_AUTHORITY_V3,
)

__all__ = [
    "GeppettoLosslessConditioningBatchV3",
    "GeppettoProductConditioningAdapterV3",
    "PRODUCT_CONDITIONING_AUTHORITY_V3",
    "LOSSLESS_EVIDENCE_CONTRACT_V1",
    "LEGACY_SUMMARY_AUTHORITY_V2",
    "AUTHORITY_REVOCATION_ID_V1",
    "AUTHORITY_REVOCATION_REASON_V1",
]
