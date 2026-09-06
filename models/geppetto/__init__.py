"""Geppetto current model and product-conditioning authorities.

The learned skeleton model remains V2. Product conditioning moved to the lossless
V3 seam after the 2026-09-06 causal-repair evidence invalidated the assumption that
the 24D summary alone was a sufficient product-authority boundary.
"""

CURRENT_PACKAGE = "models.geppetto.v2"
CURRENT_MODEL_PACKAGE = "models.geppetto.v2"
CURRENT_PRODUCT_CONDITIONING_PACKAGE = "models.geppetto.v3"
LOSSY_V2_PRODUCT_CONDITIONING_REVOKED = True
AUTHORITY_REVOCATION_ID = "REALSAS_20260906_CAUSAL_REPAIR_233ec3bcd77e0f02"
