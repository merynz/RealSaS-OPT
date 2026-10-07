from __future__ import annotations

"""Canonical product-state dispatch with carrier-native mechanical seal."""
from .product_state_legacy_v2 import *  # noqa: F401,F403
from .product_state_legacy_v2 import _require_caa_final_mesh_candidate_binding
from .product_state_v3 import seal_complete_puppet_stage
