from __future__ import annotations

"""Canonical mesh adapter dispatch.

Stages outside carrier-native mechanics retain the pre-promotion V2 implementation.
Stages 34/35/36 consume exact carrier-native W_M. Historical helpers remain
explicitly re-exported for tests/tools that import the stable adapter module.
"""
from .mesh_legacy_v2 import *  # noqa: F401,F403
from .mesh_legacy_v2 import (
    _DEMO_STAGE18_FALLBACK_RULE,
    _artifact_root,
    _axis_contract,
    _component_observations,
    _demo_stage18_fallback_prereg,
    _load_camera_set,
    _load_candidate_and_policy,
    _load_partition_and_carrier,
    _load_skeleton,
    _load_surface,
    _relation_parent_quality_report,
    _write_ir,
    _write_json,
)
from .mesh_carrier_v3 import (
    seal_deformation_capability_envelope,
    qualify_canonical_mesh_stage,
    bind_qualified_mesh_skin_stage,
)
