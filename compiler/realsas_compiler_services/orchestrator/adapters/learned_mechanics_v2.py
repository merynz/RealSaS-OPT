from __future__ import annotations

"""Canonical AXIS/MIRA adapter dispatch.

The pre-promotion V2 surface-skin implementation is byte-preserved in
``learned_mechanics_legacy_v2`` for historical replay. Public stage entrypoint
names stay stable while the current mechanics lane dispatches to carrier-native V3.
"""
from .learned_mechanics_legacy_v2 import *  # noqa: F401,F403
from .learned_mechanics_legacy_v2 import (
    _assert_geppetto_surface_scope,
    _execution_json,
    _load_mechanical_carrier_evidence,
    _skeleton_proposal,
    _skin_proposal,
)
from .learned_mechanics_v3 import (
    preregister_geppetto_fit_stage,
    execute_geppetto_fit_stage,
    qualify_skeleton_stage,
    seal_geppetto_checkpoint_stage,
    preregister_arachne_fit_stage,
    execute_arachne_fit_stage,
    qualify_skin_stage,
    seal_arachne_checkpoint_stage,
)
