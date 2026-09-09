"""FIT1-frozen Geppetto reference-strength mainline package.

The frozen base candidate is byte-preserved. Its historical tensorization import
name is rebound in ``sys.modules`` to this package's byte-identical promoted
module before the candidate is imported, so the current model implementation
does not execute a dated ``experiments.*`` tensorization implementation.
"""

from __future__ import annotations

import sys

from . import rigging_surface_tensorization_v1 as _promoted_tensorization

# Byte-preservation compatibility for the frozen base candidate's historical
# import string. The object bound here is the promoted models/ implementation.
_HISTORICAL_TENSORIZATION_MODULE = (
    "experiments.geppetto_reference_strength_fullstack_v1."
    "rigging_surface_tensorization_v1"
)
sys.modules[_HISTORICAL_TENSORIZATION_MODULE] = _promoted_tensorization

from .geppetto_reference_strength_candidate_v1 import (  # noqa: E402
    GeppettoReferenceStrengthCandidateV1,
    GeppettoReferenceStrengthConfigV1,
    GeppettoReferenceStrengthRawOutputV1,
)
from .geppetto_reference_strength_no_learned_slot_v1 import (  # noqa: E402
    ARCHITECTURE_ID,
    VIEW_CONTRACT,
    GeppettoReferenceStrengthNoLearnedSlotV1,
    assert_no_learned_view_slot_identity_v1,
    deterministic_canonical_view_code_v1,
)
from .rigging_surface_tensorization_v1 import RiggingSurfaceTensorV1

__all__ = [
    "ARCHITECTURE_ID",
    "VIEW_CONTRACT",
    "GeppettoReferenceStrengthCandidateV1",
    "GeppettoReferenceStrengthConfigV1",
    "GeppettoReferenceStrengthRawOutputV1",
    "GeppettoReferenceStrengthNoLearnedSlotV1",
    "RiggingSurfaceTensorV1",
    "assert_no_learned_view_slot_identity_v1",
    "deterministic_canonical_view_code_v1",
]
