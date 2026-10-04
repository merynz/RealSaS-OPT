"""Canonical MIRA model namespace.

MIRA = Mechanical Influence & Rig Attachment.

This namespace is alias-first: it preserves legacy Arachne checkpoint/tensor
semantics while new code migrates to canonical RealSaS naming.
"""
from models.arachne.v4.arachne_candidate_v4 import (
    ArachneA1V4 as MIRABackboneV4,
    ArachneA1ConfigV4 as MIRABackboneConfigV4,
    ArachneA1RawOutputV4 as MIRABackboneRawOutputV4,
)
from models.arachne.v5.arachne_candidate_v5 import (
    ArachneA1V5 as MIRACandidateV5,
    ArachneA1ConfigV5 as MIRACandidateConfigV5,
    ArachneA1OutputV5 as MIRACandidateOutputV5,
    DirectSimplexDecoderV5 as MIRADirectSimplexDecoderV5,
)

__all__ = [
    "MIRABackboneV4",
    "MIRABackboneConfigV4",
    "MIRABackboneRawOutputV4",
    "MIRACandidateV5",
    "MIRACandidateConfigV5",
    "MIRACandidateOutputV5",
    "MIRADirectSimplexDecoderV5",
]
