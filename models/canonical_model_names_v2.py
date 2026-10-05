from __future__ import annotations

"""Canonical public identities for the learned RealSaS model stack.

This registry changes naming/role authority only. It does not rewrite legacy
checkpoint keys, tensor semantics, schema identifiers, or artifact hashes.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class LearnedModelIdentityV2:
    name: str
    expansion: str
    scientific_object: str
    legacy_names: tuple[str, ...]
    canonical_authority: str = "LEARNED_PROPOSAL_OR_EVIDENCE_ONLY"


IRIS = LearnedModelIdentityV2(
    name="IRIS",
    expansion="Image-based Relational Inference for Structure",
    scientific_object="OBSERVATION_GROUNDED_RELATIONAL_STRUCTURE_EVIDENCE",
    legacy_names=(),
)

TESSA = LearnedModelIdentityV2(
    name="TESSA",
    expansion="Topological Evidence for Surface Structure Approximation",
    scientific_object="MECHANICAL_SURFACE_TOPOLOGY_HYPOTHESIS",
    legacy_names=(),
)

AXIS = LearnedModelIdentityV2(
    name="AXIS",
    expansion="Articulation eXtraction through Inferred Structure",
    scientific_object="ARTICULATION_SKELETON_HYPOTHESIS",
    legacy_names=("ATLAS", "Geppetto"),
)

MIRA = LearnedModelIdentityV2(
    name="MIRA",
    expansion="Mesh-Informed Rigging Affinity",
    scientific_object="CONTINUOUS_RIG_AFFINITY_FIELD",
    legacy_names=("Arachne",),
)

CANONICAL_MODEL_STACK = (IRIS, TESSA, AXIS, MIRA)
CANONICAL_MODEL_NAMES = tuple(model.name for model in CANONICAL_MODEL_STACK)

__all__ = [
    "LearnedModelIdentityV2",
    "IRIS",
    "TESSA",
    "AXIS",
    "MIRA",
    "CANONICAL_MODEL_STACK",
    "CANONICAL_MODEL_NAMES",
]
