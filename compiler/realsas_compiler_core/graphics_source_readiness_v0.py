"""Maturity gate between source Graphics evidence and puppet synthesis.

Cannot authenticate original source bytes or mint qualification. Caller MUST
construct admitted_artifacts through an independent verified Registry/court.
Self-described PASS is not evidence.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Mapping
import re
from .types import QualificationError

_SHA = re.compile(r"^[0-9a-f]{64}$")
OPERATION_SOURCE_FACTS_V0: dict[str, tuple[str, ...]] = {
    "INDEPENDENT_ATLAS_EDITS": ("SOURCE_ADDRESS_STABILITY", "INDEPENDENT_ORIGINAL_SOURCE_EDIT_CONSEQUENCES"),
    "POSE_AWARE_SOURCE_UV": ("AUTHORED_NATIVE_SOURCE_POSE", "SOURCE_UV_EDIT_CONSEQUENCE_ACROSS_POSES"),
    "LOCAL_DEPTH_SOURCE_OVER": ("SOURCE_LOCAL_DEPTH_GEOMETRY", "SOURCE_RGBA_LOCAL_DEPTH_CONSEQUENCE"),
    "HIDDEN_EDIT_STATE_PERSISTENCE": ("SOURCE_OCCLUDE_REVEAL_OPERATION", "SOURCE_HIDDEN_EDIT_STATE_RECOVERY"),
    "INDEPENDENT_ATTACHMENT_RELEASE": ("SOURCE_ATTACHMENT_RELEASE_CONSEQUENCE",),
    "SECONDARY_MOTION": ("SOURCE_SECONDARY_MOTION_CONSEQUENCE",),
    "NONBINARY_ALPHA_SOURCE_PARITY": ("SOURCE_NONBINARY_ALPHA_COMPOSITING_CONSEQUENCE",),
}
SUPPORTED_FACTS = frozenset(x for v in OPERATION_SOURCE_FACTS_V0.values() for x in v)
SOURCE_AUTHORITY = "ORIGINAL_SOURCE_PHYSICAL_OBSERVATION"
NON_SOURCE_KINDS = frozenset((
    "LEARNED_RELATION_PROPOSAL", "TOPOLOGY_OR_KINEMATIC_PROXY",
    "REPRESENTATION_POLICY_PREFERENCE", "REFERENCE_EXECUTOR_FIXTURE",
))


def _require(condition, code):
    if not condition:
        raise QualificationError(code)


def _digest(value):
    _require(type(value) is str and _SHA.fullmatch(value) is not None,
             "GRAPHICS_READINESS_SHA256_REQUIRED")


@dataclass(frozen=True)
class GraphicsSourceClaimV0:
    fact: str
    source_graphics_binding_hash: str
    scope_hash: str
    record_hash: str
    evidence_artifact_hash: str
    independent_court_hash: str
    origin: str
    observation: str
    schema_version: str = "RealSaS.GraphicsSourceClaim.v0"

    def validate(self):
        _require(self.fact in SUPPORTED_FACTS, "GRAPHICS_READINESS_FACT_INVALID")
        for value in (self.source_graphics_binding_hash, self.scope_hash,
                      self.record_hash, self.evidence_artifact_hash,
                      self.independent_court_hash):
            _digest(value)
        _require(self.origin in {SOURCE_AUTHORITY, *NON_SOURCE_KINDS},
                 "GRAPHICS_READINESS_ORIGIN_INVALID")
        _require(self.observation in {"OBSERVED", "UNKNOWN", "CONTRADICTED"},
                 "GRAPHICS_READINESS_OBSERVATION_INVALID")


@dataclass(frozen=True)
class GraphicsAdmittedArtifactV0:
    """Caller-supplied via independent verifier of sealed original bytes only.

    Constructing this Python object by hand does not qualify a source; the
    authoritative registry adapter still needs implementing and auditing.
    """
    source_graphics_binding_hash: str
    scope_hash: str
    independent_court_hash: str
    admitted_record_hashes: frozenset[str]


@dataclass(frozen=True)
class GraphicsSourceReadinessV0:
    source_graphics_binding_hash: str
    per_operation_source_observability: dict[str, str]
    missing_obligations: dict[str, tuple[str, ...]]
    observed_scope_hashes: dict[str, tuple[str, ...]]
    accepted_claim_count: int
    ignored_unqualified_claim_count: int
    source_parity_with_compiled_puppet: str = "UNKNOWN"
    artist_drawable_ownership: str = "UNKNOWN"
    training_authorized: bool = False
    schema_version: str = "RealSaS.GraphicsSourceReadiness.v0"

    def to_dict(self):
        return asdict(self)


def audit_graphics_source_readiness_v0(
    *, source_graphics_binding_hash: str, required_operations: tuple[str, ...],
    claims: tuple[GraphicsSourceClaimV0, ...] = (),
    admitted_artifacts: Mapping[str, GraphicsAdmittedArtifactV0] | None = None,
) -> GraphicsSourceReadinessV0:
    """Admit only matching source/court/record; never join unmatched scopes."""
    _digest(source_graphics_binding_hash)
    _require(bool(required_operations)
             and len(set(required_operations)) == len(required_operations)
             and set(required_operations) <= set(OPERATION_SOURCE_FACTS_V0),
             "GRAPHICS_READINESS_OPERATIONS_INVALID")
    admitted_artifacts = admitted_artifacts or {}
    actual = {}
    ignored = 0
    seen = set()
    for claim in claims:
        claim.validate()
        _require(claim.source_graphics_binding_hash == source_graphics_binding_hash,
                 "GRAPHICS_READINESS_SOURCE_BINDING_DRIFT")
        key = (claim.evidence_artifact_hash, claim.record_hash)
        _require(key not in seen, "GRAPHICS_READINESS_DUPLICATE_RECORD")
        seen.add(key)
        a = admitted_artifacts.get(claim.evidence_artifact_hash)
        is_admitted = (
            claim.origin == SOURCE_AUTHORITY
            and isinstance(a, GraphicsAdmittedArtifactV0)
            and a.source_graphics_binding_hash == source_graphics_binding_hash
            and a.scope_hash == claim.scope_hash
            and a.independent_court_hash == claim.independent_court_hash
            and claim.record_hash in a.admitted_record_hashes
        )
        if not is_admitted or claim.observation == "UNKNOWN":
            ignored += 1
            continue
        _require(claim.observation == "OBSERVED",
                 "GRAPHICS_READINESS_ADMITTED_SOURCE_CONTRADICTION")
        actual.setdefault(claim.scope_hash, set()).add(claim.fact)
    states, missing, scopes = {}, {}, {}
    for op in required_operations:
        obligations = set(OPERATION_SOURCE_FACTS_V0[op])
        complete = tuple(sorted(scope for scope, facts in actual.items()
                                if obligations <= facts))
        scopes[op] = complete
        states[op] = "OBSERVED_IN_ADMITTED_SOURCE_SCOPE" if complete else "UNKNOWN"
        if complete:
            missing[op] = ()
        else:
            candidates = (obligations - found for found in actual.values())
            least_missing = min(candidates, key=lambda x: (len(x), sorted(x)),
                                default=obligations)
            missing[op] = tuple(sorted(least_missing))
    return GraphicsSourceReadinessV0(
        source_graphics_binding_hash=source_graphics_binding_hash,
        per_operation_source_observability=states,
        missing_obligations=missing,
        observed_scope_hashes=scopes,
        accepted_claim_count=sum(len(facts) for facts in actual.values()),
        ignored_unqualified_claim_count=ignored,
    )
