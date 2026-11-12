from __future__ import annotations

"""Epistemic relation court for one RSS.vNext product mesh candidate.

This module deliberately does NOT infer relations from part names, triangle
adjacency or logits. StdGEN-style geometry/semantic proposals remain raw
hypotheses. Only independently qualified source/deformation/renderer witnesses
can subsequently admit executable abilities. The ledger preserves opposing
claims and never votes, averages depths, or silently converts UNKNOWN to false.

Face order is view-specific. Contact is pose-specific. Geometry facts about
the candidate are not automatically source truth. No caller can mint a
ProductRevision with this court's always-false product authority flags.
"""

from dataclasses import asdict, dataclass, replace
import math
from typing import Iterable

from .hashing import content_sha256
from .types import QualificationError

SCHEMA = "RealSaS.RSSvNext.RelationEpistemicCourt.v1"
_MAX_CLAIMS = 16_384
_MAX_FACE_INDEX = 16_000_000
_TYPES = frozenset((
    "SAME_DRAWABLE", "CUT_BOUNDARY", "DIRECTED_ATTACHMENT",
    "FRONT_ORDER", "MECHANICAL_CONTACT", "STRUCTURAL_CONTINUITY",
))
_UNDIRECTED = frozenset((
    "SAME_DRAWABLE", "CUT_BOUNDARY", "MECHANICAL_CONTACT",
    "STRUCTURAL_CONTINUITY",
))
_OBSERVED_ORIGINS = frozenset((
    "SOURCE_NATIVE_UNVERIFIED", "SOURCE_RENDERED_UNVERIFIED",
))
_ORIGINS = _OBSERVED_ORIGINS | frozenset((
    "LEARNED_HYPOTHESIS", "DERIVED_CANDIDATE_GEOMETRY",
    "DERIVED_CANDIDATE_INTERVENTION", "REPRESENTATION_POLICY",
))
_TRUTHS = frozenset(("POSITIVE", "NEGATIVE", "UNKNOWN"))


@dataclass(frozen=True)
class RSSvNextTypedRelationAssertionV1:
    relation_type: str
    from_face_index: int
    to_face_index: int
    inference_evidence_hash: str
    candidate_mesh_sha256: str | None
    witness_ref: str
    origin: str
    truth: str = "UNKNOWN"
    predicted_probability: float | None = None
    from_unknown_probability: float | None = None
    to_unknown_probability: float | None = None
    # "observed" means an original input camera, NOT a generated auxiliary view.
    observed_view_index: int | None = None
    pose_binding_sha256: str | None = None


@dataclass(frozen=True)
class RSSvNextEpistemicRelationDecisionV1:
    relation_type: str
    from_face_index: int
    to_face_index: int
    observed_view_index: int | None
    pose_binding_sha256: str | None
    state: str
    cause: str
    assertions: tuple[RSSvNextTypedRelationAssertionV1, ...]
    executable_authority: bool = False
    source_truth_authority: bool = False


@dataclass(frozen=True)
class RSSvNextRelationEpistemicCourtV1:
    inference_evidence_hash: str
    candidate_mesh_sha256: str | None
    state: str
    decisions: tuple[RSSvNextEpistemicRelationDecisionV1, ...]
    unrepresentable_assertions: tuple[RSSvNextTypedRelationAssertionV1, ...]
    schema: str = SCHEMA
    attachment_authority: bool = False
    source_drawable_authority: bool = False
    source_visibility_authority: bool = False
    source_contact_authority: bool = False
    executable_collision_authority: bool = False


def _fail(code: str) -> None:
    raise QualificationError("RSS_VNEXT_RELATION_EPISTEMIC_" + code)


def _sha(value: object) -> bool:
    return (type(value) is str and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _validate_assertion(
    value: RSSvNextTypedRelationAssertionV1, *,
    inference_evidence_hash: str,
    candidate_mesh_sha256: str | None,
) -> bool:
    if type(value) is not RSSvNextTypedRelationAssertionV1:
        _fail("TYPED_ASSERTION_REQUIRED")
    if (value.inference_evidence_hash != inference_evidence_hash
        or value.candidate_mesh_sha256 != candidate_mesh_sha256
        or not _sha(value.witness_ref)):
        _fail("SOURCE_AND_WITNESS_LINEAGE_DRIFT")
    if (value.origin not in _ORIGINS or value.truth not in _TRUTHS
        or type(value.origin) is not str or type(value.truth) is not str):
        _fail("EVIDENCE_ORIGIN_OR_TRUTH_INVALID")
    if (
        type(value.from_face_index) is not int
        or type(value.to_face_index) is not int
        or not 0 <= value.from_face_index < _MAX_FACE_INDEX
        or not 0 <= value.to_face_index < _MAX_FACE_INDEX
        or value.from_face_index == value.to_face_index
    ):
        _fail("FACE_ADDRESS_INVALID")
    if (value.observed_view_index is not None
        and (type(value.observed_view_index) is not int
             or value.observed_view_index not in (0, 1))):
        _fail("OBSERVED_VIEW_SCOPE_INVALID")
    if value.pose_binding_sha256 is not None and not _sha(value.pose_binding_sha256):
        _fail("POSE_LINEAGE_INVALID")
    for v in (value.predicted_probability, value.from_unknown_probability,
              value.to_unknown_probability):
        if v is not None and (type(v) not in (float, int)
                              or not math.isfinite(v) or not 0 <= v <= 1):
            _fail("PREDICTION_OR_UNKNOWN_MASS_INVALID")
    if (value.predicted_probability is not None
        and (value.origin != "LEARNED_HYPOTHESIS" or value.truth != "UNKNOWN")):
        _fail("PREDICTION_CANNOT_ASSERT_SOURCE_TRUTH")
    if (value.origin == "LEARNED_HYPOTHESIS"
        and (value.from_unknown_probability is None
             or value.to_unknown_probability is None)):
        _fail("LEARNED_UNKNOWN_MASS_REQUIRED")
    # Non-representable is not a fabricated negative and not an exception:
    # it remains a source-addressed, hashed claim in the output ledger.
    if value.relation_type not in _TYPES:
        return False
    if value.candidate_mesh_sha256 is None:
        return False
    if value.relation_type == "FRONT_ORDER" and value.observed_view_index is None:
        return False
    if value.relation_type == "MECHANICAL_CONTACT" and value.pose_binding_sha256 is None:
        return False
    return True


def adjudicate_rss_vnext_relations_v1(
    assertions: Iterable[RSSvNextTypedRelationAssertionV1], *,
    inference_evidence_hash: str, candidate_mesh_sha256: str | None,
) -> RSSvNextRelationEpistemicCourtV1:
    """Deterministic, non-promoting and bounded reconciliation of explicit claims.

    Every result retains its exact original assertions. Opposite POLARITIES on
    the SAME predicate, face-address pair, view and pose are contradictions.
    DIFFERENT predicates (e.g. same-drawable vs cut probability) must never
    be treated as logical negations merely by score arithmetic.
    """
    if not _sha(inference_evidence_hash):
        _fail("INFERENCE_PARENT_REQUIRED")
    if candidate_mesh_sha256 is not None and not _sha(candidate_mesh_sha256):
        _fail("MESH_PARENT_INVALID")
    if not isinstance(assertions, (list, tuple)):
        _fail("BOUNDED_ASSERTION_SEQUENCE_REQUIRED")
    if len(assertions) > _MAX_CLAIMS:
        _fail("ASSERTION_RESOURCE_LIMIT")
    by_key: dict[tuple[object, ...], list[RSSvNextTypedRelationAssertionV1]] = {}
    unrepresentable = []
    for claim in assertions:
        if not _validate_assertion(
            claim, inference_evidence_hash=inference_evidence_hash,
            candidate_mesh_sha256=candidate_mesh_sha256,
        ):
            unrepresentable.append(claim)
            continue
        a, b = claim.from_face_index, claim.to_face_index
        if claim.relation_type in _UNDIRECTED:
            a, b = sorted((a, b))
        key = (
            claim.relation_type, a, b,
            claim.observed_view_index, claim.pose_binding_sha256,
        )
        by_key.setdefault(key, []).append(claim)
    sort_claim = lambda row: content_sha256(asdict(row))
    decisions = []
    for (kind, a, b, view, pose), raw in sorted(
        by_key.items(), key=lambda item: str(item[0]),
    ):
        ordered = tuple(sorted(raw, key=sort_claim))
        positive = any(x.truth == "POSITIVE" for x in ordered)
        negative = any(x.truth == "NEGATIVE" for x in ordered)
        scored = any(x.predicted_probability is not None for x in ordered)
        unverified_source = any(x.origin in _OBSERVED_ORIGINS for x in ordered)
        if positive and negative:
            state = "CONFLICT_UNVERIFIED_SOURCE_CLAIMS" if unverified_source else "CONFLICT_UNQUALIFIED_HYPOTHESES"
            cause = "OPPOSITE_POLARITY_SAME_PREDICATE_AND_SCOPE"
        elif unverified_source:
            state = "ABSTAIN_SOURCE_WITNESS_REQUIRES_INDEPENDENT_ADMISSION"
            cause = "SOURCE_ORIGIN_IS_NOT_SOURCE_QUALIFICATION"
        elif scored:
            state = "PREDICTED_RELATION_UNADMITTED"
            cause = "NEURAL_SCORE_IS_NOT_BOOLEAN_OR_EXECUTABLE_WITNESS"
        elif positive or negative:
            state = "UNQUALIFIED_ASSERTION"
            cause = "POLARITY_WITHOUT_INDEPENDENT_SOURCE_OR_INTERVENTION_COURT"
        else:
            state = "ABSTAIN_NO_POLARITY"
            cause = "UNKNOWN_IS_NOT_A_NEGATIVE"
        decisions.append(RSSvNextEpistemicRelationDecisionV1(
            relation_type=kind, from_face_index=a, to_face_index=b,
            observed_view_index=view, pose_binding_sha256=pose,
            state=state, cause=cause, assertions=ordered,
        ))
    unknown = tuple(sorted(unrepresentable, key=sort_claim))
    if unknown:
        state = "HAS_UNREPRESENTABLE_EVIDENCE_RETAINED"
    elif any(x.state.startswith("CONFLICT") for x in decisions):
        state = "CONFLICTS_RETAINED_NO_SILENT_ARBITRATION"
    elif decisions:
        state = "TYPED_HYPOTHESES_UNQUALIFIED"
    else:
        state = "ABSTAIN_NO_RELATION_EVIDENCE"
    return RSSvNextRelationEpistemicCourtV1(
        inference_evidence_hash=inference_evidence_hash,
        candidate_mesh_sha256=candidate_mesh_sha256,
        state=state, decisions=tuple(decisions),
        unrepresentable_assertions=unknown,
    )


def learned_face_relation_assertions_v1(
    relation: object, *,
    inference_evidence_hash: str,
    candidate_mesh_sha256: str,
) -> tuple[RSSvNextTypedRelationAssertionV1, ...]:
    """Adapter for already computed RSS R×R conditional pair marginals.

    SAME, CUT and directed ATTACHMENT are three INDEPENDENT channels. A high
    probability is still UNKNOWN polarity. Both UNKNOWN masses are retained.
    The parent inference hash, not a fake independent source witness, owns it.
    """
    from .rss_vnext_face_relation_schedule_v1 import RSSvNextSparseFaceRelationScheduleV1
    if (type(relation) is not RSSvNextSparseFaceRelationScheduleV1
        or relation.inference_evidence_hash != inference_evidence_hash
        or relation.candidate_mesh_sha256 != candidate_mesh_sha256
        or relation.attachment_authority or relation.independent_drawable_authority):
        _fail("SPARSE_RELATION_PARENT_OR_AUTHORITY_DRIFT")
    result = []
    for row in relation.face_relations:
        for kind, score in (
            ("SAME_DRAWABLE", row.same_drawable_affinity),
            ("CUT_BOUNDARY", row.cut_affinity),
            ("DIRECTED_ATTACHMENT", row.directed_attachment_affinity),
        ):
            result.append(RSSvNextTypedRelationAssertionV1(
                relation_type=kind,
                from_face_index=row.from_face_index,
                to_face_index=row.to_face_index,
                inference_evidence_hash=inference_evidence_hash,
                candidate_mesh_sha256=candidate_mesh_sha256,
                witness_ref=inference_evidence_hash,
                origin="LEARNED_HYPOTHESIS",
                truth="UNKNOWN",
                predicted_probability=score,
                from_unknown_probability=row.from_unknown_probability,
                to_unknown_probability=row.to_unknown_probability,
            ))
    return tuple(result)


__all__ = [
    "SCHEMA", "RSSvNextTypedRelationAssertionV1",
    "RSSvNextEpistemicRelationDecisionV1", "RSSvNextRelationEpistemicCourtV1",
    "adjudicate_rss_vnext_relations_v1", "learned_face_relation_assertions_v1",
]
