from __future__ import annotations

"""Typed relation admission is not probability averaging or source truth."""

from dataclasses import asdict, replace

import pytest

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.rss_vnext_relation_epistemic_court_v1 import (
    RSSvNextTypedRelationAssertionV1,
    adjudicate_rss_vnext_relations_v1,
)
from compiler.realsas_compiler_core.types import QualificationError

EVIDENCE = "a" * 64
MESH = "b" * 64


def _claim(**updates):
    c = RSSvNextTypedRelationAssertionV1(
        relation_type="SAME_DRAWABLE",
        from_face_index=3, to_face_index=5,
        inference_evidence_hash=EVIDENCE,
        candidate_mesh_sha256=MESH,
        witness_ref=EVIDENCE,
        origin="LEARNED_HYPOTHESIS",
        truth="UNKNOWN",
        predicted_probability=.94,
        from_unknown_probability=.1,
        to_unknown_probability=.3,
    )
    return replace(c, **updates)


def _court(*claims, mesh=MESH):
    return adjudicate_rss_vnext_relations_v1(
        list(claims), inference_evidence_hash=EVIDENCE,
        candidate_mesh_sha256=mesh,
    )


def test_independent_same_cut_attachment_scores_are_not_negated_or_voted():
    same = _claim(predicted_probability=.97)
    cut = _claim(relation_type="CUT_BOUNDARY", predicted_probability=.99)
    attachment = _claim(
        relation_type="DIRECTED_ATTACHMENT", predicted_probability=.88,
    )
    court = _court(same, cut, attachment)
    assert court.state == "TYPED_HYPOTHESES_UNQUALIFIED"
    assert len(court.decisions) == 3
    assert all(d.state == "PREDICTED_RELATION_UNADMITTED" for d in court.decisions)
    asserted_scores = {
        d.relation_type: d.assertions[0].predicted_probability
        for d in court.decisions
    }
    assert asserted_scores == {
        "SAME_DRAWABLE": .97,
        "CUT_BOUNDARY": .99,
        "DIRECTED_ATTACHMENT": .88,
    }
    assert all(not d.executable_authority and not d.source_truth_authority
               for d in court.decisions)


def test_opposite_same_predicate_is_conflict_not_average_and_order_is_deterministic():
    positive = _claim(truth="POSITIVE", predicted_probability=None)
    negative = _claim(
        truth="NEGATIVE", predicted_probability=None,
        from_face_index=5, to_face_index=3, witness_ref="c" * 64,
    )
    first = _court(positive, negative)
    reversed_input = _court(negative, positive)
    assert first == reversed_input
    assert first.state == "CONFLICTS_RETAINED_NO_SILENT_ARBITRATION"
    assert first.decisions[0].state == "CONFLICT_UNQUALIFIED_HYPOTHESES"
    assert {x.truth for x in first.decisions[0].assertions} == {"POSITIVE", "NEGATIVE"}
    assert {x.witness_ref for x in first.decisions[0].assertions} == {EVIDENCE, "c"*64}
    assert not first.source_drawable_authority
    assert content_sha256(asdict(first)) == content_sha256(asdict(reversed_input))


def test_claimed_source_vs_neural_conflict_is_not_evidence_promotion():
    predicted = _claim(truth="POSITIVE", predicted_probability=None)
    source_claim = _claim(
        origin="SOURCE_RENDERED_UNVERIFIED", truth="NEGATIVE",
        predicted_probability=None,
        from_unknown_probability=None, to_unknown_probability=None,
        witness_ref="d"*64,
    )
    court = _court(predicted, source_claim)
    assert court.decisions[0].state == "CONFLICT_UNVERIFIED_SOURCE_CLAIMS"
    assert not court.source_drawable_authority
    assert not court.decisions[0].source_truth_authority
    only_source = _court(source_claim)
    assert only_source.decisions[0].state == "ABSTAIN_SOURCE_WITNESS_REQUIRES_INDEPENDENT_ADMISSION"
    assert not only_source.decisions[0].executable_authority


def test_different_view_front_orders_do_not_invent_global_cycle():
    p = _claim(
        relation_type="FRONT_ORDER", truth="POSITIVE",
        predicted_probability=None, observed_view_index=0,
    )
    q = _claim(
        relation_type="FRONT_ORDER", truth="POSITIVE",
        predicted_probability=None, observed_view_index=1,
        from_face_index=5, to_face_index=3,
    )
    result = _court(p, q)
    assert len(result.decisions) == 2
    assert result.state == "TYPED_HYPOTHESES_UNQUALIFIED"
    assert all(not x.source_truth_authority for x in result.decisions)
    missing_view = _court(replace(p, observed_view_index=None))
    assert missing_view.state == "HAS_UNREPRESENTABLE_EVIDENCE_RETAINED"
    assert missing_view.unrepresentable_assertions == (replace(p, observed_view_index=None),)
    assert missing_view.decisions == ()


def test_unknown_is_not_negative_and_unrepresentable_evidence_is_never_dropped():
    unknown = _claim(predicted_probability=None)
    odd = _claim(relation_type="UNMAPPED_HAIR_CONTACT_STATE")
    result = _court(unknown, odd)
    assert result.state == "HAS_UNREPRESENTABLE_EVIDENCE_RETAINED"
    assert result.decisions[0].state == "ABSTAIN_NO_POLARITY"
    assert result.unrepresentable_assertions == (odd,)
    assert result.unrepresentable_assertions[0].witness_ref == EVIDENCE
    no_pose = _claim(
        relation_type="MECHANICAL_CONTACT", predicted_probability=None,
    )
    assert _court(no_pose).unrepresentable_assertions == (no_pose,)


@pytest.mark.parametrize("modify,code", [
    ({"inference_evidence_hash": "f" * 64}, "SOURCE_AND_WITNESS_LINEAGE_DRIFT"),
    ({"witness_ref": "not-a-digest"}, "SOURCE_AND_WITNESS_LINEAGE_DRIFT"),
    ({"predicted_probability": float("nan")}, "PREDICTION_OR_UNKNOWN_MASS_INVALID"),
    ({"to_unknown_probability": None}, "LEARNED_UNKNOWN_MASS_REQUIRED"),
    ({"truth": "NEGATIVE", "predicted_probability": .8}, "PREDICTION_CANNOT_ASSERT_SOURCE_TRUTH"),
    ({"observed_view_index": 2}, "OBSERVED_VIEW_SCOPE_INVALID"),
])
def test_bad_lineage_scope_and_unqualified_score_claims_are_rejected(modify, code):
    with pytest.raises(QualificationError, match=code):
        _court(_claim(**modify))


def test_abstain_after_topology_failure_and_finite_resource_envelope():
    assert _court(mesh=None).state == "ABSTAIN_NO_RELATION_EVIDENCE"
    v = _court(_claim(), _claim(relation_type="CUT_BOUNDARY"))
    assert all(not x.executable_authority for x in v.decisions)
    with pytest.raises(QualificationError, match="ASSERTION_RESOURCE_LIMIT"):
        _court(*([_claim()] * 16_385))
