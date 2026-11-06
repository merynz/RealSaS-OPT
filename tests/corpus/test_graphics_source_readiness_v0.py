"""Proof boundary tests for generic Graphics Source Readiness v0.

All admission fixtures here are synthetic; no fixture certifies Cat/Triceratops
source. The real Registry must independently re-open physical evidence bytes.
"""
import hashlib

import pytest

from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_core.graphics_source_readiness_v0 import (
    GraphicsSourceClaimV0, GraphicsAdmittedArtifactV0,
    SOURCE_AUTHORITY, audit_graphics_source_readiness_v0,
)
from compiler.realsas_compiler_core.operational_representation_synthesis_v0 import (
    SourceGraphicsEvidenceV0, PuppetCapabilityContractV0,
    synthesize_operational_representations_v0,
)

SRC = "c" * 64
COURT = "e" * 64
SCOPE = "1" * 64
OTHER_SCOPE = "2" * 64
FACTS = (
    "SOURCE_ADDRESS_STABILITY",
    "INDEPENDENT_ORIGINAL_SOURCE_EDIT_CONSEQUENCES",
    "AUTHORED_NATIVE_SOURCE_POSE",
    "SOURCE_UV_EDIT_CONSEQUENCE_ACROSS_POSES",
)


def digest(x):
    return hashlib.sha256(x.encode()).hexdigest()


def claim(fact, *, scope=SCOPE, origin=SOURCE_AUTHORITY,
          observation="OBSERVED", artifact=None, court=COURT):
    return GraphicsSourceClaimV0(
        fact=fact, source_graphics_binding_hash=SRC, scope_hash=scope,
        record_hash=digest(f"record:{fact}:{scope}"),
        evidence_artifact_hash=artifact or digest(f"artifact:{scope}"),
        independent_court_hash=court, origin=origin,
        observation=observation,
    )


def admission(rows, *, scope=SCOPE, court=COURT):
    return GraphicsAdmittedArtifactV0(
        source_graphics_binding_hash=SRC, scope_hash=scope,
        independent_court_hash=court,
        admitted_record_hashes=frozenset(r.record_hash for r in rows),
    )


def audit(rows=(), registry=None, operations=(
    "INDEPENDENT_ATLAS_EDITS", "POSE_AWARE_SOURCE_UV",
)):
    return audit_graphics_source_readiness_v0(
        source_graphics_binding_hash=SRC,
        required_operations=operations,
        claims=tuple(rows), admitted_artifacts=registry,
    )


def test_no_receipts_is_unknown_with_explicit_obligations():
    r = audit()
    assert set(r.per_operation_source_observability.values()) == {"UNKNOWN"}
    assert len(r.missing_obligations["INDEPENDENT_ATLAS_EDITS"]) == 2
    assert len(r.missing_obligations["POSE_AWARE_SOURCE_UV"]) == 2
    assert r.source_parity_with_compiled_puppet == "UNKNOWN"
    assert r.artist_drawable_ownership == "UNKNOWN"
    assert r.training_authorized is False


def test_source_producer_cannot_self_admit_without_independent_registry():
    rows = tuple(map(claim, FACTS))
    r = audit(rows)
    assert r.accepted_claim_count == 0
    assert r.ignored_unqualified_claim_count == len(rows)
    assert set(r.per_operation_source_observability.values()) == {"UNKNOWN"}


def test_same_scope_admission_is_only_source_observability_not_puppet_parity():
    rows = tuple(map(claim, FACTS))
    registry = {rows[0].evidence_artifact_hash: admission(rows)}
    r = audit(rows, registry)
    assert r.accepted_claim_count == len(rows)
    assert set(r.per_operation_source_observability.values()) == {
        "OBSERVED_IN_ADMITTED_SOURCE_SCOPE"
    }
    assert all(v == () for v in r.missing_obligations.values())
    assert r.source_parity_with_compiled_puppet == "UNKNOWN"
    assert r.artist_drawable_ownership == "UNKNOWN"
    assert not r.training_authorized


def test_distinct_scopes_never_spliced_to_fake_complete_source():
    left = claim(FACTS[0], scope=SCOPE)
    right = claim(FACTS[1], scope=OTHER_SCOPE)
    registry = {
        left.evidence_artifact_hash: admission((left,), scope=SCOPE),
        right.evidence_artifact_hash: admission((right,), scope=OTHER_SCOPE),
    }
    r = audit((left, right), registry, ("INDEPENDENT_ATLAS_EDITS",))
    assert r.per_operation_source_observability["INDEPENDENT_ATLAS_EDITS"] == "UNKNOWN"
    assert r.missing_obligations["INDEPENDENT_ATLAS_EDITS"]
    assert r.observed_scope_hashes["INDEPENDENT_ATLAS_EDITS"] == ()


@pytest.mark.parametrize("change", (
    "wrong_court", "wrong_source", "missing_record", "wrong_scope",
))
def test_sha_join_mismatches_never_qualify(change):
    row = claim(FACTS[0])
    kwargs = dict(
        source_graphics_binding_hash=SRC, scope_hash=SCOPE,
        independent_court_hash=COURT,
        admitted_record_hashes=frozenset((row.record_hash,)),
    )
    if change == "wrong_court":
        kwargs["independent_court_hash"] = "f" * 64
    if change == "wrong_source":
        kwargs["source_graphics_binding_hash"] = "f" * 64
    if change == "missing_record":
        kwargs["admitted_record_hashes"] = frozenset()
    if change == "wrong_scope":
        kwargs["scope_hash"] = OTHER_SCOPE
    r = audit((row,), {row.evidence_artifact_hash: GraphicsAdmittedArtifactV0(**kwargs)},
              ("INDEPENDENT_ATLAS_EDITS",))
    assert r.accepted_claim_count == 0
    assert r.per_operation_source_observability["INDEPENDENT_ATLAS_EDITS"] == "UNKNOWN"


@pytest.mark.parametrize("origin", (
    "LEARNED_RELATION_PROPOSAL", "TOPOLOGY_OR_KINEMATIC_PROXY",
    "REPRESENTATION_POLICY_PREFERENCE", "REFERENCE_EXECUTOR_FIXTURE",
))
def test_proxies_never_admit_source_facts_even_with_synthetic_matching_registry(origin):
    row = claim(FACTS[0], origin=origin)
    r = audit((row,), {row.evidence_artifact_hash: admission((row,))},
              ("INDEPENDENT_ATLAS_EDITS",))
    assert r.accepted_claim_count == 0
    assert r.ignored_unqualified_claim_count == 1


def test_admitted_contradicted_source_fails_closed():
    row = claim(FACTS[0], observation="CONTRADICTED")
    with pytest.raises(QualificationError, match="CONTRADICTION"):
        audit((row,), {row.evidence_artifact_hash: admission((row,))},
              ("INDEPENDENT_ATLAS_EDITS",))


def test_source_binding_mismatch_and_duplicate_record_fail_closed():
    row = claim(FACTS[0])
    with pytest.raises(QualificationError, match="DUPLICATE"):
        audit((row, row), {}, ("INDEPENDENT_ATLAS_EDITS",))
    wrong = GraphicsSourceClaimV0(**{
        **row.__dict__, "source_graphics_binding_hash": "f" * 64,
    })
    with pytest.raises(QualificationError, match="BINDING"):
        audit((wrong,), {}, ("INDEPENDENT_ATLAS_EDITS",))


def test_unknown_record_not_promoted_even_if_artifact_is_admitted():
    row = claim(FACTS[0], observation="UNKNOWN")
    r = audit((row,), {row.evidence_artifact_hash: admission((row,))},
              ("INDEPENDENT_ATLAS_EDITS",))
    assert r.accepted_claim_count == 0
    assert r.ignored_unqualified_claim_count == 1


def test_synthesis_consumes_evidence_coverage_without_minting_puppet_capability():
    rows = tuple(map(claim, FACTS[:2]))
    out = synthesize_operational_representations_v0(
        SourceGraphicsEvidenceV0(
            mesh_binding_hash="a" * 64, skin_binding_hash="b" * 64,
            source_graphics_binding_hash=SRC, face_count=4,
            input_mode="TEACHER_SOURCE_KNOWN",
        ),
        PuppetCapabilityContractV0(
            2, ("INDEPENDENT_ATLAS_EDITS", "HIDDEN_EDIT_STATE_PERSISTENCE")
        ),
        source_claims=rows,
        independently_admitted_artifacts={
            rows[0].evidence_artifact_hash: admission(rows)
        },
    )
    assert out.source_evidence_readiness.per_operation_source_observability[
        "INDEPENDENT_ATLAS_EDITS"] == "OBSERVED_IN_ADMITTED_SOURCE_SCOPE"
    assert out.source_evidence_readiness.per_operation_source_observability[
        "HIDDEN_EDIT_STATE_PERSISTENCE"] == "UNKNOWN"
    assert out.source_evidence_readiness.missing_obligations[
        "HIDDEN_EDIT_STATE_PERSISTENCE"] == (
            "SOURCE_HIDDEN_EDIT_STATE_RECOVERY",
            "SOURCE_OCCLUDE_REVEAL_OPERATION",
        )
    assert set(out.capability_states.values()) == {"UNKNOWN"}
    assert len(out.executable_candidate_ids) >= 2
    assert not out.source_qualified_candidate_ids
    assert not any(out.authority.values())


def test_student_and_synthetic_inputs_cannot_consume_teacher_claims():
    rows = tuple(map(claim, FACTS[:2]))
    for mode in ("STUDENT_INFERRED_UNQUALIFIED", "SYNTHETIC_FIXTURE"):
        ev = SourceGraphicsEvidenceV0(
            mesh_binding_hash="a" * 64, skin_binding_hash="b" * 64,
            source_graphics_binding_hash=SRC, face_count=4, input_mode=mode,
        )
        with pytest.raises(QualificationError, match="TEACHER_SOURCE_EVIDENCE_FORBIDDEN"):
            synthesize_operational_representations_v0(
                ev, PuppetCapabilityContractV0(2, ("INDEPENDENT_ATLAS_EDITS",)),
                source_claims=rows,
                independently_admitted_artifacts={
                    rows[0].evidence_artifact_hash: admission(rows)
                },
            )


def test_student_source_face_grouping_from_teacher_is_rejected():
    ev = SourceGraphicsEvidenceV0(
        mesh_binding_hash="a" * 64, skin_binding_hash="b" * 64,
        source_graphics_binding_hash=SRC, face_count=4,
        source_face_group_ids=(0, 0, 1, 1),
        input_mode="STUDENT_INFERRED_UNQUALIFIED",
    )
    with pytest.raises(QualificationError, match="SOURCE_FACE_GROUPS_FORBIDDEN"):
        synthesize_operational_representations_v0(
            ev, PuppetCapabilityContractV0(2, ("INDEPENDENT_ATLAS_EDITS",))
        )
