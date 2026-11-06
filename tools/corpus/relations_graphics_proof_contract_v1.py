#!/usr/bin/env python3
"""Relations/Graphics proof contract V1: source truth != representation policy.

This kernel gives *finite-model* constructive certificates and counterexamples.
It deliberately does NOT infer original artist-owned layers or create IRIS labels.

A source observation is indexed by an exact immutable provenance/scope. In a
flat Boolean uniform-handle grammar, observed opposite values prohibit faces
sharing a handle. For complete measurements the unique COARSEST feasible
partition groups identical operation profiles; other feasible partitions can
still split these groups. Partial matrices induce a conflict graph, not a
spurious same/different-drawable target. A masked-address grammar can perform
all binary edits without any packet split, illustrating non-identifiability.

Mathematical domain here is FINITE and grammar-relative; no universal proof
of arbitrary graphics, cross-pose transport or one-image student inference.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import itertools
import json
from typing import Any


SCHEMA = "RealSaS.RelationsGraphicsEvidenceProofContract.v1"
CONTRACT_POLICY = "SOURCE_OBSERVATION_THEN_CAPABILITY_GRAMMAR_THEN_POLICY"
FALSE_AUTHORITY = {
    "graphics_owner_authorized": False,
    "same_drawable_training_pairs_qualified": 0,
    "different_drawable_training_pairs_qualified": 0,
    "iris_training_authorized": False,
    "tessa_training_authorized": False,
    "training_authorized": False,
    "materialization_authorized": False,
}


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError("RELATIONS_GRAPHICS_PROOF_" + reason)


def digest(value: Any) -> str:
    data = (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class SourceScope:
    """Do not join independent source experiments by face/UV count alone.

    Source SHA equality is necessary but not enough for an operational join.
    The context names have *literal* source policy semantics. A view sampled
    with different stochastic ray offsets is not the same observation.
    """
    source_sha256: str
    locus_domain: str
    locus_address_sha256: str
    pose_scope: str
    camera_scope: str
    sample_scope: str
    operation_scope: str
    observation_policy_sha256: str

    def __post_init__(self) -> None:
        for v in (self.source_sha256, self.locus_address_sha256,
                  self.observation_policy_sha256):
            _require(len(v) == 64 and
                     all(c in "0123456789abcdef" for c in v),
                     "INVALID_SCOPE_PARENT_SHA")
        for v in (self.locus_domain, self.pose_scope, self.camera_scope,
                  self.sample_scope, self.operation_scope):
            _require(bool(v and v.strip()), "EMPTY_SCOPE_AXIS")


def compare_source_scopes(a: SourceScope, b: SourceScope) -> dict[str, Any]:
    """Proof of a joinable observation domain, NOT proof of part ownership."""
    if a.source_sha256 != b.source_sha256:
        return {"status": "DIFFERENT_ORIGINAL_SOURCE__NO_JOIN",
                "same_original_source": False,
                "local_consequence_join_authorized": False}
    differing = [
        k for k in ("locus_domain", "locus_address_sha256", "pose_scope",
                    "camera_scope", "sample_scope", "operation_scope",
                    "observation_policy_sha256")
        if getattr(a, k) != getattr(b, k)
    ]
    return {
        "status": ("EXACT_SOURCE_SCOPE_JOIN_CANDIDATE" if not differing
                   else "SUBJECT_LEVEL_ONLY__NO_LOCAL_CONSEQUENCE_JOIN"),
        "same_original_source": True,
        "mismatched_scope_axes": differing,
        "local_consequence_join_authorized": not differing,
        "artist_owner_join_authorized": False,
    }


@dataclass(frozen=True)
class PartialBooleanTrace:
    """Observed source control responses by locus and declared operation.

    Rows correspond to original loci; columns to named source interventions.
    None means UNOBSERVED, not False. This is already a *court-admitted*
    Boolean trace matrix: the producer cannot upgrade weak instrumentation.
    """
    loci: tuple[str, ...]
    operations: tuple[str, ...]
    values: tuple[tuple[bool | None, ...], ...]
    source_scope: SourceScope

    def __post_init__(self) -> None:
        n, m = len(self.loci), len(self.operations)
        _require(n > 0 and m > 0 and n <= 2_000_000 and m <= 512,
                 "TRACE_MATRIX_DOMAIN")
        _require(len(set(self.loci)) == n and len(set(self.operations)) == m
                 and all(isinstance(x, str) and x for x in self.loci)
                 and all(isinstance(x, str) and x for x in self.operations),
                 "TRACE_IDS_DUPLICATED_OR_EMPTY")
        _require(len(self.values) == n and
                 all(len(row) == m and
                     all(value is None or type(value) is bool for value in row)
                     for row in self.values), "INVALID_OBSERVED_TRACE")
        _require(n * m <= 50_000_000, "SOURCE_TRACE_SIZE_OVERFLOW")

    def observed_disagreements(self, i: int, j: int) -> tuple[str, ...]:
        _require(0 <= i < len(self.loci) and
                 0 <= j < len(self.loci) and i != j,
                 "INVALID_SOURCE_PAIR")
        return tuple(
            op for op, a, b in zip(self.operations,
                                   self.values[i], self.values[j])
            if a is not None and b is not None and a != b
        )

    def conflicting_pairs(self) -> tuple[dict[str, Any], ...]:
        # Bounded for full enumeration; large complete workloads use
        # prove_complete_quotient in O(n * operations) instead.
        _require(len(self.loci) <= 32, "CONFLICT_GRAPH_USE_BOUNDED")
        return tuple(
            {"locus_a": self.loci[i], "locus_b": self.loci[j],
             "distinguishing_operations":
                 list(self.observed_disagreements(i, j))}
            for i, j in itertools.combinations(range(len(self.loci)), 2)
            if self.observed_disagreements(i, j)
        )

    def prove_complete_quotient(self) -> dict[str, Any]:
        """Exact unique coarsest FLAT uniform-handle grouping when complete.

        Every feasible partition refines equality of the COMPLETE binary
        source-task signatures. Conversely the signature blocks are feasible.
        Extra subdividing is allowed and never made source-author truth.
        """
        _require(all(x is not None for row in self.values for x in row),
                 "COMPLETE_QUOTIENT_REQUIRES_ALL_TRACE_BITS")
        distinct: dict[tuple[bool, ...], list[str]] = {}
        for locus, row in zip(self.loci, self.values):
            distinct.setdefault(row, []).append(locus)
        signatures = sorted(distinct)
        certificates = []
        for a, b in itertools.combinations(signatures, 2):
            ti = next(i for i, (v, w) in
                      enumerate(zip(a, b)) if v != w)
            certificates.append({
                "profile_a": list(a),
                "profile_b": list(b),
                "source_operation_witness": self.operations[ti],
                "observed_values": [a[ti], b[ti]],
            })
        groups = [{"trace_signature": list(row), "source_loci": sorted(distinct[row])}
                  for row in signatures]
        return {
            "status": "EXACT_UNIQUE_COARSEST_UNIFORM_WORKLOAD_QUOTIENT",
            "source_operation_count": len(self.operations),
            "original_locus_count": len(self.loci),
            "unique_complete_trace_profile_count": len(groups),
            "coarsest_uniform_groups": groups,
            "pairwise_profile_separation_certificates": certificates,
            "proved_only_for_grammar": "UNIFORM_BOOLEAN_PER_HANDLE",
            "all_feasible_refinements_still_possible": True,
            "requires_artwork_split_in_masked_grammar": False,
            "original_artist_drawable_ownership": "UNKNOWN",
            **FALSE_AUTHORITY,
        }


def _restricted_growth_partitions(n: int):
    """Enumerate all set partitions exactly (Bell(n)), no heuristics."""
    def visit(prefix: tuple[int, ...], max_used: int):
        if len(prefix) == n:
            yield prefix
            return
        for nxt in range(max_used + 2):
            yield from visit(prefix + (nxt,), max(max_used, nxt))
    yield from visit((0,), 0)


def prove_bounded_partition_version_space(trace: PartialBooleanTrace,
                                          *, grammar: str) -> dict[str, Any]:
    """Exact finite version-space of representations in a declared grammar.

    FLAT: a whole packet shares ONE Boolean selection value per operation.
    MASKED: an address mask can vary per original locus inside one packet.
    These are two different *capability contracts*, not two source truths.
    """
    _require(grammar in ("UNIFORM_BOOLEAN_PER_HANDLE",
                         "INDEPENDENT_MASK_PER_LOCUS"),
             "UNKNOWN_REPRESENTATION_GRAMMAR")
    n = len(trace.loci)
    _require(n <= 8, "EXHAUSTIVE_PROOF_LIMIT_EIGHT_LOCI")
    conflicts = {
        (i, j): trace.observed_disagreements(i, j)
        for i, j in itertools.combinations(range(n), 2)
    }
    certificates = [
        {"source_locus_a": trace.loci[i],
         "source_locus_b": trace.loci[j],
         "source_operation_witnesses": list(witness)}
        for (i, j), witness in conflicts.items() if witness
    ]
    viable = []
    feasible_counts = Counter()
    for partition in _restricted_growth_partitions(n):
        if grammar == "UNIFORM_BOOLEAN_PER_HANDLE" and any(
                witness and partition[i] == partition[j]
                for (i, j), witness in conflicts.items()):
            continue
        viable.append(partition)
        feasible_counts[max(partition) + 1] += 1
    _require(bool(viable), "NO_FEASIBLE_REPRESENTATION_IN_DECLARED_GRAMMAR")
    minimum = min(feasible_counts)
    optimal = [p for p in viable if max(p) + 1 == minimum]
    # Intrinsic artist owner cannot be concluded from any partition chosen
    # by the runtime; separation is a capability-specific property only.
    return {
        "schema": SCHEMA,
        "status": "EXACT_FINITE_GRAMMAR_VERSION_SPACE_PROVED",
        "declared_representation_grammar": grammar,
        "source_locus_count": n,
        "source_operation_count": len(trace.operations),
        "observed_opposite_value_witnesses": certificates,
        "feasible_partition_count": len(viable),
        "feasible_by_packet_count": dict(sorted(feasible_counts.items())),
        "minimum_packet_count_in_declared_grammar": minimum,
        "optimal_partition_count": len(optimal),
        "optimal_partition_witnesses": [list(p) for p in optimal],
        "single_packet_witness_exists": any(max(p) == 0 for p in viable),
        "capability_conditioned_must_separate_pairs": (
            certificates if grammar == "UNIFORM_BOOLEAN_PER_HANDLE" else []),
        "same_source_drawable_relation": "UNKNOWN",
        "different_source_drawable_relation": "UNKNOWN",
        "minimum_is_source_intrinsic": False,
        "source_fit_vs_unseen_qualification": "NOT_EVALUATED",
        "proof_method": "EXHAUSTIVE_SET_PARTITION_ENUMERATION",
        **FALSE_AUTHORITY,
    }


def regression_certificate(trace: PartialBooleanTrace) -> dict[str, Any]:
    complete = all(x is not None for row in trace.values for x in row)
    full = (trace.prove_complete_quotient() if complete else
            {"status": "PARTIAL_TRACE__NO_UNIQUE_COARSEST_QUOTIENT_CLAIM"})
    result = {
        "schema": SCHEMA,
        "policy": CONTRACT_POLICY,
        "source_original_sha256": trace.source_scope.source_sha256,
        "declared_ops": list(trace.operations),
        "complete_trace": complete,
        "complete_uniform_quotient": full,
        "artist_part_identifiability": "NOT_IDENTIFIABLE_FROM_EDIT_TRACES_ALONE",
        "witness_construction": (
            "SAME_PER_LOCUS_RGB_ALPHA_DEPTH_FUNCTION_AND_CONTROL_MASKS__"
            "DIFFERENT_PACKET_LABELS__IDENTICAL_ALL_FINITE_TRACES"),
        "compositing_or_alpha_sufficiency": "UNMEASURED_NOT_DERIVED",
        **FALSE_AUTHORITY,
    }
    result["certificate_sha256"] = digest(result)
    return result
