"""Finite mathematical truth proofs, including adversarial unknowns and models."""
import itertools
from dataclasses import replace
import pytest

from tools.corpus.relations_graphics_proof_contract_v1 import (
    SourceScope, PartialBooleanTrace, compare_source_scopes,
    prove_bounded_partition_version_space, regression_certificate,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def scope(**kwargs):
    s = SourceScope(
        source_sha256=SHA_A,
        locus_domain="ORIGINAL_SOURCE_FACE_INDEX",
        locus_address_sha256=SHA_B,
        pose_scope="SOURCE_AUTHORED_POSE0",
        camera_scope="ORIGINAL_SOURCE_VIEW_V1_V3",
        sample_scope="CYCLES_ORIGINAL_OFFSETS_0_4",
        operation_scope="NATIVE_SELECTION",
        observation_policy_sha256=SHA_C,
    )
    return replace(s, **kwargs)


def mk(values, operations=("ctrl0", "ctrl1")):
    return PartialBooleanTrace(
        loci=tuple(f"face{i}" for i in range(len(values))),
        operations=operations,
        values=tuple(tuple(row) for row in values),
        source_scope=scope(),
    )


def test_complete_workload_uniform_quotient_has_exact_witness_and_no_artist_labels():
    tr = mk(((True, False), (True, False), (False, False)))
    proof = tr.prove_complete_quotient()
    assert proof["unique_complete_trace_profile_count"] == 2
    assert sorted(len(x["source_loci"]) for x in proof["coarsest_uniform_groups"]) == [1, 2]
    assert len(proof["pairwise_profile_separation_certificates"]) == 1
    assert proof["pairwise_profile_separation_certificates"][0]["source_operation_witness"] == "ctrl0"
    assert proof["original_artist_drawable_ownership"] == "UNKNOWN"
    assert not proof["iris_training_authorized"]


def test_exact_version_space_exhibits_different_runtime_grammars_same_source():
    tr = mk(((True, False), (True, False), (False, False)))
    uniform = prove_bounded_partition_version_space(
        tr, grammar="UNIFORM_BOOLEAN_PER_HANDLE")
    masked = prove_bounded_partition_version_space(
        tr, grammar="INDEPENDENT_MASK_PER_LOCUS")
    assert uniform["minimum_packet_count_in_declared_grammar"] == 2
    assert masked["minimum_packet_count_in_declared_grammar"] == 1
    assert not uniform["single_packet_witness_exists"]
    assert masked["single_packet_witness_exists"]
    assert uniform["optimal_partition_count"] >= 1
    assert masked["optimal_partition_count"] == 1
    assert uniform["same_source_drawable_relation"] == "UNKNOWN"
    assert masked["different_source_drawable_relation"] == "UNKNOWN"
    assert uniform["capability_conditioned_must_separate_pairs"]
    assert masked["capability_conditioned_must_separate_pairs"] == []


def test_complete_equal_controls_never_prove_original_artist_must_link():
    tr = mk(((True, True), (True, True)))
    x = prove_bounded_partition_version_space(
        tr, grammar="UNIFORM_BOOLEAN_PER_HANDLE")
    assert x["feasible_partition_count"] == 2  # merged or separate
    assert x["minimum_packet_count_in_declared_grammar"] == 1
    assert x["same_source_drawable_relation"] == "UNKNOWN"
    assert x["different_source_drawable_relation"] == "UNKNOWN"


def test_missing_is_never_negative_or_unconditional_same():
    tr = mk(((True, None), (True, False), (False, False)))
    with pytest.raises(ValueError, match="REQUIRES_ALL_TRACE_BITS"):
        tr.prove_complete_quotient()
    q = regression_certificate(tr)
    assert q["complete_uniform_quotient"]["status"] == (
        "PARTIAL_TRACE__NO_UNIQUE_COARSEST_QUOTIENT_CLAIM")
    p = prove_bounded_partition_version_space(
        tr, grammar="UNIFORM_BOOLEAN_PER_HANDLE")
    assert p["minimum_packet_count_in_declared_grammar"] == 2
    assert p["same_source_drawable_relation"] == "UNKNOWN"


def test_partial_nontransitive_compatibility_does_not_mint_equivalence():
    # A and B are compatible; B and C are compatible; A and C conflict.
    tr = mk(((True, None), (None, False), (False, False)))
    p = prove_bounded_partition_version_space(
        tr, grammar="UNIFORM_BOOLEAN_PER_HANDLE")
    assert len(p["observed_opposite_value_witnesses"]) == 1
    assert p["minimum_packet_count_in_declared_grammar"] == 2
    assert p["optimal_partition_count"] == 2
    assert not any(k.startswith("same_drawable_training") and p[k] for k in p)


def test_adding_qualified_bit_only_strengthens_opposite_value_certificate():
    incomplete = mk(((True, None), (True, False), (False, False)))
    completed = mk(((True, True), (True, False), (False, False)))
    old = {tuple(sorted((r["locus_a"], r["locus_b"])))
           for r in incomplete.conflicting_pairs()}
    new = {tuple(sorted((r["locus_a"], r["locus_b"])))
           for r in completed.conflicting_pairs()}
    assert old <= new
    assert len(new) == len(old) + 1


def test_different_source_or_sampler_policy_blocks_operational_join():
    a = scope()
    b = scope(source_sha256=SHA_B)
    assert compare_source_scopes(a, b)["status"] == (
        "DIFFERENT_ORIGINAL_SOURCE__NO_JOIN")
    c = scope(sample_scope="CYCLES_ORIGINAL_OFFSETS_0_16")
    j = compare_source_scopes(a, c)
    assert j["status"] == "SUBJECT_LEVEL_ONLY__NO_LOCAL_CONSEQUENCE_JOIN"
    assert j["mismatched_scope_axes"] == ["sample_scope"]
    assert not j["local_consequence_join_authorized"]
    assert compare_source_scopes(a, scope())["local_consequence_join_authorized"]
    assert not compare_source_scopes(a, scope())["artist_owner_join_authorized"]


def test_complete_quotient_certificate_all_signatures_pairwise_distinguishable():
    rows = list(itertools.product((False, True), repeat=3))
    tr = mk(rows, operations=("A", "B", "C"))
    proof = tr.prove_complete_quotient()
    assert proof["unique_complete_trace_profile_count"] == 8
    assert len(proof["pairwise_profile_separation_certificates"]) == 28
    assert all(c["observed_values"][0] != c["observed_values"][1]
               for c in proof["pairwise_profile_separation_certificates"])


def test_invalid_trace_masks_and_unqualified_scope_sha_fail_closed():
    with pytest.raises(ValueError, match="INVALID_OBSERVED_TRACE"):
        mk(((True, 1), (False, None)))
    with pytest.raises(ValueError, match="INVALID_SCOPE_PARENT_SHA"):
        scope(source_sha256="bad")
    with pytest.raises(ValueError, match="TRACE_IDS_DUPLICATED_OR_EMPTY"):
        PartialBooleanTrace(
            loci=("same", "same"), operations=("c",),
            values=((True,), (False,)), source_scope=scope())


def test_exhaustive_solver_does_not_claim_unbounded_k_for_large_source():
    tr = mk(tuple((True, False) for _ in range(9)))
    with pytest.raises(ValueError, match="EXHAUSTIVE_PROOF_LIMIT_EIGHT"):
        prove_bounded_partition_version_space(
            tr, grammar="UNIFORM_BOOLEAN_PER_HANDLE")
    assert tr.prove_complete_quotient()["unique_complete_trace_profile_count"] == 1


def test_packet_labels_are_not_an_observable_in_masked_carrier():
    # Construct two different packet assignments operating on exactly the
    # same source face signals. Per-locus edit masks and sample-local depth
    # sorting are packet-independent in this explicitly declared grammar.
    faces = (("front", 0.3, (0.3, 0.0, 0.0, 0.5)),
             ("back", 0.8, (0.0, 0.4, 0.0, 1.0)))
    operations = ({"front"}, {"back"}, {"front", "back"})
    one_packet = {"front": 0, "back": 0}
    two_packet = {"front": 0, "back": 1}

    def render(partition, edited):
        assert set(partition) == {face[0] for face in faces}
        rgb = [0.0, 0.0, 0.0]
        alpha = 0.0
        for label, depth, (r, g, b, a) in sorted(
                faces, key=lambda x: x[1], reverse=True):
            if label in edited:
                r, g, b = 1-r, 1-g, 1-b
            rgb = [c * a + old * (1-a)
                   for c, old in zip((r, g, b), rgb)]
            alpha = a + alpha * (1-a)
        return (*rgb, alpha)

    assert one_packet != two_packet
    assert all(render(one_packet, op) == render(two_packet, op)
               for op in operations)
    # This is a mathematical witness for representation non-uniqueness,
    # not original artist object reconstruction.


def test_invalid_grammar_not_interpreted_as_new_truth():
    with pytest.raises(ValueError, match="UNKNOWN_REPRESENTATION_GRAMMAR"):
        prove_bounded_partition_version_space(
            mk(((True, False),)), grammar="ARTIST_LAYER")
