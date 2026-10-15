import pytest

from tools.corpus.fit8_static_draw_order_capacity_court_v1 import (
    COMPONENT_TOTAL_ORDER_FALSIFIED,
    COMPONENT_TOTAL_ORDER_NOT_FALSIFIED,
    evaluate_cohort_static_draw_order_capacity_v1,
    evaluate_subject_static_draw_order_capacity_v1,
)


def _summary(subject_key: str, mixed_counts: list[int], max_stacks: list[int]):
    assert len(mixed_counts) == len(max_stacks) == 8
    return {
        "schema": "RealSaS.FIT8StaticRayCensus.v1",
        "subject_key": subject_key,
        "policy": {"resolution": 128},
        "component_count": 3,
        "static_views": [
            {"view_index": i, "max_distinct_component_stack": int(max_stacks[i])}
            for i in range(8)
        ],
        "pair_evidence": {
            str(i): {
                "mixed_pair_count": int(mixed_counts[i]),
                "mixed_pairs": [[0, 1, 2, 3]] * int(mixed_counts[i]),
            }
            for i in range(8)
        },
    }


def test_mixed_front_back_pair_falsifies_whole_component_total_order():
    row = evaluate_subject_static_draw_order_capacity_v1(
        _summary("snake", [2, 1, 1, 1, 0, 1, 1, 1], [2, 2, 3, 2, 2, 2, 3, 2])
    )
    assert row.component_total_order_state == COMPONENT_TOTAL_ORDER_FALSIFIED
    assert row.mixed_pair_view_count == 7
    assert row.maximum_distinct_component_stack == 3


def test_absence_of_mixed_pair_is_not_promoted_to_proof_of_sufficiency():
    row = evaluate_subject_static_draw_order_capacity_v1(
        _summary("trex", [0] * 8, [1] * 8)
    )
    assert row.component_total_order_state == COMPONENT_TOTAL_ORDER_NOT_FALSIFIED


def test_cohort_falsifies_component_total_order_but_does_not_freeze_replacement():
    cohort = evaluate_cohort_static_draw_order_capacity_v1([
        _summary("trex", [0] * 8, [1] * 8),
        _summary("spider", [12, 9, 6, 4, 0, 4, 6, 9], [6, 5, 5, 5, 6, 5, 5, 5]),
    ])
    assert cohort.component_total_order_state == COMPONENT_TOTAL_ORDER_FALSIFIED
    assert cohort.replacement_mechanism_frozen is False
    assert "PER_PIXEL_OR_GEOMETRIC_DEPTH_RESOLUTION" in cohort.replacement_candidates
    assert "VIEW_POSE_CONDITIONED_FRAGMENT_DAG" in cohort.replacement_candidates


def test_pair_evidence_cardinality_mismatch_fails_closed():
    summary = _summary("bad", [1] + [0] * 7, [2] * 8)
    summary["pair_evidence"]["0"]["mixed_pairs"] = []
    with pytest.raises(ValueError, match="PAIR_EVIDENCE_INVALID"):
        evaluate_subject_static_draw_order_capacity_v1(summary)
