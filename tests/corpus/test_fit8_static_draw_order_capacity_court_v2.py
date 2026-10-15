import pytest

from tools.corpus.fit8_static_draw_order_capacity_court_v1 import (
    COMPONENT_TOTAL_ORDER_FALSIFIED,
    COMPONENT_TOTAL_ORDER_NOT_FALSIFIED,
)
from tools.corpus.fit8_static_draw_order_capacity_court_v2 import (
    evaluate_cohort_static_draw_order_capacity_v2,
    evaluate_subject_static_draw_order_capacity_v2,
)


def _summary(edges_by_view, *, subject_key="unit", component_count=3):
    pair_evidence = {}
    for view_index in range(8):
        counts = {(int(a), int(b)): int(c) for a, b, c in edges_by_view.get(view_index, [])}
        ordered = [[a, b, counts[(a, b)]] for a, b in sorted(counts)]
        mixed = []
        for lo, hi in sorted({tuple(sorted(edge)) for edge in counts}):
            forward = counts.get((lo, hi), 0)
            reverse = counts.get((hi, lo), 0)
            if forward and reverse:
                mixed.append([lo, hi, forward, reverse])
        pair_evidence[str(view_index)] = {
            "ordered_pair_count": len(ordered),
            "ordered_pairs": ordered,
            "mixed_pair_count": len(mixed),
            "mixed_pairs": mixed,
        }
    return {
        "schema": "RealSaS.FIT8StaticRayCensus.v3",
        "subject_key": subject_key,
        "policy": {"resolution": 128},
        "component_count": component_count,
        "static_views": [
            {"view_index": i, "max_distinct_component_stack": component_count}
            for i in range(8)
        ],
        "pair_evidence": pair_evidence,
    }


def test_two_cycle_from_mixed_pair_falsifies_total_order():
    row = evaluate_subject_static_draw_order_capacity_v2(
        _summary({0: [(0, 1, 4), (1, 0, 3)]})
    )
    assert row.component_total_order_state == COMPONENT_TOTAL_ORDER_FALSIFIED
    assert row.mixed_pair_view_count == 1
    assert row.directed_cycle_view_count == 1
    assert row.cycle_witnesses[0].cycle == (0, 1, 0)


def test_three_cycle_without_any_mixed_pair_also_falsifies_total_order():
    row = evaluate_subject_static_draw_order_capacity_v2(
        _summary({0: [(0, 1, 5), (1, 2, 6), (2, 0, 7)]})
    )
    assert row.mixed_pair_view_count == 0
    assert row.directed_cycle_view_count == 1
    assert row.component_total_order_state == COMPONENT_TOTAL_ORDER_FALSIFIED
    assert row.cycle_witnesses[0].cycle == (0, 1, 2, 0)


def test_dag_is_only_not_falsified_not_promoted_to_generic_pass():
    row = evaluate_subject_static_draw_order_capacity_v2(
        _summary({0: [(0, 1, 5), (0, 2, 6), (1, 2, 7)]})
    )
    assert row.directed_cycle_view_count == 0
    assert row.component_total_order_state == COMPONENT_TOTAL_ORDER_NOT_FALSIFIED


def test_mixed_pair_summary_must_match_complete_ordered_graph():
    summary = _summary({0: [(0, 1, 5), (1, 0, 3)]})
    summary["pair_evidence"]["0"]["mixed_pairs"] = []
    summary["pair_evidence"]["0"]["mixed_pair_count"] = 0
    with pytest.raises(ValueError, match="MIXED_PAIR_EVIDENCE_MISMATCH"):
        evaluate_subject_static_draw_order_capacity_v2(summary)


def test_cohort_preserves_open_replacement_decision():
    cohort = evaluate_cohort_static_draw_order_capacity_v2(
        [
            _summary({0: [(0, 1, 1), (1, 2, 1), (2, 0, 1)]}, subject_key="cyclic"),
            _summary({0: [(0, 1, 1), (1, 2, 1)]}, subject_key="dag"),
        ]
    )
    assert cohort.component_total_order_state == COMPONENT_TOTAL_ORDER_FALSIFIED
    assert cohort.replacement_mechanism_frozen is False
    assert "VIEW_POSE_CONDITIONED_FRAGMENT_DAG" in cohort.replacement_candidates
