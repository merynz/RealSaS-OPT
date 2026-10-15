from tools.corpus.fit8_static_draw_order_capacity_court_v1 import (
    COMPONENT_TOTAL_ORDER_FALSIFIED,
    evaluate_subject_static_draw_order_capacity_v1,
)


def test_authority_safe_static_ray_v2_is_accepted_without_changing_capacity_semantics():
    summary = {
        "schema": "RealSaS.FIT8StaticRayCensus.v2",
        "subject_key": "UNIT_V2_DRAW_ORDER",
        "policy": {"resolution": 128},
        "component_count": 2,
        "static_views": [
            {"view_index": i, "max_distinct_component_stack": 2}
            for i in range(8)
        ],
        "pair_evidence": {
            str(i): {
                "mixed_pair_count": 1 if i == 0 else 0,
                "mixed_pairs": [[0, 1, 4, 3]] if i == 0 else [],
            }
            for i in range(8)
        },
        "claims": {
            "mechanical_owner_authority_state": "DIAGNOSTIC",
            "mechanical_owner_authority": False,
        },
    }
    row = evaluate_subject_static_draw_order_capacity_v1(summary)
    assert row.component_total_order_state == COMPONENT_TOTAL_ORDER_FALSIFIED
    assert row.mixed_pair_view_count == 1
