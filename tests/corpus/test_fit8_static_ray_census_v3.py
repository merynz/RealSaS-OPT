import numpy as np
import pytest

from tools.corpus.fit8_static_ray_census_v3 import augment_static_ray_order_graph_v3


def _summary_v2():
    return {
        "schema": "RealSaS.FIT8StaticRayCensus.v2",
        "subject_key": "UNIT_ORDER_GRAPH",
        "pair_evidence": {
            str(i): {"mixed_pair_count": 0, "mixed_pairs": []}
            for i in range(8)
        },
        "claims": {"mechanical_owner_authority_state": "DIAGNOSTIC"},
    }


def _arrays():
    arrays = {}
    for i in range(8):
        arrays[f"v{i}_ordered_component_pair_counts"] = np.empty((0, 3), dtype=np.int64)
    return arrays


def test_v3_exports_complete_ordered_graph_in_canonical_order():
    summary = _summary_v2()
    arrays = _arrays()
    arrays["v0_ordered_component_pair_counts"] = np.asarray(
        [[2, 0, 4], [0, 1, 9], [1, 2, 3]], dtype=np.int64
    )
    result = augment_static_ray_order_graph_v3(summary, arrays)
    assert result["schema"] == "RealSaS.FIT8StaticRayCensus.v3"
    assert result["parent_static_ray_schema"] == "RealSaS.FIT8StaticRayCensus.v2"
    assert result["pair_evidence"]["0"]["ordered_pair_count"] == 3
    assert result["pair_evidence"]["0"]["ordered_pairs"] == [
        [0, 1, 9], [1, 2, 3], [2, 0, 4]
    ]
    assert result["claims"]["complete_ordered_component_relation_graph_exposed"] is True


def test_v3_rejects_invalid_ordered_pair_values():
    summary = _summary_v2()
    arrays = _arrays()
    arrays["v0_ordered_component_pair_counts"] = np.asarray([[0, 0, 3]], dtype=np.int64)
    with pytest.raises(ValueError, match="ORDERED_PAIR_ARRAY_VALUE_INVALID"):
        augment_static_ray_order_graph_v3(summary, arrays)


def test_v3_requires_all_eight_ordered_pair_arrays():
    summary = _summary_v2()
    arrays = _arrays()
    del arrays["v7_ordered_component_pair_counts"]
    with pytest.raises(ValueError, match="ORDERED_PAIR_ARRAY_MISSING"):
        augment_static_ray_order_graph_v3(summary, arrays)
