from __future__ import annotations

import numpy as np

from skeleton_projection_corpus_audit_v1 import (
    _skin_audit_v1,
    aggregate_projection_audit_v1,
    helper_skip_hops_for_deform_v1,
    nearest_deform_ancestor_v1,
    summarize_numeric_v1,
)


def synthetic():
    parents = np.asarray([-1, 0, 1, -1, 3, 4, -1], np.int64)
    deform = np.asarray([1, 0, 1, 1, 0, 1, 0], bool)
    skin = np.asarray([
        [.55, .10, .10, .10, .05, .05, .05],
        [.50, .10, .10, .10, .10, .05, .05],
        [.45, .10, .10, .10, .10, .10, .05],
        [.50, .05, .10, .10, .10, .10, .05],
    ], np.float64)
    return parents, deform, skin


def test_helpers():
    p, d, _ = synthetic()
    assert nearest_deform_ancestor_v1(1, p, d) == 0
    assert nearest_deform_ancestor_v1(4, p, d) == 3
    assert nearest_deform_ancestor_v1(6, p, d) is None
    hops = helper_skip_hops_for_deform_v1(p, d)
    assert hops == {0: 0, 2: 1, 3: 0, 5: 1}


def test_skin_counterfactual_only():
    p, d, w = synthetic()
    x = _skin_audit_v1(w, p, d)
    assert x["transport_was_applied"] is False
    assert x["helper_columns_nonzero"] == 3
    assert x["helper_columns_nonzero_without_deform_ancestor"] == 1
    assert x["candidate_nearest_ancestor_transportable_mass"] > 0
    assert x["candidate_nearest_ancestor_untransportable_mass"] > 0
    assert np.isclose(x["total_mass"], 4.0)


def test_no_silent_transpose():
    p, d, w = synthetic()
    try:
        _skin_audit_v1(w.T, p, d)
    except ValueError as e:
        assert "silent transpose/reindex is forbidden" in str(e)
    else:
        raise AssertionError("transposed skin was silently accepted")


def test_aggregate():
    row = {
        "canonical_asset_id": "asset_test",
        "source_registry_id": "TEST",
        "status": "PASS",
        "rig": {
            "deform_control_count": 4,
            "source_bone_count": 7,
            "skipped_helper_count": 3,
            "projected_root_count": 2,
            "multi_root": True,
            "exact_zero_length_deform_bone_count": 1,
            "near_zero_length_deform_bone_count_rel1e8_bbox": 1,
            "helper_skip_hops_histogram": {0: 2, 1: 2},
            "max_helper_skip_hops": 1,
            "deform_controls_with_helper_skip": 2,
        },
        "skin": {
            "total_mass": 4.0,
            "nondeform_mass_total": .8,
            "nondeform_mass_fraction_total": .2,
            "candidate_nearest_ancestor_untransportable_mass": .2,
        },
    }
    a = aggregate_projection_audit_v1([row])
    assert a["multi_root_asset_count"] == 1
    assert a["deform_control_count"]["max"] == 4.0
    assert a["helper_skip_hops_global"]["p50"] == 0.5
    assert np.isclose(a["skin"]["corpus_nondeform_mass_fraction"], .2)
    assert np.isclose(a["skin"]["candidate_nearest_ancestor_untransportable_fraction_of_nondeform_mass"], .25)


def main():
    test_helpers()
    test_skin_counterfactual_only()
    test_no_silent_transpose()
    test_aggregate()
    assert summarize_numeric_v1([])["count"] == 0
    print("SKELETON_PROJECTION_CORPUS_AUDIT_V1_TEST_PASS")


if __name__ == "__main__":
    main()
