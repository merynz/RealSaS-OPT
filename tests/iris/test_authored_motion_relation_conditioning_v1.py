from __future__ import annotations

import numpy as np

from tools.iris_vnext.audit_authored_motion_relation_conditioning_v1 import compare_first_two_hits_v1


def _hits(first, second):
    return {
        "first_face": np.asarray(first, dtype=np.int64),
        "second_face": np.asarray(second, dtype=np.int64),
    }


def test_compare_hits_counts_stable_flip_changed_and_support_drift():
    rest = _hits(
        [1, 3, 5, 7, -1, 11],
        [2, 4, 6, 8, -1, 12],
    )
    pose = _hits(
        [1, 4, 5, -1, 9, 12],
        [2, 3, 9, -1, 10, 11],
    )
    row = compare_first_two_hits_v1(rest, pose)
    assert row["ray_count"] == 6
    assert row["rest_two_hit"] == 5
    assert row["pose_two_hit"] == 5
    assert row["joint_two_hit"] == 4
    assert row["same_unordered_pair"] == 3
    assert row["same_pair_same_order"] == 1
    assert row["same_pair_reversed_order"] == 2
    assert row["changed_pair"] == 1
    assert row["lost_two_hit"] == 1
    assert row["gained_two_hit"] == 1
    assert row["same_pair_flip_fraction"] == 2.0 / 3.0
    assert row["joint_pair_membership_drift_fraction"] == 1.0 / 4.0
    assert row["rest_two_hit_retention_fraction"] == 4.0 / 5.0


def test_compare_hits_empty_support_is_finite_and_zero_fraction():
    row = compare_first_two_hits_v1(_hits([-1, -1], [-1, -1]), _hits([-1, -1], [-1, -1]))
    assert row["joint_two_hit"] == 0
    assert row["same_unordered_pair"] == 0
    assert row["same_pair_flip_fraction"] == 0.0
    assert row["joint_pair_membership_drift_fraction"] == 0.0
    assert row["rest_two_hit_retention_fraction"] == 0.0


def test_compare_hits_rejects_shape_drift():
    try:
        compare_first_two_hits_v1(_hits([1], [2]), _hits([1, 2], [2, 3]))
    except ValueError as exc:
        assert "HIT_SHAPE_DRIFT" in str(exc)
    else:
        raise AssertionError("shape drift must fail closed")


def test_motion_audit_is_diagnostic_only_no_training_entry():
    from pathlib import Path

    source = Path('tools/iris_vnext/audit_authored_motion_relation_conditioning_v1.py').read_text(encoding='utf-8')
    forbidden = (
        'optimizer.step',
        'torch.optim',
        'IRISVNextFocusedRelationTrainer',
        'AUTHORIZE_IRIS_V9_FOCUSED_RELATION_OPTIMIZER_STEPS',
    )
    assert all(token not in source for token in forbidden)
    assert 'training_authorized": False' in source
    assert 'COURT_ONLY__CURRENT_HEAD_HAS_NO_EXPLICIT_POSE_CONDITIONING' in source
