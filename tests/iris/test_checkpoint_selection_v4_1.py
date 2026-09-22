from models.iris.v4.checkpoint_selection_v4_1 import (
    HardSignCheckpointCandidateV41,
    dominates_hard_sign_v41,
    pareto_front_hard_sign_v41,
    stage13_checkpoint_selection_score_v41,
)


def _candidate(step, fg, bg, ext):
    return HardSignCheckpointCandidateV41(
        step=step,
        max_foreground_miss_fraction=fg,
        max_background_zero_crossing_fraction=bg,
        source_exterior_negative_fraction=ext,
    )


def test_pareto_front_does_not_allow_background_only_collapse_to_dominate():
    collapsed = _candidate(160, 1.0, 0.09375, 0.0)
    balanced = _candidate(4480, 0.03125, 0.765625, 0.00390625)
    assert not dominates_hard_sign_v41(collapsed, balanced)
    assert not dominates_hard_sign_v41(balanced, collapsed)
    front = pareto_front_hard_sign_v41((collapsed, balanced))
    assert {row.step for row in front} == {160, 4480}


def test_pareto_front_removes_strictly_worse_candidate():
    a = _candidate(100, 0.10, 0.40, 0.01)
    b = _candidate(200, 0.20, 0.50, 0.02)
    front = pareto_front_hard_sign_v41((a, b))
    assert tuple(row.step for row in front) == (100,)


def _thresholds():
    return {
        "min_recall": 0.999,
        "min_precision": 0.999,
        "max_largest_coherent_hole_fraction": 0.00025,
        "max_interior_uncovered_fraction": 0.0005,
        "min_component_recall": 0.999,
        "max_silhouette_edge_p95_px": 0.5,
    }


def _view(**overrides):
    row = {
        "recall": 0.9995,
        "precision": 0.9995,
        "minimum_eligible_component_recall": 0.9995,
        "largest_coherent_hole_fraction": 0.0001,
        "interior_uncovered_fraction": 0.0002,
        "silhouette_edge_p95_px": 0.25,
    }
    row.update(overrides)
    return row


def test_stage13_score_prefers_actual_pass_over_lower_scalar_surrogate():
    passed = stage13_checkpoint_selection_score_v41(
        step=4000,
        per_view_rows=tuple(_view() for _ in range(8)),
        thresholds=_thresholds(),
    )
    failed_rows = [_view() for _ in range(8)]
    failed_rows[0] = _view(recall=0.95)
    failed = stage13_checkpoint_selection_score_v41(
        step=160,
        per_view_rows=tuple(failed_rows),
        thresholds=_thresholds(),
    )
    assert passed.failed_rule_count == 0
    assert failed.failed_rule_count == 1
    assert passed < failed


def test_stage13_score_uses_frozen_gate_excess_not_training_loss():
    mild_rows = [_view() for _ in range(8)]
    severe_rows = [_view() for _ in range(8)]
    mild_rows[0] = _view(silhouette_edge_p95_px=1.0)
    severe_rows[0] = _view(silhouette_edge_p95_px=20.0)
    mild = stage13_checkpoint_selection_score_v41(
        step=1000,
        per_view_rows=tuple(mild_rows),
        thresholds=_thresholds(),
    )
    severe = stage13_checkpoint_selection_score_v41(
        step=2000,
        per_view_rows=tuple(severe_rows),
        thresholds=_thresholds(),
    )
    assert mild.failed_rule_count == severe.failed_rule_count == 1
    assert mild.worst_normalized_excess < severe.worst_normalized_excess
    assert mild < severe
