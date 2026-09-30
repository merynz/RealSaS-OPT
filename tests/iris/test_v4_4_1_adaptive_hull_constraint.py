from __future__ import annotations

import pytest
import torch

from models.iris.v4.adaptive_hull_constraint_v4_4_1 import (
    AdaptiveHullConstraintPolicyV441,
    augmented_hull_objective_v441,
    calibrate_frozen_rho_v441,
    initialize_controller_v441,
    reanchor_controller_v441,
    update_lambda_from_replay_v441,
    violation_conditioned_sign_control_v441,
)
from models.iris.v4.source_hull_lattice_v4_3 import SourceHullLatticePolicyV43


def _four_band_fixture(sdf_values, margin=0.02):
    d=torch.tensor([1.0,4.0,16.0,64.0],dtype=torch.float32)
    m=torch.full((4,),float(margin),dtype=torch.float32)
    s=torch.tensor(sdf_values,dtype=torch.float32)
    c=torch.ones(4,dtype=torch.bool)
    return s,m,d,c


def test_v441_control_zero_when_all_sign_constraints_satisfied():
    s,m,d,c=_four_band_fixture([0.01,0.02,0.03,0.04])
    row=violation_conditioned_sign_control_v441(s,m,d,c)
    assert float(row["control"])==pytest.approx(0.0)
    assert int(row["nonpositive_sign_count"])==0


def test_v441_metric_deficit_without_sign_violation_does_not_drive_dual():
    s,m,d,c=_four_band_fixture([0.01,0.01,0.01,0.01],margin=0.02)
    row=violation_conditioned_sign_control_v441(s,m,d,c)
    assert int(row["metric_deficit_violation_count"])==4
    assert int(row["nonpositive_sign_count"])==0
    assert float(row["control"])==pytest.approx(0.0)


def test_v441_sparse_tail_cannot_be_cancelled_by_satisfied_majority():
    # 1000 satisfied certified points plus one sign-violating point in a single band.
    s=torch.cat([torch.full((1000,),0.05),torch.tensor([-0.01])])
    m=torch.full_like(s,0.02)
    d=torch.ones_like(s)
    c=torch.ones_like(s,dtype=torch.bool)
    row=violation_conditioned_sign_control_v441(
        s,m,d,c,require_all_bands=False
    )
    assert int(row["nonpositive_sign_count"])==1
    assert float(row["control"])>0.0


def test_v441_equal_band_control_keeps_zero_for_clean_bands():
    s,m,d,c=_four_band_fixture([-0.01,0.02,0.03,0.04])
    row=violation_conditioned_sign_control_v441(s,m,d,c)
    # First band severity clip((0.02-(-0.01))/beta,0,1)=1 with frozen beta;
    # remaining three clean bands contribute zero -> equal-band control 0.25.
    assert float(row["control"])==pytest.approx(0.25)
    assert int(row["nonpositive_sign_count"])==1


def test_v441_replay_delta_is_anchored_to_full_population():
    p=AdaptiveHullConstraintPolicyV441(full_scan_interval_steps=100,initial_lambda=1.0)
    state=initialize_controller_v441(
        full_anchor_control=0.20,replay_anchor_control=0.80,anchor_step=100,policy=p
    )
    nxt,diag=update_lambda_from_replay_v441(state,replay_control_now=0.70,policy=p)
    assert diag["estimated_population_control"]==pytest.approx(0.10)
    assert diag["controller_mode"]=="ACCUMULATE"
    assert nxt.lambda_value==pytest.approx(1.001)


def test_v441_lambda_never_decreases_while_estimated_violation_positive():
    p=AdaptiveHullConstraintPolicyV441(full_scan_interval_steps=100,initial_lambda=1.0)
    state=initialize_controller_v441(
        full_anchor_control=0.01,replay_anchor_control=0.50,anchor_step=100,policy=p
    )
    nxt,diag=update_lambda_from_replay_v441(state,replay_control_now=0.50,policy=p)
    assert diag["estimated_population_control"]==pytest.approx(0.01)
    assert nxt.lambda_value>state.lambda_value


def test_v441_lambda_leaks_only_at_zero_control():
    p=AdaptiveHullConstraintPolicyV441(full_scan_interval_steps=100,initial_lambda=1.0)
    state=initialize_controller_v441(
        full_anchor_control=0.0,replay_anchor_control=0.2,anchor_step=100,policy=p
    )
    nxt,diag=update_lambda_from_replay_v441(state,replay_control_now=0.2,policy=p)
    assert diag["estimated_population_control"]==pytest.approx(0.0)
    assert diag["controller_mode"]=="ZERO_ONLY_LEAK"
    assert nxt.lambda_value==pytest.approx(0.99)


def test_v441_reanchor_does_not_reset_lambda():
    state=initialize_controller_v441(
        full_anchor_control=0.2,replay_anchor_control=0.3,anchor_step=0
    )
    state,_=update_lambda_from_replay_v441(state,replay_control_now=0.4)
    anchored=reanchor_controller_v441(
        state,full_anchor_control=0.1,replay_anchor_control=0.7,anchor_step=100
    )
    assert anchored.lambda_value==pytest.approx(state.lambda_value)


def test_v441_rho_formula_is_unchanged():
    assert calibrate_frozen_rho_v441(base_gradient_norm=6.0,hull_gradient_norm=2.0)==pytest.approx(3.0)


def test_v441_augmented_objective_is_unchanged():
    out=augmented_hull_objective_v441(
        base_total=torch.tensor(2.0),
        hull_total=torch.tensor(0.5),
        lambda_value=3.0,
        rho=4.0,
    )
    assert float(out)==pytest.approx(4.0)


def test_v441_scan_cadence_and_initial_lambda_match_v43():
    p=AdaptiveHullConstraintPolicyV441()
    h=SourceHullLatticePolicyV43()
    assert p.full_scan_interval_steps==h.full_scan_interval_steps==100
    assert p.initial_lambda==pytest.approx(h.top_level_weight)
