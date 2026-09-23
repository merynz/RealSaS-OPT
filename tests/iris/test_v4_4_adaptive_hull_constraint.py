from __future__ import annotations

import math

import pytest
import torch

from models.iris.v4.adaptive_hull_constraint_v4_4 import (
    AdaptiveHullConstraintPolicyV44,
    augmented_hull_objective_v44,
    calibrate_frozen_rho_v44,
    initialize_controller_v44,
    reanchor_controller_v44,
    signed_equal_band_control_v44,
    update_lambda_from_replay_v44,
)
from models.iris.v4.source_hull_lattice_v4_3 import SourceHullLatticePolicyV43


def _fixture(deficits):
    # one point per frozen distance band; margin=0.02, sdf=margin-deficit
    d=torch.tensor([1.0,4.0,16.0,64.0],dtype=torch.float32)
    m=torch.full((4,),0.02,dtype=torch.float32)
    x=torch.tensor(deficits,dtype=torch.float32)
    s=m-x
    c=torch.ones(4,dtype=torch.bool)
    return s,m,d,c


def test_v44_signed_control_is_negative_when_comfortably_satisfied():
    s,m,d,c=_fixture([-0.02,-0.02,-0.02,-0.02])
    row=signed_equal_band_control_v44(s,m,d,c)
    assert float(row["control"])==pytest.approx(-1.0)
    assert int(row["nonpositive_sign_count"])==0


def test_v44_signed_control_is_positive_when_deficit_present_in_all_bands():
    s,m,d,c=_fixture([0.02,0.02,0.02,0.02])
    row=signed_equal_band_control_v44(s,m,d,c)
    assert float(row["control"])==pytest.approx(1.0)
    assert int(row["metric_deficit_violation_count"])==4


def test_v44_replay_probe_may_use_fixed_subset_of_active_bands():
    s,m,d,c=_fixture([0.01,0.01,0.01,0.01])
    keep=torch.tensor([True,False,True,False])
    row=signed_equal_band_control_v44(
        s[keep],m[keep],d[keep],c[keep],require_all_bands=False
    )
    assert row["active_band_names"]==("0_2","8_32")
    assert float(row["control"])==pytest.approx(1.0)


def test_v44_controller_uses_replay_delta_not_raw_bank_scale():
    p=AdaptiveHullConstraintPolicyV44(full_scan_interval_steps=100,initial_lambda=1.0)
    state=initialize_controller_v44(
        full_anchor_control=0.20,replay_anchor_control=0.80,anchor_step=100,policy=p
    )
    nxt,diag=update_lambda_from_replay_v44(state,replay_control_now=0.70,policy=p)
    # estimated population control is 0.20 + (0.70-0.80)=0.10, not raw 0.70.
    assert diag["estimated_population_control"]==pytest.approx(0.10)
    assert nxt.lambda_value==pytest.approx(1.001)


def test_v44_signed_controller_can_decrease_lambda():
    p=AdaptiveHullConstraintPolicyV44(full_scan_interval_steps=100,initial_lambda=1.0)
    state=initialize_controller_v44(
        full_anchor_control=-0.40,replay_anchor_control=-0.25,anchor_step=100,policy=p
    )
    nxt,_=update_lambda_from_replay_v44(state,replay_control_now=-0.25,policy=p)
    assert nxt.lambda_value==pytest.approx(0.996)


def test_v44_reanchor_does_not_reset_lambda():
    state=initialize_controller_v44(
        full_anchor_control=0.2,replay_anchor_control=0.3,anchor_step=0
    )
    state,_=update_lambda_from_replay_v44(state,replay_control_now=0.4)
    anchored=reanchor_controller_v44(
        state,full_anchor_control=-0.1,replay_anchor_control=0.7,anchor_step=100
    )
    assert anchored.lambda_value==pytest.approx(state.lambda_value)
    assert anchored.full_anchor_control==pytest.approx(-0.1)
    assert anchored.replay_anchor_control==pytest.approx(0.7)


def test_v44_rho_is_one_shot_gradient_norm_ratio():
    assert calibrate_frozen_rho_v44(base_gradient_norm=6.0,hull_gradient_norm=2.0)==pytest.approx(3.0)
    with pytest.raises(ValueError):
        calibrate_frozen_rho_v44(base_gradient_norm=1.0,hull_gradient_norm=0.0)


def test_v44_augmented_objective_formula():
    base=torch.tensor(2.0)
    phi=torch.tensor(0.5)
    out=augmented_hull_objective_v44(base_total=base,hull_total=phi,lambda_value=3.0,rho=4.0)
    assert float(out)==pytest.approx(4.0)


def test_v44_policy_matches_v43_scan_cadence():
    p=AdaptiveHullConstraintPolicyV44()
    h=SourceHullLatticePolicyV43()
    assert p.full_scan_interval_steps==h.full_scan_interval_steps==100
    assert p.initial_lambda==pytest.approx(h.top_level_weight)
