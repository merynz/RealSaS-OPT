from __future__ import annotations

"""Coulomb contact policy: sticking, bounded sliding, and moving-frame replay."""

from dataclasses import replace
import json

import numpy as np
import pytest

from compiler.realsas_compiler_core.mechanics_xpbd_contact_friction_v1 import (
    apply_xpbd_coulomb_friction_v1,
)
from compiler.realsas_compiler_core.mechanics_xpbd_surface_v1 import (
    XPBDSurfacePolicyV1, _state_hash, xpbd_surface_state_from_dict_v1,
)
from compiler.realsas_compiler_core.mechanics_xpbd_secondary_v1 import XPBDPlaneColliderV1
from compiler.realsas_compiler_core.types import QualificationError
from tests.compiler.test_mechanics_xpbd_surface_v1 import _setup,_simulate


def _v(*args):
    return np.asarray(args,dtype=np.float64)


def test_static_coulomb_friction_sticks_without_changing_contact_normal():
    position=_v(.03,.02,0.)
    start=_v(0.,0.,0.)
    end,applied=apply_xpbd_coulomb_friction_v1(
        position,start,_v(0.,1.,0.),.01,
        static_friction=4.,dynamic_friction=1.,
    )
    assert applied is True
    assert end==pytest.approx((0.,.02,0.))
    assert position==pytest.approx((.03,.02,0.))


def test_dynamic_coulomb_friction_is_bounded_by_normal_contact_impulse():
    end,applied=apply_xpbd_coulomb_friction_v1(
        _v(.03,.02,0.),_v(0.,0.,0.),_v(0.,1.,0.),.01,
        static_friction=.5,dynamic_friction=.2,
    )
    assert applied is True
    assert end==pytest.approx((.028,.02,0.))
    zero,modified=apply_xpbd_coulomb_friction_v1(
        _v(.03,.02,0.),_v(0.,0.,0.),_v(0.,1.,0.),.01,
        static_friction=0.,dynamic_friction=0.,
    )
    assert not modified
    assert zero==pytest.approx((.03,.02,0.))


def test_kinematic_collider_motion_is_subtracted_from_relative_slip():
    # A particle carried along by the moving body must not experience a
    # fabricated tangent slip. A genuine relative slip can be corrected.
    same,moved=apply_xpbd_coulomb_friction_v1(
        _v(.02,.02,0.),_v(0.,0.,0.),_v(0.,1.,0.),.01,
        static_friction=1.,dynamic_friction=.2,
        collider_step_displacement=_v(.02,0.,0.),
    )
    assert moved is False
    assert same==pytest.approx((.02,.02,0.))
    corrected,moved=apply_xpbd_coulomb_friction_v1(
        _v(.03,.02,0.),_v(0.,0.,0.),_v(0.,1.,0.),.01,
        static_friction=2.,dynamic_friction=.2,
        collider_step_displacement=_v(.02,0.,0.),
    )
    assert moved is True
    assert corrected==pytest.approx((.02,.02,0.))


def test_bad_coulomb_coefficients_nonfinite_or_nonunit_normal_fail_closed():
    with pytest.raises(QualificationError,match="FRICTION_POLICY_INVALID"):
        apply_xpbd_coulomb_friction_v1(
            _v(1.,0.,0.),_v(0.,0.,0.),_v(0.,1.,0.),.1,
            static_friction=.2,dynamic_friction=.3,
        )
    with pytest.raises(QualificationError,match="FRICTION_NORMAL_NOT_UNIT"):
        apply_xpbd_coulomb_friction_v1(
            _v(1.,0.,0.),_v(0.,0.,0.),_v(0.,3.,0.),.1,
            static_friction=.5,dynamic_friction=.2,
        )
    with pytest.raises(QualificationError,match="FRICTION_VECTOR_INVALID"):
        apply_xpbd_coulomb_friction_v1(
            _v(float("nan"),0.,0.),_v(0.,0.,0.),_v(0.,1.,0.),.1,
            static_friction=.5,dynamic_friction=.2,
        )


def _moving_state(state, vx):
    altered=replace(state,particles=tuple(
        replace(p,world_velocity=(vx,0.,0.)) for p in state.particles
    ),state_hash="")
    return replace(altered,state_hash=_state_hash(altered))


def test_surface_runtime_friction_changes_sliding_and_preserves_state_replay():
    floor=XPBDPlaneColliderV1("GROUND",(0.,0.,0.),(0.,1.,0.),margin=.002)
    base={
        "planes":(floor,),
        "gravity_xyz":(0.,-9.81,0.),
        "stretch_compliance":.001,
        "bend_compliance":.001,
        "area_compliance":.001,
        "max_stretch_relative_error":.3,
        "max_area_relative_error":.3,
    }
    parents,domain,friction,primary=_setup(
        **base,static_friction=50.,dynamic_friction=2.,
    )
    baseline=replace(friction,static_friction=0.,dynamic_friction=0.)
    start_f=_simulate(parents,domain,friction,primary)
    start_b=_simulate(parents,domain,baseline,primary)
    incoming_f=_moving_state(start_f.state,3.)
    incoming_b=_moving_state(start_b.state,3.)
    sticky=_simulate(parents,domain,friction,primary,
                     previous_state=incoming_f,dt_seconds=1./120.)
    sliding=_simulate(parents,domain,baseline,primary,
                      previous_state=incoming_b,dt_seconds=1./120.)
    assert sticky.status=="PASS_XPBD_SURFACE_RUNTIME"
    assert sliding.status=="PASS_XPBD_SURFACE_RUNTIME"
    assert sticky.metadata["declared_rigid_contact_coulomb_friction_qualified"] is True
    assert sticky.metadata["friction_contact_corrections"]>0
    assert sticky.deformation.positions!=sliding.deformation.positions
    assert sliding.metadata["friction_contact_corrections"]==0
    replay=xpbd_surface_state_from_dict_v1(
        json.loads(json.dumps(sticky.state.to_dict()))
    )
    next_a=_simulate(parents,domain,friction,primary,
                     previous_state=sticky.state,dt_seconds=1./120.)
    next_b=_simulate(parents,domain,friction,primary,
                     previous_state=replay,dt_seconds=1./120.)
    assert next_a.state==next_b.state
    assert next_a.deformation.positions==next_b.deformation.positions


def test_surface_policy_rejects_negative_friction_or_dynamic_above_static():
    parents,domain,policy,primary=_setup()
    with pytest.raises(QualificationError,match="STATIC_FRICTION_INVALID"):
        _simulate(parents,domain,replace(policy,static_friction=-.1),primary)
    with pytest.raises(QualificationError,match="COULOMB_FRICTION_ORDER_INVALID"):
        _simulate(parents,domain,replace(policy,
            static_friction=.2,dynamic_friction=.4),primary)
