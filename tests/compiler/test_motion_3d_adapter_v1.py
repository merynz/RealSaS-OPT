from __future__ import annotations

from types import SimpleNamespace
import inspect

import pytest

from compiler.realsas_compiler_core.motion_3d_adapter_v1 import (
    authored_clip_sample_times_v1,
    build_joint_specs_v1,
    compile_motion_clip_to_d1_v1,
)
from compiler.realsas_compiler_core.types import QualificationError


def _skeleton():
    return SimpleNamespace(joints=(
        SimpleNamespace(canonical_joint_id="root", position=(0.0, 0.0, 0.0), parent_canonical_id=None),
        SimpleNamespace(canonical_joint_id="child", position=(1.0, 0.0, 0.0), parent_canonical_id="root"),
        SimpleNamespace(canonical_joint_id="helper", position=(2.0, 0.0, 0.0), parent_canonical_id="child"),
    ))


def _axis_contract():
    return {
        "schema": "Synthetic.AxisContract.v1",
        "status": "PASS",
        "joint_axes": [
            {
                "canonical_joint_id": "root",
                "axis_xyz": [0.0, 0.0, 1.0],
                "legacy_scalar_to_semantic_sign": 1.0,
                "role": "ROOT",
            },
            {
                "canonical_joint_id": "child",
                "axis_xyz": [0.0, 0.0, 1.0],
                "legacy_scalar_to_semantic_sign": -1.0,
                "role": "HINGE",
            },
        ],
    }


def _key(t, rotation, *, translation=(0.0, 0.0), scale=(1.0, 1.0), depth=0.0):
    return SimpleNamespace(
        time_sec=float(t),
        rotation_deg=float(rotation),
        translation_xy=translation,
        scale_xy=scale,
        depth_offset=float(depth),
    )


def _motion(*, helper_rotation=None, child_translation=(0.0, 0.0)):
    tracks = [
        SimpleNamespace(
            clip_id="run",
            canonical_joint_id="child",
            keys=(
                _key(0.0, 0.0, translation=child_translation),
                _key(0.5, 20.0, translation=child_translation),
                _key(1.0, 0.0, translation=child_translation),
            ),
        )
    ]
    if helper_rotation is not None:
        tracks.append(SimpleNamespace(
            clip_id="run",
            canonical_joint_id="helper",
            keys=(_key(0.0, helper_rotation), _key(1.0, helper_rotation)),
        ))
    return SimpleNamespace(
        clips=(SimpleNamespace(clip_id="run", duration_sec=1.0, loop=True),),
        joint_tracks=tuple(tracks),
        motion_state_hash="a" * 64,
    )


def test_axisless_helper_joint_stays_axisless_instead_of_receiving_fake_axis():
    specs, bindings, _hash = build_joint_specs_v1(_skeleton(), _axis_contract())
    assert specs[2].joint_id == "helper"
    assert specs[2].axis_rest is None
    assert "helper" not in bindings


def test_legacy_scalar_sign_is_mapped_into_semantic_3d_angle():
    compiled = compile_motion_clip_to_d1_v1(
        skeleton=_skeleton(),
        axis_contract=_axis_contract(),
        motion_state=_motion(),
        clip_id="run",
    )
    assert compiled.times == pytest.approx((0.0, 0.5, 1.0))
    assert compiled.semantic_rotation_degrees_by_frame[1]["child"] == pytest.approx(-20.0)
    # -20 around +Z sends the helper/descendant below the X axis.
    assert compiled.poses[1].joint_positions[2][1] < 0.0


def test_axisless_joint_zero_track_is_allowed_but_nonzero_track_fails_closed():
    zero = compile_motion_clip_to_d1_v1(
        skeleton=_skeleton(),
        axis_contract=_axis_contract(),
        motion_state=_motion(helper_rotation=0.0),
        clip_id="run",
    )
    assert zero.poses
    with pytest.raises(QualificationError, match="ANIMATED_JOINT_WITHOUT_FROZEN_AXIS:helper"):
        compile_motion_clip_to_d1_v1(
            skeleton=_skeleton(),
            axis_contract=_axis_contract(),
            motion_state=_motion(helper_rotation=3.0),
            clip_id="run",
        )


def test_legacy_translation_is_not_silently_upgraded_to_3d_motion():
    with pytest.raises(QualificationError, match="TRANSLATION_SEMANTICS_NOT_QUALIFIED"):
        compile_motion_clip_to_d1_v1(
            skeleton=_skeleton(),
            axis_contract=_axis_contract(),
            motion_state=_motion(child_translation=(1.0, 0.0)),
            clip_id="run",
        )


def test_authored_sample_times_are_union_of_track_keys_and_clip_endpoints():
    motion = _motion()
    assert authored_clip_sample_times_v1(motion, "run") == pytest.approx((0.0, 0.5, 1.0))


def test_d1_adapter_is_subject_agnostic():
    source = inspect.getsource(__import__(
        "compiler.realsas_compiler_core.motion_3d_adapter_v1",
        fromlist=["dummy"],
    )).lower()
    assert "mage" not in source
