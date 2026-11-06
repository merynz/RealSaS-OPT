from dataclasses import replace

import numpy as np
import pytest

from compiler.realsas_compiler_core.canonical_motion_ir_v2 import (
    canonical_motion_from_dict_v2, canonical_motion_lineage_hash_v2,
    mint_canonical_motion_v2,
)
from compiler.realsas_compiler_core.control_motion_ir_v1 import (
    qualify_control_motion_v1, seal_control_motion_proposal_v1,
)
from compiler.realsas_compiler_core.control_motion_runtime_v2 import (
    sample_control_transforms_v2, solve_motion_pose_v2,
)
from compiler.realsas_compiler_core.types import QualificationError
from tests.compiler.test_control_motion_v2 import _rig, _mechanics, _proposal


def _qualified_motion(*, loop=True):
    rig = _rig()
    mechanics = _mechanics(rig)
    source = _proposal(rig, mechanics)
    if source.loop != loop:
        source = seal_control_motion_proposal_v1(
            replace(source, loop=loop, proposal_lineage_hash="")
        )
    qualified = qualify_control_motion_v1(source, rig=rig, mechanics=mechanics)
    return rig, mechanics, mint_canonical_motion_v2(qualified)


def _position(motion, rig, mechanics, t):
    pose = solve_motion_pose_v2(
        motion, rig=rig, mechanics=mechanics, time_seconds=t
    )
    return np.asarray(pose.joints[0].solved_P)


def test_motion_loop_wrap_and_authored_endpoint_are_explicit():
    rig, mechanics, motion = _qualified_motion()
    assert np.allclose(_position(motion, rig, mechanics, -0.25), (1.5, 0, 0))
    assert np.allclose(_position(motion, rig, mechanics, 1.25), (0.5, 0, 0))
    assert np.allclose(_position(motion, rig, mechanics, 1.0), (2.0, 0, 0))
    assert motion.metadata["direct_joint_track_authority"] is False


def test_nonloop_motion_clamps_without_inventing_a_terminal_repair():
    rig, mechanics, motion = _qualified_motion(loop=False)
    assert np.allclose(_position(motion, rig, mechanics, -12), (0, 0, 0))
    assert np.allclose(_position(motion, rig, mechanics, 12), (2, 0, 0))


def test_motion_refuses_tampered_canonical_track_even_when_form_is_valid():
    rig, mechanics, motion = _qualified_motion()
    modified = motion.to_dict()
    modified["control_tracks"][0]["keys"][1]["translation"] = [50, 0, 0]
    with pytest.raises(QualificationError, match="CANONICAL_MOTION_V2_LINEAGE_HASH_DRIFT"):
        canonical_motion_from_dict_v2(modified)
    assert canonical_motion_from_dict_v2(motion.to_dict()) == motion


def test_motion_refuses_wrong_live_mechanics_lineage_without_retargeting():
    rig, mechanics, motion = _qualified_motion()
    wrong_mechanics = replace(mechanics, mechanics_lineage_hash="f" * 64)
    with pytest.raises(QualificationError, match="MOTION_RUNTIME_MECHANICS_LINEAGE_DRIFT"):
        sample_control_transforms_v2(
            motion, rig=rig, mechanics=wrong_mechanics, time_seconds=.25
        )


def test_motion_refuses_unknown_control_even_when_modified_clip_is_resealed():
    rig, mechanics, motion = _qualified_motion()
    bad_track = replace(motion.control_tracks[0], canonical_control_id="UNKNOWN")
    bad = replace(motion, control_tracks=(bad_track,), motion_lineage_hash="")
    bad = replace(bad, motion_lineage_hash=canonical_motion_lineage_hash_v2(bad))
    with pytest.raises(QualificationError, match="MOTION_RUNTIME_CONTROL_UNKNOWN"):
        sample_control_transforms_v2(
            bad, rig=rig, mechanics=mechanics, time_seconds=.25
        )


def test_motion_slerp_takes_short_rotation_arc_for_quaternion_sign_equivalence():
    rig, mechanics, motion = _qualified_motion()
    keys = list(motion.control_tracks[0].keys)
    keys[1] = replace(keys[1], rotation_quat_wxyz=(-1., 0., 0., 0.))
    track = replace(motion.control_tracks[0], keys=tuple(keys))
    signed = replace(motion, control_tracks=(track,), motion_lineage_hash="")
    signed = replace(signed, motion_lineage_hash=canonical_motion_lineage_hash_v2(signed))
    control, = sample_control_transforms_v2(
        signed, rig=rig, mechanics=mechanics, time_seconds=.5
    )
    assert np.allclose(control.rotation_quat_wxyz, (1., 0., 0., 0.))
