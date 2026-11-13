from __future__ import annotations

"""AXIS socket -> Mechanics pose -> Studio projected handle continuity.

Synthetic source only: no sword discovery, source contact, collision,
secondary motion or hidden appearance admission is asserted.
"""

from dataclasses import replace
import math

import numpy as np
import pytest

from compiler.realsas_compiler_core.canonical_rig_ir_v2 import (
    CanonicalAttachmentAnchorIRV2,
    _hash_control_program,
    canonical_rig_lineage_hash_v2,
)
from compiler.realsas_compiler_core.controllable_puppet_runtime_v1 import (
    ControllablePuppetSessionV1,
)
from compiler.realsas_compiler_core.mechanics_attachment_socket_runtime_v1 import (
    solve_attachment_socket_world_v1,
)
from compiler.realsas_compiler_core.mechanics_control_solver_v1 import (
    ControlTransformIRV1,
)
from compiler.realsas_compiler_core.playback_full_surface_v3 import (
    project_points_xyz_v3,
)
from compiler.realsas_compiler_core.types import QualificationError
from tests.compiler.test_controllable_puppet_runtime_v1 import (
    qualified_synthetic_bundle,
)


def _bundle_with_sockets():
    bundle = qualified_synthetic_bundle()
    anchors = (
        CanonicalAttachmentAnchorIRV2(
            canonical_anchor_id="A:held",
            parent_canonical_control_id="C_ROOT",
            local_P=(0.0, 0.25, 0.0),
            source_proposal_anchor_id="P:A:tool",
        ),
        CanonicalAttachmentAnchorIRV2(
            canonical_anchor_id="A:joint",
            parent_canonical_joint_id="J0",
            local_P=(0.2, 0.0, 0.0),
            source_proposal_anchor_id="P:A:joint",
        ),
    )
    rig = replace(bundle.rig, attachment_anchors=anchors,
                  control_program_hash="", rig_lineage_hash="")
    rig = replace(rig, control_program_hash=_hash_control_program(rig))
    rig = replace(rig, rig_lineage_hash=canonical_rig_lineage_hash_v2(rig))
    return replace(bundle, rig=rig)


def test_editor_socket_transport_uses_actual_pose_and_active_camera():
    bundle = _bundle_with_sockets()
    session = ControllablePuppetSessionV1(bundle)
    rest = session.frame()
    assert set(rest.socket_world_positions) == {"A:held", "A:joint"}
    assert rest.socket_source_contact_qualified is False
    assert np.allclose(rest.socket_world_positions["A:held"], (0.0, .25, 0.0))

    session.drag_world("C_ROOT", (0.4, 0.1, 0.0))
    moved = session.frame()
    assert np.allclose(moved.socket_world_positions["A:held"], (.4, .35, 0))
    assert moved.socket_world_positions["A:held"] != rest.socket_world_positions["A:held"]
    assert moved.deformed.positions != rest.deformed.positions
    assert moved.socket_source_contact_qualified is False

    session.set_view_index(1)
    rotated_view = session.frame()
    assert rotated_view.socket_world_positions == moved.socket_world_positions
    assert rotated_view.socket_view_positions != moved.socket_view_positions
    assert rotated_view.socket_source_contact_qualified is False
    for key, point in rotated_view.socket_world_positions.items():
        expected = project_points_xyz_v3(
            np.asarray([point], dtype=np.float64), session.cameras["V1"],
        )[0]
        assert np.allclose(rotated_view.socket_view_positions[key], expected)


def test_control_local_socket_rotation_uses_parent_frame_not_fixed_world_offset():
    bundle = _bundle_with_sockets()
    session = ControllablePuppetSessionV1(bundle)
    theta = math.pi/2
    session.set_control(ControlTransformIRV1(
        "C_ROOT", rotation_quat_wxyz=(math.cos(theta/2), 0., 0., math.sin(theta/2)),
    ))
    rotated = session.frame()
    assert np.allclose(rotated.socket_world_positions["A:held"], (-.25, 0, 0), atol=1e-7)
    assert rotated.socket_source_contact_qualified is False


def test_socket_consumer_rejects_foreign_pose_or_ambiguous_parent():
    bundle = _bundle_with_sockets()
    session = ControllablePuppetSessionV1(bundle)
    frame = session.frame()
    with pytest.raises(QualificationError, match="MECHANICS_SOCKET_POSE_PARENT_DRIFT"):
        solve_attachment_socket_world_v1(
            bundle.rig, replace(frame.pose, rig_lineage_hash="f"*64),
        )
    bad = replace(bundle.rig.attachment_anchors[0], parent_canonical_joint_id="J0")
    rig = replace(bundle.rig, attachment_anchors=(bad,),
                  control_program_hash="", rig_lineage_hash="")
    rig = replace(rig, control_program_hash=_hash_control_program(rig))
    rig = replace(rig, rig_lineage_hash=canonical_rig_lineage_hash_v2(rig))
    with pytest.raises(QualificationError, match="MECHANICS_SOCKET_PARENT_AMBIGUOUS"):
        solve_attachment_socket_world_v1(rig, replace(
            frame.pose, rig_lineage_hash=rig.rig_lineage_hash,
            control_program_hash=rig.control_program_hash,
        ))
