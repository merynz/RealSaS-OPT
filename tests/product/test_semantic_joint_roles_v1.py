from types import SimpleNamespace

import pytest

from compiler.realsas_compiler_core.semantic_joint_roles import (
    canonical_joint_for_attachment_semantic,
    infer_humanoid_joint_roles,
)
from compiler.realsas_compiler_core.types import QualificationError


def _joint(jid, x, z, parent=None, proposal="P:SHUFFLED"):
    return SimpleNamespace(
        canonical_joint_id=jid,
        position=(float(x), 0.0, float(z)),
        parent_canonical_id=parent,
        source_proposal_id=proposal,
    )


def _skeleton():
    joints = (
        _joint("ROOT", 0.0, 0.40, None, "P:99"),
        _joint("SPINE", 0.0, 0.60, "ROOT", "P:14"),
        _joint("CHEST_J", 0.0, 0.95, "SPINE", "P:03"),
        _joint("HEAD_J", 0.0, 1.25, "CHEST_J", "P:08"),
        _joint("L_SHOULDER", -0.2, 0.96, "CHEST_J", "P:13"),
        _joint("L_ELBOW", -0.5, 0.95, "L_SHOULDER", "P:12"),
        _joint("L_HAND", -0.9, 0.94, "L_ELBOW", "P:06"),
        _joint("R_SHOULDER", 0.2, 0.96, "CHEST_J", "P:11"),
        _joint("R_ELBOW", 0.5, 0.95, "R_SHOULDER", "P:00"),
        _joint("R_HAND", 0.9, 0.94, "R_ELBOW", "P:01"),
        _joint("L_HIP", -0.17, 0.50, "ROOT", "P:17"),
        _joint("L_KNEE", -0.17, 0.28, "L_HIP", "P:02"),
        _joint("R_HIP", 0.17, 0.50, "ROOT", "P:04"),
        _joint("R_KNEE", 0.17, 0.28, "R_HIP", "P:05"),
    )
    return SimpleNamespace(joints=joints, deform_root_ids=("ROOT",))


def test_attachment_roles_ignore_source_proposal_indices():
    binding = infer_humanoid_joint_roles(_skeleton())
    assert binding.chest_joint_id == "CHEST_J"
    assert binding.head_joint_id == "HEAD_J"
    assert binding.left_hand_slot_joint_id == "L_HAND"
    assert binding.right_hand_slot_joint_id == "R_HAND"
    assert binding.metadata["source_proposal_index_used"] is False
    assert canonical_joint_for_attachment_semantic(binding, "handslot.l") == "L_HAND"
    assert canonical_joint_for_attachment_semantic(binding, "handslot.r") == "R_HAND"
    assert canonical_joint_for_attachment_semantic(binding, "HEAD") == "HEAD_J"
    assert canonical_joint_for_attachment_semantic(binding, "CHEST") == "CHEST_J"


def test_unknown_attachment_semantic_fails_closed():
    binding = infer_humanoid_joint_roles(_skeleton())
    with pytest.raises(QualificationError, match="UNSUPPORTED_ATTACHMENT_SEMANTIC"):
        canonical_joint_for_attachment_semantic(binding, "source_control_8")


def test_ambiguous_multiple_deform_roots_fail_closed():
    sk = _skeleton()
    sk.deform_root_ids = ("ROOT", "SPINE")
    with pytest.raises(QualificationError, match="SINGLE_DEFORM_ROOT"):
        infer_humanoid_joint_roles(sk)
