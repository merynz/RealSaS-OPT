import numpy as np
import pytest

from compiler.realsas_compiler_core.control_basis_pruning_v1 import (
    collapse_weight_axis_to_parent_v1,
    prune_qualified_skeleton_control_v1,
)
from compiler.realsas_compiler_core.types import (
    QualificationError,
    QualifiedJoint,
    QualifiedSkeletonIR,
)


def _skeleton():
    joints = (
        QualifiedJoint("root", (0.0, 0.0, 0.0), None, (), "p0"),
        QualifiedJoint("mid", (0.0, 1.0, 0.0), "root", (), "p1"),
        QualifiedJoint("leaf", (0.0, 2.0, 0.0), "mid", (), "p2"),
        QualifiedJoint("side", (1.0, 1.0, 0.0), "root", (), "p3"),
    )
    return QualifiedSkeletonIR(joints, "root", {"status": "TEST"}, "S0")


def test_prune_middle_reparents_children_and_changes_lineage():
    out, receipt = prune_qualified_skeleton_control_v1(_skeleton(), "mid")
    by = {j.canonical_joint_id: j for j in out.joints}
    assert set(by) == {"root", "leaf", "side"}
    assert by["leaf"].parent_canonical_id == "root"
    assert by["side"].parent_canonical_id == "root"
    assert out.root_id == "root"
    assert out.skeleton_lineage_hash != "S0"
    assert receipt.removed_control_id == "mid"
    assert receipt.replacement_parent_id == "root"
    assert receipt.reparented_child_ids == ("leaf",)
    assert receipt.source_control_count == 4
    assert receipt.pruned_control_count == 3


def test_prune_leaf_is_legal_and_has_no_reparented_children():
    out, receipt = prune_qualified_skeleton_control_v1(_skeleton(), "leaf")
    assert {j.canonical_joint_id for j in out.joints} == {"root", "mid", "side"}
    assert receipt.reparented_child_ids == ()


def test_root_prune_is_forbidden():
    with pytest.raises(QualificationError, match="CONTROL_PRUNE_ROOT_FORBIDDEN"):
        prune_qualified_skeleton_control_v1(_skeleton(), "root")


def test_weight_axis_collapse_conserves_mass_and_removes_column():
    W = np.asarray(
        [
            [0.2, 0.3, 0.5],
            [0.0, 0.75, 0.25],
        ],
        dtype=np.float64,
    )
    out, ids = collapse_weight_axis_to_parent_v1(
        W,
        ("root", "mid", "leaf"),
        removed_control_id="mid",
        replacement_parent_id="root",
    )
    assert ids == ("root", "leaf")
    np.testing.assert_allclose(out, [[0.5, 0.5], [0.75, 0.25]])
    np.testing.assert_allclose(out.sum(axis=1), 1.0)


def test_weight_axis_rejects_unknown_binding():
    with pytest.raises(QualificationError, match="CONTROL_PRUNE_WEIGHT_BINDING_INVALID"):
        collapse_weight_axis_to_parent_v1(
            np.asarray([[0.5, 0.5]]),
            ("root", "mid"),
            removed_control_id="ghost",
            replacement_parent_id="root",
        )
