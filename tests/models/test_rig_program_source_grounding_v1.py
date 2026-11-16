from __future__ import annotations

"""Source-truth-only mathematical regression for AXIS/MIRA RigProgram."""

from types import SimpleNamespace

import pytest
import torch

from models.rss_vnext.training_rig_program_grounding_v1 import (
    GROUNDING_SCHEMA_V1,
    maximal_source_bone_chains_v1,
    qualified_rig_program_grounding_objective_v1,
    source_rig_grounding_digest_v1,
)


def _fixture(*, include_pose=True, include_frame=True, include_control=True):
    torch.manual_seed(7)
    # Root + branching arm (1->2) and accessory bone (3).
    parents = torch.tensor([-1, 0, 1, 0], dtype=torch.long)
    source_joints = torch.tensor([[
        [-0.75, -0.75, 0.0],
        [-0.25, -0.25, 0.0],
        [0.25, 0.25, 0.0],
        [-0.25, 0.75, 0.0],
    ]], dtype=torch.float32)
    gt_skin = torch.tensor([[
        [0.8, 0.2, 0.0, 0.0],
        [0.0, 0.0, 0.1, 0.9],
        [0.0, 0.2, 0.8, 0.0],
    ]], dtype=torch.float32)
    prediction = torch.nn.Parameter(torch.tensor([[
        [0.5, 0.3, 0.1, 0.1],
        [0.0, 0.3, 0.1, 0.6],
        [0.0, 0.4, 0.5, 0.1],
    ]], dtype=torch.float32))
    # Joint rows, four frame rows, control, anchor; these are teacher tokens
    # and are not synthetic SOURCE truth for control consequences.
    kinds = [1, 2, 2, 2, 2, 3, 4, 4, 4, 4, 5, 6, 8, 10, 14, 15, 16, 17, 18]
    ptr = [-1] * len(kinds)
    for index, value in enumerate((0, 1, 2, 3), start=6):
        ptr[index] = value
    teacher = SimpleNamespace(
        kind_ids=torch.tensor([kinds], dtype=torch.long),
        pointer_a=torch.tensor([ptr], dtype=torch.long),
    )
    S = len(kinds)
    coord = torch.nn.Parameter(torch.randn(1, S, 3, 8) * 0.5)
    frames = torch.zeros((1, S, 6))
    frames[..., 0] = 1
    frames[..., 4] = 1
    frames.requires_grad_()
    dof = torch.nn.Parameter(torch.randn((1, S, 9)))
    outputs = {"coord_logits": coord, "frame_6d": frames, "dof_logits": dof}

    full = {
        "source_joint_xyz": source_joints,
        "source_parent_indices": parents,
        "source_skin_weights": gt_skin,
        "source_frame_basis": torch.eye(3)[None, None].expand(1, 4, 3, 3).clone()
        if include_frame else None,
        "source_dof_known_mask": torch.tensor([[
            [True] + [False] * 8,
            [True] + [False] * 8,
            [True] + [False] * 8,
            [True] + [False] * 8,
        ]]) if include_frame else None,
        "source_dof_allowed": torch.tensor([[
            [1.] + [0.] * 8,
            [0.] + [0.] * 8,
            [1.] + [0.] * 8,
            [0.] + [0.] * 8,
        ]]) if include_frame else None,
        "source_control_rest_xyz": torch.tensor([[[0.125, 0., 0.]]])
        if include_control else None,
        "source_anchor_local_xyz": torch.tensor([[[0.25, 0., 0.]]])
        if include_control else None,
        "source_rest_vertices": None,
        "source_pose_matrices": None,
        "source_posed_vertices": None,
        "source_pose_mask": None,
    }
    if include_pose:
        rest = torch.tensor([[
            [-0.4, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.3, 0.2, 0.1],
        ]])
        matrices = torch.eye(4)[None, None, None].expand(1, 2, 4, 4, 4).clone()
        # Move bones independently: not an uninformative global rigid pose.
        matrices[0, 1, 1, 0, 3] = 0.25
        matrices[0, 1, 3, 1, 3] = -0.20
        hom = torch.cat((rest, rest.new_ones(1, 3, 1)), -1)
        transported = torch.einsum("bpjik,bvk->bpvji", matrices, hom)[..., :3]
        posed = (transported * gt_skin[:, None, :, :, None]).sum(-2)
        full.update(
            source_rest_vertices=rest, source_pose_matrices=matrices,
            source_posed_vertices=posed, source_pose_mask=torch.ones(1, 2, dtype=torch.bool),
        )
    source_hash_fields = {
        "joint_xyz": full["source_joint_xyz"],
        "parents": full["source_parent_indices"],
        "skin_weights": full["source_skin_weights"],
        "frame_basis": full["source_frame_basis"],
        "dof_known_mask": full["source_dof_known_mask"],
        "dof_allowed": full["source_dof_allowed"],
        "control_rest_xyz": full["source_control_rest_xyz"],
        "anchor_local_xyz": full["source_anchor_local_xyz"],
        "rest_vertices": full["source_rest_vertices"],
        "pose_matrices": full["source_pose_matrices"],
        "posed_vertices": full["source_posed_vertices"],
        "pose_mask": full["source_pose_mask"],
    }
    receipt = {
        "admission_status": "QUALIFIED_SOURCE_RIG_PROGRAM_MATH_V1",
        "schema": GROUNDING_SCHEMA_V1,
        "dataset_manifest_sha256": "a" * 64,
        "source_qualification_key": "source-1",
        "axis_teacher_sequence_sha256": "b" * 64,
        "mesh_teacher_topology_digest": "c" * 64,
        "source_tensors_sha256": source_rig_grounding_digest_v1(**source_hash_fields),
        "joint_index_and_mesh_vertex_reindex_proven": True,
        "source_semantics": "NATIVE_RIG_SKIN_NOT_PSEUDOLABEL",
        "dof_truth_authority": "INDEPENDENT_SOURCE_INTERVENTION",
        "dof_truth_witness_sha256": "d" * 64,
        "control_rest_authority": "NATIVE_AUTHORED_CONTROL",
        "anchor_local_authority": "SOURCE_NATIVE_ANCHOR",
        "pose_truth_authority": "INDEPENDENT_SOURCE_LBS_POSES",
    }
    args = {
        "teacher": teacher,
        "outputs": outputs,
        "predicted_skin": prediction,
        **full,
        "trusted_receipt": receipt,
        "expected_dataset_manifest_sha256": "a" * 64,
        "expected_source_qualification_key": "source-1",
        "expected_axis_teacher_sha256": "b" * 64,
        "expected_mesh_teacher_topology_digest": "c" * 64,
    }
    return args, coord, prediction, frames, dof


def test_maximal_chains_are_generic_and_disjoint():
    assert maximal_source_bone_chains_v1(
        torch.tensor([-1, 0, 1, 0], dtype=torch.long)
    ) == ((0,), (1, 2), (3,))
    assert maximal_source_bone_chains_v1(
        torch.tensor([-1, -1, 0, 1], dtype=torch.long)
    ) == ((0,), (1,), (2,), (3,))


def test_chain_partition_rejects_cycles():
    with pytest.raises(ValueError, match="CYCLE"):
        maximal_source_bone_chains_v1(torch.tensor([1, 0], dtype=torch.long))


def test_complete_mathematics_backward_reaches_axis_and_mira():
    args, coord, pred, frames, dof = _fixture()
    loss, parts = qualified_rig_program_grounding_objective_v1(**args)
    assert set(parts) == {
        "chain_skin", "bone_skin", "joint_position", "bone_projection",
        "frame_orientation", "observed_dof", "control_rest",
        "attachment_anchor", "posed_lbs",
    }
    assert torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    assert coord.grad is not None and coord.grad.abs().sum() > 0
    assert dof.grad is not None and dof.grad.abs().sum() > 0
    assert frames.grad is not None and torch.isfinite(frames.grad).all()


def test_unknown_dof_is_not_a_negative_label():
    args, _, _, _, dof = _fixture()
    args["source_dof_known_mask"][0, 0, 0] = False
    # A source mutation must invalidate its trusted receipt first.
    with pytest.raises(ValueError, match="RECEIPT_UNQUALIFIED"):
        qualified_rig_program_grounding_objective_v1(**args)
    args["trusted_receipt"]["source_tensors_sha256"] = source_rig_grounding_digest_v1(
        joint_xyz=args["source_joint_xyz"], parents=args["source_parent_indices"],
        skin_weights=args["source_skin_weights"], frame_basis=args["source_frame_basis"],
        dof_known_mask=args["source_dof_known_mask"], dof_allowed=args["source_dof_allowed"],
        control_rest_xyz=args["source_control_rest_xyz"], anchor_local_xyz=args["source_anchor_local_xyz"],
        rest_vertices=args["source_rest_vertices"], pose_matrices=args["source_pose_matrices"],
        posed_vertices=args["source_posed_vertices"], pose_mask=args["source_pose_mask"],
    )
    loss, parts = qualified_rig_program_grounding_objective_v1(**args)
    assert "observed_dof" in parts
    loss.backward()
    # Non-excited/unknown axes do not receive a false negative gradient.
    assert torch.count_nonzero(dof.grad[..., 1:]) == 0


def test_unobserved_optional_fields_are_not_fabricated():
    args, _, _, _, _ = _fixture(include_frame=False, include_pose=False, include_control=False)
    loss, parts = qualified_rig_program_grounding_objective_v1(**args)
    assert torch.isfinite(loss)
    assert {"chain_skin", "bone_skin", "joint_position", "bone_projection"} <= set(parts)
    assert "observed_dof" not in parts
    assert "posed_lbs" not in parts
    assert "control_rest" not in parts


def test_forged_receipts_cannot_supply_teacher_truth():
    args, _, _, _, _ = _fixture()
    args["trusted_receipt"] = {**args["trusted_receipt"], "source_tensors_sha256": "0" * 64}
    with pytest.raises(ValueError, match="RECEIPT_UNQUALIFIED"):
        qualified_rig_program_grounding_objective_v1(**args)


def test_unknown_skin_mass_is_not_zero_training_truth():
    args, _, _, _, _ = _fixture()
    args["source_skin_weights"][0, 1] = 0
    with pytest.raises(ValueError, match="SOURCE_NUMERIC_OR_SKIN_INVALID"):
        qualified_rig_program_grounding_objective_v1(**args)
