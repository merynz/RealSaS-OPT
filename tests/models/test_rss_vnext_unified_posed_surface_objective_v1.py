from __future__ import annotations

"""Synthetic regression ONLY: source receipts in this file are NOT real truth."""

from dataclasses import replace

import pytest
import torch

from models.axis.rig_program_model_v6 import (
    AXISSequenceV6, RigProgramEventV6, RigTokenKindV6,
)
from models.rss_vnext.animation_ready_mesh import posed_lbs_from_joint_matrices
from models.rss_vnext.training_engine import (
    RSSvNextStructuralTrainer, axis_mesh_teacher_address_digest_v1,
    axis_teacher_sequence_sha256_v6, source_surface_tensor_sha256,
)
from tests.models.test_rss_vnext_axis_mesh_matched_product_training_v1 import (
    fixture as axis_mesh_fixture,
)


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(118)
    torch.set_num_threads(1)


def _two_joint_teacher(net):
    K = RigTokenKindV6
    events = [
        RigProgramEventV6(K.BOS),
        RigProgramEventV6(K.JOINT, coord_indices=(1, 2, 3)),
        RigProgramEventV6(K.JOINT, pointer_a=0, coord_indices=(2, 3, 4)),
        RigProgramEventV6(K.FRAMES_START),
        RigProgramEventV6(
            K.FRAME, pointer_a=0,
            frame_6d=(1., 0., 0., 0., 1., 0.),
            dof_mask=(0., 0., 0., 1., 1., 1., 0., 0., 0.),
        ),
        RigProgramEventV6(
            K.FRAME, pointer_a=1,
            frame_6d=(1., 0., 0., 0., 1., 0.),
            dof_mask=(0., 0., 0., 1., 1., 1., 0., 0., 0.),
        ),
        RigProgramEventV6(K.CONTROLS_START),
        RigProgramEventV6(K.CHAINS_START),
        RigProgramEventV6(K.CONSTRAINTS_START),
        RigProgramEventV6(K.ANCHORS_START),
        RigProgramEventV6(K.SKIN_START),
    ]
    events.extend(
        RigProgramEventV6(K.SKIN_CODE, skin_code_id=i % net.mira.codec.quantizer.vocab_size)
        for i in range(2 * net.cfg.skin_tokens_per_joint)
    )
    events.append(RigProgramEventV6(K.EOS))
    return AXISSequenceV6.from_events(events, device=torch.device("cpu"))


def _qualified_synthetic_case():
    net, batch, existing, axis_receipt = axis_mesh_fixture()
    teacher = _two_joint_teacher(net)
    batch = replace(batch, axis_teacher_sequence=teacher)
    axis_receipt = {
        **axis_receipt,
        "teacher_sequence_sha256": axis_teacher_sequence_sha256_v6(teacher),
    }
    state, _, _ = net.encode_observations(batch.rgba, batch.camera_features)
    mesh = net.build_mesh_candidate(
        state, batch.topology_teacher_tokens[0],
        source_authority="TEACHER_FORCED_TRAINING_ONLY",
    )
    # In a real run ALL of these targets/lineages must come from an
    # independent native source evaluator, not a student-generated mesh.
    rest = mesh.rest_vertices_iris.detach().clone()
    v = rest.shape[1]
    skin = torch.zeros((1, v, 2), dtype=rest.dtype)
    skin[0, :, 0] = torch.linspace(.2, .8, v)
    skin[0, :, 1] = 1 - skin[0, :, 0]
    matrices = torch.eye(4, dtype=rest.dtype)[None, None, None].repeat(1, 3, 2, 1, 1)
    matrices[0, 1, 1, 0, 3] = .7
    matrices[0, 2, 0, 1, 3] = -.25
    matrices[0, 2, 1, 1, 3] = .45
    posed = posed_lbs_from_joint_matrices(rest, skin, matrices).detach()[0]
    valid = torch.tensor([False, True, True])
    batch = replace(
        batch,
        axis_pose_joint_skin_matrices=matrices,
        axis_pose_rest_surface_samples=rest[0],
        axis_pose_surface_samples=posed,
        axis_pose_valid_mask=valid,
        axis_pose_admission_key="synthetic-pose",
    )
    receipt = {
        "admission_status": "QUALIFIED_SOURCE_NATIVE_AXIS_TESSA_POSED_SURFACE",
        "dataset_manifest_sha256": "3" * 64,
        "source_qualification_key": batch.source_qualification_key,
        "source_binding_sha256": batch.product_source_binding_sha256,
        "observed_rgba_sha256": source_surface_tensor_sha256(batch.rgba),
        "observed_camera_sha256": source_surface_tensor_sha256(batch.camera_features),
        "mesh_teacher_topology_digest": axis_mesh_teacher_address_digest_v1(mesh),
        "axis_teacher_sequence_sha256": axis_teacher_sequence_sha256_v6(teacher),
        "source_joint_skin_matrices_sha256": source_surface_tensor_sha256(matrices),
        "source_rest_surface_samples_sha256": source_surface_tensor_sha256(rest[0]),
        "source_posed_surface_samples_sha256": source_surface_tensor_sha256(posed),
        "source_pose_valid_mask_sha256": source_surface_tensor_sha256(valid),
        "pose_witness_source": "INDEPENDENT_SOURCE_NATIVE_LBS_MESH",
        "generated_view_is_motion_teacher": False,
        "reindexed_tessa_vertices_and_axis_joints": True,
        "normalized_geometry_domain": "IRIS_NORMALIZED_SOURCE_XYZ",
    }
    weights = replace(existing.weights, axis_posed_surface=1.)
    def trainer(registry):
        return RSSvNextStructuralTrainer(
            net, torch.optim.AdamW(net.parameters(), lr=1.e-3),
            dataset_manifest_sha256="3" * 64, weights=weights,
            axis_training_policy="PRODUCT_MESH_MATCHED_REQUIRED",
            trusted_axis_teacher_registry={"axis": axis_receipt},
            trusted_axis_pose_registry=registry,
        )
    return net, batch, trainer, receipt


def test_unified_source_pose_reaches_axis_skin_and_tessa_surface_gradients():
    net, batch, factory, receipt = _qualified_synthetic_case()
    trainer = factory({"synthetic-pose": receipt})
    result = trainer.objective(batch)
    assert result.axis_training_conditioning == "PRODUCT_MESH_MATCHED_TESSA_H"
    assert result.axis_posed_surface_loss is not None
    assert result.eligible_axis_pose_count == 2
    assert torch.isfinite(result.axis_posed_surface_loss)
    result.axis_posed_surface_loss.backward()
    assert net.axis.skin_code_head.weight.grad is not None
    assert float(net.axis.skin_code_head.weight.grad.abs().sum()) > 0
    assert net.surface_refine[-1].weight.grad is not None
    assert float(net.surface_refine[-1].weight.grad.abs().sum()) > 0
    assert not any(p.grad is not None for p in net.mira.codec.parameters())


def test_unified_source_pose_rejects_changed_targets_and_untrusted_truth():
    _, batch, factory, receipt = _qualified_synthetic_case()
    trainer = factory({"synthetic-pose": receipt})
    with pytest.raises(ValueError, match="POSE_RECEIPT_UNQUALIFIED"):
        trainer.objective(replace(
            batch,
            axis_pose_surface_samples=batch.axis_pose_surface_samples + .5,
        ))
    with pytest.raises(ValueError, match="POSE_RECEIPT_UNQUALIFIED"):
        factory({"synthetic-pose": {
            **receipt, "generated_view_is_motion_teacher": True,
        }}).objective(batch)
    with pytest.raises(ValueError, match="POSE_SUPERVISION_INCOMPLETE"):
        trainer.objective(replace(batch, axis_pose_valid_mask=None))


def test_unified_source_pose_rejects_idle_as_skin_teacher():
    _, batch, factory, receipt = _qualified_synthetic_case()
    matrices = torch.eye(4)[None, None, None].repeat(
        1, 3, 2, 1, 1,
    )
    amended = replace(batch, axis_pose_joint_skin_matrices=matrices)
    valid_receipt = {
        **receipt,
        "source_joint_skin_matrices_sha256": source_surface_tensor_sha256(matrices),
    }
    with pytest.raises(ValueError, match="NO_DIFFERENTIAL_JOINT_MOTION"):
        factory({"synthetic-pose": valid_receipt}).objective(amended)


def test_unified_pose_source_registry_is_checkpoint_bound(tmp_path):
    _, batch, factory, receipt = _qualified_synthetic_case()
    trainer = factory({"synthetic-pose": receipt})
    path = tmp_path / "pose-trainer.pt"
    trainer.save_checkpoint(path, sampler_cursor=9)
    assert trainer.load_checkpoint(path) == 9
    with pytest.raises(ValueError, match="AXIS_POSE_REGISTRY_DRIFT"):
        factory({"synthetic-pose": {
            **receipt, "normalized_geometry_domain": "OTHER",
        }}).load_checkpoint(path)


def test_unified_pose_objective_opt_in_not_silent():
    _, batch, factory, receipt = _qualified_synthetic_case()
    trainer = factory({"synthetic-pose": receipt})
    no_pose = replace(
        batch,
        axis_pose_joint_skin_matrices=None,
        axis_pose_rest_surface_samples=None,
        axis_pose_surface_samples=None,
        axis_pose_valid_mask=None,
        axis_pose_admission_key="",
    )
    result = trainer.objective(no_pose)
    assert result.axis_posed_surface_loss is None
    assert result.axis_rig_skin_loss is not None
