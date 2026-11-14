from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from models.rss_vnext.animation_ready_mesh import posed_lbs_from_joint_matrices
from models.rss_vnext.mira_embedded_codec_source_training_v1 import (
    RSSvNextEmbeddedMIRASourceCodecTrainerV1,
)
from models.rss_vnext.training_engine import (
    source_surface_tensor_sha256, axis_mesh_teacher_address_digest_v1,
    axis_teacher_sequence_sha256_v6,
)
from tests.models.test_rss_vnext_mira_embedded_codec_source_training_v1 import (
    DATASET, _fixture, _trainer,
)


@pytest.fixture(autouse=True)
def seeded():
    torch.manual_seed(211)
    torch.set_num_threads(1)


def pose_fixture():
    model,batch,skin_registry=_fixture()
    with torch.no_grad():
        mesh=model.build_mesh_candidate(
            batch.scene,batch.topology_teacher_tokens[0],
            source_authority="TEACHER_FORCED_TRAINING_ONLY",
        )
    source_rest=mesh.rest_vertices_iris.detach().clone()
    v=source_rest.shape[1]
    j=batch.source_dense_weights.shape[-1]
    assert j==2
    matrices=torch.eye(4).view(1,1,1,4,4).expand(1,3,j,4,4).clone()
    # First pose at rest. In poses 2–3 joint-1 moves independently;
    # correctly decoded source weights must transmit partial influence.
    matrices[:,1,1,0,3]=.9
    matrices[:,2,1,1,3]=-.7
    poses=posed_lbs_from_joint_matrices(
        source_rest,batch.source_dense_weights,matrices,
    ).detach()
    mask=torch.ones((1,3,v),dtype=torch.bool)
    mask[0,1,0]=False
    batch=replace(
        batch,source_rest_vertices=source_rest,
        source_joint_skin_matrices=matrices,
        source_posed_vertices=poses,
        source_pose_vertex_mask=mask,
        source_pose_admission_key="source-native-motion",
    )
    receipt={
        "admission_status":"QUALIFIED_NATIVE_POSED_LBS_VERTEX_TEACHER",
        "dataset_manifest_sha256":DATASET,
        "source_qualification_key":batch.source_qualification_key,
        "source_binding_sha256":batch.source_binding_sha256,
        "scene_generator_binding_sha256":batch.scene.generating_checkpoint_sha256,
        "mesh_teacher_address_digest":axis_mesh_teacher_address_digest_v1(mesh),
        "axis_teacher_sequence_sha256":axis_teacher_sequence_sha256_v6(
            batch.axis_teacher_sequence,
        ),
        "source_rest_vertices_sha256":source_surface_tensor_sha256(source_rest),
        "source_joint_skin_matrices_sha256":source_surface_tensor_sha256(matrices),
        "source_posed_vertices_sha256":source_surface_tensor_sha256(poses),
        "source_pose_vertex_mask_sha256":source_surface_tensor_sha256(mask),
        "reindexed_tessa_vertices_and_axis_joints":True,
        "pose_witness_source":"INDEPENDENT_SOURCE_NATIVE_ANIMATED_MESH",
        "generated_view_is_motion_teacher":False,
    }
    return model,batch,skin_registry,{"source-native-motion":receipt}


def trainer(model,skin_registry,pose_registry,weight=2.):
    return RSSvNextEmbeddedMIRASourceCodecTrainerV1(
        model,torch.optim.AdamW(model.mira.codec.parameters(),lr=1e-3),
        dataset_manifest_sha256=DATASET,
        trusted_source_skin_registry=skin_registry,
        trusted_source_pose_registry=pose_registry,
        pose_consequence_weight=weight,
    )


def test_mira_source_native_pose_loss_drives_codec_and_hard_runtime_parity():
    net,batch,skin,pose=pose_fixture()
    engine=trainer(net,skin,pose)
    ref=net.mira.codec.bias_head.weight.detach().clone()
    result=engine.train_step(batch)
    assert result.source_pose_deformation_loss is not None
    assert torch.isfinite(result.source_pose_deformation_loss)
    assert float(result.source_pose_deformation_loss)>0
    assert result.qualified_pose_vertex_count==int(batch.source_pose_vertex_mask.sum())
    assert result.hard_token_pose_replay_max_abs_error<=1e-5
    assert result.hard_decoder_replay_max_abs_error<=1e-5
    assert result.sparse_zero_influence_error is not None
    assert result.sparse_nonzero_influence_error is not None
    assert net.mira.codec.bias_head.weight.grad is not None
    assert net.mira.codec.bias_head.weight.grad.abs().sum()>0
    assert not torch.equal(net.mira.codec.bias_head.weight,ref)
    assert net.axis.kind_head.weight.grad is None
    assert not result.generated_skin_program_qualified
    assert not result.product_skin_qualified


def test_pose_supervision_changes_gradient_not_only_diagnostic_metric():
    net,batch,skin,pose=pose_fixture()
    control=_trainer(net,skin)
    # The same codec and exact source example: compare backprop without
    # deform loss, vs with source-native pose teacher enabled.
    no_pose=replace(
        batch,source_rest_vertices=None,source_joint_skin_matrices=None,
        source_posed_vertices=None,source_pose_vertex_mask=None,
        source_pose_admission_key="",
    )
    net.mira.codec.zero_grad(set_to_none=True)
    baseline=control.objective(no_pose)
    baseline.loss.backward()
    gradient_baseline=net.mira.codec.bias_head.weight.grad.detach().clone()
    net.mira.codec.zero_grad(set_to_none=True)
    trained=trainer(net,skin,pose).objective(batch)
    trained.loss.backward()
    gradient_pose=net.mira.codec.bias_head.weight.grad.detach().clone()
    assert torch.isfinite(gradient_pose).all()
    assert not torch.allclose(gradient_baseline,gradient_pose,atol=1e-8)


def test_pose_teacher_absence_missing_lineage_and_tamper_fail_closed():
    net,batch,skin,pose=pose_fixture()
    engine=trainer(net,skin,pose)
    with pytest.raises(ValueError,match="POSE_RECEIPT_DRIFT"):
        engine.objective(replace(
            batch,source_posed_vertices=batch.source_posed_vertices+.1,
        ))
    with pytest.raises(ValueError,match="POSE_SUPERVISION_INCOMPLETE"):
        engine.objective(replace(
            batch,source_joint_skin_matrices=None,
        ))
    with pytest.raises(ValueError,match="POSE_RECEIPT_DRIFT"):
        trainer(net,skin,{"source-native-motion":{
            **pose["source-native-motion"],
            "generated_view_is_motion_teacher":True,
        }}).objective(batch)
    mask=batch.source_pose_vertex_mask.clone()
    mask[:]=False
    with pytest.raises(ValueError,match="SOURCE_POSE_DOMAIN"):
        engine.objective(replace(batch,source_pose_vertex_mask=mask))
    distorted=batch.source_joint_skin_matrices.clone()
    distorted[...,3,3]=0
    with pytest.raises(ValueError,match="SOURCE_POSE_DOMAIN"):
        engine.objective(replace(batch,source_joint_skin_matrices=distorted))


def test_pose_loss_policy_and_source_registry_bound_to_resume(tmp_path):
    net,batch,skin,pose=pose_fixture()
    engine=trainer(net,skin,pose)
    engine.train_step(batch)
    path=tmp_path/"source-native-pose-codec.pt"
    engine.save_checkpoint(path)
    peer=trainer(net,skin,pose)
    peer.load_checkpoint(path)
    assert peer.step==1
    no_motion=RSSvNextEmbeddedMIRASourceCodecTrainerV1(
        net,torch.optim.AdamW(net.mira.codec.parameters(),lr=1e-3),
        dataset_manifest_sha256=DATASET,
        trusted_source_skin_registry=skin,
    )
    with pytest.raises(ValueError,match="SOURCE_CODEC_RESUME_DRIFT"):
        no_motion.load_checkpoint(path)
    with pytest.raises(ValueError,match="POSE_LOSS_POLICY"):
        trainer(net,skin,pose,weight=-1.)
