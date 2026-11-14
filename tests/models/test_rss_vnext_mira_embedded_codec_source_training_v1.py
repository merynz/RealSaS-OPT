from __future__ import annotations

from dataclasses import replace
from hashlib import sha256

import pytest
import torch

from models.axis.rig_program_model_v6 import (
    RigTokenKindV6, RigProgramEventV6, AXISSequenceV6,
)
from models.rss_vnext import (
    RSSvNextConfig, RSSvNextPatchEncoderForTests, RSSvNextUnifiedCharacterModel,
)
from models.rss_vnext.realsas_virtual_camera_policy_v1 import (
    target_canonical_and_six_cameras_v1,
)
from models.rss_vnext.skin_surface_mass_v1 import POOLING_ABI
from models.rss_vnext.mira_embedded_codec_source_training_v1 import (
    RSSvNextEmbeddedMIRACodecBatchV1,
    RSSvNextEmbeddedMIRASourceCodecTrainerV1,
)
from models.rss_vnext.training_engine import (
    axis_mesh_teacher_address_digest_v1,
    axis_teacher_sequence_sha256_v6, source_surface_tensor_sha256,
)
from tests.models.test_rss_vnext_structural_trainer import make_sample


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(101)
    torch.set_num_threads(1)


DATASET=sha256(b"source-qualified-synthetic-codec-test").hexdigest()


def two_joints(cfg):
    frame=(1.,0.,0.,0.,1.,0.)
    dof=(0.,0.,0.,1.,1.,1.,0.,0.,0.)
    events=[
        RigProgramEventV6(RigTokenKindV6.BOS),
        RigProgramEventV6(RigTokenKindV6.JOINT,coord_indices=(1,2,3)),
        RigProgramEventV6(RigTokenKindV6.JOINT,coord_indices=(4,5,6),pointer_a=0),
        RigProgramEventV6(RigTokenKindV6.FRAMES_START),
        RigProgramEventV6(RigTokenKindV6.FRAME,pointer_a=0,frame_6d=frame,dof_mask=dof),
        RigProgramEventV6(RigTokenKindV6.FRAME,pointer_a=1,frame_6d=frame,dof_mask=dof),
        RigProgramEventV6(RigTokenKindV6.CONTROLS_START),
        RigProgramEventV6(RigTokenKindV6.CHAINS_START),
        RigProgramEventV6(RigTokenKindV6.CONSTRAINTS_START),
        RigProgramEventV6(RigTokenKindV6.ANCHORS_START),
        RigProgramEventV6(RigTokenKindV6.SKIN_START),
    ]
    events.extend(
        RigProgramEventV6(RigTokenKindV6.SKIN_CODE,skin_code_id=i)
        for i in range(cfg.skin_tokens_per_joint*2)
    )
    events.append(RigProgramEventV6(RigTokenKindV6.EOS))
    return AXISSequenceV6.from_events(events,device=torch.device("cpu"))


def _fixture(mode="DOT_PRODUCT_BASELINE"):
    cfg=replace(RSSvNextConfig.tiny_for_test(),skin_decoder_mode=mode)
    model=RSSvNextUnifiedCharacterModel(
        cfg,vision_encoder=RSSvNextPatchEncoderForTests(cfg.iris.image_token_dim),
    )
    # Like a staged VQ/token codec, do not allow the latent feature producer
    # to learn or to change dropout statistics during codec bootstrap.
    model.eval()
    for name, parameter in model.named_parameters():
        if mode == "POINT_QUERY_CROSS_ATTENTION":
            selected=(
                name.startswith(("mira.codec.vertex_encode.",
                                 "mira.codec.joint_encode.",
                                 "mira.codec.latent_head."))
                or name.startswith("mira.spatial_decoder.")
            )
        else:
            selected=name.startswith("mira.codec.")
        parameter.requires_grad_(selected)
    model.mira.codec.train()
    if mode == "POINT_QUERY_CROSS_ATTENTION":
        model.mira.spatial_decoder.train()
    original=torch.rand((1,2,4,16,16))
    generated=torch.rand((1,6,4,16,16))
    normals=torch.rand((1,6,3,16,16))*2.-1.
    _,cams=target_canonical_and_six_cameras_v1(batch_size=1)
    scene_out=model.forward_generated_multiview(
        original,None,generated_rgba=generated,
        generated_cameras=cams,generated_normals=normals,
        generation_checkpoint_sha256="a"*64,
        source_binding_sha256="b"*64,
    )
    topology=make_sample(model).topology_teacher_tokens
    teacher=two_joints(cfg)
    with torch.no_grad():
        mesh=model.build_mesh_candidate(
            scene_out.state,topology[0],
            source_authority="TEACHER_FORCED_TRAINING_ONLY",
        )
    n=mesh.rest_vertices_iris.shape[1]
    weights=torch.zeros((1,n,2))
    for idx in range(n):
        weight=idx/max(1,n-1)
        weights[0,idx]=torch.tensor([1.-weight,weight])
    batch=RSSvNextEmbeddedMIRACodecBatchV1(
        scene=scene_out.state, structural_memory=scene_out.structural_memory,
        topology_teacher_tokens=topology, axis_teacher_sequence=teacher,
        source_dense_weights=weights, source_binding_sha256="b"*64,
        source_qualification_key="SYNTHETIC_UNIT_TEST_SOURCE_NOT_PRODUCT",
        admission_key="qualified-codec",
    )
    registry={"qualified-codec":{
        "admission_status":
            "QUALIFIED_SOURCE_MIRA_DENSE_SKIN_TESSA_AXIS_JOINT_REINDEX",
        "dataset_manifest_sha256":DATASET,
        "source_binding_sha256":batch.source_binding_sha256,
        "source_qualification_key":batch.source_qualification_key,
        "scene_generator_binding_sha256":scene_out.state.generating_checkpoint_sha256,
        "mesh_teacher_tokens_sha256":source_surface_tensor_sha256(topology),
        "mesh_teacher_address_digest":axis_mesh_teacher_address_digest_v1(mesh),
        "axis_teacher_sequence_sha256":axis_teacher_sequence_sha256_v6(teacher),
        "source_dense_weights_sha256":source_surface_tensor_sha256(weights),
        "source_mesh_face_reindex_proven":True,
        "source_rig_joint_reindex_proven":True,
        "full_source_weight_coverage":True,
        "teacher_semantics":"SOURCE_NATIVE_DENSE_SKIN_NOT_INFERRED_WEIGHTS",
        "codec_feature_abi":"IRIS_LOCAL_H_PLUS_REST_XYZ__AXIS_CAUSAL_PRE_JOINT_H",
        "skin_decoder_mode":mode,
        "codec_source_encoder_pooling_abi":POOLING_ABI,
    }}
    return model,batch,registry


def _trainer(model,registry):
    return RSSvNextEmbeddedMIRASourceCodecTrainerV1(
        model,torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),lr=1.e-3,
        ),
        dataset_manifest_sha256=DATASET,trusted_source_skin_registry=registry,
    )


def test_embedded_product_codec_source_step_and_exact_hard_token_replay():
    model,batch,registry=_fixture()
    trainer=_trainer(model,registry)
    old=model.mira.codec.bias_head.weight.detach().clone()
    outcome=trainer.train_step(batch)
    assert trainer.step==1
    assert outcome.token_ids.shape==(1,2,model.cfg.skin_tokens_per_joint)
    assert outcome.hard_decoder_replay_max_abs_error<=1e-5
    assert not outcome.product_skin_qualified
    assert not outcome.generated_skin_program_qualified
    assert torch.isfinite(outcome.loss)
    assert model.mira.codec.bias_head.weight.grad is not None
    assert model.mira.codec.bias_head.weight.grad.abs().sum()>0
    assert not torch.equal(model.mira.codec.bias_head.weight,old)
    assert model.axis.kind_head.weight.grad is None
    assert model.iris.scene.scene_queries.grad is None


def test_source_weight_provenance_and_topology_reindex_are_not_optional():
    model,batch,registry=_fixture()
    trainer=_trainer(model,registry)
    broken=batch.source_dense_weights.clone()
    broken[0,0]=torch.tensor([.8,.8])
    with pytest.raises(ValueError,match="SOURCE_WEIGHT_SIMPLEX_DOMAIN"):
        trainer.objective(replace(batch,source_dense_weights=broken))
    reordered=batch.topology_teacher_tokens.clone()
    reordered[0,-2]^=1
    with pytest.raises(ValueError,match="SOURCE_ADMISSION_DRIFT"):
        trainer.objective(replace(batch,topology_teacher_tokens=reordered))
    altered={"qualified-codec":{**registry["qualified-codec"],
        "source_rig_joint_reindex_proven":False}}
    with pytest.raises(ValueError,match="SOURCE_ADMISSION_DRIFT"):
        _trainer(model,altered).objective(batch)


def test_codec_checkpoint_pins_source_registry_and_model_feature_architecture(tmp_path):
    model,batch,registry=_fixture()
    trainer=_trainer(model,registry)
    trainer.train_step(batch)
    target=tmp_path/"mira-source-codec.pt"
    trainer.save_checkpoint(target)
    clone=_trainer(model,registry)
    clone.load_checkpoint(target)
    assert clone.step==1
    changed={"qualified-codec":{**registry["qualified-codec"],
        "source_dense_weights_sha256":"c"*64}}
    with pytest.raises(ValueError,match="SOURCE_CODEC_RESUME_DRIFT"):
        _trainer(model,changed).load_checkpoint(target)


def test_point_query_decoder_cannot_claim_dot_product_bootstrap_coverage():
    model,batch,registry=_fixture()
    model.cfg=replace(model.cfg,skin_decoder_mode="POINT_QUERY_CROSS_ATTENTION")
    with pytest.raises(ValueError,match="CODEC_TRAINER_POLICY"):
        _trainer(model,registry)



def test_codec_bootstrap_optimizer_cannot_silently_update_or_skip_other_experts():
    model,batch,registry=_fixture()
    with pytest.raises(ValueError,match="CODEC_OPTIMIZER_SCOPE_DRIFT"):
        RSSvNextEmbeddedMIRASourceCodecTrainerV1(
            model,torch.optim.AdamW(model.parameters(),lr=1e-3),
            dataset_manifest_sha256=DATASET,
            trusted_source_skin_registry=registry,
        )
    only_bias=torch.optim.AdamW(model.mira.codec.bias_head.parameters(),lr=1e-3)
    with pytest.raises(ValueError,match="CODEC_OPTIMIZER_SCOPE_DRIFT"):
        RSSvNextEmbeddedMIRASourceCodecTrainerV1(
            model,only_bias,dataset_manifest_sha256=DATASET,
            trusted_source_skin_registry=registry,
        )


def test_codec_first_rejects_unfrozen_or_stochastic_upstream():
    model,batch,registry=_fixture()
    upstream=model.axis.kind_head.weight
    upstream.requires_grad_(True)
    with pytest.raises(RuntimeError,match="UPSTREAM_FREEZE_DRIFT"):
        _trainer(model,registry)
    upstream.requires_grad_(False)
    model.iris.train()
    with pytest.raises(RuntimeError,match="UPSTREAM_TRAIN_MODE_DRIFT"):
        _trainer(model,registry)
    model.iris.eval()
    engine=_trainer(model,registry)
    model.axis.train()
    with pytest.raises(RuntimeError,match="UPSTREAM_TRAIN_MODE_DRIFT"):
        engine.train_step(batch)


def test_codec_first_catches_upstream_mutation_and_resume_identity(tmp_path):
    model,batch,registry=_fixture()
    engine=_trainer(model,registry)
    engine.train_step(batch)
    checkpoint=tmp_path/"stationary-mira.pt"
    engine.save_checkpoint(checkpoint)
    reference=_trainer(model,registry)
    reference.load_checkpoint(checkpoint)
    with torch.no_grad():
        model.axis.kind_head.weight.add_(0.01)
    with pytest.raises(RuntimeError,match="UPSTREAM_PARAMETER_MUTATED"):
        engine.train_step(batch)
    with pytest.raises(RuntimeError,match="UPSTREAM_PARAMETER_MUTATED"):
        engine.save_checkpoint(checkpoint)
    with pytest.raises(RuntimeError,match="UPSTREAM_PARAMETER_MUTATED"):
        reference.load_checkpoint(checkpoint)


def test_codec_first_rejects_optimizer_membership_drift_and_clears_stale_grad():
    model,batch,registry=_fixture()
    engine=_trainer(model,registry)
    frozen=model.axis.kind_head.weight
    frozen.grad=torch.ones_like(frozen)
    engine.train_step(batch)
    assert frozen.grad is None
    # Parameter group changes after trainer construction must not be accepted.
    engine.optimizer.param_groups[0]["params"].pop()
    with pytest.raises((RuntimeError,ValueError),match="OPTIMIZER"):
        engine.train_step(batch)


def test_spatial_fsq_source_training_updates_actual_point_decoder_and_token_encoder():
    model,batch,registry=_fixture("POINT_QUERY_CROSS_ATTENTION")
    engine=_trainer(model,registry)
    old=model.mira.spatial_decoder.pair_score[-1].weight.detach().clone()
    result=engine.train_step(batch)
    assert engine.step==1
    assert torch.isfinite(result.loss)
    assert result.hard_decoder_replay_max_abs_error<=1.e-5
    assert result.token_ids.shape==(1,2,model.cfg.skin_tokens_per_joint)
    assert result.source_qualified is False
    assert result.product_skin_qualified is False
    assert model.mira.spatial_decoder.pair_score[-1].weight.grad is not None
    assert model.mira.spatial_decoder.pair_score[-1].weight.grad.abs().sum()>0
    assert not torch.equal(model.mira.spatial_decoder.pair_score[-1].weight,old)
    assert model.mira.codec.latent_head.weight.grad is not None
    assert model.mira.codec.latent_head.weight.grad.abs().sum()>0
    assert model.mira.codec.bias_head.weight.grad is None  # unused dot decoder
    assert model.axis.kind_head.weight.grad is None
    assert model.iris.scene.scene_queries.grad is None


def test_spatial_source_checkpoint_replays_exact_decoder_and_cannot_cross_mode(tmp_path):
    model,batch,registry=_fixture("POINT_QUERY_CROSS_ATTENTION")
    trainer=_trainer(model,registry)
    trainer.train_step(batch)
    path=tmp_path/"source-spatial-fsq.pt"
    trainer.save_checkpoint(path)
    expected={
        k:v.detach().clone()
        for k,v in model.mira.spatial_decoder.state_dict().items()
    }
    resumed=_trainer(model,registry)
    resumed.load_checkpoint(path)
    assert resumed.step==1
    assert all(torch.equal(model.mira.spatial_decoder.state_dict()[k],v)
               for k,v in expected.items())
    # The source registry is mode-bound even when every observed weight is
    # the same. Dot-product source-teacher receipts are not interchangeable.
    wrong={"qualified-codec":{
        **registry["qualified-codec"],
        "skin_decoder_mode":"DOT_PRODUCT_BASELINE",
    }}
    with pytest.raises(ValueError,match="SOURCE_ADMISSION_DRIFT"):
        _trainer(model,wrong).objective(batch)
    tampered=replace(batch,source_dense_weights=batch.source_dense_weights.flip(-1))
    with pytest.raises(ValueError,match="SOURCE_ADMISSION_DRIFT"):
        resumed.objective(tampered)
    with torch.no_grad():
        model.mira.codec.bias_head.weight.add_(0.01)
    with pytest.raises(RuntimeError,match="UPSTREAM_PARAMETER_MUTATED"):
        resumed.train_step(batch)


def test_spatial_source_fsq_requires_complete_phase_optimizer_scope():
    model,batch,registry=_fixture("POINT_QUERY_CROSS_ATTENTION")
    only_codec=torch.optim.AdamW(model.mira.codec.parameters(),lr=1.e-3)
    with pytest.raises(ValueError,match="CODEC_OPTIMIZER_SCOPE_DRIFT"):
        RSSvNextEmbeddedMIRASourceCodecTrainerV1(
            model,only_codec,dataset_manifest_sha256=DATASET,
            trusted_source_skin_registry=registry,
        )
    engine=_trainer(model,registry)
    model.mira.spatial_decoder.eval()
    with pytest.raises(RuntimeError,match="UPSTREAM_TRAIN_MODE_DRIFT"):
        engine.train_step(batch)
