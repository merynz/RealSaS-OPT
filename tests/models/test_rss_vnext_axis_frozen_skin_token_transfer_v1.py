from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from models.axis.rig_program_model_v6 import RigTokenKindV6
from models.rss_vnext.axis_frozen_skin_token_transfer_v1 import (
    RSSvNextStationarySkinTokenBatchV1,
    RSSvNextStationaryAXISSkinTokenTrainerV1,
)
from models.rss_vnext.training_engine import (
    axis_mesh_teacher_address_digest_v1,
    axis_teacher_sequence_sha256_v6,
    frozen_mira_codec_state_sha256_v6,
    source_surface_tensor_sha256,
)
from tests.models.test_rss_vnext_mira_embedded_codec_source_training_v1 import (
    DATASET, _fixture,
)


UPSTREAM="c"*64


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(391)
    torch.set_num_threads(1)


def _case():
    model, source_batch, _ = _fixture()
    model.eval()
    model.requires_grad_(False)
    model.axis.skin_code_head.requires_grad_(True)
    scene=source_batch.scene
    topo=source_batch.topology_teacher_tokens
    memory=source_batch.structural_memory
    teacher=source_batch.axis_teacher_sequence
    with torch.no_grad():
        mesh=model.build_mesh_candidate(
            scene,topo[0],source_authority="TEACHER_FORCED_TRAINING_ONLY",
        )
        conditioned=model._control_memory(scene,memory,mesh)
        hidden=model.axis(conditioned,teacher)["hidden"]
        joints=torch.where(teacher.kind_ids[0] == int(RigTokenKindV6.JOINT))[0]
        skins=torch.where(teacher.kind_ids[0] == int(RigTokenKindV6.SKIN_CODE))[0]
        codes,_,_=model.mira.codec.encode_tokens(
            source_batch.source_dense_weights,
            torch.cat((mesh.local_features,mesh.rest_vertices_iris),dim=-1),
            hidden[:,joints-1],
        )
    assert codes.numel()==len(skins)
    new_codes=teacher.skin_code_ids.clone()
    new_codes[0,skins]=codes.reshape(-1)
    teacher=replace(teacher,skin_code_ids=new_codes)
    batch=RSSvNextStationarySkinTokenBatchV1(
        scene=scene,structural_memory=memory.detach(),
        topology_teacher_tokens=topo,
        axis_teacher_sequence=teacher,
        source_dense_weights=source_batch.source_dense_weights,
        source_binding_sha256=source_batch.source_binding_sha256,
        source_qualification_key=source_batch.source_qualification_key,
        admission_key="source-skin-bridge",
    )
    registry={"source-skin-bridge":{
        "admission_status":"QUALIFIED_NATIVE_SOURCE_FROZEN_AXIS_MIRA_SKIN_CODE_HANDOFF",
        "dataset_manifest_sha256":DATASET,
        "source_qualification_key":batch.source_qualification_key,
        "source_binding_sha256":batch.source_binding_sha256,
        "upstream_frozen_checkpoint_sha256":UPSTREAM,
        "mira_codec_state_sha256":frozen_mira_codec_state_sha256_v6(model),
        "scene_generator_binding_sha256":scene.generating_checkpoint_sha256,
        "mesh_teacher_tokens_sha256":source_surface_tensor_sha256(topo),
        "axis_teacher_sequence_sha256":axis_teacher_sequence_sha256_v6(teacher),
        "source_dense_weights_sha256":source_surface_tensor_sha256(
            batch.source_dense_weights,
        ),
        "structural_memory_sha256":source_surface_tensor_sha256(batch.structural_memory),
        "mesh_teacher_address_digest":axis_mesh_teacher_address_digest_v1(mesh),
        "source_mesh_face_reindex_proven":True,
        "source_rig_joint_reindex_proven":True,
        "source_codec_hard_reconstruction_qualified":True,
        "generated_view_is_skin_teacher":False,
        "teacher_semantics":"EXACT_FROZEN_FSQ_SOURCE_SKIN_TOKENS",
        "max_admitted_source_codec_weight_mae":1.,
    }}
    return model,batch,registry


def _trainer(model,registry):
    return RSSvNextStationaryAXISSkinTokenTrainerV1(
        model,
        torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),lr=1.e-3,
        ),
        dataset_manifest_sha256=DATASET,
        upstream_frozen_checkpoint_sha256=UPSTREAM,
        trusted_source_bridge_registry=registry,
    )


def test_skin_teacher_is_exact_frozen_codec_and_only_axis_skin_head_can_train():
    model,batch,registry=_case()
    trainer=_trainer(model,registry)
    frozen=model.iris.scene.scene_queries.detach().clone()
    frozen_codec=model.mira.codec.bias_head.weight.detach().clone()
    frozen_embedding=model.axis.skin_embedding.weight.detach().clone()
    before=model.axis.skin_code_head.weight.detach().clone()
    first=trainer.train_step(batch)
    assert first.target_code_count==(
        model.cfg.skin_tokens_per_joint*2
    )
    assert torch.isfinite(first.skin_code_ce)
    assert first.target_code_accuracy>=0.
    assert first.hard_codec_runtime_replay_max_error<=1e-5
    assert first.independent_source_codec_weight_mae<=1.
    assert not first.trained_skin_program_release_qualified
    assert model.axis.skin_code_head.weight.grad is not None
    assert model.axis.skin_code_head.weight.grad.abs().sum()>0
    assert model.iris.scene.scene_queries.grad is None
    assert model.mira.codec.bias_head.weight.grad is None
    assert torch.equal(model.iris.scene.scene_queries,frozen)
    assert torch.equal(model.mira.codec.bias_head.weight,frozen_codec)
    assert torch.equal(model.axis.skin_embedding.weight,frozen_embedding)
    assert model.axis.skin_embedding.weight.grad is None
    assert not torch.equal(model.axis.skin_code_head.weight,before)
    # Token labels must stay stationary even after AXIS learns its SkinTokens
    # prediction head; even the null/padding embedding is frozen.
    second=trainer.objective(batch)
    assert second.target_code_count==first.target_code_count


def test_frozen_conditioning_cannot_be_silently_unfrozen_or_swapped():
    model,batch,registry=_case()
    model.iris.scene.scene_queries.requires_grad_(True)
    with pytest.raises(ValueError,match="UPSTREAM_MUST_BE_FROZEN"):
        _trainer(model,registry)
    model.iris.scene.scene_queries.requires_grad_(False)
    model.axis.skin_embedding.requires_grad_(True)
    with pytest.raises(ValueError,match="UPSTREAM_MUST_BE_FROZEN"):
        _trainer(model,registry)
    model.axis.skin_embedding.requires_grad_(False)
    trainer=_trainer(model,registry)
    model.axis.train()
    with pytest.raises(ValueError,match="UPSTREAM_MUST_BE_FROZEN"):
        trainer.objective(batch)
    model.axis.eval()
    model.mira.codec.bias_head.weight.data.add_(.02)
    with pytest.raises(RuntimeError,match="FSQ_CODEC_STATE_DRIFT"):
        trainer.objective(batch)


def test_same_shape_forged_skin_ids_not_accepted_as_frozen_codec_teacher():
    model,batch,registry=_case()
    altered=batch.axis_teacher_sequence.skin_code_ids.clone()
    target=torch.where(
        batch.axis_teacher_sequence.kind_ids[0] == int(RigTokenKindV6.SKIN_CODE)
    )[0][0]
    altered[0,target]=(altered[0,target]+1)%model.axis.config.skin_vocab_size
    forged=replace(batch.axis_teacher_sequence,skin_code_ids=altered)
    other=replace(batch,axis_teacher_sequence=forged)
    # Simulate a malicious preparation job resigning the new sequence bytes.
    signed={"source-skin-bridge":{
        **registry["source-skin-bridge"],
        "axis_teacher_sequence_sha256":axis_teacher_sequence_sha256_v6(forged),
    }}
    with pytest.raises(ValueError,match="LABELS_NOT_FROZEN_CODEC_CODES"):
        _trainer(model,signed).objective(other)


def test_missing_native_weights_and_mismatched_vertex_address_fail_closed():
    model,batch,registry=_case()
    bad=batch.source_dense_weights.clone()
    bad[0,0]=torch.tensor([.6,.6])
    signed={"source-skin-bridge":{
        **registry["source-skin-bridge"],
        "source_dense_weights_sha256":source_surface_tensor_sha256(bad),
    }}
    with pytest.raises(ValueError,match="SOURCE_WEIGHTS_DOMAIN"):
        _trainer(model,signed).objective(replace(batch,source_dense_weights=bad))
    forged_topology=batch.topology_teacher_tokens.clone()
    forged_topology[0,-2]^=1
    with pytest.raises(ValueError,match="SOURCE_RECEIPT_DRIFT"):
        _trainer(model,registry).objective(
            replace(batch,topology_teacher_tokens=forged_topology),
        )
    with pytest.raises(ValueError,match="SOURCE_RECEIPT_DRIFT"):
        _trainer(model,registry).objective(
            replace(batch,source_binding_sha256="e"*64),
        )


def test_source_codec_reconstruction_threshold_is_independent_gate():
    model,batch,registry=_case()
    policy={"source-skin-bridge":{
        **registry["source-skin-bridge"],
        "max_admitted_source_codec_weight_mae":0.,
    }}
    with pytest.raises(ValueError,match="CODEC_QUALITY_NOT_ADMITTED"):
        _trainer(model,policy).objective(batch)


def test_token_handoff_checkpoint_pins_architecture_codec_registry_and_source(tmp_path):
    model,batch,registry=_case()
    trainer=_trainer(model,registry)
    trainer.train_step(batch)
    target=tmp_path/"axis-skin-stage.pt"
    trainer.save_checkpoint(target)
    replay=_trainer(model,registry)
    replay.load_checkpoint(target)
    assert replay.step==1
    changed={"source-skin-bridge":{
        **registry["source-skin-bridge"],
        "source_codec_hard_reconstruction_qualified":False,
    }}
    with pytest.raises(ValueError,match="RESUME_POLICY_DRIFT"):
        _trainer(model,changed).load_checkpoint(target)
    shifted=RSSvNextStationaryAXISSkinTokenTrainerV1(
        model,torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),lr=1e-3,
        ),
        dataset_manifest_sha256=DATASET,
        upstream_frozen_checkpoint_sha256="d"*64,
        trusted_source_bridge_registry=registry,
    )
    with pytest.raises(ValueError,match="RESUME_POLICY_DRIFT"):
        shifted.load_checkpoint(target)
