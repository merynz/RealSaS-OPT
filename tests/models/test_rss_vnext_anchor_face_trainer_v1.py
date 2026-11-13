from __future__ import annotations

"""Source-receipted unified optimizer for causal AXIS -> TESSA face ownership.

Test fixtures are explicitly synthetic verification inputs. Only an independent
source foundry can mint real native attachment face reindex receipts.
"""
from dataclasses import replace
import pytest
import torch

from models.axis.rig_program_model_v6 import (
    AXISSequenceV6,RigProgramEventV6,RigTokenKindV6,
    AXIS_V6_PRODUCT_ARCHITECTURE_ID,
)
from models.rss_vnext.attachment_face_correspondence_v1 import (
    axis_sequence_digest_v1,mesh_topology_digest_v1,_digest_tensor,
)
from models.rss_vnext.training_engine import (
    RSSvNextStructuralTrainer,RSSvNextStructuralTrainingBatch,
    RSSvNextTrainingWeights,
    axis_teacher_sequence_sha256_v6,frozen_mira_codec_state_sha256_v6,
    source_surface_tensor_sha256,
)
from tests.models.test_rss_vnext_attachment_face_correspondence_v1 import _tokens
from tests.models.test_rss_vnext_inference_checkpoint_bridge_v1 import _network,_images


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(538)
    torch.set_num_threads(1)


def _qualified_axis_sequence(net):
    kinds=[
        RigProgramEventV6(RigTokenKindV6.BOS),
        RigProgramEventV6(RigTokenKindV6.JOINT,coord_indices=(11,22,33)),
        RigProgramEventV6(RigTokenKindV6.FRAMES_START),
        RigProgramEventV6(RigTokenKindV6.FRAME,pointer_a=0,
                          frame_6d=(1.,0.,0.,0.,1.,0.),
                          dof_mask=(0.,0.,0.,1.,1.,1.,0.,0.,0.)),
        RigProgramEventV6(RigTokenKindV6.CONTROLS_START),
        RigProgramEventV6(RigTokenKindV6.CONTROL,aux_type_id=7,
                          coord_indices=(44,55,66),pointer_a=-1),
        RigProgramEventV6(RigTokenKindV6.CONTROL_TARGET,pointer_a=0),
        RigProgramEventV6(RigTokenKindV6.CHAINS_START),
        RigProgramEventV6(RigTokenKindV6.CONSTRAINTS_START),
        RigProgramEventV6(RigTokenKindV6.ANCHORS_START),
        RigProgramEventV6(RigTokenKindV6.ANCHOR,coord_indices=(77,88,99),
                          anchor_parent_kind=1,pointer_b=0),
        RigProgramEventV6(RigTokenKindV6.SKIN_START),
    ]
    kinds.extend(RigProgramEventV6(RigTokenKindV6.SKIN_CODE,skin_code_id=i)
                 for i in range(net.cfg.skin_tokens_per_joint))
    kinds.append(RigProgramEventV6(RigTokenKindV6.EOS))
    return AXISSequenceV6.from_events(kinds,device=torch.device("cpu"))


def _training_court():
    net=_network()
    net.train()
    net.mira.codec.requires_grad_(False)
    rgba,cam=_images()
    topo=_tokens(net)[None]
    teacher=_qualified_axis_sequence(net)
    # Only model-owned views/features are admitted to the forward graph.
    with torch.no_grad():
        state,aux,_=net.encode_observations(rgba,cam)
        memory=net._structural_memory(state,aux)
        mesh=net.build_mesh_candidate(
            state,topo[0],source_authority="TEACHER_FORCED_TRAINING_ONLY",
        )
        axis=net.axis(net._control_memory(state,memory,mesh),teacher)
        candidate=net.anchor_face_correspondence(
            mesh=mesh,axis_hidden=axis["hidden"],axis_sequence=teacher,
            region_relation_logits=state.region_relation_logits,
        )
    assert candidate.anchor_event_indices==(10,)
    labels=torch.tensor([[0,1]],dtype=torch.long)
    mask=torch.tensor([[True,True]])
    receipt={
        "admission_status":"QUALIFIED_SOURCE_NATIVE_ATTACHMENT_FACE_REINDEX",
        "source_semantics":"NATIVE_RIG_ATTACHMENT_HELD_HOLDER_FACE_OR_VERIFIED_NO_FACE",
        "source_asset_sha256":"a"*64,
        "dataset_manifest_sha256":"3"*64,
        "source_qualification_key":"synthetic-unit-only",
        "observed_rgba_sha256":source_surface_tensor_sha256(rgba),
        "observed_camera_sha256":source_surface_tensor_sha256(cam),
        "tessa_mesh_topology_sha256":candidate.mesh_topology_digest,
        "axis_teacher_sequence_sha256":axis_sequence_digest_v1(teacher),
        "source_face_targets_sha256":_digest_tensor(labels),
        "source_face_mask_sha256":_digest_tensor(mask),
        "source_reindex_proven":True,
    }
    axis_receipt={
        "admission_status":"QUALIFIED_SOURCE_RIG_SKIN_TOKEN_TEACHER",
        "dataset_manifest_sha256":"3"*64,
        "source_qualification_key":"synthetic-unit-only",
        "architecture_id":AXIS_V6_PRODUCT_ARCHITECTURE_ID,
        "teacher_sequence_sha256":axis_teacher_sequence_sha256_v6(teacher),
        "mira_codec_state_sha256":frozen_mira_codec_state_sha256_v6(net),
        "teacher_semantics":"SOURCE_RIG_AND_PINNED_FSQ_NOT_INFERRED_CONTROLS",
        "mesh_teacher_semantics":"SOURCE_NATIVE_TESSA_TEACHER_TOPOLOGY_FOR_AXIS_PRODUCT",
        "mesh_teacher_source_reindex_proven":True,
        "mesh_teacher_tokens_sha256":source_surface_tensor_sha256(topo),
        "mesh_teacher_valid_mask_sha256":source_surface_tensor_sha256(torch.tensor([True])),
        "mesh_teacher_topology_digest":mesh_topology_digest_v1(mesh),
        "observed_rgba_sha256":source_surface_tensor_sha256(rgba),
        "observed_camera_sha256":source_surface_tensor_sha256(cam),
        "intervention_qualification_status":"QUALIFIED_SOURCE_CONTROL_CONSEQUENCE",
        "source_intervention_sha256":"9"*64,
    }
    weights=RSSvNextTrainingWeights(
        topology=0.,motion_preference=0.,source_semantics=0.,
        flexicubes_surface=0.,axis_rig_skin_codes=1.,
        anchor_face_correspondence=1.,
    )
    def new_trainer(
        anchor_row=receipt,axis_row=axis_receipt,
        axis_policy="PRODUCT_MESH_MATCHED_REQUIRED",
    ):
        return RSSvNextStructuralTrainer(
            net,torch.optim.AdamW(net.parameters(),lr=1.e-3),
            dataset_manifest_sha256="3"*64,weights=weights,
            axis_training_policy=axis_policy,
            trusted_axis_teacher_registry={"native-axis":axis_row},
            trusted_anchor_face_registry={"native-attachment":anchor_row},
        )
    batch=RSSvNextStructuralTrainingBatch(
        rgba=rgba,camera_features=cam,
        topology_teacher_tokens=topo,
        topology_valid_mask=torch.tensor([True]),
        axis_teacher_sequence=teacher,
        axis_teacher_admission_key="native-axis",
        source_anchor_face_targets=labels,
        source_anchor_face_mask=mask,
        source_anchor_face_admission_key="native-attachment",
        source_qualification_key="synthetic-unit-only",
        admission_status="QUALIFIED_FOR_THIS_SUPERVISION_FAMILY",
    )
    return net,new_trainer,batch,receipt,axis_receipt


def test_optimizer_jointly_updates_native_face_head_and_shared_iris_axis(monkeypatch):
    net,make,batch,receipt,axis= _training_court()
    trainer=make()
    before=net.anchor_face_correspondence.held_query.weight.detach().clone()
    seen_mesh=[]
    seen_mem=[]
    original_mesh=net.build_mesh_candidate
    original_axis=net.axis.forward
    def watch_mesh(*args,**kwargs):
        row=original_mesh(*args,**kwargs)
        seen_mesh.append(row)
        return row
    def watch_axis(scene_memory,sequence):
        seen_mem.append(scene_memory)
        return original_axis(scene_memory,sequence)
    monkeypatch.setattr(net,"build_mesh_candidate",watch_mesh)
    monkeypatch.setattr(net.axis,"forward",watch_axis)
    result=trainer.train_step(batch)
    assert len(seen_mesh)==1
    assert len(seen_mem)==2
    # Both causal loss and held/holder head are conditioned on THE SAME
    # scene+TESSA mesh memory tensor, not two independent mesh decodings.
    assert seen_mem[0] is seen_mem[1]
    assert result.anchor_face_correspondence_loss is not None
    assert result.eligible_anchor_held_faces==1
    assert result.eligible_anchor_holder_faces==1
    assert result.eligible_anchor_negative_faces==0
    assert result.axis_rig_skin_loss is not None
    assert result.axis_training_conditioning=="PRODUCT_MESH_MATCHED_TESSA_H"
    assert torch.isfinite(result.total_loss)
    assert trainer.global_step==1
    assert not torch.equal(before,net.anchor_face_correspondence.held_query.weight)
    for grad in (
        net.anchor_face_correspondence.held_query.weight.grad,
        net.anchor_face_correspondence.holder_query.weight.grad,
        net.axis.scene_cross_attention.in_proj_weight.grad,
        net.iris.scene.scene_queries.grad,
    ):
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert grad.abs().sum()>0


def test_untrusted_target_or_source_reindex_is_rejected_before_training():
    net,make,batch,receipt,axis=_training_court()
    trainer=make()
    with pytest.raises(ValueError,match="SOURCE_RECEIPT_DRIFT"):
        trainer.objective(replace(
            batch,source_anchor_face_targets=torch.tensor([[1,0]]),
        ))
    corrupted={**receipt,"source_reindex_proven":False}
    with pytest.raises(ValueError,match="SOURCE_RECEIPT_DRIFT"):
        make(anchor_row=corrupted).objective(batch)
    corrupted={**receipt,"observed_rgba_sha256":"f"*64}
    with pytest.raises(ValueError,match="SOURCE_OR_AXIS_TEACHER_RECEIPT_DRIFT"):
        make(anchor_row=corrupted).objective(batch)


def test_teacher_sequence_and_topology_must_be_bound_without_silent_repair():
    net,make,batch,receipt,axis=_training_court()
    trainer=make()
    with pytest.raises(ValueError,match="SINGLE_TEACHER_MESH_REQUIRED"):
        trainer.objective(replace(batch,topology_valid_mask=torch.tensor([False])))
    with pytest.raises(ValueError,match="TEACHER_INCOMPLETE"):
        trainer.objective(replace(batch,topology_teacher_tokens=None))
    altered=batch.axis_teacher_sequence.coord_indices.clone()
    altered[0,10,0]+=1
    # Existing AXIS token/codec gate rejects tampering even earlier than
    # the new face-correspondence gate; both must remain fail-closed.
    with pytest.raises(ValueError,match="AXIS_TEACHER_SOURCE_CODEC_OR_SEQUENCE_DRIFT"):
        trainer.objective(replace(
            batch,axis_teacher_sequence=replace(
                batch.axis_teacher_sequence,coord_indices=altered,
            ),
        ))


def test_resume_is_bound_to_native_anchor_face_admission_registry(tmp_path):
    net,make,batch,receipt,axis=_training_court()
    trainer=make()
    trainer.objective(batch)
    checkpoint=tmp_path/"source_native_anchor.pt"
    trainer.save_checkpoint(checkpoint,sampler_cursor=19)
    assert make().load_checkpoint(checkpoint)==19
    changed={**receipt,"source_face_mask_sha256":"c"*64}
    with pytest.raises(ValueError,match="CHECKPOINT_ANCHOR_FACE_REGISTRY_DRIFT"):
        make(anchor_row=changed).load_checkpoint(checkpoint)


def test_missing_source_labels_and_disabled_objective_do_not_train_negative():
    net,make,batch,receipt,axis=_training_court()
    trainer=make()
    with pytest.raises(ValueError,match="TEACHER_INCOMPLETE"):
        trainer.objective(replace(batch,source_anchor_face_mask=None))
    disabled=RSSvNextStructuralTrainer(
        net,torch.optim.AdamW(net.parameters(),lr=1.e-3),
        dataset_manifest_sha256="3"*64,
        weights=RSSvNextTrainingWeights(
            topology=0,motion_preference=0,source_semantics=0,
            flexicubes_surface=0,axis_rig_skin_codes=1,
            anchor_face_correspondence=0,
        ),
        trusted_axis_teacher_registry={"native-axis":axis},
        trusted_anchor_face_registry={"native-attachment":receipt},
    )
    with pytest.raises(ValueError,match="SOCKET_FACE_TRAIN_WEIGHT_REQUIRED"):
        disabled.objective(batch)


def test_product_axis_without_source_admitted_tessa_mesh_fails_closed():
    net,make,batch,receipt,axis=_training_court()
    # No source topology: cannot train a PRODUCT rig conditioned on a mesh
    # while its forward uses only the unindexed structural stream.
    with pytest.raises(ValueError,match="AXIS_PRODUCT_MATCHED_TESSA_MESH_REQUIRED"):
        make().objective(replace(
            batch,topology_teacher_tokens=None,
            source_anchor_face_targets=None,source_anchor_face_mask=None,
            source_anchor_face_admission_key="",
        ))
    bad_receipt={**axis,"mesh_teacher_source_reindex_proven":False}
    with pytest.raises(ValueError,match="AXIS_PRODUCT_TESSA_SOURCE_BINDING_DRIFT"):
        make(axis_row=bad_receipt).objective(batch)
    bad_receipt={**axis,"mesh_teacher_topology_digest":"9"*64}
    with pytest.raises(ValueError,match="AXIS_PRODUCT_TESSA_FACE_REINDEX_DRIFT"):
        make(axis_row=bad_receipt).objective(batch)


def test_pretrain_cannot_upgrade_unmatched_axis_socket_anchor_or_train_source_face():
    net,make,batch,receipt,axis=_training_court()
    # Even without native anchor-face labels, a rig-only lane may not
    # expose the source ANCHOR and then claim product AXIS conditioning.
    no_face=replace(
        batch,source_anchor_face_targets=None,source_anchor_face_mask=None,
        source_anchor_face_admission_key="",
    )
    with pytest.raises(ValueError,match="AXIS_RIG_ONLY_ANCHOR_REQUIRES_MESH"):
        make(axis_policy="RIG_ONLY_PRETRAIN").objective(no_face)
    with pytest.raises(ValueError,match="AXIS_RIG_ONLY_ANCHOR_REQUIRES_MESH"):
        make(axis_policy="RIG_ONLY_PRETRAIN").objective(batch)


def test_product_pretrain_policy_cannot_be_swapped_during_checkpoint_resume(tmp_path):
    net,make,batch,receipt,axis=_training_court()
    valid=make()
    result=valid.objective(batch)
    assert result.axis_training_conditioning=="PRODUCT_MESH_MATCHED_TESSA_H"
    out=tmp_path/"matched-product.pt"
    valid.save_checkpoint(out,sampler_cursor=31)
    with pytest.raises(ValueError,match="CHECKPOINT_OBJECTIVE_POLICY_DRIFT"):
        make(axis_policy="RIG_ONLY_PRETRAIN").load_checkpoint(out)
    assert make().load_checkpoint(out)==31
    with pytest.raises(ValueError,match="CONDITIONING_POLICY_UNKNOWN"):
        make(axis_policy="PERMISSIVE_MESH_NEARBY")


def test_product_axis_uses_actual_teacher_mesh_not_fake_same_signature():
    net,make,batch,receipt,axis=_training_court()
    # Source teacher mesh face IDs are bound to exact token tensor bytes.
    corrupted=batch.topology_teacher_tokens.clone()
    corrupted[0,1]+=1
    with pytest.raises(ValueError,match="AXIS_PRODUCT_TESSA_SOURCE_BINDING_DRIFT"):
        make().objective(replace(batch,topology_teacher_tokens=corrupted))
