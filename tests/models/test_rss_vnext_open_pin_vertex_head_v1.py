from __future__ import annotations

"""Tiny synthetic OPEN M* pin-source tests. Not model release/cloth proof."""
from dataclasses import replace

import pytest
import torch

from models.rss_vnext.product_open_pin_vertex_head_v1 import (
    OpenPinVertexNeuralOutputV1,
    SourceNativeOpenPinVertexTargetV1,
    TEACHER_SCHEMA,open_hem_boundary_mask_v1,
    propose_open_pin_vertex_indices_v1,
    source_native_open_pin_vertex_loss_v1,
    _digest_tensor,
)
from tests.models.test_rss_vnext_attachment_face_correspondence_v1 import _forward


@pytest.fixture(autouse=True)
def set_cpu():
    torch.manual_seed(20261120)
    torch.set_num_threads(1)


def _case():
    model,result,seq,rgba,camera=_forward()
    mesh=result.mesh_candidate
    memory=model._control_memory(
        result.state,result.structural_memory,mesh,
    )
    controls=model.axis(memory,seq)
    out=model.open_pin_vertex(
        mesh=mesh,axis_hidden=controls["hidden"],axis_sequence=seq,
    )
    assert isinstance(out,OpenPinVertexNeuralOutputV1)
    assert out.pin_logits.shape==(1,6)
    assert out.anchor_event_indices==(8,)
    assert out.open_boundary_mask.tolist()==[True]*6
    target=torch.zeros((1,6),dtype=torch.bool)
    target[0,0]=True
    target[0,1]=True
    known=torch.zeros_like(target)
    known[0,:3]=True
    teacher=SourceNativeOpenPinVertexTargetV1(
        source_asset_sha256="a"*64,
        source_qualification_key="source-native-open-sheet",
        native_open_hem_sha256="1"*64,
        native_pin_witness_sha256="2"*64,
        heldout_host_only_intervention_sha256="3"*64,
        source_mstar_transport_sha256="4"*64,
        pin_targets=target,known_mask=known,
    )
    source_faces=torch.tensor([[0,1]],dtype=torch.long)
    source_face_mask=torch.ones((1,2),dtype=torch.bool)
    receipt={
        "admission_status":"QUALIFIED_SOURCE_NATIVE_OPEN_PIN_VERTEX_REINDEX",
        "schema":TEACHER_SCHEMA,
        "dataset_manifest_sha256":"f"*64,
        "source_asset_sha256":"a"*64,
        "source_qualification_key":"source-native-open-sheet",
        "original_rgba_sha256":"5"*64,
        "original_camera_sha256":"6"*64,
        "combined_generation_checkpoint_sha256":"7"*64,
        "candidate_mstar_topology_address_sha256":"8"*64,
        "mesh_topology_digest":out.mesh_topology_digest,
        "axis_sequence_digest":out.axis_sequence_digest,
        "source_anchor_face_targets_sha256":_digest_tensor(source_faces),
        "source_anchor_face_mask_sha256":_digest_tensor(source_face_mask),
        "source_held_holder_are_distinct_components":True,
        "native_open_hem_sha256":teacher.native_open_hem_sha256,
        "native_pin_witness_sha256":teacher.native_pin_witness_sha256,
        "heldout_host_only_intervention_sha256":
            teacher.heldout_host_only_intervention_sha256,
        "source_mstar_transport_sha256":teacher.source_mstar_transport_sha256,
        "source_pin_targets_sha256":_digest_tensor(target),
        "source_known_mask_sha256":_digest_tensor(known),
        "source_pin_support_from_independent_native_boundary":True,
        "source_native_host_intervention_verified":True,
        "source_to_final_mstar_vertex_transport_verified":True,
        "student_nearest_vertex_as_pin_truth":False,
        "generated_views_as_pin_truth":False,
        "physical_material_or_xpbd_authority":False,
    }
    ledger={"native-open-pin":receipt}
    def train(*, labels=teacher,output=out, registry=ledger,
              faces=source_faces,face_mask=source_face_mask):
        return source_native_open_pin_vertex_loss_v1(
            output=output,teacher=labels,
            admission_key="native-open-pin",
            trusted_registry=registry,
            dataset_manifest_sha256="f"*64,
            candidate_mstar_topology_address_sha256="8"*64,
            source_asset_sha256="a"*64,
            source_qualification_key="source-native-open-sheet",
            source_anchor_face_targets=faces,
            source_anchor_face_mask=face_mask,
            original_rgba_sha256="5"*64,
            original_camera_sha256="6"*64,
            combined_generation_checkpoint_sha256="7"*64,
        )
    return model,result,seq,mesh,out,teacher,ledger,train


def test_open_hem_vertex_head_has_true_indexed_candidates_and_gradients():
    model,result,seq,mesh,out,teacher,ledger,train=_case()
    loss,n=train()
    assert n==3 and loss.requires_grad
    assert torch.isfinite(loss)
    result.state.scene_triplanes.retain_grad()
    loss.backward()
    assert model.open_pin_vertex.query.weight.grad is not None
    assert model.open_pin_vertex.key.weight.grad is not None
    assert model.open_pin_vertex.query.weight.grad.abs().sum()>0
    assert result.state.scene_triplanes.grad is not None
    assert result.state.scene_triplanes.grad.abs().sum()>0
    assert out.source_pin_truth_qualified is False
    assert out.automatic_physics_activation is False


def test_native_source_pin_receipt_tamper_and_unknown_labels_rejected():
    _,_,_,mesh,out,teacher,ledger,train=_case()
    bad={"native-open-pin":dict(ledger["native-open-pin"])}
    bad["native-open-pin"]["source_native_host_intervention_verified"]=False
    with pytest.raises(ValueError,match="RECEIPT_DRIFT"):
        train(registry=bad)
    with pytest.raises(ValueError,match="RECEIPT_DRIFT"):
        train(labels=replace(
            teacher,pin_targets=teacher.pin_targets.roll(1,dims=1),
        ))
    with pytest.raises(ValueError,match="TEACHER_DOMAIN"):
        train(labels=replace(
            teacher,pin_targets=torch.ones_like(teacher.pin_targets),
        ))
    unknown=teacher.known_mask.clone()
    unknown[0,0]=False
    with pytest.raises(ValueError,match="TEACHER_DOMAIN"):
        train(labels=replace(teacher,known_mask=unknown))
    assert source_native_open_pin_vertex_loss_v1(
        output=None,teacher=None,admission_key="",trusted_registry={},
        dataset_manifest_sha256="f"*64,
        candidate_mstar_topology_address_sha256="8"*64,
        source_asset_sha256="a"*64,source_qualification_key="x",
        source_anchor_face_targets=None,source_anchor_face_mask=None,
        original_rgba_sha256="5"*64,original_camera_sha256="6"*64,
        combined_generation_checkpoint_sha256="7"*64,
    )==(None,0)


def test_editor_pin_proposal_is_never_physics_approval():
    _,_,_,mesh,out,teacher,ledger,train=_case()
    logits=torch.full_like(out.pin_logits,-8.)
    logits[0,0]=10.
    logits[0,1]=10.
    fake=replace(out,pin_logits=logits)
    p=propose_open_pin_vertex_indices_v1(fake,min_probability=.9)
    assert len(p)==1
    assert p[0].candidate_pin_vertex_indices==(0,1)
    assert p[0].state=="UNQUALIFIED_REVIEW_PIN_VERTICES"
    assert not p[0].source_witness_claimed
    assert not p[0].executable_xpbd_claimed
    with pytest.raises(ValueError,match="PREVIEW_DOMAIN"):
        propose_open_pin_vertex_indices_v1(
            replace(fake,automatic_physics_activation=True)
        )
    assert propose_open_pin_vertex_indices_v1(
        replace(fake,pin_logits=torch.full_like(logits,-8.))
    )[0].state=="ABSTAIN_NO_CONFIDENT_OPEN_PIN_SET"


def test_open_hem_boundary_excludes_manifold_interior_and_rejects_mixed_component():
    model,result,seq,mesh,out,teacher,ledger,train=_case()
    # A square split by one interior diagonal has its corner vertices on
    # the hem; a central joined vertex in a triangle fan is truly interior.
    from dataclasses import replace
    v=mesh.rest_vertices_iris.shape[1]
    # Existing fixture has six vertices in two independent OPEN triangles;
    # make a standalone closed tetra from first 4 for a negative case.
    tetra=torch.tensor([
        [0,1,2],[0,3,1],[0,2,3],[1,3,2],
    ],dtype=torch.long)
    closed=replace(
        mesh,faces=tetra,face_component_ids=torch.zeros(4,dtype=torch.long),
        vertex_component_ids=torch.zeros(v,dtype=torch.long),
    )
    assert not bool(open_hem_boundary_mask_v1(closed).any())
    mixed=replace(mesh,face_component_ids=torch.ones_like(mesh.face_component_ids))
    with pytest.raises(ValueError,match="CROSS_COMPONENT_FACE"):
        open_hem_boundary_mask_v1(mixed)


def test_ninth_pin_support_family_matches_exact_weighted_total():
    from types import SimpleNamespace
    from models.rss_vnext.product_multitask_gradient_control_v1 import (
        admitted_product_task_losses_v1,
    )
    x=torch.tensor([.2,-.3],requires_grad=True)
    geometry=x.square().sum()
    pin=(x[0]-.4).square()
    result=SimpleNamespace(
        total_loss=geometry+.7*pin,seeded_tessa_loss=geometry,
        component_udf_loss=None,product_semantic_loss=None,
        graphics_source_rgb_loss=None,realsas_joint_slrm_loss=None,
        product_mstar_pose_raster_loss=None,
        product_native_layer_clearance_loss=None,
        product_native_control_intervention_loss=None,
        product_open_pin_vertex_loss=pin,
        product_axis_mira_loss=None,anchor_face_correspondence_loss=None,
    )
    parts=admitted_product_task_losses_v1(
        result,component_udf_weight=0.,graphics_rgb_weight=0.,
        graphics_joint_ray_weight=0.,anchor_face_weight=0.,
        open_pin_vertex_weight=.7,
    )
    assert set(parts)=={"GEOMETRY","OPEN_PIN_SUPPORT"}
    torch.testing.assert_close(sum(parts.values()),result.total_loss)


def test_real_seeded_pin_head_policy_never_invents_labels_and_is_checkpoint_pinned(
    monkeypatch,tmp_path,
):
    from models.rss_vnext.product_axis_mira_seeded_training_v1 import (
        ProductSeededAXISMIRAWeightsV1,
    )
    from tests.models.test_rss_vnext_product_seeded_axis_mira_training_v1 import (
        _rig_case,
    )
    net,batch,make,*_=_rig_case(monkeypatch)
    # The original fixture is one CLOSED tetra with no hem. A source pin
    # label must NOT be manufactured, but a sparse-source model should
    # still train its unrelated geometry/rig losses.
    teacher=ProductSeededAXISMIRAWeightsV1(axis_program=1.)
    anchor_registry={"known-source-only":{"admission_status":"NOT_FOR_THIS_ASSET"}}
    pin_registry={"known-source-only":{"admission_status":"NOT_FOR_THIS_ASSET"}}
    trainer=make(
        weights=teacher,sparse=True,anchor_weight=1.,
        anchor_registry=anchor_registry,
        pin_weight=.5,pin_registry=pin_registry,
    )
    req=trainer.source_supervision_requirements_v1()
    assert "source_native_open_hem_pin_vertices" in req["loss_required_source_families"]
    assert req["model_release_ready"] is False
    result=trainer.train_step(batch)
    assert result.product_open_pin_vertex_loss is None
    assert result.eligible_open_pin_vertex_labels==0
    path=tmp_path/"seeded-open-pin-source.pt"
    trainer.save_checkpoint(path,sampler_cursor=3)
    other=make(
        weights=teacher,sparse=True,anchor_weight=1.,
        anchor_registry=anchor_registry,
        pin_weight=.5,pin_registry=pin_registry,
    )
    assert other.load_checkpoint(path)==3
    with pytest.raises(ValueError,match="OBJECTIVE_POLICY_DRIFT"):
        make(weights=teacher,sparse=True,anchor_weight=1.,
             anchor_registry=anchor_registry,
             pin_weight=.7,pin_registry=pin_registry).load_checkpoint(path)
    with pytest.raises(ValueError,match="OPEN_PIN_VERTEX_PHASE_POLICY"):
        make(weights=teacher,sparse=True,anchor_weight=1.,
             anchor_registry=anchor_registry,
             pin_weight=.5).objective(batch)


def test_native_open_pin_held_component_cannot_change_without_source_reindex():
    _,_,_,mesh,out,teacher,ledger,train=_case()
    with pytest.raises(ValueError,match="HELD_COMPONENT_DRIFT"):
        train(faces=torch.tensor([[1,0]],dtype=torch.long))
    with pytest.raises(ValueError,match="HELD_HOLDER_REQUIRED"):
        train(face_mask=torch.tensor([[True,False]],dtype=torch.bool))
    # Even a fabricated registry with self-consistent source face hashes
    # cannot promote a student OPEN garment vertex belonging to the holder.
    swapped=torch.tensor([[1,0]],dtype=torch.long)
    forged={"native-open-pin":dict(ledger["native-open-pin"])}
    forged["native-open-pin"]["source_anchor_face_targets_sha256"]=_digest_tensor(swapped)
    with pytest.raises(ValueError,match="HELD_COMPONENT_DRIFT"):
        train(faces=swapped,registry=forged)
