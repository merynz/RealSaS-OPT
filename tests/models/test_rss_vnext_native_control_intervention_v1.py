from __future__ import annotations

"""Synthetic control-response ABI regressions, never product rig/cloth proof."""
from dataclasses import replace
from hashlib import sha256

import pytest
import torch

from models.rss_vnext.product_native_control_intervention_v1 import (
    SCHEMA, SourceNativeControlInterventionTeacherV1,
    qualified_native_control_intervention_loss_v1,
)


def _digest(t: torch.Tensor) -> str:
    return sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def _case():
    rest = torch.tensor([[
        [0.,0.,0.],[1.,0.,0.],[0.,1.,0.],
        [0.,0.,.4],[1.,0.,.4],[0.,1.,.4],
    ]])
    source = rest[:,None].repeat(1,2,1,1)
    source[0,1,3:,0] += .35  # independently authored wing-only response
    source_mask = torch.ones((1,2,6),dtype=torch.bool)
    mats=torch.eye(4)[None,None,None].repeat(1,2,2,1,1)
    mats[0,1,0,0,3]=.35
    teacher=SourceNativeControlInterventionTeacherV1(
        source_asset_sha256="a"*64,
        source_qualification_key="licensed-native-branch",
        source_control_event_sha256="b"*64,
        source_scope="SOURCE_BRANCH_ONLY",
        baseline_pose_index=0,intervention_pose_index=1,
        changed_skin_joint_matrix_mask=torch.tensor([True,False]),
        responding_vertex_ids=torch.tensor([3,4,5],dtype=torch.long),
        protected_vertex_ids=torch.tensor([0,1,2],dtype=torch.long),
    )
    receipt={
        "schema":SCHEMA,
        "admission_status":"QUALIFIED_SOURCE_NATIVE_ONE_CONTROL_INTERVENTION",
        "dataset_manifest_sha256":"f"*64,
        "source_asset_sha256":"a"*64,
        "source_qualification_key":"licensed-native-branch",
        "observed_rgba_sha256":"c"*64,
        "generation_checkpoint_sha256":"d"*64,
        "candidate_mstar_topology_address_sha256":"e"*64,
        "native_pose_transport_receipt_sha256":"0"*64,
        "source_control_event_sha256":"b"*64,
        "source_scope":"SOURCE_BRANCH_ONLY",
        "baseline_pose_index":0,"intervention_pose_index":1,
        "source_posed_sha256":_digest(source),
        "source_pose_mask_sha256":_digest(source_mask),
        "source_pose_matrices_sha256":_digest(mats),
        "changed_skin_joint_matrix_mask_sha256":
            _digest(teacher.changed_skin_joint_matrix_mask),
        "responding_vertex_ids_sha256":
            _digest(teacher.responding_vertex_ids),
        "protected_vertex_ids_sha256":
            _digest(teacher.protected_vertex_ids),
        "source_authored_exactly_one_control_changed":True,
        "source_response_observed_independently_of_student":True,
        "source_to_final_mstar_face_and_joint_transport_verified":True,
        "protected_not_a_negative_attachment_label":True,
        "student_generated_views_not_source_truth":True,
        "physical_material_or_xpbd_authority":False,
    }
    registry={"intervention":receipt}
    def evaluate(
        *, pred=None, src=source, mask=source_mask,
        joints=mats, t=teacher, reg=registry, enabled=True,
    ):
        if pred is None:
            pred=source.detach().clone().requires_grad_()
        return qualified_native_control_intervention_loss_v1(
            predicted_posed=pred, source_posed=src,
            source_pose_mask=mask, source_pose_matrices=joints,
            teacher=t if enabled else None,
            admission_key="intervention" if enabled else "",
            trusted_registry=reg, dataset_manifest_sha256="f"*64,
            mstar_address_sha256="e"*64,
            native_pose_transport_sha256="0"*64,
            source_asset_sha256="a"*64,
            source_qualification_key="licensed-native-branch",
            observed_rgba_sha256="c"*64,
            generation_checkpoint_sha256="d"*64,
            tensor_digest=_digest,
        )
    return evaluate,source,source_mask,mats,teacher,registry


def test_matching_source_branch_only_pose_has_zero_response_loss():
    evaluate,*_= _case()
    loss,parts,count=evaluate()
    assert float(loss)==pytest.approx(0.,abs=1e-9)
    assert count==6
    assert set(parts)=={
        "responding_delta_error","protected_delta_error",
        "relative_response_contrast","protected_motion_leakage",
    }
    assert evaluate(enabled=False)==(None,None,0)


def test_missed_wing_response_and_body_skin_bleeding_backpropagate_independently():
    evaluate,source,*_= _case()
    pred=source.detach().clone()
    pred[0,1,3:,0]=pred[0,0,3:,0] # failure: wing did not move
    pred[0,1,:3,0] += .20          # failure: body also moves
    pred.requires_grad_(True)
    loss,parts,count=evaluate(pred=pred)
    assert count==6 and float(loss)>0
    assert float(parts["responding_delta_error"])>0
    assert float(parts["protected_motion_leakage"])>0
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    assert bool((pred.grad[0,1,3:,0].abs()>0).any())
    assert bool((pred.grad[0,1,:3,0].abs()>0).any())


def test_source_author_control_receipt_and_changed_joint_mask_are_not_inferred():
    evaluate,source,source_mask,mats,t,registry=_case()
    corrupted={"intervention":dict(registry["intervention"])}
    corrupted["intervention"]["source_authored_exactly_one_control_changed"]=False
    with pytest.raises(ValueError,match="RECEIPT_DRIFT"):
        evaluate(reg=corrupted)
    changed=replace(t,changed_skin_joint_matrix_mask=torch.tensor([False,True]))
    with pytest.raises(ValueError,match="JOINT_WITNESS"):
        evaluate(t=changed)
    changed=replace(t,responding_vertex_ids=torch.tensor([0,4,5]))
    with pytest.raises(ValueError,match="VERTEX_SUPPORT"):
        evaluate(t=changed)
    changed=replace(t,source_control_event_sha256="1"*64)
    with pytest.raises(ValueError,match="RECEIPT_DRIFT"):
        evaluate(t=changed)


def test_unverified_source_response_and_unknown_vertex_pose_abstain():
    evaluate,source,source_mask,mats,t,registry=_case()
    bad=source.clone()
    bad[0,1,0,0]+=.15  # SOURCE protected body moved: no independent label
    with pytest.raises(ValueError,match="SOURCE_RESPONSE_NOT_VERIFIED"):
        evaluate(src=bad)
    badmask=source_mask.clone()
    badmask[0,1,3]=False
    with pytest.raises(ValueError,match="VERTEX_SUPPORT"):
        evaluate(mask=badmask)


def test_intervention_loss_is_eighth_distinct_training_family():
    from types import SimpleNamespace
    from models.rss_vnext.product_multitask_gradient_control_v1 import (
        admitted_product_task_losses_v1,
    )
    x=torch.tensor([.2,-.3],requires_grad=True)
    geo=x.square().sum()
    c=(x[0]-.3).square()
    result=SimpleNamespace(
        total_loss=geo+.4*c, seeded_tessa_loss=geo,
        component_udf_loss=None, product_semantic_loss=None,
        graphics_source_rgb_loss=None,realsas_joint_slrm_loss=None,
        product_mstar_pose_raster_loss=None,
        product_native_layer_clearance_loss=None,
        product_native_control_intervention_loss=c,
        product_axis_mira_loss=None,anchor_face_correspondence_loss=None,
    )
    groups=admitted_product_task_losses_v1(
        result,component_udf_weight=0.,graphics_rgb_weight=0.,
        graphics_joint_ray_weight=0.,anchor_face_weight=0.,
        native_control_intervention_weight=.4,
    )
    assert set(groups)=={"GEOMETRY","CONTROL_INTERVENTION"}
    torch.testing.assert_close(sum(groups.values()),result.total_loss)
