from __future__ import annotations

"""Source-native signed body/garment pose clearance: tiny ABI regressions only."""
from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace

import pytest
import torch

from models.rss_vnext.product_native_layer_clearance_v1 import (
    SCHEMA, SourceNativeLayerClearanceTeacherV1,
    native_layer_clearance_objective_v1,
)


def _digest(t: torch.Tensor) -> str:
    return sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def _case():
    # Source mesh has a body face (outward +Z) and a separate garment face.
    rest = torch.tensor([[
        [0.,0.,0.],[1.,0.,0.],[0.,1.,0.],
        [.25,.25,.2],[.50,.25,.15],[.35,.5,.16],
    ]])
    faces = torch.tensor([[0,1,2],[3,4,5]], dtype=torch.long)
    mesh = SimpleNamespace(
        rest_vertices_iris=rest, faces=faces,
        vertex_component_ids=torch.tensor([0,0,0,1,1,1]),
        face_component_ids=torch.tensor([0,1]),
    )
    source = rest[:,None].repeat(1,2,1,1)
    source[0,1,:,0] += .3
    source[0,1,3:,2] += .05
    mask = torch.ones((1,2,6),dtype=torch.bool)
    pair = SourceNativeLayerClearanceTeacherV1(
        source_asset_sha256="a"*64,
        source_qualification_key="independent-native-source",
        garment_vertex_ids=torch.tensor([3,4],dtype=torch.long),
        body_face_ids=torch.tensor([0,0],dtype=torch.long),
        body_face_barycentric=torch.tensor([
            [.50,.25,.25],[.25,.50,.25],
        ]),
        qualified_pose_pair_mask=torch.ones((1,2,2),dtype=torch.bool),
    )
    proof = {
        "schema":SCHEMA,
        "admission_status":"QUALIFIED_SOURCE_NATIVE_BODY_GARMENT_CLEARANCE",
        "dataset_manifest_sha256":"f"*64,
        "source_asset_sha256":"a"*64,
        "source_qualification_key":"independent-native-source",
        "candidate_mstar_topology_address_sha256":"b"*64,
        "native_pose_transport_receipt_sha256":"c"*64,
        "source_candidate_posed_sha256":_digest(source),
        "source_pose_mask_sha256":_digest(mask),
        "garment_vertex_ids_sha256":_digest(pair.garment_vertex_ids),
        "body_face_ids_sha256":_digest(pair.body_face_ids),
        "body_face_barycentric_sha256":_digest(pair.body_face_barycentric),
        "qualified_pose_pair_mask_sha256":_digest(pair.qualified_pose_pair_mask),
        "source_native_body_face_winding_outward_certified":True,
        "source_body_shell_signed_solid_verified":True,
        "source_native_pair_correspondence_verified":True,
        "source_rights_independently_verified":True,
        "source_generated_views_as_clearance_truth":False,
        "student_nearest_pairing_forbidden":True,
        "physical_material_or_xpbd_authority":False,
    }
    registry = {"native-clearance":proof}
    def run(*, pred=None, t=pair, rows=registry, src=source, valid=mask):
        if pred is None:
            pred = src.detach().clone().requires_grad_()
        return native_layer_clearance_objective_v1(
            mesh=mesh,predicted_posed=pred,source_native_posed=src,
            source_pose_mask=valid,teacher=t,
            admission_key="native-clearance",trusted_registry=rows,
            dataset_manifest_sha256="f"*64,
            native_pose_transport_sha256="c"*64,
            mstar_address_sha256="b"*64,
            source_asset_sha256="a"*64,
            source_qualification_key="independent-native-source",
            tensor_digest=_digest,
        )
    return run, mesh, source, mask, pair, registry


def test_source_same_mstar_rest_and_pose_has_zero_clearance_loss():
    run,*_ = _case()
    total,parts,count=run()
    assert total.item() == pytest.approx(0.,abs=1e-8)
    assert count==4 and set(parts)=={
        "source_native_signed_gap","student_signed_gap",
        "source_gap_regression","separation_barrier",
    }


def test_collapse_gradients_reach_predicted_garment_and_body_mstar_vertices():
    run,mesh,source,*_ = _case()
    predicted=source.detach().clone()
    predicted[0,:,3:,2] -= .3
    predicted.requires_grad_()
    total,parts,count=run(pred=predicted)
    assert float(total)>0
    assert float(parts["separation_barrier"])>0
    total.backward()
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()
    assert predicted.grad[0,:,3:,2].abs().sum()>0
    assert predicted.grad[0,:,:3,2].abs().sum()>0
    assert count==4


def test_nonindependent_or_stale_source_native_pair_receipt_is_refused():
    run,mesh,source,mask,pair,registry=_case()
    forged = {"native-clearance":dict(registry["native-clearance"])}
    forged["native-clearance"]["source_native_body_face_winding_outward_certified"]=False
    with pytest.raises(ValueError,match="RECEIPT_DRIFT"):
        run(rows=forged)
    with pytest.raises(ValueError,match="RECEIPT_DRIFT"):
        run(t=replace(pair,body_face_ids=torch.tensor([1,1])))
    bad=replace(pair,garment_vertex_ids=torch.tensor([0,4]))
    with pytest.raises(ValueError,match="COMPONENT_COLLAPSE"):
        run(t=bad)
    bad=replace(pair,qualified_pose_pair_mask=torch.ones((1,2,3),dtype=torch.bool))
    with pytest.raises(ValueError,match="TEACHER_DOMAIN"):
        run(t=bad)


def test_unknown_pose_mask_and_source_body_penetration_do_not_teach_as_valid():
    run,mesh,source,mask,pair,registry=_case()
    corrupt=mask.clone()
    corrupt[0,1,3]=False
    with pytest.raises(ValueError,match="UNSUPPORTED_POSE"):
        run(valid=corrupt)
    # A source garment that already passes through the body is not a
    # qualified separation teacher even if rehashed into a new registry.
    penetrated=source.clone()
    penetrated[0,:,3:,2]=-.07
    forged={"native-clearance":dict(registry["native-clearance"])}
    forged["native-clearance"]["source_candidate_posed_sha256"]=_digest(penetrated)
    with pytest.raises(ValueError,match="NOT_POSITIVE_SOURCE_GAP"):
        run(src=penetrated, rows=forged)
