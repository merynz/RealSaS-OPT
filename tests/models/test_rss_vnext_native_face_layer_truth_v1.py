from __future__ import annotations

"""Independent native face-address truth; no decoded student geometry teacher."""
from dataclasses import replace
import pytest
import torch

from models.rss_vnext.native_face_layer_truth_v1 import (
    SCHEMA, NativeFaceLayerSamplesV1, validate_native_face_samples_v1,
)


def _proof():
    # Deliberately coincident body and cape, distinct native V/F addresses.
    base=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    vertices=torch.cat((base,base),dim=0)
    witness=NativeFaceLayerSamplesV1(
        vertices=vertices,
        faces=torch.tensor([[0,1,2],[3,4,5]],dtype=torch.long),
        face_layer_ids=torch.tensor([0,1],dtype=torch.long),
        face_component_ids=torch.tensor([7,9],dtype=torch.long),
        sample_face_ids=torch.tensor([0,1,0,1],dtype=torch.long),
        sample_barycentric=torch.tensor([
            [0.2,0.3,0.5],[0.2,0.3,0.5],
            [1.,0.,0.],[1.,0.,0.],
        ]),
        source_asset_sha256="a"*64,
        layer_namespace_sha256="b"*64,
    )
    xyz,roles,components=witness.replay()
    receipt={
        "source_native_face_witness_schema":SCHEMA,
        "source_native_face_witness_sha256":witness.receipt_sha256(),
        "source_native_face_witness_derivation":
            "SOURCE_NATIVE_INDEXED_TRIANGLE_BARYCENTRIC_NOT_STUDENT",
    }
    return witness,xyz,roles,components,receipt


def _court(w,xyz,roles,components,receipt):
    return validate_native_face_samples_v1(
        w,source_xyz=xyz,source_layer_ids=roles,
        source_component_ids=components,
        source_asset_sha256="a"*64,
        layer_namespace_sha256="b"*64,receipt=receipt,
    )


def test_native_face_replay_preserves_distinct_labels_at_identical_xyz():
    w,xyz,roles,components,receipt=_proof()
    torch.testing.assert_close(xyz[0],xyz[1])
    assert roles.tolist()==[0,1,0,1]
    assert components.tolist()==[7,9,7,9]
    _court(w,xyz,roles,components,receipt)


def test_wrong_layer_face_or_barycentric_cannot_relabel_overlapping_cloth():
    w,xyz,roles,components,receipt=_proof()
    with pytest.raises(ValueError,match="FACE_REPLAY_MISMATCH"):
        _court(w,xyz,1-roles,components,receipt)
    with pytest.raises(ValueError,match="FACE_REPLAY_MISMATCH"):
        _court(w,xyz+0.01,roles,components,receipt)
    with pytest.raises(ValueError,match="FACE_REPLAY_MISMATCH"):
        _court(w,xyz,roles,torch.zeros_like(components),receipt)
    forged_faces=replace(w,faces=w.faces.flip(0))
    with pytest.raises(ValueError,match="FACE_RECEIPT_DRIFT"):
        _court(forged_faces,xyz,roles,components,receipt)
    forged_bary=replace(w,sample_barycentric=w.sample_barycentric.roll(1,0))
    with pytest.raises(ValueError,match="FACE_RECEIPT_DRIFT"):
        _court(forged_bary,xyz,roles,components,receipt)


def test_face_graph_shared_vertex_crosses_role_is_rejected_despite_distinct_xyz():
    w,*_=_proof()
    corrupted=replace(
        w,faces=torch.tensor([[0,1,2],[0,4,5]],dtype=torch.long),
    )
    with pytest.raises(ValueError,match="CROSS_LAYER_OR_COMPONENT"):
        corrupted.replay()


def test_barycentric_domain_and_receipt_must_be_exact():
    w,xyz,roles,components,receipt=_proof()
    corrupted=replace(
        w,sample_barycentric=w.sample_barycentric+0.1,
    )
    with pytest.raises(ValueError,match="BARYCENTRIC_DOMAIN"):
        corrupted.replay()
    with pytest.raises(ValueError,match="FACE_RECEIPT_DRIFT"):
        _court(w,xyz,roles,components,{**receipt,
              "source_native_face_witness_sha256":"0"*64})
    with pytest.raises(ValueError,match="WITNESS_REQUIRED"):
        _court(None,xyz,roles,components,receipt)


def test_provenance_sampling_retains_gradient_on_independent_3d_source_only():
    w,_,_,_,_=_proof()
    v=w.vertices.clone().requires_grad_()
    new=replace(w,vertices=v)
    xyz,*_=new.replay()
    xyz.square().sum().backward()
    assert v.grad is not None
    assert bool(v.grad.abs().sum()>0)
