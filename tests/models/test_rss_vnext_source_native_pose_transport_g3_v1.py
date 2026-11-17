from __future__ import annotations

"""Native face support, pose replay and worst-G3 risk are independent of loss averages."""
from dataclasses import replace

import pytest
import torch

from models.rss_vnext.native_face_layer_truth_v1 import (
    NativeFaceLayerSamplesV1,tensor_sha256_v1,
)
from models.rss_vnext.native_pose_transport_v1 import (
    SCHEMA,SOURCE_RECEIPT_SCHEMA,
    SourceNativeCandidatePoseTransportV1,
    validate_source_native_pose_candidate_teacher_v1,
)
from models.rss_vnext.seeded_pose_consequence_objective_v1 import (
    exact_m_qualified_pose_strain_v1,
)


def _source():
    xyz=torch.tensor([
        [0.,0.,.4],[-.4,-.3,-.2],[.4,-.3,-.2],[0.,.4,-.2],
    ])
    f=torch.tensor([
        [0,2,1],[0,1,3],[0,3,2],[1,2,3],
    ],dtype=torch.long)
    witness=NativeFaceLayerSamplesV1(
        vertices=xyz,faces=f,
        face_layer_ids=torch.zeros(4,dtype=torch.long),
        face_component_ids=torch.zeros(4,dtype=torch.long),
        sample_face_ids=torch.arange(4,dtype=torch.long),
        sample_barycentric=torch.full((4,3),1./3.),
        source_asset_sha256="a"*64,
        layer_namespace_sha256="b"*64,
    )
    native_skin=torch.tensor([
        [1.,0.],[1.,0.],[1.,0.],[.8,.2],
    ])
    mats=torch.eye(4).repeat(1,2,2,1,1)
    mats[0,1,1,0,3]=.4
    parent_ids=torch.tensor([0,0,0,1],dtype=torch.long)
    bary=torch.tensor([
        [1.,0.,0.],[0.,0.,1.],[0.,1.,0.],[0.,0.,1.],
    ])
    mask=torch.ones((1,2,4),dtype=torch.bool)
    key="native-bary-source-support"
    support=SourceNativeCandidatePoseTransportV1(
        source_asset_sha256=witness.source_asset_sha256,
        native_face_witness_sha256=witness.receipt_sha256(),
        source_qualification_key=key,
        native_skin=native_skin,native_skin_matrices=mats,
        candidate_native_face_ids=parent_ids,
        candidate_barycentric=bary,candidate_pose_mask=mask,
        source_candidate_mesh_face_sha256=tensor_sha256_v1(f),
    )
    registry={key:{
        "schema":SOURCE_RECEIPT_SCHEMA,
        "admission_status":"QUALIFIED_INDEPENDENT_SOURCE_FACE_TO_CANDIDATE_SUPPORT",
        "source_asset_sha256":witness.source_asset_sha256,
        "native_face_witness_sha256":witness.receipt_sha256(),
        "pose_transport_receipt_sha256":support.receipt_sha256(),
        "candidate_faces_sha256":tensor_sha256_v1(f),
        "candidate_source_support_not_student_nearest":True,
        "source_rights_independently_verified":True,
        "generated_views_as_source_pose_truth":False,
        "source_skin_is_native_vertex_order":True,
    }}
    receipt={
        "native_pose_transport_schema":SCHEMA,
        "native_pose_transport_receipt_sha256":support.receipt_sha256(),
        "native_pose_transport_independently_audited":True,
    }
    return witness,support,registry,receipt


def test_native_original_face_barycentric_replays_exact_joint_order_and_poses():
    witness,support,registry,receipt=_source()
    rest=witness.vertices[None].clone().requires_grad_()
    skin,target,mask,source_rest=support.replay(
        witness=witness,candidate_faces=witness.faces,
        candidate_rest=rest,trusted_source_registry=registry,
    )
    torch.testing.assert_close(source_rest,rest)
    torch.testing.assert_close(skin,support.native_skin[None])
    assert target.shape==(1,2,4,3)
    assert target[0,1,3,0].item()==pytest.approx(.08,abs=1.e-6)
    validate_source_native_pose_candidate_teacher_v1(
        support=support,native_face_witness=witness,
        candidate_faces=witness.faces,candidate_rest=rest,
        source_candidate_skin=skin,source_candidate_posed=target,
        source_pose_mask=mask,source_pose_matrices=support.native_skin_matrices,
        teacher_source_asset_sha256=witness.source_asset_sha256,
        teacher_receipt=receipt,trusted_source_registry=registry,
    )


def test_pose_transport_requires_registered_source_and_exact_candidate_indices():
    witness,support,registry,receipt=_source()
    with pytest.raises(ValueError,match="RECEIPT_REQUIRED"):
        support.replay(
            witness=witness,candidate_faces=witness.faces,
            candidate_rest=witness.vertices[None],
            trusted_source_registry={},
        )
    with pytest.raises(ValueError,match="SOURCE_IDENTITY"):
        support.replay(
            witness=witness,candidate_faces=witness.faces.flip(0),
            candidate_rest=witness.vertices[None],
            trusted_source_registry=registry,
        )
    skin,target,mask,_=support.replay(
        witness=witness,candidate_faces=witness.faces,
        candidate_rest=witness.vertices[None],
        trusted_source_registry=registry,
    )
    with pytest.raises(ValueError,match="REPLAY_OR_RECEIPT_DRIFT"):
        validate_source_native_pose_candidate_teacher_v1(
            support=support,native_face_witness=witness,
            candidate_faces=witness.faces,
            candidate_rest=witness.vertices[None],
            source_candidate_skin=skin,
            source_candidate_posed=target+1.,
            source_pose_mask=mask,
            source_pose_matrices=support.native_skin_matrices,
            teacher_source_asset_sha256=witness.source_asset_sha256,
            teacher_receipt=receipt,trusted_source_registry=registry,
        )


def test_no_student_nearest_or_large_rest_drift_can_qualify_pose_teacher():
    witness,support,registry,_=_source()
    broken=dict(next(iter(registry.values())))
    broken["candidate_source_support_not_student_nearest"]=False
    with pytest.raises(ValueError,match="RECEIPT_REQUIRED"):
        support.replay(
            witness=witness,candidate_faces=witness.faces,
            candidate_rest=witness.vertices[None],
            trusted_source_registry={support.source_qualification_key:broken},
        )
    wrong=witness.vertices[None].clone()
    wrong[0,3,2]+=.2
    with pytest.raises(ValueError,match="REST_SUPPORT_DRIFT"):
        support.replay(
            witness=witness,candidate_faces=witness.faces,
            candidate_rest=wrong,trusted_source_registry=registry,
        )


def test_all_face_pose_consequence_gradient_and_one_collapsed_face_barrier():
    witness,support,registry,_=_source()
    rest=witness.vertices[None].clone()
    skin,target,mask,_=support.replay(
        witness=witness,candidate_faces=witness.faces,
        candidate_rest=rest,trusted_source_registry=registry,
    )
    rest.requires_grad_()
    from models.rss_vnext.animation_ready_mesh import (
        posed_lbs_from_joint_matrices,
    )
    pred=posed_lbs_from_joint_matrices(
        rest,skin,support.native_skin_matrices,
    )
    safe=exact_m_qualified_pose_strain_v1(
        rest_xyz=rest,faces=witness.faces,
        predicted_poses=pred,source_native_posed=target,
        source_pose_mask=mask,source_candidate_skin=skin,
        scale=rest.new_tensor(1.),
    )
    assert safe["total"].item()<1.e-5
    # Collapse one of the four faces only in the second pose. Remaining
    # faces/poses cannot dilute the G3 worst-face barrier.
    bad=pred.detach().clone()
    bad[0,1,3]=bad[0,1,1]
    bad.requires_grad_()
    result=exact_m_qualified_pose_strain_v1(
        rest_xyz=rest,faces=witness.faces,
        predicted_poses=bad,source_native_posed=target,
        source_pose_mask=mask,source_candidate_skin=skin,
        scale=rest.new_tensor(1.),
    )
    assert result["g3_g3b_worst_barrier"].item()>0.9
    result["total"].backward()
    assert bad.grad is not None
    assert torch.isfinite(bad.grad).all()
    assert bad.grad.abs().sum()>0
