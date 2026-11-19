from __future__ import annotations

"""Source-native skin can teach MIRA without invented motion/physics."""
from dataclasses import replace

import pytest
import torch

from models.rss_vnext.native_face_layer_truth_v1 import (
    NativeFaceLayerSamplesV1, tensor_sha256_v1,
)
from models.rss_vnext.seeded_mstar_teacher_address_v1 import (
    seeded_mstar_teacher_address_sha256_v1,
)
from models.rss_vnext.tessa_seeded_surface_v1 import (
    build_tessa_seeded_surface_v1,
)
from models.rss_vnext.native_skin_transport_v1 import (
    DERIVATION, RECEIPT_SCHEMA, SourceNativeCandidateSkinTransportV1,
)


def _fixture():
    vertices=torch.tensor([
        [0.,0.,0.],[1.,0.,0.],[0.,1.,0.],
    ])
    faces=torch.tensor([[0,1,2]],dtype=torch.long)
    witness=NativeFaceLayerSamplesV1(
        vertices=vertices,faces=faces,
        face_layer_ids=torch.tensor([0]),face_component_ids=torch.tensor([0]),
        sample_face_ids=torch.tensor([0],dtype=torch.long),
        sample_barycentric=torch.tensor([[.25,.25,.5]]),
        source_asset_sha256="a"*64,layer_namespace_sha256="b"*64,
    )
    seed=build_tessa_seeded_surface_v1(
        iris_surface_xyz=vertices[None],iris_surface_faces=faces,
        component_ids_by_vertex=torch.zeros(3,dtype=torch.long),
        scene_sha256="e"*64,original_surface_sha256="f"*64,
    )
    mstar_address=seeded_mstar_teacher_address_sha256_v1(seed)
    transport=SourceNativeCandidateSkinTransportV1(
        source_asset_sha256=witness.source_asset_sha256,
        source_qualification_key="native-rig-skin-only",
        native_face_witness_sha256=witness.receipt_sha256(),
        source_candidate_mesh_face_sha256=tensor_sha256_v1(faces),
        axis_teacher_sequence_sha256="9"*64,
        native_skin_joint_order_sha256="a"*64,
        native_skin=torch.tensor([[1.,0.],[.5,.5],[0.,1.]]),
        candidate_native_face_ids=torch.tensor([0,0,0],dtype=torch.long),
        candidate_barycentric=torch.eye(3),
        candidate_mstar_topology_address_sha256=mstar_address,
    )
    receipt={
        "schema":RECEIPT_SCHEMA,
        "admission_status":"QUALIFIED_INDEPENDENT_SOURCE_FACE_TO_CANDIDATE_SKIN",
        "dataset_manifest_sha256":"3"*64,
        "source_asset_sha256":witness.source_asset_sha256,
        "native_face_witness_sha256":witness.receipt_sha256(),
        "skin_transport_receipt_sha256":transport.receipt_sha256(),
        "axis_teacher_sequence_sha256":"9"*64,
        "native_skin_joint_order_sha256":"a"*64,
        "native_skin_columns_match_axis_joint_order":True,
        "candidate_faces_sha256":tensor_sha256_v1(faces),
        "candidate_mstar_topology_address_sha256":mstar_address,
        "candidate_source_support_not_student_nearest":True,
        "source_rights_independently_verified":True,
        "generated_views_as_source_skin_truth":False,
        "source_skin_is_native_vertex_order":True,
    }
    return witness,transport,faces,vertices[None],receipt


def _replay(w,t,f,xyz,receipt):
    return t.replay(
        witness=w,candidate_faces=f,candidate_rest=xyz,
        trusted_source_registry={t.source_qualification_key:receipt},
        dataset_manifest_sha256="3"*64,
        expected_axis_teacher_sequence_sha256="9"*64,
        active_mstar_topology_address_sha256=t.candidate_mstar_topology_address_sha256,
        chunk_size=2,
    )


def test_exact_native_face_skin_replays_without_pose_or_clip():
    witness,transport,faces,xyz,receipt=_fixture()
    out=_replay(witness,transport,faces,xyz,receipt)
    assert out.shape==(1,3,2)
    assert torch.allclose(out[0],transport.native_skin,atol=0,rtol=0)
    assert transport.source_candidate_support_derivation==DERIVATION


def test_skin_transport_rejects_false_rights_rehashed_support_and_nearest_guess():
    witness,transport,faces,xyz,receipt=_fixture()
    with pytest.raises(ValueError,match="SOURCE_RECEIPT_REQUIRED"):
        _replay(witness,transport,faces,xyz,{**receipt,"source_rights_independently_verified":False})
    with pytest.raises(ValueError,match="SOURCE_RECEIPT_REQUIRED"):
        _replay(witness,replace(transport,native_skin=torch.tensor([
            [.5,.5],[.5,.5],[0.,1.],
        ])),faces,xyz,receipt)
    with pytest.raises(ValueError,match="REST_SUPPORT_DRIFT"):
        shifted=xyz.clone()
        shifted[:,1,0]+=.3
        _replay(witness,transport,faces,shifted,receipt)


def test_skin_transport_refuses_invalid_native_weights_or_face_admission():
    witness,transport,faces,xyz,receipt=_fixture()
    invalid=replace(transport,native_skin=torch.tensor([
        [1.,0.],[.8,.8],[0.,1.],
    ]))
    with pytest.raises(ValueError,match="TENSOR_DOMAIN"):
        _replay(witness,invalid,faces,xyz,{
            **receipt,"skin_transport_receipt_sha256":invalid.receipt_sha256(),
        })
    changed=faces[:,[0,2,1]]
    with pytest.raises(ValueError,match="SOURCE_IDENTITY"):
        _replay(witness,transport,changed,xyz,receipt)


def test_skin_receipt_must_bind_exact_dataset_manifest():
    w,t,f,xyz,r=_fixture()
    with pytest.raises(ValueError,match="SOURCE_RECEIPT_REQUIRED"):
        _replay(w,t,f,xyz,{**r,"dataset_manifest_sha256":"4"*64})


def test_native_skin_rejects_axial_joint_order_forgery():
    w,t,f,xyz,receipt=_fixture()
    with pytest.raises(ValueError,match="SOURCE_RECEIPT_REQUIRED"):
        _replay(w,t,f,xyz,{
            **receipt,"native_skin_columns_match_axis_joint_order":False,
        })
    with pytest.raises(ValueError,match="SOURCE_RECEIPT_REQUIRED"):
        _replay(w,t,f,xyz,{
            **receipt,"native_skin_joint_order_sha256":"b"*64,
        })
    with pytest.raises(ValueError,match="SOURCE_IDENTITY"):
        t.replay(
            witness=w,candidate_faces=f,candidate_rest=xyz,
            dataset_manifest_sha256="3"*64,
            expected_axis_teacher_sequence_sha256="0"*64,
            active_mstar_topology_address_sha256=t.candidate_mstar_topology_address_sha256,
            trusted_source_registry={t.source_qualification_key:receipt},
        )



def test_native_skin_rejects_new_mstar_revision_even_if_face_index_buffer_same():
    w,t,f,xyz,receipt=_fixture()
    # Same source 3D, same exact triangle array, same native barycentrics.
    # Repartitioning the M* component domain invalidates old source truth.
    source=build_tessa_seeded_surface_v1(
        iris_surface_xyz=xyz,iris_surface_faces=f,
        component_ids_by_vertex=torch.zeros(3,dtype=torch.long),
        scene_sha256="e"*64,original_surface_sha256="f"*64,
    )
    altered=replace(
        source,
        vertex_component_ids=torch.ones(3,dtype=torch.long),
        face_component_ids=torch.ones(1,dtype=torch.long),
    )
    altered.validate()
    changed=seeded_mstar_teacher_address_sha256_v1(altered)
    assert changed!=t.candidate_mstar_topology_address_sha256
    with pytest.raises(ValueError,match="MSTAR_TEACHER_TOPOLOGY_ADDRESS_DRIFT"):
        t.replay(
            witness=w,candidate_faces=f,candidate_rest=xyz,
            active_mstar_topology_address_sha256=changed,
            trusted_source_registry={t.source_qualification_key:receipt},
            dataset_manifest_sha256="3"*64,
            expected_axis_teacher_sequence_sha256="9"*64,
        )


def test_native_skin_rejects_missing_topology_receipt():
    w,t,f,xyz,receipt=_fixture()
    forged=dict(receipt)
    forged.pop("candidate_mstar_topology_address_sha256")
    with pytest.raises(ValueError,match="SOURCE_RECEIPT_REQUIRED"):
        _replay(w,t,f,xyz,forged)
