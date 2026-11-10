from __future__ import annotations

"""One two-view RSS H/mesh/rig/skin candidate through REAL Compiler and runtime.

Teacher vertices and rig events are supplied only as deliberately synthetic
unit fixture to exercise contracts. NOT a real learned checkout or ProductRevision.
"""
from dataclasses import replace
import math
from io import BytesIO

import numpy as np
import pytest
import torch

from models.axis.rig_program_model_v6 import RigProgramEventV6 as E, RigTokenKindV6 as K
from models.rss_vnext import (
    RSSvNextConfig, RSSvNextPatchEncoderForTests,
    RSSvNextUnifiedCharacterModel,
)
from models.rss_vnext.inference_candidate_v1 import (
    build_rss_vnext_inference_evidence_v1,
)
from models.tessa.v1.model_v1 import TESSAConfigV1
from models.tessa.v1.tokenization_v1 import encode_adjacent_teacher_mesh_v1

from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import (
    CanonicalMeshIR, CanonicalMeshVertexIR, canonical_mesh_lineage_hash_v1,
)
from compiler.realsas_compiler_core.rss_vnext_mechanics_bridge_v1 import (
    RSSvNextCanonicalMeshCandidateBindingV1,
    compile_rss_vnext_mechanics_candidate_v1,
)
from compiler.realsas_compiler_core.playback_full_surface_v3 import (
    build_full_surface_runtime_meshes_v3,
)
from compiler.realsas_compiler_core.playback_runtime_v3 import (
    RuntimeV3Slot, RuntimeV3PlaybackContract,
    RuntimeV3AppearancePatch, AppearanceProvenance,
)
from compiler.realsas_compiler_core.controllable_puppet_runtime_v1 import (
    VIEW_IDS_V1, ControllablePuppetBundleV1, ControllablePuppetSessionV1,
    RuntimeSelectionBindingV1,
)
from compiler.realsas_compiler_core.types import QualificationError, SurfaceSupportBinding


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(19)
    torch.set_num_threads(1)


def model():
    cfg = RSSvNextConfig.tiny_for_test()
    return RSSvNextUnifiedCharacterModel(
        cfg,vision_encoder=RSSvNextPatchEncoderForTests(cfg.iris.image_token_dim),
    ).eval()


def source_state(net):
    rgba=torch.rand(1,2,4,8,8)
    cam=torch.zeros((1,2,13))
    cam[:,:,3]=cam[:,:,7]=cam[:,:,11]=cam[:,:,12]=1
    state,_,_=net.encode_observations(rgba,cam)
    return state


def rig_events():
    # Three typed source-program joints. No humanoid prior implied by labels.
    frame=(1.,0.,0.,0.,1.,0.)
    return tuple([
        E(K.BOS),
        E(K.JOINT,pointer_a=-1,coord_indices=(127,127,127)),
        E(K.JOINT,pointer_a=0,coord_indices=(127,190,127)),
        E(K.JOINT,pointer_a=1,coord_indices=(127,254,127)),
        E(K.FRAMES_START),
        E(K.FRAME,pointer_a=0,frame_6d=frame),
        E(K.FRAME,pointer_a=1,frame_6d=frame),
        E(K.FRAME,pointer_a=2,frame_6d=frame),
        E(K.CONTROLS_START),
        E(K.CONTROL,aux_type_id=1,pointer_a=-1,coord_indices=(127,127,127)),
        E(K.CONTROL_TARGET,pointer_a=0),
        E(K.CONTROL,aux_type_id=3,pointer_a=0,coord_indices=(127,254,127)),
        E(K.CONTROL_TARGET,pointer_a=2),
        E(K.CONTROL,aux_type_id=4,pointer_a=0,coord_indices=(254,190,127)),
        E(K.CHAINS_START),
        E(K.CHAIN,pointer_a=0,pointer_b=2),
        E(K.CONSTRAINTS_START),
        E(K.CONSTRAINT,aux_type_id=0,pointer_a=0,scalar_params=(1.,1.,1.,0.)),
        E(K.CONSTRAINT_INPUT,pointer_a=1),
        E(K.CONSTRAINT_INPUT,pointer_a=2),
        E(K.CONSTRAINT_OUTPUT,pointer_a=2),
        E(K.ANCHORS_START),
        E(K.SKIN_START),
        *[E(K.SKIN_CODE,skin_code_id=0) for _ in range(12)],
        E(K.EOS),
    ])


def candidate(net):
    state=source_state(net)
    tri=encode_adjacent_teacher_mesh_v1(
        np.array([[-.25,-.25,0],[.25,-.25,0],[.25,.25,0]],dtype=np.float64),
        np.array([[0,1,2]],dtype=np.int64),
        component_id="unit",chart_id="unit",
        cfg=TESSAConfigV1(coordinate_bins=net.cfg.tessa.coordinate_bins),
    )
    mesh=net.build_mesh_candidate(
        state,torch.tensor(tri.token_ids,dtype=torch.long),
        source_authority="LEARNED_PREDICTED_TOPOLOGY",
    )
    n=mesh.rest_vertices_iris.shape[1]
    assert n==3
    result={
        "state":state,
        "mesh_candidate":mesh,
        "mesh_status":"UNQUALIFIED_MESH_CANDIDATE",
        "qualification_status":"UNQUALIFIED_NEURAL_PROPOSAL",
        "observed_view_count":2,
        "control_event_proposal":rig_events(),
        "skin_weight_proposal":torch.eye(3)[None],
        "joint_valid_mask":torch.ones((1,3),dtype=torch.bool),
    }
    return build_rss_vnext_inference_evidence_v1(
        result,
        model_architecture_id=net.architecture_id,
        checkpoint_sha256="e"*64,
        source_image_sha256=("a"*64,"b"*64),
    )


def canonical_source_mesh(evidence):
    with np.load(BytesIO(evidence.array_archive_npz),allow_pickle=False) as r:
        xyz=r["mesh_rest_xyz"][0]
        faces=r["mesh_faces"]
    def support(i):
        return SurfaceSupportBinding("EXACT_SURFACE_NODE",((f"TESTSRC:{i}",1.),),{})
    vertices=tuple(CanonicalMeshVertexIR(
        canonical_mesh_vertex_id=f"MV:{i}",
        support_binding=support(i),
        component_id="body",
        P=tuple(float(x) for x in row),
        source_candidate_vertex_id=f"CV:{i}",
    ) for i,row in enumerate(xyz))
    ids=tuple(v.canonical_mesh_vertex_id for v in vertices)
    oriented_faces=tuple(tuple(ids[int(i)] for i in f) for f in faces)
    edges=tuple(sorted({
        tuple(sorted((f[i],f[(i+1)%3])))
        for f in oriented_faces for i in range(3)
    }))
    mesh=CanonicalMeshIR(
        vertices=vertices,faces=oriented_faces,edges=edges,
        source_evidence_binding_hash="1"*64,
        partition_binding_hash="2"*64,
        carrier_policy_binding_hash="3"*64,
        static_qualification_hash="4"*64,
        carrier_evidence_hash="5"*64,
        carrier_topology_hash="6"*64,
        carrier_geometry_hash="7"*64,
        proposal_evidence_refs=("8"*64,),
        source_candidate_lineage_hash="9"*64,
        mesh_lineage_hash="",
        metadata={"authority":"COMPILER_CANONICAL_MESH",
                  "unit_test_not_a_valid_source_proof":True},
    )
    mesh=replace(mesh,mesh_lineage_hash=canonical_mesh_lineage_hash_v1(mesh))
    bind=RSSvNextCanonicalMeshCandidateBindingV1(
        candidate_index_to_carrier_vertex_id=("CV:0","CV:1","CV:2"),
        component_names_by_model_index=("body",),
        normalization_center=(0.,0.,0.),normalization_scale=1.,
        canonical_mesh_source_qualification_receipt_sha256="f"*64,
    )
    return mesh,bind


def test_unified_candidate_to_canonical_rig_skin_and_real_compiler_rest_pose():
    net=model()
    bundle=candidate(net)
    mesh,bind=canonical_source_mesh(bundle)
    merged=compile_rss_vnext_mechanics_candidate_v1(
        bundle,mesh,bind,axis_config=net.cfg.axis_config(),
        codec_architecture_id="RealSaS.MIRA.CarrierNativeFSQSkinCodec.v6_0",
        verify_independent_mesh_qualification=lambda m,b:
            m.metadata.get("unit_test_not_a_valid_source_proof") is True,
    )
    assert merged.authority=="COMPILER_MECHANICS_QUALIFIED_NOT_A_PRODUCT_REVISION"
    assert merged.rig.mesh_lineage_hash==mesh.mesh_lineage_hash
    assert merged.skin.mesh_lineage_hash==mesh.mesh_lineage_hash
    assert merged.skin.deform_structure_hash==merged.rig.deform_structure_hash
    assert len(merged.rig.joints)==3
    assert len(merged.rig.constraints)==1
    assert len(merged.skin.rows)==len(mesh.vertices)
    assert merged.rest_deformation.canonical_mesh_vertex_ids == tuple(
        v.canonical_mesh_vertex_id for v in mesh.vertices
    )
    assert merged.skin.skin_codec_checkpoint_hash==bundle.manifest["checkpoint_sha256"]


def test_model_compiler_runtime_reprojects_one_mechanical_identity_eight_ways():
    net=model()
    candidate_bundle=candidate(net)
    mesh,binding=canonical_source_mesh(candidate_bundle)
    compiled=compile_rss_vnext_mechanics_candidate_v1(
        candidate_bundle,mesh,binding,
        axis_config=net.cfg.axis_config(),
        codec_architecture_id="RealSaS.MIRA.CarrierNativeFSQSkinCodec.v6_0",
        verify_independent_mesh_qualification=lambda *_:True,
    )
    cameras={}
    for i, view in enumerate(VIEW_IDS_V1):
        angle=math.pi * i/4
        si,co=math.sin(angle),math.cos(angle)
        cameras[view]={
            "view_index":i,
            "origin":(3*si,0.,3*co),
            "right":(co,0.,-si),
            "screen_up":(0.,1.,0.),
            "forward":(-si,0.,-co),
            "half_extent":1.,
            "resolution":128,
        }
    rest=np.asarray([v.P for v in mesh.vertices])
    fid={v.canonical_mesh_vertex_id:i for i,v in enumerate(mesh.vertices)}
    faces=np.asarray([[fid[x] for x in f] for f in mesh.faces],dtype=np.int64)
    surface=build_full_surface_runtime_meshes_v3(
        rest,faces,cameras=cameras,
        uv_by_view={v:np.asarray([[0.,0.],[.5,1.],[1.,0.]]) for v in VIEW_IDS_V1},
        slot_id="S0",attachment_id="A0",
    )
    playback=RuntimeV3PlaybackContract(
        slots=(RuntimeV3Slot("S0",compiled.rig.joints[0].canonical_joint_id,0,"A0"),),
        meshes=surface.meshes,
        appearance_patches=tuple(RuntimeV3AppearancePatch(
            patch_id=f"AP:{view}",mesh_id=f"{view}:A0",
            face_indices=(0,),provenance=AppearanceProvenance.UNSEEN,
            donor_view_index=None,atlas_id=None,
        ) for view in VIEW_IDS_V1),
    )
    session=ControllablePuppetSessionV1(ControllablePuppetBundleV1(
        mesh=mesh,rig=compiled.rig,skin=compiled.skin,playback=playback,
        cameras_by_view=cameras,
        vertex_ids_by_attachment={"A0":tuple(v.canonical_mesh_vertex_id for v in mesh.vertices)},
        selection_bindings=(RuntimeSelectionBindingV1(
            attachment_id="A0",canonical_component_id="body",
            source_support_evidence_id="SYNTHETIC_UNIT_ONLY",
        ),),
        product_revision_id="SYNTHETIC_TEST_ONLY",
        product_qualification_receipt="SYNTHETIC_NOT_PRODUCT",
    ))
    baseline=session.frame()
    session.set_view_index(1)
    turned=session.frame()
    assert turned.deformed.canonical_mesh_vertex_ids==baseline.deformed.canonical_mesh_vertex_ids
    assert turned.deformed.positions==baseline.deformed.positions
    assert tuple((x.x,x.y) for x in turned.meshes[0].vertices)!=tuple(
        (x.x,x.y) for x in baseline.meshes[0].vertices
    )


def test_reject_mesh_redistribution_missing_court_and_fake_components():
    net=model();evidence=candidate(net);mesh,b=canonical_source_mesh(evidence)
    kwargs=dict(axis_config=net.cfg.axis_config(),
        codec_architecture_id="RealSaS.MIRA.CarrierNativeFSQSkinCodec.v6_0")
    with pytest.raises(QualificationError,match="EXTERNAL_MESH_COURT_REJECTED"):
        compile_rss_vnext_mechanics_candidate_v1(
            evidence,mesh,b,verify_independent_mesh_qualification=lambda *_:False,
            **kwargs,
        )
    bad_mesh=replace(mesh,vertices=tuple(
        replace(v,P=(v.P[0]+.5,v.P[1],v.P[2])) if i==0 else v
        for i,v in enumerate(mesh.vertices)
    ))
    bad_mesh=replace(bad_mesh,mesh_lineage_hash=canonical_mesh_lineage_hash_v1(bad_mesh))
    with pytest.raises(QualificationError,match="REST_GEOMETRY_DRIFT"):
        compile_rss_vnext_mechanics_candidate_v1(
            evidence,bad_mesh,b,
            verify_independent_mesh_qualification=lambda *_:True,**kwargs,
        )
    invalid_binding=replace(b,component_names_by_model_index=("weapon",))
    with pytest.raises(QualificationError,match="COMPONENT_OWNERSHIP_DRIFT"):
        compile_rss_vnext_mechanics_candidate_v1(
            evidence,mesh,invalid_binding,
            verify_independent_mesh_qualification=lambda *_:True,**kwargs,
        )


def test_untrained_mesh_only_evidence_cannot_mint_rig_or_product():
    net=model()
    original=candidate(net)
    with np.load(BytesIO(original.array_archive_npz),allow_pickle=False) as z:
        assert z["mesh_faces"].shape[-1]==3
    m=dict(original.manifest)
    m["status"]="MESH_CANDIDATE"
    from models.rss_vnext.inference_candidate_v1 import RSSvNextInferenceEvidenceBundleV1
    evidence=RSSvNextInferenceEvidenceBundleV1(m,original.array_archive_npz)
    mesh,b=canonical_source_mesh(original)
    with pytest.raises(QualificationError,match="REQUIRES_CONTROL_SKIN_EVIDENCE"):
        compile_rss_vnext_mechanics_candidate_v1(
            evidence,mesh,b,
            axis_config=net.cfg.axis_config(),
            codec_architecture_id="RealSaS.MIRA.CarrierNativeFSQSkinCodec.v6_0",
            verify_independent_mesh_qualification=lambda *_:True,
        )
