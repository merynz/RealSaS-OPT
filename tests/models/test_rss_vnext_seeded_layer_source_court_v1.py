from __future__ import annotations

"""Body/cape at identical coordinates: source truth cannot bleed across.

This fixture authors independent native layer annotations and proof receipts.
It is NOT evidence of real corpus labels, mesh reconstruction or GPU quality.
"""
from dataclasses import replace

import pytest
import torch

from models.rss_vnext import (
    RSSvNextConfig, RSSvNextPatchEncoderForTests,
    RSSvNextUnifiedCharacterModel,
)
from models.rss_vnext.layered_iris_seed_v1 import layer_namespace_digest_v1
from models.rss_vnext.native_face_layer_truth_v1 import (
    SCHEMA as NATIVE_FACE_SCHEMA, NativeFaceLayerSamplesV1,
)
from models.rss_vnext.native_layer_open_boundary_v1 import (
    SCHEMA as OPEN_HEM_SCHEMA, DERIVATION as OPEN_HEM_DERIVATION,
    NativeLayerOpenBoundaryV1,exact_open_boundary_edges_v1,
)
from models.rss_vnext.seeded_layer_source_court_v1 import (
    SOURCE_LAYER_AUTHORITY, SOURCE_LAYER_DERIVATION,
)
from models.rss_vnext.tessa_seeded_surface_v1 import (
    build_tessa_seeded_surface_v1,
)
from models.rss_vnext.training_seeded_tessa_v1 import (
    RSSvNextSeededTESSATrainerV1, SeededTESSASourceTeacherV1,
    SeededTESSALossWeightsV1, PHASE, SCHEMA, _digest,
    _bounded_nearest_source_indices_v1,
)


def _setup(*, source_component_ids=None, weights=None):
    torch.manual_seed(2026111701)
    torch.set_num_threads(1)
    cfg=RSSvNextConfig.tiny_for_test()
    model=RSSvNextUnifiedCharacterModel(
        cfg,vision_encoder=RSSvNextPatchEncoderForTests(
            cfg.iris.image_token_dim,
        ),
    ).train()
    rgba=torch.rand(1,2,4,8,8)
    cameras=torch.zeros(1,2,13)
    cameras[...,3]=cameras[...,7]=cameras[...,11]=cameras[...,12]=1
    state,_,_=model.encode_observations(rgba,cameras)
    base=torch.tensor([
        [-.4,.4,0.],[.4,.4,0.],[-.4,-.4,0.],[.4,-.4,0.],
    ])
    # Same four geometric points, different native face roles.
    xyz=torch.cat((base,base),dim=0)[None]
    faces=torch.tensor([
        [0,2,1],[1,2,3],  # body front facing
        [4,5,6],[5,7,6],  # cape reverse facing
    ],dtype=torch.long)
    layer_ids=torch.tensor([0]*4+[1]*4,dtype=torch.long)
    layer_names=("body","cape")
    layer_roles=(("primary_form",),("outer_covering",))
    namespace=layer_namespace_digest_v1(layer_names,layer_roles)
    seed=build_tessa_seeded_surface_v1(
        iris_surface_xyz=xyz,iris_surface_faces=faces,
        component_ids_by_vertex=layer_ids,
        scene_sha256="a"*64,
        original_surface_sha256="b"*64,
        original_vertex_layer_ids=layer_ids,
        layer_names=layer_names,layer_roles=layer_roles,
        layer_namespace_sha256=namespace,
    )
    mesh=model.build_seeded_mechanics_candidate_v1(
        state,seed,
    )["mesh_candidate"]
    # Force a distinctive metric target for both layers; source importance
    # remains teacher-only. A global XYZ-nearest label would map both to body.
    learned_logits=torch.tensor(
        [[-7.]*4+[7.]*4],requires_grad=True,
    )
    mesh=replace(mesh,tessa_metric_logits=learned_logits)
    native_face_samples=NativeFaceLayerSamplesV1(
        vertices=xyz[0].clone(),faces=faces.clone(),
        face_layer_ids=torch.tensor([0,0,1,1],dtype=torch.long),
        face_component_ids=torch.tensor([
            int((source_component_ids if source_component_ids is not None
                else layer_ids)[0]),
            int((source_component_ids if source_component_ids is not None
                else layer_ids)[0]),
            int((source_component_ids if source_component_ids is not None
                else layer_ids)[4]),
            int((source_component_ids if source_component_ids is not None
                else layer_ids)[4]),
        ],dtype=torch.long),
        sample_face_ids=torch.tensor([0,0,0,1,2,2,2,3],dtype=torch.long),
        sample_barycentric=torch.tensor([
            [1.,0.,0.],[0.,0.,1.],[0.,1.,0.],[0.,0.,1.],
            [1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[0.,1.,0.],
        ]),
        source_asset_sha256="c"*64,
        layer_namespace_sha256=namespace,
    )
    teacher=SeededTESSASourceTeacherV1(
        native_face_layer_samples=native_face_samples,
        source_asset_sha256="c"*64,
        source_qualification_key="qualified-native-layer-test",
        source_xyz=xyz[0].clone(),
        source_normals=mesh.normals[0].detach().clone(),
        source_point_mask=torch.ones(8,dtype=torch.bool),
        source_importance=torch.tensor([0.]*4+[1.]*4),
        source_importance_mask=torch.ones(8,dtype=torch.bool),
        source_component_ids=(
            source_component_ids if source_component_ids is not None
            else layer_ids.clone()
        ),
        source_layer_ids=layer_ids.clone(),
    )
    receipt={
        "schema":SCHEMA,
        "admission_status":"QUALIFIED_INDEPENDENT_SOURCE_NATIVE_3D",
        "source_asset_sha256":teacher.source_asset_sha256,
        "source_alignment_identity":teacher.source_alignment_identity,
        "required_phase":PHASE,
        "source_rights_independently_verified":True,
        "generated_view_as_3d_teacher":False,
        "source_layer_partition_authority":SOURCE_LAYER_AUTHORITY,
        "source_layer_partition_derivation":SOURCE_LAYER_DERIVATION,
        "source_layer_namespace_sha256":namespace,
        "source_layer_native_asset_sha256":teacher.source_asset_sha256,
        "native_source_layer_partition_sha256":_digest(teacher.source_layer_ids),
        "source_layer_independently_audited":True,
        "source_label_student_checkpoint_dependency":False,
        "source_native_face_witness_schema":NATIVE_FACE_SCHEMA,
        "source_native_face_witness_sha256":
            native_face_samples.receipt_sha256(),
        "source_native_face_witness_derivation":
            "SOURCE_NATIVE_INDEXED_TRIANGLE_BARYCENTRIC_NOT_STUDENT",
        **{k+"_sha256":_digest(v) for k,v in teacher.tensor_fields().items()},
    }
    driver=RSSvNextSeededTESSATrainerV1(
        model,torch.optim.AdamW(model.parameters(),lr=1e-3),
        weights=weights or SeededTESSALossWeightsV1(),
        trusted_source_registry={
            teacher.source_qualification_key:receipt,
        },
    )
    return mesh,teacher,driver,learned_logits,receipt


def test_coincident_body_cape_no_cross_layer_normal_or_importance_borrowing():
    mesh,teacher,driver,logits,_=_setup()
    naive,_=_bounded_nearest_source_indices_v1(
        mesh.rest_vertices_iris[0],teacher.source_xyz,
    )
    # A global nearest-neighbor teacher *does* make this mistake.
    assert bool((
        mesh.seeded_surface.vertex_layer_ids
        !=teacher.source_layer_ids[naive]
    ).any())
    out=driver.objective(mesh,teacher,source_face_budget=4)
    assert int(out["qualified_source_layer_count"])==2
    assert out["normal"].item()<1.e-5
    assert out["structural_importance"].item()<0.01
    assert out["rest_surface"].isfinite()
    out["total"].backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum())>0


def test_missing_native_layer_truth_fails_even_with_perfect_positions():
    mesh,teacher,driver,_,_=_setup()
    with pytest.raises(ValueError,match="SOURCE_RECEIPT_REQUIRED"):
        driver.objective(
            mesh,replace(teacher,source_layer_ids=None),
            source_face_budget=4,
        )


def test_wrong_native_3d_component_mixing_is_not_a_source_layer():
    mesh,teacher,driver,_,_=_setup(
        source_component_ids=torch.zeros(8,dtype=torch.long),
    )
    with pytest.raises(ValueError,match="NATIVE_COMPONENT_CROSS_ROLE"):
        driver.objective(mesh,teacher,source_face_budget=4)


def test_spoofed_namespace_and_reassigned_ids_are_rejected():
    mesh,teacher,driver,_,receipt=_setup()
    fake=replace(
        mesh,seeded_surface=replace(
            mesh.seeded_surface,layer_namespace_sha256="f"*64,
        ),
    )
    with pytest.raises(ValueError,match="SOURCE_RECEIPT_REQUIRED"):
        driver.objective(fake,teacher,source_face_budget=4)
    swapped=replace(
        teacher,source_layer_ids=torch.tensor([1]*4+[0]*4),
    )
    with pytest.raises(ValueError,match="SOURCE_RECEIPT_DRIFT"):
        driver.objective(mesh,swapped,source_face_budget=4)
    changed=dict(receipt)
    changed["source_layer_independently_audited"]=False
    adversarial=RSSvNextSeededTESSATrainerV1(
        driver.model,driver.optimizer,weights=driver.weights,
        trusted_source_registry={teacher.source_qualification_key:changed},
    )
    with pytest.raises(ValueError,match="SOURCE_RECEIPT_REQUIRED"):
        adversarial.objective(mesh,teacher,source_face_budget=4)


def test_layered_open_boundary_and_pose_losses_require_distinct_reindex_proof():
    weights=replace(SeededTESSALossWeightsV1(),boundary=0.5)
    mesh,teacher,driver,_,_=_setup(weights=weights)
    with pytest.raises(ValueError,match="NATIVE_OPEN_BOUNDARY_WITNESS_REQUIRED"):
        driver.objective(mesh,teacher,source_face_budget=4)


def test_layered_open_cape_and_body_hem_supervision_is_source_native():
    """Open hems can train without a signed-solid hole fill or rig program."""
    weights=replace(SeededTESSALossWeightsV1(),boundary=0.5)
    mesh,teacher,driver,_,receipt=_setup(weights=weights)
    native=teacher.native_face_layer_samples
    assert native is not None
    edges=exact_open_boundary_edges_v1(native.faces)
    ids=torch.unique(edges.reshape(-1),sorted=True)
    witness=NativeLayerOpenBoundaryV1(
        boundary_edges=edges,boundary_vertex_ids=ids,
        open_layer_ids=torch.tensor([0,1],dtype=torch.long),
        source_asset_sha256=teacher.source_asset_sha256,
        layer_namespace_sha256=mesh.seeded_surface.layer_namespace_sha256,
    )
    boundary_xyz=native.vertices[ids].clone()
    qualified=replace(
        teacher,native_open_boundary=witness,
        source_boundary_xyz=boundary_xyz,
        source_boundary_mask=torch.ones(len(ids),dtype=torch.bool),
    )
    revised_registry={
        **receipt,
        "source_native_open_boundary_schema":OPEN_HEM_SCHEMA,
        "source_native_open_boundary_sha256":witness.receipt_sha256(),
        "source_native_open_boundary_derivation":OPEN_HEM_DERIVATION,
        **{k+"_sha256":_digest(v)
           for k,v in qualified.tensor_fields().items()},
    }
    admitted=RSSvNextSeededTESSATrainerV1(
        driver.model,driver.optimizer,weights=weights,
        trusted_source_registry={
            qualified.source_qualification_key:revised_registry,
        },
    )
    loss=admitted.objective(mesh,qualified,source_face_budget=4)
    assert bool(loss["boundary"].isfinite())
    assert float(loss["boundary"])<1.e-6
    assert int(loss["qualified_source_layer_count"])==2

    # Hem evidence is source-bound: cannot silently swap the cape and body.
    forged=replace(
        qualified,native_open_boundary=replace(
            witness,open_layer_ids=torch.tensor([1],dtype=torch.long),
        ),
    )
    with pytest.raises(ValueError,match="BOUNDARY_RECEIPT_DRIFT"):
        admitted.objective(mesh,forged,source_face_budget=4)
