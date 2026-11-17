from __future__ import annotations

"""Production trainer route: two observed RGBA, generated-six conditioned H(x),
one IRIS indexed seed -> TESSA metric -> exactly the StructuralTrainer optimizer.

Mock extracted triangles are a regression fixture, NOT a signed-source or
GPU/sparse extraction proof. Real pinned FlexiCubes is tested separately.
"""
from dataclasses import replace
import hashlib
import json

import pytest
import torch

from models.rss_vnext import (
    RSSvNextConfig, RSSvNextPatchEncoderForTests,
    RSSvNextUnifiedCharacterModel,
)
from models.rss_vnext.semantic_geometry_prior import RSSvNextSemanticLayerSpec
from models.rss_vnext.flexicubes_surface_evidence import (
    RSSvNextFlexiCubesSurfaceEvidence,
)
from models.rss_vnext.training_engine import (
    RSSvNextStructuralTrainer,RSSvNextStructuralTrainingBatch,
    RSSvNextTrainingWeights,source_surface_tensor_sha256,
)
from models.rss_vnext.training_seeded_tessa_v1 import (
    SCHEMA,PHASE,SeededTESSALossWeightsV1,
    SeededTESSASourceTeacherV1,_digest,
)
from models.rss_vnext.realsas_virtual_camera_policy_v1 import (
    target_canonical_and_six_cameras_v1,
)
from models.rss_vnext.realsas_two_source_construction_v1 import (
    paired_generation_binding_sha256_v1,
)


def _scenario(monkeypatch):
    torch.manual_seed(20261117)
    torch.set_num_threads(1)
    cfg=replace(
        RSSvNextConfig.tiny_for_test(),
        tessa_surface_requirement="REQUIRE_SIGNED_IRIS_SURFACE",
    )
    net=RSSvNextUnifiedCharacterModel(
        cfg,vision_encoder=RSSvNextPatchEncoderForTests(
            cfg.iris.image_token_dim,
        ),
    ).train()
    # Independently authored, consistently wound closed tetrahedron.
    xyz=torch.tensor([[.0,.0,.35],[-.27,-.2,-.2],[.27,-.2,-.2],[0.,.3,-.2]])
    faces=torch.tensor([[0,2,1],[0,3,2],[0,1,3],[1,2,3]],dtype=torch.long)
    normals=torch.nn.functional.normalize(xyz-xyz.mean(dim=0),dim=-1)
    role_count=cfg.semantic_role_schema().count

    def generated_signed_extraction(
        scene_triplanes, *,extractor,role_head,schema,layer,field_policy,
        high_resolution_training=False,
    ):
        return RSSvNextFlexiCubesSurfaceEvidence(
            xyz=xyz[None],faces=faces,
            local_features=net.iris.query(scene_triplanes,xyz[None])["H"],
            presentation_role_logits=torch.zeros(1,4,role_count),
            regularizer=xyz.new_zeros(()),
            sample_grid_xyz=xyz[None],
            signed_field_grid=xyz.new_tensor([[-1.,1.,1.,1.]]),
            cube_parameters=xyz.new_zeros(1,21),
            source_policy=field_policy,
            layer_name=layer.name,
        )
    monkeypatch.setattr(net.fc_surface_bridge,"forward",generated_signed_extraction)
    observed=torch.rand(1,2,4,16,16)
    # Teacher-only source cameras are never passed to H(x).
    cameras=torch.zeros(1,2,13)
    cameras[...,3]=cameras[...,7]=cameras[...,11]=cameras[...,12]=1
    canonical_cameras,six=target_canonical_and_six_cameras_v1(
        1,device="cpu",dtype=observed.dtype,
    )
    generated=torch.rand(1,6,4,16,16)
    generated_normals=torch.zeros(1,6,3,16,16)
    source_binding="c"*64
    canonical_ckpt="a"*64
    multiview_ckpt="b"*64
    combined=paired_generation_binding_sha256_v1(
        canonical_checkpoint_sha256=canonical_ckpt,
        multiview_checkpoint_sha256=multiview_ckpt,
        source_binding_sha256=source_binding,
    )
    layer=RSSvNextSemanticLayerSpec("all",cfg.semantic_role_schema().roles)
    source_asset_sha="d"*64
    native=SeededTESSASourceTeacherV1(
        source_asset_sha256=source_asset_sha,
        source_qualification_key="approved-asset-source-001",
        source_xyz=xyz.clone(),source_normals=normals,
        source_point_mask=torch.ones(4,dtype=torch.bool),
        source_importance=torch.tensor([0.1,0.8,0.6,0.2]),
        source_importance_mask=torch.ones(4,dtype=torch.bool),
    )
    manifest="f"*64
    source_qualification_key="qualified-source-001"
    signed_key="signed-closed-source-001"
    generated_key="pinned-six-generated-001"
    native_registry={native.source_qualification_key:{
        "schema":SCHEMA,
        "admission_status":"QUALIFIED_INDEPENDENT_SOURCE_NATIVE_3D",
        "source_asset_sha256":source_asset_sha,
        "source_alignment_identity":native.source_alignment_identity,
        "required_phase":PHASE,
        "source_rights_independently_verified":True,
        "generated_view_as_3d_teacher":False,
        "dataset_manifest_sha256":manifest,
        "source_qualification_key":source_qualification_key,
        "observed_rgba_sha256":source_surface_tensor_sha256(observed),
        "observed_camera_sha256":source_surface_tensor_sha256(cameras),
        "source_binding_sha256":source_binding,
        "combined_generation_checkpoint_sha256":combined,
        "signed_source_admission_key":signed_key,
        "semantic_layer_name":"all",
        **{k+"_sha256":_digest(v) for k,v in native.tensor_fields().items()},
    }}
    generated_registry={generated_key:{
        "admission_status":"PINNED_GENERATOR_OUTPUT_NOT_SOURCE_TEACHER",
        "dataset_manifest_sha256":manifest,
        "source_qualification_key":source_qualification_key,
        "source_observed_rgba_sha256":source_surface_tensor_sha256(observed),
        "original_camera_is_model_input":False,
        "generated_rgba_sha256":source_surface_tensor_sha256(generated),
        "generated_normals_sha256":source_surface_tensor_sha256(generated_normals),
        "generated_cameras_sha256":source_surface_tensor_sha256(six),
        "source_binding_sha256":source_binding,
        "canonical_stage_checkpoint_sha256":canonical_ckpt,
        "multiview_stage_checkpoint_sha256":multiview_ckpt,
        "combined_generation_checkpoint_sha256":combined,
        "generated_views_are_observed_source":False,
    }}
    signed_registry={signed_key:{
        "admission_status":"QUALIFIED_SOURCE_SIGNED_SOLID",
        "source_component_domain":"SINGLE_QUALIFIED_CLOSED_COMPONENT",
        "dataset_manifest_sha256":manifest,
        "source_qualification_key":source_qualification_key,
        "role_schema_id":cfg.semantic_role_schema_id,
        "layer_name":"all",
        "surface_target_sha256":source_surface_tensor_sha256(xyz[None]),
    }}
    batch=RSSvNextStructuralTrainingBatch(
        rgba=observed,camera_features=cameras,
        product_generated_rgba=generated,
        product_generated_normals=generated_normals,
        product_generated_cameras=six,
        product_source_binding_sha256=source_binding,
        product_canonical_checkpoint_sha256=canonical_ckpt,
        product_multiview_checkpoint_sha256=multiview_ckpt,
        product_generation_checkpoint_sha256=combined,
        product_conditioning_admission_key=generated_key,
        flexicubes_extractor=object(),
        flexicubes_layer=layer,
        flexicubes_reference_surface_xyz=xyz[None],
        flexicubes_signed_source_admission_key=signed_key,
        seeded_tessa_source_teacher=native,
        seeded_tessa_source_face_budget=4,
        source_qualification_key=source_qualification_key,
        admission_status="QUALIFIED_FOR_THIS_SUPERVISION_FAMILY",
    )
    weights=replace(
        SeededTESSALossWeightsV1(),boundary=0.0,
    )
    optimizer=torch.optim.AdamW(net.parameters(),lr=1.e-3)
    def trainer(native_registry=native_registry,
                signed_registry_override=None):
        return RSSvNextStructuralTrainer(
            net,optimizer,dataset_manifest_sha256=manifest,
            scene_conditioning_policy="PRODUCT_GENERATED_SIX_VIEW",
            training_profile="PRODUCT_SEEDED_IRIS_TESSA_V1",
            seeded_tessa_weights=weights,
            trusted_seeded_tessa_registry=native_registry,
            trusted_generated_conditioning_registry=generated_registry,
            trusted_signed_source_registry=(
                signed_registry if signed_registry_override is None
                else signed_registry_override
            ),
        )
    return net,batch,trainer,native_registry,generated_registry


def test_v1_seeded_product_training_step_and_optimizer_owner(monkeypatch):
    net,batch,make_trainer,*_=_scenario(monkeypatch)
    trainer=make_trainer()
    assert all(
        value == 0 for key,value in vars(trainer.weights).items()
        if key != "clip_gradient_norm"
    )
    before=net.tessa_seeded_head[-1].weight.detach().clone()
    result=trainer.train_step(batch)
    assert result.seeded_tessa_loss is not None
    assert result.seeded_tessa_loss.isfinite()
    assert result.topology_loss is None
    assert result.seeded_tessa_source_point_count==4
    assert result.gradient_norm.isfinite()
    assert not torch.equal(before,net.tessa_seeded_head[-1].weight)
    assert trainer.global_step==1


def test_bad_native_teacher_and_old_topology_cannot_train_product(monkeypatch):
    net,batch,make_trainer,*_=_scenario(monkeypatch)
    trainer=make_trainer()
    forged=replace(batch,topology_teacher_tokens=torch.ones(1,3,dtype=torch.long))
    with pytest.raises(ValueError,match="UNCONSUMED_SOURCE_TEACHER"):
        trainer.objective(forged)
    # Other teacher families may not be silently ignored either.
    unrelated=replace(batch,source_udf_target=torch.zeros(1,4))
    with pytest.raises(ValueError,match="UNCONSUMED_SOURCE_TEACHER"):
        trainer.objective(unrelated)
    teacher=replace(
        batch.seeded_tessa_source_teacher,
        source_importance=torch.zeros(4),
    )
    with pytest.raises(ValueError,match="SOURCE_RECEIPT_DRIFT"):
        trainer.objective(replace(batch,seeded_tessa_source_teacher=teacher))


def test_generated_scene_tamper_and_checkpoint_resume_policy(monkeypatch,tmp_path):
    net,batch,make_trainer,registry,*_=_scenario(monkeypatch)
    trainer=make_trainer()
    fake_views=batch.product_generated_rgba.clone()
    fake_views[0,0,0,0,0]=1-fake_views[0,0,0,0,0]
    with pytest.raises(ValueError,match="GENERATED_SCENE_RECEIPT_DRIFT"):
        trainer.objective(replace(batch,product_generated_rgba=fake_views))
    trainer.train_step(batch)
    path=tmp_path/"seeded.pt"
    trainer.save_checkpoint(path,sampler_cursor=7)
    assert make_trainer().load_checkpoint(path)==7
    changed={k:dict(v) for k,v in registry.items()}
    changed[next(iter(changed))]["source_rights_independently_verified"]=False
    other=make_trainer(changed)
    with pytest.raises(ValueError,match="OBJECTIVE_POLICY_DRIFT|REGISTRY"):
        other.load_checkpoint(path)


def test_product_phase_rejects_legacy_or_frozen_head(monkeypatch):
    net,batch,make_trainer,*_=_scenario(monkeypatch)
    trainer=make_trainer()
    assert trainer.training_profile=="PRODUCT_SEEDED_IRIS_TESSA_V1"
    with pytest.raises(ValueError,match="LEGACY_MAY_NOT_ACCEPT_SEEDED_TEACHER"):
        RSSvNextStructuralTrainer(
            net,trainer.optimizer,dataset_manifest_sha256="f"*64,
            trusted_seeded_tessa_registry={"fake":{"x":"y"}},
        )
    for p in net.tessa_seeded_head.parameters():
        p.requires_grad_(False)
    with pytest.raises(ValueError,match="GRADIENT_OWNERS_FROZEN"):
        make_trainer()


def test_layered_source_truth_executes_one_student_h_with_two_signed_receipts(
    monkeypatch,
):
    from models.rss_vnext.layered_iris_seed_v1 import (
        layer_namespace_digest_v1,
    )
    from models.rss_vnext.native_face_layer_truth_v1 import (
        SCHEMA as NATIVE_FACE_SCHEMA, NativeFaceLayerSamplesV1,
    )
    from models.rss_vnext.seeded_layer_source_court_v1 import (
        SOURCE_LAYER_AUTHORITY,SOURCE_LAYER_DERIVATION,
    )
    net,batch,make_trainer,registry,_=_scenario(monkeypatch)
    a=batch.seeded_tessa_source_teacher
    first=batch.flexicubes_reference_surface_xyz[0]
    source=torch.cat((first,first),dim=0)
    layer_ids=torch.tensor([0]*4+[1]*4,dtype=torch.long)
    layers=(
        RSSvNextSemanticLayerSpec("body",("primary_form","focal_detail")),
        RSSvNextSemanticLayerSpec("cape",("outer_covering",)),
    )
    namespace=layer_namespace_digest_v1(
        tuple(s.name for s in layers),
        tuple(s.keep_roles for s in layers),
    )
    native_triangles=torch.tensor([
        [0,2,1],[0,3,2],[0,1,3],[1,2,3],
        [4,6,5],[4,7,6],[4,5,7],[5,6,7],
    ],dtype=torch.long)
    native_face_witness=NativeFaceLayerSamplesV1(
        vertices=source.clone(),faces=native_triangles,
        face_layer_ids=torch.tensor([0]*4+[1]*4,dtype=torch.long),
        face_component_ids=torch.tensor([0]*4+[1]*4,dtype=torch.long),
        sample_face_ids=torch.tensor([0,0,0,1,4,4,4,5],dtype=torch.long),
        sample_barycentric=torch.tensor([
            [1.,0.,0.],[0.,0.,1.],[0.,1.,0.],[0.,1.,0.],
            [1.,0.,0.],[0.,0.,1.],[0.,1.,0.],[0.,1.,0.],
        ]),
        source_asset_sha256=a.source_asset_sha256,
        layer_namespace_sha256=namespace,
    )
    teacher=replace(
        a,native_face_layer_samples=native_face_witness,
        source_xyz=source,
        source_normals=torch.cat((a.source_normals,a.source_normals),dim=0),
        source_point_mask=torch.ones(8,dtype=torch.bool),
        source_importance=torch.tensor([0.1]*4+[0.9]*4),
        source_importance_mask=torch.ones(8,dtype=torch.bool),
        source_component_ids=layer_ids.clone(),
        source_layer_ids=layer_ids.clone(),
    )
    layer_keys=("signed-closed-source-001","signed-closed-source-002")
    native_row=dict(registry[a.source_qualification_key])
    native_row.update({
        "semantic_layer_name":"body",
        "source_layer_partition_authority":SOURCE_LAYER_AUTHORITY,
        "source_layer_partition_derivation":SOURCE_LAYER_DERIVATION,
        "source_layer_namespace_sha256":namespace,
        "source_layer_native_asset_sha256":teacher.source_asset_sha256,
        "native_source_layer_partition_sha256":_digest(teacher.source_layer_ids),
        "source_layer_independently_audited":True,
        "source_label_student_checkpoint_dependency":False,
        "source_native_face_witness_schema":NATIVE_FACE_SCHEMA,
        "source_native_face_witness_sha256":
            native_face_witness.receipt_sha256(),
        "source_native_face_witness_derivation":
            "SOURCE_NATIVE_INDEXED_TRIANGLE_BARYCENTRIC_NOT_STUDENT",
        "signed_layer_admission_keys":list(layer_keys),
        **{k+"_sha256":_digest(v) for k,v in teacher.tensor_fields().items()},
    })
    registry={teacher.source_qualification_key:native_row}
    signed={}
    for layer_id,(name,key) in enumerate(zip(layers,layer_keys)):
        signed[key]={
            "admission_status":"QUALIFIED_SOURCE_SIGNED_SOLID",
            "source_component_domain":"SINGLE_QUALIFIED_CLOSED_COMPONENT",
            "dataset_manifest_sha256":"f"*64,
            "source_qualification_key":batch.source_qualification_key,
            "role_schema_id":net.cfg.semantic_role_schema_id,
            "layer_name":name.name,
            "surface_target_sha256":source_surface_tensor_sha256(
                teacher.source_xyz[teacher.source_layer_ids==layer_id][None],
            ),
            "source_layer_partition_authority":SOURCE_LAYER_AUTHORITY,
            "source_layer_partition_derivation":SOURCE_LAYER_DERIVATION,
            "native_source_layer_partition_sha256":_digest(
                teacher.source_layer_ids,
            ),
            "source_layer_namespace_sha256":namespace,
            "source_layer_native_asset_sha256":teacher.source_asset_sha256,
            "native_source_layer_index":layer_id,
            "source_layer_independently_audited":True,
            "source_label_student_checkpoint_dependency":False,
        }
    layered_batch=replace(
        batch,flexicubes_layer=layers[0],
        flexicubes_layers=layers,
        flexicubes_reference_surface_xyz=source[None],
        flexicubes_signed_layer_admission_keys=layer_keys,
        seeded_tessa_source_teacher=teacher,
        seeded_tessa_source_face_budget=8,
    )
    trainer=make_trainer(
        native_registry=registry,signed_registry_override=signed,
    )
    out=trainer.train_step(layered_batch)
    assert out.seeded_tessa_qualified_layer_count==2
    assert out.seeded_tessa_source_point_count==8
    assert out.seeded_tessa_loss.isfinite()
    assert trainer.global_step==1
    bad={k:dict(v) for k,v in signed.items()}
    bad[layer_keys[1]]["source_layer_namespace_sha256"]="0"*64
    bad_trainer=make_trainer(
        native_registry=registry,signed_registry_override=bad,
    )
    with pytest.raises(ValueError,match="NATIVE_SIGNED_LAYER_RECEIPT_DRIFT"):
        bad_trainer.objective(layered_batch)
