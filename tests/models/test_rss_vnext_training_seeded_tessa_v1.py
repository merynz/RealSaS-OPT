from __future__ import annotations

"""Source-qualification/gradient contract for new non-token seeded TESSA.

Authored square is a regression fixture; not a real 21k corpus teacher.
"""
from dataclasses import replace

import pytest
import torch

from models.rss_vnext import (
    RSSvNextConfig,RSSvNextPatchEncoderForTests,RSSvNextUnifiedCharacterModel,
)
from models.rss_vnext.tessa_seeded_surface_v1 import (
    build_tessa_seeded_surface_v1,
)
from models.rss_vnext.training_seeded_tessa_v1 import (
    SCHEMA,PHASE,SeededTESSASourceTeacherV1,SeededTESSALossWeightsV1,
    RSSvNextSeededTESSATrainerV1,_digest,
)


def _fixture():
    torch.manual_seed(390)
    torch.set_num_threads(1)
    cfg=RSSvNextConfig.tiny_for_test()
    net=RSSvNextUnifiedCharacterModel(
        cfg,vision_encoder=RSSvNextPatchEncoderForTests(
            cfg.iris.image_token_dim,
        ),
    ).train()
    rgba=torch.rand(1,2,4,8,8)
    cameras=torch.zeros(1,2,13)
    cameras[...,3]=cameras[...,7]=cameras[...,11]=cameras[...,12]=1
    state,_,_=net.encode_observations(rgba,cameras)
    xyz=torch.tensor([[[-.4,.4,0.],[.4,.4,0.],
                      [-.4,-.4,0.],[.4,-.4,0.]]])
    faces=torch.tensor([[0,2,1],[1,2,3]],dtype=torch.long)
    seed=build_tessa_seeded_surface_v1(
        iris_surface_xyz=xyz,iris_surface_faces=faces,
        component_ids_by_vertex=torch.zeros(4,dtype=torch.long),
        scene_sha256="a"*64,original_surface_sha256="b"*64,
    )
    m=net.build_seeded_mechanics_candidate_v1(state,seed)["mesh_candidate"]
    normals=torch.tensor([[0.,0.,1.]]).expand(4,3).contiguous()
    teacher=SeededTESSASourceTeacherV1(
        source_asset_sha256="c"*64,
        source_qualification_key="SOURCE_UNIT_01",
        source_xyz=xyz[0].clone(),source_normals=normals,
        source_point_mask=torch.ones(4,dtype=torch.bool),
        source_importance=torch.tensor([0.2,0.85,0.65,0.1]),
        source_importance_mask=torch.ones(4,dtype=torch.bool),
        source_boundary_xyz=xyz[0].clone(),
        source_boundary_mask=torch.ones(4,dtype=torch.bool),
    )
    registry={"SOURCE_UNIT_01":{
        "schema":SCHEMA,
        "admission_status":"QUALIFIED_INDEPENDENT_SOURCE_NATIVE_3D",
        "source_asset_sha256":teacher.source_asset_sha256,
        "source_alignment_identity":teacher.source_alignment_identity,
        "required_phase":PHASE,
        "source_rights_independently_verified":True,
        "generated_view_as_3d_teacher":False,
        **{k+"_sha256":_digest(v) for k,v in teacher.tensor_fields().items()},
    }}
    weights=replace(SeededTESSALossWeightsV1(),boundary=0.5)
    opt=torch.optim.AdamW(net.parameters(),lr=1e-3)
    trainer=RSSvNextSeededTESSATrainerV1(
        net,opt,weights=weights,trusted_source_registry=registry,
    )
    return net,m,teacher,trainer


def test_seeded_tessa_objective_gradient_reaches_real_learned_metric_and_iris():
    net,mesh,teacher,trainer=_fixture()
    parts=trainer.objective(mesh,teacher,source_face_budget=2)
    assert set(parts)>={
        "rest_surface","normal","boundary","quality",
        "face_budget","structural_importance","total",
    }
    assert all(bool(torch.isfinite(v).all()) for v in parts.values())
    parts["total"].backward()
    head=net.tessa_seeded_head[-1]
    assert head.weight.grad is not None
    assert head.weight.grad.abs().sum()>0
    assert net.iris.scene.scene_queries.grad is not None
    assert net.iris.scene.scene_queries.grad.abs().sum()>0
    assert net.surface_mechanics_context.conditioner[-1].weight.grad is None


def test_missing_source_label_or_tampered_mask_fail_closed():
    _,mesh,teacher,trainer=_fixture()
    corrupted=replace(
        teacher,source_importance_mask=torch.zeros_like(
            teacher.source_importance_mask,
        ),
    )
    with pytest.raises(ValueError,match="SOURCE_SURFACE_DOMAIN"):
        trainer.objective(mesh,corrupted,source_face_budget=2)
    altered=replace(
        teacher,source_importance=torch.ones_like(teacher.source_importance),
    )
    with pytest.raises(ValueError,match="SOURCE_RECEIPT_DRIFT"):
        trainer.objective(mesh,altered,source_face_budget=2)


def test_cannot_train_fabricated_pose_or_treat_legacy_mesh_as_new_tessa():
    net,mesh,teacher,_=_fixture()
    weights=replace(SeededTESSALossWeightsV1(),deformation=2.)
    trainer=RSSvNextSeededTESSATrainerV1(
        net,torch.optim.AdamW(net.parameters(),lr=1e-3),
        weights=weights,
        trusted_source_registry={
            teacher.source_qualification_key:{
                "schema":SCHEMA,
                "admission_status":"QUALIFIED_INDEPENDENT_SOURCE_NATIVE_3D",
                "source_asset_sha256":teacher.source_asset_sha256,
                "source_alignment_identity":teacher.source_alignment_identity,
                "required_phase":PHASE,
                "source_rights_independently_verified":True,
                "generated_view_as_3d_teacher":False,
                **{k+"_sha256":_digest(v) for k,v in teacher.tensor_fields().items()},
            },
        },
    )
    with pytest.raises(ValueError,match="POSE_TRUTH_REQUIRED"):
        trainer.objective(mesh,teacher,source_face_budget=2)
    from dataclasses import replace as rep
    with pytest.raises(ValueError,match="SEEDED_MESH_REQUIRED"):
        trainer.objective(
            rep(mesh,seeded_surface=None),
            teacher,source_face_budget=2,
        )


def test_bounded_source_nearest_indices_equal_dense_without_losing_ambiguous_ties():
    from models.rss_vnext.training_seeded_tessa_v1 import (
        _bounded_nearest_source_indices_v1,
    )
    torch.manual_seed(111)
    pred=torch.randn(522,3)
    source=torch.randn(1055,3)
    dense=torch.cdist(pred,source)
    expected_dist,expected_ids=dense.min(dim=1)
    ids,dist=_bounded_nearest_source_indices_v1(pred,source)
    torch.testing.assert_close(ids,expected_ids)
    torch.testing.assert_close(dist,expected_dist,atol=1e-5,rtol=1e-5)
    # Exact tie split over different source tiles chooses smaller ID.
    source[1050]=source[2]
    pred[0]=source[2]
    ids,_=_bounded_nearest_source_indices_v1(pred,source)
    assert int(ids[0])==2
