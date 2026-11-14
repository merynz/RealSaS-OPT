from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from models.rss_vnext.mira_embedded_codec_source_training_v1 import (
    RSSvNextEmbeddedMIRASourceCodecTrainerV1,
)
from models.rss_vnext.training_engine import (
    axis_mesh_teacher_address_digest_v1,
    source_surface_tensor_sha256,
)
from tests.models.test_rss_vnext_mira_embedded_codec_source_training_v1 import (
    DATASET,_fixture,
)


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(283)
    torch.set_num_threads(1)


def source_case():
    model,batch,skin_registry=_fixture()
    with torch.no_grad():
        mesh=model.build_mesh_candidate(
            batch.scene,batch.topology_teacher_tokens[0],
            source_authority="TEACHER_FORCED_TRAINING_ONLY",
        )
    rest=mesh.rest_vertices_iris.detach().clone()
    faces=mesh.faces.detach().clone()
    mask=torch.ones((faces.shape[0],),dtype=torch.bool)
    if faces.shape[0]>1:
        mask[0]=False
    batch=replace(
        batch,source_area_rest_vertices=rest,
        source_area_faces=faces,
        source_area_face_mask=mask,
        source_area_admission_key="qualified-source-surface",
    )
    receipt={"qualified-source-surface":{
        "admission_status":"QUALIFIED_SOURCE_TESSA_REINDEXED_SKIN_AREA",
        "dataset_manifest_sha256":DATASET,
        "source_qualification_key":batch.source_qualification_key,
        "source_binding_sha256":batch.source_binding_sha256,
        "scene_generator_binding_sha256":batch.scene.generating_checkpoint_sha256,
        "mesh_teacher_address_digest":axis_mesh_teacher_address_digest_v1(mesh),
        "source_dense_weights_sha256":source_surface_tensor_sha256(batch.source_dense_weights),
        "source_rest_vertices_sha256":source_surface_tensor_sha256(rest),
        "source_faces_sha256":source_surface_tensor_sha256(faces),
        "source_face_mask_sha256":source_surface_tensor_sha256(mask),
        "reindexed_tessa_vertices_and_faces_proven":True,
        "source_area_authority":"INDEPENDENT_SOURCE_NATIVE_REST_MESH",
        "generated_view_is_source_area_teacher":False,
    }}
    return model,batch,skin_registry,receipt


def trainer(model,skin_registry,area_registry,area_weight=1.):
    return RSSvNextEmbeddedMIRASourceCodecTrainerV1(
        model,torch.optim.AdamW(model.mira.codec.parameters(),lr=1e-3),
        dataset_manifest_sha256=DATASET,
        trusted_source_skin_registry=skin_registry,
        trusted_source_area_registry=area_registry,
        source_area_skin_weight=area_weight,
    )


def test_source_native_faces_contribute_gradients_to_exact_embedded_codec():
    model,batch,skin,area=source_case()
    engine=trainer(model,skin,area)
    result=engine.train_step(batch)
    assert result.source_area_skin_loss is not None
    assert torch.isfinite(result.source_area_skin_loss)
    assert float(result.source_area_skin_loss)>0
    assert result.qualified_source_area_faces==int(batch.source_area_face_mask.sum())
    assert result.qualified_source_surface_area>0
    assert result.hard_decoder_replay_max_abs_error<=1e-5
    assert model.mira.codec.bias_head.weight.grad is not None
    assert model.mira.codec.bias_head.weight.grad.abs().sum()>0.
    assert model.axis.kind_head.weight.grad is None
    assert not result.product_skin_qualified


def test_area_quadrature_is_a_real_separate_loss_not_a_report_only():
    model,batch,skin,area=source_case()
    ignored=replace(
        batch,source_area_rest_vertices=None,source_area_faces=None,
        source_area_face_mask=None,source_area_admission_key="",
    )
    without=trainer(model,skin,{},area_weight=0.)
    model.mira.codec.zero_grad(set_to_none=True)
    base=without.objective(ignored)
    base.loss.backward()
    g0=model.mira.codec.bias_head.weight.grad.detach().clone()
    model.mira.codec.zero_grad(set_to_none=True)
    with_area=trainer(model,skin,area,area_weight=3.).objective(batch)
    with_area.loss.backward()
    g1=model.mira.codec.bias_head.weight.grad.detach().clone()
    assert not torch.allclose(g0,g1,atol=1e-8)
    assert float(with_area.loss)>float(base.loss)


def test_predicted_tessa_mesh_cannot_replace_independent_source_face_teacher():
    model,batch,skin,area=source_case()
    altered=batch.source_area_faces.clone()
    altered[0]=altered[0].roll(1)
    signed={"qualified-source-surface":{
        **area["qualified-source-surface"],
        "source_faces_sha256":source_surface_tensor_sha256(altered),
    }}
    with pytest.raises(ValueError,match="SOURCE_AREA_FACE_REINDEX_DRIFT"):
        trainer(model,skin,signed).objective(
            replace(batch,source_area_faces=altered),
        )
    missing=replace(batch,source_area_rest_vertices=None)
    with pytest.raises(ValueError,match="SOURCE_AREA_INCOMPLETE"):
        trainer(model,skin,area).objective(missing)


def test_unqualified_source_area_and_missing_coverage_fail_closed():
    model,batch,skin,area=source_case()
    forged={"qualified-source-surface":{
        **area["qualified-source-surface"],
        "generated_view_is_source_area_teacher":True,
    }}
    with pytest.raises(ValueError,match="SOURCE_AREA_RECEIPT_DRIFT"):
        trainer(model,skin,forged).objective(batch)
    missing_mask=batch.source_area_face_mask.clone()
    missing_mask[:]=False
    signed={"qualified-source-surface":{
        **area["qualified-source-surface"],
        "source_face_mask_sha256":source_surface_tensor_sha256(missing_mask),
    }}
    with pytest.raises(ValueError,match="NO_ELIGIBLE_FACES"):
        trainer(model,skin,signed).objective(
            replace(batch,source_area_face_mask=missing_mask),
        )
    with pytest.raises(ValueError,match="SOURCE_AREA_POLICY"):
        trainer(model,skin,area,area_weight=-1.)


def test_source_area_teacher_does_not_receive_gradient_and_is_checkpoint_bound(tmp_path):
    model,batch,skin,area=source_case()
    rest=batch.source_area_rest_vertices.detach().clone().requires_grad_(True)
    weights=batch.source_dense_weights.detach().clone().requires_grad_(True)
    changed=replace(
        batch,source_area_rest_vertices=rest,
        source_dense_weights=weights,
    )
    # Same bytes, differing autograd metadata: valid receipt with no gradient
    # authority over source tensors.
    engine=trainer(model,skin,area)
    result=engine.train_step(changed)
    assert result.source_area_skin_loss is not None
    assert rest.grad is None
    assert weights.grad is None
    path=tmp_path/"mira-area-stage.pt"
    engine.save_checkpoint(path)
    clone=trainer(model,skin,area)
    clone.load_checkpoint(path)
    assert clone.step==1
    no_area=trainer(model,skin,{},area_weight=0.)
    with pytest.raises(ValueError,match="SOURCE_CODEC_RESUME_DRIFT"):
        no_area.load_checkpoint(path)
