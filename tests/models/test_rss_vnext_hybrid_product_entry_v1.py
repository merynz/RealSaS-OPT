from __future__ import annotations

"""Real two-stage builder and raw Shape policy consume the SAME hybrid mode.

These are policy/transit tests with fake pinned generation weights, not a
source-native garment or trained GPU release proof.
"""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from compiler.rss_vnext.workers.rss_vnext_shape_executor_v4 import (
    _raw_shape_hybrid_open_policy_v1,
)
from models.rss_vnext.product_hybrid_open_seed_v1 import (
    ProductHybridOpenM0PolicyV1,
)
from models.rss_vnext.semantic_geometry_prior import RSSvNextSemanticLayerSpec
from models.rss_vnext.realsas_two_source_construction_v1 import (
    run_two_source_neural_construction_v1,
)
from models.rss_vnext.realsas_cleanroom_diffusion_training_v1 import (
    STAGE_CANONICAL,STAGE_MULTIVIEW,
)
from tests.models.test_rss_vnext_two_stage_generation_v1 import (
    stage,source_cameras,six_cameras,
)
from models.rss_vnext import (
    RSSvNextConfig,RSSvNextPatchEncoderForTests,
    RSSvNextUnifiedCharacterModel,
)


def _product_model():
    """Use the production required-signed topology policy, not tiny ablation."""
    cfg=replace(
        RSSvNextConfig.tiny_for_test(),
        constructed_total_views=0,
        tessa_surface_requirement="REQUIRE_SIGNED_IRIS_SURFACE",
    )
    return RSSvNextUnifiedCharacterModel(
        cfg,vision_encoder=RSSvNextPatchEncoderForTests(
            cfg.iris.image_token_dim,
        ),
    ).eval()


def _policy_request():
    return {
        "hybrid_open_policy":{
            "open_layer_name":"cape",
            "open_presentation_roles":["outer_covering"],
            "resolution":[12,12,12],
            "query_chunk":256,
        },
        "hybrid_signed_layer":{
            "name":"body","keep_roles":["primary_form"],
        },
    }


def test_shape_policy_typed_and_model_only_with_no_source_part_hint():
    net=_product_model()
    open_policy,signed=_raw_shape_hybrid_open_policy_v1(
        _policy_request(),net,
    )
    assert open_policy.open_layer_name=="cape"
    assert open_policy.open_presentation_roles==("outer_covering",)
    assert open_policy.resolution==(12,12,12)
    assert signed==RSSvNextSemanticLayerSpec(
        "body",("primary_form",),
    )
    assert _raw_shape_hybrid_open_policy_v1({},net)==(None,None)


@pytest.mark.parametrize("bad",[
    {"hybrid_open_policy":{}},
    {"hybrid_open_policy":{"source_part_index":0},
     "hybrid_signed_layer":{"name":"body","keep_roles":["primary_form"]}},
    {"hybrid_open_policy":{"open_layer_name":"body"},
     "hybrid_signed_layer":{"name":"body","keep_roles":["primary_form"]}},
    {"hybrid_open_policy":{"open_presentation_roles":["primary_form"]},
     "hybrid_signed_layer":{"name":"body","keep_roles":["primary_form"]}},
    {"hybrid_open_policy":{"min_open_probability":1.5},
     "hybrid_signed_layer":{"name":"body","keep_roles":["primary_form"]}},
])
def test_shape_rejects_partial_or_forged_hybrid_source_hints(bad):
    with pytest.raises(RuntimeError,match="RSS_VNEXT_SHAPE_HYBRID_POLICY"):
        _raw_shape_hybrid_open_policy_v1(bad,_product_model())


def test_pinned_two_stage_routes_open_policy_to_same_model_without_teacher(tmp_path,monkeypatch):
    torch.manual_seed(1608)
    torch.set_num_threads(1)
    net=_product_model()
    signed=RSSvNextSemanticLayerSpec("body",("primary_form",))
    policy=ProductHybridOpenM0PolicyV1(
        open_layer_name="cape",open_presentation_roles=("outer_covering",),
        resolution=(12,12,12),
    )
    canon=stage(STAGE_CANONICAL)
    multi=stage(STAGE_MULTIVIEW)
    for name,obj in (("canon",canon),("multi",multi)):
        location=tmp_path/(name+".pt")
        sha=obj.save_training_weights(location)
        obj.load_pinned_task_weights(location,expected_sha256=sha)
    captured={}
    def propose(rgba,cameras,**kwargs):
        captured.update(kwargs)
        return {"state":SimpleNamespace(
            scene_conditioning_authority="GENERATED_SIX_VIEW_SHARED_SCENE",
            generated_view_count=6,
            observed_image_tokens=torch.zeros(1,2,1,8),
        )}
    monkeypatch.setattr(net,"propose_teacher_free",propose)
    result=run_two_source_neural_construction_v1(
        model=net,
        observed_rgba=torch.rand(1,2,4,16,16),
        original_camera_features=source_cameras(),
        source_binding_sha256="a"*64,
        canonical_cameras=six_cameras()[:,[0,3]],
        six_view_cameras=six_cameras(),
        canonical_sampler=canon,multiview_sampler=multi,
        image_size=16,denoise_steps=1,
        canonical_seed=10,multiview_seed=20,
        flexicubes_extractor=object(),
        hybrid_open_policy=policy,hybrid_signed_layer=signed,
    )
    assert result.original_observed_view_count==2
    assert result.generated_view_count==6
    assert result.product_revision_minted is False
    assert captured["hybrid_open_policy"] is policy
    assert captured["flexicubes_layer"]==signed
    assert captured["flexicubes_layers"] is None
    assert captured["generation_checkpoint_sha256"]==(
        result.construction_checkpoint_binding_sha256
    )


def test_two_stage_hybrid_never_uses_unpinned_or_implicit_signed_layer():
    net=_product_model()
    policy=ProductHybridOpenM0PolicyV1()
    with pytest.raises(ValueError,match="HYBRID_EXPLICIT_POLICY_REQUIRED"):
        run_two_source_neural_construction_v1(
            model=net,observed_rgba=torch.rand(1,2,4,16,16),
            original_camera_features=source_cameras(),
            source_binding_sha256="a"*64,
            canonical_cameras=six_cameras()[:,[0,3]],
            six_view_cameras=six_cameras(),
            canonical_sampler=stage(STAGE_CANONICAL),
            multiview_sampler=stage(STAGE_MULTIVIEW),
            image_size=16,denoise_steps=1,
            canonical_seed=10,multiview_seed=20,
            hybrid_open_policy=policy,
        )
