from __future__ import annotations

"""Real product-seeded objective wiring with mock source receipts (not corpus proof)."""
import pytest
import torch

from models.rss_vnext.product_multitask_gradient_control_v1 import (
    ProductMultitaskGradientPolicyV1,
)
from models.rss_vnext.product_multitask_curriculum_v1 import (
    ProductSeededCurriculumV1,
)
from tests.models.test_rss_vnext_product_graphics_source_training_v1 import (
    _appearance,
)


@pytest.fixture(autouse=True)
def reproducible_cpu():
    torch.manual_seed(20261120)
    torch.set_num_threads(1)


def test_live_seeded_geometry_plus_graphics_uses_same_shared_H(monkeypatch):
    net, batch, make, *_ = _appearance(monkeypatch)
    policy = ProductMultitaskGradientPolicyV1(
        mode="BOUNDED_SHARED_PCGRAD",
        shared_owner_prefixes=("iris.scene.",),
        max_shared_elements=2_000_000,
    )
    trainer = make(gradient_route_policy=policy)
    vanilla = make()
    assert trainer.objective_policy_sha256 != vanilla.objective_policy_sha256
    result = trainer.train_step(batch)
    assert result.product_gradient_route_report is not None
    report = result.product_gradient_route_report
    assert report["schema"] == (
        "RealSaS.RSSvNext.ProductMultitaskGradientRouting.v1"
    )
    assert set(report["active_shared_task_families"]) == {
        "GEOMETRY", "GRAPHICS", "JOINT_RENDER"
    }
    assert report["task_private_gradients_preserved"]
    assert report["first_order_total_descent_preserved"]
    assert report["product_release_qualified"] is False
    assert trainer.global_step == 1
    assert torch.isfinite(result.total_loss)
    assert any(
        p.grad is not None and bool(torch.any(p.grad != 0))
        for p in net.graphics.parameters()
    )


def test_policy_sha_prevents_resuming_into_a_different_router(monkeypatch, tmp_path):
    _, batch, make, *_ = _appearance(monkeypatch)
    policy = ProductMultitaskGradientPolicyV1(
        mode="BOUNDED_SHARED_PCGRAD",
        shared_owner_prefixes=("iris.scene.",),
        every_n_steps=2,
    )
    trainer = make(gradient_route_policy=policy)
    trainer.train_step(batch)
    path = tmp_path / "source-pinned-router.pt"
    trainer.save_checkpoint(path, sampler_cursor=3)
    assert make(gradient_route_policy=policy).load_checkpoint(path) == 3
    with pytest.raises(ValueError, match="OBJECTIVE_POLICY_DRIFT"):
        make().load_checkpoint(path)


def test_reject_non_seeded_usage_before_optimizer_step(monkeypatch):
    from tests.models.test_rss_vnext_structural_trainer import make_sample
    from models.rss_vnext.training_engine import RSSvNextStructuralTrainer
    from models.rss_vnext import RSSvNextConfig, RSSvNextPatchEncoderForTests
    from models.rss_vnext.unified_character_model import RSSvNextUnifiedCharacterModel

    cfg = RSSvNextConfig.tiny_for_test()
    net = RSSvNextUnifiedCharacterModel(
        cfg, vision_encoder=RSSvNextPatchEncoderForTests(
            cfg.iris.image_token_dim
        ),
    )
    optimizer = torch.optim.AdamW(net.parameters(), lr=1.e-3)
    with pytest.raises(ValueError, match="PRODUCT_SEEDED_ONLY"):
        RSSvNextStructuralTrainer(
            net, optimizer, dataset_manifest_sha256="a"*64,
            product_gradient_route_policy=ProductMultitaskGradientPolicyV1(
                mode="BOUNDED_SHARED_PCGRAD"
            ),
        )


def test_foundry_requirements_are_exactly_loss_activated_not_product_truth(
    monkeypatch,
):
    _, batch, make, *_ = _appearance(monkeypatch)
    trainer = make()
    manifest = trainer.source_supervision_requirements_v1()
    families = manifest["loss_required_source_families"]
    assert manifest["objective_policy_sha256"] == trainer.objective_policy_sha256
    assert manifest["model_release_ready"] is False
    assert manifest["product_proof_ready"] is False
    assert "seeded_mstar_surface" in families
    assert "original_two_view_rgba" in families
    assert "generated_six_view_conditioning" in families
    assert "original_visible_rgb" in families
    assert "source_3d_ray_rgb_alpha_layers" in families
    assert "native_axis_rig" not in families
    assert "native_lbs_pose_consequence" not in families
    assert "PRODUCT_MSTAR_STAGE3_NATIVE_MOVING_RGBA_RASTER_NOT_CONNECTED" in (
        manifest["compiler_acceptance_gaps"]
    )


def test_curriculum_stage_and_checkpoint_fingerprint_cannot_be_overclaimed(
    monkeypatch,
):
    _, _, make, *_ = _appearance(monkeypatch)
    with pytest.raises(ValueError, match="OBJECTIVE_STAGE_MISMATCH"):
        make(curriculum=ProductSeededCurriculumV1(
            stage="IRIS_GEOMETRY_SLR_GRAPHICS",
        ))
    with pytest.raises(ValueError, match="OBJECTIVE_STAGE_MISMATCH"):
        make(curriculum=ProductSeededCurriculumV1(
            stage="FULL_SHARED_H_FINETUNE",
        ))
    a = make()
    b = make(curriculum=ProductSeededCurriculumV1(
        stage="COMPATIBILITY_UNSCHEDULED",
    ))
    assert a.objective_policy_sha256 == b.objective_policy_sha256
