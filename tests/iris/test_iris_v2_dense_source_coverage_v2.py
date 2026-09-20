from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import models.iris.v2.train_v2 as train_v2
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_services.orchestrator.adapters.iris_geometry_v2 import (
    _validate_dense_coverage_execution_metadata,
)


ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT / "canonical" / "IRIS_V2_DENSE_SOURCE_COVERAGE_POLICY_20260920.json"


def _domain(*, q: int = 16, view: int = 0):
    return SimpleNamespace(
        anchor_view=torch.full((1, q), view, dtype=torch.long),
        anchor_stride_px=256,
        construction_authority="RGB_CAMERA_FULL_FRAME_LATTICE_V1",
    )


def _output(probability: float, *, q: int = 16, modes: int = 2):
    p = torch.full(
        (1, q, modes),
        float(probability),
        dtype=torch.float32,
        requires_grad=True,
    )
    return SimpleNamespace(
        depth_output=SimpleNamespace(support_probability=p)
    )


def test_dense_source_coverage_loss_rewards_foreground_support_and_backpropagates():
    masks = torch.ones((1, 8, 1024, 1024), dtype=torch.bool)
    high = _output(0.98)
    low = _output(0.05)

    high_loss = train_v2._dense_source_coverage_loss(high, _domain(), masks)
    low_loss = train_v2._dense_source_coverage_loss(low, _domain(), masks)

    assert torch.isfinite(high_loss)
    assert torch.isfinite(low_loss)
    assert float(high_loss.detach()) < float(low_loss.detach())

    low_loss.backward()
    grad = low.depth_output.support_probability.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert float(grad.mean()) < 0.0


def test_dense_source_coverage_loss_uses_selected_anchor_view_target():
    masks = torch.zeros((1, 8, 1024, 1024), dtype=torch.bool)
    masks[:, 3] = True
    output = _output(0.95)
    matching = train_v2._dense_source_coverage_loss(
        output,
        _domain(view=3),
        masks,
    )
    background = train_v2._dense_source_coverage_loss(
        _output(0.95),
        _domain(view=2),
        masks,
    )
    assert float(matching.detach()) < float(background.detach())


def test_production_train_step_requires_dense_target_but_never_forwards_it(monkeypatch):
    parameter = torch.nn.Parameter(torch.tensor(1.0))

    class Learner:
        def parameters(self):
            yield parameter

    class Apparatus:
        runtime_seal = object()
        source_contract_hash = "sealed"
        learner = Learner()

        def __init__(self):
            self.forward_args = None

        def train(self):
            return self

        def __call__(self, images, domain):
            self.forward_args = (images, domain)
            return SimpleNamespace()

    class Optimizer:
        def zero_grad(self, set_to_none=True):
            parameter.grad = None

        def step(self):
            pass

    monkeypatch.setattr(
        train_v2,
        "require_cuda_forward_live_set_v2",
        lambda apparatus, domain: SimpleNamespace(
            forward_live_lower_bound_gib=0.0
        ),
    )

    seen = {}

    def fake_loss(output, domain, teacher_depth, teacher_support, **kwargs):
        seen["masks"] = kwargs.get("source_foreground_masks")
        return {
            "total": parameter * 0.0 + 1.0,
            "dense_source_coverage": parameter * 0.0,
        }

    monkeypatch.setattr(train_v2, "iris_v2_loss", fake_loss)

    apparatus = Apparatus()
    domain = object()
    images = torch.zeros((1, 8, 3, 4, 4))
    masks = torch.zeros((1, 8, 1024, 1024), dtype=torch.bool)
    batch = {
        "images": images,
        "domain": domain,
        "teacher_depth": torch.zeros((1, 1)),
        "teacher_support": torch.ones((1, 1), dtype=torch.bool),
        "source_foreground_masks": masks,
    }
    train_v2.train_step_production_v2(apparatus, Optimizer(), batch)
    assert apparatus.forward_args == (images, domain)
    assert seen["masks"] is masks

    missing = dict(batch)
    missing.pop("source_foreground_masks")
    with pytest.raises(
        ValueError,
        match="requires dense source foreground targets",
    ):
        train_v2.train_step_production_v2(
            Apparatus(),
            Optimizer(),
            missing,
        )


def test_frozen_policy_matches_training_implementation_constants():
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    impl = train_v2.DENSE_SOURCE_COVERAGE_POLICY_V2
    assert policy["target_authority"] == impl["target_authority"]
    assert policy["target_resolution"] == impl["target_resolution"]
    assert policy["production_q_domain_authority"] == "RGB_CAMERA_FULL_FRAME_LATTICE_V1"
    assert policy["mask_is_forward_input"] is impl["mask_is_forward_input"] is False
    assert policy["loss"]["bce_weight"] == impl["bce_weight"]
    assert policy["loss"]["soft_dice_weight"] == impl["soft_dice_weight"]
    assert policy["loss"]["boundary_multiplier"] == impl["boundary_multiplier"]
    assert policy["loss"]["boundary_radius_px"] == impl["boundary_radius_px"]


def test_execution_receipt_requires_all_eight_balanced_dense_supervised_views():
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    good = {
        "dense_source_coverage": {
            "policy_sha256": "a" * 64,
            "source_foreground_masks_forward_input": False,
            "target_resolution": 1024,
            "q_domain_authority": "RGB_CAMERA_FULL_FRAME_LATTICE_V1",
            "view_schedule": "ROUND_ROBIN_V0_TO_V7",
            "anchor_view_step_counts": [20] * 8,
            "reported_loss_terms": [
                "depth_mode",
                "refined_depth",
                "support",
                "dense_source_coverage",
            ],
        }
    }
    report = _validate_dense_coverage_execution_metadata(
        good,
        policy_sha256="a" * 64,
        policy=policy,
    )
    assert report["total_dense_supervised_steps"] == 160
    assert report["maximum_step_imbalance"] == 0

    missing_view = json.loads(json.dumps(good))
    missing_view["dense_source_coverage"]["anchor_view_step_counts"][-1] = 0
    with pytest.raises(
        QualificationError,
        match="VIEW_COVERAGE_INCOMPLETE",
    ):
        _validate_dense_coverage_execution_metadata(
            missing_view,
            policy_sha256="a" * 64,
            policy=policy,
        )

    leaking = json.loads(json.dumps(good))
    leaking["dense_source_coverage"]["source_foreground_masks_forward_input"] = True
    with pytest.raises(QualificationError, match="MASK_LEAK"):
        _validate_dense_coverage_execution_metadata(
            leaking,
            policy_sha256="a" * 64,
            policy=policy,
        )
