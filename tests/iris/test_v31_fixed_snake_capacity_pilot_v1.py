import json
from pathlib import Path

import torch

from models.iris.v31.fixed_snake_capacity_pilot_v1 import (
    ARCHITECTURE_ID,
    FIXED_FOLD_ID,
    FIXED_HELDOUT,
    IRISCapacityPilotConfigV1,
    MECHANICS_TARGET_TRANSFORM,
    mechanics_target_log1p_v1,
)
from models.iris.v31.model_v31 import IRISV31


ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / "canonical/IRIS_V31_FIXED_SNAKE_CAPACITY_PILOT_V1.json"


def test_capacity_pilot_exact_parameter_count_and_fixed_holdout():
    recipe = json.loads(RECIPE.read_text())
    cfg = dict(recipe["model_config"])
    cfg["mechanical_channels"] = tuple(cfg["mechanical_channels"])
    cfg["pair_channels"] = tuple(cfg["pair_channels"])
    config = IRISCapacityPilotConfigV1(**cfg)
    config.validate()
    model = IRISV31(config)
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert count == 9_193_710
    assert count == recipe["capacity_policy"]["expected_trainable_parameters"]
    assert recipe["architecture_id"] == ARCHITECTURE_ID
    assert recipe["pilot"]["fixed_fold_id"] == FIXED_FOLD_ID
    assert recipe["pilot"]["fixed_heldout_subject"] == FIXED_HELDOUT
    assert recipe["pilot"]["lofo_expansion_authorized"] is False


def test_mechanics_log1p_transform_preserves_tail_without_clipping():
    raw = torch.tensor([[[0.0, 0.0], [0.015, 0.2], [3098.55, 28601.44]]], dtype=torch.float32)
    transformed = mechanics_target_log1p_v1(raw)
    assert torch.isfinite(transformed).all()
    assert torch.all(transformed >= 0)
    restored = torch.expm1(transformed)
    assert torch.allclose(restored, raw, rtol=1e-5, atol=1e-4)
    assert MECHANICS_TARGET_TRANSFORM == "LOG1P_NONNEGATIVE_MAGNITUDES"
