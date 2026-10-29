from __future__ import annotations

import json
from pathlib import Path

import torch

from models.iris.v31.evidence_adapter_model_v2 import (
    IRISEvidenceAdapterConfigV2,
    IRISV31EvidenceAdapterV2,
)
from models.iris.v31.mechanics_student_t_v3 import student_t_mechanical_loss_v3
from models.iris.v31.presentation_objective_v1 import heteroscedastic_mechanical_loss_v1


ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / "canonical/IRIS_V31_FIXED_SNAKE_EVIDENCE_STUDENT_T_PILOT_V3.json"


def _config(recipe: dict) -> IRISEvidenceAdapterConfigV2:
    row = dict(recipe["model_config"])
    row["mechanical_channels"] = tuple(row["mechanical_channels"])
    row["pair_channels"] = tuple(row["pair_channels"])
    return IRISEvidenceAdapterConfigV2(**row)


def test_student_t_tail_is_finite_and_far_less_quadratic_than_gaussian():
    # Same four-channel output contract: mean0, logvar0, mean1, logvar1.
    raw = torch.tensor([[[0.0, -6.0, 0.0, -6.0]]], dtype=torch.float32)
    target = torch.tensor([[[10.0, 0.0]]], dtype=torch.float32)
    valid = torch.ones((1, 1), dtype=torch.bool)

    robust = student_t_mechanical_loss_v3(
        raw, target, valid, degrees_of_freedom=3.0,
        minimum_log_variance=-6.0, maximum_log_variance=3.0,
    )
    gaussian = heteroscedastic_mechanical_loss_v1(raw, target, valid)

    assert torch.isfinite(robust)
    assert torch.isfinite(gaussian)
    assert float(robust) < 25.0
    assert float(gaussian) > 1_000.0


def test_student_t_retains_gradient_for_large_tail():
    raw = torch.tensor([[[0.0, 0.0, 0.0, 0.0]]], dtype=torch.float32, requires_grad=True)
    target = torch.tensor([[[10.0, 8.0]]], dtype=torch.float32)
    valid = torch.ones((1, 1), dtype=torch.bool)
    loss = student_t_mechanical_loss_v3(raw, target, valid, degrees_of_freedom=3.0)
    loss.backward()
    assert raw.grad is not None
    assert bool(torch.isfinite(raw.grad).all())
    assert float(raw.grad[..., 0].abs().max()) > 0.0
    assert float(raw.grad[..., 2].abs().max()) > 0.0


def test_recipe_keeps_architecture_truth_partition_and_capacity_fixed():
    recipe = json.loads(RECIPE.read_text())
    assert recipe["schema"] == "RealSaS.IRISV31FixedSnakeEvidenceStudentTPilotRecipe.v3"
    assert recipe["pilot"]["fixed_heldout_subject"] == "fit8_snake_v1"
    assert recipe["pilot"]["selected_evidence_step"] == 5000
    assert recipe["pilot"]["lofo_expansion_authorized"] is False
    assert recipe["partition_policy"] == "UNCHANGED_HASHED_ROW_80_20_V1"
    assert recipe["mechanics_target_transform"] == "LOG1P_NONNEGATIVE_MAGNITUDES"
    assert recipe["mechanics_transform_policy"]["target_clipping"] is False
    assert recipe["mechanics_transform_policy"]["truth_rows_deleted"] is False
    assert recipe["mechanics_objective"]["degrees_of_freedom"] == 3.0
    assert recipe["mechanics_objective"]["subject_specific_weights"] is False
    assert recipe["foundation_import_policy"]["copy_foundation_parameters_only"] is True
    assert recipe["foundation_import_policy"]["fresh_evidence_initialization"] is True
    assert recipe["claims"]["gaussian_mechanics_objective_used"] is False

    model = IRISV31EvidenceAdapterV2(_config(recipe))
    assert sum(p.numel() for p in model.parameters()) == 9_768_366
    assert sum(p.numel() for _, p in model.foundation_named_parameters()) == 4_564_644
    assert sum(p.numel() for _, p in model.evidence_named_parameters()) == 1_972_745
