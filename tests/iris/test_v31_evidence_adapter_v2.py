from __future__ import annotations

import json
from pathlib import Path

import torch

from models.iris.v31.evidence_adapter_model_v2 import (
    IRISEvidenceAdapterConfigV2,
    IRISV31EvidenceAdapterV2,
)


ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / "canonical/IRIS_V31_FIXED_SNAKE_EVIDENCE_ADAPTER_PILOT_V2.json"


def _model():
    recipe = json.loads(RECIPE.read_text())
    row = dict(recipe["model_config"])
    row["mechanical_channels"] = tuple(row["mechanical_channels"])
    row["pair_channels"] = tuple(row["pair_channels"])
    return recipe, IRISV31EvidenceAdapterV2(IRISEvidenceAdapterConfigV2(**row))


def test_parameter_courts_and_family_separation():
    recipe, model = _model()
    total = sum(p.numel() for p in model.parameters())
    foundation = sum(p.numel() for _, p in model.foundation_named_parameters())
    evidence = sum(p.numel() for _, p in model.evidence_named_parameters())
    assert total == recipe["capacity_policy"]["expected_trainable_parameters_total"] == 9_768_366
    assert foundation == recipe["capacity_policy"]["expected_foundation_graph_parameters"] == 4_564_644
    assert evidence == recipe["capacity_policy"]["expected_evidence_adapter_parameters"] == 1_972_745

    foundation_ids = {id(p) for _, p in model.foundation_named_parameters()}
    evidence_ids = {id(p) for _, p in model.evidence_named_parameters()}
    relation_ids = {id(p) for _, p in model.frozen_unused_named_parameters()}
    assert foundation_ids.isdisjoint(evidence_ids)
    assert foundation_ids.isdisjoint(relation_ids)
    assert evidence_ids.isdisjoint(relation_ids)
    assert len(foundation_ids | evidence_ids | relation_ids) == len(list(model.parameters()))


def test_new_evidence_adapters_are_exact_identity_at_initialization():
    _, model = _model()
    torch.manual_seed(7)
    state = torch.randn(2, 11, model.config.query_width)
    assert torch.equal(model.mm(state), state)
    assert torch.equal(model.gp(state), state)


def test_mechanics_and_geometry_do_not_share_private_adapter_parameters():
    _, model = _model()
    gm = {id(p) for p in model.gm.parameters()}
    mm = {id(p) for p in model.mm.parameters()}
    assert gm
    assert mm
    assert gm.isdisjoint(mm)


def test_recipe_forbids_cross_stage_selector_and_shared_joint_unfreeze():
    recipe = json.loads(RECIPE.read_text())
    assert recipe["pilot"]["selection_rule"] == "FIXED_FINAL_STEP__NO_CROSS_STAGE_SCALAR_CHECKPOINT_COMPETITION"
    assert recipe["pilot"]["selected_step"] == 6000
    assert recipe["foundation_policy"]["freeze_after_foundation"] is True
    assert recipe["evidence_adapter_policy"]["shared_foundation_unfreeze_authorized"] is False
    assert recipe["evidence_adapter_policy"]["mechanics_geometry_private_path_shared"] is False
    assert [row["id"] for row in recipe["schedule"]] == [
        "GEOMETRY_FOUNDATION",
        "EVIDENCE_ADAPTER_ACQUISITION",
    ]
