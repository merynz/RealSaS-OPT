from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_h_geo_p3_efreq_prereg_is_zero_optimizer_and_fail_closed():
    p = json.loads((ROOT / "canonical/H_GEO_P3_EFREQ_PREREGISTRATION_V1.json").read_text())
    assert p["schema"] == "RealSaS.HGeoP3EFreqPreregistration.v1"
    assert p["optimizer_policy"]["carrier_optimizer_steps"] == 0
    assert p["optimizer_policy"]["probe_optimizer_steps"] == 0
    assert p["optimizer_policy"]["any_parameter_mutation_forbidden"] is True
    assert p["optimizer_policy"]["checkpoint_write_forbidden"] is True
    assert p["optimizer_policy"]["tessa_invoked"] is False
    assert p["interpretation"]["any_result_authorizes_surf_a2"] is False
    assert p["interpretation"]["any_result_authorizes_tessa"] is False
    assert p["decision_rules"]["thresholds_may_change_after_results"] is False
    assert p["perturbation_audit"]["epsilon_scales_iris"] == [0.00390625,0.0078125,0.015625]


def test_h_geo_p3_efreq_binding_is_exact_p3_result_chain():
    b = json.loads((ROOT / "canonical/H_GEO_P3_EFREQ_INPUT_BINDING_V1.json").read_text())
    assert b["schema"] == "RealSaS.HGeoP3EFreqInputBinding.v1"
    assert b["trigger_result"]["p3_status"] == "SCIENTIFIC_FAIL_H_GEO_P3_EXPLICIT_HIRES_CARRIER"
    assert b["p3"]["checkpoint"]["sha256"] == "95af736e1e9163cad9d568ea9c67770fe2fbf3391bbdb72d8c03c472b5b5d74b"
    assert b["p2_comparison"]["checkpoint"]["sha256"] == "0bdcd24af776e322bfbb1a004d6034167770b7950ecccb9b04d6d689597854ee"
    assert b["v8"]["checkpoint"]["sha256"] == "0be15df5614a05343cf3850aff747ba4d31f00a322f058819cb5a54d5a5d2fc5"
    assert len(b["teachers"]["subjects"]) == 8
    assert len(b["prepared"]["subjects"]) == 8


def test_h_geo_p3_efreq_source_cannot_optimize_or_mutate_model():
    text = (ROOT / "tools/iris_vnext/audit_h_geo_p3_efreq_v1.py").read_text()
    forbidden = ("torch.optim", ".backward(", ".step()", "requires_grad_(True)", "TESSA", "FlexiCubes")
    for literal in forbidden:
        assert literal not in text
    assert "requires_grad_(False)" in text
    assert '"optimizer_steps_executed": 0' in text
    assert '"probe_optimizer_steps_executed": 0' in text
    assert "SCIENTIFIC_FAIL_H_GEO_P3_EXPLICIT_HIRES_CARRIER" in text
    assert "P3_TRAINING_AUX_USED_ONLY_TO_RANK_ENDPOINT_TAIL_NOT_TO_REQUALIFY_P3" in text


def test_h_geo_p3_result_is_sealed_before_efreq():
    r = json.loads((ROOT / "docs/research/IRIS_H_GEO_P3_RESULT_20261020.json").read_text())
    assert r["status"] == "SCIENTIFIC_FAIL_H_GEO_P3_EXPLICIT_HIRES_CARRIER"
    assert r["optimizer_steps_executed"] == 4800
    assert r["shared_v8_preserved_exact"] is True
    assert r["gate_summary"]["absolute_endpoint"] is False
    assert r["next_action"] == "AUDIT_EXPLICIT_GEOMETRY_CARRIER_ENDPOINT_FREQUENCY_CEILING__DO_NOT_TRAIN_TESSA"
