from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_scale_xfer_prereg_is_zero_optimizer_and_locked():
    p = json.loads((ROOT / "canonical/H_GEO_P3_SCALE_XFER_PREREGISTRATION_V1.json").read_text())
    assert p["schema"] == "RealSaS.HGeoP3ScaleTransferPreregistration.v1"
    assert p["optimizer_policy"]["carrier_optimizer_steps_authorized"] == 0
    assert p["optimizer_policy"]["probe_optimizer_steps_authorized"] == 0
    assert p["optimizer_policy"]["checkpoint_write_allowed"] is False
    assert p["query_authority"]["reuse_efreq_query_csv_byte_exact"] is True
    assert p["query_authority"]["tail_and_control_membership_may_not_be_reselected"] is True
    assert p["spatial_scales_iris"] == [1/1024, 1/512, 1/256, 1/128, 1/64, 1/32]
    assert p["authorization"]["p4_training_authorized"] is False
    assert p["authorization"]["surf_a2_authorized"] is False
    assert p["authorization"]["tessa_authorized"] is False


def test_scale_xfer_binding_is_chained_to_sealed_efreq():
    b = json.loads((ROOT / "canonical/H_GEO_P3_SCALE_XFER_INPUT_BINDING_V1.json").read_text())
    assert b["schema"] == "RealSaS.HGeoP3ScaleTransferInputBinding.v1"
    assert b["trigger_result"]["efreq_classification"] == "UNRESOLVED"
    assert b["trigger_result"]["efreq_audit_sha256"] == "ab4ca656442de0addb9468c6dcd7bf3e1e23b5a874fb4e5b675f2079720f2d04"
    assert b["trigger_result"]["efreq_query_csv_sha256"] == "41bf4f6bf1d90fa639e01fc799c032fb276ef51dd833fed1891ce4ce7c87a16d"
    assert b["transport_policy"]["checkpoint_write_allowed"] is False


def test_scale_xfer_runtime_has_no_training_entrypoint_or_optimizer():
    src = (ROOT / "tools/iris_vnext/audit_h_geo_p3_scale_xfer_v1.py").read_text()
    assert "torch.optim" not in src
    assert ".backward(" not in src
    assert "optimizer.step" not in src
    assert "train_fit8" not in src
    assert "tessa" in src.lower()  # explicit false result boundary is present


def test_sealed_efreq_result_requires_more_diagnosis_not_p4_training():
    r = json.loads((ROOT / "docs/research/IRIS_H_GEO_P3_EFREQ_RESULT_20261020.json").read_text())
    assert r["classification"] == "UNRESOLVED"
    assert r["optimizer_steps_executed"] == 0
    assert r["next_action"] == "H-GEO-P3-SCALE-XFER_ZERO_OPTIMIZER_STAGE_LOCALIZATION"
    assert r["surf_a2_authorized"] is False
    assert r["tessa_authorized"] is False
