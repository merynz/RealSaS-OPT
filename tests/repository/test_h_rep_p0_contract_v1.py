import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FROZEN_H_REP_SOURCE = "479e8e53cf0d7df234a3e64b824bb704b861bf19"

def _json(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))

def test_surf_a1_and_h_rep_history_remain_sealed_after_h_geo_progression():
    result = _json("docs/research/IRIS_SURF_A1_FROZEN_H_RESULT_20261019.json")
    ledger = _json("canonical/IRIS_RESEARCH_EXPERIMENT_LEDGER_V1.json")
    state = (ROOT / "CURRENT_STATE.md").read_text(encoding="utf-8")
    plan = (ROOT / "CURRENT_PLAN.md").read_text(encoding="utf-8")
    assert result["execution_source_sha"] == "afc18debbc4d5858a4a632884b7cfd2141dcc35a"
    assert result["preflight"]["status"] == "PASS_SURF_A1_A100_PREFLIGHT__TRAINING_AUTHORIZED"
    assert result["training"]["optimizer_steps_executed"] == 2400
    assert result["independent_court"]["status"] == "SCIENTIFIC_FAIL_SURF_A1_FROZEN_H_REPRESENTATION_RESCUE"
    assert result["decision"]["surf_a2_authorized"] is False
    assert result["decision"]["tessa_authorized"] is False
    rows={row["id"]:row for row in ledger["experiments"]}
    allowed=("H-GEO-P2-SCENE-FIELD", "H-GEO-P3-EXPLICIT-HIRES-CARRIER", "H-GEO-P3-EFREQ", "H-GEO-P3-SCALE-XFER")
    assert ledger["active_experiment"] in allowed
    hrep=rows["H-REP-P0"]
    assert hrep["status"] == "SCIENTIFIC_FAIL"
    assert hrep["frozen_execution_source_sha"] == FROZEN_H_REP_SOURCE
    assert hrep["scientific_result_available"] is True
    assert hrep["tessa_authorized"] is False
    assert rows["H-GEO-P1"]["status"] == "SCIENTIFIC_FAIL"
    p2=rows["H-GEO-P2-SCENE-FIELD"]
    assert p2["status"] in ("SOURCE_PREPARATION", "SCIENTIFIC_FAIL")
    if ledger["active_experiment"] in ("H-GEO-P3-EXPLICIT-HIRES-CARRIER", "H-GEO-P3-EFREQ", "H-GEO-P3-SCALE-XFER"):
        assert p2["status"] == "SCIENTIFIC_FAIL"
    if ledger["active_experiment"] == "H-GEO-P3-EXPLICIT-HIRES-CARRIER":
        assert rows["H-GEO-P3-EXPLICIT-HIRES-CARRIER"]["status"] == "SOURCE_PREPARATION"
    if ledger["active_experiment"] == "H-GEO-P3-EFREQ":
        assert rows["H-GEO-P3-EXPLICIT-HIRES-CARRIER"]["status"] == "SCIENTIFIC_FAIL"
        assert rows["H-GEO-P3-EFREQ"]["status"] == "SOURCE_PREPARATION"
        assert rows["H-GEO-P3-EFREQ"]["optimizer_steps_authorized"] == 0
        assert rows["H-GEO-P3-EFREQ"]["probe_optimizer_steps_authorized"] == 0
    if ledger["active_experiment"] == "H-GEO-P3-SCALE-XFER":
        assert rows["H-GEO-P3-EXPLICIT-HIRES-CARRIER"]["status"] == "SCIENTIFIC_FAIL"
        assert rows["H-GEO-P3-EFREQ"]["status"] == "COMPLETED_UNRESOLVED"
        assert rows["H-GEO-P3-EFREQ"]["optimizer_steps_executed"] == 0
        assert rows["H-GEO-P3-EFREQ"]["probe_optimizer_steps_executed"] == 0
        sx=rows["H-GEO-P3-SCALE-XFER"]
        assert sx["status"] == "SOURCE_PREPARATION"
        assert sx["optimizer_steps_authorized"] == 0
        assert sx["probe_optimizer_steps_authorized"] == 0
        assert sx["p4_training_authorized"] is False
    assert rows["SURF-A2-HIRES"]["status"].startswith("BLOCKED_")
    assert rows["TESSA-V2-B1"]["status"] == "BLOCKED_ON_SURF_A2_READINESS"
    assert "H-GEO-P1" in state and "H-GEO-P2" in state
    assert ledger["active_experiment"] in state
    assert ledger["active_experiment"] in plan

def test_h_rep_binding_is_exactly_triggered_by_a1_evidence():
    binding = _json("canonical/H_REP_P0_INPUT_BINDING_V1.json")
    result = _json("docs/research/IRIS_SURF_A1_FROZEN_H_RESULT_20261019.json")
    assert binding["schema"] == "RealSaS.HRepP0InputBinding.v1"
    assert binding["v8"]["checkpoint"]["sha256"] == "0be15df5614a05343cf3850aff747ba4d31f00a322f058819cb5a54d5a5d2fc5"
    assert binding["prepared"]["manifest"]["sha256"] == "6561bf2087f4fac3699c673b5c93fb3d8dbc2d1ff90a5065d976cae70688b4d8"
    assert binding["trigger_evidence"]["surf_a1_preflight"]["sha256"] == result["preflight"]["sha256"]
    assert binding["trigger_evidence"]["surf_a1_training_result"]["sha256"] == result["training"]["sha256"]
    assert binding["trigger_evidence"]["surf_a1_independent_court"]["sha256"] == result["independent_court"]["sha256"]
    assert binding["transport_policy"]["single_drive_consent"] is True
    assert binding["transport_policy"]["drive_tree_scan_forbidden"] is True
    assert binding["transport_policy"]["second_google_oauth_forbidden"] is True

def test_h_rep_preregistration_keeps_representation_question_narrow():
    prereg = _json("canonical/H_REP_P0_PREREGISTRATION_V1.json")
    assert prereg["schema"] == "RealSaS.HRepP0Preregistration.v1"
    assert tuple(prereg["probe_arms"]) == ("H_XYZ","RAW_FIELD_XYZ","XYZ_ONLY","SHUFFLED_H_XYZ")
    architecture=prereg["shared_probe_architecture"]
    assert architecture["input_dim_all_arms"] == 387
    assert architecture["equal_parameter_count_across_arms"] is True
    assert architecture["subject_id_input"] is False
    assert architecture["subject_specific_parameters"] is False
    assert prereg["probe_truth"]["distance_semantics"] == "POINT_CLOUD_PROXY__NOT_EXACT_SDF"
    assert prereg["probe_truth"]["teacher_truth_in_probe_input"] is False
    assert prereg["heldout_court"]["thresholds_may_change_after_results"] is False

def test_h_rep_dependency_graph_is_extractor_free():
    paths=["models/iris/vnext/h_rep_probe_v1.py","tools/iris_vnext/h_rep_probe_authority_v1.py","tools/iris_vnext/build_fit8_h_rep_probe_dataset_v1.py","tools/iris_vnext/h_rep_probe_runtime_v1.py","tools/iris_vnext/train_fit8_h_rep_probe_v1.py","tools/iris_vnext/audit_fit8_h_rep_probe_v1.py"]
    text="\n".join((ROOT/p).read_text() for p in paths)
    for token in ("materialize_fit8_direct_surface_v1","surface_rescue_training_v1","SparseRegularTetra","models.iris.vnext.sparse_surface","PinnedFlexiCubesV1","FrozenHSurfaceRescueV1","models.tessa","decode_vnext_sparse_surface"):
        assert token not in text
