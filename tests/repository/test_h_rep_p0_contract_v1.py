import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _json(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def test_surf_a1_result_is_sealed_as_fail_and_h_rep_is_active():
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
    assert result["decision"]["carrier_retraining_authorized"] is False
    assert result["decision"]["next_action"] == "H-REP-P0"

    assert ledger["active_experiment"] == "H-REP-P0"
    experiments = {row["id"]: row for row in ledger["experiments"]}
    assert experiments["SURF-A1-FROZEN-H"]["status"] == "SCIENTIFIC_FAIL"
    assert experiments["H-REP-P0"]["tessa_authorized"] is False
    assert experiments["SURF-A2-HIRES"]["status"].startswith("BLOCKED_")
    assert experiments["TESSA-V2-B1"]["status"] == "BLOCKED_ON_SURF_A2_READINESS"
    assert "SURF-A1 COMPLETED SCIENTIFIC FAIL" in state
    assert "H-REP-P0" in state
    assert "SURF-A1 CLOSED FAIL; H-REP-P0 ACTIVE" in plan
    assert "IRIS_H_REP_P0_DECISION_LEDGER_20261019.md" in state


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
    assert tuple(prereg["probe_arms"]) == (
        "H_XYZ",
        "RAW_FIELD_XYZ",
        "XYZ_ONLY",
        "SHUFFLED_H_XYZ",
    )
    architecture = prereg["shared_probe_architecture"]
    assert architecture["input_dim_all_arms"] == 387
    assert architecture["equal_parameter_count_across_arms"] is True
    assert architecture["subject_id_input"] is False
    assert architecture["subject_specific_parameters"] is False
    assert prereg["probe_truth"]["distance_semantics"] == "POINT_CLOUD_PROXY__NOT_EXACT_SDF"
    assert prereg["probe_truth"]["teacher_truth_in_probe_input"] is False
    runtime = prereg["runtime_admission"]
    assert runtime["accelerator"] == "NVIDIA_A100_CLASS"
    assert runtime["minimum_cuda_memory_gib"] == 39
    assert runtime["exact_execution_source_sha_required"] is True
    assert runtime["source_precompile_required"] is True
    assert runtime["optimizer_step_allowed_before_all_authority_gates"] is False
    assert runtime["single_drive_mount_consent_required"] is True
    assert runtime["drive_tree_discovery_allowed"] is False
    assert runtime["additional_google_oauth_allowed"] is False
    assert runtime["reduced_capacity_or_subject_fallback_allowed"] is False
    assert prereg["heldout_court"]["thresholds_may_change_after_results"] is False
    assert prereg["interpretation"]["pass_authorizes_surf_a2"] is False
    assert prereg["interpretation"]["pass_authorizes_tessa"] is False
    assert prereg["interpretation"]["fail_authorizes_carrier_retraining"] is False


def test_h_rep_source_does_not_import_mesh_extraction_or_tessa():
    paths = [
        "models/iris/vnext/h_rep_probe_v1.py",
        "tools/iris_vnext/build_fit8_h_rep_probe_dataset_v1.py",
        "tools/iris_vnext/h_rep_probe_runtime_v1.py",
        "tools/iris_vnext/train_fit8_h_rep_probe_v1.py",
        "tools/iris_vnext/audit_fit8_h_rep_probe_v1.py",
    ]
    text = "\n".join((ROOT / path).read_text(encoding="utf-8") for path in paths)
    forbidden_import_tokens = (
        "PinnedFlexiCubesV1",
        "FrozenHSurfaceRescueV1",
        "models.tessa",
        "decode_vnext_sparse_surface",
    )
    for token in forbidden_import_tokens:
        assert token not in text
