import json
from pathlib import Path


SOURCE = "66b1fb0cccc6145057aa325cc7aac877239b297c"
DECODER = (
    "RealSaS.ZeroSurfaceDecoder.SparseRegularT512T1024IndexedMT."
    "ExactShellContainment.v5_1"
)


def test_structural_evidence_ab_execution_binding_v4():
    value = json.loads(
        Path("canonical/STRUCTURAL_EVIDENCE_AB_EXECUTION_BINDING_V4.json")
        .read_text(encoding="utf-8")
    )
    assert value["schema"] == "RealSaS.StructuralEvidenceABExecutionBinding.v4"
    assert value["execution_source"]["merge_sha"] == SOURCE
    assert value["execution_source"]["pull_request"] == 145

    observed = value["observed_preoptimizer_failure"]
    assert observed["subject_id"] == "Snake"
    assert observed["optimizer_steps_executed"] == 0
    assert observed["backward_steps_executed"] == 0
    assert observed["scientific_pass_or_fail_observed"] is False

    correction = value["direct_surface_gate_correction"]
    assert correction["decoder_id"] == DECODER
    assert correction["exact_shell_containment_required"] is True
    assert correction["legacy_boundary_crossing_tets_role"] == "TELEMETRY_ONLY"
    assert correction["legacy_boundary_crossing_tets_is_truncation_gate"] is False
    assert correction["base_cells"] == 512
    assert correction["fine_cells"] == 1024
    assert correction["field_semantics_changed"] is False
    assert correction["coordinate_semantics_changed"] is False
    assert correction["model_parameter_bytes_changed"] is False
    assert correction["optimizer_or_schedule_changed"] is False
    assert correction["court_thresholds_changed"] is False

    authority = value["locked_authority"]
    assert authority["direct_surface_input_binding"] == "canonical/VNEXT_DIRECT_SURFACE_INPUT_BINDING_V2.json"
    assert authority["direct_surface_decoder"] == DECODER
    assert authority["admitted_v2_oriented_pair_rows"] == 3499296

    program = value["execution_program"]
    assert program["direct_surface_materializer"] == "tools/iris_vnext/materialize_fit8_direct_surface_resumable_v3.py"
    assert program["arm_order"] == ["A_NULL_PRESENTATION", "B_ROLE_FREE_PRESENTATION"]
    assert program["optimizer_steps_per_arm"] == 2400
    assert program["seed_set"] == [26091991]

    notebook = value["notebook_contract"]
    assert notebook["requires_checkout_execution_merge_sha"] == SOURCE
    assert notebook["requires_direct_surface_input_binding_v2"] is True
    assert notebook["requires_exact_shell_containment_before_surface_extraction"] is True
    assert notebook["requires_legacy_boundary_proxy_telemetry_only"] is True
    assert notebook["requires_zero_optimizer_preflight_before_training"] is True
    assert notebook["requires_all_three_independent_courts"] is True

    facts = value["execution_facts_at_binding_time"]
    assert facts["scientific_training_executed"] is False
    assert facts["optimizer_steps_executed"] == 0
    assert facts["scientific_ab_court_executed"] is False
    assert facts["scientific_pass_or_fail_observed"] is False


def test_binding_replays_locked_blobs_at_execution_source():
    value = json.loads(
        Path("canonical/STRUCTURAL_EVIDENCE_AB_EXECUTION_BINDING_V4.json")
        .read_text(encoding="utf-8")
    )
    locked = value["locked_authority"]
    assert locked["preregistration_git_blob_sha"] == "10706b00fe7a201ef006e67c545c3a26ca85703f"
    assert locked["court_metric_definitions_git_blob_sha"] == "6f4e33da465edda45eaacdbcbeced487ae5d26a8"
    assert locked["evidence_admission_git_blob_sha"] == "e9383617f74497f7e56aaa772394090786c28955"
    assert locked["direct_surface_input_binding_git_blob_sha"] == "2a3d252f27e08a24a506552447a5d44614f8e698"
