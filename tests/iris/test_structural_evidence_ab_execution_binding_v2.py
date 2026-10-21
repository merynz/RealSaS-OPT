import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BINDING = ROOT / "canonical/STRUCTURAL_EVIDENCE_AB_EXECUTION_BINDING_V2.json"


def test_execution_binding_v2_closes_manifest_contract_without_scientific_drift():
    value = json.loads(BINDING.read_text(encoding="utf-8"))
    assert value["schema"] == "RealSaS.StructuralEvidenceABExecutionBinding.v2"
    assert value["execution_source"]["merge_sha"] == "68689f1ae6617482da850a684e8709ab58715214"
    assert value["execution_source"]["source_mutation_after_merge_for_scientific_execution_allowed"] is False

    closure = value["compatibility_closure"]
    assert closure["adapter_path"] == "tools/iris_vnext/normalize_direct_presentation_manifest_contract_v1.py"
    assert closure["presentation_subject_payloads_read_by_adapter"] is False
    assert closure["presentation_subject_payloads_mutated_by_adapter"] is False
    assert closure["evidence_tensor_values_mutated"] is False
    assert closure["model_architecture_mutated"] is False
    assert closure["optimizer_or_schedule_mutated"] is False
    assert closure["court_thresholds_mutated"] is False

    assert value["execution_program"]["arm_order"] == ["A_NULL_PRESENTATION", "B_ROLE_FREE_PRESENTATION"]
    assert value["execution_program"]["optimizer_steps_per_arm"] == 2400
    assert value["execution_program"]["seed_set"] == [26091991]
    assert value["execution_program"]["sampling"] is False

    assert value["notebook_contract"]["requires_checkout_execution_merge_sha"] == value["execution_source"]["merge_sha"]
    assert value["notebook_contract"]["requires_manifest_contract_normalization_before_ab_preflight"] is True
    assert value["notebook_contract"]["requires_zero_optimizer_preflight_before_training"] is True
    assert value["notebook_contract"]["requires_all_three_independent_courts"] is True

    facts = value["execution_facts_at_binding_time"]
    assert facts["optimizer_steps_executed"] == 0
    assert facts["scientific_training_executed"] is False
    assert facts["scientific_ab_court_executed"] is False
    assert facts["scientific_pass_or_fail_observed"] is False

    ci = value["source_ci"]
    assert ci["all_triggered_workflows_completed_success"] is True
    assert all(row["conclusion"] == "success" for row in ci["workflows"])
    assert ci["optimizer_steps_executed_by_ci"] == 0
    assert ci["scientific_training_executed_by_ci"] is False
