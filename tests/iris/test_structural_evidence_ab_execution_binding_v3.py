import json
from pathlib import Path


def test_structural_evidence_ab_execution_binding_v3():
    path = Path("canonical/STRUCTURAL_EVIDENCE_AB_EXECUTION_BINDING_V3.json")
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["schema"] == "RealSaS.StructuralEvidenceABExecutionBinding.v3"
    assert value["execution_source"]["merge_sha"] == "39476c024b09fa61bc5bc3e0404123204afeed50"
    runtime = value["runtime_hardening_closure"]
    assert runtime["orchestration_only"] is True
    assert runtime["decoder_semantics_changed"] is False
    assert runtime["field_semantics_changed"] is False
    assert runtime["coordinate_semantics_changed"] is False
    assert runtime["model_parameter_bytes_changed"] is False
    assert runtime["direct_surface_decoder"] == "RealSaS.ZeroSurfaceDecoder.SparseRegularT512T1024IndexedMT.v5"
    assert runtime["base_cells"] == 512
    assert runtime["fine_cells"] == 1024
    program = value["execution_program"]
    assert program["direct_surface_materializer"] == "tools/iris_vnext/materialize_fit8_direct_surface_resumable_v2.py"
    assert program["arm_order"] == ["A_NULL_PRESENTATION", "B_ROLE_FREE_PRESENTATION"]
    assert program["optimizer_steps_per_arm"] == 2400
    notebook = value["notebook_contract"]
    assert notebook["requires_checkout_execution_merge_sha"] == value["execution_source"]["merge_sha"]
    assert notebook["requires_resumable_isolated_direct_surface_materialization"] is True
    assert notebook["requires_direct_surface_failure_receipt_on_child_failure"] is True
    assert notebook["requires_preservation_of_completed_direct_surface_subjects"] is True
    facts = value["execution_facts_at_binding_time"]
    assert facts["scientific_training_executed"] is False
    assert facts["optimizer_steps_executed"] == 0
    assert facts["scientific_pass_or_fail_observed"] is False
