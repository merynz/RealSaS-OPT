import pytest

from compiler.realsas_compiler_services.proof.causal_attribution import (
    build_controlled_owner_attribution_v1,
)
from compiler.realsas_compiler_services.proof.stage41_controlled_intervention_v1 import (
    build_stage41_controlled_intervention_v1,
)
from compiler.realsas_compiler_services.proof.stage41_failure_context_v1 import (
    build_stage41_failure_attribution_context_v1,
    build_stage41_probe_receipt_v1,
)


def _bindings(**overrides):
    value = {
        "mechanical_state_binding_hash": "STATE:BASE",
        "skeleton_binding_hash": "SK:BASE",
        "mesh_binding_hash": "MESH:BASE",
        "mesh_skin_binding_hash": "SKIN:BASE",
        "qualified_motion_binding_hash": "MOTION:BASE",
        "constraint_set_binding_hash": "CONSTRAINT:BASE",
        "presentation_binding_hash": "PRESENTATION:BASE",
        "mechanical_carrier_evidence_hash": "CARRIER:EVIDENCE:BASE",
        "mechanical_carrier_topology_hash": "CARRIER:TOPOLOGY:BASE",
        "mechanical_carrier_geometry_hash": "CARRIER:GEOMETRY:BASE",
        "static_mesh_qualification_binding_hash": "CARRIER:STATIC:BASE",
        "motion_source_set_binding_hash": "SOURCE:SET",
        "motion_source_seal_binding_hash": "SOURCE:SEAL",
    }
    value.update(overrides)
    return value


def _context(bindings=None, *, policy="POLICY"):
    return build_stage41_failure_attribution_context_v1(
        measurements={
            "clip_id": "run",
            "time_seconds": 0.25,
            "max_triangle_condition_number": 22.0,
        },
        bindings=bindings or _bindings(),
        mesh_policy_hash=policy,
        camera_binding_hashes=tuple(f"CAM:{i}" for i in range(8)),
        observation_set_hash="OBS",
        evaluator_semantic_version="EVAL:V1",
    )


def _receipt(bindings, *, policy="POLICY"):
    return build_stage41_probe_receipt_v1(
        bindings=bindings,
        mesh_policy_hash=policy,
        camera_binding_hashes=tuple(f"CAM:{i}" for i in range(8)),
        observation_set_hash="OBS",
        evaluator_semantic_version="EVAL:V1",
    )


SIGNATURE = {
    "signature_id": "SIG:COND",
    "proof_domain": "MOTION",
    "failure_family": "triangle_condition_exceeded",
    "observed_value": 22.0,
    "allowed_value": 16.0,
}


def test_stage41_real_child_can_keep_same_probe_while_rig_state_changes():
    baseline = _context()
    child = _receipt(
        _bindings(
            mechanical_state_binding_hash="STATE:CHILD",
            skeleton_binding_hash="SK:CHILD",
            mesh_skin_binding_hash="SKIN:CHILD",
            qualified_motion_binding_hash="MOTION:CHILD",
            constraint_set_binding_hash="CONSTRAINT:CHILD",
        )
    )
    assert child["proof_probe_fingerprint"] == baseline["proof_probe_fingerprint"]
    assert child["product_bindings_hash"] != baseline["baseline_product_bindings_hash"]

    intervention = build_stage41_controlled_intervention_v1(
        baseline_failure_context=baseline,
        failure_signature=SIGNATURE,
        child_probe_receipt=child,
        child_measurements={"max_triangle_condition_number": 12.0},
        owner_id="ATLAS_CONTROL_BASIS",
        changed_owner_ids=("ATLAS_CONTROL_BASIS",),
        bounded_change_passed=True,
        counterfactual_status="PASS",
        material_improvement_margin=0.5,
        mutation_summary={"operation": "single_control_prune"},
    )
    finding = build_controlled_owner_attribution_v1(
        failure_signatures=(SIGNATURE,),
        proof_domain="MOTION",
        baseline_source_product_state_hash="STATE:BASE",
        baseline_measurement_report_hash=baseline["baseline_measurement_report_hash"],
        operator_policy_hashes=baseline["operator_policy_hashes"],
        probe_specification=baseline["probe_specification"],
        interventions=(intervention,),
    )[0]
    assert finding["status"] == "attributed"
    assert finding["selected_owner_id"] == "ATLAS_CONTROL_BASIS"
    assert finding["automatic_repair_eligible"] is True


def test_carrier_counterfactual_changes_state_not_external_probe():
    baseline = _context()
    child = _receipt(
        _bindings(
            mechanical_state_binding_hash="STATE:CARRIER_CHILD",
            mesh_binding_hash="MESH:CARRIER_CHILD",
            mesh_skin_binding_hash="SKIN:CARRIER_CHILD",
            mechanical_carrier_evidence_hash="CARRIER:EVIDENCE:CHILD",
            mechanical_carrier_topology_hash="CARRIER:TOPOLOGY:CHILD",
            mechanical_carrier_geometry_hash="CARRIER:GEOMETRY:CHILD",
            static_mesh_qualification_binding_hash="CARRIER:STATIC:CHILD",
            skeleton_binding_hash="SK:CARRIER_CHILD",
            qualified_motion_binding_hash="MOTION:CARRIER_CHILD",
            constraint_set_binding_hash="CONSTRAINT:CARRIER_CHILD",
            presentation_binding_hash="PRESENTATION:CARRIER_CHILD",
        )
    )
    intervention = build_stage41_controlled_intervention_v1(
        baseline_failure_context=baseline,
        failure_signature=SIGNATURE,
        child_probe_receipt=child,
        child_measurements={"max_triangle_condition_number": 14.0},
        owner_id="MECHANICAL_CARRIER",
        changed_owner_ids=("MECHANICAL_CARRIER",),
        bounded_change_passed=True,
        counterfactual_status="PASS",
        material_improvement_margin=0.5,
    )
    assert intervention.probe_fingerprint == baseline["proof_probe_fingerprint"]
    assert (
        intervention.counterfactual_source_product_state_hash
        == "STATE:CARRIER_CHILD"
    )


def test_stage41_intervention_rejects_changed_external_probe_or_multi_owner():
    baseline = _context()
    changed_source = _receipt(
        _bindings(
            mechanical_state_binding_hash="STATE:CHILD",
            motion_source_set_binding_hash="SOURCE:SET:CHANGED",
        )
    )
    with pytest.raises(ValueError, match="PROBE_FINGERPRINT_DRIFT"):
        build_stage41_controlled_intervention_v1(
            baseline_failure_context=baseline,
            failure_signature=SIGNATURE,
            child_probe_receipt=changed_source,
            child_measurements={"max_triangle_condition_number": 12.0},
            owner_id="ATLAS_CONTROL_BASIS",
            changed_owner_ids=("ATLAS_CONTROL_BASIS",),
            bounded_change_passed=True,
            counterfactual_status="PASS",
            material_improvement_margin=0.5,
        )

    child = _receipt(
        _bindings(mechanical_state_binding_hash="STATE:CHILD")
    )
    with pytest.raises(ValueError, match="NOT_SINGLE_OWNER_LOCAL"):
        build_stage41_controlled_intervention_v1(
            baseline_failure_context=baseline,
            failure_signature=SIGNATURE,
            child_probe_receipt=child,
            child_measurements={"max_triangle_condition_number": 12.0},
            owner_id="ATLAS_CONTROL_BASIS",
            changed_owner_ids=("ATLAS_CONTROL_BASIS", "MIRA_SKIN"),
            bounded_change_passed=True,
            counterfactual_status="PASS",
            material_improvement_margin=0.5,
        )
