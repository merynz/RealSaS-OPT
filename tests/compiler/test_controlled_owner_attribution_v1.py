from compiler.realsas_compiler_services.proof.causal_attribution import (
    ControlledInterventionEvidenceV1,
    build_controlled_owner_attribution_v1,
    proof_probe_fingerprint_v1,
)


def _base(owner="mesh", *, after=1.1, probe="same", changed=None, regression=()):
    fp = proof_probe_fingerprint_v1(operator_policy_hashes=("P",), probe_specification={"k": 1})
    return ControlledInterventionEvidenceV1(
        intervention_id=f"I:{owner}",
        target_signature_id="S:1",
        proof_domain="MOTION",
        owner_id=owner,
        baseline_source_product_state_hash="BASE",
        baseline_measurement_report_hash="MR",
        counterfactual_source_product_state_hash=f"CHILD:{owner}",
        probe_fingerprint=fp if probe == "same" else "WRONG",
        changed_owner_ids=tuple(changed or (owner,)),
        bounded_change_passed=True,
        target_metric="max_area_change_ratio",
        target_metric_before=2.0,
        target_metric_after=after,
        higher_is_better=False,
        material_improvement_margin=0.2,
        baseline_status="FAIL",
        counterfactual_status="FAIL",
        protected_invariant_regressions=tuple(regression),
    )


def _run(*rows):
    return build_controlled_owner_attribution_v1(
        failure_signatures=({"signature_id": "S:1", "failure_family": "area_change_exceeded"},),
        proof_domain="MOTION",
        baseline_source_product_state_hash="BASE",
        baseline_measurement_report_hash="MR",
        operator_policy_hashes=("P",),
        probe_specification={"k": 1},
        interventions=rows,
    )[0]


def test_single_owner_material_counterfactual_attributes():
    out = _run(_base("mesh", after=1.1), _base("weight", after=1.95))
    assert out["status"] == "attributed"
    assert out["selected_owner_id"] == "mesh"
    assert out["automatic_repair_eligible"] is True


def test_multiple_material_owners_abstain():
    out = _run(_base("mesh", after=1.1), _base("weight", after=1.2))
    assert out["status"] == "abstained"
    assert out["selected_owner_id"] is None
    assert out["abstention_reason"] == "multiple_owner_counterfactuals_materially_improve"


def test_probe_change_or_cross_owner_mutation_cannot_attribute():
    out = _run(
        _base("mesh", after=1.0, probe="wrong"),
        _base("weight", after=1.0, changed=("weight", "mesh")),
    )
    assert out["status"] == "abstained"
    assert out["abstention_reason"] == "all_controlled_interventions_rejected"
    blockers = {b for row in out["rejected_interventions"] for b in row["blockers"]}
    assert "probe_specification_changed" in blockers
    assert "counterfactual_not_single_owner_local" in blockers


def test_protected_regression_rejects_credit():
    out = _run(_base("mesh", after=1.0, regression=("loop_seam_regressed",)))
    assert out["status"] == "abstained"
    assert out["abstention_reason"] == "all_controlled_interventions_rejected"
