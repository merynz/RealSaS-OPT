from compiler.realsas_compiler_services.proof.repair_loop import (
    BoundedRepairOperationV1,
    RepairApplicationRecordV1,
    RepairReproofEvidenceV1,
    build_bounded_repair_directives_v1,
    evaluate_repair_effect_v1,
    validate_repair_application_v1,
)


FINDING = {
    "finding_id": "OWNER_ATTR:1",
    "signature_id": "S:1",
    "status": "attributed",
    "selected_owner_id": "mesh",
    "automatic_repair_eligible": True,
}


def _op(owner="mesh", *, qualified=True):
    return BoundedRepairOperationV1(
        operation_id="OP:1",
        owner_id=owner,
        operation_family="local_retriangulate",
        target_signature_ids=("S:1",),
        qualification_hash="QUAL" if qualified else "",
        automatic_execution_qualified=qualified,
        allowed_change_paths=("directional_renderables.mesh.topology",),
        bounded_change_spec={"max_local_faces": 8},
        protected_invariants=("motion_probe_policy", "canonical_joint_ids"),
    )


def _directive():
    rows = build_bounded_repair_directives_v1(
        attribution_finding=FINDING,
        parent_product_state_hash="PARENT",
        baseline_measurement_report_hash="MR",
        proof_probe_fingerprint="PROBE",
        operation_candidates=(_op(),),
    )
    assert len(rows) == 1
    return rows[0]


def _application(d, *, owners=("mesh",), paths=("directional_renderables.mesh.topology",), parent="PARENT"):
    return RepairApplicationRecordV1(
        application_id="APP:1",
        directive_id=d.directive_id,
        parent_product_state_hash="PARENT",
        child_product_state_hash="CHILD",
        child_parent_state_hash=parent,
        applied_operation_id="OP:1",
        changed_owner_ids=tuple(owners),
        changed_paths=tuple(paths),
        bounded_change_passed=True,
    )


def _reproof(d, *, probe="PROBE", regression=(), improved=True):
    return RepairReproofEvidenceV1(
        directive_id=d.directive_id,
        child_product_state_hash="CHILD",
        child_proof_bundle_hash="PB:CHILD",
        proof_probe_fingerprint=probe,
        target_signature_id="S:1",
        target_materially_improved=improved,
        target_resolved=False,
        child_domain_status="FAIL",
        protected_invariant_regressions=tuple(regression),
        target_metric_deltas={"max_area_change_ratio": -0.7},
    )


def test_attributed_owner_yields_executable_qualified_directive():
    d = _directive()
    assert d.executable
    assert d.selected_owner_id == "mesh"
    assert d.metadata["mandatory_reproof"] is True


def test_unattributed_or_unqualified_does_not_create_executable_repair():
    assert build_bounded_repair_directives_v1(
        attribution_finding={**FINDING, "status": "abstained"},
        parent_product_state_hash="PARENT",
        baseline_measurement_report_hash="MR",
        proof_probe_fingerprint="PROBE",
        operation_candidates=(_op(),),
    ) == ()
    rows = build_bounded_repair_directives_v1(
        attribution_finding=FINDING,
        parent_product_state_hash="PARENT",
        baseline_measurement_report_hash="MR",
        proof_probe_fingerprint="PROBE",
        operation_candidates=(_op(qualified=False),),
    )
    assert len(rows) == 1 and not rows[0].executable


def test_child_application_must_be_distinct_single_owner_and_in_scope():
    d = _directive()
    ok, blockers = validate_repair_application_v1(d, _application(d, owners=("mesh", "weight")))
    assert not ok and "repair_not_single_owner_local" in blockers
    ok, blockers = validate_repair_application_v1(d, _application(d, paths=("mechanical_state.skeleton",)))
    assert not ok and "repair_changed_path_outside_operation_scope" in blockers


def test_same_probe_reproof_with_material_improvement_accepts_effect():
    d = _directive()
    out = evaluate_repair_effect_v1(directive=d, application=_application(d), reproof=_reproof(d))
    assert out["repair_accepted"] is True
    assert out["blockers"] == []


def test_probe_change_or_protected_regression_rejects_effect():
    d = _directive()
    out = evaluate_repair_effect_v1(
        directive=d,
        application=_application(d),
        reproof=_reproof(d, probe="DIFFERENT", regression=("loop_seam_regressed",)),
    )
    assert out["repair_accepted"] is False
    assert "repair_reproof_probe_changed" in out["blockers"]
    assert "repair_protected_invariant_regression" in out["blockers"]
