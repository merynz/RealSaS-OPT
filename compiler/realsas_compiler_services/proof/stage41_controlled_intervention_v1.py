from __future__ import annotations

"""Build admissible Stage41 controlled-intervention evidence.

This module does not attribute an owner. It only converts one audited child
reproof into the generic ControlledInterventionEvidenceV1 contract after proving
that parent and child used the same external Stage41 evaluation probe.
"""

from typing import Any, Mapping, Sequence

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_services.proof.causal_attribution import (
    ControlledInterventionEvidenceV1,
)


_METRICS = {
    "triangle_area_ratio_below_min": ("min_triangle_area_ratio", True),
    "triangle_area_ratio_above_max": ("max_triangle_area_ratio", False),
    "triangle_condition_exceeded": ("max_triangle_condition_number", False),
    "new_self_intersection": ("new_self_intersection_pair_count", False),
    "rest_intersection_worsened": (
        "max_rest_existing_intersection_severity_growth",
        False,
    ),
    "contact_drift_exceeded": ("max_contact_drift", False),
}


def build_stage41_controlled_intervention_v1(
    *,
    baseline_failure_context: Mapping[str, Any],
    failure_signature: Mapping[str, Any],
    child_probe_receipt: Mapping[str, Any],
    child_measurements: Mapping[str, Any],
    owner_id: str,
    changed_owner_ids: Sequence[str],
    bounded_change_passed: bool,
    counterfactual_status: str,
    material_improvement_margin: float,
    protected_invariant_regressions: Sequence[str] = (),
    mutation_summary: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ControlledInterventionEvidenceV1:
    baseline = dict(baseline_failure_context)
    signature = dict(failure_signature)
    child = dict(child_probe_receipt)
    child_measurements = dict(child_measurements)

    signature_id = str(signature.get("signature_id") or "")
    family = str(signature.get("failure_family") or "")
    proof_domain = str(signature.get("proof_domain") or "")
    if not signature_id or proof_domain != "MOTION":
        raise ValueError("STAGE41_INTERVENTION_SIGNATURE_INVALID")
    if family not in _METRICS:
        raise ValueError(
            "STAGE41_INTERVENTION_FAILURE_FAMILY_UNSUPPORTED:" + family
        )
    if float(material_improvement_margin) <= 0.0:
        raise ValueError("STAGE41_INTERVENTION_MARGIN_MUST_BE_POSITIVE")

    baseline_probe = str(baseline.get("proof_probe_fingerprint") or "")
    child_probe = str(child.get("proof_probe_fingerprint") or "")
    if not baseline_probe or not child_probe or baseline_probe != child_probe:
        raise ValueError("STAGE41_INTERVENTION_PROBE_FINGERPRINT_DRIFT")

    baseline_bindings = dict(baseline.get("bindings") or {})
    child_bindings = dict(child.get("bindings") or {})
    baseline_state = str(baseline_bindings.get("mechanical_state_binding_hash") or "")
    child_state = str(child_bindings.get("mechanical_state_binding_hash") or "")
    if not baseline_state or not child_state or baseline_state == child_state:
        raise ValueError("STAGE41_INTERVENTION_CHILD_STATE_NOT_DISTINCT")

    metric_key, higher_is_better = _METRICS[family]
    if metric_key not in child_measurements:
        raise ValueError(
            "STAGE41_INTERVENTION_CHILD_METRIC_MISSING:" + metric_key
        )
    before = float(signature["observed_value"])
    after = float(child_measurements[metric_key])

    changed = tuple(sorted(set(str(x) for x in changed_owner_ids if x)))
    owner = str(owner_id)
    if not owner or changed != (owner,):
        raise ValueError("STAGE41_INTERVENTION_NOT_SINGLE_OWNER_LOCAL")

    payload = {
        "schema": "RealSaS.Stage41ControlledInterventionSeed.v1",
        "target_signature_id": signature_id,
        "owner_id": owner,
        "baseline_state": baseline_state,
        "child_state": child_state,
        "baseline_measurement_report_hash": str(
            baseline.get("baseline_measurement_report_hash") or ""
        ),
        "probe_fingerprint": baseline_probe,
        "metric_key": metric_key,
        "before": before,
        "after": after,
        "higher_is_better": bool(higher_is_better),
        "counterfactual_status": str(counterfactual_status),
        "changed_owner_ids": changed,
        "bounded_change_passed": bool(bounded_change_passed),
        "protected_invariant_regressions": tuple(
            sorted(set(map(str, protected_invariant_regressions)))
        ),
        "mutation_summary": dict(mutation_summary or {}),
    }
    if not payload["baseline_measurement_report_hash"]:
        raise ValueError("STAGE41_INTERVENTION_BASELINE_MEASUREMENT_HASH_MISSING")

    return ControlledInterventionEvidenceV1(
        intervention_id="STAGE41_INT:" + content_sha256(payload)[:24],
        target_signature_id=signature_id,
        proof_domain="MOTION",
        owner_id=owner,
        baseline_source_product_state_hash=baseline_state,
        baseline_measurement_report_hash=payload[
            "baseline_measurement_report_hash"
        ],
        counterfactual_source_product_state_hash=child_state,
        probe_fingerprint=baseline_probe,
        changed_owner_ids=changed,
        bounded_change_passed=bool(bounded_change_passed),
        target_metric=metric_key,
        target_metric_before=before,
        target_metric_after=after,
        higher_is_better=bool(higher_is_better),
        material_improvement_margin=float(material_improvement_margin),
        baseline_status="FAIL",
        counterfactual_status=str(counterfactual_status),
        protected_invariant_regressions=tuple(
            sorted(set(map(str, protected_invariant_regressions)))
        ),
        mutation_summary=dict(mutation_summary or {}),
        metadata={
            "failure_family": family,
            "baseline_product_bindings_hash": baseline.get(
                "baseline_product_bindings_hash"
            ),
            "child_product_bindings_hash": child.get("product_bindings_hash"),
            **dict(metadata or {}),
        },
    )


__all__ = ["build_stage41_controlled_intervention_v1"]
