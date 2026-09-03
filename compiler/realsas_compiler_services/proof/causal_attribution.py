from __future__ import annotations

"""Controlled-intervention causal owner attribution for current RealSaS proof state.

This service is deliberately subordinate. A failed invariant is not an owner label.
An owner may be attributed only when a bounded, single-owner counterfactual child
attempt is replayed under the same proof probe/policy and materially improves the
target failure without protected-invariant regression.
"""

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from typing import Any, Iterable, Mapping

SERVICE_ID = "RealSaS.CompilerServices.ControlledOwnerAttribution.v1"
SCHEMA = "RealSaS.ControlledOwnerAttributionFinding.v1"

Json = dict[str, Any]


def _stable_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return sha256(raw).hexdigest()


def proof_probe_fingerprint_v1(*, operator_policy_hashes: Iterable[str], probe_specification: Mapping[str, Any]) -> str:
    return _stable_hash({
        "schema": "RealSaS.ProofProbeFingerprint.v1",
        "operator_policy_hashes": list(operator_policy_hashes),
        "probe_specification": dict(probe_specification),
    })


@dataclass(frozen=True)
class ControlledInterventionEvidenceV1:
    intervention_id: str
    target_signature_id: str
    proof_domain: str
    owner_id: str
    baseline_source_product_state_hash: str
    baseline_measurement_report_hash: str
    counterfactual_source_product_state_hash: str
    probe_fingerprint: str
    changed_owner_ids: tuple[str, ...]
    bounded_change_passed: bool
    target_metric: str
    target_metric_before: float
    target_metric_after: float
    higher_is_better: bool
    material_improvement_margin: float
    baseline_status: str
    counterfactual_status: str
    protected_invariant_regressions: tuple[str, ...] = ()
    mutation_summary: Json = field(default_factory=dict)
    metadata: Json = field(default_factory=dict)
    schema_version: str = "RealSaS.ControlledInterventionEvidence.v1"

    def to_dict(self) -> Json:
        return asdict(self)


def _signed_improvement(e: ControlledInterventionEvidenceV1) -> float:
    before = float(e.target_metric_before)
    after = float(e.target_metric_after)
    return after - before if bool(e.higher_is_better) else before - after


def _materially_improves(e: ControlledInterventionEvidenceV1) -> bool:
    status_improved = str(e.baseline_status) != "PASS" and str(e.counterfactual_status) == "PASS"
    return bool(status_improved or _signed_improvement(e) >= float(e.material_improvement_margin))


def _validate_intervention(
    e: ControlledInterventionEvidenceV1,
    *,
    target_signature_id: str,
    proof_domain: str,
    baseline_source_product_state_hash: str,
    baseline_measurement_report_hash: str,
    probe_fingerprint: str,
) -> tuple[bool, tuple[str, ...]]:
    blockers: list[str] = []
    if not e.intervention_id:
        blockers.append("missing_intervention_id")
    if e.target_signature_id != target_signature_id:
        blockers.append("target_signature_mismatch")
    if e.proof_domain != proof_domain:
        blockers.append("proof_domain_mismatch")
    if e.baseline_source_product_state_hash != baseline_source_product_state_hash:
        blockers.append("baseline_product_state_mismatch")
    if e.baseline_measurement_report_hash != baseline_measurement_report_hash:
        blockers.append("baseline_measurement_report_mismatch")
    if not e.counterfactual_source_product_state_hash or e.counterfactual_source_product_state_hash == baseline_source_product_state_hash:
        blockers.append("counterfactual_child_state_not_distinct")
    if e.probe_fingerprint != probe_fingerprint:
        blockers.append("probe_specification_changed")
    changed = tuple(sorted(set(str(x) for x in e.changed_owner_ids if x)))
    if changed != (str(e.owner_id),):
        blockers.append("counterfactual_not_single_owner_local")
    if not bool(e.bounded_change_passed):
        blockers.append("bounded_change_policy_failed")
    if float(e.material_improvement_margin) < 0.0:
        blockers.append("negative_material_improvement_margin")
    if e.protected_invariant_regressions:
        blockers.append("protected_invariant_regression")
    return not blockers, tuple(blockers)


def attribute_signature_owner_v1(
    *,
    signature: Mapping[str, Any],
    proof_domain: str,
    baseline_source_product_state_hash: str,
    baseline_measurement_report_hash: str,
    operator_policy_hashes: Iterable[str],
    probe_specification: Mapping[str, Any],
    interventions: Iterable[ControlledInterventionEvidenceV1],
) -> Json:
    """Attribute one failure signature from controlled interventions or abstain."""
    signature_id = str(signature.get("signature_id", ""))
    if not signature_id:
        raise ValueError("failure signature requires signature_id")
    fingerprint = proof_probe_fingerprint_v1(
        operator_policy_hashes=operator_policy_hashes,
        probe_specification=probe_specification,
    )
    relevant = [e for e in interventions if e.target_signature_id == signature_id]
    valid_rows: list[tuple[ControlledInterventionEvidenceV1, float, bool]] = []
    rejected: list[Json] = []
    for e in relevant:
        valid, blockers = _validate_intervention(
            e,
            target_signature_id=signature_id,
            proof_domain=str(proof_domain),
            baseline_source_product_state_hash=str(baseline_source_product_state_hash),
            baseline_measurement_report_hash=str(baseline_measurement_report_hash),
            probe_fingerprint=fingerprint,
        )
        if not valid:
            rejected.append({"intervention_id": e.intervention_id, "owner_id": e.owner_id, "blockers": list(blockers)})
            continue
        valid_rows.append((e, _signed_improvement(e), _materially_improves(e)))

    improving = [(e, delta) for e, delta, material in valid_rows if material]
    improving_owners = sorted(set(e.owner_id for e, _ in improving))
    selected_owner = None
    status = "abstained"
    reason = "no_controlled_owner_intervention_materially_improves"
    if len(improving_owners) == 1:
        selected_owner = improving_owners[0]
        status = "attributed"
        reason = None
    elif len(improving_owners) > 1:
        reason = "multiple_owner_counterfactuals_materially_improve"
    elif not relevant:
        reason = "no_controlled_intervention_evidence"
    elif relevant and not valid_rows:
        reason = "all_controlled_interventions_rejected"

    evidence_rows = [
        {
            "intervention_id": e.intervention_id,
            "owner_id": e.owner_id,
            "target_metric": e.target_metric,
            "target_metric_before": float(e.target_metric_before),
            "target_metric_after": float(e.target_metric_after),
            "signed_improvement": float(delta),
            "material_improvement_margin": float(e.material_improvement_margin),
            "materially_improved": bool(material),
            "baseline_status": e.baseline_status,
            "counterfactual_status": e.counterfactual_status,
            "counterfactual_source_product_state_hash": e.counterfactual_source_product_state_hash,
            "mutation_summary": dict(e.mutation_summary),
        }
        for e, delta, material in valid_rows
    ]
    payload = {
        "schema_version": SCHEMA,
        "service_id": SERVICE_ID,
        "signature_id": signature_id,
        "failure_family": signature.get("failure_family"),
        "proof_domain": str(proof_domain),
        "selected_owner_id": selected_owner,
        "status": status,
        "abstention_reason": reason,
        "probe_fingerprint": fingerprint,
        "baseline_source_product_state_hash": str(baseline_source_product_state_hash),
        "baseline_measurement_report_hash": str(baseline_measurement_report_hash),
        "controlled_interventions": evidence_rows,
        "rejected_interventions": rejected,
        "automatic_repair_eligible": bool(status == "attributed"),
        "causal_owner_requires_intervention": True,
        "failure_label_alone_is_not_owner_evidence": True,
    }
    payload["finding_id"] = "OWNER_ATTR:" + _stable_hash(payload)[:24]
    return payload


def build_controlled_owner_attribution_v1(
    *,
    failure_signatures: Iterable[Mapping[str, Any]],
    proof_domain: str,
    baseline_source_product_state_hash: str,
    baseline_measurement_report_hash: str,
    operator_policy_hashes: Iterable[str],
    probe_specification: Mapping[str, Any],
    interventions: Iterable[ControlledInterventionEvidenceV1],
) -> tuple[Json, ...]:
    rows = tuple(interventions)
    return tuple(
        attribute_signature_owner_v1(
            signature=s,
            proof_domain=proof_domain,
            baseline_source_product_state_hash=baseline_source_product_state_hash,
            baseline_measurement_report_hash=baseline_measurement_report_hash,
            operator_policy_hashes=operator_policy_hashes,
            probe_specification=probe_specification,
            interventions=rows,
        )
        for s in failure_signatures
    )
