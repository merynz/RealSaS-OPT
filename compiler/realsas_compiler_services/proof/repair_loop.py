from __future__ import annotations

"""Bounded repair directives and mandatory re-proof acceptance for current RealSaS.

This service does not mutate CanonicalPuppetGraphV3. It converts already-proven
controlled causal attribution into owner-local, operation-qualified child-attempt
contracts and validates the effect of a separately produced child state after
same-probe re-proof.
"""

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from typing import Any, Iterable, Mapping

SERVICE_ID = "RealSaS.CompilerServices.BoundedRepairLoop.v1"
Json = dict[str, Any]


def _stable_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return sha256(raw).hexdigest()


@dataclass(frozen=True)
class BoundedRepairOperationV1:
    operation_id: str
    owner_id: str
    operation_family: str
    target_signature_ids: tuple[str, ...]
    qualification_hash: str
    automatic_execution_qualified: bool
    allowed_change_paths: tuple[str, ...]
    bounded_change_spec: Json
    parameters: Json = field(default_factory=dict)
    protected_invariants: tuple[str, ...] = ()
    metadata: Json = field(default_factory=dict)
    schema_version: str = "RealSaS.BoundedRepairOperation.v1"

    def to_dict(self) -> Json:
        return asdict(self)


@dataclass(frozen=True)
class RepairDirectiveV1:
    directive_id: str
    parent_product_state_hash: str
    baseline_measurement_report_hash: str
    proof_probe_fingerprint: str
    attribution_finding_id: str
    target_signature_id: str
    selected_owner_id: str
    operation: BoundedRepairOperationV1
    automatic_execution_allowed: bool
    blockers: tuple[str, ...] = ()
    metadata: Json = field(default_factory=dict)
    schema_version: str = "RealSaS.RepairDirective.v1"

    @property
    def executable(self) -> bool:
        return bool(self.automatic_execution_allowed and not self.blockers)

    def to_dict(self) -> Json:
        return asdict(self)


@dataclass(frozen=True)
class RepairApplicationRecordV1:
    application_id: str
    directive_id: str
    parent_product_state_hash: str
    child_product_state_hash: str
    child_parent_state_hash: str
    applied_operation_id: str
    changed_owner_ids: tuple[str, ...]
    changed_paths: tuple[str, ...]
    bounded_change_passed: bool
    mutation_summary: Json = field(default_factory=dict)
    metadata: Json = field(default_factory=dict)
    schema_version: str = "RealSaS.RepairApplicationRecord.v1"

    def to_dict(self) -> Json:
        return asdict(self)


@dataclass(frozen=True)
class V2VisualRepairNonRegressionEvidenceV1:
    child_product_state_hash: str
    implementation_closure_hash: str
    static_fidelity_bank_hash: str
    dynamic_fidelity_bank_hash: str
    static_fidelity_status: str
    dynamic_fidelity_status: str
    static_fidelity_passed: bool
    dynamic_fidelity_passed: bool
    metadata: Json = field(default_factory=dict)
    schema_version: str = "RealSaS.V2VisualRepairNonRegressionEvidence.v1"

    def to_dict(self) -> Json:
        return asdict(self)


def validate_v2_visual_repair_nonregression_v1(
    *,
    child_product_state_hash: str,
    evidence: V2VisualRepairNonRegressionEvidenceV1,
) -> tuple[bool, tuple[str, ...]]:
    blockers: list[str] = []
    if evidence.child_product_state_hash != str(child_product_state_hash):
        blockers.append("visual_nonregression_child_product_mismatch")
    if not evidence.implementation_closure_hash:
        blockers.append("visual_nonregression_implementation_closure_missing")
    if not evidence.static_fidelity_bank_hash:
        blockers.append("visual_nonregression_static_bank_hash_missing")
    if not evidence.dynamic_fidelity_bank_hash:
        blockers.append("visual_nonregression_dynamic_bank_hash_missing")
    if evidence.static_fidelity_status != "PASS":
        blockers.append("visual_nonregression_static_bank_not_pass")
    if evidence.dynamic_fidelity_status != "PASS":
        blockers.append("visual_nonregression_dynamic_bank_not_pass")
    if not bool(evidence.static_fidelity_passed):
        blockers.append("visual_nonregression_static_bank_failed")
    if not bool(evidence.dynamic_fidelity_passed):
        blockers.append("visual_nonregression_dynamic_bank_failed")
    # Both banks are required to describe one exact implementation closure.
    static_impl = str(evidence.metadata.get("static_implementation_closure_hash") or "")
    dynamic_impl = str(evidence.metadata.get("dynamic_implementation_closure_hash") or "")
    if (
        not static_impl
        or not dynamic_impl
        or static_impl != evidence.implementation_closure_hash
        or dynamic_impl != evidence.implementation_closure_hash
    ):
        blockers.append("visual_nonregression_cross_bank_implementation_drift")
    return not blockers, tuple(blockers)


@dataclass(frozen=True)
class RepairReproofEvidenceV1:
    directive_id: str
    child_product_state_hash: str
    child_proof_bundle_hash: str
    proof_probe_fingerprint: str
    target_signature_id: str
    target_materially_improved: bool
    target_resolved: bool
    child_domain_status: str
    protected_invariant_regressions: tuple[str, ...] = ()
    target_metric_deltas: Json = field(default_factory=dict)
    metadata: Json = field(default_factory=dict)
    schema_version: str = "RealSaS.RepairReproofEvidence.v1"

    def to_dict(self) -> Json:
        return asdict(self)


def _operation_blockers(operation: BoundedRepairOperationV1, *, selected_owner_id: str, target_signature_id: str) -> tuple[str, ...]:
    blockers: list[str] = []
    if not operation.operation_id:
        blockers.append("missing_operation_id")
    if operation.owner_id != selected_owner_id:
        blockers.append("operation_owner_mismatch")
    if target_signature_id not in set(operation.target_signature_ids):
        blockers.append("operation_does_not_target_attributed_signature")
    if not operation.qualification_hash:
        blockers.append("operation_qualification_missing")
    if not bool(operation.automatic_execution_qualified):
        blockers.append("operation_not_qualified_for_automatic_execution")
    if not operation.allowed_change_paths:
        blockers.append("operation_allowed_change_paths_empty")
    if not operation.bounded_change_spec:
        blockers.append("operation_bounded_change_spec_missing")
    return tuple(blockers)


def build_bounded_repair_directives_v1(
    *,
    attribution_finding: Mapping[str, Any],
    parent_product_state_hash: str,
    baseline_measurement_report_hash: str,
    proof_probe_fingerprint: str,
    operation_candidates: Iterable[BoundedRepairOperationV1],
) -> tuple[RepairDirectiveV1, ...]:
    """Emit one immutable child-attempt directive per qualified owner-local operation."""
    if str(attribution_finding.get("status")) != "attributed":
        return ()
    selected_owner = str(attribution_finding.get("selected_owner_id") or "")
    signature_id = str(attribution_finding.get("signature_id") or "")
    finding_id = str(attribution_finding.get("finding_id") or "")
    if not selected_owner or not signature_id or not finding_id:
        return ()
    if not bool(attribution_finding.get("automatic_repair_eligible")):
        return ()
    if not parent_product_state_hash or not baseline_measurement_report_hash or not proof_probe_fingerprint:
        raise ValueError("repair directive requires exact baseline and probe identity")

    directives: list[RepairDirectiveV1] = []
    for op in operation_candidates:
        if op.owner_id != selected_owner or signature_id not in set(op.target_signature_ids):
            continue
        blockers = _operation_blockers(op, selected_owner_id=selected_owner, target_signature_id=signature_id)
        payload = {
            "parent": parent_product_state_hash,
            "baseline_measurement": baseline_measurement_report_hash,
            "probe": proof_probe_fingerprint,
            "finding": finding_id,
            "signature": signature_id,
            "owner": selected_owner,
            "operation": op.to_dict(),
            "blockers": list(blockers),
        }
        directives.append(RepairDirectiveV1(
            directive_id="REPAIR_DIRECTIVE:" + _stable_hash(payload)[:24],
            parent_product_state_hash=str(parent_product_state_hash),
            baseline_measurement_report_hash=str(baseline_measurement_report_hash),
            proof_probe_fingerprint=str(proof_probe_fingerprint),
            attribution_finding_id=finding_id,
            target_signature_id=signature_id,
            selected_owner_id=selected_owner,
            operation=op,
            automatic_execution_allowed=not blockers,
            blockers=blockers,
            metadata={
                "service_id": SERVICE_ID,
                "one_owner_counterfactual_per_child_attempt": True,
                "in_place_product_mutation_forbidden": True,
                "mandatory_reproof": True,
                "v2_static_dynamic_visual_nonregression_required": True,
            },
        ))
    return tuple(directives)


def validate_repair_application_v1(directive: RepairDirectiveV1, application: RepairApplicationRecordV1) -> tuple[bool, tuple[str, ...]]:
    blockers: list[str] = []
    if not directive.executable:
        blockers.append("directive_not_executable")
    if application.directive_id != directive.directive_id:
        blockers.append("application_directive_mismatch")
    if application.parent_product_state_hash != directive.parent_product_state_hash:
        blockers.append("application_parent_product_mismatch")
    if not application.child_product_state_hash or application.child_product_state_hash == directive.parent_product_state_hash:
        blockers.append("repair_child_state_not_distinct")
    if application.child_parent_state_hash != directive.parent_product_state_hash:
        blockers.append("repair_child_parent_lineage_mismatch")
    if application.applied_operation_id != directive.operation.operation_id:
        blockers.append("repair_operation_mismatch")
    changed_owners = tuple(sorted(set(str(x) for x in application.changed_owner_ids if x)))
    if changed_owners != (directive.selected_owner_id,):
        blockers.append("repair_not_single_owner_local")
    if not bool(application.bounded_change_passed):
        blockers.append("repair_bounded_change_audit_failed")
    allowed = set(directive.operation.allowed_change_paths)
    changed_paths = set(str(x) for x in application.changed_paths if x)
    if not changed_paths:
        blockers.append("repair_changed_paths_empty")
    elif not changed_paths.issubset(allowed):
        blockers.append("repair_changed_path_outside_operation_scope")
    return not blockers, tuple(blockers)


def evaluate_repair_effect_v1(
    *,
    directive: RepairDirectiveV1,
    application: RepairApplicationRecordV1,
    reproof: RepairReproofEvidenceV1,
    visual_nonregression: V2VisualRepairNonRegressionEvidenceV1 | None = None,
) -> Json:
    """Accept repair effect only after exact-lineage, same-probe re-proof."""
    app_ok, app_blockers = validate_repair_application_v1(directive, application)
    blockers = list(app_blockers)
    if reproof.directive_id != directive.directive_id:
        blockers.append("reproof_directive_mismatch")
    if reproof.child_product_state_hash != application.child_product_state_hash:
        blockers.append("reproof_child_product_mismatch")
    if not reproof.child_proof_bundle_hash:
        blockers.append("child_reproof_bundle_missing")
    if reproof.proof_probe_fingerprint != directive.proof_probe_fingerprint:
        blockers.append("repair_reproof_probe_changed")
    if reproof.target_signature_id != directive.target_signature_id:
        blockers.append("repair_reproof_target_signature_mismatch")
    if not (bool(reproof.target_materially_improved) or bool(reproof.target_resolved)):
        blockers.append("repair_target_not_materially_improved")
    if reproof.protected_invariant_regressions:
        blockers.append("repair_protected_invariant_regression")
    if bool(
        directive.metadata.get(
            "v2_static_dynamic_visual_nonregression_required",
            False,
        )
    ):
        if visual_nonregression is None:
            blockers.append("repair_v2_visual_nonregression_evidence_missing")
        else:
            visual_ok, visual_blockers = validate_v2_visual_repair_nonregression_v1(
                child_product_state_hash=application.child_product_state_hash,
                evidence=visual_nonregression,
            )
            if not visual_ok:
                blockers.extend(visual_blockers)
    accepted = bool(app_ok and not blockers)
    payload = {
        "schema_version": "RealSaS.RepairEffectReport.v1",
        "service_id": SERVICE_ID,
        "directive_id": directive.directive_id,
        "parent_product_state_hash": directive.parent_product_state_hash,
        "child_product_state_hash": application.child_product_state_hash,
        "child_proof_bundle_hash": reproof.child_proof_bundle_hash,
        "proof_probe_fingerprint": reproof.proof_probe_fingerprint,
        "target_signature_id": directive.target_signature_id,
        "selected_owner_id": directive.selected_owner_id,
        "operation_id": directive.operation.operation_id,
        "target_materially_improved": bool(reproof.target_materially_improved),
        "target_resolved": bool(reproof.target_resolved),
        "child_domain_status": str(reproof.child_domain_status),
        "target_metric_deltas": dict(reproof.target_metric_deltas),
        "protected_invariant_regressions": list(reproof.protected_invariant_regressions),
        "repair_accepted": accepted,
        "blockers": blockers,
        "same_probe_reproof_required": True,
        "v2_static_dynamic_visual_nonregression_required": bool(
            directive.metadata.get(
                "v2_static_dynamic_visual_nonregression_required",
                False,
            )
        ),
        "visual_nonregression_evidence": (
            None
            if visual_nonregression is None
            else visual_nonregression.to_dict()
        ),
        "in_place_product_mutation_forbidden": True,
    }
    payload["report_id"] = "REPAIR_EFFECT:" + _stable_hash(payload)[:24]
    return payload
