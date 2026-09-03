from __future__ import annotations

"""Core authority gate for counterfactual repair child product attempts.

Services may propose directives and executors may eventually produce candidate
children, but only current Compiler authority may decide whether a real
CanonicalPuppetGraph.v3 child is a lineage-correct, owner-scoped counterfactual.
"""

from dataclasses import asdict, dataclass
from fnmatch import fnmatchcase
from typing import Any

from .hashing import content_sha256
from .repair_registry import operation_authority_blockers_v1, operation_authority_v1
from .v4 import validate_product_ontology


@dataclass(frozen=True)
class RepairChildAttemptAuditV1:
    directive_id: str
    operation_id: str
    selected_owner_id: str
    parent_product_state_hash: str
    child_product_state_hash: str
    child_parent_state_hash: str | None
    actual_changed_paths: tuple[str, ...]
    allowed_change_patterns: tuple[str, ...]
    status: str
    blockers: tuple[str, ...]
    audit_hash: str
    schema_version: str = "RealSaS.RepairChildAttemptAudit.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _h(value: Any) -> str:
    return content_sha256(value)


def product_semantic_fingerprint_v1(product) -> dict[str, str]:
    """Return non-aggregate semantic leaf identities used for repair delta audit."""
    m = product.mechanical_state
    out = {
        "mechanical.surface": str(m.surface.geometry_lineage_hash),
        "mechanical.skeleton": str(m.skeleton.skeleton_lineage_hash),
        "mechanical.skin": str(m.skin.skin_lineage_hash),
        "mechanical.contract": _h({
            "equivalence": m.mechanical_equivalence_class,
            "full_3d": bool(m.full_3d_reconstruction_authority),
        }),
        "capability_contract": str(product.capability_contract.capability_contract_hash),
        "motion": str(product.motion_state.motion_state_hash),
        "runtime_policy": _h(dict(product.runtime_policy or {})),
        "editable_metadata": _h(dict(product.editable_metadata or {})),
        "product.contract": _h({
            "schema": product.schema_version,
            "representation": product.representation_class,
            "equivalence": product.mechanical_equivalence_class,
            "full_3d": bool(product.full_3d_reconstruction_authority),
            "export_contract": product.export_contract_version,
        }),
    }
    for direction in sorted(product.directional_renderables.directions, key=lambda d: int(d.view_index)):
        dp = f"directional_visual.direction.{int(direction.view_index)}"
        out[f"{dp}.camera"] = str(direction.camera_binding_hash)
        out[f"{dp}.metadata"] = _h(dict(direction.metadata or {}))
        for component in sorted(direction.components, key=lambda c: str(c.component_id)):
            cp = f"{dp}.component.{component.component_id}"
            out[f"{cp}.mesh"] = str(component.mesh.mesh_lineage_hash)
            out[f"{cp}.mesh_skin"] = str(component.mesh_skin.mesh_skin_lineage_hash)
            out[f"{cp}.appearance"] = str(component.appearance.appearance_lineage_hash)
            out[f"{cp}.render_contract"] = _h({
                "setup_order": int(component.setup_order),
                "coverage_classification": str(component.coverage_classification),
                "default_visible": bool(component.default_visible),
                "metadata": dict(component.metadata or {}),
            })
            for completion in sorted(component.completions, key=lambda c: str(c.completion_id)):
                out[f"{cp}.completion.{completion.completion_id}"] = str(completion.completion_lineage_hash)
    return out


def product_semantic_delta_v1(parent, child) -> tuple[str, ...]:
    before = product_semantic_fingerprint_v1(parent)
    after = product_semantic_fingerprint_v1(child)
    keys = set(before) | set(after)
    return tuple(sorted(k for k in keys if before.get(k) != after.get(k)))


def _matches_any(path: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatchcase(path, pattern) for pattern in patterns)


def audit_repair_child_attempt_v1(parent, child, directive) -> RepairChildAttemptAuditV1:
    """Bind a real child product to a promoted repair directive or fail closed."""
    validate_product_ontology(parent)
    validate_product_ontology(child)
    blockers: list[str] = []
    operation = directive.operation
    blockers.extend(operation_authority_blockers_v1(operation))
    try:
        authority = operation_authority_v1(operation.operation_id)
        allowed = tuple(authority.allowed_change_patterns)
    except KeyError:
        allowed = ()
    if not bool(getattr(directive, "executable", False)):
        blockers.append("repair_directive_not_executable")
    if str(directive.parent_product_state_hash) != str(parent.product_state_hash):
        blockers.append("repair_directive_parent_state_mismatch")
    if str(directive.selected_owner_id) != str(operation.owner_id):
        blockers.append("repair_directive_owner_operation_mismatch")
    if not child.parent_state_hash or str(child.parent_state_hash) != str(parent.product_state_hash):
        blockers.append("repair_child_parent_lineage_mismatch")
    if str(child.product_state_hash) == str(parent.product_state_hash):
        blockers.append("repair_child_state_not_distinct")
    changed = product_semantic_delta_v1(parent, child)
    if not changed:
        blockers.append("repair_child_has_no_semantic_change")
    outside = tuple(path for path in changed if not _matches_any(path, allowed))
    if outside:
        blockers.append("repair_child_actual_change_outside_authorized_scope:" + ",".join(outside))
    payload = {
        "directive_id": str(directive.directive_id),
        "operation_id": str(operation.operation_id),
        "selected_owner_id": str(directive.selected_owner_id),
        "parent_product_state_hash": str(parent.product_state_hash),
        "child_product_state_hash": str(child.product_state_hash),
        "child_parent_state_hash": child.parent_state_hash,
        "actual_changed_paths": list(changed),
        "allowed_change_patterns": list(allowed),
        "blockers": blockers,
    }
    return RepairChildAttemptAuditV1(
        directive_id=str(directive.directive_id),
        operation_id=str(operation.operation_id),
        selected_owner_id=str(directive.selected_owner_id),
        parent_product_state_hash=str(parent.product_state_hash),
        child_product_state_hash=str(child.product_state_hash),
        child_parent_state_hash=child.parent_state_hash,
        actual_changed_paths=changed,
        allowed_change_patterns=allowed,
        status="PASS" if not blockers else "REJECTED",
        blockers=tuple(blockers),
        audit_hash=_h(payload),
    )
