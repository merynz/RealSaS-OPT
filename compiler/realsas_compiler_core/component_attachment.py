from __future__ import annotations

"""Typed Compiler authority for visible component mechanics and attachments.

The module deliberately separates evidence from qualification.  It never infers
mechanical class, parent joint, socket semantics or detachability from a filename,
semantic label, or a one-hot skin row alone.  A current product may embed several
logical source components inside one render mesh; geometry_membership_refs preserve
that ownership without creating a second mesh authority.
"""

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping

from .hashing import content_sha256
from .types import Json, QualificationError
from .v4 import directional_visual_state_hash


_MECHANICAL_CLASSES = {
    "DEFORMABLE_COMPONENT",
    "RIGID_SKINNED_COMPONENT",
    "RIGID_BONE_ATTACHMENT",
    "NON_MECHANICAL_VISUAL_COMPONENT",
    "EXCLUDED_SOURCE_COMPONENT",
}
_DETACHABILITY_CLASSES = {
    "FIXED_COMPONENT",
    "DETACHABLE_COMPONENT",
    "SWAPPABLE_SLOT_COMPONENT",
    "DETACHABILITY_UNKNOWN",
}
_REQUIRED_VIEWS = set(range(8))


@dataclass(frozen=True)
class ComponentAttachmentEvidenceIR:
    component_id: str
    source_provenance_refs: tuple[str, ...]
    mechanical_class: str
    directional_render_membership: tuple[tuple[int, str], ...]
    geometry_membership_refs: tuple[str, ...]
    geometry_lineage_hash: str
    skin_deformer_lineage_hash: str
    canonical_parent_joint_id: str
    socket_id: str
    bind_state_authority_hash: str
    detachability_class: str
    visible_required: bool
    one_hot_carry_verified: bool
    one_hot_carry_joint_id: str
    visual_only_qualification_hash: str
    exclusion_reason: str
    qualification_evidence_hash: str
    schema_version: str = "RealSaS.ComponentAttachmentEvidenceIR.v1"
    metadata: Json = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class QualifiedComponentAttachmentIR:
    component_id: str
    source_provenance_refs: tuple[str, ...]
    product_view_membership: tuple[int, ...]
    directional_render_membership: tuple[tuple[int, str], ...]
    mechanical_class: str
    canonical_parent_joint_id: str
    socket_id: str
    bind_state_authority_hash: str
    geometry_membership_refs: tuple[str, ...]
    geometry_lineage_hash: str
    skin_deformer_lineage_hash: str
    detachability_class: str
    visible_required: bool
    qualification_report: Json
    component_lineage_hash: str
    schema_version: str = "RealSaS.QualifiedComponentAttachmentIR.v1"
    metadata: Json = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class QualifiedComponentAssemblyIR:
    components: tuple[QualifiedComponentAttachmentIR, ...]
    required_visible_component_ids: tuple[str, ...]
    qualification_report: Json
    component_assembly_hash: str
    schema_version: str = "RealSaS.QualifiedComponentAssemblyIR.v1"
    metadata: Json = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def _hash_without(value, field_name: str) -> str:
    payload = value.to_dict()
    payload.pop(field_name, None)
    return content_sha256(payload)


def component_attachment_lineage_hash(value: QualifiedComponentAttachmentIR) -> str:
    return _hash_without(value, "component_lineage_hash")


def component_assembly_hash(value: QualifiedComponentAssemblyIR) -> str:
    return _hash_without(value, "component_assembly_hash")


def _canonical_joint_ids(skeleton) -> set[str]:
    ids = {str(j.canonical_joint_id) for j in tuple(getattr(skeleton, "joints", ())) }
    if not ids:
        raise QualificationError("COMPONENT_ATTACHMENT_REQUIRES_QUALIFIED_SKELETON")
    return ids


def _render_membership_map(directional_renderables) -> dict[int, set[str]]:
    result: dict[int, set[str]] = {}
    for direction in tuple(getattr(directional_renderables, "directions", ())):
        view = int(direction.view_index)
        if view in result:
            raise QualificationError("COMPONENT_ASSEMBLY_DUPLICATE_DIRECTION")
        result[view] = {str(component.component_id) for component in tuple(direction.components)}
    if result and set(result) != _REQUIRED_VIEWS:
        raise QualificationError("COMPONENT_ASSEMBLY_RENDERABLE_SET_MUST_HAVE_8_VIEWS")
    return result


def _validate_source_evidence(evidence: ComponentAttachmentEvidenceIR) -> None:
    if not evidence.component_id:
        raise QualificationError("COMPONENT_ATTACHMENT_ID_REQUIRED")
    if not evidence.source_provenance_refs or any(not str(x) for x in evidence.source_provenance_refs):
        raise QualificationError("COMPONENT_ATTACHMENT_SOURCE_PROVENANCE_REQUIRED")
    if evidence.mechanical_class not in _MECHANICAL_CLASSES:
        raise QualificationError("COMPONENT_ATTACHMENT_INVALID_MECHANICAL_CLASS")
    if evidence.detachability_class not in _DETACHABILITY_CLASSES:
        raise QualificationError("COMPONENT_ATTACHMENT_INVALID_DETACHABILITY_CLASS")
    if not evidence.qualification_evidence_hash:
        raise QualificationError("COMPONENT_ATTACHMENT_QUALIFICATION_EVIDENCE_HASH_REQUIRED")

    pairs = tuple((int(v), str(c)) for v, c in evidence.directional_render_membership)
    if len(pairs) != len(set(pairs)):
        raise QualificationError("COMPONENT_ATTACHMENT_DUPLICATE_RENDER_MEMBERSHIP")
    if any(v not in _REQUIRED_VIEWS or not cid for v, cid in pairs):
        raise QualificationError("COMPONENT_ATTACHMENT_INVALID_RENDER_MEMBERSHIP")
    if evidence.visible_required and not pairs:
        raise QualificationError("COMPONENT_ATTACHMENT_VISIBLE_COMPONENT_REQUIRES_RENDER_MEMBERSHIP")
    if evidence.visible_required and evidence.detachability_class == "DETACHABILITY_UNKNOWN":
        raise QualificationError("COMPONENT_ATTACHMENT_REQUIRED_VISIBLE_DETACHABILITY_UNRESOLVED")

    mechanical = evidence.mechanical_class
    if mechanical == "EXCLUDED_SOURCE_COMPONENT":
        if evidence.visible_required:
            raise QualificationError("COMPONENT_ATTACHMENT_REQUIRED_VISIBLE_COMPONENT_CANNOT_BE_EXCLUDED")
        if pairs:
            raise QualificationError("COMPONENT_ATTACHMENT_EXCLUDED_COMPONENT_CANNOT_BE_RENDERED")
        if not evidence.exclusion_reason:
            raise QualificationError("COMPONENT_ATTACHMENT_EXCLUSION_REASON_REQUIRED")
        return

    if not evidence.geometry_membership_refs or not evidence.geometry_lineage_hash:
        raise QualificationError("COMPONENT_ATTACHMENT_GEOMETRY_MEMBERSHIP_AUTHORITY_REQUIRED")

    if mechanical == "DEFORMABLE_COMPONENT":
        if not evidence.skin_deformer_lineage_hash:
            raise QualificationError("COMPONENT_ATTACHMENT_DEFORMABLE_REQUIRES_SKIN_DEFORMER_LINEAGE")
    elif mechanical == "RIGID_SKINNED_COMPONENT":
        if not evidence.skin_deformer_lineage_hash:
            raise QualificationError("COMPONENT_ATTACHMENT_RIGID_SKINNED_REQUIRES_SKIN_LINEAGE")
        if not evidence.canonical_parent_joint_id:
            raise QualificationError("COMPONENT_ATTACHMENT_RIGID_SKINNED_REQUIRES_PARENT_JOINT")
        if not evidence.one_hot_carry_verified:
            raise QualificationError("COMPONENT_ATTACHMENT_RIGID_SKINNED_REQUIRES_VERIFIED_ONE_HOT_CARRY")
        if evidence.one_hot_carry_joint_id != evidence.canonical_parent_joint_id:
            raise QualificationError("COMPONENT_ATTACHMENT_ONE_HOT_PARENT_JOINT_MISMATCH")
    elif mechanical == "RIGID_BONE_ATTACHMENT":
        if not evidence.canonical_parent_joint_id:
            raise QualificationError("COMPONENT_ATTACHMENT_RIGID_BONE_REQUIRES_PARENT_JOINT")
        if not evidence.bind_state_authority_hash:
            raise QualificationError("COMPONENT_ATTACHMENT_RIGID_BONE_REQUIRES_BIND_AUTHORITY")
    elif mechanical == "NON_MECHANICAL_VISUAL_COMPONENT":
        if not evidence.visual_only_qualification_hash:
            raise QualificationError("COMPONENT_ATTACHMENT_VISUAL_ONLY_QUALIFICATION_REQUIRED")
        if evidence.canonical_parent_joint_id or evidence.socket_id:
            raise QualificationError("COMPONENT_ATTACHMENT_VISUAL_ONLY_CANNOT_CLAIM_MECHANICAL_BINDING")


def qualify_component_attachment(
    evidence: ComponentAttachmentEvidenceIR,
    skeleton,
    *,
    qualification_report: Mapping[str, Any],
) -> QualifiedComponentAttachmentIR:
    """Qualify one logical source component without semantic-name inference."""
    _validate_source_evidence(evidence)
    report = dict(qualification_report)
    if not bool(report.get("passed", False)):
        raise QualificationError("COMPONENT_ATTACHMENT_QUALIFICATION_FAILED")
    if not bool(report.get("source_authority_verified", False)):
        raise QualificationError("COMPONENT_ATTACHMENT_SOURCE_AUTHORITY_NOT_VERIFIED")
    if bool(report.get("classification_inferred_from_filename", False)):
        raise QualificationError("COMPONENT_ATTACHMENT_FILENAME_CLASSIFICATION_FORBIDDEN")

    joint_ids = _canonical_joint_ids(skeleton)
    if evidence.canonical_parent_joint_id and evidence.canonical_parent_joint_id not in joint_ids:
        raise QualificationError("COMPONENT_ATTACHMENT_PARENT_JOINT_NOT_CANONICAL")
    if evidence.one_hot_carry_joint_id and evidence.one_hot_carry_joint_id not in joint_ids:
        raise QualificationError("COMPONENT_ATTACHMENT_ONE_HOT_JOINT_NOT_CANONICAL")

    if evidence.mechanical_class == "DEFORMABLE_COMPONENT" and not bool(report.get("deformable_multi_joint_support", False)):
        raise QualificationError("COMPONENT_ATTACHMENT_DEFORMABLE_MULTI_JOINT_PROOF_REQUIRED")
    if evidence.mechanical_class == "RIGID_BONE_ATTACHMENT" and not bool(report.get("explicit_bone_or_socket_binding", False)):
        raise QualificationError("COMPONENT_ATTACHMENT_EXPLICIT_BONE_BINDING_PROOF_REQUIRED")

    views = tuple(sorted({int(v) for v, _ in evidence.directional_render_membership}))
    value = QualifiedComponentAttachmentIR(
        component_id=str(evidence.component_id),
        source_provenance_refs=tuple(str(x) for x in evidence.source_provenance_refs),
        product_view_membership=views,
        directional_render_membership=tuple(sorted((int(v), str(c)) for v, c in evidence.directional_render_membership)),
        mechanical_class=str(evidence.mechanical_class),
        canonical_parent_joint_id=str(evidence.canonical_parent_joint_id),
        socket_id=str(evidence.socket_id),
        bind_state_authority_hash=str(evidence.bind_state_authority_hash),
        geometry_membership_refs=tuple(str(x) for x in evidence.geometry_membership_refs),
        geometry_lineage_hash=str(evidence.geometry_lineage_hash),
        skin_deformer_lineage_hash=str(evidence.skin_deformer_lineage_hash),
        detachability_class=str(evidence.detachability_class),
        visible_required=bool(evidence.visible_required),
        qualification_report={
            **report,
            "qualification_evidence_hash": str(evidence.qualification_evidence_hash),
            "one_hot_carry_verified": bool(evidence.one_hot_carry_verified),
            "one_hot_carry_joint_id": str(evidence.one_hot_carry_joint_id),
            "visual_only_qualification_hash": str(evidence.visual_only_qualification_hash),
            "exclusion_reason": str(evidence.exclusion_reason),
        },
        component_lineage_hash="",
        metadata=dict(evidence.metadata),
    )
    value = replace(value, component_lineage_hash=component_attachment_lineage_hash(value))
    validate_qualified_component_attachment(value, skeleton)
    return value


def validate_qualified_component_attachment(value: QualifiedComponentAttachmentIR, skeleton) -> None:
    if value.mechanical_class not in _MECHANICAL_CLASSES:
        raise QualificationError("QUALIFIED_COMPONENT_INVALID_MECHANICAL_CLASS")
    if value.detachability_class not in _DETACHABILITY_CLASSES:
        raise QualificationError("QUALIFIED_COMPONENT_INVALID_DETACHABILITY_CLASS")
    if not bool(value.qualification_report.get("passed", False)):
        raise QualificationError("QUALIFIED_COMPONENT_REPORT_NOT_PASS")
    if not bool(value.qualification_report.get("source_authority_verified", False)):
        raise QualificationError("QUALIFIED_COMPONENT_SOURCE_AUTHORITY_NOT_VERIFIED")
    if bool(value.qualification_report.get("classification_inferred_from_filename", False)):
        raise QualificationError("QUALIFIED_COMPONENT_FILENAME_CLASSIFICATION_FORBIDDEN")
    if value.visible_required and value.detachability_class == "DETACHABILITY_UNKNOWN":
        raise QualificationError("QUALIFIED_COMPONENT_REQUIRED_VISIBLE_DETACHABILITY_UNRESOLVED")

    joint_ids = _canonical_joint_ids(skeleton)
    if value.canonical_parent_joint_id and value.canonical_parent_joint_id not in joint_ids:
        raise QualificationError("QUALIFIED_COMPONENT_PARENT_JOINT_NOT_CANONICAL")
    if value.mechanical_class in {"RIGID_SKINNED_COMPONENT", "RIGID_BONE_ATTACHMENT"} and not value.canonical_parent_joint_id:
        raise QualificationError("QUALIFIED_COMPONENT_RIGID_PARENT_REQUIRED")
    if value.mechanical_class == "RIGID_SKINNED_COMPONENT":
        if not bool(value.qualification_report.get("one_hot_carry_verified", False)):
            raise QualificationError("QUALIFIED_COMPONENT_RIGID_CARRY_NOT_VERIFIED")
        if value.qualification_report.get("one_hot_carry_joint_id") != value.canonical_parent_joint_id:
            raise QualificationError("QUALIFIED_COMPONENT_RIGID_CARRY_JOINT_MISMATCH")
    if value.mechanical_class == "DEFORMABLE_COMPONENT" and not value.skin_deformer_lineage_hash:
        raise QualificationError("QUALIFIED_COMPONENT_DEFORMER_LINEAGE_REQUIRED")
    if value.mechanical_class == "NON_MECHANICAL_VISUAL_COMPONENT" and not value.qualification_report.get("visual_only_qualification_hash"):
        raise QualificationError("QUALIFIED_COMPONENT_VISUAL_ONLY_PROOF_REQUIRED")
    if value.mechanical_class == "EXCLUDED_SOURCE_COMPONENT":
        if value.visible_required or value.directional_render_membership:
            raise QualificationError("QUALIFIED_COMPONENT_EXCLUSION_CONFLICT")
        if not value.qualification_report.get("exclusion_reason"):
            raise QualificationError("QUALIFIED_COMPONENT_EXCLUSION_REASON_REQUIRED")

    if value.component_lineage_hash != component_attachment_lineage_hash(value):
        raise QualificationError("QUALIFIED_COMPONENT_LINEAGE_HASH_MISMATCH")


def validate_component_assembly(
    value: QualifiedComponentAssemblyIR,
    skeleton,
    *,
    directional_renderables=None,
) -> None:
    component_ids = [component.component_id for component in value.components]
    if not component_ids or len(component_ids) != len(set(component_ids)):
        raise QualificationError("COMPONENT_ASSEMBLY_REQUIRES_UNIQUE_COMPONENTS")
    for component in value.components:
        validate_qualified_component_attachment(component, skeleton)

    required = set(value.required_visible_component_ids)
    if len(required) != len(value.required_visible_component_ids):
        raise QualificationError("COMPONENT_ASSEMBLY_DUPLICATE_REQUIRED_COMPONENT")
    by_id = {component.component_id: component for component in value.components}
    if not required.issubset(by_id):
        raise QualificationError("COMPONENT_ASSEMBLY_REQUIRED_COMPONENT_MISSING")
    for component_id in required:
        component = by_id[component_id]
        if not component.visible_required:
            raise QualificationError("COMPONENT_ASSEMBLY_REQUIRED_COMPONENT_NOT_MARKED_VISIBLE_REQUIRED")
        if component.mechanical_class == "EXCLUDED_SOURCE_COMPONENT":
            raise QualificationError("COMPONENT_ASSEMBLY_REQUIRED_COMPONENT_EXCLUDED")

    if directional_renderables is not None:
        available = _render_membership_map(directional_renderables)
        for component in value.components:
            for view_index, render_component_id in component.directional_render_membership:
                if render_component_id not in available.get(int(view_index), set()):
                    raise QualificationError("COMPONENT_ASSEMBLY_RENDER_MEMBERSHIP_NOT_PRESENT")

    report = value.qualification_report
    if not bool(report.get("passed", False)):
        raise QualificationError("COMPONENT_ASSEMBLY_REPORT_NOT_PASS")
    if float(report.get("visible_required_accounting", 0.0)) != 1.0:
        raise QualificationError("COMPONENT_ASSEMBLY_VISIBLE_REQUIRED_ACCOUNTING_INCOMPLETE")
    if value.component_assembly_hash != component_assembly_hash(value):
        raise QualificationError("COMPONENT_ASSEMBLY_HASH_MISMATCH")


def qualify_component_assembly(
    components: tuple[QualifiedComponentAttachmentIR, ...],
    skeleton,
    *,
    required_visible_component_ids: tuple[str, ...],
    directional_renderables=None,
    metadata: Mapping[str, Any] | None = None,
) -> QualifiedComponentAssemblyIR:
    required = tuple(sorted(str(x) for x in required_visible_component_ids))
    by_id = {component.component_id: component for component in components}
    if len(by_id) != len(components):
        raise QualificationError("COMPONENT_ASSEMBLY_DUPLICATE_COMPONENT_ID")
    accounted = sum(
        1
        for component_id in required
        if component_id in by_id
        and by_id[component_id].visible_required
        and by_id[component_id].mechanical_class != "EXCLUDED_SOURCE_COMPONENT"
    )
    fraction = 1.0 if not required else float(accounted) / float(len(required))
    value = QualifiedComponentAssemblyIR(
        components=tuple(sorted(components, key=lambda component: component.component_id)),
        required_visible_component_ids=required,
        qualification_report={
            "passed": bool(fraction == 1.0),
            "visible_required_accounting": fraction,
            "required_visible_count": len(required),
            "accounted_visible_count": accounted,
            "silent_disappearance_forbidden": True,
            "rigidity_does_not_imply_detachability": True,
        },
        component_assembly_hash="",
        metadata=dict(metadata or {}),
    )
    value = replace(value, component_assembly_hash=component_assembly_hash(value))
    validate_component_assembly(value, skeleton, directional_renderables=directional_renderables)
    return value


def _directional_component_authority_signature(directional_renderables):
    rows = []
    for direction in sorted(
        directional_renderables.directions,
        key=lambda row: int(row.view_index),
    ):
        for component in sorted(
            direction.components,
            key=lambda row: (int(row.setup_order), str(row.component_id)),
        ):
            rows.append((
                int(direction.view_index),
                str(component.component_id),
                str(component.component_state_hash),
                str(component.mesh.mesh_lineage_hash),
                str(component.mesh_skin.mesh_skin_lineage_hash),
                str(component.appearance.appearance_lineage_hash),
            ))
    return tuple(rows)


def bind_component_assembly_to_directional_renderable_set(
    directional_renderables,
    component_assembly: QualifiedComponentAssemblyIR,
    skeleton,
):
    """Hash typed component ownership into the directional/product visual state.

    CanonicalPuppetGraph.v3 already commits to directional_visual_state_hash, so this
    narrow binding makes component assembly part of product-state identity without
    relabelling the scientific MechanicalStateIR surface or changing mesh authority.
    """
    validate_component_assembly(
        component_assembly,
        skeleton,
        directional_renderables=directional_renderables,
    )
    before_authority = _directional_component_authority_signature(
        directional_renderables
    )
    metadata = dict(getattr(directional_renderables, "metadata", {}) or {})
    metadata["qualified_component_assembly"] = component_assembly.to_dict()
    metadata["component_assembly_hash"] = component_assembly.component_assembly_hash
    metadata["component_assembly_authority"] = "COMPILER_QUALIFIED_TYPED_ATTACHMENT_V1"
    rebound = replace(
        directional_renderables,
        directional_visual_state_hash="",
        metadata=metadata,
    )
    rebound = replace(
        rebound,
        directional_visual_state_hash=directional_visual_state_hash(rebound),
    )
    after_authority = _directional_component_authority_signature(rebound)
    if after_authority != before_authority:
        raise QualificationError(
            "COMPONENT_ASSEMBLY_MUTATED_DIRECTIONAL_COMPONENT_AUTHORITY"
        )
    return rebound
