from __future__ import annotations

from dataclasses import replace

from .component_attachment import (
    ComponentEvidenceIR,
    QualifiedComponentIR,
    QualifiedComponentSetIR,
    qualify_component_set_v1,
    qualified_component_lineage_hash,
    qualified_component_set_lineage_hash,
)
from .types import QualificationError


def qualify_component_set_against_mechanical_v2(
    evidence_rows: tuple[ComponentEvidenceIR, ...],
    *,
    surface,
    skeleton,
    skin,
    min_rigid_owner_weight: float = 0.999,
    max_rigid_other_mass: float = 0.001,
    require_full_visible_surface_accounting: bool = True,
) -> QualifiedComponentSetIR:
    """Qualify component/attachment semantics against the exact current mechanics.

    V1 establishes typed provenance/class legality. V2 additionally proves that the
    declared component membership exists on the exact current surface and that a
    RIGID_SKINNED_COMPONENT is actually rigid under the exact current QualifiedSkinIR.
    It also forbids silent visible-surface disappearance when requested.
    """
    if not (0.0 < float(min_rigid_owner_weight) <= 1.0):
        raise ValueError("min_rigid_owner_weight must be within (0,1]")
    if not (0.0 <= float(max_rigid_other_mass) < 1.0):
        raise ValueError("max_rigid_other_mass must be within [0,1)")

    if skin.surface_binding_hash != surface.geometry_lineage_hash:
        raise QualificationError("COMPONENT_V2_SKIN_SURFACE_LINEAGE_MISMATCH")
    if skin.skeleton_binding_hash != skeleton.skeleton_lineage_hash:
        raise QualificationError("COMPONENT_V2_SKIN_SKELETON_LINEAGE_MISMATCH")

    known_surface_ids = {str(node.surface_id) for node in surface.surface_nodes}
    known_joint_ids = {str(joint.canonical_joint_id) for joint in skeleton.joints}
    base = qualify_component_set_v1(
        evidence_rows,
        known_surface_ids=known_surface_ids,
        known_joint_ids=known_joint_ids,
        surface_lineage_hash=surface.geometry_lineage_hash,
        skeleton_lineage_hash=skeleton.skeleton_lineage_hash,
        skin_lineage_hash=skin.skin_lineage_hash,
    )

    skin_rows = {str(row.surface_id): row for row in skin.rows}
    if len(skin_rows) != len(skin.rows):
        raise QualificationError("COMPONENT_V2_DUPLICATE_SKIN_SURFACE_ROW")

    surface_owner: dict[str, str] = {}
    qualified_rows: list[QualifiedComponentIR] = []
    rigid_verified = 0
    for component in base.components:
        for sid in component.surface_ids:
            old = surface_owner.get(sid)
            if old is not None and old != component.component_id:
                raise QualificationError(
                    f"COMPONENT_V2_SURFACE_MEMBERSHIP_OVERLAP:{sid}:{old}:{component.component_id}"
                )
            surface_owner[sid] = component.component_id

        report = dict(component.qualification_report)
        report.update({
            "mechanical_evidence_verified": True,
            "exact_surface_lineage_verified": True,
            "exact_skeleton_lineage_verified": True,
            "exact_skin_lineage_verified": True,
        })

        if component.mechanical_class == "RIGID_SKINNED_COMPONENT":
            owner = str(component.parent_joint_id or "")
            if not owner:
                raise QualificationError(
                    f"COMPONENT_V2_RIGID_SKINNED_REQUIRES_CANONICAL_PARENT:{component.component_id}"
                )
            if owner not in known_joint_ids:
                raise QualificationError(
                    f"COMPONENT_V2_RIGID_OWNER_UNKNOWN:{component.component_id}:{owner}"
                )
            min_owner = 1.0
            max_other = 0.0
            for sid in component.surface_ids:
                row = skin_rows.get(sid)
                if row is None:
                    raise QualificationError(
                        f"COMPONENT_V2_RIGID_SKIN_ROW_MISSING:{component.component_id}:{sid}"
                    )
                influences = dict(row.influences)
                owner_weight = float(influences.get(owner, 0.0))
                other_mass = float(sum(float(w) for jid, w in row.influences if str(jid) != owner))
                min_owner = min(min_owner, owner_weight)
                max_other = max(max_other, other_mass)
            if min_owner < float(min_rigid_owner_weight) or max_other > float(max_rigid_other_mass):
                raise QualificationError(
                    "COMPONENT_V2_RIGID_SKIN_EVIDENCE_FAIL:"
                    f"{component.component_id}:owner_min={min_owner}:other_max={max_other}"
                )
            report.update({
                "rigid_skin_evidence_verified": True,
                "rigid_owner_joint_id": owner,
                "min_rigid_owner_weight": float(min_owner),
                "max_rigid_other_mass": float(max_other),
                "rigid_owner_weight_threshold": float(min_rigid_owner_weight),
                "rigid_other_mass_threshold": float(max_rigid_other_mass),
            })
            rigid_verified += 1
        elif component.mechanical_class == "DEFORMABLE_COMPONENT":
            missing = tuple(sid for sid in component.surface_ids if sid not in skin_rows)
            if missing:
                raise QualificationError(
                    f"COMPONENT_V2_DEFORMABLE_SKIN_ROW_MISSING:{component.component_id}:{missing[:4]}"
                )
            report["deformable_skin_rows_verified"] = len(component.surface_ids)
        elif component.mechanical_class == "RIGID_BONE_ATTACHMENT":
            # A bone/socket attachment is not inferred from skin weights. The V1 bind
            # authority remains the explicit source of truth.
            report["rigid_bone_binding_verified_without_skin_inference"] = True

        updated = replace(component, qualification_report=report, component_lineage_hash="")
        updated = replace(updated, component_lineage_hash=qualified_component_lineage_hash(updated))
        qualified_rows.append(updated)

    visible_surface_ids = {
        str(node.surface_id) for node in surface.surface_nodes if tuple(node.support_views)
    }
    accounted_visible = visible_surface_ids.intersection(surface_owner)
    missing_visible = tuple(sorted(visible_surface_ids - accounted_visible))
    accounting_fraction = (
        1.0 if not visible_surface_ids else float(len(accounted_visible)) / float(len(visible_surface_ids))
    )
    if require_full_visible_surface_accounting and missing_visible:
        raise QualificationError(
            f"COMPONENT_V2_VISIBLE_SURFACE_ACCOUNTING_INCOMPLETE:{len(missing_visible)}"
        )

    report = dict(base.qualification_report)
    report.update({
        "status": "PASS_MECHANICAL_EVIDENCE_VERIFIED",
        "mechanical_evidence_verification_version": "RealSaS.ComponentMechanicalQualification.v2",
        "rigid_skinned_component_verified_count": int(rigid_verified),
        "visible_surface_count": len(visible_surface_ids),
        "accounted_visible_surface_count": len(accounted_visible),
        "visible_surface_accounting_fraction": float(accounting_fraction),
        "unaccounted_visible_surface_count": len(missing_visible),
        "full_visible_surface_accounting_required": bool(require_full_visible_surface_accounting),
        "silent_visible_surface_disappearance_forbidden": True,
    })
    provisional = replace(
        base,
        components=tuple(qualified_rows),
        qualification_report=report,
        component_set_lineage_hash="",
    )
    return replace(
        provisional,
        component_set_lineage_hash=qualified_component_set_lineage_hash(provisional),
    )
