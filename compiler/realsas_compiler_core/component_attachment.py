from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any

from .hashing import content_sha256
from .types import QualificationError


MECHANICAL_CLASSES = {
    "DEFORMABLE_COMPONENT",
    "RIGID_SKINNED_COMPONENT",
    "RIGID_BONE_ATTACHMENT",
    "NON_MECHANICAL_VISUAL_COMPONENT",
    "EXCLUDED_SOURCE_COMPONENT",
}

DETACHABILITY_CLASSES = {
    "FIXED_COMPONENT",
    "DETACHABLE_COMPONENT",
    "SWAPPABLE_SLOT_COMPONENT",
    "DETACHABILITY_UNKNOWN",
}


@dataclass(frozen=True)
class ComponentEvidenceIR:
    component_id: str
    source_component_refs: tuple[str, ...]
    observation_views: tuple[int, ...]
    surface_ids: tuple[str, ...]
    proposed_mechanical_class: str
    proposed_parent_joint_id: str = ""
    proposed_socket_id: str = ""
    detachability_class: str = "DETACHABILITY_UNKNOWN"
    bind_state_authority_hash: str = ""
    source_geometry_hash: str = ""
    skin_or_deformer_lineage_hash: str = ""
    required_visible_component: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "RealSaS.ComponentEvidenceIR.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QualifiedComponentIR:
    component_id: str
    source_component_refs: tuple[str, ...]
    observation_views: tuple[int, ...]
    surface_ids: tuple[str, ...]
    mechanical_class: str
    parent_joint_id: str
    socket_id: str
    detachability_class: str
    bind_state_authority_hash: str
    source_geometry_hash: str
    skin_or_deformer_lineage_hash: str
    qualification_report: dict[str, Any]
    component_lineage_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "RealSaS.QualifiedComponentIR.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QualifiedComponentSetIR:
    components: tuple[QualifiedComponentIR, ...]
    surface_lineage_hash: str
    skeleton_lineage_hash: str
    skin_lineage_hash: str
    qualification_report: dict[str, Any]
    component_set_lineage_hash: str
    schema_version: str = "RealSaS.QualifiedComponentSetIR.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _component_payload(c: QualifiedComponentIR) -> dict[str, Any]:
    d = c.to_dict()
    d.pop("component_lineage_hash", None)
    return d


def qualified_component_lineage_hash(c: QualifiedComponentIR) -> str:
    return content_sha256(_component_payload(c))


def qualified_component_set_lineage_hash(s: QualifiedComponentSetIR) -> str:
    payload = s.to_dict()
    payload.pop("component_set_lineage_hash", None)
    return content_sha256(payload)


def qualify_component_evidence_v1(
    evidence: ComponentEvidenceIR,
    *,
    known_surface_ids: set[str],
    known_joint_ids: set[str],
) -> QualifiedComponentIR:
    cid = str(evidence.component_id).strip()
    if not cid:
        raise QualificationError("component_id must be non-empty")
    if not evidence.source_component_refs:
        raise QualificationError(f"component {cid} has no source provenance")

    views = tuple(sorted(set(int(v) for v in evidence.observation_views)))
    if any(v < 0 or v > 7 for v in views):
        raise QualificationError(f"component {cid} has invalid observation view")
    if evidence.required_visible_component and not views:
        raise QualificationError(f"required visible component {cid} has no admitted observation view")

    klass = str(evidence.proposed_mechanical_class).strip().upper()
    if klass not in MECHANICAL_CLASSES:
        raise QualificationError(f"component {cid} has unqualified mechanical class {klass!r}")

    detach = str(evidence.detachability_class).strip().upper()
    if detach not in DETACHABILITY_CLASSES:
        raise QualificationError(f"component {cid} has invalid detachability class {detach!r}")

    sids = tuple(sorted(set(map(str, evidence.surface_ids))))
    missing_surface = tuple(sorted(set(sids) - set(known_surface_ids)))
    if missing_surface:
        raise QualificationError(f"component {cid} references unknown surface ids: {missing_surface[:4]}")

    parent = str(evidence.proposed_parent_joint_id or "")
    socket = str(evidence.proposed_socket_id or "")
    bind_hash = str(evidence.bind_state_authority_hash or "")
    skin_hash = str(evidence.skin_or_deformer_lineage_hash or "")

    if klass == "DEFORMABLE_COMPONENT":
        if not sids:
            raise QualificationError(f"deformable component {cid} has no mechanical surface membership")
        if not skin_hash:
            raise QualificationError(f"deformable component {cid} has no skin/deformer authority")
    elif klass in {"RIGID_SKINNED_COMPONENT", "RIGID_BONE_ATTACHMENT"}:
        if not parent and not socket:
            raise QualificationError(f"rigid component {cid} has no canonical joint/socket binding")
        if parent and parent not in known_joint_ids:
            raise QualificationError(f"rigid component {cid} binds unknown joint {parent!r}")
        if not bind_hash:
            raise QualificationError(f"rigid component {cid} has no bind-state authority hash")
        if klass == "RIGID_SKINNED_COMPONENT" and not skin_hash:
            raise QualificationError(f"rigid skinned component {cid} has no skin authority")
    elif klass == "NON_MECHANICAL_VISUAL_COMPONENT":
        if sids:
            raise QualificationError(f"visual-only component {cid} may not masquerade as mechanical surface")
        if parent or socket or skin_hash:
            raise QualificationError(f"visual-only component {cid} has mechanical binding authority")
    elif klass == "EXCLUDED_SOURCE_COMPONENT":
        if evidence.required_visible_component:
            reason = str((evidence.metadata or {}).get("exclusion_reason", "")).strip()
            if not reason:
                raise QualificationError(f"required component {cid} cannot be excluded without product-valid reason")

    report = {
        "status": "PASS",
        "required_visible_component": bool(evidence.required_visible_component),
        "observation_view_count": len(views),
        "surface_count": len(sids),
        "mechanical_class": klass,
        "detachability_class": detach,
        "silent_disappearance_forbidden": True,
        "rigidity_does_not_imply_detachability": True,
    }
    provisional = QualifiedComponentIR(
        component_id=cid,
        source_component_refs=tuple(map(str, evidence.source_component_refs)),
        observation_views=views,
        surface_ids=sids,
        mechanical_class=klass,
        parent_joint_id=parent,
        socket_id=socket,
        detachability_class=detach,
        bind_state_authority_hash=bind_hash,
        source_geometry_hash=str(evidence.source_geometry_hash or ""),
        skin_or_deformer_lineage_hash=skin_hash,
        qualification_report=report,
        component_lineage_hash="",
        metadata=dict(evidence.metadata or {}),
    )
    return QualifiedComponentIR(**{**provisional.to_dict(), "component_lineage_hash": qualified_component_lineage_hash(provisional)})


def qualify_component_set_v1(
    evidence_rows: tuple[ComponentEvidenceIR, ...],
    *,
    known_surface_ids: set[str],
    known_joint_ids: set[str],
    surface_lineage_hash: str,
    skeleton_lineage_hash: str,
    skin_lineage_hash: str,
) -> QualifiedComponentSetIR:
    if not evidence_rows:
        raise QualificationError("component set cannot be empty")
    ids = [str(x.component_id) for x in evidence_rows]
    if len(ids) != len(set(ids)):
        raise QualificationError("component ids must be unique")

    components = tuple(
        qualify_component_evidence_v1(
            row,
            known_surface_ids=set(known_surface_ids),
            known_joint_ids=set(known_joint_ids),
        )
        for row in evidence_rows
    )

    required = tuple(c for c in components if bool(c.qualification_report.get("required_visible_component")))
    if not required:
        raise QualificationError("component set must account for at least one required visible component")

    unknown = tuple(c.component_id for c in components if c.mechanical_class not in MECHANICAL_CLASSES)
    if unknown:
        raise QualificationError(f"component set contains unknown final classes: {unknown}")

    report = {
        "status": "PASS",
        "component_count": len(components),
        "required_visible_component_count": len(required),
        "required_visible_component_accounting_fraction": 1.0,
        "unknown_or_ambiguous_required_count": 0,
        "silent_disappearance_forbidden": True,
    }
    provisional = QualifiedComponentSetIR(
        components=components,
        surface_lineage_hash=str(surface_lineage_hash),
        skeleton_lineage_hash=str(skeleton_lineage_hash),
        skin_lineage_hash=str(skin_lineage_hash),
        qualification_report=report,
        component_set_lineage_hash="",
    )
    return QualifiedComponentSetIR(**{
        **provisional.to_dict(),
        "component_set_lineage_hash": qualified_component_set_lineage_hash(provisional),
    })
