from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from .component_attachment import QualifiedComponentSetIR, qualified_component_set_lineage_hash
from .hashing import content_sha256
from .types import QualificationError
from .v4 import mechanical_state_hash, validate_mechanical_state
from .v4_types import MechanicalStateIR


@dataclass(frozen=True)
class QualifiedMechanicalAssemblyIR:
    """Typed mechanical state plus explicit component/attachment authority.

    This is deliberately an adjunct to MechanicalStateIR.v2 during the reclosure.
    It prevents rigid/deformable/visual component semantics from being inferred from
    skin weights or render visibility after the fact. Promotion into a future product
    schema must consume this qualified assembly rather than silently dropping it.
    """

    mechanical_state: MechanicalStateIR
    component_set: QualifiedComponentSetIR
    assembly_lineage_hash: str
    schema_version: str = "RealSaS.QualifiedMechanicalAssemblyIR.v1"

    def to_dict(self):
        return asdict(self)


def qualified_mechanical_assembly_hash(value: QualifiedMechanicalAssemblyIR) -> str:
    payload = value.to_dict()
    payload.pop("assembly_lineage_hash", None)
    return content_sha256(payload)


def validate_qualified_mechanical_assembly(value: QualifiedMechanicalAssemblyIR) -> None:
    validate_mechanical_state(value.mechanical_state)
    components = value.component_set
    mechanical = value.mechanical_state

    if components.component_set_lineage_hash != qualified_component_set_lineage_hash(components):
        raise QualificationError("MECHANICAL_ASSEMBLY_COMPONENT_SET_HASH_MISMATCH")
    if components.surface_lineage_hash != mechanical.surface.geometry_lineage_hash:
        raise QualificationError("MECHANICAL_ASSEMBLY_SURFACE_LINEAGE_MISMATCH")
    if components.skeleton_lineage_hash != mechanical.skeleton.skeleton_lineage_hash:
        raise QualificationError("MECHANICAL_ASSEMBLY_SKELETON_LINEAGE_MISMATCH")
    if components.skin_lineage_hash != mechanical.skin.skin_lineage_hash:
        raise QualificationError("MECHANICAL_ASSEMBLY_SKIN_LINEAGE_MISMATCH")
    if components.qualification_report.get("status") != "PASS":
        raise QualificationError("MECHANICAL_ASSEMBLY_COMPONENT_QUALIFICATION_NOT_PASS")
    if float(components.qualification_report.get("required_visible_component_accounting_fraction", 0.0)) != 1.0:
        raise QualificationError("MECHANICAL_ASSEMBLY_REQUIRED_VISIBLE_COMPONENT_ACCOUNTING_INCOMPLETE")
    if int(components.qualification_report.get("unknown_or_ambiguous_required_count", 1)) != 0:
        raise QualificationError("MECHANICAL_ASSEMBLY_REQUIRED_COMPONENT_AMBIGUITY")
    if mechanical.mechanical_state_hash != mechanical_state_hash(mechanical):
        raise QualificationError("MECHANICAL_ASSEMBLY_MECHANICAL_STATE_HASH_MISMATCH")
    if value.assembly_lineage_hash != qualified_mechanical_assembly_hash(value):
        raise QualificationError("MECHANICAL_ASSEMBLY_LINEAGE_HASH_MISMATCH")


def build_qualified_mechanical_assembly(
    mechanical_state: MechanicalStateIR,
    component_set: QualifiedComponentSetIR,
) -> QualifiedMechanicalAssemblyIR:
    value = QualifiedMechanicalAssemblyIR(
        mechanical_state=mechanical_state,
        component_set=component_set,
        assembly_lineage_hash="",
    )
    value = replace(value, assembly_lineage_hash=qualified_mechanical_assembly_hash(value))
    validate_qualified_mechanical_assembly(value)
    return value
