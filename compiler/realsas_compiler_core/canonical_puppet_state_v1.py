from __future__ import annotations

"""Canonical mechanical product-state seal for the current QualifiedMesh architecture.

This is not the presentation graph and not a motion state. It exists to make the
compatible set S/G/W/partition/carrier/envelope/policy/M/B one exact immutable identity
before presentation, appearance and motion are allowed to consume it.
"""

from dataclasses import asdict, dataclass, field, replace
from typing import Any

from .hashing import content_sha256
from .product_authority_v1 import (
    ComponentCarrierPolicyIR,
    DeformationCapabilityEnvelopeIR,
    MechanicalPartitionIR,
    MeshQualificationPolicyIR,
    QualifiedMeshIR,
    validate_component_carrier_policy,
    validate_deformation_capability_envelope,
    validate_mechanical_partition,
    validate_mesh_qualification_policy,
    validate_qualified_mesh,
)
from .product_mesh_skin_v1 import validate_product_mesh_skin
from .types import (
    QualifiedMeshSkinIR,
    QualifiedSkeletonIR,
    QualifiedSkinIR,
    QualificationError,
    RiggingSurfaceIR,
)

Json=dict[str,Any]


@dataclass(frozen=True)
class CanonicalPuppetStateIR:
    surface_lineage_hash:str
    skeleton_lineage_hash:str
    skin_lineage_hash:str
    partition_lineage_hash:str
    carrier_policy_lineage_hash:str
    deformation_envelope_lineage_hash:str
    mesh_policy_lineage_hash:str
    mesh_lineage_hash:str
    mesh_skin_lineage_hash:str
    qualification_ledger:tuple[Json,...]
    product_state_hash:str
    schema_version:str="RealSaS.CanonicalPuppetStateIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)


def canonical_puppet_state_hash(value:CanonicalPuppetStateIR)->str:
    payload=value.to_dict()
    payload.pop("product_state_hash",None)
    return content_sha256(payload)


def _expected_ledger(
    *,
    surface,
    skeleton,
    skin,
    partition,
    carrier_policy,
    envelope,
    policy,
    mesh,
    mesh_skin,
)->tuple[Json,...]:
    return (
        {"stage":"RIGGING_SURFACE","hash":surface.geometry_lineage_hash},
        {
            "stage":"SKELETON_QUALIFICATION",
            "hash":skeleton.skeleton_lineage_hash,
            "report_hash":content_sha256(skeleton.qualification_report),
        },
        {
            "stage":"SKIN_QUALIFICATION",
            "hash":skin.skin_lineage_hash,
            "report_hash":content_sha256(skin.qualification_report),
        },
        {
            "stage":"MECHANICAL_PARTITION",
            "hash":partition.partition_lineage_hash,
        },
        {
            "stage":"COMPONENT_CARRIER_POLICY",
            "hash":carrier_policy.carrier_policy_lineage_hash,
        },
        {
            "stage":"DEFORMATION_CAPABILITY_ENVELOPE",
            "hash":envelope.envelope_lineage_hash,
            "probe_plan_hash":envelope.probe_plan_hash,
            "axis_contract_hash":envelope.axis_contract_hash,
        },
        {
            "stage":"MESH_QUALIFICATION_POLICY",
            "hash":policy.qualification_policy_lineage_hash,
        },
        {
            "stage":"QUALIFIED_MESH",
            "hash":mesh.mesh_lineage_hash,
            "report_hash":content_sha256(mesh.qualification_report),
        },
        {
            "stage":"QUALIFIED_MESH_SKIN",
            "hash":mesh_skin.mesh_skin_lineage_hash,
            "report_hash":content_sha256(mesh_skin.qualification_report),
        },
    )


def validate_canonical_puppet_state(
    value:CanonicalPuppetStateIR,
    *,
    surface:RiggingSurfaceIR,
    skeleton:QualifiedSkeletonIR,
    skin:QualifiedSkinIR,
    partition:MechanicalPartitionIR,
    carrier_policy:ComponentCarrierPolicyIR,
    envelope:DeformationCapabilityEnvelopeIR,
    policy:MeshQualificationPolicyIR,
    mesh:QualifiedMeshIR,
    mesh_skin:QualifiedMeshSkinIR,
)->None:
    if skin.surface_binding_hash!=surface.geometry_lineage_hash:
        raise QualificationError("PUPPET_STATE_SKIN_SURFACE_LINEAGE_MISMATCH")
    if skin.skeleton_binding_hash!=skeleton.skeleton_lineage_hash:
        raise QualificationError("PUPPET_STATE_SKIN_SKELETON_LINEAGE_MISMATCH")

    validate_mechanical_partition(partition,surface)
    validate_component_carrier_policy(carrier_policy,partition)
    validate_deformation_capability_envelope(
        envelope,
        known_joint_ids={joint.canonical_joint_id for joint in skeleton.joints},
    )
    if envelope.skeleton_lineage_hash!=skeleton.skeleton_lineage_hash:
        raise QualificationError("PUPPET_STATE_ENVELOPE_SKELETON_LINEAGE_MISMATCH")
    validate_mesh_qualification_policy(policy)
    validate_qualified_mesh(
        mesh,
        surface=surface,
        partition=partition,
        carrier_policy=carrier_policy,
        envelope=envelope,
        policy=policy,
    )
    validate_product_mesh_skin(
        mesh_skin,
        surface=surface,
        skeleton=skeleton,
        skin=skin,
        mesh=mesh,
        partition=partition,
        carrier_policy=carrier_policy,
        envelope=envelope,
        policy=policy,
    )

    expected={
        "surface_lineage_hash":surface.geometry_lineage_hash,
        "skeleton_lineage_hash":skeleton.skeleton_lineage_hash,
        "skin_lineage_hash":skin.skin_lineage_hash,
        "partition_lineage_hash":partition.partition_lineage_hash,
        "carrier_policy_lineage_hash":carrier_policy.carrier_policy_lineage_hash,
        "deformation_envelope_lineage_hash":envelope.envelope_lineage_hash,
        "mesh_policy_lineage_hash":policy.qualification_policy_lineage_hash,
        "mesh_lineage_hash":mesh.mesh_lineage_hash,
        "mesh_skin_lineage_hash":mesh_skin.mesh_skin_lineage_hash,
    }
    for field_name,expected_hash in expected.items():
        if getattr(value,field_name)!=expected_hash:
            raise QualificationError(f"PUPPET_STATE_BINDING_MISMATCH:{field_name}")

    expected_ledger=_expected_ledger(
        surface=surface,
        skeleton=skeleton,
        skin=skin,
        partition=partition,
        carrier_policy=carrier_policy,
        envelope=envelope,
        policy=policy,
        mesh=mesh,
        mesh_skin=mesh_skin,
    )
    if tuple(value.qualification_ledger)!=expected_ledger:
        raise QualificationError("PUPPET_STATE_QUALIFICATION_LEDGER_DRIFT")
    if value.product_state_hash!=canonical_puppet_state_hash(value):
        raise QualificationError("PUPPET_STATE_HASH_MISMATCH")


def build_canonical_puppet_state(
    *,
    surface:RiggingSurfaceIR,
    skeleton:QualifiedSkeletonIR,
    skin:QualifiedSkinIR,
    partition:MechanicalPartitionIR,
    carrier_policy:ComponentCarrierPolicyIR,
    envelope:DeformationCapabilityEnvelopeIR,
    policy:MeshQualificationPolicyIR,
    mesh:QualifiedMeshIR,
    mesh_skin:QualifiedMeshSkinIR,
    metadata:Json|None=None,
)->CanonicalPuppetStateIR:
    ledger=_expected_ledger(
        surface=surface,
        skeleton=skeleton,
        skin=skin,
        partition=partition,
        carrier_policy=carrier_policy,
        envelope=envelope,
        policy=policy,
        mesh=mesh,
        mesh_skin=mesh_skin,
    )
    value=CanonicalPuppetStateIR(
        surface.geometry_lineage_hash,
        skeleton.skeleton_lineage_hash,
        skin.skin_lineage_hash,
        partition.partition_lineage_hash,
        carrier_policy.carrier_policy_lineage_hash,
        envelope.envelope_lineage_hash,
        policy.qualification_policy_lineage_hash,
        mesh.mesh_lineage_hash,
        mesh_skin.mesh_skin_lineage_hash,
        ledger,
        "",
        metadata={
            "presentation_bound":False,
            "motion_bound":False,
            "runtime_projection_bound":False,
            "single_product_mesh_authority":True,
            **dict(metadata or {}),
        },
    )
    value=replace(value,product_state_hash=canonical_puppet_state_hash(value))
    validate_canonical_puppet_state(
        value,
        surface=surface,
        skeleton=skeleton,
        skin=skin,
        partition=partition,
        carrier_policy=carrier_policy,
        envelope=envelope,
        policy=policy,
        mesh=mesh,
        mesh_skin=mesh_skin,
    )
    return value
