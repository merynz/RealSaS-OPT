from __future__ import annotations

"""Carrier-native canonical mechanical state for exact (M,G,W_M)."""

from dataclasses import replace

from .canonical_puppet_state_v1 import CanonicalPuppetStateIR, canonical_puppet_state_hash
from .hashing import content_sha256
from .product_authority_v1 import (
    validate_component_carrier_policy, validate_deformation_capability_envelope,
    validate_mechanical_partition, validate_mesh_qualification_policy, validate_qualified_mesh,
)
from .product_mesh_skin_carrier_v1 import validate_carrier_native_mesh_skin_v1
from .types import QualificationError


def _ledger(*,surface,skeleton,carrier_skin,partition,carrier_policy,envelope,policy,mesh,mesh_skin,carrier):
    return (
        {"stage":"RIGGING_SURFACE_EVIDENCE","hash":surface.geometry_lineage_hash},
        {"stage":"STATIC_MECHANICAL_CARRIER","hash":carrier.carrier_evidence_hash,"topology_hash":carrier.topology_hash,"geometry_hash":carrier.geometry_hash},
        {"stage":"SKELETON_QUALIFICATION","hash":skeleton.skeleton_lineage_hash,"report_hash":content_sha256(skeleton.qualification_report)},
        {"stage":"CARRIER_NATIVE_SKIN_QUALIFICATION","hash":carrier_skin.skin_lineage_hash,"report_hash":content_sha256(carrier_skin.qualification_report)},
        {"stage":"MECHANICAL_PARTITION","hash":partition.partition_lineage_hash},
        {"stage":"COMPONENT_CARRIER_POLICY","hash":carrier_policy.carrier_policy_lineage_hash},
        {"stage":"DEFORMATION_CAPABILITY_ENVELOPE","hash":envelope.envelope_lineage_hash,"probe_plan_hash":envelope.probe_plan_hash,"axis_contract_hash":envelope.axis_contract_hash},
        {"stage":"MESH_QUALIFICATION_POLICY","hash":policy.qualification_policy_lineage_hash},
        {"stage":"QUALIFIED_MESH","hash":mesh.mesh_lineage_hash,"report_hash":content_sha256(mesh.qualification_report)},
        {"stage":"QUALIFIED_MESH_SKIN_IDENTITY_REKEY","hash":mesh_skin.mesh_skin_lineage_hash,"report_hash":content_sha256(mesh_skin.qualification_report)},
    )


def build_canonical_puppet_state_carrier_v2(*,surface,skeleton,carrier_skin,carrier,partition,carrier_policy,envelope,policy,mesh,mesh_skin,metadata=None):
    if carrier_skin.carrier_evidence_hash!=carrier.carrier_evidence_hash or carrier_skin.skeleton_binding_hash!=skeleton.skeleton_lineage_hash:
        raise QualificationError("PUPPET_STATE_V2_CARRIER_SKIN_BINDING_DRIFT")
    validate_mechanical_partition(partition,surface); validate_component_carrier_policy(carrier_policy,partition)
    validate_deformation_capability_envelope(envelope,known_joint_ids={j.canonical_joint_id for j in skeleton.joints})
    validate_mesh_qualification_policy(policy)
    validate_qualified_mesh(mesh,surface=surface,partition=partition,carrier_policy=carrier_policy,envelope=envelope,policy=policy)
    validate_carrier_native_mesh_skin_v1(mesh_skin,carrier=carrier,skeleton=skeleton,carrier_skin=carrier_skin,mesh=mesh)
    ledger=_ledger(surface=surface,skeleton=skeleton,carrier_skin=carrier_skin,partition=partition,carrier_policy=carrier_policy,envelope=envelope,policy=policy,mesh=mesh,mesh_skin=mesh_skin,carrier=carrier)
    value=CanonicalPuppetStateIR(
        surface_lineage_hash=surface.geometry_lineage_hash,
        skeleton_lineage_hash=skeleton.skeleton_lineage_hash,
        skin_lineage_hash=carrier_skin.skin_lineage_hash,
        partition_lineage_hash=partition.partition_lineage_hash,
        carrier_policy_lineage_hash=carrier_policy.carrier_policy_lineage_hash,
        deformation_envelope_lineage_hash=envelope.envelope_lineage_hash,
        mesh_policy_lineage_hash=policy.qualification_policy_lineage_hash,
        mesh_lineage_hash=mesh.mesh_lineage_hash,
        mesh_skin_lineage_hash=mesh_skin.mesh_skin_lineage_hash,
        qualification_ledger=ledger,
        product_state_hash="",
        metadata={
            "mechanical_state_semantics":"EXACT_SAME_FROZEN_M_G_WM",
            "carrier_evidence_hash":carrier.carrier_evidence_hash,
            "carrier_topology_hash":carrier.topology_hash,
            "carrier_geometry_hash":carrier.geometry_hash,
            "semantic_skin_transfer_performed":False,
            "presentation_bound":False,"motion_bound":False,"runtime_projection_bound":False,
            "single_product_mesh_authority":True,
            **dict(metadata or {}),
        },
    )
    return replace(value,product_state_hash=canonical_puppet_state_hash(value))

__all__=["build_canonical_puppet_state_carrier_v2"]
