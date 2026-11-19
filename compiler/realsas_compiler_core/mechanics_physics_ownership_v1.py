from __future__ import annotations

"""Mechanics.Physics: deterministic ownership of canonical dynamic vertices.

The production Mechanics order is PRIMARY FK/IK -> optional SECONDARY JOINT
dynamics -> canonical MIRA LBS -> optional SURFACE XPBD. A surface vertex may
have one *dynamic* owner only. Primary pose is an input to every mode, not a
competing physics owner; pinned surface vertices are kinematic boundaries.

This is a legality court for declared programs. It does NOT infer cloth,
wing material, mass, stiffness, contact or physical activation from semantics.
"""
from dataclasses import dataclass
from typing import Literal

from .canonical_rig_ir_v2 import CanonicalRigIRV2
from .canonical_skin_ir_v2 import CanonicalSkinIRV2
from .mechanics_secondary_motion_v1 import SecondaryMotionPolicyV1
from .mechanics_xpbd_domain_v1 import XPBDSecondaryDomainV1
from .mechanics_xpbd_surface_v1 import XPBDSurfaceDomainV1
from .types import QualificationError

JointMode = Literal["NONE", "DECLARED_OSCILLATOR", "DECLARED_XPBD_JOINT"]
SurfaceMode = Literal["NONE", "DECLARED_XPBD_SURFACE"]


@dataclass(frozen=True)
class MechanicsPhysicsOwnershipAuditV1:
    secondary_joint_mode: JointMode
    surface_mode: SurfaceMode
    secondary_joint_count: int
    secondary_skinned_vertex_count: int
    surface_dynamic_vertex_count: int
    disjoint: bool = True
    source_material_truth_claimed: bool = False
    authority: str = "MECHANICS_DECLARED_RUNTIME_DOMAIN_NOT_LEARNED_MATERIAL"


def qualify_mechanics_physics_domain_ownership_v1(
    rig: CanonicalRigIRV2,
    skin: CanonicalSkinIRV2,
    *,
    secondary_motion_policy: SecondaryMotionPolicyV1 | None = None,
    xpbd_secondary_domain: XPBDSecondaryDomainV1 | None = None,
    xpbd_surface_domain: XPBDSurfaceDomainV1 | None = None,
) -> MechanicsPhysicsOwnershipAuditV1:
    """Admit disjoint Physics ownership; no scene/semantic inference.

    A positive canonical skin influence gives a dynamic joint ownership of
    that vertex. A tiny positive influence is still an actual contribution
    until the *canonical skin* itself has explicitly pruned it. We cannot
    unilaterally choose a threshold to conceal double simulation.
    """
    if not isinstance(rig,CanonicalRigIRV2) or not isinstance(skin,CanonicalSkinIRV2):
        raise QualificationError("MECHANICS_PHYSICS_CANONICAL_PARENT_REQUIRED")
    if secondary_motion_policy is not None and xpbd_secondary_domain is not None:
        raise QualificationError("LIVE_MECHANICS_SECONDARY_SOLVER_DOUBLE_OWNERSHIP")
    mode: JointMode = "NONE"
    joint_ids: set[str] = set()
    owned_vertices: set[str] = set()

    if secondary_motion_policy is not None:
        if (
            not isinstance(secondary_motion_policy,SecondaryMotionPolicyV1)
            or secondary_motion_policy.rig_lineage_hash!=rig.rig_lineage_hash
            or secondary_motion_policy.control_program_hash!=rig.control_program_hash
        ):
            raise QualificationError("SECONDARY_POLICY_AUTHORITY_LINEAGE_DRIFT")
        chains={c.canonical_chain_id:c for c in rig.chains}
        for p in secondary_motion_policy.chains:
            declared=chains.get(p.canonical_chain_id)
            if declared is None or len(declared.canonical_joint_ids)<2:
                raise QualificationError("SECONDARY_CHAIN_NOT_DECLARED")
            actual=set(declared.canonical_joint_ids[1:])
            if actual & joint_ids:
                raise QualificationError("SECONDARY_JOINT_MULTI_CHAIN_OWNERSHIP")
            joint_ids.update(actual)
        mode="DECLARED_OSCILLATOR"
    elif xpbd_secondary_domain is not None:
        if not isinstance(xpbd_secondary_domain,XPBDSecondaryDomainV1):
            raise QualificationError("MECHANICS_PHYSICS_XPBD_JOINT_DOMAIN_INVALID")
        for region in xpbd_secondary_domain.regions:
            if set(region.dynamic_joint_ids) & joint_ids:
                raise QualificationError("MECHANICS_PHYSICS_XPBD_JOINT_MULTIPLE_OWNERS")
            joint_ids.update(region.dynamic_joint_ids)
            owned_vertices.update(region.skinned_vertex_ids)
        mode="DECLARED_XPBD_JOINT"

    surface_mode: SurfaceMode = (
        "DECLARED_XPBD_SURFACE" if xpbd_surface_domain is not None else "NONE"
    )
    if xpbd_surface_domain is not None:
        if not isinstance(xpbd_surface_domain,XPBDSurfaceDomainV1):
            raise QualificationError("MECHANICS_PHYSICS_XPBD_SURFACE_DOMAIN_INVALID")
        dynamic=set(xpbd_surface_domain.dynamic_vertex_ids)
        if len(dynamic)!=len(xpbd_surface_domain.dynamic_vertex_ids):
            raise QualificationError("MECHANICS_PHYSICS_SURFACE_VERTEX_DUPLICATE")
        if secondary_motion_policy is not None:
            # Only the independently declared surface dynamic subset is
            # scanned; surface pins can deliberately follow moving bones.
            for row in skin.rows:
                if row.canonical_mesh_vertex_id not in dynamic:
                    continue
                if any(
                    jid in joint_ids and weight>0.0
                    for jid,weight in row.influences
                ):
                    raise QualificationError("LIVE_SECONDARY_JOINT_SURFACE_DOUBLE_OWNERSHIP")
        elif xpbd_secondary_domain is not None and dynamic & owned_vertices:
            raise QualificationError("LIVE_XPBD_JOINT_SURFACE_DOUBLE_OWNERSHIP")
    else:
        dynamic=set()

    return MechanicsPhysicsOwnershipAuditV1(
        secondary_joint_mode=mode,
        surface_mode=surface_mode,
        secondary_joint_count=len(joint_ids),
        secondary_skinned_vertex_count=len(owned_vertices),
        surface_dynamic_vertex_count=len(dynamic),
    )


__all__=[
    "MechanicsPhysicsOwnershipAuditV1",
    "qualify_mechanics_physics_domain_ownership_v1",
]
