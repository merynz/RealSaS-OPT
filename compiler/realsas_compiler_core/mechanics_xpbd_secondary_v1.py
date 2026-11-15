from __future__ import annotations

"""Opt-in, canonical-domain-bound XPBD secondary joint dynamics.

This is an independently implemented XPBD numeric lane. The old v0.5 archive
is a historical reference, NOT an implicitly imported/product-qualified
backend. XPBD here means compliance-scaled constraints and accumulated
Lagrange multipliers *within* each fixed simulation substep.

Primary AXIS/Mechanics FK, IK and grip are authoritative and solved first.
Only joints admitted into a DeclaredXPBDSecondaryDomainV1 may move. Every
dynamics coefficient, collider, pin and activation is explicit RUNTIME policy;
neither PartVerse semantics nor generated 3D faces claim physical truth.

The result is a canonical MechanicsPoseIRV1. Its downstream deformation must
reuse canonical MIRA skin and the same Graphics mesh; no duplicate puppet.
"""

from dataclasses import asdict, dataclass, field, replace
import math
from typing import Sequence

import numpy as np

from .canonical_rig_ir_v2 import CanonicalRigIRV2
from .hashing import content_sha256
from .mechanics_control_solver_v1 import (
    MechanicsPoseIRV1, _minimal_rotation,
)
from .mechanics_xpbd_domain_v1 import (
    XPBDSecondaryDomainV1, XPBD_SECONDARY_DOMAIN_VERSION,
)
from .types import QualificationError


Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class XPBDPlaneColliderV1:
    collider_id: str
    point: Vec3
    normal: Vec3
    margin: float = 0.0


@dataclass(frozen=True)
class XPBDSphereColliderV1:
    collider_id: str
    center: Vec3
    radius: float
    margin: float = 0.0


@dataclass(frozen=True)
class XPBDSecondaryChainPolicyV1:
    canonical_chain_id: str
    inverse_mass: float = 1.0
    stretch_compliance: float = 0.0
    bend_compliance: float = 0.001
    follow_compliance: float = 0.003
    capsule_radius: float = 0.0


@dataclass(frozen=True)
class XPBDSecondaryPolicyV1:
    domain_hash: str
    chains: tuple[XPBDSecondaryChainPolicyV1, ...]
    planes: tuple[XPBDPlaneColliderV1, ...] = ()
    spheres: tuple[XPBDSphereColliderV1, ...] = ()
    gravity_xyz: Vec3 = (0.0, -9.81, 0.0)
    velocity_damping_per_second: float = 1.5
    fixed_step_seconds: float = 1.0 / 120.0
    max_dt_seconds: float = 0.25
    maximum_substeps: int = 32
    iterations: int = 32
    maximum_segment_relative_error: float = 0.02
    maximum_contact_penetration: float = 1e-4
    schema_version: str = "RealSaS.XPBD.SecondaryPolicy.v1"


@dataclass(frozen=True)
class XPBDSecondaryJointStateV1:
    canonical_joint_id: str
    world_P: Vec3
    world_velocity: Vec3


@dataclass(frozen=True)
class XPBDSecondaryStateV1:
    domain_hash: str
    runtime_policy_hash: str
    rig_lineage_hash: str
    control_program_hash: str
    joints: tuple[XPBDSecondaryJointStateV1, ...]
    pinned_roots: tuple[tuple[str, Vec3], ...]
    simulation_tick_count: int
    frame_count: int
    schema_version: str = "RealSaS.XPBD.SecondaryState.v1"
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class XPBDSecondaryResultV1:
    pose: MechanicsPoseIRV1
    state: XPBDSecondaryStateV1
    status: str
    max_segment_relative_error: float
    max_contact_penetration: float
    applied_contact_projections: int
    simulation_substeps: int
    metadata: dict = field(default_factory=dict)


def _vec(value: object, error: str) -> np.ndarray:
    try:
        a = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise QualificationError(error) from exc
    if a.shape != (3,) or not np.isfinite(a).all():
        raise QualificationError(error)
    return a


def _number(value: object, *, code: str, low: float = 0.0,
            high: float = 1e5, inclusive_low: bool = True) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise QualificationError(code) from exc
    if (not math.isfinite(v) or v > high or
            (v < low if inclusive_low else v <= low)):
        raise QualificationError(code)
    return v


def _policy(domain: XPBDSecondaryDomainV1,
            policy: XPBDSecondaryPolicyV1):
    if (
        domain.schema_version != XPBD_SECONDARY_DOMAIN_VERSION
        or not domain.regions
        or domain.metadata.get("domain_authority") != "EXPLICIT_RUNTIME_POLICY"
        or domain.domain_hash != content_sha256({
            k: v for k, v in domain.to_dict().items() if k != "domain_hash"
        })
    ):
        raise QualificationError("XPBD_DOMAIN_UNQUALIFIED")
    if (
        policy.schema_version != "RealSaS.XPBD.SecondaryPolicy.v1"
        or policy.domain_hash != domain.domain_hash
        or not policy.chains
        or len(policy.chains) != len(domain.regions)
        or len(policy.chains) > 32
        or tuple(c.canonical_chain_id for c in policy.chains)
           != tuple(r.canonical_chain_id for r in domain.regions)
    ):
        raise QualificationError("XPBD_POLICY_DOMAIN_DRIFT")
    _vec(policy.gravity_xyz, "XPBD_GRAVITY_NONFINITE")
    _number(policy.velocity_damping_per_second,
            code="XPBD_DAMPING_INVALID", high=100.0)
    h = _number(policy.fixed_step_seconds,
                code="XPBD_FIXED_STEP_INVALID", low=0.0,
                high=0.1, inclusive_low=False)
    _number(policy.max_dt_seconds,
            code="XPBD_MAX_DT_INVALID", low=h, high=0.25)
    if (type(policy.maximum_substeps) is not int
            or not 1 <= policy.maximum_substeps <= 128
            or type(policy.iterations) is not int
            or not 1 <= policy.iterations <= 128):
        raise QualificationError("XPBD_ITERATION_RESOURCE_BOUND")
    _number(policy.maximum_segment_relative_error,
            code="XPBD_LENGTH_TOLERANCE_INVALID", low=0.0, high=0.5)
    _number(policy.maximum_contact_penetration,
            code="XPBD_CONTACT_TOLERANCE_INVALID", low=0.0, high=0.1)
    configs = {}
    for cfg in policy.chains:
        _number(cfg.inverse_mass, code="XPBD_MASS_INVALID",
                low=0., high=1e4, inclusive_low=False)
        for v in (
            cfg.stretch_compliance, cfg.bend_compliance,
            cfg.follow_compliance,
        ):
            _number(v, code="XPBD_COMPLIANCE_INVALID", high=1e3)
        _number(cfg.capsule_radius,
                code="XPBD_CAPSULE_RADIUS_INVALID", high=1e3)
        configs[cfg.canonical_chain_id] = cfg
    colliders = {}
    for plane in policy.planes:
        if (not plane.collider_id or plane.collider_id in colliders):
            raise QualificationError("XPBD_COLLIDER_ID_DUPLICATE")
        _vec(plane.point, "XPBD_PLANE_POINT_INVALID")
        n = _vec(plane.normal, "XPBD_PLANE_NORMAL_INVALID")
        if abs(float(np.linalg.norm(n)) - 1.0) > 1e-6:
            raise QualificationError("XPBD_PLANE_NORMAL_NOT_UNIT")
        _number(plane.margin, code="XPBD_COLLIDER_MARGIN_INVALID",
                high=1e3)
        colliders[plane.collider_id] = plane
    for sphere in policy.spheres:
        if (not sphere.collider_id or sphere.collider_id in colliders):
            raise QualificationError("XPBD_COLLIDER_ID_DUPLICATE")
        _vec(sphere.center, "XPBD_SPHERE_CENTER_INVALID")
        _number(sphere.radius, code="XPBD_SPHERE_RADIUS_INVALID",
                low=0., high=1e3, inclusive_low=False)
        _number(sphere.margin, code="XPBD_COLLIDER_MARGIN_INVALID",
                high=1e3)
        colliders[sphere.collider_id] = sphere
    total_joints = sum(len(r.dynamic_joint_ids) for r in domain.regions)
    if total_joints > 256 or (len(policy.planes) + len(policy.spheres)) > 32:
        raise QualificationError("XPBD_PARTICLE_OR_COLLIDER_RESOURCE_BOUND")
    # O(N^2) capsules are deliberately limited; no unbounded frame cost.
    if sum(len(r.dynamic_joint_ids) for r, c in zip(domain.regions, policy.chains)
           if c.capsule_radius > 0.0) > 64:
        raise QualificationError("XPBD_CAPSULE_RESOURCE_BOUND")
    return configs, content_sha256(asdict(policy))


def _closest(a: np.ndarray, b: np.ndarray,
             c: np.ndarray, d: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Closest points on finite segments (Ericson clamp formulation)."""
    u, v, w = b - a, d - c, a - c
    A, B, C = float(u@u), float(u@v), float(v@v)
    D, E = float(u@w), float(v@w)
    if A < 1e-18 or C < 1e-18:
        # A degenerate capsule is still a valid point collision case.
        if A < 1e-18 and C < 1e-18:
            return 0.0, 0.0, a-c
        if A < 1e-18:
            t = float(np.clip(E / C, 0., 1.))
            return 0., t, a - (c+t*v)
        s = float(np.clip(-D / A, 0., 1.))
        return s, 0., (a+s*u)-c
    den = A*C-B*B
    s = 0. if den < 1e-18 else float(np.clip((B*E-C*D)/den, 0., 1.))
    t = float(np.clip((B*s+E)/C, 0., 1.))
    s = float(np.clip((B*t-D)/A, 0., 1.))
    t = float(np.clip((B*s+E)/C, 0., 1.))
    return s, t, (a+s*u)-(c+t*v)


def step_declared_xpbd_secondary_v1(
    rig: CanonicalRigIRV2,
    primary_pose: MechanicsPoseIRV1,
    domain: XPBDSecondaryDomainV1,
    policy: XPBDSecondaryPolicyV1,
    *,
    previous_state: XPBDSecondaryStateV1 | None = None,
    dt_seconds: float | None = None,
) -> XPBDSecondaryResultV1:
    """Run hard-pinned primary→XPBD secondary chain and contact constraints.

    Lagrange multipliers are accumulated during solver iterations and reset at
    each fixed substep; velocity persists across calls. Fixed-tick dt is exact
    (a non-integral number of ticks is rejected, never silently rounded).
    """

    cfgs, signature = _policy(domain, policy)
    if (domain.rig_lineage_hash != rig.rig_lineage_hash
            or domain.control_program_hash != rig.control_program_hash
            or primary_pose.rig_lineage_hash != rig.rig_lineage_hash
            or primary_pose.control_program_hash != rig.control_program_hash):
        raise QualificationError("XPBD_POSE_OR_RIG_LINEAGE_DRIFT")
    primary = {j.canonical_joint_id: j for j in primary_pose.joints}
    if len(primary) != len(primary_pose.joints) or set(primary) != {
        j.canonical_joint_id for j in rig.joints
    }:
        raise QualificationError("XPBD_POSE_JOINT_COVERAGE_DRIFT")
    pp = {jid: _vec(j.solved_P, "XPBD_PRIMARY_POSE_NONFINITE")
          for jid, j in primary.items()}
    for j in primary_pose.joints:
        R = np.asarray(j.rotation_basis, dtype=np.float64)
        if (R.shape != (3, 3) or not np.isfinite(R).all()
                or np.max(np.abs(R.T@R-np.eye(3))) > 1e-4
                or np.linalg.det(R) < 0.999):
            raise QualificationError("XPBD_PRIMARY_ROTATION_INVALID")

    dyn = tuple(sorted(jid for r in domain.regions for jid in r.dynamic_joint_ids))
    if len(set(dyn)) != len(dyn) or any(jid not in pp for jid in dyn):
        raise QualificationError("XPBD_DYNAMIC_ADDRESS_DRIFT")
    roots = tuple(sorted(set(r.pinned_root_joint_id for r in domain.regions)))
    if any(root not in pp or root in dyn for root in roots):
        raise QualificationError("XPBD_PIN_ROOT_INVALID")
    rest = {r.canonical_chain_id: np.asarray(r.segment_rest_lengths, dtype=np.float64)
            for r in domain.regions}
    for region in domain.regions:
        chain = (region.pinned_root_joint_id,) + region.dynamic_joint_ids
        if any(abs(np.linalg.norm(pp[b]-pp[a])-float(L)) > max(0.1*L, 1e-6)
               for (a,b), L in zip(zip(chain, chain[1:]),
                                    rest[region.canonical_chain_id])):
            raise QualificationError("XPBD_PRIMARY_SEGMENT_STRETCH_CONFLICT")
    pos = {jid: p.copy() for jid, p in pp.items()}
    velocities = {jid: np.zeros(3, dtype=np.float64) for jid in dyn}
    if previous_state is None:
        if dt_seconds is not None:
            raise QualificationError("XPBD_INITIALIZE_WITHOUT_DT")
        ticks, frames, steps = 0, 0, 0
        previous_roots = {jid: pp[jid] for jid in roots}
    else:
        if (
            previous_state.schema_version != "RealSaS.XPBD.SecondaryState.v1"
            or previous_state.domain_hash != domain.domain_hash
            or previous_state.runtime_policy_hash != signature
            or previous_state.rig_lineage_hash != rig.rig_lineage_hash
            or previous_state.control_program_hash != rig.control_program_hash
        ):
            raise QualificationError("XPBD_STATE_DOMAIN_OR_POLICY_DRIFT")
        old = {j.canonical_joint_id: j for j in previous_state.joints}
        if len(old) != len(previous_state.joints) or set(old) != set(dyn):
            raise QualificationError("XPBD_STATE_JOINT_COVERAGE_DRIFT")
        previous_roots = {
            root: _vec(p, "XPBD_STATE_ROOT_INVALID")
            for root, p in previous_state.pinned_roots
        }
        if len(previous_roots) != len(previous_state.pinned_roots) or set(previous_roots) != set(roots):
            raise QualificationError("XPBD_STATE_ROOT_COVERAGE_DRIFT")
        if (
            type(previous_state.simulation_tick_count) is not int
            or previous_state.simulation_tick_count < 0
            or type(previous_state.frame_count) is not int
            or previous_state.frame_count < 0
        ):
            raise QualificationError("XPBD_STATE_COUNTER_INVALID")
        for jid in dyn:
            pos[jid] = _vec(old[jid].world_P, "XPBD_STATE_POSITION_INVALID")
            velocities[jid] = _vec(
                old[jid].world_velocity, "XPBD_STATE_VELOCITY_INVALID"
            )
        dt = _number(dt_seconds, code="XPBD_DT_INVALID",
                     low=0., high=policy.max_dt_seconds, inclusive_low=False)
        nominal = dt / policy.fixed_step_seconds
        steps = int(round(nominal))
        if (steps < 1 or steps > policy.maximum_substeps
                or abs(nominal-steps) > 1e-7):
            raise QualificationError("XPBD_FIXED_TICK_PARTITION_INVALID")
        ticks, frames = previous_state.simulation_tick_count, previous_state.frame_count

    gravity = _vec(policy.gravity_xyz, "XPBD_GRAVITY_NONFINITE")
    h = policy.fixed_step_seconds
    damping = math.exp(-policy.velocity_damping_per_second*h)
    contact_projections = 0
    # XPBD lambda is per (constraint, time substep), not an empirical spring.
    for sub in range(steps):
        alpha = (sub+1)/steps
        for root in roots:
            pos[root] = previous_roots[root] + alpha*(pp[root]-previous_roots[root])
        before = {jid: pos[jid].copy() for jid in dyn}
        for jid in dyn:
            pos[jid] = pos[jid] + velocities[jid]*h*damping + gravity*h*h
        inv_mass = {
            jid: cfgs[reg.canonical_chain_id].inverse_mass
            for reg in domain.regions for jid in reg.dynamic_joint_ids
        }
        for root in roots:
            inv_mass[root] = 0.
        segment_constraints: list[tuple[str,str,float,float]] = []
        bend_constraints: list[tuple[str,str,float,float]] = []
        follow_constraints: list[tuple[str,np.ndarray,float]] = []
        capsules: list[tuple[str,str,float,str,int]] = []
        for reg in domain.regions:
            cfg = cfgs[reg.canonical_chain_id]
            chain = (reg.pinned_root_joint_id,) + reg.dynamic_joint_ids
            segment_constraints.extend(
                (a,b,float(L),cfg.stretch_compliance)
                for (a,b), L in zip(zip(chain,chain[1:]),rest[reg.canonical_chain_id])
            )
            for i in range(len(chain)-2):
                a,b = chain[i],chain[i+2]
                # Bending constraints follow current primary articulated pose,
                # preserving authored primary shape; no source stiffness claim.
                bend_constraints.append(
                    (a,b,float(np.linalg.norm(pp[b]-pp[a])),
                     cfg.bend_compliance)
                )
            follow_constraints.extend(
                (jid, pp[jid], cfg.follow_compliance)
                for jid in reg.dynamic_joint_ids
            )
            if cfg.capsule_radius > 0.0:
                capsules.extend(
                    (a,b,cfg.capsule_radius,reg.canonical_chain_id,i)
                    for i,(a,b) in enumerate(zip(chain,chain[1:]))
                )
        lam_dist = np.zeros(len(segment_constraints),dtype=np.float64)
        lam_bend = np.zeros(len(bend_constraints),dtype=np.float64)
        lam_follow = np.zeros((len(follow_constraints),3),dtype=np.float64)
        lam_plane = np.zeros((len(dyn),len(policy.planes)),dtype=np.float64)
        lam_sphere = np.zeros((len(dyn),len(policy.spheres)),dtype=np.float64)
        pair_ids = tuple((i,j) for i in range(len(capsules))
                         for j in range(i+1,len(capsules))
                         if (capsules[i][3] != capsules[j][3] or
                             abs(capsules[i][4]-capsules[j][4]) >= 2)
                         and not ({capsules[i][0],capsules[i][1]}
                                  & {capsules[j][0],capsules[j][1]}))
        lam_capsule = np.zeros(len(pair_ids),dtype=np.float64)
        for _ in range(policy.iterations):
            for rows, lam in ((segment_constraints,lam_dist),
                              (bend_constraints,lam_bend)):
                for ci,(a,b,L,compliance) in enumerate(rows):
                    vec = pos[a]-pos[b]
                    dist = float(np.linalg.norm(vec))
                    if dist < 1e-12:
                        raise QualificationError("XPBD_COLLAPSED_DISTANCE_CONSTRAINT")
                    n = vec/dist
                    w0,w1=inv_mass[a],inv_mass[b]
                    alpha_tilde = compliance/(h*h)
                    dl = -(dist-L+alpha_tilde*lam[ci])/(w0+w1+alpha_tilde)
                    lam[ci] += dl
                    pos[a] = pos[a]+w0*dl*n
                    pos[b] = pos[b]-w1*dl*n
            for ci,(jid,target,compliance) in enumerate(follow_constraints):
                w = inv_mass[jid]
                at = compliance/(h*h)
                dl = -(pos[jid]-target+at*lam_follow[ci])/(w+at)
                lam_follow[ci] += dl
                pos[jid] = pos[jid]+w*dl
            # Collisions are unilateral: signed gap >= 0; lambda >= 0.
            # Contact projection is last so distance constraints cannot
            # silently replace a penetration correction with a fake PASS.
            for di,jid in enumerate(dyn):
                w=inv_mass[jid]
                for ci,collider in enumerate(policy.planes):
                    n=np.asarray(collider.normal,dtype=np.float64)
                    C=float(np.dot(pos[jid]-collider.point,n)-collider.margin)
                    dl=max(0.0,lam_plane[di,ci]-C/w)-lam_plane[di,ci]
                    lam_plane[di,ci]+=dl
                    pos[jid]=pos[jid]+w*dl*n
                    if dl>1e-12: contact_projections+=1
                for ci,collider in enumerate(policy.spheres):
                    radius=collider.radius+collider.margin
                    offset=pos[jid]-collider.center
                    length=float(np.linalg.norm(offset))
                    n=offset/length if length>1e-12 else np.array((1.,0.,0.))
                    C=length-radius
                    dl=max(0.0,lam_sphere[di,ci]-C/w)-lam_sphere[di,ci]
                    lam_sphere[di,ci]+=dl
                    pos[jid]=pos[jid]+w*dl*n
                    if dl>1e-12: contact_projections+=1
            for ci,(i,j) in enumerate(pair_ids):
                a,b,ra,_,_=capsules[i]
                c,d,rb,_,_=capsules[j]
                s,t,delta=_closest(pos[a],pos[b],pos[c],pos[d])
                length=float(np.linalg.norm(delta))
                C=length-(ra+rb)
                if length>1e-12:
                    n=delta/length
                else:
                    n=np.cross(pos[b]-pos[a],pos[d]-pos[c])
                    z=float(np.linalg.norm(n))
                    n=n/z if z>1e-12 else np.array((1.,0.,0.))
                factors=(1-s,s,-(1-t),-t)
                names=(a,b,c,d)
                den=sum(inv_mass[k]*factor*factor
                        for k,factor in zip(names,factors))
                if den<=1e-18:
                    if C<0.: raise QualificationError("XPBD_CAPSULE_FIXED_PENETRATION")
                    continue
                dl=max(0.,lam_capsule[ci]-C/den)-lam_capsule[ci]
                lam_capsule[ci]+=dl
                for jid,factor in zip(names,factors):
                    pos[jid]=pos[jid]+inv_mass[jid]*factor*dl*n
                if dl>1e-12:contact_projections+=1
        for jid in dyn:
            velocities[jid]=(pos[jid]-before[jid])/h
            if not np.isfinite(velocities[jid]).all():
                raise QualificationError("XPBD_VELOCITY_NONFINITE")
        if any(not np.isfinite(pos[j]).all() for j in dyn):
            raise QualificationError("XPBD_POSITION_NONFINITE")
        ticks+=1

    max_len=0.
    for reg in domain.regions:
        chain = (reg.pinned_root_joint_id,) + reg.dynamic_joint_ids
        for (a,b),L in zip(zip(chain,chain[1:]),rest[reg.canonical_chain_id]):
            max_len=max(max_len,abs(float(np.linalg.norm(pos[b]-pos[a]))-L)/L)
    penetration=0.
    for jid in dyn:
        for plane in policy.planes:
            penetration=max(penetration,max(0.,plane.margin-
                float((pos[jid]-plane.point)@plane.normal)))
        for sphere in policy.spheres:
            penetration=max(penetration,max(0.,
                sphere.radius+sphere.margin-float(np.linalg.norm(pos[jid]-sphere.center))))
    # Self-capsule constraints must be checked after *all* corrections.
    capsules=[(reg.canonical_chain_id,i,a,b,cfgs[reg.canonical_chain_id].capsule_radius)
              for reg in domain.regions
              for i,(a,b) in enumerate(zip(
                  (reg.pinned_root_joint_id,)+reg.dynamic_joint_ids,
                  reg.dynamic_joint_ids
              ))
              if cfgs[reg.canonical_chain_id].capsule_radius>0.]
    for i,ca in enumerate(capsules):
        for cb in capsules[i+1:]:
            if ((ca[0] == cb[0] and abs(ca[1]-cb[1])<2)
                    or {ca[2],ca[3]} & {cb[2],cb[3]}):
                continue
            _,_,delta=_closest(pos[ca[2]],pos[ca[3]],
                               pos[cb[2]],pos[cb[3]])
            penetration=max(penetration,max(0.,
                ca[4]+cb[4]-float(np.linalg.norm(delta))))
    angles={}
    for reg in domain.regions:
        chain=(reg.pinned_root_joint_id,)+reg.dynamic_joint_ids
        for a,b in zip(chain,chain[1:]):
            native=pp[b]-pp[a]
            updated=pos[b]-pos[a]
            if float(np.linalg.norm(updated))<=1e-12:
                raise QualificationError("XPBD_RESULT_SEGMENT_COLLAPSED")
            swing=_minimal_rotation(native,updated)
            angles[b]=swing@np.asarray(primary[b].rotation_basis,dtype=np.float64)
    pose_rows=tuple(
        replace(j,solved_P=tuple(map(float,pos[j.canonical_joint_id])),
                rotation_basis=tuple(tuple(map(float,row)) for row in angles[j.canonical_joint_id]))
        if j.canonical_joint_id in angles else j
        for j in primary_pose.joints
    )
    state=XPBDSecondaryStateV1(
        domain_hash=domain.domain_hash,
        runtime_policy_hash=signature,
        rig_lineage_hash=rig.rig_lineage_hash,
        control_program_hash=rig.control_program_hash,
        joints=tuple(XPBDSecondaryJointStateV1(
            jid,tuple(map(float,pos[jid])),tuple(map(float,velocities[jid]))
        ) for jid in dyn),
        pinned_roots=tuple((jid,tuple(map(float,pp[jid]))) for jid in roots),
        simulation_tick_count=ticks,
        frame_count=frames+int(steps>0),
        metadata={
            "authority":"EXPLICIT_XPBD_SECONDARY_RUNTIME",
            "physical_truth_minted":False,
            "training_authorized":False,
            "constraints_lambdas_reset_per_substep":True,
            "source_contact_qualified":False,
        },
    )
    status=("INITIALIZED_XPBD_UNSIMULATED" if steps==0 else
            "PASS_XPBD_SECONDARY_RUNTIME" if (
                max_len<=policy.maximum_segment_relative_error
                and penetration<=policy.maximum_contact_penetration)
            else "FAIL_XPBD_RESIDUAL_OR_PENETRATION")
    return XPBDSecondaryResultV1(
        pose=replace(primary_pose,joints=pose_rows,metadata={
            **primary_pose.metadata,
            "secondary_motion":"XPBD_CONSTRAINT_SOLVER_V1",
            "xpbd_domain_hash":domain.domain_hash,
            "primary_controls_not_modified":True,
            "physical_parameters_source_qualified":False,
        }),
        state=state,
        status=status,
        max_segment_relative_error=max_len,
        max_contact_penetration=penetration,
        applied_contact_projections=contact_projections,
        simulation_substeps=steps,
        metadata={
            "solver_family":"EXTENDED_POSITION_BASED_DYNAMICS",
            "compliance_scaled_by_inverse_dt_squared":True,
            "lambda_accumulated_per_substep":True,
            "rig_skin_immutable":True,
            "source_physical_truth_claimed":False,
            "self_collision_type":"NONADJACENT_CHAIN_CAPSULE",
            "world_contacts":"DECLARED_PLANE_AND_SPHERE",
            "cloth_surface_particles_simulated":False,
        },
    )
