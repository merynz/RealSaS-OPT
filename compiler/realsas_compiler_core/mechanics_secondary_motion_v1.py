from __future__ import annotations

"""Opt-in deterministic secondary joint motion after primary Mechanics.

Primary FK/IK and world contact constraints are solved first. ONLY explicitly
declared secondary chains may lag. Secondary motion changes no declared control,
source truth, CanonicalMotion, primary IK goal or contact pin.

The critically damped oscillator frequency and angular cap are RUNTIME POLICY;
they do not assert measured mass, physical joint stiffness or source truth.
No Graphics draw ordering, collision, cloth solver or friction is inferred.
"""

from dataclasses import dataclass, field, replace
import math

import numpy as np

from .canonical_rig_ir_v2 import CanonicalRigIRV2
from .mechanics_control_solver_v1 import (
    MechanicsPoseIRV1, SolvedJointTransformIRV1, _minimal_rotation,
)
from .types import QualificationError

Vec3 = tuple[float, float, float]
Mat3 = tuple[Vec3, Vec3, Vec3]


@dataclass(frozen=True)
class SecondaryChainPolicyV1:
    canonical_chain_id: str
    frequency_hz: float = 3.0
    max_deviation_rad: float = 0.85


@dataclass(frozen=True)
class SecondaryMotionPolicyV1:
    rig_lineage_hash: str
    control_program_hash: str
    chains: tuple[SecondaryChainPolicyV1, ...]
    maximum_dt_seconds: float = 0.20
    fixed_step_seconds: float = 1.0 / 60.0
    maximum_substeps: int = 32
    schema_version: str = "RealSaS.SecondaryMotionPolicy.v1"


@dataclass(frozen=True)
class SecondaryJointStateV1:
    canonical_joint_id: str
    world_P: Vec3
    world_velocity: Vec3


@dataclass(frozen=True)
class SecondaryMotionStateV1:
    rig_lineage_hash: str
    control_program_hash: str
    secondary_joint_ids: tuple[str, ...]
    joints: tuple[SecondaryJointStateV1, ...]
    step_count: int
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SecondaryMotionStepResultV1:
    pose: MechanicsPoseIRV1
    state: SecondaryMotionStateV1
    max_secondary_segment_error: float
    max_secondary_angular_deviation_rad: float
    status: str
    metadata: dict = field(default_factory=dict)


def _v3(value: object, code: str) -> np.ndarray:
    try:
        p = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise QualificationError(code) from exc
    if p.shape != (3,) or not np.isfinite(p).all():
        raise QualificationError(code)
    return p


def _unit(v: np.ndarray, code: str) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise QualificationError(code)
    return v / norm


def _params(rig: CanonicalRigIRV2, policy: SecondaryMotionPolicyV1):
    if (policy.schema_version != "RealSaS.SecondaryMotionPolicy.v1"
            or policy.rig_lineage_hash != rig.rig_lineage_hash
            or policy.control_program_hash != rig.control_program_hash):
        raise QualificationError("SECONDARY_POLICY_AUTHORITY_LINEAGE_DRIFT")
    if (not math.isfinite(float(policy.maximum_dt_seconds))
            or not 0 < policy.maximum_dt_seconds <= 0.25):
        raise QualificationError("SECONDARY_DT_RESOURCE_POLICY_INVALID")
    if (not math.isfinite(float(policy.fixed_step_seconds))
            or policy.fixed_step_seconds <= 0
            or policy.fixed_step_seconds > policy.maximum_dt_seconds
            or not isinstance(policy.maximum_substeps, int)
            or not 1 <= policy.maximum_substeps <= 64):
        raise QualificationError("SECONDARY_SUBSTEP_POLICY_INVALID")
    if not policy.chains or len(policy.chains) > 32:
        raise QualificationError("SECONDARY_CHAIN_RESOURCE_POLICY_INVALID")
    joint_by = {j.canonical_joint_id: j for j in rig.joints}
    chain_by = {c.canonical_chain_id: c for c in rig.chains}
    if len(joint_by) != len(rig.joints) or len(chain_by) != len(rig.chains):
        raise QualificationError("SECONDARY_RIG_DUPLICATE_JOINT_OR_CHAIN")
    # A root/anchor may be read by many secondary branches, but a secondary
    # animated joint is owned by exactly one chain.
    owned: set[str] = set()
    seen_chains: set[str] = set()
    forbidden: set[str] = set()
    for constraint in rig.constraints:
        if constraint.canonical_chain_id:
            chain = chain_by.get(constraint.canonical_chain_id)
            if chain is None:
                raise QualificationError("SECONDARY_CONSTRAINT_CHAIN_MISSING")
            forbidden.update(chain.canonical_joint_ids)
        forbidden.update(constraint.output_canonical_joint_ids)
    validated=[]
    for declared in policy.chains:
        cid=declared.canonical_chain_id
        if cid in seen_chains:
            raise QualificationError("SECONDARY_CHAIN_DUPLICATE")
        seen_chains.add(cid)
        chain=chain_by.get(cid)
        if chain is None:
            raise QualificationError("SECONDARY_CHAIN_NOT_DECLARED")
        ids=tuple(chain.canonical_joint_ids)
        if len(ids)<2 or len(ids)>128 or len(set(ids))!=len(ids):
            raise QualificationError("SECONDARY_CHAIN_LENGTH_OR_CYCLE_INVALID")
        if (not math.isfinite(float(declared.frequency_hz))
                or not 0.05 <= declared.frequency_hz <= 30.0
                or not math.isfinite(float(declared.max_deviation_rad))
                or not 0 < declared.max_deviation_rad < math.pi/2):
            raise QualificationError("SECONDARY_DYNAMICS_POLICY_INVALID")
        for index, jid in enumerate(ids):
            if jid not in joint_by:
                raise QualificationError("SECONDARY_CHAIN_JOINT_UNKNOWN")
            if index and joint_by[jid].parent_canonical_joint_id != ids[index-1]:
                raise QualificationError("SECONDARY_CHAIN_PARENT_DRIFT")
        secondary_ids=ids[1:]
        if any(jid in forbidden for jid in secondary_ids):
            raise QualificationError("SECONDARY_PRIMARY_CONSTRAINT_OWNERSHIP_CONFLICT")
        if set(secondary_ids) & owned:
            raise QualificationError("SECONDARY_JOINT_MULTI_CHAIN_OWNERSHIP")
        # An explicitly attached prop cannot silently be allowed to lag
        # behind the parent joint carrying a sword, etc. A separate
        # authored secondary attachment policy is needed.
        attached={anchor.parent_canonical_joint_id for anchor in rig.attachment_anchors
                  if anchor.parent_canonical_joint_id is not None}
        if set(secondary_ids) & attached:
            raise QualificationError("SECONDARY_ATTACHMENT_OWNER_CONFLICT")
        owned.update(secondary_ids)
        validated.append((declared,ids))
    return joint_by,tuple(validated),tuple(sorted(owned))


def step_secondary_motion_v1(
    rig: CanonicalRigIRV2,
    primary_pose: MechanicsPoseIRV1,
    policy: SecondaryMotionPolicyV1,
    *,
    previous_state: SecondaryMotionStateV1 | None = None,
    dt_seconds: float | None = None,
) -> SecondaryMotionStepResultV1:
    joint_by, validated, owned = _params(rig,policy)
    if (primary_pose.rig_lineage_hash != rig.rig_lineage_hash or
            primary_pose.control_program_hash != rig.control_program_hash):
        raise QualificationError("SECONDARY_PRIMARY_POSE_LINEAGE_DRIFT")
    primary={j.canonical_joint_id:j for j in primary_pose.joints}
    if len(primary)!=len(primary_pose.joints) or set(primary)!=set(joint_by):
        raise QualificationError("SECONDARY_PRIMARY_POSE_JOINT_COVERAGE_DRIFT")
    primary_positions={jid:_v3(row.solved_P,"SECONDARY_PRIMARY_POSITION_INVALID")
                       for jid,row in primary.items()}
    for j in primary_pose.joints:
        R=np.asarray(j.rotation_basis,dtype=np.float64)
        if (R.shape != (3,3) or not np.isfinite(R).all()
                or np.max(np.abs(R.T@R-np.eye(3)))>1e-5 or
                np.linalg.det(R)<0.999):
            raise QualificationError("SECONDARY_PRIMARY_ROTATION_INVALID")
    if previous_state is None:
        if dt_seconds is not None:
            raise QualificationError("SECONDARY_INIT_REQUIRES_NO_DT")
        previous={}
        step_count=0
        integration_steps=0
        prior_tick_count=0
    else:
        if (previous_state.rig_lineage_hash != rig.rig_lineage_hash or
                previous_state.control_program_hash != rig.control_program_hash or
                previous_state.secondary_joint_ids != owned):
            raise QualificationError("SECONDARY_STATE_LINEAGE_OR_DOMAIN_DRIFT")
        if dt_seconds is None:
            raise QualificationError("SECONDARY_STEP_DT_REQUIRED")
        try:
            dt=float(dt_seconds)
        except (ValueError,TypeError) as exc:
            raise QualificationError("SECONDARY_STEP_DT_INVALID") from exc
        if not math.isfinite(dt) or not 0 < dt <= policy.maximum_dt_seconds:
            raise QualificationError("SECONDARY_STEP_DT_INVALID")
        previous={j.canonical_joint_id:j for j in previous_state.joints}
        if len(previous)!=len(previous_state.joints) or set(previous)!=set(owned):
            raise QualificationError("SECONDARY_STATE_JOINT_COVERAGE_DRIFT")
        step_count=previous_state.step_count+1
        integration_steps=max(1,int(math.ceil(
            (dt/policy.fixed_step_seconds)-1e-10
        )))
        if integration_steps>policy.maximum_substeps:
            raise QualificationError("SECONDARY_SUBSTEP_RESOURCE_BOUND")
        prior_tick_count=int(previous_state.metadata.get("simulation_tick_count",0))
        if prior_tick_count<0 or prior_tick_count>1000000000:
            raise QualificationError("SECONDARY_STATE_TICK_COUNT_INVALID")
    result_positions=dict(primary_positions)
    state_out:dict[str,SecondaryJointStateV1]={}
    rotations:dict[str,np.ndarray]={}
    max_segment_error=0.0
    max_deviation=0.0
    for cfg,ids in validated:
        omega=2*math.pi*float(cfg.frequency_hz)
        for parent_id, jid in zip(ids,ids[1:]):
            parent=result_positions[parent_id]
            primary_parent=primary_positions[parent_id]
            original=primary_positions[jid]
            native_segment=original-primary_parent
            length=float(np.linalg.norm(native_segment))
            if not math.isfinite(length) or length<=1e-8:
                raise QualificationError("SECONDARY_DEGENERATE_SEGMENT")
            target=parent+native_segment
            if previous_state is None:
                new_position=target.copy()
                new_velocity=np.zeros(3,dtype=np.float64)
            else:
                old_position=_v3(previous[jid].world_P,"SECONDARY_STATE_POSITION_INVALID")
                old_velocity=_v3(previous[jid].world_velocity,"SECONDARY_STATE_VELOCITY_INVALID")
                # Exact critically damped homogeneous response to a fixed
                # target over dt. Target moves across steps, never inferred
                # from unobserved source forces.
                # Independent fixed-step integration of the bounded position
                # response. For a fixed primary target, one 1/30s frame and
                # two 1/60s frames follow identical 1/60s physics ticks.
                # No hidden variable-step lumping or arbitrary frame-rate
                # dependence is introduced within the declared substep grid.
                sub_dt=dt/integration_steps
                new_position=old_position.copy()
                new_velocity=old_velocity.copy()
                forward=native_segment/length
                for _ in range(integration_steps):
                    error=new_position-target
                    decay=math.exp(-omega*sub_dt)
                    coefficient=new_velocity+omega*error
                    unconstrained=target+(error+coefficient*sub_dt)*decay
                    radial=unconstrained-parent
                    if np.linalg.norm(radial)<=1e-10:
                        radial=native_segment.copy()
                    direction=_unit(radial,"SECONDARY_RADIAL_DEGENERATE")
                    cos_angle=float(np.clip(np.dot(direction,forward),-1.,1.))
                    angle=math.acos(cos_angle)
                    if angle>cfg.max_deviation_rad:
                        transverse=direction-cos_angle*forward
                        transverse=_unit(transverse,"SECONDARY_CONE_DEGENERATE")
                        direction=(math.cos(cfg.max_deviation_rad)*forward
                                   +math.sin(cfg.max_deviation_rad)*transverse)
                    updated=parent+length*direction
                    new_velocity=(updated-new_position)/sub_dt
                    if not np.isfinite(new_velocity).all():
                        raise QualificationError("SECONDARY_VELOCITY_NONFINITE")
                    new_position=updated
            segment=new_position-parent
            max_segment_error=max(max_segment_error,
                                  abs(float(np.linalg.norm(segment))-length))
            deviation=math.acos(float(np.clip(
                np.dot(segment/length,native_segment/length),-1.,1.)))
            max_deviation=max(max_deviation,deviation)
            result_positions[jid]=new_position
            swing=_minimal_rotation(native_segment,segment)
            rotations[jid]=swing@np.asarray(primary[jid].rotation_basis,dtype=np.float64)
            state_out[jid]=SecondaryJointStateV1(
                jid,tuple(map(float,new_position)),tuple(map(float,new_velocity)),
            )
    rows=tuple(
        replace(j,
            solved_P=tuple(map(float,result_positions[j.canonical_joint_id])),
            rotation_basis=tuple(tuple(map(float,r)) for r in rotations[j.canonical_joint_id])
                                 ) if j.canonical_joint_id in owned else j
        for j in primary_pose.joints
    )
    state=SecondaryMotionStateV1(
        rig_lineage_hash=rig.rig_lineage_hash,
        control_program_hash=rig.control_program_hash,
        secondary_joint_ids=owned,
        joints=tuple(state_out[jid] for jid in owned),
        step_count=step_count,
        metadata={"authority":"SECONDARY_DETERMINISTIC_RUNTIME_STATE_ONLY",
                  "physical_truth_minted":False,"training_authorized":False,
                  "simulation_tick_count":prior_tick_count+integration_steps,
                  "integration_substeps_this_frame":integration_steps,
                  "fixed_step_seconds":policy.fixed_step_seconds},
    )
    return SecondaryMotionStepResultV1(
        pose=replace(primary_pose,joints=rows,metadata={
            **primary_pose.metadata,
            "secondary_motion":"CRITICALLY_DAMPED_POSITION_LAG_RESEARCH_ONLY",
            "secondary_joint_ids":owned,
            "primary_constraint_execution_unchanged":True,
            "physical_parameters_source_qualified":False,
        }),
        state=state,
        max_secondary_segment_error=max_segment_error,
        max_secondary_angular_deviation_rad=max_deviation,
        status=("PASS_SECONDARY_MOTION_BOUNDED" if (
            max_segment_error <= 1e-8 and
            max_deviation <= max(cfg.max_deviation_rad for cfg,_ in validated)+1e-8
        ) else "FAIL_SECONDARY_MOTION_BOUNDS"),
        metadata={"rig_or_skin_mutated":False,
                  "primary_ik_or_contact_controls_mutated":False,
                  "secondary_policy_is_not_source_truth":True,
                  "physics_step_policy":"BOUNDED_FIXED_SUBSTEPS_V1",
                  "simulation_substeps":integration_steps,
                  "reset_is_explicit_previous_state_none":True,
                  "collision_cloth_force_simulation_claimed":False,
                  "graphics_authority_claimed":False},
    )


__all__=[
    "SecondaryChainPolicyV1","SecondaryMotionPolicyV1",
    "SecondaryJointStateV1","SecondaryMotionStateV1",
    "SecondaryMotionStepResultV1","step_secondary_motion_v1",
]
