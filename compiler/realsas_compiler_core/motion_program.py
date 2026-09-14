from __future__ import annotations

"""Semantic motion-program authoring for the recovered RealSaS motion engine.

Historical SaS separated animation intent from the concrete rig. This module
restores that boundary on top of the current canonical compiler: presets target
semantic selectors (CHEST, ARM_R:1, HAND_R, ...), then compile deterministically
against the current qualified HumanoidJointRoleBindingIR.

The program compiler authors rotation-only tracks and typed intent metadata. It
never infers or replaces mesh, rig, skin, components, pixels or canonical identity.
Contact locking, arc solving, corrective deformation and secondary motion remain
separate qualified stages; program-level hints are intent, never proof.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from .hashing import content_sha256
from .motion_quality import MotionPhaseSpec, MotionQualityProfile
from .semantic_joint_roles import HumanoidJointRoleBindingIR, infer_humanoid_joint_roles
from .types import QualificationError
from .v4 import build_joint_track, build_motion_state, validate_motion_against_mechanical
from .v4_types import JointTransformKeyIR, MotionClipIR, MotionStateIR


SCHEMA = "RealSaS.SemanticMotionProgram.v1"
PRODUCER = "RealSaS.MotionEngine.SemanticProgramCompiler.v1"


@dataclass(frozen=True)
class MotionProgramRotationKeyIR:
    normalized_time: float
    rotation_deg: float

    def validate(self) -> None:
        if not (0.0 <= float(self.normalized_time) <= 1.0):
            raise ValueError("motion program key normalized_time out of range")
        if not (-720.0 <= float(self.rotation_deg) <= 720.0):
            raise ValueError("motion program rotation outside bounded policy")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MotionProgramRoleTrackIR:
    role_selector: str
    keys: tuple[MotionProgramRotationKeyIR, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not str(self.role_selector).strip() or len(self.keys) < 2:
            raise ValueError("motion program role track requires selector and >=2 keys")
        previous = -1.0
        for key in self.keys:
            key.validate()
            if float(key.normalized_time) <= previous:
                raise ValueError("motion program role track times must be strictly increasing")
            previous = float(key.normalized_time)
        if abs(float(self.keys[0].normalized_time)) > 1e-12 or abs(float(self.keys[-1].normalized_time)-1.0) > 1e-12:
            raise ValueError("motion program role track must cover normalized 0..1")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MotionProgramContactIntentIR:
    contact_id: str
    role_selector: str
    normalized_start: float
    normalized_end: float
    side: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.contact_id or not self.role_selector:
            raise ValueError("motion program contact intent requires id and role selector")
        if not (0.0 <= float(self.normalized_start) < float(self.normalized_end) <= 1.0):
            raise ValueError("motion program contact interval invalid")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MotionProgramArcIntentIR:
    arc_id: str
    role_selector: str
    normalized_start: float
    normalized_end: float
    expected_arc_normal_xy: tuple[float, float]
    minimum_arc_height: float
    maximum_arc_height: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.arc_id or not self.role_selector:
            raise ValueError("motion program arc intent requires id and role selector")
        if not (0.0 <= float(self.normalized_start) < float(self.normalized_end) <= 1.0):
            raise ValueError("motion program arc interval invalid")
        if float(self.minimum_arc_height) < 0.0 or float(self.maximum_arc_height) < float(self.minimum_arc_height):
            raise ValueError("motion program arc height bounds invalid")
        x,y=map(float,self.expected_arc_normal_xy)
        if abs(x)+abs(y) <= 1e-12:
            raise ValueError("motion program arc normal cannot be zero")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MotionProgramSecondaryIntentIR:
    intent_id: str
    component_semantic: str
    normalized_time: float
    impulse_xy: tuple[float,float]
    magnitude: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.intent_id or not self.component_semantic:
            raise ValueError("secondary intent requires id and component semantic")
        if not (0.0 <= float(self.normalized_time) <= 1.0):
            raise ValueError("secondary intent time invalid")
        if not (0.0 <= float(self.magnitude) <= 4.0):
            raise ValueError("secondary intent magnitude outside bounded policy")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SemanticMotionProgramIR:
    program_id: str
    clip_id: str
    intent: str
    duration_sec: float
    loop: bool
    phases: tuple[MotionPhaseSpec, ...]
    role_tracks: tuple[MotionProgramRoleTrackIR, ...]
    contact_intents: tuple[MotionProgramContactIntentIR, ...] = ()
    arc_intents: tuple[MotionProgramArcIntentIR, ...] = ()
    secondary_intents: tuple[MotionProgramSecondaryIntentIR, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA

    def validate(self) -> None:
        if not self.program_id or not self.clip_id or not self.intent:
            raise ValueError("motion program requires ids and intent")
        if not (0.02 <= float(self.duration_sec) <= 60.0):
            raise ValueError("motion program duration outside bounded policy")
        if len(self.phases) < 2 or abs(float(self.phases[0].normalized_time)) > 1e-12 or abs(float(self.phases[-1].normalized_time)-1.0) > 1e-12:
            raise ValueError("motion program phases must span normalized 0..1")
        previous=-1.0
        for phase in self.phases:
            if float(phase.normalized_time) <= previous:
                raise ValueError("motion program phase times must be strictly increasing")
            previous=float(phase.normalized_time)
        selectors=set()
        for track in self.role_tracks:
            track.validate()
            if track.role_selector in selectors:
                raise ValueError("motion program duplicate role selector track")
            selectors.add(track.role_selector)
            if self.loop and abs(float(track.keys[0].rotation_deg)-float(track.keys[-1].rotation_deg)) > 1e-9:
                raise ValueError("looping motion program track must close exactly")
        if not self.role_tracks:
            raise ValueError("motion program requires at least one role track")
        for row in self.contact_intents: row.validate()
        for row in self.arc_intents: row.validate()
        for row in self.secondary_intents: row.validate()

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def program_hash(self) -> str:
        self.validate()
        return content_sha256(self.to_dict())


def resolve_role_selector(binding: HumanoidJointRoleBindingIR, selector: str) -> str:
    raw=str(selector).strip().upper()
    direct={
        "ROOT":binding.root_joint_id,
        "CHEST":binding.chest_joint_id,
        "HEAD":binding.head_joint_id,
        "HAND_L":binding.left_hand_slot_joint_id,
        "HAND_R":binding.right_hand_slot_joint_id,
        "HAND_SLOT_LEFT":binding.left_hand_slot_joint_id,
        "HAND_SLOT_RIGHT":binding.right_hand_slot_joint_id,
    }
    if raw in direct:
        return str(direct[raw])
    chain_map={
        "SPINE":tuple(binding.spine_chain),
        "ARM_L":tuple(binding.left_arm_chain),
        "ARM_R":tuple(binding.right_arm_chain),
        "LEG_L":tuple(binding.left_leg_chain),
        "LEG_R":tuple(binding.right_leg_chain),
    }
    if ":" not in raw:
        raise QualificationError(f"MOTION_PROGRAM_UNSUPPORTED_ROLE_SELECTOR:{selector}")
    family,index_text=raw.split(":",1)
    if family not in chain_map:
        raise QualificationError(f"MOTION_PROGRAM_UNSUPPORTED_ROLE_SELECTOR:{selector}")
    try:
        index=int(index_text)
    except ValueError as exc:
        raise QualificationError(f"MOTION_PROGRAM_INVALID_ROLE_INDEX:{selector}") from exc
    chain=chain_map[family]
    if index<0:
        index=len(chain)+index
    if index<0 or index>=len(chain):
        raise QualificationError(f"MOTION_PROGRAM_ROLE_INDEX_OUT_OF_RANGE:{selector}:{len(chain)}")
    return str(chain[index])


def quality_profile_from_program(program: SemanticMotionProgramIR, *, sample_count: int = 33) -> MotionQualityProfile:
    program.validate()
    return MotionQualityProfile(
        profile_id=f"PROGRAM_QUALITY:{program.program_id}",
        intent=str(program.intent),
        sample_count=int(sample_count),
        phases=tuple(program.phases),
        minimum_jerk_mix=float(program.metadata.get("minimum_jerk_mix",0.32)),
        hermite_tension=float(program.metadata.get("hermite_tension",0.12)),
        exact_loop_repair=bool(program.loop),
        contact_schedule_is_intent_only=True,
        metadata={
            "source_program_hash":program.program_hash,
            "contact_lock_claimed":False,
            "arc_constraint_claimed":False,
            "secondary_motion_claimed":False,
        },
    )


def compile_semantic_motion_programs(programs: Sequence[SemanticMotionProgramIR], mechanical) -> MotionStateIR:
    rows=tuple(programs)
    if not rows:
        raise ValueError("semantic motion compiler requires program")
    binding=infer_humanoid_joint_roles(mechanical.skeleton)
    by_id={j.canonical_joint_id:j for j in mechanical.skeleton.joints}
    clips=[]; tracks=[]; program_rows=[]
    clip_ids=set()
    for program in rows:
        program.validate()
        if program.clip_id in clip_ids:
            raise ValueError("semantic motion compiler duplicate clip id")
        clip_ids.add(program.clip_id)
        resolved={track.role_selector:resolve_role_selector(binding,track.role_selector) for track in program.role_tracks}
        if len(set(resolved.values())) != len(resolved):
            raise QualificationError("MOTION_PROGRAM_MULTIPLE_ROLE_TRACKS_RESOLVE_TO_SAME_JOINT")
        contact_intents=[]
        for row in program.contact_intents:
            contact_intents.append({**row.to_dict(),"canonical_joint_id":resolve_role_selector(binding,row.role_selector),"qualified":False})
        arc_intents=[]
        for row in program.arc_intents:
            arc_intents.append({**row.to_dict(),"canonical_joint_id":resolve_role_selector(binding,row.role_selector),"qualified":False})
        spec={
            "schema":SCHEMA,
            "producer":PRODUCER,
            "program_hash":program.program_hash,
            "role_binding_hash":binding.binding_hash,
            "resolved_role_map":dict(sorted(resolved.items())),
            "contact_intents":contact_intents,
            "arc_intents":arc_intents,
            "secondary_intents":[{**row.to_dict(),"qualified":False} for row in program.secondary_intents],
        }
        clip_hash=content_sha256(spec)
        clips.append(MotionClipIR(
            program.clip_id,"PRESET",clip_hash,("PRESET_MOTION","WEIGHTED_DEFORM_2D_2P5D"),
            source_ref=PRODUCER,duration_sec=float(program.duration_sec),loop=bool(program.loop),
            metadata={
                "intent":program.intent,
                "display_name":str(program.metadata.get("display_name",program.program_id)),
                "source_program_hash":program.program_hash,
                "role_binding_hash":binding.binding_hash,
                "semantic_program_compiled":True,
                "motion_phases":[phase.to_dict() for phase in program.phases],
                "contact_intents":contact_intents,
                "arc_intents":arc_intents,
                "secondary_intents":[{**row.to_dict(),"qualified":False} for row in program.secondary_intents],
                "contact_lock_claimed":False,
                "arc_constraint_claimed":False,
                "secondary_motion_claimed":False,
                "corrective_deformation_claimed":False,
                "translation_policy":"ZERO_ALL_JOINT_TRANSLATION_XY",
                **dict(program.metadata),
            },
        ))
        for track in program.role_tracks:
            jid=resolved[track.role_selector]
            if jid not in by_id:
                raise QualificationError("MOTION_PROGRAM_RESOLVED_JOINT_MISSING")
            keys=tuple(JointTransformKeyIR(
                float(key.normalized_time)*float(program.duration_sec),(0.0,0.0),float(key.rotation_deg),(1.0,1.0),0.0
            ) for key in track.keys)
            tracks.append(build_joint_track(
                f"TRACK:PROGRAM:{program.clip_id}:{jid}",program.clip_id,jid,keys,
                metadata={
                    "producer":PRODUCER,
                    "source_program_hash":program.program_hash,
                    "semantic_role_selector":track.role_selector,
                    "role_binding_hash":binding.binding_hash,
                    "translation_policy":"ZERO_ALL_JOINT_TRANSLATION_XY",
                    **dict(track.metadata),
                },
            ))
        program_rows.append({"program_id":program.program_id,"clip_id":program.clip_id,"program_hash":program.program_hash,"resolved_role_map":dict(sorted(resolved.items()))})
    state=build_motion_state(tuple(clips),tuple(tracks),metadata={
        "producer":PRODUCER,
        "semantic_motion_program_compiler":True,
        "role_binding_hash":binding.binding_hash,
        "programs":program_rows,
        "translation_policy":"ZERO_ALL_JOINT_TRANSLATION_XY",
        "contact_lock_claimed":False,
        "arc_constraint_claimed":False,
        "secondary_motion_claimed":False,
        "corrective_deformation_claimed":False,
    })
    validate_motion_against_mechanical(state,mechanical)
    return state


def _keys(rows) -> tuple[MotionProgramRotationKeyIR,...]:
    return tuple(MotionProgramRotationKeyIR(float(t),float(v)) for t,v in rows)


def build_mage_slash_program() -> SemanticMotionProgramIR:
    """A recovered SaS-style authored slash intent, retargetable through semantic roles."""
    phases=(
        MotionPhaseSpec("ANTICIPATION",0.0,"HOLD_THEN_SNAP"),
        MotionPhaseSpec("COMMIT",0.22,"SNAP"),
        MotionPhaseSpec("IMPACT",0.46,"RECOIL"),
        MotionPhaseSpec("FOLLOW_THROUGH",0.58,"HEAVY"),
        MotionPhaseSpec("RECOVER",0.76,"SETTLE"),
        MotionPhaseSpec("SETTLE",1.0,"SMOOTH"),
    )
    times=(0.0,0.22,0.46,0.58,0.76,1.0)
    def track(selector,values,**metadata):
        return MotionProgramRoleTrackIR(selector,_keys(zip(times,values)),metadata)
    tracks=(
        track("ROOT",(0.0,-2.0,4.0,6.0,2.0,0.0),purpose="weight_shift"),
        track("CHEST",(0.0,-9.0,18.0,25.0,8.0,0.0),purpose="anticipation_commit_followthrough"),
        track("HEAD",(0.0,4.0,-5.0,-9.0,-2.0,0.0),purpose="counter_motion"),
        track("ARM_R:0",(0.0,-24.0,48.0,70.0,20.0,0.0),purpose="weapon_shoulder_arc"),
        track("ARM_R:1",(0.0,-12.0,30.0,46.0,16.0,0.0),purpose="elbow_followthrough"),
        track("ARM_R:2",(0.0,-5.0,18.0,30.0,8.0,0.0),purpose="wrist_delay"),
        track("ARM_L:0",(0.0,8.0,-14.0,-18.0,-5.0,0.0),purpose="counter_balance"),
    )
    return SemanticMotionProgramIR(
        program_id="MAGE_SLASH_HISTORICAL_RECOVERY_V1",
        clip_id="mage_fit1_slash_v1",
        intent="SLASH",
        duration_sec=0.72,
        loop=False,
        phases=phases,
        role_tracks=tracks,
        arc_intents=(MotionProgramArcIntentIR(
            "WEAPON_HAND_ARC","HAND_R",0.18,0.62,(0.0,1.0),0.04,0.45,
            metadata={"historical_semantics":"SaSAuthoringArcPosePlanner","constraint_claimed":False},
        ),),
        secondary_intents=(MotionProgramSecondaryIntentIR(
            "SLASH_CAPE_IMPULSE","CAPE",0.50,(-1.0,0.15),0.65,
            metadata={"historical_semantics":"secondary_impulse","qualified":False},
        ),),
        metadata={
            "display_name":"Slash Historical",
            "minimum_jerk_mix":0.24,
            "hermite_tension":0.10,
            "impact_hit_stop_intent":True,
            "historical_lineage":"SaS_phase_arc_contact_secondary_action_authoring",
        },
    )


__all__=[
    "MotionProgramArcIntentIR","MotionProgramContactIntentIR","MotionProgramRoleTrackIR","MotionProgramRotationKeyIR",
    "MotionProgramSecondaryIntentIR","PRODUCER","SemanticMotionProgramIR","build_mage_slash_program",
    "compile_semantic_motion_programs","quality_profile_from_program","resolve_role_selector",
]
