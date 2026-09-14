from __future__ import annotations

"""Recovered contact/foot-sliding analysis for qualification-owned directional motion.

Historical SaS had contact plans, foot-plant windows and sliding analyzers. This
recovery restores that measurement boundary without pretending to have solved
contact correction yet. Contact windows bind canonical joint ids to the exact
motion/skeleton authority; directional sliding is measured in admitted raster
coordinates from the Compiler-owned joint/view binding.

No root translation units are invented and no pose is repaired in this module.
A failing plan stays a failing plan until a separately qualified contact solver
produces a new motion state.
"""

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Sequence

import numpy as np

from .hashing import content_sha256
from .semantic_joint_roles import infer_humanoid_joint_roles
from .types import QualificationError


SCHEMA = "RealSaS.MotionContactPlan.v1"
PRODUCER = "RealSaS.MotionEngine.HistoricalContactRecovery.v1"


@dataclass(frozen=True)
class ContactWindowIR:
    contact_id: str
    canonical_joint_id: str
    normalized_start: float
    normalized_end: float
    side: str
    contact_kind: str = "FOOT_PLANT"
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.contact_id or not self.canonical_joint_id:
            raise ValueError("contact window requires id and canonical joint")
        if not (0.0 <= float(self.normalized_start) < float(self.normalized_end) <= 1.0):
            raise ValueError("contact window normalized interval invalid")
        if str(self.side) not in {"LEFT", "RIGHT", "CENTER", "OTHER"}:
            raise ValueError("contact window side invalid")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MotionContactPlanIR:
    clip_id: str
    source_motion_state_hash: str
    skeleton_lineage_hash: str
    windows: tuple[ContactWindowIR, ...]
    plan_hash: str
    schema_version: str = SCHEMA
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ContactSlidingMeasurementIR:
    contact_id: str
    view_index: int
    canonical_joint_id: str
    sample_count: int
    max_sliding_px: float
    rms_sliding_px: float
    start_xy: tuple[float, float]
    end_xy: tuple[float, float]
    passed: bool
    measurement_hash: str
    schema_version: str = "RealSaS.ContactSlidingMeasurement.v1"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ContactSlidingReportIR:
    contact_plan_hash: str
    directional_binding_hash: str
    max_allowed_sliding_px: float
    measurements: tuple[ContactSlidingMeasurementIR, ...]
    max_sliding_px: float
    failed_contact_ids: tuple[str, ...]
    passed: bool
    report_hash: str
    schema_version: str = "RealSaS.ContactSlidingReport.v1"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def build_contact_plan(
    *,
    clip_id: str,
    source_motion_state_hash: str,
    skeleton_lineage_hash: str,
    windows: Sequence[ContactWindowIR],
    metadata: dict | None = None,
) -> MotionContactPlanIR:
    rows = tuple(windows)
    if not clip_id or not source_motion_state_hash or not skeleton_lineage_hash or not rows:
        raise ValueError("contact plan requires exact clip/motion/skeleton binding and windows")
    ids = set()
    for row in rows:
        row.validate()
        if row.contact_id in ids:
            raise ValueError("duplicate contact window id")
        ids.add(row.contact_id)
    payload = {
        "schema": SCHEMA,
        "clip_id": str(clip_id),
        "source_motion_state_hash": str(source_motion_state_hash),
        "skeleton_lineage_hash": str(skeleton_lineage_hash),
        "windows": [row.to_dict() for row in rows],
        "metadata": dict(metadata or {}),
    }
    return MotionContactPlanIR(
        str(clip_id), str(source_motion_state_hash), str(skeleton_lineage_hash), rows,
        content_sha256(payload), metadata=dict(metadata or {}),
    )


def build_mage_run_contact_plan(mechanical, motion_state) -> MotionContactPlanIR:
    binding = infer_humanoid_joint_roles(mechanical.skeleton)
    if not binding.left_leg_chain or not binding.right_leg_chain:
        raise QualificationError("MOTION_CONTACT_REQUIRES_BILATERAL_LEG_CHAINS")
    clip_id = "mage_fit1_run_v2"
    if clip_id not in {clip.clip_id for clip in motion_state.clips}:
        raise QualificationError("MOTION_CONTACT_MAGE_RUN_CLIP_MISSING")
    left = binding.left_leg_chain[-1]
    right = binding.right_leg_chain[-1]
    windows = (
        ContactWindowIR("RUN_LEFT_PLANT_A", left, 0.00, 0.16, "LEFT", metadata={"phase":"LEFT_PLANT"}),
        ContactWindowIR("RUN_RIGHT_PLANT", right, 0.44, 0.66, "RIGHT", metadata={"phase":"RIGHT_PLANT"}),
        ContactWindowIR("RUN_LEFT_PLANT_B", left, 0.88, 1.00, "LEFT", metadata={"phase":"LEFT_PLANT_WRAP"}),
    )
    return build_contact_plan(
        clip_id=clip_id,
        source_motion_state_hash=motion_state.motion_state_hash,
        skeleton_lineage_hash=mechanical.skeleton.skeleton_lineage_hash,
        windows=windows,
        metadata={
            "producer": PRODUCER,
            "contact_lock_claimed": False,
            "contact_windows_from_historical_phase_locomotion": True,
            "terminal_leg_joint_policy": "CANONICAL_SEMANTIC_LEG_CHAIN_TERMINAL",
        },
    )


def _track_map(motion_state, clip_id: str):
    out = {}
    for track in motion_state.joint_tracks:
        if str(track.clip_id) != str(clip_id):
            continue
        jid = str(track.canonical_joint_id)
        if jid in out:
            raise QualificationError("MOTION_CONTACT_DUPLICATE_JOINT_TRACK")
        out[jid] = track
    return out


def _rotation(track, time_sec: float) -> float:
    if track is None or not track.keys:
        return 0.0
    keys = tuple(track.keys)
    t = float(time_sec)
    if t <= float(keys[0].time_sec):
        return float(keys[0].rotation_deg)
    if t >= float(keys[-1].time_sec):
        return float(keys[-1].rotation_deg)
    for a, b in zip(keys[:-1], keys[1:]):
        ta, tb = float(a.time_sec), float(b.time_sec)
        if ta <= t <= tb:
            if tb <= ta:
                return float(b.rotation_deg)
            u = (t - ta) / (tb - ta)
            return (1.0-u)*float(a.rotation_deg)+u*float(b.rotation_deg)
    return float(keys[-1].rotation_deg)


def _T(x: float, y: float):
    out=np.eye(3,dtype=np.float64); out[0,2]=float(x); out[1,2]=float(y); return out


def _R(deg: float):
    a=math.radians(float(deg)); c,s=math.cos(a),math.sin(a)
    return np.asarray(((c,-s,0.0),(s,c,0.0),(0.0,0.0,1.0)),dtype=np.float64)


def _topological_skeleton(skeleton):
    by_id={str(j.canonical_joint_id):j for j in skeleton.joints}
    if not by_id:
        raise QualificationError("MOTION_CONTACT_EMPTY_SKELETON")
    visiting=set(); done=set(); order=[]
    def visit(jid):
        if jid in done: return
        if jid in visiting: raise QualificationError("MOTION_CONTACT_SKELETON_CYCLE")
        visiting.add(jid)
        parent=by_id[jid].parent_canonical_id
        if parent is not None:
            parent=str(parent)
            if parent not in by_id: raise QualificationError("MOTION_CONTACT_MISSING_PARENT")
            visit(parent)
        visiting.remove(jid); done.add(jid); order.append(jid)
    for jid in sorted(by_id): visit(jid)
    return tuple(order),by_id


def _pivot_lookup(binding, view_index: int) -> dict[str, tuple[float,float]]:
    rows={}
    for pivot in binding.joint_pivots:
        if int(pivot.view_index) != int(view_index):
            continue
        jid=str(pivot.canonical_joint_id)
        if jid in rows:
            raise QualificationError("MOTION_CONTACT_DUPLICATE_DIRECTIONAL_PIVOT")
        xy=tuple(map(float,pivot.raster_xy))
        if len(xy)!=2 or not all(math.isfinite(v) for v in xy):
            raise QualificationError("MOTION_CONTACT_INVALID_DIRECTIONAL_PIVOT")
        rows[jid]=xy
    return rows


def joint_positions_for_view(skeleton, motion_state, clip_id: str, binding, view_index: int, time_sec: float) -> dict[str, tuple[float,float]]:
    order,by_id=_topological_skeleton(skeleton)
    pivots=_pivot_lookup(binding,view_index)
    if set(by_id)-set(pivots):
        raise QualificationError("MOTION_CONTACT_DIRECTIONAL_PIVOT_SET_INCOMPLETE")
    tracks=_track_map(motion_state,clip_id)
    posed={}
    out={}
    for jid in order:
        pivot=np.asarray(pivots[jid],dtype=np.float64)
        parent=by_id[jid].parent_canonical_id
        delta=_R(_rotation(tracks.get(jid),time_sec))
        if parent is None:
            posed[jid]=_T(*pivot)@delta
        else:
            parent=str(parent)
            parent_pivot=np.asarray(pivots[parent],dtype=np.float64)
            posed[jid]=posed[parent]@_T(*(pivot-parent_pivot))@delta
        xy=(posed[jid]@np.asarray((0.0,0.0,1.0),dtype=np.float64))[:2]
        out[jid]=(float(xy[0]),float(xy[1]))
    return out


def measure_contact_sliding(
    *,
    product,
    directional_binding,
    plan: MotionContactPlanIR,
    samples_per_window: int = 9,
    max_allowed_sliding_px: float = 2.0,
) -> ContactSlidingReportIR:
    if int(samples_per_window)<3:
        raise ValueError("contact sliding requires >=3 samples/window")
    if not (0.0 < float(max_allowed_sliding_px) <= 64.0):
        raise ValueError("contact sliding pixel threshold invalid")
    if str(plan.source_motion_state_hash) != str(product.motion_state.motion_state_hash):
        raise QualificationError("MOTION_CONTACT_STALE_MOTION_PLAN")
    if str(plan.skeleton_lineage_hash) != str(product.mechanical_state.skeleton.skeleton_lineage_hash):
        raise QualificationError("MOTION_CONTACT_STALE_SKELETON_PLAN")
    if str(directional_binding.source_product_state_hash) != str(product.product_state_hash):
        raise QualificationError("MOTION_CONTACT_STALE_DIRECTIONAL_BINDING")
    clips={str(c.clip_id):c for c in product.motion_state.clips}
    clip=clips.get(str(plan.clip_id))
    if clip is None or float(clip.duration_sec)<=0.0:
        raise QualificationError("MOTION_CONTACT_CLIP_BINDING_MISSING")
    duration=float(clip.duration_sec)
    measurements=[]
    for view in range(8):
        for window in plan.windows:
            times=np.linspace(float(window.normalized_start)*duration,float(window.normalized_end)*duration,int(samples_per_window))
            xy=[]
            for t in times:
                positions=joint_positions_for_view(product.mechanical_state.skeleton,product.motion_state,plan.clip_id,directional_binding,view,float(t))
                if window.canonical_joint_id not in positions:
                    raise QualificationError("MOTION_CONTACT_JOINT_POSITION_MISSING")
                xy.append(np.asarray(positions[window.canonical_joint_id],dtype=np.float64))
            anchor=xy[0]
            distances=np.asarray([float(np.linalg.norm(p-anchor)) for p in xy],dtype=np.float64)
            max_slide=float(distances.max(initial=0.0)); rms=float(np.sqrt(np.mean(distances*distances)))
            passed=max_slide<=float(max_allowed_sliding_px)
            provisional={
                "contact_id":window.contact_id,"view_index":view,"canonical_joint_id":window.canonical_joint_id,
                "sample_count":len(xy),"max_sliding_px":max_slide,"rms_sliding_px":rms,
                "start_xy":tuple(map(float,xy[0])),"end_xy":tuple(map(float,xy[-1])),"passed":bool(passed),
            }
            measurements.append(ContactSlidingMeasurementIR(
                window.contact_id,view,window.canonical_joint_id,len(xy),max_slide,rms,
                tuple(map(float,xy[0])),tuple(map(float,xy[-1])),bool(passed),content_sha256(provisional),
                metadata={"contact_kind":window.contact_kind,"side":window.side},
            ))
    failed=tuple(sorted({m.contact_id for m in measurements if not m.passed}))
    max_slide=max((m.max_sliding_px for m in measurements),default=0.0)
    provisional={
        "contact_plan_hash":plan.plan_hash,
        "directional_binding_hash":str(directional_binding.binding_set_hash),
        "max_allowed_sliding_px":float(max_allowed_sliding_px),
        "measurement_hashes":[m.measurement_hash for m in measurements],
        "max_sliding_px":max_slide,
        "failed_contact_ids":failed,
        "passed":not failed,
    }
    return ContactSlidingReportIR(
        plan.plan_hash,str(directional_binding.binding_set_hash),float(max_allowed_sliding_px),tuple(measurements),
        float(max_slide),failed,not failed,content_sha256(provisional),
        metadata={
            "producer":PRODUCER,
            "measurement_space":"QUALIFIED_DIRECTIONAL_RASTER_PIXELS",
            "contact_correction_applied":False,
            "root_translation_invented":False,
        },
    )


__all__=[
    "ContactSlidingMeasurementIR","ContactSlidingReportIR","ContactWindowIR","MotionContactPlanIR","PRODUCER",
    "build_contact_plan","build_mage_run_contact_plan","joint_positions_for_view","measure_contact_sliding",
]
