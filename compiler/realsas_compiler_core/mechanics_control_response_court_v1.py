from __future__ import annotations

"""Finite-intervention effect court for already-declared Mechanics controls.

The graph-only surface selector gives POTENTIAL causal reachability, not proof
of effect. This research court actually executes the canonical solver and LBS
on small two-sided interventions at a proven canonical face, and reports only
observed local response. An unobserved control may act at another pose or through
another constraint; do not call it "dead" or automatically delete it.

No new handles, motion synthesis, graphics decisions, learned inference or
training authority. Not a per-mouse-frame algorithm; expensive compile-time
inspection only.
"""

from dataclasses import dataclass, field, replace
import math

import numpy as np

from .canonical_mesh_ir_v1 import CanonicalMeshIR
from .canonical_rig_ir_v2 import CanonicalRigIRV2
from .canonical_skin_ir_v2 import CanonicalSkinIRV2
from .canonical_mechanics_ir_v1 import CanonicalMechanicsIRV1
from .mechanics_control_solver_v1 import ControlTransformIRV1
from .mechanics_live_runtime_v1 import evaluate_mechanics_intervention_frame_v1
from .mechanics_surface_control_effects_v1 import surface_control_effect_evidence_v1
from .types import QualificationError


@dataclass(frozen=True)
class MechanicsControlResponseV1:
    canonical_control_id: str
    control_kind: str
    graph_support_mass: float
    measured_max_gain: float
    passed_probes: int
    failed_probes: int
    status: str   # OBSERVED_LOCAL_EFFECT | NO_LOCAL_EFFECT_OBSERVED | INCONCLUSIVE


@dataclass(frozen=True)
class MechanicsControlResponseCourtV1:
    triangle_vertex_ids: tuple[str, str, str]
    barycentric: tuple[float, float, float]
    graph_status: str
    controls: tuple[MechanicsControlResponseV1, ...]
    status: str
    metadata: dict = field(default_factory=dict)


def _quat_mul(a: tuple[float, float, float, float],
              b: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    aw,ax,ay,az = a
    bw,bx,by,bz = b
    out=(aw*bw-ax*bx-ay*by-az*bz,
         aw*bx+ax*bw+ay*bz-az*by,
         aw*by-ax*bz+ay*bw+az*bx,
         aw*bz+ax*by-ay*bx+az*bw)
    q=np.asarray(out,dtype=np.float64)
    n=float(np.linalg.norm(q))
    if not math.isfinite(n) or n<=1e-12:
        raise QualificationError("MECHANICS_RESPONSE_QUAT_INVALID")
    return tuple(float(x) for x in q/n)


def _hit_point(frame, vids, bary) -> np.ndarray:
    lookup={v:np.asarray(p,dtype=np.float64)
            for v,p in zip(frame.deformation.canonical_mesh_vertex_ids,
                           frame.deformation.positions)}
    if len(lookup)!=len(frame.deformation.positions) or any(v not in lookup for v in vids):
        raise QualificationError("MECHANICS_RESPONSE_DEFORM_VERTEX_BINDING_DRIFT")
    return sum((w*lookup[v] for v,w in zip(vids,bary)),np.zeros(3,dtype=np.float64))


def evaluate_declared_control_response_v1(
    mesh: CanonicalMeshIR,
    rig: CanonicalRigIRV2,
    skin: CanonicalSkinIRV2,
    mechanics: CanonicalMechanicsIRV1,
    *,
    triangle_vertex_ids: tuple[str, str, str],
    barycentric: tuple[float, float, float],
    baseline_overrides: tuple[ControlTransformIRV1, ...] = (),
    translation_step_world: float = 1e-3,
    rotation_step_rad: float = 1e-3,
    minimum_gain: float = 1e-5,
    maximum_controls: int = 16,
) -> MechanicsControlResponseCourtV1:
    if (not all(math.isfinite(float(x)) and x>0 for x in
                (translation_step_world, rotation_step_rad, minimum_gain)) or
            translation_step_world>0.1 or rotation_step_rad>0.1 or
            maximum_controls<1):
        raise QualificationError("MECHANICS_RESPONSE_PROBE_POLICY_INVALID")
    if len(rig.controls)>maximum_controls:
        raise QualificationError("MECHANICS_RESPONSE_CONTROL_RESOURCE_BOUND")

    graph=surface_control_effect_evidence_v1(
        mesh,rig,skin,triangle_vertex_ids=triangle_vertex_ids,
        barycentric=barycentric,maximum_candidates=maximum_controls,
    )
    baseline_by={c.canonical_control_id:c for c in baseline_overrides}
    if len(baseline_by)!=len(baseline_overrides):
        raise QualificationError("MECHANICS_RESPONSE_DUPLICATE_BASE_CONTROL")
    authored_ids={c.canonical_control_id for c in rig.controls}
    if set(baseline_by)-authored_ids:
        raise QualificationError("MECHANICS_RESPONSE_BASE_CONTROL_UNKNOWN")
    base=evaluate_mechanics_intervention_frame_v1(
        mesh,rig,skin,mechanics,overrides=baseline_overrides,
    )
    if base.status!="PASS_MECHANICS_BEHAVIORAL_COURT":
        raise QualificationError("MECHANICS_RESPONSE_BASELINE_NOT_SOLVED")
    p0=_hit_point(base,triangle_vertex_ids,barycentric)
    control_order={c.canonical_control_id:i for i,c in enumerate(rig.controls)}
    result:list[MechanicsControlResponseV1]=[]
    for candidate in graph.candidates:
        cid=candidate.canonical_control_id
        ckind=candidate.control_kind
        original=baseline_by.get(cid,ControlTransformIRV1(cid))
        trials=[]
        for axis in range(3):
            for sign in (-1,1):
                vec=np.zeros(3,dtype=np.float64)
                vec[axis]=sign*translation_step_world
                translation=tuple(float(x) for x in
                                  np.asarray(original.translation,dtype=np.float64)+vec)
                trials.append((replace(original,translation=translation),
                               translation_step_world))
        if ckind in {"GLOBAL","ROOT","FK","SOCKET_CONTROL"}:
            for axis in range(3):
                for sign in (-1,1):
                    q=np.zeros(4,dtype=np.float64)
                    q[0]=math.cos(rotation_step_rad/2)
                    q[axis+1]=sign*math.sin(rotation_step_rad/2)
                    quat=_quat_mul(original.rotation_quat_wxyz,tuple(q))
                    trials.append((replace(original,rotation_quat_wxyz=quat),
                                   rotation_step_rad))
        gains:list[float]=[]
        failed=0
        for perturbed,step in trials:
            edited={**baseline_by,cid:perturbed}
            # Deterministic order independent of API override insertion order.
            ordered=tuple(edited[x.canonical_control_id] for x in rig.controls
                          if x.canonical_control_id in edited)
            try:
                f=evaluate_mechanics_intervention_frame_v1(
                    mesh,rig,skin,mechanics,overrides=ordered,
                )
            except QualificationError:
                failed+=1
                continue
            if f.status!="PASS_MECHANICS_BEHAVIORAL_COURT":
                failed+=1
                continue
            p=_hit_point(f,triangle_vertex_ids,barycentric)
            gains.append(float(np.linalg.norm(p-p0))/step)
        gain=max(gains,default=0.)
        outcome=("OBSERVED_LOCAL_EFFECT" if gain>=minimum_gain else
                 "INCONCLUSIVE" if failed else "NO_LOCAL_EFFECT_OBSERVED")
        result.append(MechanicsControlResponseV1(
            canonical_control_id=cid,control_kind=ckind,
            graph_support_mass=candidate.support_mass,
            measured_max_gain=gain,passed_probes=len(gains),
            failed_probes=failed,status=outcome,
        ))
    status=("OBSERVED_EFFECTS" if any(r.status=="OBSERVED_LOCAL_EFFECT" for r in result)
            else "INCONCLUSIVE" if any(r.status=="INCONCLUSIVE" for r in result)
            else "NO_LOCAL_EFFECT_OBSERVED")
    return MechanicsControlResponseCourtV1(
        triangle_vertex_ids=triangle_vertex_ids,barycentric=barycentric,
        graph_status=graph.status,controls=tuple(result),status=status,
        metadata={
            "authority":"RESEARCH_MECHANICS_RESPONSE_ONLY",
            "local_probes_are_not_source_authored_intent":True,
            "no_local_effect_does_not_mean_permanently_inert":True,
            "solver_invoked_for_each_probe":True,
            "face_topology_verified":True,
            "controls_built_or_modified":False,
            "graphics_authority_claimed":False,
            "training_allowed":False,
        },
    )


__all__=[
    "MechanicsControlResponseV1","MechanicsControlResponseCourtV1",
    "evaluate_declared_control_response_v1",
]
