from __future__ import annotations

"""Reproducible CPU full-frame cross-parent 6D grip + planted third limb.

Synthetic fixture, explicitly declared authored relationship, three-vertex
mesh and SciPy CPU reference. No claim of a source-native sword, a humanoid
training truth, a whole product runtime or Graphics/CAA FPS.
"""

import argparse
import json
import math
import platform
from pathlib import Path
import time

import numpy as np
import scipy

from compiler.realsas_compiler_core.mechanics_live_runtime_v1 import (
    evaluate_mechanics_intervention_frame_v1,
)
from compiler.realsas_compiler_core.mechanics_prepared_skin_v1 import (
    prepare_mechanics_skin_deformer_v1,
)
from tests.compiler.test_mechanics_relative_attachment_bridge_v1 import (
    _three_branch_foot_contact_scene, _rotated_targets, _drag_to_world,
)

SCHEMA="RealSaS.MechanicsCrossParentBridgeFullFrameCPUResearchCourt.v1"


def benchmark(*,frames: int=12, warmup: int=3)->dict:
    if not 4<=frames<=60 or not 0<=warmup<=10:
        raise ValueError("RELATIVE_BRIDGE_BENCHMARK_BUDGET_INVALID")
    mesh,rig,skin,mechanics,bridge=_three_branch_foot_contact_scene()
    prepared=prepare_mechanics_skin_deformer_v1(mesh,rig,skin)
    durations=[]
    drag_res=[]
    pin_res=[]
    bridge_position_res=[]
    bridge_angular_res=[]
    strains=[]
    prior=None
    failures=[]
    angles=np.linspace(.06,.18,frames+warmup)
    for i,angle in enumerate(angles):
        target,_=_rotated_targets(float(angle))
        started=time.perf_counter_ns()
        frame=evaluate_mechanics_intervention_frame_v1(
            mesh,rig,skin,mechanics,
            overrides=_drag_to_world(target),
            solver_mode="DECLARED_MULTI_GOAL_V1",
            multi_goal_jacobian_mode="ANALYTIC_RESEARCH_V1",
            pinned_contact_control_ids=("C_CONTACT",),
            pinned_world_targets={"C_CONTACT":(.6,-2.2,0.)},
            relative_attachment_bridges=(bridge,),
            prepared_skin=prepared,
            initial_parameters_rad=prior,
            max_allowed_constraint_residual=1e-3,
        )
        elapsed=(time.perf_counter_ns()-started)/1e6
        if frame.status!="PASS_MECHANICS_BEHAVIORAL_COURT":
            failures.append({"frame":i,"status":frame.status,"reasons":frame.failures})
            prior=None
        else:
            prior=frame.joint_parameter_solution_rad
        if i>=warmup:
            durations.append(elapsed)
            pin_res.append(float(frame.max_constraint_residual))
            bridge_position_res.append(
                frame.relative_attachment_court.max_position_residual_world
            )
            bridge_angular_res.append(
                frame.relative_attachment_court.max_orientation_residual_rad
            )
            strains.append(frame.max_segment_length_relative_error)
            joints={j.canonical_joint_id:np.asarray(j.solved_P) for j in frame.pose.joints}
            drag_res.append(float(np.linalg.norm(joints["A2"]-target)))
    if failures or len(durations)!=frames:
        raise AssertionError("RELATIVE_BRIDGE_FULL_FRAME_COURT_FAIL:"+repr(failures))
    if (max(pin_res)>1e-3 or max(bridge_position_res)>1e-3
            or max(bridge_angular_res)>1e-3 or max(strains)>1e-8
            or max(drag_res)>1e-3):
        raise AssertionError("RELATIVE_BRIDGE_FULL_FRAME_CONSTRAINT_DRIFT")
    return {
        "schema":SCHEMA,
        "status":"PASS_SYNTHETIC_3_BRANCH_CROSS_PARENT_RUNTIME_COURT",
        "sampled_frames":frames,"warmup_frames":warmup,
        "joint_count":len(rig.joints),
        "rotational_dof_count":5,
        "attachment_bridges":1,
        "simultaneous_grips":2,
        "fixed_contact_controls":["C_CONTACT"],
        "skin_vertex_count":len(mesh.vertices),
        "prepared_skin_reused":True,
        "measured_frame_time_ms":{
            "p50":float(np.median(durations)),
            "p95":float(np.percentile(durations,95)),
            "max":float(max(durations)),
        },
        "max_drag_position_residual_world":max(drag_res),
        "max_contact_position_residual_world":max(pin_res),
        "max_bridge_position_residual_world":max(bridge_position_res),
        "max_bridge_orientation_residual_rad":max(bridge_angular_res),
        "max_segment_length_relative_error":max(strains),
        "timing_scope":"CPU SciPy full interactive Mechanics pose+independent bridge court+prepared 3-vertex LBS; no Graphics/renderer",
        "environment":{
            "python":platform.python_version(),
            "platform":platform.platform(),
            "numpy":np.__version__,"scipy":scipy.__version__,
        },
        "authority":{
            "synthetic_only":True,
            "source_to_canonical_binding_proven":False,
            "shared_prop_source_truth_minted":False,
            "physics_forces_or_friction_proven":False,
            "whole_product_fps_certified":False,
            "training_authorized":False,
        },
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--frames",type=int,default=12)
    parser.add_argument("--warmup",type=int,default=3)
    args=parser.parse_args()
    result=benchmark(frames=args.frames,warmup=args.warmup)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,sort_keys=True))


if __name__=="__main__":
    main()
