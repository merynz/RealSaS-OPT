from __future__ import annotations

"""Differential DOF-scaling court for numeric vs exact FK Jacobian (CPU research).

Each subject is an explicit synthetic branching 5-joint graph, NOT source or
model output. Vary declared DOF count 3/6/9/15, solve the same reachable
drag + planted point, compare residuals, and measure warm-start wall time.
No timing thresholds are admission criteria and no graphics/FPS is claimed.
"""

import argparse
from dataclasses import replace
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import scipy

from compiler.realsas_compiler_core.mechanics_multigoal_solver_v1 import (
    solve_declared_multi_goal_v1,
)
from compiler.realsas_compiler_core.mechanics_control_solver_v1 import ControlTransformIRV1
from tests.compiler.test_mechanics_multigoal_solver_v1 import _branched_rig

SCHEMA="RealSaS.MechanicsRotationalFKJacobianScaleCourt.v1"


def _variant(n: int):
    if n not in (3,6,9,15):
        raise ValueError("MECHANICS_JACOBIAN_SCALING_DOF_POLICY")
    rig=_branched_rig()
    labels={"BASE":["RZ"],"A1":["RZ"],"A2":[],"B1":["RZ"],"B2":[]}
    extras=[
        ("BASE","RY"),("A1","RY"),("B1","RY"),
        ("BASE","RX"),("A1","RX"),("B1","RX"),
        ("A2","RZ"),("A2","RY"),("A2","RX"),
        ("B2","RZ"),("B2","RY"),("B2","RX"),
    ]
    for jid,label in extras[:n-3]:
        labels[jid].append(label)
    return replace(rig,local_frames=tuple(
        replace(r,dof_labels=tuple(labels[r.canonical_joint_id]))
        for r in rig.local_frames
    ))


def measure(*,frames: int=10,warmup: int=2)->dict:
    if not 4<=frames<=100 or not 0<=warmup<=10:
        raise ValueError("MECHANICS_JACOBIAN_SCALING_FRAMES_INVALID")
    output=[]
    for n in (3,6,9,15):
        rig=_variant(n)
        methods={}
        for mode in ("NUMERIC_V1","ANALYTIC_RESEARCH_V1"):
            elapsed=[];passed=0;errors=[];nfev=[];previous=None
            for i in range(frames+warmup):
                angle=.08+.18*(.5+.5*math.sin(2*math.pi*i/max(frames,4)))
                edit=(ControlTransformIRV1(
                    "A_HANDLE",translation=(math.cos(angle)-1.,math.sin(angle),0.),
                ),)
                t=time.perf_counter_ns()
                result=solve_declared_multi_goal_v1(
                    rig,edit,pinned_contact_control_ids=("B_HANDLE",),
                    initial_parameters_rad=previous,
                    goal_tolerance=1e-3,
                    jacobian_mode=mode,
                )
                dt=(time.perf_counter_ns()-t)/1e6
                success=result.status=="PASS_MULTIGOAL_BEHAVIORAL_COURT"
                if i>=warmup:
                    elapsed.append(dt)
                    passed+=int(success)
                    errors.append(max(result.max_drag_residual,result.max_pin_residual))
                    nfev.append(result.solver_evaluations)
                previous=result.parameters_rad if success else None
            methods[mode]={
                "passed_frames":passed,
                "max_goal_residual_world":float(max(errors)),
                "solver_nfev_p50":float(np.median(nfev)),
                "solver_time_p50_ms":float(np.median(elapsed)),
                "solver_time_p95_ms":float(np.percentile(elapsed,95)),
            }
        # Independent central-difference verification once OUTSIDE timing.
        audit=solve_declared_multi_goal_v1(
            rig,
            (ControlTransformIRV1("A_HANDLE",translation=(math.cos(.19)-1.,math.sin(.19),0.)),),
            pinned_contact_control_ids=("B_HANDLE",),
            goal_tolerance=1e-3,
            jacobian_mode="ANALYTIC_RESEARCH_V1",
            verify_analytic_jacobian=True,
        )
        derivative_error=audit.metadata["analytic_jacobian_finite_difference_max_error"]
        if (methods["NUMERIC_V1"]["passed_frames"]!=frames
                or methods["ANALYTIC_RESEARCH_V1"]["passed_frames"]!=frames
                or derivative_error>1e-6):
            raise AssertionError("MECHANICS_JACOBIAN_SCALE_FUNCTIONAL_OR_DERIVATIVE_DRIFT")
        output.append({
            "rotational_dof_count":n,
            "numeric":methods["NUMERIC_V1"],
            "analytic":methods["ANALYTIC_RESEARCH_V1"],
            "central_difference_max_abs_derivative_error":derivative_error,
            "p50_speed_ratio_numeric_over_analytic":(
                methods["NUMERIC_V1"]["solver_time_p50_ms"]
                / methods["ANALYTIC_RESEARCH_V1"]["solver_time_p50_ms"]
            ),
        })
    return {
        "schema":SCHEMA,
        "status":"PASS_DIFFERENTIAL_FK_JACOBIAN_RESEARCH_COURT",
        "frame_count_per_mode":frames,
        "warmup_per_mode":warmup,
        "fixture":"FIVE_JOINT_BRANCHED_DOF_STRESS_NOT_REAL_SOURCE",
        "variants":output,
        "timing_environment":{
            "python":platform.python_version(),"platform":platform.platform(),
            "numpy":np.__version__,"scipy":scipy.__version__,
            "scope":"SOLVER_ONLY_CPU_NO_SKIN_OR_GRAPHICS",
        },
        "authority":{
            "source_teacher_used":False,
            "learned_axis_output_used":False,
            "physical_DOF_limits_qualified":False,
            "whole_product_FPS_certified":False,
            "research_only":True,
        },
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--frames",type=int,default=10)
    p.add_argument("--warmup",type=int,default=2)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    data=measure(frames=a.frames,warmup=a.warmup)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(data,indent=2,sort_keys=True)+"\n")
    print(json.dumps(data,sort_keys=True))


if __name__=="__main__":
    main()
