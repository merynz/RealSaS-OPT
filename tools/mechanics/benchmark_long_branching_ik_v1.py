from __future__ import annotations

"""Morphology/DOF stress court for independent Mechanics research.

These are *synthetic* declared-rig programs, never source teacher truth:
two-branch, four-branch, six-branch articulated trees with 12 movable joints
per limb. Neighboring limbs occupy different XY/XZ/YZ rest planes.

Critical distinction: every declared rotation DOF can locally influence its
declared tip, but a tip's position is only a 3D observation. Rank deficiency
is measured and REPORTED; never claim 72 independently identifiable DOFs
from six endpoint observations.

A single planned branch-root rotation defines a reachable DRAG target;
other explicitly declared goals are WORLD pins. This is a mechanics executor
court, not an automatic control-program discovery claim or 60 FPS proof.
"""

import argparse
from dataclasses import dataclass
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import scipy

from compiler.realsas_compiler_core.canonical_rig_ir_v2 import (
    CanonicalDeformJointIRV2, CanonicalJointFrameIRV2,
    CanonicalRigChainIRV2, CanonicalRigControlIRV2,
    CanonicalRigConstraintIRV2, CanonicalRigIRV2,
)
from compiler.realsas_compiler_core.mechanics_control_solver_v1 import ControlTransformIRV1
from compiler.realsas_compiler_core.mechanics_multigoal_solver_v1 import (
    MechanicsMultiGoalResultV1, solve_declared_multi_goal_v1,
)

SCHEMA="RealSaS.MechanicsLongBranchingMorphologyCourt.v1"
BASIS=((1.,0.,0.),(0.,1.,0.),(0.,0.,1.))
PLANES=(
    ("RZ",np.array((1.,0.,0.)),np.array((0.,1.,0.)),np.array((0.,0.,1.))),
    ("RY",np.array((1.,0.,0.)),np.array((0.,0.,1.)),np.array((0.,1.,0.))),
    ("RX",np.array((0.,1.,0.)),np.array((0.,0.,1.)),np.array((1.,0.,0.))),
)


@dataclass(frozen=True)
class MorphologyCaseV1:
    rig: CanonicalRigIRV2
    overrides: tuple[ControlTransformIRV1,...]
    pins: tuple[str,...]
    pin_world_targets: dict[str,tuple[float,float,float]]
    joint_axes: tuple[tuple[str,str],...]
    targets_world: dict[str,tuple[float,float,float]]
    morphology: str


def _world_point(p: np.ndarray) -> tuple[float,float,float]:
    return tuple(map(float,p))


def _rotated_about_axis(v: np.ndarray, axis: np.ndarray, theta: float) -> np.ndarray:
    # Independent Rodrigues reference, not solver forward-kinematic helper.
    return (
        v*math.cos(theta)
        + np.cross(axis,v)*math.sin(theta)
        + axis*float(np.dot(axis,v))*(1.-math.cos(theta))
    )


def make_morphology_case_v1(dof_count: int) -> MorphologyCaseV1:
    if dof_count not in (24,48,72):
        raise ValueError("MECHANICS_LONG_MORPHOLOGY_DOF_POLICY")
    branches=dof_count//12
    root="J:ROOT"
    joints=[CanonicalDeformJointIRV2(root,(0.,0.,0.),None,"fixture:root")]
    frames=[CanonicalJointFrameIRV2(root,BASIS,())]
    controls=[];chains=[];constraints=[]
    overrides=[];pins=[];anchors={};targets={}
    dof_axis_records=[]
    for branch in range(branches):
        axis,basis_a,basis_b,rot_axis=PLANES[branch%len(PLANES)]
        center=.37+branch*1.31
        prev=root
        path=[root]
        p=np.zeros(3)
        first=None
        for local in range(12):
            # Curved rest path, not a straight kinematic rod.
            heading=center+0.22*math.sin(local*.65+branch*.19)
            step=.24*(math.cos(heading)*basis_a+math.sin(heading)*basis_b)
            p=p+step
            jid=f"J:{branch}:{local}"
            joints.append(CanonicalDeformJointIRV2(
                jid,_world_point(p),prev,f"fixture:{jid}",
            ))
            frames.append(CanonicalJointFrameIRV2(jid,BASIS,(axis,)))
            dof_axis_records.append((jid,axis))
            path.append(jid)
            prev=jid
            if first is None: first=p.copy()
        last_heading=center+0.22*math.sin(12*.65+branch*.19)
        p=p+.24*(math.cos(last_heading)*basis_a+math.sin(last_heading)*basis_b)
        tip=f"TIP:{branch}"
        joints.append(CanonicalDeformJointIRV2(
            tip,_world_point(p),prev,f"fixture:{tip}",
        ))
        frames.append(CanonicalJointFrameIRV2(tip,BASIS,()))
        path.append(tip)
        cid=f"C:{branch}"
        chain_id=f"CHAIN:{branch}"
        controls.append(CanonicalRigControlIRV2(
            cid,"IK_EFFECTOR",(tip,),None,_world_point(p),f"control:{branch}",
        ))
        chains.append(CanonicalRigChainIRV2(
            chain_id,tuple(path),"GENERIC_KINEMATIC_CHAIN",f"chain:{branch}",
        ))
        constraints.append(CanonicalRigConstraintIRV2(
            f"Q:{branch}","IK",(cid,),(tip,),chain_id,
            {"p0":1.,"p2":1.},f"constraint:{branch}",
        ))
        if branch%2==0:
            pins.append(cid)
            anchors[cid]=_world_point(p)
            targets[cid]=_world_point(p)
        else:
            angle=.10+.012*(branch%3)
            target=first+_rotated_about_axis(p-first,rot_axis,angle)
            overrides.append(ControlTransformIRV1(
                cid,translation=_world_point(target-p),
            ))
            targets[cid]=_world_point(target)
    rig=CanonicalRigIRV2(
        joints=tuple(joints),local_frames=tuple(frames),
        controls=tuple(controls),chains=tuple(chains),
        constraints=tuple(constraints),attachment_anchors=(),
        deform_root_joint_ids=(root,),
        mesh_lineage_hash="1"*64,
        feature_evidence_binding_hash="2"*64,
        proposal_evidence_hash="3"*64,
        rig_program_qualification_hash="4"*64,
        deform_structure_hash="5"*64,
        control_program_hash="6"*64,
        rig_lineage_hash="7"*64,
        metadata={"authority":"SYNTHETIC_MECHANICS_MORPHOLOGY_RESEARCH_ONLY"},
    )
    assert len(dof_axis_records)==dof_count
    return MorphologyCaseV1(rig,tuple(overrides),tuple(pins),anchors,
                            tuple(dof_axis_records),targets,
                            {2:"BIFURCATED_2",4:"QUADRUPED_LIKE_4",6:"RADIAL_6"}[branches])


def _observability(case: MorphologyCaseV1) -> dict:
    """Independent rest-position endpoint Jacobian based on Rodrigues/cross."""
    by_id={j.canonical_joint_id:j for j in case.rig.joints}
    normal={"RX":np.array((1.,0.,0.)),
            "RY":np.array((0.,1.,0.)),
            "RZ":np.array((0.,0.,1.))}
    entries=np.zeros((3*len(case.rig.controls),len(case.joint_axes)))
    for gi,control in enumerate(case.rig.controls):
        tip=control.target_canonical_joint_ids[0]
        tip_p=np.asarray(by_id[tip].P,dtype=np.float64)
        current=tip
        ancestors=set()
        while current is not None:
            ancestors.add(current)
            current=by_id[current].parent_canonical_joint_id
        for col,(jid,axis) in enumerate(case.joint_axes):
            if jid not in ancestors:
                continue
            pivot=np.asarray(by_id[jid].P,dtype=np.float64)
            entries[3*gi:3*gi+3,col]=np.cross(normal[axis],tip_p-pivot)
    magnitude=np.linalg.norm(entries,axis=0)
    nonzero=int(np.sum(magnitude>1e-8))
    rank=int(np.linalg.matrix_rank(entries,tol=1e-8))
    return {
        "declared_rotation_dof_count":len(case.joint_axes),
        "locally_nonzero_endpoint_columns":nonzero,
        "endpoint_position_observation_dimension":entries.shape[0],
        "endpoint_jacobian_rank":rank,
        "unidentifiable_joint_space_dimension":len(case.joint_axes)-rank,
        "all_dofs_individually_affect_an_endpoint":bool(nonzero==len(case.joint_axes)),
        "all_dofs_independently_identifiable":bool(rank==len(case.joint_axes)),
        "measurement_type":"REST_ENDPOINT_POSITION_DERIVATIVE_ONLY",
        "physical_DOF_limit_authority":False,
    }


def solve_and_court_v1(
    case: MorphologyCaseV1,
    *,
    jacobian_mode: str="ANALYTIC_RESEARCH_V1",
    max_nfev: int=90,
    derivative_check: bool=False,
    active_parameter_keys: tuple[tuple[str,str],...] | None=None,
    active_parameter_policy: str="EXPLICIT_OR_ALL",
) -> dict:
    started=time.perf_counter()
    result: MechanicsMultiGoalResultV1=solve_declared_multi_goal_v1(
        case.rig,case.overrides,
        pinned_contact_control_ids=case.pins,
        pinned_world_targets=case.pin_world_targets,
        jacobian_mode=jacobian_mode,
        goal_tolerance=2e-3,
        regularization=1e-8,
        max_nfev=max_nfev,
        verify_analytic_jacobian=derivative_check,
        active_parameter_keys=active_parameter_keys,
        active_parameter_policy=active_parameter_policy,
    )
    elapsed=(time.perf_counter()-started)*1000
    by_joint={j.canonical_joint_id:np.asarray(j.solved_P) for j in result.pose.joints}
    by_rest={j.canonical_joint_id:np.asarray(j.P) for j in case.rig.joints}
    maximum_strain=0.
    for j in case.rig.joints:
        parent=j.parent_canonical_joint_id
        if parent is None:
            continue
        d0=float(np.linalg.norm(by_rest[j.canonical_joint_id]-by_rest[parent]))
        d1=float(np.linalg.norm(by_joint[j.canonical_joint_id]-by_joint[parent]))
        maximum_strain=max(maximum_strain,abs(d1-d0))
    by_control={g.canonical_control_id:g for g in result.goals}
    if set(by_control)!=set(case.targets_world):
        raise AssertionError("MECHANICS_MORPHOLOGY_GOAL_COVERAGE_DRIFT")
    if any(np.linalg.norm(np.asarray(g.target_P)-case.targets_world[cid])>1e-10
           for cid,g in by_control.items()):
        raise AssertionError("MECHANICS_MORPHOLOGY_GOAL_TARGET_DRIFT")
    return {
        "status":result.status,
        "solver_mode":jacobian_mode,
        "runtime_ms":elapsed,
        "solver_evaluations":result.solver_evaluations,
        "active_solver_parameters":result.metadata["active_solver_parameter_count"],
        "frozen_solver_parameters":result.metadata["frozen_solver_parameter_count"],
        "rank_policy_candidate_count":result.metadata.get(
            "rank_revealing_candidate_task_rank"
        ),
        "rank_policy_full_fallback_used":result.metadata.get(
            "rank_revealing_fallback_to_all"
        ),
        "joint_count":len(case.rig.joints),
        "control_count":len(case.rig.controls),
        "pin_count":len(case.pins),
        "drag_count":len(case.overrides),
        "max_pin_residual_world":result.max_pin_residual,
        "max_drag_residual_world":result.max_drag_residual,
        "max_segment_length_absolute_error_world":maximum_strain,
        "differential_derivative_error":result.metadata.get(
            "analytic_jacobian_finite_difference_max_error"),
        "no_model_or_graphics_claim":True,
    }


def run_court_v1(*,dois:tuple[int,...]=(24,48,72),max_nfev:int=90)->dict:
    cases=[]
    for dof in dois:
        case=make_morphology_case_v1(dof)
        measured=solve_and_court_v1(case,max_nfev=max_nfev,
                                    derivative_check=(dof==24))
        observed=_observability(case)
        # The source-free synthetic fixture *specifies* each drag as a
        # single joint-0 rotation. Its matching sparse actuator policy is a
        # known fixture oracle, not generic control discovery or truth.
        sparse_keys=tuple(
            (f"J:{index}:0",PLANES[index%3][0])
            for index in range(len(case.rig.controls))
            if index%2==1
        )
        reduced=solve_and_court_v1(
            case,max_nfev=max_nfev,active_parameter_keys=sparse_keys,
        )
        auto=solve_and_court_v1(
            case,max_nfev=max_nfev,
            active_parameter_policy="RANK_REVEALING_RESEARCH_V1",
        )
        if (measured["status"]!="PASS_MULTIGOAL_BEHAVIORAL_COURT" or
                reduced["status"]!="PASS_MULTIGOAL_BEHAVIORAL_COURT" or
                auto["status"]!="PASS_MULTIGOAL_BEHAVIORAL_COURT" or
                measured["max_segment_length_absolute_error_world"]>1e-8 or
                reduced["max_segment_length_absolute_error_world"]>1e-8 or
                not observed["all_dofs_individually_affect_an_endpoint"]):
            raise AssertionError("MECHANICS_MORPHOLOGY_FUNCTIONAL_COURT_FAIL")
        cases.append({
            "morphology":case.morphology,**observed,**measured,
            "generic_task_jacobian_policy":{
                "selection_evidence":"DECLARED_KINEMATIC_TASK_JACOBIAN_ONLY",
                "candidate_rank":auto["rank_policy_candidate_count"],
                "fallback_to_full":auto["rank_policy_full_fallback_used"],
                "actual_solved_parameters":auto["active_solver_parameters"],
                "runtime_ms":auto["runtime_ms"],
                "max_drag_residual_world":auto["max_drag_residual_world"],
                "max_pin_residual_world":auto["max_pin_residual_world"],
                "full_over_rank_policy_runtime_ratio_observed":(
                    measured["runtime_ms"]/auto["runtime_ms"]
                ),
                "policy_is_not_source_DOFS_or_physical_truth":True,
            },
            "explicit_synthetic_oracle_policy":{
                "scope":"KNOWN_FIXTURE_TARGET_GENERATION_ONLY_NOT_GENERIC",
                "selected_actuators":list(map(list,sparse_keys)),
                "active_dof_count":reduced["active_solver_parameters"],
                "frozen_dof_count":reduced["frozen_solver_parameters"],
                "solver_status":reduced["status"],
                "solver_runtime_ms":reduced["runtime_ms"],
                "max_drag_residual_world":reduced["max_drag_residual_world"],
                "max_pin_residual_world":reduced["max_pin_residual_world"],
                "max_segment_length_absolute_error_world":reduced[
                    "max_segment_length_absolute_error_world"
                ],
                "full_over_sparse_wall_ratio_observed":(
                    measured["runtime_ms"]/reduced["runtime_ms"]
                ),
                "unselected_dof_truth_is_not_invalidated":True,
            },
        })
    return {
        "schema":SCHEMA,"status":"PASS_MORPHOLOGY_RESEARCH_COURT",
        "cases":cases,
        "environment":{"python":platform.python_version(),
            "numpy":np.__version__,"scipy":scipy.__version__,
            "platform":platform.platform()},
        "authority":{
            "research_only":True,
            "fixture_is_synthetic":True,
            "source_to_canonical_binding_qualified":False,
            "independent_student_training_authorized":False,
            "real_source_physical_DOF_limits_proven":False,
            "full_product_runtime_FPS_certified":False,
            "contact_is_world_position_not_force_or_friction":True,
            "sparse_actuator_policy_is_oracle_for_known_synthetic_target":True,
            "policy_for_arbitrary_targets_qualified":False,
            "rank_policy_dynamic_fallback_can_restore_full_declared_DOFs":True,
        },
    }


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--max-nfev",type=int,default=90)
    args=parser.parse_args()
    result=run_court_v1(max_nfev=args.max_nfev)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,sort_keys=True))


if __name__=="__main__":
    main()
