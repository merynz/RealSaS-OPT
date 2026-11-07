#!/usr/bin/env python3
"""Source-only generic CCD target behavior under BLINDED native controls.

The auto-discovered AXIS IK candidate chain is moved in memory to novel
world-space relative targets with no native IK constraints/targets as input.
Measures target error + other-bone motion + evaluated skin displacement +
pose restoration; does NOT validate joint limits, animator usability, foot
contact, semantic end-effectors or native original controller identity.

This is a RIGPROGRAM provider research assay, NOT a Compiler fallback.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.corpus.blender_corpus4008_native_source_probe_v1 import require,sha
from tools.corpus.blender_corpus4008_blind_ik_discovery_v1 import (
    skeleton_skin_snapshot,
)
from tools.corpus.derive_corpus4008_rigprogram_prior_v1 import (
    discover,
)
from tools.corpus.blender_corpus4008_native_ik_consequence_v1 import (
    sense,max_motion,bounds_span,
)

SCHEMA="RealSaS.Corpus4008DerivedSourceOnlyIKTargetBehaviorAssay.v1"


def mute_source_constraints(bpy):
    # Mask every source control/constraint, WITHOUT reading its kind, target,
    # owner, native chain length, pole or output semantics. No oracle labels
    # can select the solver chain; it was already frozen by source-only code.
    changes=[]
    for ob in bpy.context.scene.objects:
        for c in ob.constraints:
            changes.append((c,c.mute))
            c.mute=True
        if ob.type=="ARMATURE":
            for pb in ob.pose.bones:
                for c in pb.constraints:
                    changes.append((c,c.mute))
                    c.mute=True
    bpy.context.view_layer.update()
    return changes


def restore_source_constraints(bpy,changes):
    for c,old in changes:c.mute=old
    bpy.context.view_layer.update()


def pose_snapshot(bpy):
    return {
        (arm.name,pb.name):(pb.matrix_basis.copy(),pb.rotation_mode)
        for arm in bpy.context.scene.objects if arm.type=="ARMATURE"
        for pb in arm.pose.bones
    }


def restore_pose(bpy,original):
    for ob in bpy.context.scene.objects:
        if ob.type!="ARMATURE":continue
        for pb in ob.pose.bones:
            key=(ob.name,pb.name)
            if key not in original:continue
            basis,mode=original[key]
            pb.rotation_mode=mode
            pb.matrix_basis=basis.copy()
    bpy.context.view_layer.update()


def solve_source_ik_chain(bpy,arm,chain,goal_world,*,iterations=16,
                          strength=.65):
    """Minimal generic CCD: no source constraint, pole or semantic labels.

    posebone.matrix is armature-object-space. A source-world rotation is
    conjugated into this space; rest and bone hierarchy remain unaltered.
    Blender recomputes evaluated skin with no native constraints active.
    """
    from mathutils import Matrix,Quaternion,Vector
    require(2<=len(chain)<=6,"SOLVER_CHAIN_LENGTH")
    require(1<=iterations<=64 and 0<strength<=1,
            "SOLVER_ITERATIONS")
    goal=Vector(goal_world)
    tip=chain[-1]
    start=(arm.matrix_world @ tip.tail).copy()
    prev=float((start-goal).length)
    for iteration in range(iterations):
        if prev<=1e-6:break
        for pb in reversed(chain):
            pivot=arm.matrix_world @ pb.head
            current=arm.matrix_world @ tip.tail
            src=current-pivot
            desired=goal-pivot
            if src.length<=1e-8 or desired.length<=1e-8:continue
            change=src.rotation_difference(desired)
            partial=Quaternion((1.,0.,0.,0.)).slerp(change,strength)
            q_arm=(arm.matrix_world.to_quaternion().inverted()
                   @partial@arm.matrix_world.to_quaternion())
            center=pb.head.copy()
            around=(Matrix.Translation(center)
                    @ q_arm.to_matrix().to_4x4()
                    @ Matrix.Translation(-center))
            pb.matrix=around@pb.matrix
            bpy.context.view_layer.update()
        endpoint=arm.matrix_world@tip.tail
        err=float((endpoint-goal).length)
        if err<=1e-6 or abs(prev-err)<=1e-8:break
        prev=err
    end=arm.matrix_world@tip.tail
    return {
        "end_effector_source_world_before":tuple(float(v) for v in start),
        "end_effector_source_world_after":tuple(float(v) for v in end),
        "target_source_world":tuple(float(v) for v in goal),
        "target_error_world":float((end-goal).length),
        "initial_target_error_world":float((start-goal).length),
        "iterations_run":iteration+1,
    }


def run(bpy,source,expected_sha,asset,*,max_chains=5):
    require(sha(source)==expected_sha,"SOURCE_SHA")
    require(not bpy.context.preferences.filepaths.use_scripts_auto_execute,
            "SOURCE_SCRIPTS_DISABLED")
    bpy.ops.wm.open_mainfile(filepath=str(source),load_ui=False)
    require(not bpy.context.preferences.filepaths.use_scripts_auto_execute,
            "SOURCE_AUTOEXEC_REENABLED")
    snapshot,metadata=skeleton_skin_snapshot(bpy)
    prediction=discover(snapshot,max_depth=5,max_candidates=64)
    source_only_sha=prediction["blind_proposal_sha256"]
    extent=bounds_span(bpy)
    require(extent>1e-8,"SOURCE_WORLD_EXTENT")
    original=pose_snapshot(bpy)
    muted=mute_source_constraints(bpy)
    try:
        bones0,mesh0=sense(bpy)
        chosen=[]
        used_ends=set()
        for p in prediction["proposals"]:
            if p["end_bone_id"] in used_ends:continue
            used_ends.add(p["end_bone_id"])
            chosen.append(p)
            if len(chosen)>=max_chains:break
        trials=[]
        delta=0.03*extent
        for idx,proposal in enumerate(chosen):
            ids=proposal["chain_bone_ids"]
            prefix,_,_=ids[0].partition("::")
            # Opaque source IDs are structural references only; do not parse
            # body-part names or original constraint target types.
            arm=bpy.data.objects.get(prefix)
            if arm is None or arm.type!="ARMATURE":continue
            if any(x.partition("::")[0]!=prefix for x in ids):
                continue
            chain=[arm.pose.bones.get(x.partition("::")[2]) for x in ids]
            if any(x is None for x in chain):continue
            for axis in range(3):
                restore_pose(bpy,original)
                b0,m0=sense(bpy)
                target=(arm.matrix_world@chain[-1].tail).copy()
                target[axis]+=delta
                outcome=None
                try:
                    outcome=solve_source_ik_chain(
                        bpy,arm,chain,target,iterations=18,strength=.65)
                    b1,m1=sense(bpy)
                    affected_other={
                        k:v for k,v in b0.items()
                        if k!=ids[-1]}
                    changed_other={
                        k:v for k,v in b1.items()
                        if k!=ids[-1]}
                    moving=max_motion(affected_other,changed_other)
                    deform=max_motion(m0,m1)
                    outcome.update({
                        "sampled_mesh_vertex_motion_max_world":deform,
                        "non_effector_joint_motion_max_world":moving,
                        "source_training_admission":False,
                        "source_native_constraint_used":False,
                        "source_rig_pole_limit_or_contact_tested":False,
                    })
                    outcome["fraction_target_error_reduction"]=(
                        1.-outcome["target_error_world"]/
                        max(outcome["initial_target_error_world"],1e-12))
                    outcome["candidate_chain_sha256"]=source_only_sha
                except Exception as exc:
                    outcome={
                        "status":"SOURCE_CCD_SOLVER_EXCEPTION",
                        "exception_type":type(exc).__name__,
                        "target_error_world":None,
                        "source_training_admission":False,
                    }
                finally:
                    restore_pose(bpy,original)
                    restored_bones,restored_mesh=sense(bpy)
                rb=max_motion(bones0,restored_bones)
                rm=max_motion(mesh0,restored_mesh)
                require(rb is not None and rm is not None,
                        "SOURCE_RIG_MESH_ADDRESS_DRIFT")
                outcome.update({
                    "source_restore_joint_residual_world":rb,
                    "source_restore_sampled_mesh_residual_world":rm,
                    "restoration_within_1e_4_extent":
                        rb<=max(1e-6,1e-4*extent) and
                        rm<=max(1e-6,1e-4*extent),
                    "candidate_index_blind_only":idx,
                    "target_axis_source_world":axis,
                    "target_delta_world":delta,
                    "native_ik_oracle_not_consulted":True,
                })
                trials.append(outcome)
    finally:
        restore_pose(bpy,original)
        restore_source_constraints(bpy,muted)
    require(prediction["blind_proposal_sha256"]==source_only_sha,
            "SOURCE_PREDICTION_MUTATED")
    measurable=[x for x in trials
                if isinstance(x.get("target_error_world"),float)]
    return {
        "schema":SCHEMA,
        "asset":asset,
        "original_source_sha256":expected_sha,
        "source_blind_prediction_sha256":source_only_sha,
        "source_control_mask":"ALL_SOURCE_CONSTRAINTS_MUTED",
        "source_constraint_graph_consumed_by_solver":False,
        "source_motion_prior":"ACTION_FCURVE_COUNTS_IF_AVAILABLE",
        "source_skin_prior":"EXACT_IMPORTED_SPARSE_WEIGHT_SUPPORT",
        "sampled_source_bone_count":metadata["source_bone_count"],
        "blind_end_effector_candidate_count":prediction[
            "source_deform_weighted_distal_count"],
        "tested_source_only_candidate_chains":len(chosen),
        "source_only_target_trials":len(trials),
        "source_only_target_trials_measurable":len(measurable),
        "source_only_targets_error_reduced":sum(
            t.get("fraction_target_error_reduction",0)>0.5
            for t in measurable),
        "source_only_targets_with_mesh_response":sum(
            (t.get("sampled_mesh_vertex_motion_max_world") or 0)>1e-6*extent
            for t in measurable),
        "all_trials":trials,
        "original_native_IK_constraint_grammar":"UNKNOWN_TO_SOLVER",
        "functional_joint_limits_courted":False,
        "source_original_ik_fidelity_courted":False,
        "semantic_effector_identity_courted":False,
        "commercial_ml_rights_courted":False,
        "model_training_admitted":False,
        "materialization_authorized":False,
        "training_authorized":False,
    }


def main():
    import bpy
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--expected-sha",required=True)
    p.add_argument("--asset",required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    r=run(bpy,a.source.resolve(),a.expected_sha,a.asset)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(r,indent=2,sort_keys=True,
                                   allow_nan=False)+"\n")
    print(json.dumps({
        "asset":a.asset,
        "blind_candidate_count":r["blind_end_effector_candidate_count"],
        "tested_chains":r["tested_source_only_candidate_chains"],
        "target_trials":r["source_only_target_trials"],
        "error_reduced_trials":r["source_only_targets_error_reduced"],
        "mesh_response_trials":r["source_only_targets_with_mesh_response"],
        "training_authorized":False,
    },sort_keys=True))


if __name__=="__main__":
    argv=sys.argv
    try:
        sys.argv=[argv[0]]+argv[argv.index("--")+1:]
        main()
    finally:
        sys.argv=argv
