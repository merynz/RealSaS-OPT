#!/usr/bin/env python3
"""Original-source IK *behavioral observation*, not functional source-truth PASS.

Pinned CC0 native .blend, --disable-autoexec. Perturb each valid IK target in
memory and measure source rig joint and evaluated mesh changes, then restore.
All outputs retain exact source identity. Imported authoring controllers may
require untrusted scripts that are deliberately disabled; missing response
does not prove source lacks native IK. Rig motion != human-reviewed anatomy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.corpus.blender_corpus4008_native_source_probe_v1 import (
    require,sha,
)

SCHEMA="RealSaS.Corpus4008NativeRigIKControlInterventionObservation.v1"


def max_motion(a,b):
    # Named address, not order: source bone and sampled skin vertex IDs must
    # match to compare; changed topology fails closed as numerical evidence.
    if set(a)!=set(b):
        return None
    return max(
        (math.dist(a[key],b[key]) for key in a),default=0.0)


def sense(bpy):
    from mathutils import Vector
    dg=bpy.context.evaluated_depsgraph_get()
    dg.update()
    bones={}
    mesh={}
    for ob in bpy.context.scene.objects:
        if ob.type=="ARMATURE":
            evaluated=ob.evaluated_get(dg)
            for pb in evaluated.pose.bones:
                xyz=evaluated.matrix_world @ pb.matrix.translation
                bones[f"{ob.name}::{pb.name}"]=tuple(map(float,xyz))
        elif ob.type=="MESH":
            eval_ob=ob.evaluated_get(dg)
            eval_mesh=eval_ob.to_mesh()
            try:
                # fixed strides span the full mesh deterministically; a
                # sampled absence of movement cannot certify no deformation.
                count=len(eval_mesh.vertices)
                stride=max(1,math.ceil(count/128))
                for i in range(0,count,stride):
                    xyz=eval_ob.matrix_world @ eval_mesh.vertices[i].co
                    mesh[f"{ob.name}::{i}"]=tuple(map(float,xyz))
            finally:
                eval_ob.to_mesh_clear()
    return bones,mesh


def bounds_span(bpy):
    from mathutils import Vector
    vertices=[]
    for ob in bpy.context.scene.objects:
        if ob.type=="MESH":
            vertices.extend(tuple(map(float,ob.matrix_world@Vector(p)))
                            for p in ob.bound_box)
    require(vertices,"SOURCE_MESH_BOUNDS")
    return max(max(p[i] for p in vertices)-min(p[i] for p in vertices)
               for i in range(3))


def perturb_pose_target(bpy, target, subtarget, displacement):
    from mathutils import Vector
    # No native/source file save. A pose bone's local controller location is
    # used when subtarget exists; world-axis equivalence is NOT implied.
    if subtarget:
        pb=target.pose.bones.get(subtarget) if target.type=="ARMATURE" else None
        if pb is None:
            return None,"IK_SUBTARGET_NOT_FOUND"
        old=pb.location.copy()
        pb.location=old+Vector(displacement)
        def undo():pb.location=old
        return undo,"POSE_BONE_LOCAL_LOCATION"
    old=target.location.copy()
    target.location=old+Vector(displacement)
    def undo():target.location=old
    return undo,"OBJECT_LOCAL_LOCATION"


def run(bpy,source:Path,expected_sha256:str,asset_key:str,
        upstream_commit:str,transport_sha256:str,
        max_ik:int=24):
    require(source.is_file() and source.suffix.lower()==".blend",
            "NATIVE_BLEND_SOURCE")
    require(sha(source)==expected_sha256,"EXACT_UNWRAPPED_SOURCE_SHA")
    require(not bpy.context.preferences.filepaths.use_scripts_auto_execute,
            "SOURCE_AUTOEXEC_DISABLED")
    require(1<=max_ik<=32,"MAX_CONSTRAINT_LIMIT")
    bpy.ops.wm.open_mainfile(filepath=str(source),load_ui=False)
    require(not bpy.context.preferences.filepaths.use_scripts_auto_execute,
            "SOURCE_AUTOEXEC_REENABLED")
    bpy.context.view_layer.update()
    span=bounds_span(bpy)
    require(span>1e-9,"INVALID_SOURCE_SCALE")
    candidate=[]
    total_ik=0
    for arm in sorted(
            (o for o in bpy.context.scene.objects if o.type=="ARMATURE"),
            key=lambda o:o.name):
        for pb in sorted(arm.pose.bones,key=lambda x:x.name):
            for ci,c in enumerate(pb.constraints):
                if c.type!="IK":continue
                total_ik+=1
                if c.target is None or c.mute or c.influence<=1e-6:
                    continue
                candidate.append((arm,pb,ci,c))
    candidate=candidate[:max_ik]
    first_bones,first_mesh=sense(bpy)
    require(first_bones and first_mesh,"NO_MESH_OR_BONES_IN_NATIVE_RIG")
    eps=span*1e-6
    delta=span*.025
    experiments=[]
    for arm,owner,ci,con in candidate:
        target=con.target
        subtarget=con.subtarget or ""
        # Changing a controller changes target itself. We separately record
        # non-target articulated bones and evaluated skinned mesh response.
        record={
            "armature_teacher_only":arm.name,
            "owner_pose_bone_teacher_only":owner.name,
            "constraint_index":ci,
            "constraint_type":con.type,
            "constraint_influence":float(con.influence),
            "target_object_teacher_only":target.name,
            "target_pose_bone_teacher_only":subtarget or None,
            "source_behavior_court_admitted":False,
            "interventions":[],
        }
        for axis in range(3):
            d=[0.0,0.0,0.0]
            d[axis]=delta
            # Local perturbation may not match rig's intended control space.
            undo,method=perturb_pose_target(bpy,target,subtarget,d)
            if undo is None:
                record["interventions"].append({
                    "axis":axis,"status":method,
                    "source_ik_functionality_courted":False,
                })
                continue
            try:
                bpy.context.view_layer.update()
                after_bones,after_mesh=sense(bpy)
                bone_delta=max_motion(first_bones,after_bones)
                mesh_delta=max_motion(first_mesh,after_mesh)
                # Exclude the target bone: directly moving the target is not
                # evidence the IK solver made other source joints respond.
                target_key=f"{target.name}::{subtarget}" if subtarget else None
                other_base={k:v for k,v in first_bones.items() if k!=target_key}
                other_after={k:v for k,v in after_bones.items() if k!=target_key}
                other_delta=max_motion(other_base,other_after)
            finally:
                undo()
                bpy.context.view_layer.update()
                restored_bones,restored_mesh=sense(bpy)
            rb=max_motion(first_bones,restored_bones)
            rm=max_motion(first_mesh,restored_mesh)
            require(rb is not None and rm is not None and
                    rb<=max(eps,.00001) and rm<=max(eps,.00001),
                    "INTERVENTION_DID_NOT_RESTORE_SOURCE_BASELINE")
            measured=(other_delta is not None and mesh_delta is not None)
            record["interventions"].append({
                "axis":axis,
                "target_displacement_source_local_units":delta,
                "perturbation_method":method,
                "rig_response_max_all_bones":bone_delta,
                "rig_response_max_excluding_direct_target":other_delta,
                "evaluated_mesh_response_max_sampled_vertices":mesh_delta,
                "restored_baseline_bone_max_residual":rb,
                "restored_baseline_mesh_max_residual":rm,
                "observed_indirect_joint_response":
                    bool(measured and other_delta>eps),
                "observed_deformation_response":
                    bool(measured and mesh_delta>eps),
                "sampled_vertex_count":len(first_mesh),
                "source_ik_functionality_courted":False,
                "native_source_control_program_complete":"UNKNOWN",
            })
        experiments.append(record)
    positive=sum(any(x.get("observed_indirect_joint_response")
                     and x.get("observed_deformation_response")
                     for x in r["interventions"])
                 for r in experiments)
    return {
        "schema":SCHEMA,
        "asset_key":asset_key,
        "source_sha256":expected_sha256,
        "source_transport_sha256":transport_sha256,
        "upstream_repo_commit":upstream_commit,
        "source_scripts_autoexec_disabled":True,
        "blender_version":bpy.app.version_string,
        "source_ik_constraints_total":total_ik,
        "source_ik_constraints_perturbable":len(experiments),
        "sampled_baseline_bones":len(first_bones),
        "sampled_baseline_evaluated_skin_vertices":len(first_mesh),
        "source_world_extent_imported":span,
        "source_unit":"IMPORTED_WORLD_UNIT_UNKNOWN",
        "intervention_scale_source_extent_fraction":.025,
        "source_ik_constraints_with_indirect_joint_and_mesh_motion_observed":positive,
        "rig_interventions":experiments,
        "source_ik_functionality_authority":"OBSERVATION_ONLY",
        "source_attachments_consequence_courted":False,
        "source_pose_rest_parity_courted":False,
        "rig_native_driver_scripting_parity_courted":False,
        "commercial_ml_use_rights_courted":False,
        "model_head_truth_admitted":False,
        "materialization_authorized":False,
        "training_authorized":False,
    }


def main():
    import bpy
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--expected-sha256",required=True)
    p.add_argument("--asset-key",required=True)
    p.add_argument("--source-transport-sha256",required=True)
    p.add_argument("--upstream-commit",required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    j=run(bpy,a.source.resolve(),a.expected_sha256,
          a.asset_key,a.upstream_commit,a.source_transport_sha256)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(j,sort_keys=True,indent=2,
                                   allow_nan=False)+"\n")
    print(json.dumps({
        "asset":a.asset_key,
        "original_ik_count":j["source_ik_constraints_total"],
        "tested_ik_count":j["source_ik_constraints_perturbable"],
        "observed_ik_constraints_affecting_bones_and_mesh":
            j["source_ik_constraints_with_indirect_joint_and_mesh_motion_observed"],
        "training_authorized":False,
    },sort_keys=True))


if __name__=="__main__":
    original=sys.argv
    try:
        sys.argv=[original[0]]+original[original.index("--")+1:]
        main()
    finally:
        sys.argv=original
