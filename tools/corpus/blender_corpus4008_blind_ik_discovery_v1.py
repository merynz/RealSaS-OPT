#!/usr/bin/env python3
"""Native source rig IK hide-and-predict experiment on independent sources.

Critical anti-leak workflow:
  1. Verify byte SHA and load original .blend with --disable-autoexec.
  2. Build an ALLOWLISTED skeleton, source skin coverage and action-curve
     snapshot. Native constraints, controller targets and constraint chain
     lengths are never passed into derived candidate builder.
  3. Freeze blind candidates to SHA before inspecting native IK oracle.
  4. Evaluate top-K chain discovery separately. This is not a geometric IK
     solver and NOT functional IK/pose/deformation parity acceptance.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.corpus.blender_corpus4008_native_source_probe_v1 import sha, require
from tools.corpus.blender_corpus4008_control_profile_v1 import _action_curves
from tools.corpus.derive_corpus4008_rigprogram_prior_v1 import (
    discover, discover_articulation_frontiers, oracle_evaluate, sha_payload,
)

SCHEMA="RealSaS.Corpus4008BlindNativeIKDiscoverySourceOracleCourt.v1"
MAX_SOURCE_BONES=10_000
MAX_SOURCE_VERTICES=3_000_000
DATA_PATH_BONE=re.compile(r'^pose\.bones\["((?:\\.|[^"\\])*)"\]')


def source_motion_curve_count(bpy):
    counts=Counter()
    discovered=Counter()
    for action in bpy.data.actions:
        method,bags=_action_curves(action)
        discovered[method]+=1
        for _,_,_,curves in bags:
            for curve in curves:
                m=DATA_PATH_BONE.match(str(curve.data_path))
                if m:
                    # Data paths are source observation hints, not labels.
                    # Blender escaped names may contain unicode/backslashes.
                    name=m.group(1).replace('\\"','"').replace('\\\\','\\')
                    counts[name]+=1
    return counts,dict(discovered)


def skeleton_skin_snapshot(bpy):
    arms=sorted(
        (x for x in bpy.context.scene.objects if x.type=="ARMATURE"),
        key=lambda o:o.name)
    require(arms,"NO_SOURCE_ARMATURE")
    mesh_objects=sorted(
        (x for x in bpy.context.scene.objects if x.type=="MESH"),
        key=lambda o:o.name)
    require(mesh_objects,"NO_SOURCE_MESH")
    total_vertices=sum(len(x.data.vertices) for x in mesh_objects)
    require(total_vertices<=MAX_SOURCE_VERTICES,
            "SOURCE_MESH_VERTEX_BUDGET")
    total_bones=sum(len(x.pose.bones) for x in arms)
    require(0<total_bones<=MAX_SOURCE_BONES,
            "SOURCE_BONE_COUNT_BUDGET")
    mass=Counter()
    vertex_support=Counter()
    for ob in mesh_objects:
        owner_arms=set()
        if ob.parent is not None and ob.parent.type=="ARMATURE":
            owner_arms.add(ob.parent)
        for m in ob.modifiers:
            if m.type=="ARMATURE" and m.object is not None:
                owner_arms.add(m.object)
        # If multiple armatures genuinely bind a mesh, observation is
        # ambiguous. Do not double-count or choose one from source naming.
        if len(owner_arms)!=1:
            continue
        owner=next(iter(owner_arms))
        groups={g.index:g.name for g in ob.vertex_groups}
        arm_bones=set(owner.data.bones.keys())
        for v in ob.data.vertices:
            for g in v.groups:
                name=groups.get(g.group)
                if name is None or name not in arm_bones:continue
                weight=float(g.weight)
                require(math.isfinite(weight) and weight>=0,
                        "SOURCE_SKIN_WEIGHT_INVALID")
                if weight<=1e-8:continue
                jid=owner.name+"::"+name
                mass[jid]+=weight
                vertex_support[jid]+=1
    curves,discovery=source_motion_curve_count(bpy)
    bones=[]
    for arm in arms:
        for b in sorted(arm.data.bones,key=lambda x:x.name):
            parent=(arm.name+"::"+b.parent.name
                    if b.parent is not None else None)
            head=arm.matrix_world@b.head_local
            tail=arm.matrix_world@b.tail_local
            bones.append({
                "bone_id":arm.name+"::"+b.name,
                "parent_bone_id":parent,
                "head_source":[float(x) for x in head],
                "tail_source":[float(x) for x in tail],
                "use_deform":bool(b.use_deform),
                "skin_weight_mass":float(mass.get(arm.name+"::"+b.name,0)),
                "skin_influenced_vertices":int(vertex_support.get(
                    arm.name+"::"+b.name,0)),
                "source_motion_curve_count":int(curves.get(b.name,0)),
            })
    return bones,{
        "armature_count":len(arms),
        "mesh_object_count":len(mesh_objects),
        "source_imported_vertex_count":total_vertices,
        "source_bone_count":len(bones),
        "weighted_deforming_bone_count":sum(
            b["use_deform"] and b["skin_weight_mass"]>1e-8
            for b in bones),
        "motion_curve_discovery_apis":discovery,
        "source_ik_constraint_census_visible_to_discovery":False,
    }


def source_oracle_ik(bpy):
    rows=[]
    for arm in sorted((o for o in bpy.context.scene.objects
                       if o.type=="ARMATURE"),key=lambda x:x.name):
        for pb in sorted(arm.pose.bones,key=lambda b:b.name):
            for ci,c in enumerate(pb.constraints):
                if c.type!="IK":continue
                if c.mute or c.influence<=1e-6 or c.target is None:
                    continue
                native_chain=[]
                cur=pb
                # chain_count=0 means all ancestors on Blender IK rigs.
                budget=c.chain_count if c.chain_count>0 else 1000
                while cur is not None and len(native_chain)<budget:
                    native_chain.append(arm.name+"::"+cur.name)
                    cur=cur.parent
                rows.append({
                    "constraint_id":f"{arm.name}::{pb.name}::{ci}",
                    "owner_bone_id":f"{arm.name}::{pb.name}",
                    "source_native_chain_bone_ids":list(
                        reversed(native_chain)),
                    "native_chain_count_evaluation_only":c.chain_count,
                    "native_has_pole_target_evaluation_only":
                        c.pole_target is not None,
                    "native_control_solver_response_not_evaluated_here":True,
                })
    return rows


def study(bpy,source,expected_sha,transport_sha,commit,asset):
    require(source.is_file() and source.suffix.lower()==".blend",
            "PINNED_NATIVE_BLEND")
    require(sha(source)==expected_sha,"EXACT_ORIGINAL_NATIVE_SHA")
    require(not bpy.context.preferences.filepaths.use_scripts_auto_execute,
            "UNTRUSTED_SOURCE_SCRIPTS_DISABLED")
    bpy.ops.wm.open_mainfile(filepath=str(source),load_ui=False)
    require(not bpy.context.preferences.filepaths.use_scripts_auto_execute,
            "ORIGINAL_FILE_CANNOT_ENABLE_AUTOEXEC")
    bpy.context.view_layer.update()
    source_bones,scan=skeleton_skin_snapshot(bpy)
    # Freeze prediction first. The later call may examine original native
    # controller hints, but cannot update predictions or ranks.
    predicted=discover(source_bones,max_depth=5,max_candidates=192)
    frontier=discover_articulation_frontiers(
        source_bones,max_depth=5,max_candidates=512)
    frozen=predicted["blind_proposal_sha256"]
    frozen_frontier=frontier["blind_proposal_sha256"]
    # Freeze BOTH hypotheses before touching source-native IK oracle fields.
    oracle=source_oracle_ik(bpy)
    evaluation=oracle_evaluate(predicted,oracle,top_k=32)
    frontier32=oracle_evaluate(frontier,oracle,top_k=32)
    frontier128=oracle_evaluate(frontier,oracle,top_k=128)
    require(frontier["blind_proposal_sha256"]==frozen_frontier,
            "ORACLE_CONTAMINATED_FRONTIER")
    require(predicted["blind_proposal_sha256"]==frozen,
            "ORACLE_CONTAMINATED_PROPOSAL")
    return {
        "schema":SCHEMA,
        "status":"SOURCE_BLIND_IK_CHAIN_DISCOVERY_AND_SEPARATE_NATIVE_ORACLE_EVALUATION_ONLY",
        "source_sha256":expected_sha,
        "transport_sha256":transport_sha,
        "source_upstream_commit":commit,
        "asset":asset,
        "source_scripts_autoexec_disabled":True,
        "blender_version":bpy.app.version_string,
        "source_blind_observation":scan,
        "source_blind_observation_sha256":sha_payload(source_bones),
        "blind_candidate_program":predicted,
        "native_ik_oracle_evaluation_only":evaluation,
        "blind_frontier_candidate_program":frontier,
        "native_ik_oracle_frontier_top32_evaluation_only":frontier32,
        "native_ik_oracle_frontier_top128_evaluation_only":frontier128,
        "frontier_ablation_informed_by_spider_dragon_dev_rigs":True,
        "heldout_rig_did_not_inform_frontier_policy":
            asset not in ("rig-spider","rig-dragon","rig-snake"),
        "source_native_ik_functional_intervention_court":
            "SEPARATE_EXISTING_EXPERIMENT",
        "derived_ik_solver_behavior_tested":False,
        "derived_ik_skin_deformation_tested":False,
        "origin_ik_control_graph_recovered":False,
        "semantic_body_part_labels_produced":False,
        "automatic_AXISMIRA_training_admission":False,
        "materialization_authorized":False,
        "training_authorized":False,
    }


def main():
    import bpy
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--expected-sha",required=True)
    p.add_argument("--transport-sha",required=True)
    p.add_argument("--source-commit",required=True)
    p.add_argument("--asset",required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    result=study(bpy,a.source.resolve(),a.expected_sha,
                 a.transport_sha,a.source_commit,a.asset)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,
                                   sort_keys=True,allow_nan=False)+"\n")
    o=result["native_ik_oracle_evaluation_only"]
    f32=result["native_ik_oracle_frontier_top32_evaluation_only"]
    f128=result["native_ik_oracle_frontier_top128_evaluation_only"]
    print(json.dumps({
        "asset":a.asset,
        "native_source_ik_oracle_count":o["oracle_ik_constraint_count"],
        "blind_candidate_count":result["blind_candidate_program"][
            "discovered_candidate_count"],
        "top32_owner_effector_hits":o[
            "exact_native_owner_end_effector_recall"],
        "top32_half_chain_overlap_hits":o["chain_overlap_at_least_half"],
        "frontier_top32_owner_hits":f32["exact_native_owner_end_effector_recall"],
        "frontier_top32_half_chain_hits":f32["chain_overlap_at_least_half"],
        "frontier_top128_owner_hits":f128["exact_native_owner_end_effector_recall"],
        "frontier_top128_half_chain_hits":f128["chain_overlap_at_least_half"],
        "actual_solver_tested":False,
        "training_authorized":False,
    },sort_keys=True))


if __name__=="__main__":
    original_argv=sys.argv
    try:
        sys.argv=[original_argv[0]]+original_argv[
            original_argv.index("--")+1:]
        main()
    finally:
        sys.argv=original_argv
