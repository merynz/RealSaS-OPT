#!/usr/bin/env python3
"""V13 hosted true native animation editing of pre-registered V11 heldout faces.

Only original SHA-locked 3-source pilot. 3 exact V9 authored poses, four
original cameras, 12 new source-face leaves per source. No owner/training labels.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np
from tools.corpus.audit_corpus370_v5_source_rest_courts_v1 import row_input
from tools.corpus.corpus370_partition_edit_plan_v7 import build_plan
from tools.corpus.court_corpus370_native_actuator_mechanics_v13 import evaluate
from tools.corpus.run_corpus370_native_animated_partition_private_v9 import (
    sha,read,ref,seal,require,pick_three_poses,V5_SHA,ROSTER_SHA,RAYS_SHA,JOIN_SHA,
    GROUP_SHA,NATIVE_SHA
)

SCHEMA="RealSaS.Corpus370NativeAnonymousActuatorPrivateBatch.v13"
SEAL_V9="RealSaS.Corpus370NativeAnimatedPartitionRealSourceResearchResult.v9"
SEAL_V10="RealSaS.Corpus370NativeEditResponseEquivalenceRealResult.v10"
SEAL_V11="RealSaS.Corpus370HeldoutGeometricSourceProbeRealPlanResult.v11"

def run(args):
    out=Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=True)
    result={"schema":SCHEMA,
            "status":"FAIL_V13_HELDOUT_NATIVE_SOURCE_RENDER_NOT_CLOSED",
            "worker":args.subject_ordinal,
            "graphics_owner_authorized":False,
            "training_authorized":False}
    started=time.monotonic()
    try:
        v9=read(args.public_v9,args.public_v9_sha256,"V13_V9_SEALED_PUBLIC_RESULT")
        v10=read(args.public_v10,args.public_v10_sha256,"V13_V10_SEALED_PUBLIC_RESULT")
        v11=read(args.public_v11,args.public_v11_sha256,"V13_V11_SEALED_PUBLIC_RESULT")
        v12=read(args.public_v12,args.public_v12_sha256,"V13_V12_REAL_HELDOUT_RENDER_SEAL_SHA")
        require(v12.get("schema")=="RealSaS.Corpus370HeldoutNativeEditRealSourceResult.v12"
                and v12["status"]=="PASS_THREE_REAL_SOURCE_V11_PRECOMMITTED_EDIT_RENDER_AND_INDEPENDENT_COURT__NO_DRAWABLE_OWNER"
                and v12["hosted"]["success_jobs"]==4
                and v12["authority"]["training_authorized"] is False,
                "V13_REAL_V12_SOURCE_MEASUREMENT_PARENT_NO_OWNER")
        require(v9.get("schema")==SEAL_V9
                and v10.get("schema")==SEAL_V10
                and v11.get("schema")==SEAL_V11
                and v9["authority"]["training_authorized"] is False
                and v10["interpretation"]["training_authorized"] is False
                and v11["limits"]["training_authorized"] is False
                and v9["hosted"]["jobs_succeeded"]==4
                and v10["github_actions_jobs_succeeded"]==4
                and v11["jobs_succeeded"]==4
                and v10["v9_public_seal_sha256"]==sha(args.public_v9)
                and v11["total_new_source_face_queries"]==36,
                "V13_REAL_PREVIOUS_SOURCE_UNALTERED_AUTHORITY")
        v5=read(args.v5_index,V5_SHA,"V13_EXACT_V5_SHA")
        roster=read(args.exact_roster,ROSTER_SHA,"V13_EXACT_SOURCE_ROSTER_SHA")
        rays=read(args.geometric_ray_index,RAYS_SHA,"V13_ORIGINAL_SOURCE_RAYS_SHA")
        join=read(Path(args.original_join_root)/
                  "CORPUS370_NATIVE_VISIBILITY_PAIR_JOIN_V1.json",
                  JOIN_SHA,"V13_ORIGINAL_SOURCE_PAIRS_SHA")
        groups=read(Path(args.allview_groups_root)/
                  "CORPUS370_ALLVIEW_GEOMETRY_GROUPS_PRIVATE_BATCH_V3.json",
                  GROUP_SHA,"V13_ORIGINAL_ANONYMOUS_GEOMETRY_BATCH_SHA")
        native=read(Path(args.native_teacher_root)/
                    "CORPUS370_V5_AUTHORED_TEMPORAL_PILOT_V1.json",
                    NATIVE_SHA,"V13_ORIGINAL_AUTHORED_MOTION_SHA")
        ids=sorted(join["subjects"])
        require(len(ids)==3 and args.subject_ordinal in range(3)
                and set(ids)==set(groups["subjects"])==set(native["selected_subject_ids"])
                and v5["prepared_subject_count"]==352
                and v5["failure_count"]==18
                and v5["training_authorized"] is False
                and native["training_authorized"] is False
                and groups["model_training_authorized"] is False,
                "V13_NONSEMANTIC_THREE_REAL_SOURCE_COHORT")
        sid=ids[args.subject_ordinal]
        label={0:"CAT",1:"TRICERATOPS",2:"COWFISH"}[args.subject_ordinal]
        source={x["subject_id"]:x for x in roster["subjects"]}[sid]
        observation={x["subject_id"]:x for x in v5["subjects"]}[sid]
        teacher_parent={x["subject_id"]:x for x in native["results"]}[sid]
        v9row=next(x for x in v9["per_subject"] if x["label"]==label)
        v10row=next(x for x in v10["per_subject"] if x["source_label"]==label)
        v11row=next(x for x in v11["per_subject"] if x["label"]==label)
        require(sid==v9row["subject_id"]==v10row["source_id"]==v11row["subject_id"]
                and source["source_sha256"]==observation["source_sha256"]==
                    teacher_parent["source_sha256"]==
                    groups["subjects"][sid]["source_sha256"]==
                    join["subjects"][sid]["source_sha256"]==
                    v9row["source_sha256"]==v10row["source_sha256"]==
                    v11row["source_sha256"]
                and v11row["new_heldout_target_count"]==12,
                "V13_REAL_SOURCE_V9_V10_V11_SUBJECT_SHA_ALIGNMENT")
        root=Path(args.extracted_source_root).resolve()
        asset=(root/source["source_relative_path"]).resolve()
        require(asset.is_relative_to(root) and asset.is_file()
                and sha(asset)==source["source_sha256"],
                "V13_RAW_REAL_ORIGINAL_SOURCE_BYTES")
        binding=row_input(observation)
        rayref=rays["subject_results"][sid]["receipt"]
        ray=read(rayref["path"],rayref["sha256"],"V13_REST_SOURCE_RAY_COURT")
        rest=ray["source_rest_geometry"]
        pairref=join["subjects"][sid]["pair_diagnostic_receipt"]
        pairs=read(pairref["path"],pairref["sha256"],
                   "V13_SOURCE_ORIGINAL_FACE_PAIR")
        require(rest["sha256"]==pairs["parents"]["rest_geometry_sha256"],
                "V13_NATIVE_REST_GEOMETRY_PAIR_PARENT")
        grouproot=Path(args.allview_groups_root)/"subjects"/sid
        proposalpath=grouproot/"ALLVIEW_ANONYMOUS_GROUP_CANDIDATES_V3.json"
        proposal=read(proposalpath,groups["subjects"][sid]["proposal_sha256"],
                      "V13_GROUNDED_ANONYMOUS_GROUP_PROPOSAL")
        courtpath=grouproot/"ALLVIEW_ANONYMOUS_EDIT_COURT_V3.json"
        oldcourt=read(courtpath,groups["subjects"][sid]["court_sha256"],
                      "V13_ORIGINAL_NONARTIST_GEOMETRIC_COURT")
        require(oldcourt["training_authorized"] is False
                and oldcourt["graphics_owner_family_authorized"] is False,
                "V13_NO_NAMED_SOURCE_OWNER")
        baseplan=build_plan(proposal,pairs)
        teacherdir=Path(args.native_teacher_root)/"subjects"/sid
        teacher_court_path=teacherdir/"TEMPORAL_TEACHER_COURT_V1.json"
        teacourt=read(teacher_court_path,sha(teacher_court_path),
                     "V13_ORIGINAL_NATIVE_TEACHER_COURT")
        require(teacourt["source_sha256"]==source["source_sha256"]
                and teacourt["teacher_arrays"]["sha256"]==
                    teacher_parent["teacher_arrays"]["sha256"]
                and teacourt["teacher_manifest"]["sha256"]==
                    teacher_parent["teacher_manifest"]["sha256"]
                and teacourt["training_authorized"] is False
                and teacourt["family_authority_minted"] is False,
                "V13_EXACT_NATIVE_SOURCE_POSE_ARRAY_LINEAGE")
        teachernpz=Path(teacourt["teacher_arrays"]["path"])
        teacherreceipt=Path(teacourt["teacher_manifest"]["path"])
        receipt=read(teacherreceipt,teacourt["teacher_manifest"]["sha256"],
                     "V13_ORIGINAL_NATIVE_TEACHER_SAMPLE_SHA")
        require(sha(teachernpz)==teacourt["teacher_arrays"]["sha256"],
                "V13_PRIVATE_AUTHORED_TEACHER_NPZ_SHA")
        require(sha(rest["path"])==rest["sha256"],
                "V13_ORIGINAL_REST_SOURCE_NPZ_SHA")
        with np.load(rest["path"],allow_pickle=False) as a,np.load(
                teachernpz,allow_pickle=False) as b:
            pose_indices=pick_three_poses(b,receipt,a["rest_vertices_source_world"])
        require(pose_indices==v9row["original_pose_indices_teacher_only"],
                "V13_DO_NOT_CHANGE_V9_TEST_POSES_AFTER_HELDOUT")
        v11path=Path(args.v11_private_root)/str(args.subject_ordinal)/(
            "PRIVATE_HELDOUT_BRANCH_PROBE_PLAN_V11.json")
        heldout=read(v11path,v11row["real_source_query_plan_private_sha256"],
                     "V13_ORIGINAL_V11_EXACT_HELDOUT_SOURCE_ADDRESSES_SHA")
        v10path=Path(args.v10_private_root)/str(args.subject_ordinal)/(
            "PRIVATE_NATIVE_EDIT_EQUIVALENCE_COURT_V10.json")
        prior=read(v10path,v10row["court_sha256"],
                   "V13_ORIGINAL_PRIVATE_V10_COURT_SHA")
        require(heldout["parents"]["original_source_sha256"]==source["source_sha256"]
                and heldout["parents"]["v9_public_seal_sha256"]==sha(args.public_v9)
                and heldout["parents"]["v10_public_seal_sha256"]==sha(args.public_v10)
                and heldout["source_rest_geometry_sha256"]==rest["sha256"]
                and heldout["parents"]["original_geometry_proposal_sha256"]==sha(proposalpath)
                and heldout["parents"]["original_source_hit_pair_sha256"]==sha(pairref["path"])
                and heldout["parents"]["original_native_v9_manifest_sha256"]==v9row["render_manifest_sha256"]
                and heldout["parents"]["independent_native_v10_court_sha256"]==sha(v10path)
                and prior["status"]==
                    "REAL_NATIVE_EDIT_RESPONSE_PAIRWISE_EQUIVALENCE_MEASURED__ARTIST_OWNER_UNKNOWN"
                and prior["source_sha256"]==source["source_sha256"]
                and prior["training_authorized"] is False,
                "V13_PRIVATE_V11_TARGETS_V10_SOURCE_PARENT_SEALED")
        v12row=next(x for x in v12["per_subject"] if x["label"]==label)
        require(v12row["subject_id"]==sid
                and v12row["source_sha256"]==source["source_sha256"]
                and v12row["v11_original_heldout_query_plan_sha256"]==sha(v11path)
                and v12row["qualified_source_vs_soup_pose_camera_cases"]==12
                and v12row["render_arms"] in (912,1200),
                "V13_ORIGINAL_V12_REAL_SOURCE_POSES_GROUP_CAA_PARENT")
        job={
            "schema":"RealSaS.Corpus370NativeAnonymousActuatorInterventionJob.v13",
            "subject_id":sid,"source":ref(asset),
            "rest_geometry":rest,"native_teacher":ref(teachernpz),
            "native_teacher_receipt":ref(teacherreceipt),
            "native_teacher_court":ref(teacher_court_path),
            "group_proposal":ref(proposalpath),
            "v11_private_plan":ref(v11path),
            "pose_indices":pose_indices,
            "native_actuation_degrees":5.0,"axes":["X","Y","Z"],
            "source_control_budget":128,
            "allow_source_rig_teacher_only":True,
            "teacher_only":True,
            "source_bone_name_as_label":False,
            "graphics_owner_authorized":False,"training_authorized":False
        }
        jobpath=out/"PRIVATE_ORIGINAL_SOURCE_ANON_ACTUATOR_JOB_V13.json"
        seal(jobpath,job)
        manifestpath=out/"PRIVATE_ORIGINAL_ANON_ACTUATOR_MANIFEST_V13.json"
        cmd=[str(Path(args.blender).resolve()),"-b","--factory-startup",
             "--python-exit-code","5","--python",
             "tools/truth_foundry/blender_corpus370_native_actuator_intervention_v13.py",
             "--","--job",str(jobpath),"--output",str(manifestpath)]
        proc=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                            text=True,check=False,timeout=14400)
        (out/"PRIVATE_ANON_ACTUATOR_BLENDER_V13_LOG.txt").write_text(proc.stdout[-380000:])
        require(proc.returncode==0,
                "V13_NATIVE_ORIGINAL_RIG_ACTUATOR_BLENDER_EXIT_"+str(proc.returncode))
        manifest=read(manifestpath,sha(manifestpath),
                      "V13_ORIGINAL_RIG_INTERVENTION_RESULT_SHA")
        require(manifest["original_selected_pose_indices_teacher_only"]==pose_indices
                and manifest["original_heldout_group_plan_sha256"]==sha(v11path)
                and manifest["native_original_pose_teacher_npz_sha256"]==sha(teachernpz)
                and manifest["source_sha256"]==source["source_sha256"],
                "V13_REAL_ORIGINAL_ANIMATED_INTERVENTION_MATCH_SOURCE_PARENTS")
        npz=Path(manifest["source_pose_articulation_array_path"])
        require(npz.is_file() and sha(npz)==manifest["source_pose_articulation_array_sha256"],
                "V13_ORIGINAL_ANON_NATIVE_ACTUATION_NPZ_SHA")
        with np.load(rest["path"],allow_pickle=False) as restarr, np.load(
            teachernpz,allow_pickle=False) as nativearr, np.load(
            npz,allow_pickle=False) as arr:
            require(set(arr.files)=={"baseline_vertices","native_control_perturbed_vertices"},
                    "V13_ACTUATOR_GEOMETRIC_RESPONSE_ARRAY_KEYS")
            court=evaluate(
                restarr["rest_vertices_source_world"],restarr["faces"],
                nativearr["posed_vertices_source_world"],
                arr["baseline_vertices"],arr["native_control_perturbed_vertices"],
                manifest,proposal,heldout,v12row)
        court["parents"]={
            "original_source_sha256":source["source_sha256"],
            "original_v12_public_sha256":sha(args.public_v12),
            "original_v11_private_plan_sha256":sha(v11path),
            "native_original_teacher_npz_sha256":sha(teachernpz),
            "native_anonymous_intervention_manifest_sha256":sha(manifestpath),
            "native_anonymous_intervention_vertex_array_sha256":sha(npz)
        }
        independent=out/"INDEPENDENT_ANONYMOUS_NATIVE_ACTUATOR_COURT_V13.json"
        seal(independent,court)
        require(court["training_authorized"] is False
                and court["graphics_owner_authorized"] is False
                and court["positive_same_drawable_labels"]==0
                and court["positive_different_drawable_labels"]==0,
                "V13_SOURCE_NATIVE_INTERVENTION_NOT_ARTIST_DRAWABLE")
        result.update({
            "status":"MEASURED_REAL_ORIGINAL_SOURCE_RIG_ANONYMOUS_ACTUATION_V13_NO_OWNER",
            "subject_id":sid,"subject_label":label,
            "source_sha256":source["source_sha256"],
            "original_native_author_pose_indices":pose_indices,
            "original_native_rig_controls_total":manifest["source_control_inventory_count"],
            "original_native_rig_controls_measured":manifest["source_controls_measured_count"],
            "original_native_rig_controls_truncated":manifest["source_control_inventory_truncated"],
            "source_native_pose_control_axis_cases":court["source_native_anonymous_mechanics_counts"]["tested_pose_control_axis_combinations"],
            "effective_native_source_control_responses":court["source_native_anonymous_mechanics_counts"]["source_effect_nonzero_count"],
            "local_nonrigid_source_control_responses":court["source_native_anonymous_mechanics_counts"]["local_nonrigid_effect_count"],
            "root_like_or_globally_rigid_responses":court["source_native_anonymous_mechanics_counts"]["root_like_or_globally_rigid_effect_count"],
            "source_control_reversibility_masked_count":court["source_native_anonymous_mechanics_counts"]["nonreversible_effect_masked_count"],
            "independent_native_mechanics_status":court["status"],
            "candidate_mixed_group_events":court["source_native_anonymous_mechanics_counts"]["candidate_mixed_group_events"],
            "native_intervention_manifest_sha256":sha(manifestpath),
            "native_vertex_response_npz_sha256":sha(npz),
            "independent_mechanics_court_sha256":sha(independent),
            "original_v12_public_result_sha256":sha(args.public_v12),
            "source_mesh_bone_action_names_as_student_labels":False,
            "artist_drawable_positive_or_negative_labels":0,
            "graphics_owner_authorized":False,
            "training_authorized":False
        })
    except (KeyboardInterrupt,MemoryError):
        raise
    except Exception as exc:
        result["status"]="FAIL_REAL_V13_SOURCE_RENDER_OR_COURT"
        result["failure_type"]=type(exc).__name__
        result["failure_reason"]=str(exc)
        seal(out/"PRIVATE_V13_SOURCE_FAILURE_RECEIPT.json",{
            "reason":str(exc),"type":type(exc).__name__,
            "traceback":traceback.format_exc(),
            "training_authorized":False,"graphics_owner_authorized":False
        })
    result["elapsed_seconds"]=round(time.monotonic()-started,3)
    seal(out/"PRIVATE_HELDOUT_NATIVE_EDIT_BATCH_V13.json",result)
    print(json.dumps({k:v for k,v in result.items()
                     if k!="failure_reason"},sort_keys=True),flush=True)
    require(result["status"]=="MEASURED_REAL_ORIGINAL_SOURCE_RIG_ANONYMOUS_ACTUATION_V13_NO_OWNER",
            "V13_HELDOUT_ANIMATED_SOURCE_RENDER_REQUIRES_REPAIR")
    return result

def main():
    p=argparse.ArgumentParser()
    for k in ("v5_index","exact_roster","geometric_ray_index",
              "extracted_source_root","original_join_root",
              "allview_groups_root","native_teacher_root",
              "v11_private_root","v10_private_root",
              "public_v9","public_v10","public_v11","public_v12","blender","output"):
        p.add_argument("--"+k.replace("_","-"),type=Path,required=True)
    for k in ("public_v9_sha256","public_v10_sha256","public_v11_sha256","public_v12_sha256"):
        p.add_argument("--"+k.replace("_","-"),required=True)
    p.add_argument("--subject-ordinal",type=int,required=True)
    run(p.parse_args())

if __name__=="__main__":
    main()
