#!/usr/bin/env python3
"""V14 hosted true original-source native mechanical control and RGBA joint experiment.

Only exact SHA-locked 3-source pilot. Same ±5deg source control 3D response
and 2D UV-preserving original rendering. No artist owner/training labels.
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
from tools.corpus.court_corpus370_mechanical_raster_join_v14 import evaluate
from tools.corpus.run_corpus370_native_animated_partition_private_v9 import (
    sha,read,ref,seal,require,pick_three_poses,V5_SHA,ROSTER_SHA,RAYS_SHA,JOIN_SHA,
    GROUP_SHA,NATIVE_SHA
)

SCHEMA="RealSaS.Corpus370MechanicalRasterJointPrivateBatch.v14"
SEAL_V9="RealSaS.Corpus370NativeAnimatedPartitionRealSourceResearchResult.v9"
SEAL_V10="RealSaS.Corpus370NativeEditResponseEquivalenceRealResult.v10"
SEAL_V11="RealSaS.Corpus370HeldoutGeometricSourceProbeRealPlanResult.v11"

def run(args):
    out=Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=True)
    result={"schema":SCHEMA,
            "status":"FAIL_V14_HELDOUT_NATIVE_SOURCE_RENDER_NOT_CLOSED",
            "worker":args.subject_ordinal,
            "graphics_owner_authorized":False,
            "training_authorized":False}
    started=time.monotonic()
    try:
        v9=read(args.public_v9,args.public_v9_sha256,"V14_V9_SEALED_PUBLIC_RESULT")
        v10=read(args.public_v10,args.public_v10_sha256,"V14_V10_SEALED_PUBLIC_RESULT")
        v11=read(args.public_v11,args.public_v11_sha256,"V14_V11_SEALED_PUBLIC_RESULT")
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
                "V14_REAL_PREVIOUS_SOURCE_UNALTERED_AUTHORITY")
        public_v12=read(args.public_v12,args.public_v12_sha256,"V14_SOURCE_V12_RGB_ALPHA_PROVENANCE_SHA")
        public_v13b=read(args.public_v13b,args.public_v13b_sha256,"V14_SOURCE_NATIVE_V13B_MECHANICS_SHA")
        require(public_v12["schema"]=="RealSaS.Corpus370HeldoutNativeEditRealSourceResult.v12"
                and public_v12["hosted"]["success_jobs"]==4
                and public_v12["authority"]["training_authorized"] is False
                and public_v13b["schema"]=="RealSaS.Corpus370NativeMixedCarrierRealResult.v13b"
                and public_v13b["hosted"]["success_jobs"]==4
                and public_v13b["authority"]["training_authorized"] is False,
                "V14_SOURCE_REAL_V12_RASTER_AND_V13B_MECHANICAL_NOT_MODEL_TRUTH")
        v5=read(args.v5_index,V5_SHA,"V14_EXACT_V5_SHA")
        roster=read(args.exact_roster,ROSTER_SHA,"V14_EXACT_SOURCE_ROSTER_SHA")
        rays=read(args.geometric_ray_index,RAYS_SHA,"V14_ORIGINAL_SOURCE_RAYS_SHA")
        join=read(Path(args.original_join_root)/
                  "CORPUS370_NATIVE_VISIBILITY_PAIR_JOIN_V1.json",
                  JOIN_SHA,"V14_ORIGINAL_SOURCE_PAIRS_SHA")
        groups=read(Path(args.allview_groups_root)/
                  "CORPUS370_ALLVIEW_GEOMETRY_GROUPS_PRIVATE_BATCH_V3.json",
                  GROUP_SHA,"V14_ORIGINAL_ANONYMOUS_GEOMETRY_BATCH_SHA")
        native=read(Path(args.native_teacher_root)/
                    "CORPUS370_V5_AUTHORED_TEMPORAL_PILOT_V1.json",
                    NATIVE_SHA,"V14_ORIGINAL_AUTHORED_MOTION_SHA")
        ids=sorted(join["subjects"])
        require(len(ids)==3 and args.subject_ordinal in range(3)
                and set(ids)==set(groups["subjects"])==set(native["selected_subject_ids"])
                and v5["prepared_subject_count"]==352
                and v5["failure_count"]==18
                and v5["training_authorized"] is False
                and native["training_authorized"] is False
                and groups["model_training_authorized"] is False,
                "V14_NONSEMANTIC_THREE_REAL_SOURCE_COHORT")
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
                "V14_REAL_SOURCE_V9_V10_V11_SUBJECT_SHA_ALIGNMENT")
        root=Path(args.extracted_source_root).resolve()
        asset=(root/source["source_relative_path"]).resolve()
        require(asset.is_relative_to(root) and asset.is_file()
                and sha(asset)==source["source_sha256"],
                "V14_RAW_REAL_ORIGINAL_SOURCE_BYTES")
        binding=row_input(observation)
        rayref=rays["subject_results"][sid]["receipt"]
        ray=read(rayref["path"],rayref["sha256"],"V14_REST_SOURCE_RAY_COURT")
        rest=ray["source_rest_geometry"]
        pairref=join["subjects"][sid]["pair_diagnostic_receipt"]
        pairs=read(pairref["path"],pairref["sha256"],
                   "V14_SOURCE_ORIGINAL_FACE_PAIR")
        require(rest["sha256"]==pairs["parents"]["rest_geometry_sha256"],
                "V14_NATIVE_REST_GEOMETRY_PAIR_PARENT")
        grouproot=Path(args.allview_groups_root)/"subjects"/sid
        proposalpath=grouproot/"ALLVIEW_ANONYMOUS_GROUP_CANDIDATES_V3.json"
        proposal=read(proposalpath,groups["subjects"][sid]["proposal_sha256"],
                      "V14_GROUNDED_ANONYMOUS_GROUP_PROPOSAL")
        courtpath=grouproot/"ALLVIEW_ANONYMOUS_EDIT_COURT_V3.json"
        oldcourt=read(courtpath,groups["subjects"][sid]["court_sha256"],
                      "V14_ORIGINAL_NONARTIST_GEOMETRIC_COURT")
        require(oldcourt["training_authorized"] is False
                and oldcourt["graphics_owner_family_authorized"] is False,
                "V14_NO_NAMED_SOURCE_OWNER")
        baseplan=build_plan(proposal,pairs)
        teacherdir=Path(args.native_teacher_root)/"subjects"/sid
        teacher_court_path=teacherdir/"TEMPORAL_TEACHER_COURT_V1.json"
        teacourt=read(teacher_court_path,sha(teacher_court_path),
                     "V14_ORIGINAL_NATIVE_TEACHER_COURT")
        require(teacourt["source_sha256"]==source["source_sha256"]
                and teacourt["teacher_arrays"]["sha256"]==
                    teacher_parent["teacher_arrays"]["sha256"]
                and teacourt["teacher_manifest"]["sha256"]==
                    teacher_parent["teacher_manifest"]["sha256"]
                and teacourt["training_authorized"] is False
                and teacourt["family_authority_minted"] is False,
                "V14_EXACT_NATIVE_SOURCE_POSE_ARRAY_LINEAGE")
        teachernpz=Path(teacourt["teacher_arrays"]["path"])
        teacherreceipt=Path(teacourt["teacher_manifest"]["path"])
        receipt=read(teacherreceipt,teacourt["teacher_manifest"]["sha256"],
                     "V14_ORIGINAL_NATIVE_TEACHER_SAMPLE_SHA")
        require(sha(teachernpz)==teacourt["teacher_arrays"]["sha256"],
                "V14_PRIVATE_AUTHORED_TEACHER_NPZ_SHA")
        require(sha(rest["path"])==rest["sha256"],
                "V14_ORIGINAL_REST_SOURCE_NPZ_SHA")
        with np.load(rest["path"],allow_pickle=False) as a,np.load(
                teachernpz,allow_pickle=False) as b:
            pose_indices=pick_three_poses(b,receipt,a["rest_vertices_source_world"])
        require(pose_indices==v9row["original_pose_indices_teacher_only"],
                "V14_DO_NOT_CHANGE_V9_TEST_POSES_AFTER_HELDOUT")
        v11path=Path(args.v11_private_root)/str(args.subject_ordinal)/(
            "PRIVATE_HELDOUT_BRANCH_PROBE_PLAN_V11.json")
        heldout=read(v11path,v11row["real_source_query_plan_private_sha256"],
                     "V14_ORIGINAL_V11_EXACT_HELDOUT_SOURCE_ADDRESSES_SHA")
        v10path=Path(args.v10_private_root)/str(args.subject_ordinal)/(
            "PRIVATE_NATIVE_EDIT_EQUIVALENCE_COURT_V10.json")
        prior=read(v10path,v10row["court_sha256"],
                   "V14_ORIGINAL_PRIVATE_V10_COURT_SHA")
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
                "V14_PRIVATE_V11_TARGETS_V10_SOURCE_PARENT_SEALED")
        original_v13brow=next(x for x in public_v13b["per_subject"] if x["label"]==label)
        require(original_v13brow["subject_id"]==sid
                and original_v13brow["original_source_sha256"]==source["source_sha256"]
                and original_v13brow["source_controls_total"]>=1
                and original_v13brow["local_nonrigid_cases"]>0,
                "V14_SOURCE_NATIVE_MECHANICAL_V13B_ORIGINAL_REAL_NOT_SYNTHETIC")
        root13=Path(args.v13b_private_root)/str(args.subject_ordinal)
        private_batch_path=root13/"PRIVATE_HELDOUT_NATIVE_EDIT_BATCH_V13B.json"
        private_batch=read(private_batch_path,sha(private_batch_path),
                           "V14_PREVIOUS_REAL_SOURCE_PRIVATE_BATCH")
        manifest13path=root13/"PRIVATE_ORIGINAL_ANON_ACTUATOR_MANIFEST_V13B.json"
        manifest13=read(manifest13path,private_batch["native_intervention_manifest_sha256"],
                        "V14_TRUE_ORIGINAL_SOURCE_RIG_OR_OBJECT_CONTROL_RECEIPT")
        array13path=root13/"PRIVATE_ORIGINAL_ANON_ACTUATOR_MANIFEST_V13B.npz"
        require(sha(array13path)==original_v13brow["raw_vertex_response_npz_sha256"]
                and manifest13["source_pose_articulation_array_sha256"]==
                    original_v13brow["raw_vertex_response_npz_sha256"]
                and private_batch["native_vertex_response_npz_sha256"]==
                    original_v13brow["raw_vertex_response_npz_sha256"]
                and private_batch["independent_mechanics_court_sha256"]==
                    original_v13brow["independent_court_sha256"]
                and private_batch["subject_id"]==sid
                and private_batch["training_authorized"] is False
                and private_batch["graphics_owner_authorized"] is False
                and manifest13["source_sha256"]==source["source_sha256"]
                and manifest13["original_selected_pose_indices_teacher_only"]==pose_indices
                and manifest13["native_control_carrier_types_teacher_only"]==
                   {"POSE_BONE":original_v13brow["source_pose_bone_controls"],
                    "IMPORTED_ANIMATED_OBJECT":original_v13brow["source_imported_object_node_controls"]},
                "V14_V13B_SOURCE_NATIVE_CARRIER_MANIFEST_NPZ_BATCH_SHA")
        from tools.corpus.corpus370_native_raster_join_plan_v14 import (
            select,arms_for_plan
        )
        with np.load(rest["path"],allow_pickle=False) as r,np.load(
            array13path,allow_pickle=False) as a:
            control_plan=select(
                r["rest_vertices_source_world"],
                a["baseline_vertices"],
                a["native_control_perturbed_vertices"],
                pose_indices,
                original_v13brow["source_controls_total"],
                pose_indices
            )
        require(control_plan["training_authorized"] is False
                and control_plan["graphics_owner_authorized"] is False
                and control_plan["raster_pixels_seen_during_control_selection"] is False,
                "V14_RASTER_BLIND_NATIVE_SOURCE_CONTROLS_NO_ARTIST_OWNER")
        selected_path=out/"PRIVATE_GEOMETRY_ONLY_NATIVE_CONTROL_SELECTION_V14.json"
        seal(selected_path,control_plan)
        arms,_=arms_for_plan(control_plan)
        job={
            "schema":"RealSaS.Corpus370SourceNativeMechanicalRasterJointJob.v14",
            "subject_id":sid,
            "source":ref(asset),
            "admission":binding["admission"],
            "candidate":binding["candidate"],
            "camera":binding["camera"],
            "rest_geometry":rest,
            "native_teacher":ref(teachernpz),
            "native_teacher_receipt":ref(teacherreceipt),
            "native_teacher_court":ref(teacher_court_path),
            "v11_private_plan":ref(v11path),
            "v13b_manifest":ref(manifest13path),
            "v13b_array":ref(array13path),
            "selected_source_controls":ref(selected_path),
            "pose_indices":pose_indices,
            "view_indices":[0,2,4,6],
            "raster_resolution":192,
            "cycles_samples":32,
            "native_actuation_degrees":5.0,
            "root_sham_source_world_fraction":.025,
            "teacher_only":True,
            "source_bone_names_as_targets":False,
            "artist_drawable_identity_inferred":False,
            "graphics_owner_authorized":False,
            "training_authorized":False
        }
        jobpath=out/"PRIVATE_NATIVE_MECHANICAL_RGBA_JOIN_JOB_V14.json"
        seal(jobpath,job)
        renderpath=out/"PRIVATE_REAL_NATIVE_MECHANICAL_RGBA_MANIFEST_V14.json"
        cmd=[str(Path(args.blender).resolve()),"-b","--factory-startup",
             "--python-exit-code","5","--python",
             "tools/truth_foundry/blender_corpus370_native_mechanical_raster_join_v14.py",
             "--","--job",str(jobpath),"--output",str(renderpath)]
        proc=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                            text=True,check=False,timeout=14400)
        (out/"PRIVATE_NATIVE_RGBA_BLENDER_LOG_V14.txt").write_text(proc.stdout[-380000:])
        require(proc.returncode==0,
                "V14_SOURCE_NATIVE_RGBA_BLENDER_EXIT_"+str(proc.returncode))
        manifest=read(renderpath,sha(renderpath),
                      "V14_REAL_NATIVE_SOURCE_MECHANICAL_RGBA_MANIFEST_SHA")
        require(manifest["source_sha256"]==source["source_sha256"],
                "V14_SOURCE_MANIFEST_EXACT_ORIGINAL_SHA")
        require(manifest["private_job_sha256"]==sha(jobpath)
                and manifest["source_private_v13b_intervention_manifest_sha256"]==sha(manifest13path)
                and manifest["source_private_v13b_intervention_array_sha256"]==sha(array13path)
                and manifest["source_private_control_selection_sha256"]==sha(selected_path)
                and manifest["rendered_arms"]==arms,
                "V14_SOURCE_RGBA_PRESELECTED_NATIVE_CONTROL_MANIFEST")
        rgba=Path(manifest["original_perturbed_pose_rgba_npz_path"])
        require(rgba.is_file() and sha(rgba)==manifest["original_perturbed_pose_rgba_npz_sha256"],
                "V14_ORIGINAL_RGBA_FULL_PIXELS_PRIVILEGED_RECEIPT_SHA")
        with np.load(rgba,allow_pickle=False) as a:
            require(set(a.files)=={"original_native_actuator_rgba"},
                    "V14_ORIGINAL_REAL_RGAB_PER_CONTROL_RENDER_NPZ")
            court=evaluate(a["original_native_actuator_rgba"],manifest,control_plan,
                           manifest13,private_batch,original_v13brow)
        court["parents"]={
            "original_source_sha256":source["source_sha256"],
            "original_v12_heldout_render_public_sha256":sha(args.public_v12),
            "original_v13b_source_mechanics_public_sha256":sha(args.public_v13b),
            "private_v13b_source_mechanical_array_sha256":sha(array13path),
            "private_original_v13b_native_carrier_manifest_sha256":sha(manifest13path),
            "raster_blind_control_plan_sha256":sha(selected_path),
            "original_native_v14_same_perturbation_rgba_npz_sha256":sha(rgba),
            "original_native_v14_renderer_manifest_sha256":sha(renderpath)
        }
        independent=out/"PRIVATE_INDEPENDENT_GEOMETRIC_MECHANICAL_RGBA_COURT_V14.json"
        seal(independent,court)
        require(court["graphics_owner_authorized"] is False
                and court["training_authorized"] is False
                and court["artist_same_drawable_pairs"]==0
                and court["artist_different_drawable_pairs"]==0,
                "V14_PUPPET_DRAWABLE_ARTIST_AUTHORITY_NOT_MINTED")
        result.update({
            "status":"MEASURED_REAL_SAME_CAUSAL_NATIVE_CONTROL_RGBA_V14__OWNER_UNKNOWN",
            "subject_id":sid,"subject_label":label,
            "source_sha256":source["source_sha256"],
            "selected_raster_blind_source_control_count":len(control_plan["controls"]),
            "source_native_control_selection_sha256":sha(selected_path),
            "real_rgba_blender_render_arms":len(arms)*12,
            "qualified_baseline_pose_camera_parity":
                court["baseline_source_vs_soup_and_root_noop_qualified_pose_views"],
            "qualified_selected_mechanical_rgba_pairs":
                court["original_native_perturbation_vs_source_soup_qualified_camera_cases"],
            "expected_selected_mechanical_rgba_pairs":
                court["original_native_perturbation_vs_source_soup_total_camera_cases"],
            "visual_perturbation_effect_pixel_instances":
                court["whole_source_perturbation_effective_pixel_instances_by_selected_control"],
            "causal_source_face_isolation_pixel_instances":
                court["causal_active_face_isolation_changed_pixel_instances_by_selected_control"],
            "independent_court_status":court["status"],
            "independent_court_all_parity_passed":
                court["all_original_source_camera_cases_passed"],
            "source_native_render_manifest_sha256":sha(renderpath),
            "source_native_rgba_npz_sha256":sha(rgba),
            "independent_court_sha256":sha(independent),
            "source_art_layer_identity_truth":False,
            "graphics_owner_authorized":False,
            "training_authorized":False
        })
    except (KeyboardInterrupt,MemoryError):
        raise
    except Exception as exc:
        result["status"]="FAIL_REAL_V14_NATIVE_SOURCE_MECHANICAL_RGBA_JOIN"
        result["failure_type"]=type(exc).__name__
        result["failure_reason"]=str(exc)
        seal(out/"PRIVATE_V14_SOURCE_FAILURE_RECEIPT.json",{
            "reason":str(exc),"type":type(exc).__name__,
            "traceback":traceback.format_exc(),
            "training_authorized":False,"graphics_owner_authorized":False
        })
    result["elapsed_seconds"]=round(time.monotonic()-started,3)
    seal(out/"PRIVATE_MECHANICAL_RGBA_JOIN_BATCH_V14.json",result)
    print(json.dumps({k:v for k,v in result.items()
                     if k!="failure_reason"},sort_keys=True),flush=True)
    require(result["status"]=="MEASURED_REAL_SAME_CAUSAL_NATIVE_CONTROL_RGBA_V14__OWNER_UNKNOWN",
            "V14_ORIGINAL_NATIVE_MECHANICAL_RGBA_COURT_NOT_COMPLETED")
    return result

def main():
    p=argparse.ArgumentParser()
    for k in ("v5_index","exact_roster","geometric_ray_index",
              "extracted_source_root","original_join_root",
              "allview_groups_root","native_teacher_root",
              "v11_private_root","v10_private_root",
              "public_v9","public_v10","public_v11","public_v12","public_v13b",
              "v13b_private_root","blender","output"):
        p.add_argument("--"+k.replace("_","-"),type=Path,required=True)
    for k in ("public_v9_sha256","public_v10_sha256","public_v11_sha256",
              "public_v12_sha256","public_v13b_sha256"):
        p.add_argument("--"+k.replace("_","-"),required=True)
    p.add_argument("--subject-ordinal",type=int,required=True)
    run(p.parse_args())

if __name__=="__main__":
    main()
