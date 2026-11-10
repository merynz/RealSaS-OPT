#!/usr/bin/env python3
"""Hosted V9 native animated geometry and real-source original/soup/edit rasters.

Exact original V5 SHA, 370 roster, visibility/ray rest court, original grouped
hypotheses and original authored teacher manifest and NPZ are independently
hydrated and checked. Only private pixel buffers and source-face IDs are output.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np

from tools.corpus.audit_corpus370_v5_source_rest_courts_v1 import row_input
from tools.corpus.corpus370_partition_edit_plan_v7 import build_plan,arm_names
from tools.corpus.court_corpus370_native_animated_partition_v9 import evaluate

SCHEMA="RealSaS.Corpus370NativeAnimatedPartitionPrivateBatch.v9"
V5_SHA="8ab2ebf526bbf91d78117580b68ed3784e53bcbafdbd326f012e143c924c0ac4"
ROSTER_SHA="925dac90ecbf949375aa3ef5d2ed2f40788ae7dcf5669a02b708ada8d9435355"
RAYS_SHA="0eeb24089322d0802567ba9736ada1de03e234bbf133bc91164132a80bec3c70"
JOIN_SHA="e72f3380c2f34cbf3e088218ebfcd4be0f4aa71c9138d4b3d1865a4f4074439a"
GROUP_SHA="1d8b8e2e8018d2cd3b94188f2683424659a8b0735f43c43eb381b6b252a3b8a1"
NATIVE_SHA="5aa3c4d9cc97ab669d2a7a951a4c9d18218ef5ea4d6f58f5b1ff1ac5b89cc272"

def require(ok,code):
    if not ok:raise ValueError(code)

def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()

def read(path,digest,code):
    path=Path(path).resolve()
    require(path.is_file() and sha(path)==digest,code)
    return json.loads(path.read_text())

def ref(path):
    path=Path(path).resolve()
    require(path.is_file(),"V9_PRIVATE_PARENT_MISSING")
    return {"path":str(path),"sha256":sha(path)}

def seal(path,doc):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".partial")
    tmp.write_text(json.dumps(doc,indent=2,sort_keys=True,allow_nan=False)+"\n")
    tmp.replace(path)

def pick_three_poses(native,receipt,rest):
    """Deterministic source-native geometric activity ranking across actions.

    Real source pose displacement, not bone names or semantic classes.
    Distinct authored actions are required; samples are never selected from
    model edits, visual render results or human segmentation.
    """
    p=np.asarray(native["posed_vertices_source_world"],np.float64)
    r=np.asarray(rest,np.float64)
    frames=receipt.get("sampled_frames",[])
    require(p.ndim==3 and len(p)==len(frames)
            and p.shape[1:]==r.shape
            and np.isfinite(p).all()
            and len(frames)>=3,
            "V9_SOURCE_NATIVE_FRAMES_AND_REST")
    score=np.sqrt(np.mean(np.sum((p-r[None])**2,axis=-1),axis=-1))
    ranked={}
    for index,frame in enumerate(frames):
        key=frame["action"]
        choice=(-float(score[index]),index)
        if key not in ranked or choice<ranked[key]:
            ranked[key]=choice
    require(len(ranked)>=3,"V9_NO_THREE_DISTINCT_NATIVE_ACTIONS")
    top=sorted((priority[0],priority[1],key)
               for key,priority in ranked.items())[:3]
    require(all(np.isfinite(-v) for v,_,_ in top),"V9_NATIVE_ACTIVITY_SCORE_NONFINITE")
    return sorted([index for _,index,_ in top])

def run(args):
    out=Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=True)
    summary={"schema":SCHEMA,
             "status":"V9_REAL_ANIMATED_SOURCE_COURT_INCOMPLETE",
             "subject_ordinal":args.subject_ordinal,
             "animated_render_caa_authorized":False,
             "graphics_owner_authorized":False,
             "training_authorized":False}
    start=time.monotonic()
    try:
        v5=read(args.v5_index,V5_SHA,"V9_ORIGINAL_V5_SHA")
        roster=read(args.exact_roster,ROSTER_SHA,"V9_370_SOURCE_ROSTER_SHA")
        rays=read(args.geometric_ray_index,RAYS_SHA,"V9_ORIGINAL_GEOMETRY_RAYS_SHA")
        join=read(Path(args.original_join_root)/
                  "CORPUS370_NATIVE_VISIBILITY_PAIR_JOIN_V1.json",
                  JOIN_SHA,"V9_NATIVE_SOURCE_HIT_JOIN_SHA")
        group=read(Path(args.allview_groups_root)/
                   "CORPUS370_ALLVIEW_GEOMETRY_GROUPS_PRIVATE_BATCH_V3.json",
                   GROUP_SHA,"V9_ORIGINAL_GROUP_COHORT_SHA")
        native=read(Path(args.native_teacher_root)/
                    "CORPUS370_V5_AUTHORED_TEMPORAL_PILOT_V1.json",
                    NATIVE_SHA,"V9_ORIGINAL_AUTHORED_NATIVE_PILOT_SHA")
        ids=sorted(join["subjects"])
        require(type(args.subject_ordinal) is int and
                0<=args.subject_ordinal<3 and len(ids)==3
                and set(ids)==set(group["subjects"])
                and set(ids)==set(native["selected_subject_ids"])
                and v5["prepared_subject_count"]==352
                and v5["failure_count"]==18
                and v5["training_authorized"] is False
                and native["training_authorized"] is False
                and group["model_training_authorized"] is False,
                "V9_THREE_REAL_SOURCE_COHORT_NO_TRAINING")
        sid=ids[args.subject_ordinal]
        source={x["subject_id"]:x for x in roster["subjects"]}[sid]
        obs={x["subject_id"]:x for x in v5["subjects"]}[sid]
        teacher_row={x["subject_id"]:x for x in native["results"]}[sid]
        require(source["source_sha256"]==obs["source_sha256"]==
                teacher_row["source_sha256"]==
                group["subjects"][sid]["source_sha256"]==
                join["subjects"][sid]["source_sha256"],
                "V9_SOURCE_SHA_PARENT_IDENTITY")
        root=Path(args.extracted_source_root).resolve()
        source_path=(root/source["source_relative_path"]).resolve()
        require(source_path.is_relative_to(root)
                and source_path.is_file()
                and sha(source_path)==source["source_sha256"],
                "V9_RAW_ORIGINAL_SOURCE_SHA")
        bind=row_input(obs)
        rayref=rays["subject_results"][sid]["receipt"]
        ray=read(rayref["path"],rayref["sha256"],"V9_SOURCE_GEOMETRY_RAYS_COURT")
        restref=ray["source_rest_geometry"]
        pairsref=join["subjects"][sid]["pair_diagnostic_receipt"]
        pairs=read(pairsref["path"],pairsref["sha256"],
                   "V9_NATIVE_ORIGINAL_SOURCE_FACE_PAIRS")
        require(restref["sha256"]==pairs["parents"]["rest_geometry_sha256"],
                "V9_SOURCE_REST_NATIVE_PAIR_SHA")
        groups_root=Path(args.allview_groups_root).resolve()/"subjects"/sid
        proposal_path=groups_root/"ALLVIEW_ANONYMOUS_GROUP_CANDIDATES_V3.json"
        prior=group["subjects"][sid]
        proposal=read(proposal_path,prior["proposal_sha256"],
                      "V9_BEFORE_RENDER_NESTED_SOURCE_PROPOSAL_SHA")
        old_court_path=groups_root/"ALLVIEW_ANONYMOUS_EDIT_COURT_V3.json"
        old_court=read(old_court_path,prior["court_sha256"],
                       "V9_ORIGINAL_ANON_GROUP_COURT_SHA")
        require(old_court["training_authorized"] is False
                and old_court["graphics_owner_family_authorized"] is False,
                "V9_NOT_AUTHOR_DRAWABLE_TRUTH")
        plan=build_plan(proposal,pairs)
        require(len(arm_names(plan))<=64 and plan["task_count"]>=2,
                "V9_AUTHORED_ANIMATED_ARMS_BOUNDED")
        teacher_dir=Path(args.native_teacher_root).resolve()/"subjects"/sid
        teacher_court_path=teacher_dir/"TEMPORAL_TEACHER_COURT_V1.json"
        teacher_court=read(teacher_court_path,sha(teacher_court_path),
                           "V9_SOURCE_AUTHORED_INDEPENDENT_COURT")
        require(teacher_court["source_sha256"]==source["source_sha256"]
                and teacher_court["teacher_arrays"]["sha256"]==
                    teacher_row["teacher_arrays"]["sha256"]
                and teacher_court["teacher_manifest"]["sha256"]==
                    teacher_row["teacher_manifest"]["sha256"]
                and teacher_court["training_authorized"] is False
                and teacher_court["family_authority_minted"] is False,
                "V9_CORPUS370_NATIVE_PILOT_ORIGINAL_AUTHORITY")
        teacher_npz=Path(teacher_court["teacher_arrays"]["path"])
        receipt_path=Path(teacher_court["teacher_manifest"]["path"])
        require(sha(teacher_npz)==teacher_court["teacher_arrays"]["sha256"],
                "V9_AUTHORED_TEACHER_ARRAY_SHA")
        receipt=read(receipt_path,teacher_court["teacher_manifest"]["sha256"],
                     "V9_AUTHORED_TEACHER_MANIFEST_SHA")
        restpath=Path(restref["path"])
        require(sha(restpath)==restref["sha256"],
                "V9_SOURCE_REST_ARRAY_SHA")
        with np.load(restpath,allow_pickle=False) as original, np.load(
                teacher_npz,allow_pickle=False) as teacher:
            poses=pick_three_poses(teacher,receipt,original["rest_vertices_source_world"])
        job={
            "schema":"RealSaS.Corpus370NativeAnimatedPartitionJob.v9",
            "subject_id":sid,
            "source":ref(source_path),
            "admission":bind["admission"],
            "candidate":bind["candidate"],
            "camera":bind["camera"],
            "rest_geometry":restref,
            "layer_proposal":ref(proposal_path),
            "layer_court":ref(old_court_path),
            "source_graphics_pairs":pairsref,
            "native_teacher":ref(teacher_npz),
            "native_teacher_receipt":ref(receipt_path),
            "native_teacher_court":ref(teacher_court_path),
            "expected_partition_sizes":plan["source_geometry_candidate_sizes"],
            "expected_proxy_task_count":plan["task_count"],
            "pose_indices":poses,
            "view_indices":[0,2,4,6],
            "raster_resolution":192,
            "cycles_samples":32,
            "reference_policy":"FINEST_AVAILABLE_GEOMETRY_HYPOTHESIS_NOT_AUTHOR_TRUTH",
            "teacher_only":True,
            "source_bone_names_as_targets":False,
            "native_full_caa_authorized":False,
            "graphics_owner_authorized":False,
            "training_authorized":False,
        }
        jobpath=out/"PRIVATE_ANIMATED_ORIGINAL_SOURCE_RENDER_JOB_V9.json"
        seal(jobpath,job)
        manifestpath=out/"PRIVATE_ANIMATED_SOURCE_RENDER_MANIFEST_V9.json"
        cmd=[str(Path(args.blender).resolve()),"-b","--factory-startup",
             "--python-exit-code","5","--python",
             "tools/truth_foundry/blender_corpus370_native_animated_partition_render_v9.py",
             "--","--job",str(jobpath),"--output",str(manifestpath)]
        subprocess_error=subprocess.run(cmd,stdout=subprocess.PIPE,
             stderr=subprocess.STDOUT,text=True,check=False,timeout=7200)
        (out/"PRIVATE_ANIMATED_BLENDER_LOG_V9.txt").write_text(
             subprocess_error.stdout[-380000:])
        require(subprocess_error.returncode==0,
                "V9_REAL_NATIVE_BLENDER_NONZERO_EXIT:"+str(subprocess_error.returncode))
        manifest=read(manifestpath,sha(manifestpath),
                      "V9_PRIVATE_ANIMATED_RENDER_MANIFEST_SHA")
        require(manifest["private_job_sha256"]==sha(jobpath)
                and manifest["pose_indices"]==poses,
                "V9_SOURCE_NATIVE_BLEND_POSES_NOT_SELECTED")
        pix=Path(manifest["array_path"])
        require(pix.is_file() and sha(pix)==manifest["array_sha256"],
                "V9_RENDER_ARRAY_PARENT_SHA")
        with np.load(pix,allow_pickle=False) as arrays:
            require(set(arrays.files)=={"animated_original_and_group_edits_rgba"},
                    "V9_RENDER_RGBA_ARRAY_ONLY")
            decision=evaluate(arrays["animated_original_and_group_edits_rgba"],
                              manifest,proposal,pairs,receipt)
        decision["parents"]={
            "source_sha256":source["source_sha256"],
            "original_v5_index_sha256":V5_SHA,
            "original_native_teacher_sha256":sha(teacher_npz),
            "original_group_proposal_sha256":sha(proposal_path),
            "real_render_manifest_sha256":sha(manifestpath),
            "real_render_rgba_arrays_sha256":sha(pix),
        }
        courtpath=out/"INDEPENDENT_ANIMATED_NATIVE_RASTER_COURT_V9.json"
        seal(courtpath,decision)
        require(decision["training_authorized"] is False
                and decision["same_drawable_positive_pairs"]==0
                and decision["different_drawable_negative_pairs"]==0,
                "V9_COURT_FALSE_DRAWABLE_PROMOTION")
        summary.update({
            "status":"MEASURED_REAL_NATIVE_RGBA_PARITY_AND_EDIT_PROXY_V9__NO_OWNER",
            "subject_id":sid,
            "source_sha256":source["source_sha256"],
            "source_authored_teacher_sha256":sha(teacher_npz),
            "real_selected_source_pose_indices_teacher_only":poses,
            "real_native_source_pose_count":len(poses),
            "real_native_camera_views_per_pose":4,
            "render_arms_per_pose_view":len(manifest["arms"]),
            "real_render_arms_total":len(poses)*4*len(manifest["arms"]),
            "render_manifest_sha256":sha(manifestpath),
            "render_rgba_npz_sha256":sha(pix),
            "independent_court_sha256":sha(courtpath),
            "qualified_parity_pose_views":decision["source_soup_parity_qualified_pose_views"],
            "unqualified_parity_pose_views":decision["unqualified_pose_views"],
            "all_pose_view_source_parity":decision["all_pose_view_source_parity"],
            "candidate_aggregates":decision["candidate_aggregates"],
            "optimal_group_count_authorized":False,
            "graphics_owner_authorized":False,
            "training_authorized":False,
        })
    except (KeyboardInterrupt,MemoryError):
        raise
    except Exception as err:
        summary["status"]="FAIL_V9_REAL_AUTHORED_SOURCE_MEASUREMENT"
        summary["error_type"]=type(err).__name__
        summary["error_reason"]=str(err)
        seal(out/"PRIVATE_ANIMATED_SOURCE_FAILURE_V9.json",{
            "error_type":type(err).__name__,"reason":str(err),
            "traceback":traceback.format_exc(),"training_authorized":False,
        })
    summary["elapsed_seconds"]=round(time.monotonic()-start,3)
    seal(out/"PRIVATE_NATIVE_ANIMATED_PARTITION_BATCH_V9.json",summary)
    print(json.dumps({k:v for k,v in summary.items()
                      if k not in ("source_sha256","error_reason")},
                     sort_keys=True),flush=True)
    require(not summary["status"].startswith("FAIL_"),
            "V9_REAL_SOURCE_ANIMATED_REPLAY_FAILURE_REQUIRES_REPAIR")
    return summary

def main():
    p=argparse.ArgumentParser()
    for name in ("v5_index","exact_roster","geometric_ray_index",
                 "extracted_source_root","original_join_root",
                 "allview_groups_root","native_teacher_root",
                 "blender","output"):
        p.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    p.add_argument("--subject-ordinal",type=int,required=True)
    run(p.parse_args())

if __name__=="__main__":
    main()
