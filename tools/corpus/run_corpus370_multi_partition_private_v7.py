#!/usr/bin/env python3
"""Real original full-FOV Blender source vs anonymous group edits, in private.

Input originals are separately SHA-hydrated original Corpus370 V5, exact
370 roster, 352 geometry ray/rest courts, and independently courted candidate
source-native anonymous groups. One subject per parallel hosted runner.
All source art, rendered RGBA arrays, source triangle labels and Blender logs
stay in private Drive. No training, owner, product release or full CAA minted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback

from tools.corpus.audit_corpus370_v5_source_rest_courts_v1 import row_input
from tools.corpus.court_corpus370_multi_partition_edit_v7 import evaluate
from tools.corpus.corpus370_partition_edit_plan_v7 import build_plan, arm_names
import numpy as np

SCHEMA="RealSaS.Corpus370MultiPartitionPrivateRunner.v7"
V5_SHA="8ab2ebf526bbf91d78117580b68ed3784e53bcbafdbd326f012e143c924c0ac4"
ROSTER_SHA="925dac90ecbf949375aa3ef5d2ed2f40788ae7dcf5669a02b708ada8d9435355"
RAY_INDEX_SHA="0eeb24089322d0802567ba9736ada1de03e234bbf133bc91164132a80bec3c70"
SOURCE_JOIN_SHA="e72f3380c2f34cbf3e088218ebfcd4be0f4aa71c9138d4b3d1865a4f4074439a"
ALLVIEW_SHA="1d8b8e2e8018d2cd3b94188f2683424659a8b0735f43c43eb381b6b252a3b8a1"


def require(ok,reason):
    if not ok:raise ValueError(reason)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def load(path,expected,code):
    p=Path(path).resolve()
    require(p.is_file() and sha(p)==expected,code)
    return json.loads(p.read_text())


def ref(path):
    p=Path(path).resolve()
    require(p.is_file(),"FULL_FRAME_PARENT_FILE_MISSING")
    return {"path":str(p),"sha256":sha(p)}


def write(path,document):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".partial")
    tmp.write_text(json.dumps(document,indent=2,sort_keys=True,allow_nan=False)+"\n")
    tmp.replace(path)


def run(v5_path,roster_path,rays_index_path,source_root,join_root,
        allview_root,blender,output,subject_ordinal):
    sources=load(roster_path,ROSTER_SHA,"FULL_FRAME_ROSTER_PARENT_SHA")
    v5=load(v5_path,V5_SHA,"FULL_FRAME_ORIGINAL_V5_SHA")
    rays=load(rays_index_path,RAY_INDEX_SHA,"FULL_FRAME_GEOMETRIC_RAYS_INDEX_SHA")
    join=load(Path(join_root)/"CORPUS370_NATIVE_VISIBILITY_PAIR_JOIN_V1.json",
              SOURCE_JOIN_SHA,"FULL_FRAME_SOURCE_NATIVE_JOIN_SHA")
    group=load(Path(allview_root)/
               "CORPUS370_ALLVIEW_GEOMETRY_GROUPS_PRIVATE_BATCH_V3.json",
               ALLVIEW_SHA,"FULL_FRAME_MULTIVIEW_SELECTED_COURT_SHA")
    ids=sorted(join["subjects"])
    require(len(ids)==3 and set(ids)==set(group["subjects"])
            and type(subject_ordinal) is int and 0<=subject_ordinal<3
            and v5["prepared_subject_count"]==352
            and v5["failure_count"]==18
            and len(rays["subject_results"])==352
            and v5["training_authorized"] is False
            and sources["training_authorized"] is False
            and rays["training_authorized"] is False
            and join["training_authorized"] is False
            and group["model_training_authorized"] is False
            and group["graphics_owner_authorized"] is False
            and group["candidate_qualified_count"]==3
            and group["failure_count"]==0,
            "FULL_FRAME_ORIGINAL_THREE_SOURCE_UNIVERSE")
    sid=ids[subject_ordinal]
    src={x["subject_id"]:x for x in sources["subjects"]}[sid]
    row={x["subject_id"]:x for x in v5["subjects"]}[sid]
    require(src["source_sha256"]==row["source_sha256"]
            and src["source_sha256"]==join["subjects"][sid]["source_sha256"]
            and src["source_sha256"]==group["subjects"][sid]["source_sha256"],
            "FULL_FRAME_SOURCE_SHA_IDENTITY")
    source=Path(source_root).resolve()/src["source_relative_path"]
    root=Path(source_root).resolve()
    require(source.resolve().is_relative_to(root)
            and sha(source)==src["source_sha256"],
            "FULL_FRAME_RAW_SOURCE_PUBLISHER_SHA")
    bindings=row_input(row)
    geometry=rays["subject_results"][sid]
    rayfile=Path(geometry["receipt"]["path"]).resolve()
    raydoc=load(rayfile,geometry["receipt"]["sha256"],
                "FULL_FRAME_ORIGINAL_GEOMETRIC_RAY_SHA")
    require(raydoc.get("source_sha256")==src["source_sha256"]
            and raydoc.get("authority",{}).get("training_authorized") is False
            and raydoc.get("status")==
                "SOURCE_GEOMETRIC_RAY_CANDIDATES_MEASURED__RENDERED_ALPHA_AND_OWNER_COURTS_REQUIRED",
            "FULL_FRAME_RAY_SOURCE_SCOPE")
    # Bind the original source-rest geometry to the independent native/pair
    # court, not to an arbitrary visually similar mesh.
    pairref=join["subjects"][sid]["pair_diagnostic_receipt"]
    pairs=load(pairref["path"],pairref["sha256"],
               "FULL_FRAME_ORIGINAL_GRAPHICS_PAIR_SHA")
    require(raydoc["source_rest_geometry"]["sha256"]==
                pairs["parents"]["rest_geometry_sha256"]
            and pairs["parents"]["source_sha256"]==src["source_sha256"],
            "FULL_FRAME_SOURCE_REST_NATIVE_RAY_COURT_LINK")
    old=group["subjects"][sid]
    vg=Path(allview_root).resolve()/"subjects"/sid
    layerfile=vg/"ALLVIEW_ANONYMOUS_GROUP_CANDIDATES_V3.json"
    courtfile=vg/"ALLVIEW_ANONYMOUS_EDIT_COURT_V3.json"
    layer=load(layerfile,old["proposal_sha256"],
               "FULL_FRAME_SELECTED_LAYER_PROPOSAL_SHA")
    lawyer=load(courtfile,old["court_sha256"],
                "FULL_FRAME_INDEPENDENT_SELECTED_GROUP_COURT_SHA")
    require(layer["source_sha256"]==src["source_sha256"]
            and lawyer["training_authorized"] is False
            and lawyer["graphics_owner_family_authorized"] is False
            and lawyer["selected_group_count"]==old["selected_group_count"],
            "FULL_FRAME_NO_FAKE_LAYER_COURT")
    plan=build_plan(layer,pairs)
    require(plan["source_geometry_candidate_sizes"]==(
            [8,16,32,64] if plan["finest_available_reference_size"]==64
            else [8,16,32])
            and plan["task_count"]>=2
            and len(arm_names(plan))<=64,
            "PARTITION_PREEXISTING_SOURCE_CANDIDATES_OR_ARM_LIMIT")
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    job={
        "schema":"RealSaS.Corpus370MultiPartitionSourceEditJob.v7",
        "subject_id":sid,
        "source":ref(source),
        "admission":bindings["admission"],
        "candidate":bindings["candidate"],
        "camera":bindings["camera"],
        "rest_geometry":raydoc["source_rest_geometry"],
        "layer_proposal":ref(layerfile),
        "layer_court":ref(courtfile),
        "source_graphics_pairs":pairref,
        "views":list(range(8)),
        "persistent_pair_anchor_view":4,
        "source_face_proxy_tasks":True,
        "reference_policy":"FINEST_AVAILABLE_GEOMETRY_HYPOTHESIS",
        "expected_partition_sizes":plan["source_geometry_candidate_sizes"],
        "expected_reference_size":plan["finest_available_reference_size"],
        "expected_proxy_task_count":plan["task_count"],
        "cycles_samples":32,
        "raster_resolution":192,
        "teacher_only":True,
        "full_frame_render_qualifies_model":False,
        "graphics_owner_authorized":False,
        "training_authorized":False,
    }
    jobfile=output/"PRIVATE_MULTI_PARTITION_JOB_V7.json"
    write(jobfile,job)
    manifestfile=output/"PRIVATE_MULTI_PARTITION_RENDER_V7.json"
    summary={
        "schema":SCHEMA,
        "status":"FULL_FRAME_SOURCE_RESEARCH_INCOMPLETE",
        "subject_ordinal":subject_ordinal,
        "source_sha256":src["source_sha256"],
        "original_v5_sha256":V5_SHA,
        "original_rest_native_visibility_join_sha256":SOURCE_JOIN_SHA,
        "geometry_only_allview_groups_sha256":ALLVIEW_SHA,
        "full_frame_original_source_appearance_authorized":False,
        "graphics_owner_authorized":False,
        "model_training_authorized":False,
        "training_authorized":False,
    }
    receipt=output/"PRIVATE_MULTI_PARTITION_BATCH_RESULT_V7.json"
    start=time.monotonic()
    try:
        cmd=[
            str(Path(blender).resolve()),"-b","--factory-startup",
            "--python-exit-code","5",
            "--python","tools/truth_foundry/blender_corpus370_multi_partition_edit_v7.py",
            "--","--job",str(jobfile),"--output",str(manifestfile),
        ]
        ret=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                           text=True,check=False,timeout=3600)
        (output/"PRIVATE_MULTI_PARTITION_BLENDER_LOG.txt").write_text(ret.stdout[-320000:])
        require(ret.returncode==0,"FULL_FRAME_BLENDER_NONZERO_EXIT:"+str(ret.returncode))
        rendered=load(manifestfile,sha(manifestfile),"FULL_FRAME_PRODUCER_RESULT_NOT_SEALED")
        frames=Path(rendered["array_path"])
        require(frames.is_file() and sha(frames)==rendered["array_sha256"],
                "FULL_FRAME_PRODUCER_ARRAYS_SHA")
        with np.load(frames,allow_pickle=False) as data:
            require(set(data.files)=={"source_original_vs_partition_edits_rgba"},
                    "PARTITION_RENDER_BYTES_EXACT")
            decision=evaluate(data["source_original_vs_partition_edits_rgba"],
                              rendered,layer,pairs)
        decision["parents"]={
            "source_sha256":src["source_sha256"],
            "render_manifest_sha256":sha(manifestfile),
            "render_array_sha256":sha(frames),
            "allview_group_court_sha256":sha(courtfile),
        }
        court_path=output/"MULTI_PARTITION_EDIT_INDEPENDENT_COURT_V7.json"
        write(court_path,decision)
        # Research render success does not imply that any layer is an authored
        # drawable or that the group count is optimal for an editor.
        require(decision["source_original_vs_soup_noop_parity_views"]==8
                and decision["available_candidate_sizes"]==
                    plan["source_geometry_candidate_sizes"]
                and decision["proxy_edit_task_count"]==plan["task_count"]
                and decision["pareto_or_optimal_partition_selected"] is False
                and decision["training_authorized"] is False,
                "PARTITION_SOURCE_EDIT_RESEARCH_COURT_FAILED")
        summary.update({
            "status":"COMPLETE_NESTED_MULTI_PARTITION_REAL_SOURCE_EDIT_COURTS__NO_OWNER_AUTHORITY",
            "render_manifest_sha256":sha(manifestfile),
            "render_arrays_sha256":sha(frames),
            "independent_court_sha256":sha(court_path),
            "candidate_sizes":plan["source_geometry_candidate_sizes"],
            "unavailable_requested_sizes":plan["unavailable_requested_sizes"],
            "finest_available_reference_size":plan["finest_available_reference_size"],
            "source_hit_edit_proxy_task_count":plan["task_count"],
            "render_arms_per_camera":len(arm_names(plan)),
            "source_noop_parity_views":decision[
                "source_original_vs_soup_noop_parity_views"],
            "candidate_aggregates":decision["candidate_aggregates"],
            "optimal_group_count_minted":False,
            "graphics_owner_authorized":False,
            "training_authorized":False,
        })
    except (KeyboardInterrupt,MemoryError):
        raise
    except Exception as err:
        summary["status"]="FAIL_REAL_FULL_FRAME_SOURCE_OR_INDEPENDENT_COURT"
        summary["error_type"]=type(err).__name__
        summary["reason"]=str(err)
        write(output/"MULTI_PARTITION_FAILURE_V7.json",{
            "source_sha256":src["source_sha256"],
            "error_type":type(err).__name__,
            "reason":str(err),"traceback":traceback.format_exc(),
            "training_authorized":False,
        })
    summary["elapsed_seconds"]=round(time.monotonic()-start,3)
    write(receipt,summary)
    print(json.dumps({k:v for k,v in summary.items()
                      if k not in ("source_sha256","reason")},
                     sort_keys=True),flush=True)
    require(not summary["status"].startswith("FAIL_"),
            "MULTI_PARTITION_SOURCE_FAILURE_REQUIRES_REPAIR")
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--v5-index",type=Path,required=True)
    p.add_argument("--exact-roster",type=Path,required=True)
    p.add_argument("--geometric-ray-index",type=Path,required=True)
    p.add_argument("--extracted-source-root",type=Path,required=True)
    p.add_argument("--original-join-root",type=Path,required=True)
    p.add_argument("--allview-groups-root",type=Path,required=True)
    p.add_argument("--blender",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--subject-ordinal",type=int,required=True)
    a=p.parse_args()
    run(a.v5_index,a.exact_roster,a.geometric_ray_index,
        a.extracted_source_root,a.original_join_root,
        a.allview_groups_root,a.blender,a.output,a.subject_ordinal)


if __name__=="__main__":main()
