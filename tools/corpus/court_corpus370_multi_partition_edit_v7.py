#!/usr/bin/env python3
"""Independent full-field source edit consequence court for competing partitions.

The finest AVAILABLE nested partition is a *proxy edit reference*, not an
artist's authored drawable segmentation. Source face addresses are privileged
teacher-only. Never mint graphics owner, CAA, mechanical or training authority.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.corpus.corpus370_partition_edit_plan_v7 import build_plan, arm_names

SCHEMA="RealSaS.Corpus370MultiPartitionSourceEditCourt.v7"
RENDER="RealSaS.Corpus370MultiPartitionSourceEditRender.v7"
SOURCE_PARITY_P99_TOL=.025
NOOP_MAX_TOL=.0025
EDIT_EFFECT_TOL=.025


def require(ok,code):
    if not ok:raise ValueError(code)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def signal(a,b):
    alpha=np.maximum(a[...,3],b[...,3])
    color=np.max(np.abs(a[...,:3]-b[...,:3]),axis=-1)
    return np.maximum(np.abs(a[...,3]-b[...,3]),color*(alpha>1e-4))


def evaluate(pixels,manifest,proposal,pair_receipt):
    plan=build_plan(proposal,pair_receipt)
    require(manifest.get("schema")==RENDER
            and manifest.get("status")==
                "SOURCE_MULTIPARTITION_REST_EDIT_PROXY__INDEPENDENT_COURT_REQUIRED"
            and manifest.get("teacher_only") is True
            and manifest.get("training_authorized") is False
            and manifest.get("graphics_owner_authorized") is False
            and manifest.get("model_training_authorized") is False
            and manifest.get("original_geometry_reimport_parity_verified") is True
            and manifest.get("reference_policy")=="FINEST_AVAILABLE_GEOMETRY_HYPOTHESIS"
            and manifest.get("source_sha256")==proposal["source_sha256"]
            and manifest.get("original_source_partition_edit_plan_teacher_only")==plan
            and manifest.get("source_group_partition_candidates")==
                plan["source_geometry_candidate_sizes"]
            and manifest.get("arms")==arm_names(plan)
            and manifest.get("views")==list(range(8)),
            "PARTITION_COURT_EXACT_PLAN_AND_FALSE_AUTHORITY")
    rows=manifest.get("private_view_details")
    require(isinstance(rows,list) and len(rows)==8
            and [x.get("view") for x in rows]==list(range(8))
            and all(x.get("source_groups_same_addresses_across_cameras") is True
                    for x in rows),"PARTITION_COURT_STABLE_SOURCE_LABELS")
    p=np.asarray(pixels)
    resolution=manifest.get("full_frame_resolution")
    arms=arm_names(plan)
    require(type(resolution) is int and 128<=resolution<=512
            and p.shape==(8,len(arms),resolution,resolution,4)
            and p.dtype.kind=="f" and np.isfinite(p).all()
            and np.all(p[...,3]>=-1e-5) and np.all(p[...,3]<=1+1e-5),
            "PARTITION_COURT_RGBA_FINITE_SHAPE")
    index={a:i for i,a in enumerate(arms)}
    parity=[]
    for v in range(8):
        reference=p[v,index["SOURCE_ORIGINAL"]]
        full=p[v,index["SOUP_FULL"]]
        noop=p[v,index["SOUP_NOOP"]]
        source_p99=float(np.quantile(signal(reference,full),.99))
        noop_max=float(signal(full,noop).max())
        parity.append({"view":v,"source_soup_p99":source_p99,
                       "soup_noop_max":noop_max,
                       "passed":source_p99<=SOURCE_PARITY_P99_TOL
                                and noop_max<=NOOP_MAX_TOL})
    require(all(x["passed"] for x in parity),
            "PARTITION_REAL_SOURCE_OR_NOOP_PARITY_FAILURE")
    sizes=plan["source_geometry_candidate_sizes"]
    fine=plan["finest_available_reference_size"]
    per_task=[]
    for task in plan["tasks"]:
        idx=task["task_index"]
        ora=index[f"FINE_REMOVE_T{idx:02d}"]
        oia=index[f"FINE_ISOLATE_T{idx:02d}"]
        by_size={}
        # Full 8-camera source image is the consumer, not merely one oracle pixel.
        reference_effects=0
        isolation_reference_effects=0
        for v in range(8):
            reference_effects+=int(np.count_nonzero(
                signal(p[v,index["SOUP_FULL"]],p[v,ora])>EDIT_EFFECT_TOL))
            isolation_reference_effects+=int(np.count_nonzero(
                signal(p[v,index["REMOVE_ALL_GROUPS"]],p[v,oia])>EDIT_EFFECT_TOL))
        for size in sizes:
            rem=index[f"GROUP_REMOVE_K{size:03d}_T{idx:02d}"]
            isolate=index[f"GROUP_ISOLATE_K{size:03d}_T{idx:02d}"]
            direct_pixels=collateral=peel_collateral=isolated_pixels=0
            direct_views=0
            for v in range(8):
                full=p[v,index["SOUP_FULL"]]
                altered=p[v,rem]
                fine_removed=p[v,ora]
                fine_isolated=p[v,oia]
                candidate_isolated=p[v,isolate]
                removed=signal(full,altered)
                effect=int(np.count_nonzero(removed>EDIT_EFFECT_TOL))
                direct_pixels+=effect
                direct_views+=int(effect>0)
                collateral+=int(np.count_nonzero(
                    signal(fine_removed,altered)>EDIT_EFFECT_TOL))
                peel_collateral+=int(np.count_nonzero(
                    signal(fine_isolated,candidate_isolated)>EDIT_EFFECT_TOL))
                isolated_pixels+=int(np.count_nonzero(
                    signal(p[v,index["REMOVE_ALL_GROUPS"]],candidate_isolated)
                    >EDIT_EFFECT_TOL))
            # The fine candidate must reproduce the exact teacher fine reference.
            if size==fine:
                require(collateral==0 and peel_collateral==0,
                        "PARTITION_FINE_REFERENCE_NOT_EQUIVALENT_TO_SELF")
            by_size[str(size)]={
                "source_group_count":size,
                "task_source_group_face_count_teacher_only":None,
                "direct_edit_effect_pixel_instances":direct_pixels,
                "direct_effect_camera_views":direct_views,
                "full_peel_isolated_pixel_instances":isolated_pixels,
                "additional_edit_delta_pixels_vs_finest_reference":collateral,
                "additional_isolated_delta_pixels_vs_finest_reference":peel_collateral,
                "additional_edit_delta_per_reference_direct_pixel":(
                    collateral/max(1,reference_effects)
                    if reference_effects>0 else None),
                "additional_isolated_delta_per_reference_pixel":(
                    peel_collateral/max(1,isolation_reference_effects)
                    if isolation_reference_effects>0 else None),
                "status":("MEASURED_SOURCE_EDIT_PROXY__NOT_DRAWABLE_TRUTH"
                          if isolation_reference_effects>0 else
                          "ABSTAIN_REFERENCE_NOT_VISIBLE_EVEN_IN_ISOLATION"),
                "same_drawable":"UNKNOWN",
                "different_drawable":"UNKNOWN",
                "training_authorized":False,
            }
        per_task.append({
            "task_index":idx,
            "privileged_source_face_teacher_only":task[
                "source_face_teacher_only"],
            "finest_available_reference_group_size":fine,
            "reference_direct_edit_pixel_instances":reference_effects,
            "reference_isolation_pixel_instances":isolation_reference_effects,
            "candidate_consequences":by_size,
            "training_authorized":False,
        })
    aggregates={}
    for size in sizes:
        rows=[x["candidate_consequences"][str(size)] for x in per_task]
        aggregates[str(size)]={
            "source_group_count":size,
            "scoped_proxy_task_count":len(per_task),
            "source_direct_edit_delta_pixels_vs_fine_sum":sum(
                x["additional_edit_delta_pixels_vs_finest_reference"] for x in rows),
            "source_peel_isolation_delta_pixels_vs_fine_sum":sum(
                x["additional_isolated_delta_pixels_vs_finest_reference"] for x in rows),
            "unobserved_reference_task_count":sum(
                int(x["status"].startswith("ABSTAIN")) for x in rows),
            "operational_complexity_group_count":size,
            "selection_authorized":False,
        }
    return {
        "schema":SCHEMA,
        "status":"MEASURED_COMPETING_NESTED_PARTITION_EDIT_PROXIES__NOT_DRAWABLE_AUTHORITY",
        "source_sha256":proposal["source_sha256"],
        "reference_policy":"FINEST_AVAILABLE_GEOMETRY_HYPOTHESIS",
        "reference_is_not_artist_truth":True,
        "available_candidate_sizes":sizes,
        "unavailable_requested_candidate_sizes":plan["unavailable_requested_sizes"],
        "proxy_edit_task_count":len(per_task),
        "camera_count":8,
        "render_arms_per_camera":len(arms),
        "source_original_vs_soup_noop_parity_views":sum(int(x["passed"]) for x in parity),
        "parity":parity,
        "per_task_teacher_only":per_task,
        "candidate_aggregates":aggregates,
        "pareto_or_optimal_partition_selected":False,
        "source_labels_student_targets":False,
        "same_drawable_positive_pairs":0,
        "different_drawable_negative_pairs":0,
        "graphics_owner_authorized":False,
        "appearance_caa_total_authority_minted":False,
        "mechanics_authorized":False,
        "tessa_teacher_authorized":False,
        "training_authorized":False,
        "materialization_authorized":False,
    }


def main():
    q=argparse.ArgumentParser()
    q.add_argument("--manifest",type=Path,required=True)
    q.add_argument("--expected-manifest-sha256",required=True)
    q.add_argument("--arrays",type=Path,required=True)
    q.add_argument("--expected-arrays-sha256",required=True)
    q.add_argument("--proposal",type=Path,required=True)
    q.add_argument("--expected-proposal-sha256",required=True)
    q.add_argument("--source-pairs",type=Path,required=True)
    q.add_argument("--expected-source-pairs-sha256",required=True)
    q.add_argument("--output",type=Path,required=True)
    a=q.parse_args()
    for path,expected in [
        (a.manifest,a.expected_manifest_sha256),
        (a.arrays,a.expected_arrays_sha256),
        (a.proposal,a.expected_proposal_sha256),
        (a.source_pairs,a.expected_source_pairs_sha256),
    ]:
        require(path.is_file() and sha(path)==expected,"PARTITION_COURT_PARENT_SHA")
    m=json.loads(a.manifest.read_text())
    require(m["array_sha256"]==a.expected_arrays_sha256
            and m["selected_layer_proposal_sha256"]==a.expected_proposal_sha256
            and m["source_graphics_pair_sha256"]==a.expected_source_pairs_sha256,
            "PARTITION_COURT_PRODUCER_SOURCE_REFERENCES")
    with np.load(a.arrays,allow_pickle=False) as f:
        require(set(f.files)=={"source_original_vs_partition_edits_rgba"},
                "PARTITION_ARRAYS_NO_EXTRA_PAYLOAD")
        result=evaluate(f["source_original_vs_partition_edits_rgba"],
                        m,json.loads(a.proposal.read_text()),
                        json.loads(a.source_pairs.read_text()))
    result["parents"]={
        "source_sha256":m["source_sha256"],
        "render_manifest_sha256":a.expected_manifest_sha256,
        "render_array_sha256":a.expected_arrays_sha256,
        "source_layer_proposal_sha256":a.expected_proposal_sha256,
        "source_pairs_sha256":a.expected_source_pairs_sha256,
    }
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":result["status"],"sizes":result["available_candidate_sizes"],
                      "tasks":result["proxy_edit_task_count"],
                      "training_authorized":False},sort_keys=True))


if __name__=="__main__":main()
