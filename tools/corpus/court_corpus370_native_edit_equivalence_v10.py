#!/usr/bin/env python3
"""V10: genuine native source-edit pairwise observational equivalence court.

Compare ALL anonymous group partition responses directly, not with finest
partition designated ground truth. Model labels, artist same/different drawable
ownership, and training admission remain UNKNOWN/FALSE. Source arrays private.
"""
from __future__ import annotations

import argparse
from itertools import combinations
import hashlib
import json
from pathlib import Path
import numpy as np

SCHEMA="RealSaS.Corpus370NativeAnimatedEditResponseEquivalenceCourt.v10"
V9="RealSaS.Corpus370NativeAnimatedPartitionRasterCourt.v9"
RENDER="RealSaS.Corpus370NativeAnimatedPartitionSourceRender.v9"
PARENT="RealSaS.Corpus370NativeAnimatedPartitionRealSourceResearchResult.v9"
SIGNAL_TOL=.025

def require(value,code):
    if not value:
        raise ValueError(code)

def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()

def read_checked(path,digest,code):
    p=Path(path)
    require(p.is_file() and sha(p)==digest,code)
    return json.loads(p.read_text())

def signal(a,b):
    alpha=np.maximum(a[...,3],b[...,3])
    color=np.max(np.abs(a[...,:3]-b[...,:3]),axis=-1)
    return np.maximum(np.abs(a[...,3]-b[...,3]),color*(alpha>1e-4))

def evaluate(pixels,manifest,v9court):
    require(manifest.get("schema")==RENDER
            and manifest.get("status")==
                "REAL_AUTHORED_NATIVE_POSE_ANIMATED_RGBA_RENDER__INDEPENDENT_COURT_REQUIRED"
            and manifest.get("teacher_only") is True
            and manifest.get("graphics_owner_authorized") is False
            and manifest.get("training_authorized") is False
            and v9court.get("schema")==V9
            and v9court.get("source_sha256")==manifest.get("source_sha256")
            and v9court.get("subject_id")==manifest.get("subject_id")
            and v9court.get("all_pose_view_source_parity") is True
            and v9court.get("source_soup_parity_qualified_pose_views")==12
            and v9court.get("unqualified_pose_views")==0
            and v9court.get("graphics_owner_authorized") is False
            and v9court.get("training_authorized") is False
            and v9court.get("same_drawable_positive_pairs")==0
            and v9court.get("different_drawable_negative_pairs")==0
            and v9court.get("available_group_sizes")==manifest.get("candidate_sizes")
            and v9court.get("source_face_proxy_task_count")==manifest.get("plan",{}).get("task_count")
            and manifest.get("pose_indices") is not None
            and len(manifest.get("pose_indices"))==3
            and manifest.get("view_indices")==[0,2,4,6],
            "V10_PRIOR_REAL_ORIGINAL_SOURCE_POSE_PARITY_AND_FALSE_LABEL_AUTHORITY")
    sizes=manifest["candidate_sizes"]
    require(isinstance(sizes,list) and
            sizes==sorted(set(sizes)) and
            len(sizes)>=2 and sizes[0]==8 and
            all(type(k) is int and k in (8,16,32,64) for k in sizes),
            "V10_SOURCE_SIZE_DOMAIN")
    arms=manifest["arms"]
    require(isinstance(arms,list) and len(arms)==len(set(arms)),
            "V10_SOURCE_EDIT_ARMS_DISTINCT")
    ix={a:i for i,a in enumerate(arms)}
    p=np.asarray(pixels)
    require(p.shape==(3,4,len(arms),192,192,4)
            and p.dtype.kind=="f" and np.isfinite(p).all()
            and np.all(p[...,3]>=-1e-5)
            and np.all(p[...,3]<=1.+1e-5),
            "V10_REAL_NATIVE_ANIMATED_RGBA_SHAPE")
    # Full source vs posed soup no-op is independently replayed, not trusted
    # merely from the previous v9 court flags.
    require(np.max(signal(p[:,:,ix["SOUP_FULL"]],
                          p[:,:,ix["SOUP_NOOP"]]))<=.0025,
            "V10_REAL_SOURCE_SOUP_NOOP_NOT_CLOSED")
    task_rows=[]
    for task in manifest["plan"]["tasks"]:
        tid=task["task_index"]
        per_pair=[]
        # Require an ACTUAL tested source edit event; an invisible probe cannot
        # establish equivalence between partitions.
        fine=manifest["plan"]["finest_available_reference_size"]
        fine_direct=p[:,:,ix[f"FINE_REMOVE_T{tid:02d}"]]
        baseline=p[:,:,ix["SOUP_FULL"]]
        reference_effect=int(np.count_nonzero(signal(fine_direct,baseline)>SIGNAL_TOL))
        require(reference_effect>0,"V10_INVISIBLE_FINE_TARGET_REFUSE_EQUIVALENCE")
        clear=p[:,:,ix["REMOVE_ALL_GROUPS"]]
        fine_isolate=p[:,:,ix[f"FINE_ISOLATE_T{tid:02d}"]]
        reference_isolate=int(np.count_nonzero(signal(fine_isolate,clear)>SIGNAL_TOL))
        require(reference_isolate>0,"V10_NO_SOURCE_ISOLATION_REVEAL_ABSTAIN")
        for a,b in combinations(sizes,2):
            ka=f"GROUP_REMOVE_K{a:03d}_T{tid:02d}"
            kb=f"GROUP_REMOVE_K{b:03d}_T{tid:02d}"
            ia=f"GROUP_ISOLATE_K{a:03d}_T{tid:02d}"
            ib=f"GROUP_ISOLATE_K{b:03d}_T{tid:02d}"
            require(all(k in ix for k in (ka,kb,ia,ib)),
                    "V10_ORIGINAL_REAL_RENDER_ARM_MISSING")
            direct=signal(p[:,:,ix[ka]],p[:,:,ix[kb]])
            isolate=signal(p[:,:,ix[ia]],p[:,:,ix[ib]])
            extra_direct=int(np.count_nonzero(direct>SIGNAL_TOL))
            extra_isolate=int(np.count_nonzero(isolate>SIGNAL_TOL))
            indistinguishable=extra_direct==0 and extra_isolate==0
            per_pair.append({
                "candidate_a":a,"candidate_b":b,
                "additional_direct_pixel_instances":extra_direct,
                "additional_isolate_pixel_instances":extra_isolate,
                "direct_discriminating_pose_views":int(np.count_nonzero(
                    np.count_nonzero(direct>SIGNAL_TOL,axis=(-2,-1))>0)),
                "isolate_discriminating_pose_views":int(np.count_nonzero(
                    np.count_nonzero(isolate>SIGNAL_TOL,axis=(-2,-1))>0)),
                "max_direct_rgba_signal":float(direct.max()),
                "max_isolate_rgba_signal":float(isolate.max()),
                "status":("INDISTINGUISHABLE_ONLY_UNDER_12_TESTED_POSE_VIEWS"
                          if indistinguishable else
                          "DISTINGUISHED_BY_SOURCE_NATIVE_ANIMATED_EDITS"),
                "same_drawable":"UNKNOWN",
                "different_drawable":"UNKNOWN",
                "trainable_group_identity":False,
            })
        task_rows.append({
            "task_index_teacher_only":tid,
            "finest_existing_reference_edit_pixel_instances":reference_effect,
            "finest_existing_isolation_pixel_instances":reference_isolate,
            "pairwise_candidate_comparisons":per_pair,
        })
    pairs=[]
    for a,b in combinations(sizes,2):
        matching=[next(x for x in row["pairwise_candidate_comparisons"]
                       if x["candidate_a"]==a and x["candidate_b"]==b)
                  for row in task_rows]
        invisible=sum(x["status"].startswith("INDISTINGUISHABLE") for x in matching)
        fully=invisible==len(matching)
        pairs.append({
            "candidate_a":a,"candidate_b":b,
            "finite_test_equivalence_class_member":fully,
            "indistinguishable_source_edit_task_count":invisible,
            "total_source_edit_task_count":len(matching),
            "direct_difference_pixel_instances":sum(
                x["additional_direct_pixel_instances"] for x in matching),
            "isolated_difference_pixel_instances":sum(
                x["additional_isolate_pixel_instances"] for x in matching),
            "status":("OBSERVATIONALLY_EQUIVALENT_ON_ALL_TESTED_SOURCE_EDITS"
                      if fully else "SOURCE_EDIT_COUNTEREXAMPLE_OBSERVED"),
            "optimal_group_count_authorized":False,
            "graphics_owner_authorized":False,
            "training_authorized":False,
        })
    return {
        "schema":SCHEMA,
        "status":"REAL_NATIVE_EDIT_RESPONSE_PAIRWISE_EQUIVALENCE_MEASURED__ARTIST_OWNER_UNKNOWN",
        "subject_id":manifest["subject_id"],
        "source_sha256":manifest["source_sha256"],
        "real_native_poses":3,
        "original_source_cameras":4,
        "source_camera_pose_cases":12,
        "source_edit_task_count":len(task_rows),
        "candidate_sizes":sizes,
        "pairwise_equivalence_classes_scoped_to_tested_source_edits":pairs,
        "private_per_task_teacher_only":task_rows,
        "finest_partition_is_artist_truth":False,
        "exact_response_equality_in_unseen_actions_or_cameras":False,
        "semantic_taxonomy_prior_used":False,
        "fixed_group_counts_are_structural_priors":True,
        "artist_same_drawable_positive_pairs":0,
        "artist_different_drawable_negative_pairs":0,
        "graphics_owner_authorized":False,
        "mechanical_owner_authorized":False,
        "native_resolution_caa_authorized":False,
        "optimal_group_count_authorized":False,
        "training_authorized":False,
        "materialization_authorized":False,
    }

def run(args):
    public=json.loads(Path(args.public_v9_result).read_text())
    require(public.get("schema")==PARENT
            and public.get("authority",{}).get("training_authorized") is False
            and public.get("hosted",{}).get("jobs_succeeded")==4,
            "V10_V9_PUBLIC_SEALED_PARENT_REQUIRED")
    sources=[x for x in public["per_subject"]
             if x["subject_id"]==args.subject_id]
    require(len(sources)==1,"V10_ORIGINAL_V9_SUBJECT_REQUIRED")
    row=sources[0]
    require(row.get("render_manifest_sha256")==args.expected_manifest_sha256,
            "V10_PUBLIC_SEALED_RENDER_MANIFEST_SHA_NOT_ARBITRARY")
    manifest=read_checked(args.manifest,row["render_manifest_sha256"],
                          "V10_RENDERMANIFEST_SHA")
    # The independent V9 public seal binds source, render manifest,
    # original RGBA bytes and the separately courted V9 verdict by SHA.
    require(manifest["source_sha256"]==row["source_sha256"]
            and manifest["array_sha256"]==row["private_render_arrays_sha256"],
            "V10_PARENT_MANIFEST_OR_ARRAY_BINDING")
    prior=read_checked(args.prior_v9_court,row["independent_private_court_sha256"],
                       "V10_INDEPENDENT_V9_COURT_SHA")
    require(sha(args.arrays)==row["private_render_arrays_sha256"],
            "V10_SOURCE_RGBA_PARENT_SHA")
    with np.load(args.arrays,allow_pickle=False) as npz:
        require(set(npz.files)=={"animated_original_and_group_edits_rgba"},
                "V10_NO_TRAINER_OR_EXTERNAL_ARRAYS")
        report=evaluate(npz["animated_original_and_group_edits_rgba"],
                        manifest,prior)
    report["parents"]={
        "public_v9_result_sha256":sha(args.public_v9_result),
        "private_v9_render_manifest_sha256":sha(args.manifest),
        "original_v9_rgba_npz_sha256":sha(args.arrays),
        "independent_v9_court_sha256":sha(args.prior_v9_court),
    }
    path=Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({
        "subject_id":args.subject_id,
        "status":report["status"],
        "equivalence_count":sum(x["finite_test_equivalence_class_member"]
                                for x in report["pairwise_equivalence_classes_scoped_to_tested_source_edits"]),
        "same_drawable_labels":0,
        "training_authorized":False,
    },sort_keys=True),flush=True)
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--public-v9-result",type=Path,required=True)
    p.add_argument("--subject-id",required=True)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--expected-manifest-sha256",required=True)
    p.add_argument("--prior-v9-court",type=Path,required=True)
    p.add_argument("--arrays",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    run(p.parse_args())

if __name__=="__main__":main()
