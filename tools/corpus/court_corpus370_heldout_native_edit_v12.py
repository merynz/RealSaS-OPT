#!/usr/bin/env python3
"""V12 independent V11-heldout real-source native edited raster court.

Different image-edit response is not DIFFERENT_DRAWABLE; identical observations
are not SAME_DRAWABLE. Original source/soup RGBA parity gates per pose/camera.
Invisible heldout source probes ABSTAIN rather than endorse equivalence.
"""
from __future__ import annotations
from itertools import combinations
import hashlib
from pathlib import Path
import numpy as np
from tools.corpus.corpus370_partition_edit_plan_v7 import build_plan
from tools.corpus.court_corpus370_native_animated_partition_v9 import signal

SCHEMA="RealSaS.Corpus370HeldoutNativeEditIndependentCourt.v12"
RENDER="RealSaS.Corpus370HeldoutNativeAnimatedSourceRender.v12"
V11="RealSaS.Corpus370GeometryOnlyHeldoutBranchProbePlan.v11"
V10="RealSaS.Corpus370NativeAnimatedEditResponseEquivalenceCourt.v10"
RGBA_TOL=.025
NOOP_TOL=.0025

def require(ok,error):
    if not ok: raise ValueError(error)

def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()

def evaluate(pixels,manifest,proposal,pairs,v11_plan,prior_v10):
    source_plan=build_plan(proposal,pairs)
    sizes=source_plan["source_geometry_candidate_sizes"]
    tasks=v11_plan["query_targets_source_teacher_only"]
    require(manifest.get("schema")==RENDER
            and manifest.get("status")==
                "REAL_V11_HELDOUT_SOURCE_NATIVE_POSE_RGBA_RENDER__INDEPENDENT_V12_COURT_REQUIRED"
            and manifest.get("teacher_only") is True
            and manifest.get("source_rest_exact_geometry_verified") is True
            and manifest.get("source_selected_pose_exact_teacher_replay_verified") is True
            and manifest.get("v11_original_heldout_targets_verified") is True
            and manifest.get("source_sha256")==source_plan["source_sha256"]
            and manifest.get("candidate_sizes")==sizes
            and manifest.get("v11_heldout_plan_sha256") is not None
            and manifest.get("source_group_proposal_sha256") is not None
            and manifest.get("source_pair_sha256") is not None
            and manifest.get("reference_policy")==
                "DIRECT_PAIRWISE_HELDOUT_SOURCE_GEOMETRIC_EDIT_NO_ARTIST_TRUTH"
            and manifest.get("graphics_owner_authorized") is False
            and manifest.get("training_authorized") is False
            and manifest.get("source_bone_names_as_targets") is False
            and manifest.get("animated_render_caa_authorized") is False
            and manifest.get("materialization_authorized") is False
            and v11_plan.get("schema")==V11
            and v11_plan.get("subject_id")==manifest.get("subject_id")
            and v11_plan.get("source_sha256")==manifest.get("source_sha256")
            and v11_plan.get("real_rgba_consequences_measured_for_heldout_queries") is False
            and v11_plan.get("graphics_owner_authorized") is False
            and v11_plan.get("training_authorized") is False
            and v11_plan.get("new_source_face_intervention_query_count")==12
            and prior_v10.get("schema")==V10
            and prior_v10.get("subject_id")==manifest.get("subject_id")
            and prior_v10.get("source_sha256")==manifest.get("source_sha256")
            and prior_v10.get("graphics_owner_authorized") is False
            and prior_v10.get("training_authorized") is False
            and prior_v10.get("artist_same_drawable_positive_pairs")==0
            and prior_v10.get("artist_different_drawable_negative_pairs")==0
            and manifest.get("prior_v10_court_sha256")==v11_plan["parents"]["independent_native_v10_court_sha256"]
            and manifest.get("prior_v9_manifest_sha256")==v11_plan["parents"]["original_native_v9_manifest_sha256"]
            and manifest.get("previous_tested_source_faces_teacher_only")==[
                t["source_face_teacher_only"] for t in source_plan["tasks"]]
            and manifest.get("view_indices")==[0,2,4,6]
            and manifest.get("raster_resolution")==192
            and manifest.get("cycles_samples")==32
            and len(manifest.get("pose_indices",[]))==3
            and len(set(manifest["pose_indices"]))==3,
            "V12_ORIGINAL_SOURCE_HELDOUT_V9_V10_V11_AUTHORITY_BINDINGS")
    labels={c["slot_count"]:c["face_slot_teacher_only"]
            for c in proposal["candidates"] if c["slot_count"] in sizes}
    require(set(labels)==set(sizes) and len(tasks)==12
            and len(set(t["finest_group_id_teacher_only"] for t in tasks))==12,
            "V12_HELDOUT_SOURCE_FACE_PROBE_DOMAIN")
    prior_leaves={int(labels[sizes[-1]][t["source_face_teacher_only"]])
                  for t in source_plan["tasks"]}
    arms=["SOURCE_ORIGINAL","SOUP_FULL","SOUP_NOOP","REMOVE_ALL_GROUPS"]
    for i,t in enumerate(tasks):
        face=t["source_face_teacher_only"]
        require(type(face) is int and 0<=face<source_plan["source_face_count"]
                and t["test_index"]==i and type(i) is int
                and t["finest_group_id_teacher_only"]==labels[sizes[-1]][face]
                and t["finest_group_id_teacher_only"] not in prior_leaves
                and t["same_drawable"]=="UNKNOWN" and t["different_drawable"]=="UNKNOWN"
                and t["training_authorized"] is False
                and all(t["source_groups_per_candidate_size_teacher_only"][str(k)]==labels[k][face] for k in sizes),
                "V12_SOURCE_NEW_FACE_QUERY_COUNTERFEIT_OR_PREVIOUS_PROBE_LEAKAGE")
        for k in sizes:
            arms.extend((f"GROUP_REMOVE_K{k:03d}_T{i:02d}",
                         f"GROUP_ISOLATE_K{k:03d}_T{i:02d}"))
    require(manifest.get("arms")==arms,"V12_EXACT_RENDER_ARM_ORDER_AND_ALL_HELDOUT_TARGETS")
    px=np.asarray(pixels)
    require(px.shape==(3,4,len(arms),192,192,4)
            and px.dtype.kind=="f" and np.isfinite(px).all()
            and np.all(px[...,3]>=-1e-5) and np.all(px[...,3]<=1+1e-5),
            "V12_REAL_SOURCE_RGBA_ARRAY_SHAPE")
    ix={a:i for i,a in enumerate(arms)}
    parity=[]
    qualified=[]
    for p in range(3):
        for v in range(4):
            orig=px[p,v,0]; full=px[p,v,1]; noop=px[p,v,2]
            occupancy=int(np.count_nonzero(orig[...,3]>.05))
            p99=float(np.quantile(signal(orig,full),.99))
            noopdiff=float(signal(noop,full).max())
            accepted=occupancy>0 and p99<=RGBA_TOL and noopdiff<=NOOP_TOL
            parity.append({
                "pose_index_teacher_only":manifest["pose_indices"][p],
                "view_index":manifest["view_indices"][v],
                "source_occupied_pixels":occupancy,
                "source_vs_soup_signal_p99":p99,
                "noop_signal_max":noopdiff,
                "qualified":accepted,
                "reason":None if accepted else "SOURCE_SOUP_CAA_OR_NOOP_OR_OCCUPANCY_FAILED"
            })
            if accepted:qualified.append((p,v))
    pair_results=[]
    task_details=[]
    for a,b in combinations(sizes,2):
        details=[]
        for i in range(12):
            effect_a=effect_b=0
            direct=isolation=0
            discriminating_direct=discriminating_isolation=0
            for p,v in qualified:
                full=px[p,v,ix["SOUP_FULL"]]
                ra=px[p,v,ix[f"GROUP_REMOVE_K{a:03d}_T{i:02d}"]]
                rb=px[p,v,ix[f"GROUP_REMOVE_K{b:03d}_T{i:02d}"]]
                ia=px[p,v,ix[f"GROUP_ISOLATE_K{a:03d}_T{i:02d}"]]
                ib=px[p,v,ix[f"GROUP_ISOLATE_K{b:03d}_T{i:02d}"]]
                mask_a=(signal(ra,full)>RGBA_TOL)|(signal(ia,full)>RGBA_TOL)
                mask_b=(signal(rb,full)>RGBA_TOL)|(signal(ib,full)>RGBA_TOL)
                effect_a+=int(np.count_nonzero(mask_a))
                effect_b+=int(np.count_nonzero(mask_b))
                delta_direct=int(np.count_nonzero(signal(ra,rb)>RGBA_TOL))
                delta_iso=int(np.count_nonzero(signal(ia,ib)>RGBA_TOL))
                direct+=delta_direct;isolation+=delta_iso
                discriminating_direct+=int(delta_direct>0)
                discriminating_isolation+=int(delta_iso>0)
            observable=len(qualified)>0 and (effect_a>0 or effect_b>0)
            status=("NO_QUALIFIED_SOURCE_RENDER_ABSTAIN" if not qualified
                    else "INVISIBLE_BOTH_SOURCE_EDITS_ABSTAIN" if not observable
                    else "SOURCE_HELDOUT_EDIT_COUNTEREXAMPLE" if direct or isolation
                    else "OBSERVATIONALLY_EQUAL_TESTED_NATIVE_SOURCE_EDIT")
            details.append({
                "heldout_query_index_teacher_only":i,
                "status":status,"source_edit_observable":observable,
                "candidate_a_edit_pixel_instances":effect_a,
                "candidate_b_edit_pixel_instances":effect_b,
                "direct_response_difference_pixel_instances":direct,
                "isolated_response_difference_pixel_instances":isolation,
                "discriminating_direct_pose_views":discriminating_direct,
                "discriminating_isolated_pose_views":discriminating_isolation,
                "same_drawable":"UNKNOWN","different_drawable":"UNKNOWN"
            })
        old=next((x for x in prior_v10["pairwise_equivalence_classes_scoped_to_tested_source_edits"]
                  if x["candidate_a"]==a and x["candidate_b"]==b),None)
        require(old is not None and old["total_source_edit_task_count"]==source_plan["task_count"],
                "V12_ORIGINAL_V10_SOURCE_PAIR_NOT_FOUND")
        observed=sum(x["source_edit_observable"] for x in details)
        counters=sum(x["status"]=="SOURCE_HELDOUT_EDIT_COUNTEREXAMPLE" for x in details)
        pair_results.append({
            "candidate_a":a,"candidate_b":b,
            "v10_original_source_hit_indistinguishable_all_tasks":
                old["finite_test_equivalence_class_member"],
            "v10_original_source_hit_tested_task_count":old["total_source_edit_task_count"],
            "heldout_source_query_count":12,
            "qualified_observable_heldout_queries":observed,
            "heldout_queries_with_pairwise_counterexamples":counters,
            "heldout_queries_not_observable":12-observed,
            "direct_difference_pixel_instances":sum(x["direct_response_difference_pixel_instances"] for x in details),
            "isolated_difference_pixel_instances":sum(x["isolated_response_difference_pixel_instances"] for x in details),
            "v10_finite_source_edit_equivalence_falsified_by_new_heldout":
                bool(old["finite_test_equivalence_class_member"] and counters),
            "v10_finite_source_edit_equivalence_still_not_falsified_by_heldout":
                bool(old["finite_test_equivalence_class_member"] and observed and not counters),
            "same_drawable":"UNKNOWN","different_drawable":"UNKNOWN",
            "authority_minted":False,
            "per_query_private_teacher_only":details
        })
    return {
        "schema":SCHEMA,
        "status":("V12_REAL_HELDOUT_NATIVE_RENDER_EDIT_PAIRWISE_COURT_QUALIFIED_NO_OWNER"
                  if len(qualified)==12
                  else "V12_REAL_HELDOUT_SOURCE_PARTIAL_PARITY_EDIT_MASKED_NO_OWNER"),
        "subject_id":manifest["subject_id"],"source_sha256":manifest["source_sha256"],
        "prior_heldout_plan_sha256":manifest["v11_heldout_plan_sha256"],
        "original_real_authored_pose_count":3,
        "original_source_camera_view_count":4,
        "real_native_pose_view_count":12,
        "source_soup_parity_qualified_pose_views":len(qualified),
        "source_soup_parity_abstained_pose_views":12-len(qualified),
        "parity_census_private":parity,
        "candidate_group_counts_are_experimental":sizes,
        "new_untested_original_source_face_targets":12,
        "source_geometry_pairwise_heldout_consequences":pair_results,
        "artist_same_drawable_positive_pairs":0,
        "artist_different_drawable_negative_pairs":0,
        "source_names_as_model_targets":False,
        "finest_group_count_as_author_label":False,
        "fixed_layer_count_authorized":False,
        "native_full_resolution_caa_authorized":False,
        "mechanical_owner_authorized":False,
        "graphics_owner_authorized":False,
        "training_authorized":False,
        "materialization_authorized":False,
    }
