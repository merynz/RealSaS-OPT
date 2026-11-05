#!/usr/bin/env python3
"""V18a independent source-to-stain candidate/heldout court.

Original V13b + V16 source geometry is reopened, and ALL proposed face groups
are checked against the precommitted stain graph. A heldout pose2 native
mechanical response is evaluated independently with bincount / SSE, and
original V17 observable pixel evidence is READ ONLY in court/diagnostics.
No training, artist same/different drawable or correct-layer authority.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from tools.corpus.compile_corpus370_v18a_reactive_staining import (
    SCHEMA as PRODUCER_SCHEMA,STATUS as PRODUCER_STATUS,
    THRESHOLDS,TRAIN_POSES,HELDOUT_POSE,BUDGET,
    require,shape_check,face_edges,component_labels,
    select_blind_control_axes,face_stain_features
)
from tools.corpus.measure_corpus370_v17_posed_source_ray_to_causal_rgba_v1 import (
    validate_and_open
)

SCHEMA="RealSaS.Corpus370IndependentReactiveStainProposalCourt.v18a"
PASS="ORIGINAL_NATIVE_SOURCE_STAIN_GRAPH_REPRODUCED__ARTIST_EDIT_TRUTH_UNKNOWN"


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def independent_heldout_fraction(faces,base,controls,picks,labels,extent):
    """Different evaluator from producer: component bincount weighted means."""
    labels=np.asarray(labels,dtype=np.int64)
    n=int(labels.max(initial=-1))+1
    require(n>0 and len(labels)==len(faces),"V18A_INDEPENDENT_LABEL_SOURCE_SHAPE")
    size=np.bincount(labels,minlength=n)
    require(np.all(size>0),"V18A_EMPTY_STAIN_COMPONENT")
    residual=0.;all_variance=0.
    for ci,axis in picks:
        for si in (0,1):
            velocity=(controls[HELDOUT_POSE,ci,axis,si]-
                      base[HELDOUT_POSE])/extent
            amplitude=np.linalg.norm(velocity,axis=-1)[faces].mean(axis=1)
            total_mean=amplitude.mean()
            all_variance+=float(np.square(amplitude-total_mean).sum())
            mean_by_group=np.bincount(labels,weights=amplitude,minlength=n)/size
            residual+=float(np.square(amplitude-mean_by_group[labels]).sum())
    return float(residual/all_variance) if all_variance>1e-15 else None


def verify_component_connectivity(labels,edges):
    labels=np.asarray(labels,dtype=np.int64)
    edges=np.asarray(edges,dtype=np.int64)
    count=len(labels)
    peers=[[] for _ in range(count)]
    for a,b in edges:
        if labels[a]==labels[b]:
            peers[a].append(int(b))
            peers[b].append(int(a))
    seen=np.zeros(count,bool)
    clusters={}
    for ix in range(count):
        k=int(labels[ix])
        if seen[ix]:continue
        if k in clusters:
            raise ValueError("V18A_DISCONNECTED_SAME_STAIN_LABEL_FALSE_GRAPH_OWNER")
        stack=[ix];seen[ix]=True;n=0
        while stack:
            v=stack.pop();n+=1
            for u in peers[v]:
                if not seen[u]:
                    require(labels[u]==k,
                            "V18A_FACE_GRAPH_CROSS_COMPONENT_EDGE")
                    seen[u]=True;stack.append(u)
        clusters[k]=n
    require(np.all(seen),"V18A_UNREACHED_SOURCE_FACE")
    return clusters


def evaluate(report_path,npz_path,source,v14_public,v16_public,v17_public,
             v14_dir,v14_job,v16_manifest,v16_court,v16_pixels,
             v17_proposal,v17_court):
    proposal=json.loads(Path(report_path).read_text())
    (identity,manifest,job,faces,bundle,oi,ni,singles,
     rgba,joint,v16_rows,parents)=validate_and_open(
        v14_public,v16_public,source,v14_dir,v14_job,
        v16_manifest,v16_court,v16_pixels)
    original=json.loads(Path(v17_proposal).read_text())
    reference=json.loads(Path(v17_court).read_text())
    public=json.loads(Path(v17_public).read_text())
    row=next((z for z in public["source_results"]
              if z["source"]==source),None)
    require(row is not None and sha(v17_court)==row["court_sha256"]
        and reference["source_pose_ray_proposal_sha256"]==sha(v17_proposal)
        and original["parents_sha256"]==parents
        and row["subject_id"]==identity["subject_id"]
        and row["original_source_sha256"]==identity["source_sha256"]
        and reference["graphics_owner_authorized"] is False
        and reference["training_authorized"] is False,
        "V18A_INDEPENDENT_ORIGINAL_V17_PRIVATE_RGB_RAY_COURT")

    require(proposal.get("schema")==PRODUCER_SCHEMA
        and proposal.get("status")==PRODUCER_STATUS
        and proposal.get("subject_id")==identity["subject_id"]
        and proposal.get("source_sha256")==identity["source_sha256"]
        and proposal.get("source_name_teacher_only")==source
        and proposal.get("fit_source_authored_poses")==[0,1]
        and proposal.get("heldout_source_authored_pose")==2
        and proposal.get("stain_control_selection_policy")==
            "FIT_POSE0_GEOMETRY_ONLY_GREEDY_SOURCE_RESPONSE_DIVERSITY"
        and proposal.get("source_appearance_transformed_into_model_input") is False
        and proposal.get("source_3d_rig_bone_or_material_identity_used_as_artist_layer") is False
        and proposal.get("source_control_id_as_model_feature") is False
        and proposal.get("number_of_artist_parts_imposed") is False
        and proposal.get("teacher_control_response_is_product_input") is False
        and proposal.get("heldout_edit_replay_proven") is False
        and proposal.get("full_alpha_draw_order_and_CAA_proven") is False
        and proposal.get("original_artist_same_drawable_labels_minted")==0
        and proposal.get("original_artist_different_drawable_labels_minted")==0
        and proposal.get("source_graphics_owner_authorized") is False
        and proposal.get("iris_training_authorized") is False
        and proposal.get("tessa_training_authorized") is False
        and proposal.get("materialization_authorized") is False
        and proposal.get("training_authorized") is False
        and proposal.get("private_candidate_npz_sha256")==sha(npz_path),
        "V18A_FALSE_PUBLIC_TRAINING_OR_SOURCE_DRAWABLE_LABEL")

    source_parent=proposal["parent_source_immutable_sha256"]
    required={
        **parents,
        "v17_public":sha(v17_public),
        "v17_private_proposal":sha(v17_proposal),
        "v17_independent_court":sha(v17_court),
        "v13b_source_intervention":sha(job["v13b_array"]["path"]),
        "source_rest_geometry":sha(job["rest_geometry"]["path"]),
        "v14_source_control_plan":sha(job["selected_source_controls"]["path"])
    }
    require(source_parent==required
        and sha(job["v13b_array"]["path"])==job["v13b_array"]["sha256"]
        and sha(job["rest_geometry"]["path"])==job["rest_geometry"]["sha256"],
        "V18A_INDEPENDENT_ORIGINAL_V13B_V14_V16_V17_SOURCE_SHA_CHAIN")

    with np.load(job["v13b_array"]["path"],allow_pickle=False) as d:
        base=np.asarray(d["baseline_vertices"],np.float64)
        controls=np.asarray(d["native_control_perturbed_vertices"],np.float64)
    with np.load(job["rest_geometry"]["path"],allow_pickle=False) as d:
        rest=np.asarray(d["rest_vertices_source_world"],np.float64)
        original_faces=np.asarray(d["faces"],np.int64)
    require(np.array_equal(faces,original_faces),
        "V18A_ORIGINAL_FACE_TRIANGULATION_CHANGED")
    shape_check(rest,faces,base,controls,joint)
    extent=max(1e-12,float(np.linalg.norm(np.ptp(rest,axis=0))))

    selected=select_blind_control_axes(base,controls,faces,extent)
    reported=[tuple(x) for x in
              proposal["actual_native_source_axes_probed_teacher_only"]]
    require(reported==selected and len(selected)<=BUDGET
        and proposal["source_control_count_for_staining"]==len(selected),
        "V18A_NATIVE_STAIN_CONTROL_PICK_NOT_SOURCE_ONLY")

    # Read original V14 blind locked A+B action list and check source private
    # mechanism association. Absolutely no author semantic part names.
    v14_plan=json.loads(Path(job["selected_source_controls"]["path"]).read_text())
    from tools.corpus import compile_corpus370_v18a_reactive_staining as algorithm
    algorithm.joint_selection_indices=tuple(
        (r["control_index_teacher_only"],r["axis_index_teacher_only"])
        for r in v14_plan["controls"])
    expected_stain=face_stain_features(
        rest,faces,base,controls,joint,selected)
    expected_edges=face_edges(faces)
    with np.load(npz_path,allow_pickle=False) as f:
        keys=set(f.files)
        expected_keys={"faces","stain_channels","graph_edges",
            "selected_source_control_axes_teacher_only"}|{
            "labels_"+str(t).replace(".","p") for t in THRESHOLDS}
        require(keys==expected_keys,
            "V18A_SOURCE_STAIN_NPZ_UNAUTHORIZED_MEMBERS")
        require(np.array_equal(f["faces"],faces)
            and np.array_equal(
                f["selected_source_control_axes_teacher_only"],
                np.asarray(selected,np.int32))
            and np.array_equal(f["graph_edges"],expected_edges)
            and np.array_equal(f["stain_channels"],expected_stain),
            "V18A_RECOMPUTED_ORIGINAL_SOURCE_STAIN_DIFFERENCE")
        labels_by_t={t:np.asarray(
            f["labels_"+str(t).replace(".","p")],np.int64)
            for t in THRESHOLDS}
    require(proposal["source_face_count"]==len(faces)
        and proposal["true_shared_edge_graph_edges"]==len(expected_edges)
        and proposal["feature_channel_count"]==expected_stain.shape[1]
        and len(proposal["candidate_policies"])==len(THRESHOLDS),
        "V18A_EXACT_NATIVE_STAIN_SOURCE_FACE_AND_CHANNEL_ACCOUNTING")
    result_rows=[]
    for t,supplied in zip(THRESHOLDS,proposal["candidate_policies"]):
        labels=labels_by_t[t]
        independently_derived=component_labels(
            expected_stain,expected_edges,t)
        require(np.array_equal(labels,independently_derived)
            and len(labels)==len(faces)
            and labels.min(initial=0)>=0,
            "V18A_STAIN_CANDIDATE_NOT_ORIGINAL_MECHANICAL_RESPONSE")
        members=verify_component_connectivity(labels,expected_edges)
        sizes=np.bincount(labels)
        independent_error=independent_heldout_fraction(
            faces,base,controls,selected,labels,extent)
        recorded_error=supplied[
            "heldout_authored_pose_2_motion_within_component_variance_fraction"]
        require(supplied["frozen_stain_similarity_threshold"]==t
            and supplied["predicted_connected_source_graphics_layer_candidate_count"]==
                len(members)
            and supplied["smallest_candidate_faces"]==int(sizes.min())
            and supplied["largest_candidate_faces"]==int(sizes.max())
            and supplied["single_face_candidate_count"]==int(np.count_nonzero(sizes==1))
            and supplied["is_original_artist_part_count"] is False
            and supplied["causal_edit_independence_proven"] is False
            and ((independent_error is None and recorded_error is None)
                 or (independent_error is not None and recorded_error is not None
                     and abs(independent_error-recorded_error)<=1e-9)),
            "V18A_INDEPENDENT_HELDOUT_POSE2_SOURCE_MECHANICS_AND_COMPLEXITY")
        result_rows.append({
            "threshold":t,
            "actual_variable_connected_candidate_count":len(members),
            "heldout_pose2_independent_source_response_error":independent_error,
            "singleton_candidate_count":int(np.count_nonzero(sizes==1)),
            "mean_faces_per_candidate":float(len(faces)/len(members)),
            "original_art_drawable_count_proven":False
        })
    v17_rows=[r for r in original["source_ray_rows_private_only"]
              if r["source_pose_ordinal"]==2]
    qualified=[r for r in v17_rows
               if r["geometric_stack_complete"] and
               not r["geometric_stack_has_depth_tie"] and
               len(r["source_geometric_all_hits_ordered_near_to_far"])>0]
    require(len(qualified)==proposal["independent_heldout_v17_geometric_ray_first_hit_cases"]
        and len(v17_rows)-len(qualified)==proposal[
            "independent_heldout_v17_unknown_ray_cases"]
        and sum(bool(r["real_original_source_joint_ab_visible_rgba_effect"])
                for r in qualified)==proposal[
                "independent_heldout_v17_visible_effect_pixel_cases"],
        "V18A_HELDOUT_ORIGINAL_RGBA_WAS_TAMPERED_AFTER_STAINING")
    return {
        "schema":SCHEMA,
        "status":PASS,
        "source_name_teacher_only":source,
        "subject_id":identity["subject_id"],
        "original_source_sha256":identity["source_sha256"],
        "private_stain_report_sha256":sha(report_path),
        "private_stain_arrays_sha256":sha(npz_path),
        "private_stain_diagnostic_png_sha256":
            proposal["private_rgb_stain_diagnostic_sha256"],
        "source_native_motion_channels_independently_replayed":len(selected),
        "source_face_count_independently_verified":len(faces),
        "fit_poses_without_hidden_original_edit_outcomes":[0,1],
        "heldout_pose_2_only_in_independent_evaluation":True,
        "independently_qualified_original_V17_heldout_pose2_rays":
            len(qualified),
        "stain_variable_candidate_results":result_rows,
        "original_source_rgba_and_edit_court_authority":"NOT_YET_MEASURED_FOR_NEW_CANDIDATES",
        "artist_same_drawable_labels_minted":0,
        "artist_different_drawable_labels_minted":0,
        "author_drawable_count_qualified":False,
        "actual_editor_action_independence_proven":False,
        "graphics_owner_authorized":False,
        "iris_training_authorized":False,
        "tessa_training_authorized":False,
        "materialization_authorized":False,
        "training_authorized":False
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True,
                   choices=("CAT","TRICERATOPS","COWFISH"))
    for k in ("report","candidate_npz","v14_public","v16_public",
              "v17_public","v14_dir","v14_job","v16_manifest",
              "v16_court","v16_pixels","v17_proposal","v17_court","output"):
        p.add_argument("--"+k.replace("_","-"),required=True,type=Path)
    a=p.parse_args()
    result=evaluate(a.report,a.candidate_npz,a.source,a.v14_public,
        a.v16_public,a.v17_public,a.v14_dir,a.v14_job,a.v16_manifest,
        a.v16_court,a.v16_pixels,a.v17_proposal,a.v17_court)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True,
                                   allow_nan=False)+"\n")
    print(json.dumps({
        "source":a.source,
        "status":PASS,
        "candidates":[
            (x["threshold"],x["actual_variable_connected_candidate_count"],
             x["heldout_pose2_independent_source_response_error"])
            for x in result["stain_variable_candidate_results"]],
        "training_authorized":False
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
