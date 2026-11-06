#!/usr/bin/env python3
"""V18o independent SOURCE-CANDIDATE viability and heldout operational evidence.

NO segmentation learned/fit here; reads SHA-sealed V18a FACE labels for all
precommitted thresholds, and original V18n physically edited material Cycles
samples for only its frozen .10 V11 source edit locus. Measures concentration,
fragmentation and observable action consequences, never artist ownership gold.

A partition with many triangles but only K=1, 300 tiny components, or
one 97% component can compose perfectly as rendered source images while
failing to represent useful graphics edit units. This court says so, without
making another generic model label from privileged source teacher IDs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.corpus.compile_corpus370_v18a_reactive_staining import (
    SCHEMA as STAIN_SCHEMA,STATUS as STAIN_STATUS,THRESHOLDS
)
from tools.corpus.court_corpus370_v18n_original_material_sample_replay import (
    MANIFEST as MATERIAL_SCHEMA,RENDER_STATUS as MATERIAL_STATUS,
    SCHEMA as MATERIAL_COURT_SCHEMA,STATUS as MATERIAL_COURT_STATUS,
    premul,TOL,CAMERAS,INDICES
)

SCHEMA="RealSaS.Corpus370IndependentAnonymousPartitionViabilityCourt.v18o"
STATUS="ORIGINAL_SOURCE_PARTITION_COMPLEXITY_AND_EDIT_CONSEQUENCES_COURTED__OWNER_UNKNOWN"
THRESHOLDS_FROZEN=(.045,.1,.18)
CANDIDATE_GROUP_T=.1
ARMS=("SOURCE_NATIVE","SOUP_FULL","SOUP_ISOLATE","SOUP_REMOVE")
NO_AUTH=("graphics_owner_authorized","iris_training_authorized",
         "tessa_training_authorized","materialization_authorized",
         "training_authorized")


def require(ok,code):
    if not ok:raise ValueError(code)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def partition_profile(labels,faces):
    """Pure source geometry+face index measurement; NO per-character settings."""
    labels=np.asarray(labels,dtype=np.int64)
    faces=np.asarray(faces,dtype=np.int64)
    require(labels.ndim==1 and faces.ndim==2 and faces.shape[1]==3
        and len(labels)==len(faces) and len(labels)>1
        and np.isfinite(labels).all() and labels.min()>=0
        and np.unique(labels).size>0,
        "V18O_FROZEN_SOURCE_FACE_LABELS_NOT_VALID")
    sizes=np.bincount(labels).astype(np.int64)
    sizes=sizes[sizes>0]
    count=int(len(sizes)); n=int(len(labels))
    require(sizes.sum()==n,"V18O_SOURCE_FACE_PARTITION_CENSUS_CHANGED")
    weights=sizes/n
    max_fraction=float(weights.max())
    singleton=int(np.count_nonzero(sizes==1))
    entropy=float(np.exp(-np.sum(weights*np.log(weights))))
    effective=float(1./np.sum(weights**2))
    # Warnings are diagnostic, not learned artists' desired part count.
    # Very large components and tiny components can BOTH coexist.
    return {
       "source_face_count":n,
       "candidate_component_count":count,
       "largest_component_faces":int(sizes.max()),
       "largest_component_face_fraction":max_fraction,
       "smallest_component_faces":int(sizes.min()),
       "singleton_component_count":singleton,
       "singleton_component_fraction":float(singleton/count),
       "singletons_source_face_fraction":float(singleton/n),
       "largest_component_has_at_least_90pct_source_faces":bool(max_fraction>=.9),
       "partition_has_at_least_64_components":bool(count>=64),
       "components_to_original_faces_ratio":float(count/n),
       "candidate_component_count_over_25pct_of_face_count":bool(count/n>=.25),
       "nontrivial_one_to_eighty_pct_face_mass_component_count":
           int(np.count_nonzero((weights>=.01)&(weights<=.8))),
       "effective_component_count_by_simpson_inverse":effective,
       "effective_component_count_by_shannon_exp":entropy,
       "source_candidate_trivially_k1":bool(count==1),
       "source_candidate_display_part_names_or_artist_gold":False,
       "source_face_count_is_not_screen_area":True,
    }


def edit_consequences(material_rgba, selected_component_size, total_faces):
    """ONLY selected SOURCE V18n physical edit, not ALL candidate identities."""
    a=np.asarray(material_rgba,dtype=np.float64)
    require(a.shape==(len(CAMERAS),len(INDICES),len(ARMS),192,192,4)
        and np.isfinite(a).all()
        and np.min(a[...,3])>=-1e-5 and np.max(a[...,3])<=1.+1e-5
        and 0<selected_component_size<total_faces,
        "V18O_ORIGINAL_MATERIAL_EDIT_RGBA_SOURCE_CENSUS")
    p=premul(a)
    native=p[:,:,0]
    full=p[:,:,1]
    isolate=p[:,:,2]
    remove=p[:,:,3]
    def mismatch(x,y):
        return np.max(np.abs(x-y),axis=-1)>TOL
    orig_soup=mismatch(native,full)
    removal=mismatch(full,remove)
    isolation=mismatch(full,isolate)
    # Deleting an entire candidate alters *many* source image pixels,
    # but does not imply that the selected candidate is a good drawable.
    rows=[]
    for i,camera in enumerate(CAMERAS):
        pixel_rem=np.any(removal[i],axis=0)
        pixel_iso=np.any(isolation[i],axis=0)
        rows.append({
          "source_camera":int(camera),
          "source_camera_partition":
             "LOCKED_EVALUATION" if camera in (4,6) else "EXPLORATORY",
          "original_source_native_to_soup_sample_failures":
              int(np.count_nonzero(orig_soup[i])),
          "original_source_group_removal_observable_sample_instances":
              int(np.count_nonzero(removal[i])),
          "original_source_group_removal_observable_screen_pixels":
              int(np.count_nonzero(pixel_rem)),
          "original_source_group_isolation_observable_sample_instances":
              int(np.count_nonzero(isolation[i])),
          "original_source_group_isolation_observable_screen_pixels":
              int(np.count_nonzero(pixel_iso)),
          "source_opacity_nonbinary_sample_instances":
              int(np.count_nonzero((a[i,:,0,:,:,3]>1e-6)&
                   (a[i,:,0,:,:,3]<1-1e-6))),
        })
    return {
      "source_selected_candidate_group_size_faces":int(selected_component_size),
      "source_selected_candidate_fraction_of_all_faces":
          float(selected_component_size/total_faces),
      "selected_candidate_original_heldout_physical_edit_only":True,
      "original_source_heldout_pose_index":2,
      "source_selected_candidate_observable_removal_across_any_view":
          bool(sum(row["original_source_group_removal_observable_sample_instances"]
                   for row in rows)>0),
      "per_original_source_camera":rows,
      "all_original_source_camera_native_soup_sample_failures":
          sum(row["original_source_native_to_soup_sample_failures"] for row in rows),
      "source_selected_group_removal_sample_effects_total":
          sum(row["original_source_group_removal_observable_sample_instances"] for row in rows),
      "source_selected_group_isolation_sample_effects_total":
          sum(row["original_source_group_isolation_observable_sample_instances"] for row in rows),
      "source_material_alpha_nonbinary_samples_across_source_cameras":
          sum(row["source_opacity_nonbinary_sample_instances"] for row in rows),
      "remaining_component_heldout_edit_independence":"UNKNOWN",
      "source_selected_component_artist_edit_intent":"UNKNOWN",
      "candidate_group_membership_is_source_teacher_only":True,
    }


def audit(stain_manifest,stain_npz,stain_court,
          v18n_manifest,v18n_npz,v18n_court,
          expected_stain_court_sha,expected_v18n_court_sha):
    require(THRESHOLDS==THRESHOLDS_FROZEN,
        "V18O_V18A_ORIGINAL_FROZEN_SOURCE_THRESHOLDS_CHANGED")
    require(all(isinstance(v,str) and len(v)==64 for v in (
        expected_stain_court_sha,expected_v18n_court_sha))
        and sha(stain_court)==expected_stain_court_sha
        and sha(v18n_court)==expected_v18n_court_sha,
        "V18O_EXACT_SOURCE_PARENT_COURTS_SHA_REQUIRED")
    s=load(stain_manifest)
    sc=load(stain_court)
    m=load(v18n_manifest)
    mc=load(v18n_court)
    require(s.get("schema")==STAIN_SCHEMA
        and s.get("status")==STAIN_STATUS
        and s.get("private_candidate_npz_sha256")==sha(stain_npz)
        and sc.get("schema")==
           "RealSaS.Corpus370IndependentReactiveStainProposalCourt.v18a"
        and sc.get("status")==
           "ORIGINAL_NATIVE_SOURCE_STAIN_GRAPH_REPRODUCED__ARTIST_EDIT_TRUTH_UNKNOWN"
        and sc.get("private_stain_report_sha256")==sha(stain_manifest)
        and sc.get("private_stain_arrays_sha256")==sha(stain_npz)
        and s.get("source_sha256")==sc.get("original_source_sha256")
        and s.get("subject_id")==sc.get("subject_id")
        and m.get("schema")==MATERIAL_SCHEMA
        and m.get("status")==MATERIAL_STATUS
        and m.get("source_original_v18n_parent_v18m_run")=="42135271057-1"
        and m.get("source_original_v18n_parent_v18l_run")=="42131579613-1"
        and m.get("source_reactive_stain_report_sha256")==sha(stain_manifest)
        and m.get("source_reactive_stain_npz_sha256")==sha(stain_npz)
        and m.get("independent_source_stain_court_sha256")==sha(stain_court)
        and m.get("original_source_sha256")==s.get("source_sha256")
        and m.get("subject_id")==s.get("subject_id")
        and m.get("new_source_actual_edited_rgba_npz_sha256")==sha(v18n_npz)
        and m.get("source_edit_arms")==list(ARMS)
        and m.get("heldout_source_camera_views")==list(CAMERAS)
        and m.get("original_source_cycles_sample_indices")==list(INDICES)
        and m.get("frozen_tested_stain_similarity_thresholds")==[.1]
        and mc.get("schema")==MATERIAL_COURT_SCHEMA
        and mc.get("status")==MATERIAL_COURT_STATUS
        and mc.get("source_original_material_render_manifest_sha256")==sha(v18n_manifest)
        and mc.get("source_original_material_rgba_sha256")==sha(v18n_npz)
        and mc.get("subject_id_private")==s.get("subject_id")
        and mc.get("source_sha256")==s.get("source_sha256")
        and all(m.get(k) is False for k in NO_AUTH)
        and all(mc.get(k) is False for k in NO_AUTH)
        and all(s.get(k if k!="graphics_owner_authorized" else
                     "source_graphics_owner_authorized") is False
                for k in NO_AUTH),
        "V18O_FROZEN_SOURCE_PRETRUTH_MANIFEST_AND_COURT_PROVENANCE")
    with np.load(stain_npz,allow_pickle=False) as data:
        required={"faces","stain_channels","graph_edges",
                  "selected_source_control_axes_teacher_only"} | {
                      "labels_"+str(t).replace(".","p") for t in THRESHOLDS}
        require(set(data.files)==required,
                "V18O_ORIGINAL_V18A_SOURCE_FACE_ARRAY_MEMBERS")
        faces=np.asarray(data["faces"],np.int64)
        labels={t:np.asarray(data["labels_"+str(t).replace(".","p")],np.int64)
                for t in THRESHOLDS}
    with np.load(v18n_npz,allow_pickle=False) as data:
        require(data.files==["source_material_cycles_sample_rgba"],
                "V18O_ORIGINAL_SOURCE_MATERIAL_RGBA_ARRAY_KEY")
        material=np.asarray(data["source_material_cycles_sample_rgba"],np.float32)
    result=[]
    for row,t in zip(s["candidate_policies"],THRESHOLDS):
        require(row["frozen_stain_similarity_threshold"]==t,
                "V18O_ORIGINAL_V18A_SOURCE_THRESHOLDS_OR_ORDER_CHANGED")
        p=partition_profile(labels[t],faces)
        require(row["predicted_connected_source_graphics_layer_candidate_count"]==
                  p["candidate_component_count"]
            and row["largest_candidate_faces"]==p["largest_component_faces"]
            and row["smallest_candidate_faces"]==p["smallest_component_faces"]
            and row["single_face_candidate_count"]==p["singleton_component_count"],
            "V18O_SOURCE_CANDIDATE_INDEPENDENT_FACE_CENSUS_NOT_REPRODUCED")
        p["frozen_original_source_threshold"]=t
        p["candidate_would_need_independent_edit_effects_for_all_components"]=True
        p["independently_edited_candidate_components_at_this_threshold"]=(
            1 if t==CANDIDATE_GROUP_T else 0)
        p["source_level_false_ownership_promotion_prevented"]=True
        result.append(p)
    selected=labels[CANDIDATE_GROUP_T]
    face_ordinal=m.get("source_face_probe_target_ordinals_teacher_only")
    require(isinstance(face_ordinal,list) and len(face_ordinal)==1
        and type(face_ordinal[0]) is int and 0<=face_ordinal[0]<len(faces),
        "V18O_ORIGINAL_HELDOUT_BLIND_SOURCE_EDIT_FACE_TARGET_MISSING")
    group_size=int(np.count_nonzero(selected==selected[face_ordinal[0]]))
    require(group_size==m.get("v18l_source_nontrivial_group_face_count")
        and len(faces)==m.get("v18l_original_source_face_count"),
        "V18O_FROZEN_GROUP_SOURCE_AREA_AND_FACE_ID_MISMATCH")
    observed=edit_consequences(material,group_size,len(faces))
    require(observed["all_original_source_camera_native_soup_sample_failures"]==
        sum(row["original_source_material_native_vs_soup_sample_failures_full_domain"]
            for row in mc["original_source_camera_reports"]),
        "V18O_SOURCE_MATERIAL_COURT_NATIVE_SOUP_NOT_REPRODUCED")
    return {
       "schema":SCHEMA,"status":STATUS,
       "original_source_id_private":s["subject_id"],
       "original_source_sha256":s["source_sha256"],
       "exact_v18a_stain_report_sha256":sha(stain_manifest),
       "exact_v18a_independent_court_sha256":sha(stain_court),
       "exact_v18a_label_array_sha256":sha(stain_npz),
       "exact_v18n_material_render_manifest_sha256":sha(v18n_manifest),
       "exact_v18n_material_rgba_sha256":sha(v18n_npz),
       "exact_v18n_independent_material_court_sha256":sha(v18n_court),
       "all_original_source_frozen_thresholds":result,
       "original_heldout_real_source_material_edit_locus":observed,
       "source_candidate_count_is_not_artist_drawable_count":True,
       "original_source_single_edit_is_only_proxy_not_all_component_validation":True,
       "source_nonbinary_alpha_fidelity":"UNTESTED_IF_NO_NONBINARY_OBSERVED",
       "original_source_varK_compositing_render_admitted":False,
       "same_drawable_truth_minted":0,
       "different_drawable_truth_minted":0,
       "artist_graphics_ownership":"UNKNOWN",
       "iris_training_authorized":False,
       "tessa_training_authorized":False,
       "graphics_owner_authorized":False,
       "materialization_authorized":False,
       "training_authorized":False,
    }


def main():
    a=argparse.ArgumentParser()
    for k in ("stain_manifest","stain_npz","stain_court",
              "v18n_manifest","v18n_npz","v18n_court","output"):
        a.add_argument("--"+k.replace("_","-"),type=Path,required=True)
    a.add_argument("--expected-stain-court-sha",required=True)
    a.add_argument("--expected-v18n-court-sha",required=True)
    v=a.parse_args()
    out=audit(v.stain_manifest,v.stain_npz,v.stain_court,
        v.v18n_manifest,v.v18n_npz,v.v18n_court,
        v.expected_stain_court_sha,v.expected_v18n_court_sha)
    v.output.parent.mkdir(parents=True,exist_ok=True)
    v.output.write_text(json.dumps(out,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({
      "status":out["status"],"subject":out["original_source_id_private"],
      "thresholds":out["all_original_source_frozen_thresholds"],
      "heldout_source_edit":out["original_heldout_real_source_material_edit_locus"],
      "training_authorized":False,
    },sort_keys=True),flush=True)


if __name__=="__main__":main()
