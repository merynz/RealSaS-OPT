#!/usr/bin/env python3
"""Independent controlled-emission/lighting-coupling falsification for V18g.

Reopen exact original V18h Cycles rendered source-soup 2-color emission frames
and strict SHA-pinned original V18f/V18g courts. No source artist labels,
semantics, student model targets, or drawable owner training grants.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.corpus.compile_corpus370_v18a_reactive_staining import THRESHOLDS
from tools.corpus.court_corpus370_v18g_operational_compositing_diagnostics import (
    compositor_probe, TOL, VIEWS,
)

RENDER="RealSaS.Corpus370OriginalEmissiveComplementRender.v18h"
SCHEMA="RealSaS.Corpus370IndependentEmissiveComplementCourt.v18h"
STATUS="ORIGINAL_SOURCE_EMISSIVE_COMPOSITING_CONTRAST_COURTED__NO_DRAWABLE_AUTHORITY"
CANDIDATE="RealSaS.Corpus370NativeReactiveStainCandidate.v18a"
V18F="RealSaS.Corpus370IndependentFullPhysicalFaceEditCourt.v18f"
V18G="RealSaS.Corpus370IndependentOriginalPhysicalCompositingDiagnostics.v18g"
RGB_MEMBER="emissive_complement_rgba"


def require(value,reason):
    if not value:raise ValueError(reason)


def sha(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file,"sha256").hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def arms():
    return ["EMISSIVE_EMPTY"]+[
        f"EMISSIVE_{str(t).replace('.', 'p')}_{mode}"
        for t in THRESHOLDS for mode in ("FULL","ISOLATE","REMOVE")
    ]


def measure(manifest_file,rgba_file,stain_npz_file,stain_court_file,
            parent_f_file,parent_g_file,expected_f_sha,expected_g_sha):
    require(len(expected_f_sha)==64 and len(expected_g_sha)==64
            and sha(parent_f_file)==expected_f_sha
            and sha(parent_g_file)==expected_g_sha,
            "V18H_INDEPENDENT_PARENT_COURT_SHA_REQUIRED")
    render=load(manifest_file)
    f=load(parent_f_file)
    g=load(parent_g_file)
    stain=load(stain_court_file)
    expected=arms()
    require(render.get("schema")==RENDER
            and render.get("status")==
                "V18H_ORIGINAL_EMISSIVE_COMPLEMENT_RENDERED__INDEPENDENT_COURT_REQUIRED"
            and f.get("schema")==V18F
            and f.get("status")==
                "ORIGINAL_FULL_PHYSICAL_EDITS_SOURCE_CONSEQUENCES_COURTED__ARTIST_DRAWABLE_UNKNOWN"
            and g.get("schema")==V18G
            and g.get("status")==
                "ORIGINAL_SOURCE_PHYSICAL_LAYER_COMPLEMENT_ALPHA_COMPOSING_DIAGNOSED__NO_OWNER_TRUTH"
            and stain.get("schema")==
                "RealSaS.Corpus370IndependentReactiveStainProposalCourt.v18a"
            and stain.get("status")==
                "ORIGINAL_NATIVE_SOURCE_STAIN_GRAPH_REPRODUCED__ARTIST_EDIT_TRUTH_UNKNOWN"
            and render.get("source_edit_arms")==expected
            and render.get("actual_render_arms_per_camera")==len(expected)
            and render.get("heldout_original_authored_pose_ordinal")==2
            and render.get("heldout_source_camera_views")==list(VIEWS)
            and render.get("frozen_tested_stain_similarity_thresholds")==list(THRESHOLDS)
            and render.get("precommitted_v11_target_query_indices")==[0]
            and len(render.get("source_face_probe_target_ordinals_teacher_only",[]))==1
            and render.get("source_material_policy")==
                "FIXED_TWO_COLOR_OPAQUE_EMISSION_NO_MATERIAL_SHADING"
            and render.get("source_material_label_authority")=="NONE"
            and render.get("physical_face_visibility_operator")==
                "ACTUAL_TRIANGLE_REMOVAL_AND_OBJECT_RENDER_HIDE"
            and render.get("original_UV_material_textures_deliberately_not_tested_under_emission") is True
            and render.get("source_rest_triangle_UV_material_authority_reused") is False
            and render.get("source_vs_original_v14_RGBA_reference_source_parity_verified") is False
            and render.get("source_v14_parity_authority_inherited_only_from_sealed_v18f_parent") is True
            and render.get("parent_v18f_independent_court_sha256")==expected_f_sha
            and render.get("parent_v18g_independent_compositing_sha256")==expected_g_sha
            and render.get("independent_source_stain_court_sha256")==sha(stain_court_file)
            and render.get("source_reactive_stain_npz_sha256")==sha(stain_npz_file)
            and render.get("new_source_actual_edited_rgba_npz_sha256")==sha(rgba_file)
            and render.get("diagnostic_emission_color_group_rgb")==[.90,.08,.035]
            and render.get("diagnostic_emission_color_complement_rgb")==[.04,.14,.92]
            and render.get("no_artist_layer_labels_minted") is True
            and render.get("physical_removal_not_author_drawable_layer") is True
            and render.get("artist_original_source_part_names_as_labels") is False
            and render.get("source_face_single_triangle_is_artist_gold") is False
            and render.get("source_editor_edit_independence_proven") is False
            and render.get("full_original_caa_draw_order_qualified") is False
            and render.get("subject_id")==f.get("original_source_id")
                ==g.get("source_subject_private")
            and render.get("original_source_sha256")==f.get("source_sha256")
                ==g.get("original_source_sha256")
                ==stain.get("original_source_sha256")
            and f.get("physical_render_face_counts_courted") is True
            and f.get("heldout_original_source_authored_pose_ordinal")==2
            and g.get("original_v18f_run")=="42114401233-1"
            and g.get("exact_v18f_independent_court_sha256")==expected_f_sha
            and f.get("artist_same_drawable_positive_pairs")==0
            and f.get("artist_different_drawable_negative_pairs")==0
            and all(render.get(k) is False for k in (
                "graphics_owner_authorized","iris_training_authorized",
                "tessa_training_authorized","materialization_authorized",
                "training_authorized"))
            and all(f.get(k) is False for k in (
                "graphics_owner_authorized","iris_training_authorized",
                "tessa_training_authorized","materialization_authorized",
                "training_authorized"))
            and all(g.get(k) is False for k in (
                "graphics_owner_authorized","iris_training_authorized",
                "tessa_training_authorized","materialization_authorized",
                "training_authorized")),
            "V18H_SEALED_SOURCE_OR_COURT_OR_AUTHORITY_MISMATCH")
    with np.load(rgba_file,allow_pickle=False) as a:
        require(a.files==[RGB_MEMBER],"V18H_UNEXPECTED_RGBA_NPZ_MEMBERS")
        images=np.asarray(a[RGB_MEMBER],np.float32)
    with np.load(stain_npz_file,allow_pickle=False) as a:
        require(set(a.files)=={"faces","stain_channels","graph_edges",
            "selected_source_control_axes_teacher_only"}|
            {"labels_"+str(t).replace(".","p") for t in THRESHOLDS},
            "V18H_SOURCE_V18A_FACE_MEMBERS_CHANGED")
        face_count=len(a["faces"])
        label_sets={t:np.asarray(a["labels_"+str(t).replace(".","p")],
                                 np.int32) for t in THRESHOLDS}
    require(images.shape==(len(VIEWS),len(expected),192,192,4)
            and np.isfinite(images).all()
            and np.min(images[...,3])>=-1e-5 and np.max(images[...,3])<=1.00001
            and face_count==f["source_frozen_faces"]
            and all(x.shape==(face_count,) and np.all(x>=0) for x in label_sets.values()),
            "V18H_REAL_RGBA_SOURCE_FACE_SHAPE")
    group_face=render["source_face_probe_target_ordinals_teacher_only"][0]
    require(type(group_face) is int and 0<=group_face<face_count,
            "V18H_ORIGINAL_FROZEN_FACE_TARGET")
    parent_rows={
        (int(x["view"]),int(x["v11_target_ordinal"]),float(x["frozen_threshold"])):x
        for x in g["per_original_source_target_view_threshold"]
    }
    require(len(parent_rows)==72,"V18H_PARENT_COMPOSITING_ROWS_MISSING")
    count_rows=render["actual_visible_original_source_triangles_per_arm_view"]
    require(len(count_rows)==len(VIEWS)
            and all(len(x)==len(expected) for x in count_rows),
            "V18H_PHYSICAL_SOURCE_FACE_CENSUS_BAD")
    entries=[]
    totals={}
    for ti,t in enumerate(THRESHOLDS):
        name=str(t).replace(".","p")
        labels=label_sets[t]
        g_id=int(labels[group_face])
        group_count=int(np.count_nonzero(labels==g_id))
        require(group_count>0,"V18H_ORIGINAL_GROUP_EMPTY")
        cases=[]
        for vi,view in enumerate(VIEWS):
            counts=count_rows[vi]
            require(counts[0]==0
                    and counts[1+3*ti:4+3*ti]==[
                      face_count,group_count,face_count-group_count],
                    "V18H_PHYSICAL_COMPLEMENT_FACE_COUNT_DRIFT")
            empty=images[vi,0]
            require(not np.any(empty[...,3]>TOL),"V18H_EMISSION_EMPTY_HAS_ALPHA")
            full=images[vi,1+3*ti]
            group=images[vi,2+3*ti]
            remainder=images[vi,3+3*ti]
            now=compositor_probe(full,group,remainder)
            prior=parent_rows[(int(view),0,float(t))]["compositor"]
            require(parent_rows[(int(view),0,float(t))]["group_source_face_count"]
                    ==group_count,
                    "V18H_PARENT_V18G_GROUP_FACE_COUNT_MISMATCH")
            cases.append({
                "camera_view":int(view),
                "source_group_faces":group_count,
                "emissive":now,
                "original_lit_v18g":prior,
                "oracle_error_pixel_difference_emissive_minus_lit":
                    now["per_pixel_order_oracle"]["pixel_count_above_original_tolerance"]
                    -prior["per_pixel_order_oracle"]["pixel_count_above_original_tolerance"],
                "emission_means_shading_only_is_not_proved":True,
            })
            entries.append(cases[-1]|{"threshold":float(t)})
        totals[str(t)]={
            "source_group_count":int(np.unique(labels).size),
            "examined_group_source_face_count":group_count,
            "nontrivial_group_under_exact_source_geometry":
                 bool(0<group_count<face_count),
            "real_emissive_group_view_cases":len(cases),
            "original_lit_oracle_failed_pixel_instances":sum(
                z["original_lit_v18g"]["per_pixel_order_oracle"]
                  ["pixel_count_above_original_tolerance"] for z in cases),
            "emissive_oracle_failed_pixel_instances":sum(
                z["emissive"]["per_pixel_order_oracle"]
                  ["pixel_count_above_original_tolerance"] for z in cases),
            "emissive_order_discriminating_overlap_pixel_instances":sum(
                z["emissive"]["order_discriminating_overlap_pixels"] for z in cases),
            "emissive_neither_order_works_on_informative_pixels":sum(
                z["emissive"]["informative_pixels_both_orders_fail"] for z in cases),
            "negative_artist_drawable_truth_created":False,
        }
    require(len(entries)==len(VIEWS)*len(THRESHOLDS),
            "V18H_EXACT_CONTROLLED_SOURCE_COMPARISON_COUNT")
    return {
        "schema":SCHEMA,"status":STATUS,
        "original_source_sha256":render["original_source_sha256"],
        "source_subject_id_private":render["subject_id"],
        "original_v18f_run":"42114401233-1",
        "original_v18g_run":"42116712393-1",
        "exact_source_v18h_render_manifest_sha256":sha(manifest_file),
        "exact_source_v18h_render_rgba_npz_sha256":sha(rgba_file),
        "exact_v18f_original_independent_court_sha256":sha(parent_f_file),
        "exact_v18g_original_independent_court_sha256":sha(parent_g_file),
        "exact_v18a_original_source_stain_npz_sha256":sha(stain_npz_file),
        "source_original_heldout_camera_views":list(VIEWS),
        "per_threshold":totals,
        "per_camera_threshold":entries,
        "source_graphic_appearance_truth_evaluated":False,
        "fixed_two_color_emission_only_diagnostic":True,
        "no_artist_semantic_or_original_layer_labels":True,
        "no_new_student_training_authority":True,
        "graphics_owner_authorized":False,
        "iris_training_authorized":False,
        "tessa_training_authorized":False,
        "materialization_authorized":False,
        "training_authorized":False,
    }


def main():
    p=argparse.ArgumentParser()
    for k in ("render_manifest","render_npz","stain_npz",
              "stain_court","parent_v18f_court","parent_v18g_court","output"):
        p.add_argument("--"+k.replace("_","-"),required=True,type=Path)
    for k in ("expected_v18f_sha","expected_v18g_sha"):
        p.add_argument("--"+k.replace("_","-"),required=True)
    a=p.parse_args()
    result=measure(a.render_manifest,a.render_npz,a.stain_npz,
                   a.stain_court,a.parent_v18f_court,a.parent_v18g_court,
                   a.expected_v18f_sha,a.expected_v18g_sha)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True,
                                   allow_nan=False)+"\n")
    print(json.dumps({
        "status":result["status"],
        "parent_v18f_run":result["original_v18f_run"],
        "parent_v18g_run":result["original_v18g_run"],
        "per_threshold":result["per_threshold"],
        "training_authorized":False,
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
