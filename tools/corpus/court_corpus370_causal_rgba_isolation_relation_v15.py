#!/usr/bin/env python3
"""V15 genuinely original-source causal mechanical -> alpha-aware RGBA support.

Independent *counterfactual effect* readback, not artist drawable ownership.
No new source art, no refit and no category, bone/material/body label as target.
Input: immutable V14 real independent PASS court, 3x4x19 original source/soup
pixel observations, native source manifests and their private SHA seals.
Compare the whole-scene original source +/- control effect with only the
active causal source-face isolation +/- effect over identical pose/view.
The latter is a teacher instrumentation effect, not an author drawable.

An isolation-only effect is a HIDDEN_OR_COMPOSITING_CONTEXT_CANDIDATE, not
independent proof of physical z-occlusion. A whole-only difference is a
CONTEXTUAL_COMPOSITING_DELTA, not independent source art. No model admission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from tools.corpus.corpus370_native_raster_join_plan_v14 import arms_for_plan

SCHEMA="RealSaS.Corpus370IndependentCausalRGBAOverlapCourt.v15"
PARENT_SCHEMA="RealSaS.Corpus370V14OriginalNativeMechanicalRGBAIndependentCourtRealResult.v1"
PARENT_STATUS="REAL_ORIGINAL_SOURCE_NATIVE_CONTROL_TO_RGBA_CAUSAL_REPAIR_3_OF_3_COURTS_PASS__DRAWABLE_OWNER_UNKNOWN"
V14_COURT="RealSaS.Corpus370SameNativeControlGeometricRasterCourt.v14"
V14_PASS="V14_REAL_CAUSAL_SOURCE_MECHANICAL_TO_RGBA_JOIN_QUALIFIED__OWNER_UNKNOWN"
V14_RENDER="RealSaS.Corpus370SourceNativeMechanicalRasterJointRender.v14"
V14_BATCH="RealSaS.Corpus370MechanicalRasterJointPrivateBatch.v14"
TOL=.025


def require(ok,code):
    if not ok:raise ValueError(code)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as fd:
        for chunk in iter(lambda:fd.read(8<<20),b""):h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def signal(a,b):
    """RGBA difference with premultiplied color; transparent RGB is inert."""
    a=np.asarray(a,np.float32)
    b=np.asarray(b,np.float32)
    require(a.shape==b.shape and a.shape[-1]==4
            and np.isfinite(a).all() and np.isfinite(b).all(),
            "V15_RGBA_IMAGE_SHAPE")
    alpha=np.abs(a[...,3]-b[...,3])
    color=np.max(np.abs(a[...,:3]*a[...,3:]-b[...,:3]*b[...,3:]),axis=-1)
    return np.maximum(alpha,color)


def contribution_masks(whole_original,whole_edit,isolated_base,isolated_edit,
                       tolerance=TOL):
    require(0<tolerance<=TOL,"V15_STRICT_RGBA_TOLERANCE")
    whole=signal(whole_original,whole_edit)>tolerance
    isolated=signal(isolated_base,isolated_edit)>tolerance
    both=whole & isolated
    isolated_only=isolated & ~whole
    whole_only=whole & ~isolated
    return {
        "whole_scene_effect_pixel_instances":int(whole.sum()),
        "isolated_active_surface_effect_pixel_instances":int(isolated.sum()),
        "same_pixel_observed_in_both_effects":int(both.sum()),
        "isolation_only_effect_pixel_instances__visibility_or_compositing_candidate":
            int(isolated_only.sum()),
        "whole_scene_only_effect_pixel_instances__contextual_compositing_candidate":
            int(whole_only.sum()),
        "whole_effect_explained_by_isolated_active_surface_fraction":
            float(both.sum()/whole.sum()) if np.any(whole) else None,
        "isolated_effect_visible_in_whole_scene_fraction":
            float(both.sum()/isolated.sum()) if np.any(isolated) else None,
        "occlusion_order_authorized":False,
        "drawable_owner_authorized":False,
    }


def verify_parents(root,canonical,source_name):
    require(canonical.get("schema")==PARENT_SCHEMA
            and canonical.get("status")==PARENT_STATUS
            and canonical.get("original_perturbed_source_vs_soup_qualified_camera_cases")==144
            and canonical.get("baseline_parity_qualified_pose_views")==36
            and canonical.get("graphics_owner_authorized") is False
            and canonical.get("iris_training_authorized") is False
            and canonical.get("tessa_training_authorized") is False
            and canonical.get("materialization_authorized") is False,
            "V15_QUALIFIED_V14_CANONICAL_PARENT")
    matches=[x for x in canonical["source_results"] if x.get("source")==source_name]
    require(len(matches)==1,"V15_SOURCE_NOT_IN_EXACT_V14_ROSTER")
    source=matches[0]
    names={
        "manifest":root/"PRIVATE_REAL_NATIVE_MECHANICAL_RGBA_MANIFEST_V14.json",
        "pixels":root/"PRIVATE_REAL_NATIVE_MECHANICAL_RGBA_MANIFEST_V14.npz",
        "court":root/"PRIVATE_INDEPENDENT_GEOMETRIC_MECHANICAL_RGBA_COURT_V14.json",
        "batch":root/"PRIVATE_MECHANICAL_RGBA_JOIN_BATCH_V14.json",
    }
    require(all(p.is_file() for p in names.values()),"V15_REQUIRED_PRIVATE_SOURCE_BYTES_ABSENT")
    hashes={k:sha(v) for k,v in names.items()}
    require(hashes["court"]==source["private_independent_court_sha256"]
            and hashes["batch"]==source["private_batch_sha256"]
            and hashes["pixels"]==source["private_native_rgba_npz_sha256"],
            "V15_FROZEN_ACTUAL_SOURCE_INPUT_SHA")
    batch=read(names["batch"]);manifest=read(names["manifest"]);court=read(names["court"])
    require(batch.get("schema")==V14_BATCH
            and batch.get("independent_court_status")==V14_PASS
            and batch.get("independent_court_sha256")==hashes["court"]
            and batch.get("source_native_render_manifest_sha256")==hashes["manifest"]
            and batch.get("source_native_rgba_npz_sha256")==hashes["pixels"]
            and batch.get("subject_id")==source["subject_id"]
            and batch.get("source_sha256")==source["source_sha256"]
            and batch.get("training_authorized") is False
            and batch.get("graphics_owner_authorized") is False,
            "V15_PRIVATE_BATCH_SOURCE_AND_COURT_CHAIN")
    require(manifest.get("schema")==V14_RENDER
            and manifest.get("status")==
                "V14_REAL_SAME_CAUSAL_SOURCE_NATIVE_CONTROL_VERTEX_AND_RGBA__INDEPENDENT_COURT_REQUIRED"
            and manifest.get("source_sha256")==source["source_sha256"]
            and manifest.get("subject_id")==source["subject_id"]
            and manifest.get("original_perturbed_pose_rgba_npz_sha256")==hashes["pixels"]
            and manifest.get("source_native_control_raster_animated_action_frozen_then_restored") is True
            and manifest.get("source_native_perturbed_vertices_verified_before_and_after_render") is True
            and manifest.get("original_source_action_name_as_model_feature") is False
            and manifest.get("source_bone_mesh_material_names_as_drawable_labels") is False
            and manifest.get("native_full_resolution_caa_authorized") is False
            and manifest.get("graphics_owner_authorized") is False
            and manifest.get("training_authorized") is False,
            "V15_SOURCE_RENDER_NATIVE_CONTROL_AUTHORITY")
    require(court.get("schema")==V14_COURT
            and court.get("status")==V14_PASS
            and court.get("subject_id")==source["subject_id"]
            and court.get("source_sha256")==source["source_sha256"]
            and court.get("baseline_source_vs_soup_and_root_noop_qualified_pose_views")==12
            and court.get("original_native_perturbation_vs_source_soup_qualified_camera_cases")==48
            and court.get("original_native_perturbation_vs_source_soup_total_camera_cases")==48
            and court.get("real_source_perturbations_with_visible_rgba_consequences") is True
            and court.get("all_original_source_camera_cases_passed") is True
            and court.get("artist_same_drawable_pairs")==0
            and court.get("artist_different_drawable_pairs")==0
            and court.get("graphics_owner_authorized") is False
            and court.get("training_authorized") is False
            and court.get("materialization_authorized") is False,
            "V15_PARENT_INDEPENDENT_CAUSAL_COURT_ACTUAL_PASS")
    return names,source,manifest,court,hashes


def evaluate(root,canonical_path,source_name):
    canonical=read(canonical_path)
    names,source,manifest,court,hashes=verify_parents(root,canonical,source_name)
    plan=manifest["selected_source_controls"]
    arms,_=arms_for_plan(plan)
    require(arms==manifest["rendered_arms"]
            and manifest["rendered_source_camera_indices"]==[0,2,4,6]
            and len(manifest["rendered_pose_indices"])==3
            and len(plan["controls"])==2
            and len(court["selected_source_native_control_paired_rgba"])==48,
            "V15_PREDICTED_CONTROL_ARM_READSET_DRIFT")
    with np.load(names["pixels"],allow_pickle=False) as f:
        require(f.files==["original_native_actuator_rgba"],
                "V15_RGBA_ARRAY_KEYS")
        raw=np.asarray(f["original_native_actuator_rgba"],np.float32)
    require(raw.shape==(3,4,len(arms),192,192,4)
            and np.isfinite(raw).all()
            and float(raw[...,3].min())>=-1e-5
            and float(raw[...,3].max())<=1.00001,
            "V15_RGBA_PIXEL_FINITE_ALPHA_DOMAIN")
    ix={name:index for index,name in enumerate(arms)}
    reference={}
    for row in court["selected_source_native_control_paired_rgba"]:
        key=(row["native_pose_ordinal"],row["source_camera_view"],
             row["precommitted_source_control_ordinal"],row["perturbation_sign"])
        require(key not in reference and
                row.get("source_perturbation_soup_parity_qualified") is True
                and row.get("artist_drawable_label")=="UNKNOWN",
                "V15_COURT_DUPLICATED_OR_UNQUALIFIED_CONTROL_RESPONSE")
        reference[key]=row
    controls=[]
    effect_rows=[]
    for ci in range(len(plan["controls"])):
        item={
            "anonymous_teacher_control_ordinal":ci,
            "original_source_mechanical_control_is_not_drawable":True,
            "all_pose_view_sign_cases":24,
            "v14_raw_RGBA_control_effect_pixel_instances":0,
            "whole_scene_effect_pixel_instances":0,
            "isolated_active_surface_effect_pixel_instances":0,
            "same_pixel_observed_in_both_effects":0,
            "isolation_only_effect_pixel_instances__visibility_or_compositing_candidate":0,
            "whole_scene_only_effect_pixel_instances__contextual_compositing_candidate":0,
            "invisible_or_zero_whole_scene_cases":0,
            "visible_scene_cases":0,
            "qualified_real_original_edited_source_vs_soup_parity_cases":0,
        }
        for p in range(3):
            for view_ord,view in enumerate([0,2,4,6]):
                original=raw[p,view_ord,ix["ORIGINAL_BASE"]]
                active_base=raw[p,view_ord,ix[f"ACTIVE_C{ci:02d}_BASE"]]
                for sign in ("PLUS","MINUS"):
                    altered=raw[p,view_ord,ix[f"SOURCE_C{ci:02d}_{sign}"]]
                    isolated=raw[p,view_ord,ix[f"ACTIVE_C{ci:02d}_{sign}"]]
                    soup=raw[p,view_ord,ix[f"SOUP_C{ci:02d}_{sign}"]]
                    row=reference[(p,view,ci,sign)]
                    # Independent replay of the ORIGINAL V14 court's raw
                    # RGBA (not only alpha-aware) reference pixel instances.
                    raw_source_diff=int(np.count_nonzero(
                        np.max(np.abs(altered-original),axis=-1)>TOL
                    ))
                    raw_isolate_diff=int(np.count_nonzero(
                        np.max(np.abs(isolated-active_base),axis=-1)>TOL
                    ))
                    require(raw_source_diff==
                        row["whole_source_perturbation_changed_pixel_instances"]
                        and raw_isolate_diff==
                        row["causal_source_face_active_isolation_changed_pixel_instances"],
                        "V15_INDEPENDENT_ORIGINAL_V14_EDITS_PIXEL_REPLAY")
                    item["v14_raw_RGBA_control_effect_pixel_instances"]+=raw_source_diff
                    require(float(np.percentile(
                        np.max(np.abs(altered-soup),axis=-1),99
                    ))<=TOL,
                        "V15_PRECOMMITTED_SOURCE_NATIVE_VS_SOUP_PARITY")
                    metrics=contribution_masks(original,altered,active_base,isolated)
                    for key in (
                        "whole_scene_effect_pixel_instances",
                        "isolated_active_surface_effect_pixel_instances",
                        "same_pixel_observed_in_both_effects",
                        "isolation_only_effect_pixel_instances__visibility_or_compositing_candidate",
                        "whole_scene_only_effect_pixel_instances__contextual_compositing_candidate"
                    ):
                        item[key]+=metrics[key]
                    item["visible_scene_cases"]+=int(
                        metrics["whole_scene_effect_pixel_instances"]>0)
                    item["invisible_or_zero_whole_scene_cases"]+=int(
                        metrics["whole_scene_effect_pixel_instances"]==0)
                    item["qualified_real_original_edited_source_vs_soup_parity_cases"]+=1
                    effect_rows.append({
                        "anonymous_teacher_control_ordinal":ci,
                        "native_pose_ordinal":p,
                        "source_camera_view":view,
                        "perturbation_sign":sign,
                        "whole_scene_effect_pixel_instances":metrics["whole_scene_effect_pixel_instances"],
                        "isolated_active_surface_effect_pixel_instances":
                            metrics["isolated_active_surface_effect_pixel_instances"],
                        "same_pixel_observed_in_both_effects":metrics["same_pixel_observed_in_both_effects"],
                        "isolation_only_effect_pixel_instances":
                            metrics["isolation_only_effect_pixel_instances__visibility_or_compositing_candidate"],
                        "whole_scene_only_effect_pixel_instances":
                            metrics["whole_scene_only_effect_pixel_instances__contextual_compositing_candidate"],
                        "artist_drawable_identity":"UNKNOWN",
                    })
        require(item["visible_scene_cases"]+item["invisible_or_zero_whole_scene_cases"]==24
                and item["qualified_real_original_edited_source_vs_soup_parity_cases"]==24,
                "V15_COUNTERFACTUAL_CASE_COVERAGE")
        item["visible_source_effect_matched_by_active_isolation_fraction"]=(
            item["same_pixel_observed_in_both_effects"]/
            item["whole_scene_effect_pixel_instances"]
        ) if item["whole_scene_effect_pixel_instances"] else None
        item["isolation_effect_observed_in_full_original_fraction"]=(
            item["same_pixel_observed_in_both_effects"]/
            item["isolated_active_surface_effect_pixel_instances"]
        ) if item["isolated_active_surface_effect_pixel_instances"] else None
        controls.append(item)
    require(len(effect_rows)==len(reference)==48,
            "V15_REFERENCE_SOURCE_EDIT_CASE_SET_INCOMPLETE")
    # Canonical V14 counts use raw max-channel RGBA. V15's premultiplied
    # alpha-aware effect may legitimately count slightly fewer pixels.
    require([c["v14_raw_RGBA_control_effect_pixel_instances"] for c in controls] ==
                source["source_perturbed_control_visible_pixel_instances"],
            "V15_CANONICAL_SOURCE_VISIBLE_CONTROL_COUNTS_DRIFT")
    total={key:sum(int(row[key]) for row in controls) for key in (
        "whole_scene_effect_pixel_instances",
        "isolated_active_surface_effect_pixel_instances",
        "same_pixel_observed_in_both_effects",
        "isolation_only_effect_pixel_instances__visibility_or_compositing_candidate",
        "whole_scene_only_effect_pixel_instances__contextual_compositing_candidate",
    )}
    return {
        "schema":SCHEMA,
        "status":"REAL_NATIVE_CAUSAL_RGBA_ALPHA_AWARE_ISOLATION_RELATION_COURTED__AUTHOR_OWNER_UNKNOWN",
        "source_name_court_only":source_name,
        "original_source_sha256":source["source_sha256"],
        "subject_id":source["subject_id"],
        "original_v14_parent_court_sha256":hashes["court"],
        "original_v14_parent_batch_sha256":hashes["batch"],
        "original_v14_rgba_npz_sha256":hashes["pixels"],
        "original_v14_render_manifest_sha256":hashes["manifest"],
        "independent_courted_source_controls":len(controls),
        "pose_view_sign_control_cases":len(effect_rows),
        "causal_source_control_effects":controls,
        "causal_source_observed_and_isolated_pixel_effect_instance_totals":total,
        "original_artist_name_or_segment_read_into_student":False,
        "source_native_control_id_training_target":False,
        "no_original_artist_material_or_bone_names_in_output":True,
        "isolated_only_is_proven_occlusion":False,
        "whole_only_is_proven_draw_order":False,
        "original_source_graphics_drawable_equivalence_proven":False,
        "author_same_drawable_positive_pairs":0,
        "author_different_drawable_negative_pairs":0,
        "mechanical_causal_rgba_diagnostic_courted":True,
        "source_native_resolution_CAA_authorized":False,
        "graphics_owner_authorized":False,
        "iris_training_authorized":False,
        "tessa_training_authorized":False,
        "materialization_authorized":False,
        "product_authorized":False,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--canonical",type=Path,required=True)
    p.add_argument("--source",required=True,choices=("CAT","TRICERATOPS","COWFISH"))
    p.add_argument("--private-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    result=evaluate(args.private_root,args.canonical,args.source)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    tmp=args.output.with_suffix(".partial")
    tmp.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    tmp.replace(args.output)
    print(json.dumps({"schema":result["schema"],"status":result["status"],
          "source":args.source,
          "cases":result["pose_view_sign_control_cases"],
          "totals":result["causal_source_observed_and_isolated_pixel_effect_instance_totals"],
          "training_authorized":False},sort_keys=True),flush=True)


if __name__=="__main__":
    main()
