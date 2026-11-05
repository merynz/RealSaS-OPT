#!/usr/bin/env python3
"""V16 independent original native A+B edit-causality and RGBA interaction court.

The V14 source single A/B and the NEW V16 simultaneous A+B source render are
observations of the same source, authored pose, camera, sign and geometry.
A+B image nonadditivity is NOT equality of artist drawables. Source/isolated
effects may be hidden or compositing-dependent. No new training rights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from tools.corpus.corpus370_native_raster_join_plan_v14 import arms_for_plan
from tools.corpus.court_corpus370_causal_rgba_isolation_relation_v15 import (
    signal as image_signal,
    verify_parents as verify_v14_parent,
)

SCHEMA="RealSaS.Corpus370IndependentOriginalDualNativeControlRGBAInteractionCourt.v16"
RENDER="RealSaS.Corpus370OriginalDualNativeControlCausalInteractionRender.v16"
SOURCE_PASS="V16_REAL_ORIGINAL_TWO_NATIVE_CONTROLS_SIMULTANEOUS_RGBA__INDEPENDENT_COURT_REQUIRED"
TOL=.025
PAIRS=("PLUS_PLUS","PLUS_MINUS","MINUS_PLUS","MINUS_MINUS")


def require(ok,code):
    if not ok:raise ValueError(code)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(8<<20),b""):h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def premultiplied(a):
    a=np.asarray(a,dtype=np.float32)
    require(a.shape[-1]==4 and np.isfinite(a).all(),
        "V16_PREMULTIPLIED_RGBA_FINITE")
    return np.concatenate((a[...,:3]*a[...,3:4],a[...,3:4]),axis=-1)


def measure(a0,a1,b0,b1,ab,isolated_base,isolated_ab,tol=TOL):
    """Counterfactual spatial effect support and nonlinear interaction."""
    require(0<tol<=TOL,"V16_FROZEN_THRESHOLD")
    img=[premultiplied(z) for z in (a0,a1,b0,b1,ab,isolated_base,isolated_ab)]
    baseline,a,again,b,joint,isolated_origin,isolated=img
    require(np.array_equal(baseline,again),
        "V16_SAME_ORIGINAL_BASELINE_FOR_A_AND_B")
    sig=lambda d:np.max(np.abs(d),axis=-1)>tol
    source_a=sig(a-baseline)
    source_b=sig(b-baseline)
    source_ab=sig(joint-baseline)
    iso_ab=sig(isolated-isolated_origin)
    # Identical active source-face union support in isolated BASE and joint ±
    # scenes: never compare isolated paint to the FULL-scene baseline.
    nonlinear=sig(joint-a-b+baseline)
    incremental_a_given_b=sig(joint-b)
    incremental_b_given_a=sig(joint-a)
    alpha_a=(np.abs(a[...,3]-baseline[...,3])>tol)
    alpha_b=(np.abs(b[...,3]-baseline[...,3])>tol)
    alpha_ab=(np.abs(joint[...,3]-baseline[...,3])>tol)
    union_ab=source_a|source_b
    both=source_a&source_b
    only_a=source_a&~source_b
    only_b=source_b&~source_a
    return {
        "single_a_effect_pixel_instances":int(source_a.sum()),
        "single_b_effect_pixel_instances":int(source_b.sum()),
        "single_effect_overlap_pixel_instances":int(both.sum()),
        "single_a_only_effect_pixel_instances":int(only_a.sum()),
        "single_b_only_effect_pixel_instances":int(only_b.sum()),
        "single_effect_union_pixel_instances":int(union_ab.sum()),
        "joint_ab_effect_pixel_instances":int(source_ab.sum()),
        "joint_isolated_active_face_effect_pixel_instances":int(iso_ab.sum()),
        "joint_full_and_isolated_effect_overlap_pixel_instances":
            int((iso_ab & source_ab).sum()),
        "joint_isolation_only_effect_pixel_instances__unresolved_visibility_or_compositing":
            int((iso_ab & ~source_ab).sum()),
        "joint_full_only_effect_pixel_instances__contextual":
            int((source_ab & ~iso_ab).sum()),
        "joint_nonadditive_rgba_response_pixel_instances":int(nonlinear.sum()),
        "incremental_a_given_b_pixel_instances":int(incremental_a_given_b.sum()),
        "incremental_b_given_a_pixel_instances":int(incremental_b_given_a.sum()),
        "joint_affected_pixels_beyond_single_union":int((source_ab&~union_ab).sum()),
        "single_affected_pixels_disappearing_under_joint":
            int((union_ab&~source_ab).sum()),
        "single_a_alpha_effect_pixel_instances":int(alpha_a.sum()),
        "single_b_alpha_effect_pixel_instances":int(alpha_b.sum()),
        "joint_alpha_effect_pixel_instances":int(alpha_ab.sum()),
        "joint_effect_active_face_isolation_observability":"EXACT_SAME_ACTIVE_SOURCE_FACE_UNION_RENDERED_IN_BASELINE_AND_JOINT_SOURCE_CONTROL",
        "source_drawable_same_as_control":False,
        "proves_artist_drawable_equivalence":False,
    }


def evaluate(source_name, v14_public_path, original_root, v16_json_path,
             v16_npz_path, v14_job_path):
    public=read(v14_public_path)
    paths,row,parent_render,parent_court,hashes=verify_v14_parent(
        original_root, public, source_name)
    joined=read(v16_json_path)
    require(joined.get("schema")==RENDER
            and joined.get("status")==SOURCE_PASS
            and joined.get("subject_id")==row["subject_id"]
            and joined.get("source_sha256")==row["source_sha256"]
            and joined.get("parent_v14_job_sha256")==sha(v14_job_path)
            and joined.get("parent_v14_render_manifest_sha256")==hashes["manifest"]
            and joined.get("parent_v14_rgba_npz_sha256")==hashes["pixels"]
            and joined.get("original_dual_control_rgba_npz_sha256")==sha(v16_npz_path)
            and joined.get("original_v13b_control_vertex_array_sha256")==
                parent_render["source_private_v13b_intervention_array_sha256"]
            and joined.get("precommitted_blind_v14_control_plan_sha256")==
                parent_render["source_private_control_selection_sha256"]
            and joined.get("original_native_source_inventory_sha256")==
                parent_render["original_native_source_carrier_inventory_sha256"]
            and joined.get("anonymous_source_controls_tested")==2
            and joined.get("source_pose_count")==3
            and joined.get("source_camera_views")==[0,2,4,6]
            and joined.get("signed_control_pairs")==list(PAIRS)
            and joined.get("blender_render_arms")==168
            and joined.get("joint_original_source_to_soup_parity_rechecked") is True
            and joined.get("original_source_action_frozen_and_exact_vertices_before_after_render") is True
            and joined.get("joint_visuals_from_real_original_native_blender_source") is True
            and joined.get("no_expected_joint_ground_truth_fabricated_from_single_v13b") is True
            and joined.get("source_artist_drawable_names_or_part_count_used") is False
            and joined.get("source_rig_bone_material_names_as_model_features") is False
            and joined.get("source_presentation_draw_order_proven") is False
            and joined.get("mechanical_controller_is_a_drawable_label") is False
            and joined.get("author_same_drawable_pairs")==0
            and joined.get("author_different_drawable_pairs")==0
            and joined.get("graphics_owner_authorized") is False
            and joined.get("training_authorized") is False
            and joined.get("materialization_authorized") is False,
            "V16_ORIGINAL_JOINT_SOURCE_MANIFEST_PRIVILEGED_AUTHORITY_OR_SHA")
    # Court must not import Blender or control carriers: only typed observed
    # source images, blind anonymous A/B handles, and frozen source SHA chain.
    expected=["ORIGINAL_BASE","ISOLATED_AB_BASE"]+[
        mode+"_"+s for s in PAIRS
        for mode in ("SOURCE_AB","SOUP_AB","ISOLATED_AB")
    ]
    require(joined["rendered_arms"]==expected and len(expected)==14,
        "V16_EXACT_ORIGINAL_SOURCE_JOINT_ARMS")
    require(len(joined["native_source_joint_mechanics_vs_independent_single_effect_witness"])==48,
        "V16_NATIVE_JOINT_MECHANICS_WITNESS_NOT_COMPLETE")

    with np.load(paths["pixels"],allow_pickle=False) as f:
        require(f.files==["original_native_actuator_rgba"],
            "V16_FROZEN_ORIGINAL_V14_RGBA_MEMBERS")
        old=np.asarray(f["original_native_actuator_rgba"],np.float32)
    with np.load(v16_npz_path,allow_pickle=False) as f:
        require(f.files==["original_native_dual_control_rgba"],
            "V16_FROZEN_JOINT_RGBA_MEMBERS")
        new=np.asarray(f["original_native_dual_control_rgba"],np.float32)
    plan=parent_render["selected_source_controls"]
    old_arms,_=arms_for_plan(plan)
    require(old_arms==parent_render["rendered_arms"]
        and old.shape==(3,4,19,192,192,4)
        and new.shape==(3,4,14,192,192,4)
        and np.isfinite(old).all() and np.isfinite(new).all()
        and np.min(old[...,3])>=-1e-5 and np.max(old[...,3])<=1.00001
        and np.min(new[...,3])>=-1e-5 and np.max(new[...,3])<=1.00001,
        "V16_REAL_SOURCE_AND_JOINT_CAMERA_POSE_RGBA_FINITE")
    oi={name:i for i,name in enumerate(old_arms)}
    ji={name:i for i,name in enumerate(expected)}
    witness={}
    for w in joined["native_source_joint_mechanics_vs_independent_single_effect_witness"]:
        k=(w["source_pose_ordinal"],w["source_camera_view"],w["sign_a"],w["sign_b"])
        require(k not in witness
            and w["source_to_soup_p99"]<=TOL
            and w["joint_native_max_displacement"]>0
            and w["joint_native_vs_single_control_additive_max_residual"]>=0
            and w["joint_native_vs_single_control_additive_mean_residual"]>=0
            and w["author_drawable_equality"]=="UNKNOWN",
            "V16_NATIVE_JOINT_WITNESS_FAKE_PARITY_OR_ARTIST_OWNER")
        witness[k]=w
    require(len(witness)==48,"V16_NATIVE_JOINT_WITNESS_DUPLICATE_OR_MISSING")
    case_rows=[]
    metric_keys=None
    aggregates={}
    all_parity=True
    for pose in range(3):
        for iv,view in enumerate([0,2,4,6]):
            b14=old[pose,iv,oi["ORIGINAL_BASE"]]
            b16=new[pose,iv,ji["ORIGINAL_BASE"]]
            require(float(np.percentile(
                np.max(np.abs(b14-b16),axis=-1),99))<=TOL,
                "V16_V14_ORIGINAL_BASELINE_DRIFT_INDEPENDENT_COURT")
            for s1 in ("PLUS","MINUS"):
                for s2 in ("PLUS","MINUS"):
                    sign=("_").join((s1,s2))
                    joint=new[pose,iv,ji["SOURCE_AB_"+sign]]
                    soup=new[pose,iv,ji["SOUP_AB_"+sign]]
                    iso=new[pose,iv,ji["ISOLATED_AB_"+sign]]
                    iso_base=new[pose,iv,ji["ISOLATED_AB_BASE"]]
                    p99=float(np.percentile(
                        np.max(np.abs(joint-soup),axis=-1),99))
                    require(p99<=TOL, "V16_REAL_ORIGINAL_AB_SOURCE_SOUP_PIXEL_PARITY")
                    key=(pose,view,1 if s1=="PLUS" else -1,1 if s2=="PLUS" else -1)
                    require(key in witness
                            and abs(witness[key]["source_to_soup_p99"]-p99)<=.00001,
                            "V16_COURT_V16_RENDERER_SOUSED_SOUSED_PARITY_DRIFT")
                    a=old[pose,iv,oi["SOURCE_C00_"+s1]]
                    b=old[pose,iv,oi["SOURCE_C01_"+s2]]
                    m=measure(b14,a,b14,b,joint,iso_base,iso)
                    metric_keys=tuple(m) if metric_keys is None else metric_keys
                    for key,value in m.items():
                        if type(value) is int: aggregates[key]=aggregates.get(key,0)+value
                    case_rows.append({
                        "source_pose_ordinal":pose,
                        "source_camera_view":view,
                        "anonymous_intervention_a_sign":s1,
                        "anonymous_intervention_b_sign":s2,
                        "source_vs_soup_RGBA_p99":p99,
                        **m,
                        "original_artist_owner_relation":"UNKNOWN"
                    })
    require(len(case_rows)==48
        and aggregates["joint_ab_effect_pixel_instances"]>0,
        "V16_SOURCE_CONTROL_PAIR_NO_REAL_CAUSAL_RESPONSE")
    return {
        "schema":SCHEMA,
        "status":"V16_REAL_JOINT_NATIVE_CONTROLS_CAUSAL_NONADDITIVITY_MEASURED__AUTHOR_OWNER_UNKNOWN",
        "source_name_court_only":source_name,
        "subject_id":row["subject_id"],
        "original_source_sha256":row["source_sha256"],
        "source_v14_private_court_sha256":hashes["court"],
        "source_v14_parent_rgba_sha256":hashes["pixels"],
        "source_v16_private_blender_manifest_sha256":sha(v16_json_path),
        "source_v16_joint_RGBA_npz_sha256":sha(v16_npz_path),
        "source_poses":3,"source_views":[0,2,4,6],
        "anonymous_native_controls":[0,1],
        "real_joint_native_sign_combinations":list(PAIRS),
        "causal_joint_visual_counterfactual_cases":len(case_rows),
        "source_native_joint_vs_original_v13b_single_mechanics_nonadditive_witness_count":len(witness),
        "independently_replayed_qualifying_source_soup_RGBA_case_count":len(case_rows),
        "causal_joint_effect_rows":case_rows,
        "aggregate_effects":aggregates,
        "joint_nonadditivity_is_not_artist_editable_layer_equivalence":True,
        "original_mechanical_source_controls_as_model_features":False,
        "source_artist_own_layer_names_as_model_targets":False,
        "original_author_same_drawable_labels":0,
        "original_author_different_drawable_labels":0,
        "same_drawable_truth_authorized":False,
        "alpha_draw_order_CAA_authorized":False,
        "graphics_owner_authorized":False,
        "iris_training_authorized":False,
        "tessa_training_authorized":False,
        "materialization_authorized":False,
        "training_authorized":False
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True,choices=("CAT","TRICERATOPS","COWFISH"))
    for f in ("v14_public","v14_private_root","v14_original_job",
              "v16_manifest","v16_npz","output"):
        p.add_argument("--"+f.replace("_","-"),required=True,type=Path)
    a=p.parse_args()
    result=evaluate(a.source,a.v14_public,a.v14_private_root,
                    a.v16_manifest,a.v16_npz,a.v14_original_job)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({
        "schema":result["schema"],"subject_id":result["subject_id"],
        "status":result["status"],
        "real_joint_cases":result["causal_joint_visual_counterfactual_cases"],
        "joint_effect_instances":result["aggregate_effects"]["joint_ab_effect_pixel_instances"],
        "joint_nonadditivity_instances":
            result["aggregate_effects"]["joint_nonadditive_rgba_response_pixel_instances"],
        "training_authorized":False},sort_keys=True),flush=True)


if __name__=="__main__":
    main()
