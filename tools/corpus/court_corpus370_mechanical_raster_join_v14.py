"""V14 independent full RGBA court: original control perturbation -> editable image.

This is NOT an artist drawable identity court. Real source native GLTF/FBX
mechanics plus UV-preserving original source/soup under *the same* ± source
control, same authored action pose and same sealed native camera. Color/alpha
render fidelity and visibility are bounded 192px measurements, not native CAA.
"""
from __future__ import annotations
import numpy as np
from tools.corpus.corpus370_native_raster_join_plan_v14 import (
    SCHEMA as PLAN_SCHEMA,arms_for_plan
)

SCHEMA="RealSaS.Corpus370SameNativeControlGeometricRasterCourt.v14"
RENDER="RealSaS.Corpus370SourceNativeMechanicalRasterJointRender.v14"
STATUS="V14_REAL_SAME_CAUSAL_SOURCE_NATIVE_CONTROL_VERTEX_AND_RGBA__INDEPENDENT_COURT_REQUIRED"
RGBA_TOL=.025
NOOP_TOL=.005

def require(ok,reason):
    if not ok:raise ValueError(reason)

def change(a,b):
    return np.max(np.abs(np.asarray(a,np.float32)-np.asarray(b,np.float32)),axis=-1)

def evaluate(pixels,render,selection,prior_v13b_manifest,prior_v13b_batch,prior_v13b_public_source):
    controls=selection.get("controls")
    arms,pairs=arms_for_plan(selection)
    require(render.get("schema")==RENDER
            and render.get("status")==STATUS
            and render.get("selected_source_controls")==selection
            and render.get("rendered_arms")==arms
            and render.get("rendered_pose_indices")==selection["native_pose_indices_teacher_only"]
            and render.get("rendered_source_camera_indices")==[0,2,4,6]
            and render.get("raster_resolution")==192
            and render.get("cycles_samples")==32
            and render.get("native_poses_exact_source_author_actions") is True
            and render.get("original_source_rest_and_animated_face_identity") is True
            and render.get("source_pose_matches_original_v13b_vertex_oracle") is True
            and render.get("perturbed_source_pose_matches_original_v13b_native_control") is True
            and render.get("source_active_faces_are_not_artist_drawables") is True
            and render.get("source_bone_mesh_material_names_as_drawable_labels") is False
            and render.get("original_source_action_name_as_model_feature") is False
            and render.get("rgba_and_visibility_only_not_actual_draw_order_proof") is True
            and render.get("root_sham")==
                "CAMERA_NEGATIVE_TRANSLATION_EQUIVALENT_TO_POSITIVE_GLOBAL_SOURCE_TRANSLATION_NOT_AUTHORED_ROOT"
            and render.get("artist_same_drawable_pairs")==0
            and render.get("artist_different_drawable_pairs")==0
            and render.get("native_full_resolution_caa_authorized") is False
            and render.get("graphics_owner_authorized") is False
            and render.get("training_authorized") is False
            and prior_v13b_manifest["source_sha256"]==render["source_sha256"]==
                prior_v13b_public_source["original_source_sha256"]
            and render["source_private_v13b_intervention_array_sha256"]==
                prior_v13b_public_source["raw_vertex_response_npz_sha256"]
            and render["source_private_v13b_intervention_manifest_sha256"]==
                prior_v13b_batch["native_intervention_manifest_sha256"]
            and prior_v13b_batch["native_vertex_response_npz_sha256"]==
                render["source_private_v13b_intervention_array_sha256"]
            and prior_v13b_batch["independent_mechanics_court_sha256"]==
                prior_v13b_public_source["independent_court_sha256"]
            and prior_v13b_manifest["source_controls_measured_count"]==
                selection["source_control_inventory_count_teacher_only"]
            and prior_v13b_public_source["source_controls_total"]==
                selection["source_control_inventory_count_teacher_only"]
            and prior_v13b_batch["subject_id"]==render["subject_id"]
            and prior_v13b_batch["training_authorized"] is False
            and prior_v13b_batch["graphics_owner_authorized"] is False
            and render["original_source_control_carrier_kinds_teacher_only"]==
                prior_v13b_manifest["native_control_carrier_types_teacher_only"]
            and prior_v13b_public_source["local_nonrigid_cases"]>0
            and prior_v13b_public_source["nonreversible_cases"]==0,
            "V14_FROZEN_INDEPENDENT_V13B_SOURCE_MECHANICS_AND_RGBA_NOT_OWNER")
    x=np.asarray(pixels,np.float32)
    require(x.shape==(3,4,len(arms),192,192,4)
            and np.isfinite(x).all()
            and float(x[...,3].min())>=-1e-5
            and float(x[...,3].max())<=1.00001,
            "V14_REAL_FULL_RGBA_PIXEL_SCHEMA")
    ix={name:i for i,name in enumerate(arms)}
    parity=[]
    base_ok=[]
    sham_change_px=0
    source_pose_occupancy=0
    for p in range(3):
        for v,vid in enumerate([0,2,4,6]):
            source=x[p,v,ix["ORIGINAL_BASE"]]
            soup=x[p,v,ix["SOUP_BASE"]]
            noop=x[p,v,ix["SOUP_NOOP"]]
            root=x[p,v,ix["ROOT_SHAM_ORIGINAL"]]
            root_soup=x[p,v,ix["ROOT_SHAM_SOUP"]]
            occupancy=int(np.count_nonzero(source[...,3]>.001))
            source_pose_occupancy+=occupancy
            base_diff=change(source,soup)
            p99=float(np.percentile(base_diff,99))
            noopmax=float(change(soup,noop).max(initial=0))
            root_p99=float(np.percentile(change(root,root_soup),99))
            qualified=bool(occupancy>0 and p99<=RGBA_TOL
                           and noopmax<=NOOP_TOL and root_p99<=RGBA_TOL)
            root_change=int(np.count_nonzero(change(root,source)>RGBA_TOL))
            sham_change_px+=root_change
            parity.append({
                "original_pose_ordinal":p,"source_camera_view":vid,
                "source_original_nonzero_alpha_pixel_count":occupancy,
                "baseline_source_vs_soup_rgba_p99":p99,
                "source_soup_noop_max":noopmax,
                "root_camera_sham_original_vs_soup_rgba_p99":root_p99,
                "global_root_equivalent_sham_changes_pixels":root_change,
                "qualified":qualified
            })
            base_ok.append(qualified)
    responses=[]
    effects_by_control=[0 for _ in controls]
    iso_by_control=[0 for _ in controls]
    validated_parity_count=0
    total_parity=3*4*len(controls)*2
    for p in range(3):
        for v,vid in enumerate([0,2,4,6]):
            base=x[p,v,ix["ORIGINAL_BASE"]]
            for ci,r in enumerate(controls):
                active_base=x[p,v,ix[f"ACTIVE_C{ci:02d}_BASE"]]
                for sign in ("PLUS","MINUS"):
                    original=x[p,v,ix[f"SOURCE_C{ci:02d}_{sign}"]]
                    soup=x[p,v,ix[f"SOUP_C{ci:02d}_{sign}"]]
                    active=x[p,v,ix[f"ACTIVE_C{ci:02d}_{sign}"]]
                    p99=float(np.percentile(change(original,soup),99))
                    # An edited source-face active support may be hidden by
                    # original depth/occluders. This is NOT author draw order
                    # or source art layer membership.
                    effect=int(np.count_nonzero(change(original,base)>RGBA_TOL))
                    isolated=int(np.count_nonzero(change(active,active_base)>RGBA_TOL))
                    visible=int(np.count_nonzero(active[...,3]>.001))
                    valid=bool(base_ok[p*4+v] and p99<=RGBA_TOL)
                    validated_parity_count+=int(valid)
                    effects_by_control[ci]+=effect if valid else 0
                    iso_by_control[ci]+=isolated if valid else 0
                    responses.append({
                        "native_pose_ordinal":p,"source_camera_view":vid,
                        "precommitted_source_control_ordinal":ci,
                        "control_response_class_source_only":r["source_response_class_teacher_only"],
                        "perturbation_sign":sign,
                        "real_original_vs_source_soup_rgba_p99":p99,
                        "whole_source_perturbation_changed_pixel_instances":effect,
                        "causal_source_face_active_isolation_changed_pixel_instances":isolated,
                        "causal_source_face_active_isolation_visible_pixel_count":visible,
                        "source_perturbation_soup_parity_qualified":valid,
                        "artist_drawable_label":"UNKNOWN"
                    })
    base_count=sum(base_ok)
    sign_effect_ok=all(x>0 for x in effects_by_control)
    source_local=controls[0]["source_response_class_teacher_only"]=="LOCAL_NONRIGID"
    passed=base_count==12 and validated_parity_count==total_parity and source_local and sign_effect_ok
    return {
        "schema":SCHEMA,
        "status":("V14_REAL_CAUSAL_SOURCE_MECHANICAL_TO_RGBA_JOIN_QUALIFIED__OWNER_UNKNOWN"
                  if passed else
                  "V14_PARTIAL_OR_FAILED_ORIGINAL_RGBA_PARITY_OR_VISIBILITY__OWNER_UNKNOWN"),
        "subject_id":render["subject_id"],
        "source_sha256":render["source_sha256"],
        "original_native_poses":3,
        "source_original_camera_views":[0,2,4,6],
        "original_render_arm_count":int(x.shape[0]*x.shape[1]*x.shape[2]),
        "source_actuator_selected_before_new_v14_raster":True,
        "selected_native_actuator_source_only":controls,
        "selected_source_control_count":len(controls),
        "baseline_source_vs_soup_and_root_noop_qualified_pose_views":base_count,
        "baseline_source_vs_soup_and_root_noop_total_pose_views":12,
        "original_native_perturbation_vs_source_soup_qualified_camera_cases":validated_parity_count,
        "original_native_perturbation_vs_source_soup_total_camera_cases":total_parity,
        "real_source_perturbations_with_visible_rgba_consequences":sign_effect_ok,
        "whole_source_perturbation_effective_pixel_instances_by_selected_control":effects_by_control,
        "causal_active_face_isolation_changed_pixel_instances_by_selected_control":iso_by_control,
        "analytic_global_root_equivalent_camera_sham_changed_pixel_instances":sham_change_px,
        "source_original_alpha_pixel_instances":source_pose_occupancy,
        "baseline_source_pose_camera_parity":parity,
        "selected_source_native_control_paired_rgba":responses,
        "all_original_source_camera_cases_passed":passed,
        "native_resolution_source_alpha_caa_authorized":False,
        "true_source_depth_and_draw_order_proven":False,
        "source_morphology_generalization_proven":False,
        "observed_local_deformation_proves_separate_drawable":False,
        "observed_shared_perturbation_proves_same_drawable":False,
        "artist_same_drawable_pairs":0,
        "artist_different_drawable_pairs":0,
        "source_skeleton_or_object_semantic_names_used_as_targets":False,
        "graphics_owner_authorized":False,
        "iris_training_authorized":False,
        "tessa_training_authorized":False,
        "materialization_authorized":False,
        "training_authorized":False
    }
