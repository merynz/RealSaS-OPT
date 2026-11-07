#!/usr/bin/env python3
"""V13 independent kinematic source-actuator response vs root/null controls.

True local interventions occur ONLY through original source armature action
evaluation in Blender, not in inferred model space. The math court receives
anonymous source vertex arrays and the pre-existing geometry hypotheses.
No semantic ontology, source control label or authored drawable is minted.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np

SCHEMA="RealSaS.Corpus370AnonymousNativeMechanicalResponseCourt.v13"
RENDER="RealSaS.Corpus370NativeAnonymousActuatorInterventionRender.v13"
RADIAL_EPS=1e-12

def require(ok,code):
    if not ok:raise ValueError(code)

def _rigid_rms(source,target):
    """Best global rigid SE(3) fit in source coordinates, no semantic input."""
    x=np.asarray(source,np.float64);y=np.asarray(target,np.float64)
    require(x.shape==y.shape and x.ndim==2 and x.shape[1]==3 and len(x)>=3
            and np.isfinite(x).all() and np.isfinite(y).all(),
            "V13_KABSCH_FINITE_MATCHED_VERTEX_DOMAIN")
    a=x.mean(axis=0);b=y.mean(axis=0)
    h=(x-a).T@(y-b)
    u,_,vh=np.linalg.svd(h)
    d=np.linalg.det(vh.T@u.T)
    diag=np.diag([1.,1.,-1. if d<0 else 1.])
    rotation=vh.T@diag@u.T
    fitted=(x-a)@rotation.T+b
    return float(np.sqrt(np.mean(np.sum((y-fitted)**2,axis=1))))

def _face_activity(displacement,faces,eps):
    perface=np.asarray(displacement,dtype=np.float64)[np.asarray(faces,dtype=np.int64)]
    # Triangular mean displacement, face IDs remain geometric, not drawable.
    return np.mean(perface,axis=1)>eps

def _mixed_groups(face_moved,labels):
    group=defaultdict(list)
    for index,label in enumerate(labels):
        group[int(label)].append(bool(face_moved[index]))
    count=sum(any(values) and not all(values) for values in group.values())
    return {"mixed_candidate_groups":count,
            "observed_candidate_groups":len(group),
            "all_static_candidate_groups":sum(not any(v) for v in group.values()),
            "all_moving_candidate_groups":sum(all(v) for v in group.values())}

def _controls_noop_and_root(rest):
    baseline=np.asarray(rest,dtype=np.float64)
    extent=float(np.linalg.norm(np.ptp(baseline,axis=0)))
    t=np.array((.025*extent,0.,0.),np.float64)
    root=baseline+t
    return {
       "null_zero_max":float(np.abs(baseline-baseline).max(initial=0)),
       "root_uniform_translation_rigid_fit_rms":_rigid_rms(baseline,root),
       "root_max_displacement":float(np.max(np.linalg.norm(root-baseline,axis=-1))),
       "root_control_sham_source_geometry_only":True
    }

def evaluate(rest,faces,native_original,baseline,measured,manifest,
             proposal,v11_plan,v12_public_result,original_v9_pose_indices):
    rest=np.asarray(rest,dtype=np.float64)
    faces=np.asarray(faces,dtype=np.int64)
    native_original=np.asarray(native_original,dtype=np.float64)
    b=np.asarray(baseline,dtype=np.float64)
    intervention=np.asarray(measured,dtype=np.float64)
    require(manifest.get("schema")==RENDER
        and manifest.get("status")==
           "V13_REAL_ORIGINAL_AUTHORED_SOURCE_ANONYMOUS_RIG_INTERVENTION__INDEPENDENT_COURT_REQUIRED"
        and manifest.get("source_sha256")==proposal.get("source_sha256")==
            v11_plan.get("source_sha256")
        and manifest.get("subject_id")==v11_plan.get("subject_id")
        and manifest.get("source_rest_vertex_count")==len(rest)
        and manifest.get("source_face_count")==len(faces)
        and manifest.get("source_controls_measured_count")<=128
        and manifest.get("source_control_inventory_count")>=manifest.get("source_controls_measured_count")
        and manifest.get("original_source_teacher_pose_count")==3
        and manifest.get("original_selected_pose_indices_teacher_only")==
             original_v9_pose_indices
        and manifest.get("axes")==["X","Y","Z"]
        and manifest.get("rotation_degrees")==5.0
        and manifest.get("no_anon_control_intervention_result_as_drawable_truth") is True
        and manifest.get("anonymous_control_count_inference_target") is False
        and manifest.get("native_source_actions_as_student_input") is False
        and manifest.get("graphics_owner_authorized") is False
        and manifest.get("training_authorized") is False
        and manifest.get("artist_same_drawable_positive_pairs")==0
        and manifest.get("artist_different_drawable_negative_pairs")==0
        and v11_plan.get("training_authorized") is False
        and v11_plan.get("graphics_owner_authorized") is False
        and v11_plan.get("new_source_face_intervention_query_count")==12
        and v12_public_result["source_sha256"]==manifest["source_sha256"]
        and v12_public_result["qualified_source_vs_soup_pose_camera_cases"]==12
        and v12_public_result["render_arms"] in (912,1200),
        "V13_ORIGINAL_NATIVE_CAUSAL_SOURCE_PARENT_AUTHORITY")
    require(rest.ndim==2 and rest.shape[1]==3
        and faces.ndim==2 and faces.shape[1]==3
        and len(rest)>=3 and len(faces)>=8
        and int(faces.min())>=0 and int(faces.max())<len(rest)
        and b.shape==(3,len(rest),3)
        and native_original.shape[1:]==rest.shape
        and intervention.shape==
            (3,manifest["source_controls_measured_count"],3,2,len(rest),3)
        and all(np.isfinite(x).all() for x in (rest,b,native_original,intervention))
        and np.max(np.abs(b-native_original[manifest["original_selected_pose_indices_teacher_only"]]))<=1e-5*max(1.,float(np.linalg.norm(np.ptp(rest,axis=0)))),
        "V13_FROZEN_ORIGINAL_AUTHORED_TEACHER_VS_BLENDER_POSE_SHAPE_AND_IDENTITY")
    candidate_labels={c["slot_count"]:np.asarray(c["face_slot_teacher_only"],dtype=np.int64)
                      for c in proposal["candidates"]}
    require(set(candidate_labels).issuperset({8,16,32})
        and all(len(a)==len(faces) and a.min()>=0 and a.max()<k
                for k,a in candidate_labels.items()),
        "V13_SOURCE_GEOMETRIC_GROUP_HYPOTHESIS_FACE_ADDRESS")
    sizes=sorted(k for k in candidate_labels if k in (8,16,32,64))
    n_controls=manifest["source_controls_measured_count"]
    extent=max(1.e-12,float(np.linalg.norm(np.ptp(rest,axis=0))))
    eps=2.e-4*extent
    sham=_controls_noop_and_root(rest)
    require(sham["null_zero_max"]==0. and
        sham["root_uniform_translation_rigid_fit_rms"]<=1e-7*max(1.,extent),
        "V13_INDEPENDENT_ROOT_SHAM_RIGID_GEOMETRY_NOOP")
    root_face=_face_activity(np.linalg.norm((rest+np.array([.025*extent,0,0]))-rest,axis=-1),
                             faces,eps)
    require(root_face.all() and
        all(_mixed_groups(root_face,candidate_labels[k])["mixed_candidate_groups"]==0 for k in sizes),
        "V13_GLOBAL_ROOT_SHAM_WRONGLY_CLASSIFIED_INDEPENDENT")
    counts={
        "tested_pose_control_axis_combinations":3*n_controls*3,
        "source_effect_nonzero_count":0,
        "root_like_or_globally_rigid_effect_count":0,
        "local_nonrigid_effect_count":0,
        "nonreversible_effect_masked_count":0,
        "no_source_effect_abstain_count":0,
        "candidate_mixed_group_events":{str(k):0 for k in sizes},
        "candidate_mixed_group_fraction_sum":{str(k):0. for k in sizes},
        "max_native_response_displacement_relative_extent":0.,
        "max_nonrigid_residual_relative_extent":0.,
        "control_inventory_truncated":manifest["source_control_inventory_truncated"],
    }
    poses=[]
    for pi in range(3):
        moving=local=static=nonrev=0
        for ci in range(n_controls):
            for axis in range(3):
                plus=intervention[pi,ci,axis,0]
                minus=intervention[pi,ci,axis,1]
                source=b[pi]
                positive=plus-source
                negative=minus-source
                disp=np.linalg.norm(positive,axis=-1)
                max_move=float(disp.max(initial=0.))
                counts["max_native_response_displacement_relative_extent"]=max(
                    counts["max_native_response_displacement_relative_extent"],max_move/extent)
                if max_move<=eps:
                    counts["no_source_effect_abstain_count"]+=1
                    static+=1
                    continue
                counts["source_effect_nonzero_count"]+=1
                moving+=1
                residual=_rigid_rms(source,plus)
                counts["max_nonrigid_residual_relative_extent"]=max(
                    counts["max_nonrigid_residual_relative_extent"],residual/extent)
                symmetry_err=np.sqrt(np.mean(np.sum((positive+negative)**2,axis=-1)))
                energy=np.sqrt(np.mean(np.sum((positive-negative)**2,axis=-1)))
                # A finite + / - response need not be perfectly antisymmetric;
                # nonlinear rig constraints may legitimately change the curve.
                reversible=energy>eps and symmetry_err/max(energy,1e-12)<=.35
                if not reversible:
                    nonrev+=1;counts["nonreversible_effect_masked_count"]+=1
                if residual<=eps:
                    counts["root_like_or_globally_rigid_effect_count"]+=1
                else:
                    counts["local_nonrigid_effect_count"]+=1
                    local+=1
                active=_face_activity(disp,faces,eps)
                for k in sizes:
                    data=_mixed_groups(active,candidate_labels[k])
                    counts["candidate_mixed_group_events"][str(k)]+=(data["mixed_candidate_groups"]>0)
                    counts["candidate_mixed_group_fraction_sum"][str(k)]+=(data["mixed_candidate_groups"]/
                        max(1,data["observed_candidate_groups"]))
        poses.append({
            "pose_index_teacher_only":manifest["original_selected_pose_indices_teacher_only"][pi],
            "native_anonymous_controls_measured":n_controls,
            "axes_per_control":3,
            "source_actuators_with_visible_geometry_effect":moving,
            "source_actuators_with_nonrigid_local_effect":local,
            "source_actuators_with_no_source_response":static,
            "nonreversible_responses_masked":nonrev,
        })
    counts["candidate_mixed_group_fraction_mean"]={
        str(k):counts["candidate_mixed_group_fraction_sum"][str(k)]/
            max(1,counts["source_effect_nonzero_count"])
        for k in sizes
    }
    status=("REAL_NATIVE_ANONYMOUS_RIG_INTERVENTIONS_OBSERVED_NO_OWNER"
            if counts["source_effect_nonzero_count"]>0
            else "ABSTAIN_NO_EFFECTIVE_NATIVE_RIG_CONTROLS_NO_OWNER")
    return {
        "schema":SCHEMA,"status":status,
        "subject_id":manifest["subject_id"],"source_sha256":manifest["source_sha256"],
        "pose_count":3,"native_control_count":n_controls,
        "native_controls_total_before_budget":manifest["source_control_inventory_count"],
        "source_native_rig_controls_are_privileged_teacher_only":True,
        "source_bone_names_or_ontology_as_inference_target":False,
        "source_generated_control_perturbation_is_art_author_label":False,
        "native_rig_angular_perturbation_degrees":5.,
        "analytic_uniform_root_translation_sham":sham,
        "original_source_restore_noop_error":manifest["all_control_restoration_noop_max_error"],
        "numeric_threshold_source_relative":eps/extent,
        "source_native_controls_by_pose":poses,
        "source_native_anonymous_mechanics_counts":counts,
        "causal_source_rig_perturbation_observed":counts["source_effect_nonzero_count"]>0,
        "mixed_source_groups_not_certified_as_drawables":True,
        "positive_different_drawable_labels":0,
        "positive_same_drawable_labels":0,
        "native_resolution_render_caa_qualified":False,
        "source_attachment_owner_authorized":False,
        "graphics_owner_authorized":False,
        "tessa_training_authorized":False,
        "iris_training_authorized":False,
        "materialization_authorized":False,
        "training_authorized":False
    }
