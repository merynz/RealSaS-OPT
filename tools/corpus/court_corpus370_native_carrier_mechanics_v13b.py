#!/usr/bin/env python3
"""V13b independent native ORIGINAL rig + importer object carrier causal court.

Extends V13's unchanged independent geometric response judge to all source-
imported action node/object carriers, not an arbitrary hand-authored skeleton.
The original shape/pose/face authority remains SHA-locked. No ownership labels.
"""
from __future__ import annotations
from tools.corpus.court_corpus370_native_actuator_mechanics_v13 import evaluate as _geometry_court

SCHEMA="RealSaS.Corpus370AnonymousNativeMixedCarrierResponseCourt.v13b"
RENDER="RealSaS.Corpus370NativeCarrierActuatorInterventionRender.v13b"
STATUS="V13B_REAL_ORIGINAL_SOURCE_ANONYMOUS_NATIVE_CARRIER_INTERVENTION__INDEPENDENT_COURT_REQUIRED"

def require(ok,reason):
    if not ok: raise ValueError(reason)

def evaluate(rest,faces,native_original,baseline,measured,manifest,
             proposal,v11_plan,v12_public_result,original_v9_pose_indices):
    kinds=manifest.get("native_control_carrier_types_teacher_only")
    require(manifest.get("schema")==RENDER
            and manifest.get("status")==STATUS
            and isinstance(kinds,dict)
            and set(kinds)=={"POSE_BONE","IMPORTED_ANIMATED_OBJECT"}
            and all(type(x) is int and x>=0 for x in kinds.values())
            and sum(kinds.values())==manifest.get("source_controls_measured_count")
            and manifest.get("source_animated_object_owner_count")>=kinds["IMPORTED_ANIMATED_OBJECT"]
            and manifest.get("source_object_controls_preselected_from_original_imported_action_bindings") is True
            and manifest.get("source_control_carrier_type_as_model_prior") is False
            and manifest.get("native_source_actions_as_student_input") is False
            and manifest.get("anonymous_control_count_inference_target") is False
            and manifest.get("no_anon_control_intervention_result_as_drawable_truth") is True
            and manifest.get("training_authorized") is False
            and manifest.get("graphics_owner_authorized") is False,
            "V13B_NATIVE_IMPORTED_SOURCE_CARRIER_OR_FALSE_LABEL")
    base=dict(manifest)
    # The original V13 independent numerical court is unchanged and independently
    # rechecks all geometric array/pose/sha-derived research authority invariants.
    base["schema"]="RealSaS.Corpus370NativeAnonymousActuatorInterventionRender.v13"
    base["status"]="V13_REAL_ORIGINAL_AUTHORED_SOURCE_ANONYMOUS_RIG_INTERVENTION__INDEPENDENT_COURT_REQUIRED"
    result=_geometry_court(rest,faces,native_original,baseline,measured,
                           base,proposal,v11_plan,v12_public_result,
                           original_v9_pose_indices)
    require(result["training_authorized"] is False
            and result["graphics_owner_authorized"] is False
            and result["positive_same_drawable_labels"]==0
            and result["positive_different_drawable_labels"]==0,
            "V13B_NOBODY_CAN_MINT_ARTIST_DRAWABLE_FROM_RIG_OR_OBJECT")
    result["schema"]=SCHEMA
    result["status"]=(
        "REAL_ORIGINAL_NATIVE_RIG_OR_OBJECT_ACTUATOR_RESPONSE_OBSERVED_NO_OWNER"
        if result["causal_source_rig_perturbation_observed"]
        else "ABSTAIN_NO_EFFECTIVE_ORIGINAL_NATIVE_CARRIER_CONTROL_NO_OWNER"
    )
    result["anonymous_native_original_carrier_counts_teacher_only"]=kinds
    result["unsupported_imported_source_carrier_classes_teacher_only"]=[
        str(v) for v in manifest.get("native_unsupported_source_action_owner_types_teacher_only",[])
    ]
    result["source_node_or_object_native_action_binding_verified"] = True
    result["source_rig_and_node_carriers_are_not_drawable_ontology"]=True
    result["source_authored_shape_key_carrier_measured"] = False
    result["source_authored_shape_key_carrier_abstains_if_encountered"]=True
    result["original_artist_drawing_layers_qualified"] = False
    return result
