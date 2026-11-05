"""V13 synthetic source actuator/root/null/sham and false-owner negative tests.

Synthetic controls only exercise contracts, not proof about real source rigs.
"""
import copy
import numpy as np
import pytest
from tools.corpus.court_corpus370_native_actuator_mechanics_v13 import evaluate

@pytest.fixture(scope="module")
def evidence():
    rng=np.random.default_rng(31415)
    rest=rng.normal(size=(40,3)).astype(np.float64)
    faces=np.stack([np.arange(36),np.arange(36)+1,np.arange(36)+2],axis=1)
    native=np.stack([rest+np.array([.02*i,0.,0.]) for i in range(3)])
    baseline=native.astype(np.float32)
    tensor=np.broadcast_to(baseline[:,None,None,None,:,:],
                           (3,2,3,2,40,3)).copy()
    for pose in range(3):
        for axis in range(3):
            for sign_index,sign in enumerate((1.,-1.)):
                tensor[pose,0,axis,sign_index,:18,axis]+=.055*sign
                tensor[pose,1,axis,sign_index,:,axis]+=.055*sign
    proposal={"source_sha256":"a"*64,
              "candidates":[{"slot_count":n,
                  "face_slot_teacher_only":[min(n-1,i//max(1,36//n))
                                           for i in range(36)]}
                  for n in (8,16,32)]}
    v11={"source_sha256":"a"*64,"subject_id":"synthetic",
         "training_authorized":False,"graphics_owner_authorized":False,
         "new_source_face_intervention_query_count":12}
    v12={"source_sha256":"a"*64,
         "same_original_authored_v9_poses_teacher_only":[0,1,2],
         "qualified_source_vs_soup_pose_camera_cases":12,
         "render_arms":912}
    manifest={"schema":"RealSaS.Corpus370NativeAnonymousActuatorInterventionRender.v13",
              "status":"V13_REAL_ORIGINAL_AUTHORED_SOURCE_ANONYMOUS_RIG_INTERVENTION__INDEPENDENT_COURT_REQUIRED",
              "source_sha256":"a"*64,"subject_id":"synthetic",
              "source_rest_vertex_count":40,"source_face_count":36,
              "source_controls_measured_count":2,"source_control_inventory_count":2,
              "source_control_inventory_truncated":False,
              "original_source_teacher_pose_count":3,
              "original_selected_pose_indices_teacher_only":[0,1,2],
              "axes":["X","Y","Z"],"rotation_degrees":5.,
              "anonymous_control_count_inference_target":False,
              "native_source_actions_as_student_input":False,
              "no_anon_control_intervention_result_as_drawable_truth":True,
              "artist_same_drawable_positive_pairs":0,
              "artist_different_drawable_negative_pairs":0,
              "graphics_owner_authorized":False,"training_authorized":False,
              "all_control_restoration_noop_max_error":0.}
    return rest,faces,native,baseline,tensor,manifest,proposal,v11,v12

def test_native_local_vs_root_control_without_owner_admission(evidence):
    result=evaluate(*evidence)
    count=result["source_native_anonymous_mechanics_counts"]
    assert result["status"]=="REAL_NATIVE_ANONYMOUS_RIG_INTERVENTIONS_OBSERVED_NO_OWNER"
    assert count["source_effect_nonzero_count"]==18
    assert count["local_nonrigid_effect_count"]==9
    assert count["root_like_or_globally_rigid_effect_count"]==9
    assert result["analytic_uniform_root_translation_sham"]["null_zero_max"]==0.
    assert result["analytic_uniform_root_translation_sham"]["root_uniform_translation_rigid_fit_rms"]<1.e-7
    assert result["positive_same_drawable_labels"]==0
    assert result["positive_different_drawable_labels"]==0
    assert result["graphics_owner_authorized"] is False
    assert result["training_authorized"] is False

@pytest.mark.parametrize("kind",[
    "forged_owner","forged_training","forged_same","forged_different",
    "bone_action_student_input","controls_as_model_outputs","fake_rig_labels",
    "source_sha","bad_v11_parent","wrong_pose_index","bad_shape",
    "wrong_axes","wrong_rotation","bad_face_domain","not_original_native",
    "spoof_materialization"
])
def test_corrupt_provenance_or_source_rig_prior_fails_closed(evidence,kind):
    values=list(copy.deepcopy(evidence))
    m=values[5];proposal=values[6];v11=values[7];v12=values[8]
    if kind=="forged_owner":m["graphics_owner_authorized"]=True
    elif kind=="forged_training":m["training_authorized"]=True
    elif kind=="forged_same":m["artist_same_drawable_positive_pairs"]=1
    elif kind=="forged_different":m["artist_different_drawable_negative_pairs"]=1
    elif kind=="bone_action_student_input":m["native_source_actions_as_student_input"]=True
    elif kind=="controls_as_model_outputs":m["anonymous_control_count_inference_target"]=True
    elif kind=="fake_rig_labels":m["no_anon_control_intervention_result_as_drawable_truth"]=False
    elif kind=="source_sha":m["source_sha256"]="x"*64
    elif kind=="bad_v11_parent":v11["training_authorized"]=True
    elif kind=="wrong_pose_index":m["original_selected_pose_indices_teacher_only"]=[0,2,1]
    elif kind=="bad_shape":values[4]=values[4][...,:-1,:]
    elif kind=="wrong_axes":m["axes"]=["X","Y","head"]
    elif kind=="wrong_rotation":m["rotation_degrees"]=25.
    elif kind=="bad_face_domain":values[1]=values[1].copy();values[1][0,0]=999
    elif kind=="not_original_native":values[3]=values[3].copy();values[3][0,0,0]+=2.
    elif kind=="spoof_materialization":v12["qualified_source_vs_soup_pose_camera_cases"]=0
    with pytest.raises((ValueError,KeyError,IndexError)):
        evaluate(*values)

def test_no_effective_source_rig_control_abstains(evidence):
    values=list(copy.deepcopy(evidence))
    values[4]=np.broadcast_to(values[3][:,None,None,None,:,:],
                               values[4].shape).copy()
    out=evaluate(*values)
    assert out["status"]=="ABSTAIN_NO_EFFECTIVE_NATIVE_RIG_CONTROLS_NO_OWNER"
    assert out["source_native_anonymous_mechanics_counts"]["no_source_effect_abstain_count"]==18
    assert out["training_authorized"] is False

def test_root_global_translation_is_not_drawable_split(evidence):
    out=evaluate(*evidence)
    assert out["analytic_uniform_root_translation_sham"]["root_control_sham_source_geometry_only"] is True
    assert out["source_bone_names_or_ontology_as_inference_target"] is False
    assert out["source_generated_control_perturbation_is_art_author_label"] is False
