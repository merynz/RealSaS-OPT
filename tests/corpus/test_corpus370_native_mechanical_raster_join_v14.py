"""V14 geometric precommit and independent original RGBA court synthetic contracts.

These tests CANNOT admit any real artist owner/source CAA or training example.
"""
import copy
import numpy as np
import pytest
from tools.corpus.corpus370_native_raster_join_plan_v14 import select,arms_for_plan
from tools.corpus.court_corpus370_mechanical_raster_join_v14 import evaluate

def native():
    rng=np.random.default_rng(42)
    rest=rng.normal(size=(12,3))
    base=np.stack([rest+np.array([.02*i,0,0]) for i in range(3)])
    pert=np.broadcast_to(base[:,None,None,None,:,:],(3,2,3,2,12,3)).copy()
    for p in range(3):
        for a in range(3):
            for si,sgn in enumerate((1.,-1.)):
                pert[p,0,a,si,:6,a]+=.1*sgn
                pert[p,1,a,si,:,a]+=.1*sgn
    return rest,base,pert

def evidence():
    r,b,p=native()
    plan=select(r,b,p,[0,1,2],2,[0,1,2])
    assert len(plan["controls"])==2
    arms,_=arms_for_plan(plan)
    rgba=np.zeros((3,4,len(arms),192,192,4),np.float32)
    ix={name:i for i,name in enumerate(arms)}
    rgba[:,:,:,30:100,40:110,3]=1.
    rgba[:,:,:,30:100,40:110,:3]=.2
    for ci in range(2):
        for sign in ("PLUS","MINUS"):
            name=f"SOURCE_C{ci:02d}_{sign}"
            soup=f"SOUP_C{ci:02d}_{sign}"
            active=f"ACTIVE_C{ci:02d}_{sign}"
            rgba[:,:,ix[name],40:55,50:65,0]=.8
            rgba[:,:,ix[soup],40:55,50:65,0]=.8
            rgba[:,:,ix[active],40:55,50:65,0]=.8
    rgba[:,:,ix["ROOT_SHAM_ORIGINAL"],20:28,30:42,3]=1.
    rgba[:,:,ix["ROOT_SHAM_SOUP"],20:28,30:42,3]=1.
    prior={
        "source_sha256":"a"*64,
        "source_controls_measured_count":2,
        "native_control_carrier_types_teacher_only":{"POSE_BONE":1,"IMPORTED_ANIMATED_OBJECT":1},
    }
    batch={
        "subject_id":"synthetic",
        "native_intervention_manifest_sha256":"c"*64,
        "native_vertex_response_npz_sha256":"d"*64,
        "independent_mechanics_court_sha256":"e"*64,
        "training_authorized":False,
        "graphics_owner_authorized":False
    }
    public={
        "subject_id":"synthetic","original_source_sha256":"a"*64,
        "raw_vertex_response_npz_sha256":"d"*64,
        "independent_court_sha256":"e"*64,
        "source_controls_total":2,
        "local_nonrigid_cases":1,
        "nonreversible_cases":0
    }
    render={
        "schema":"RealSaS.Corpus370SourceNativeMechanicalRasterJointRender.v14",
        "status":"V14_REAL_SAME_CAUSAL_SOURCE_NATIVE_CONTROL_VERTEX_AND_RGBA__INDEPENDENT_COURT_REQUIRED",
        "selected_source_controls":plan,"rendered_arms":arms,
        "rendered_pose_indices":[0,1,2],"rendered_source_camera_indices":[0,2,4,6],
        "raster_resolution":192,"cycles_samples":32,
        "native_poses_exact_source_author_actions":True,
        "original_source_rest_and_animated_face_identity":True,
        "source_pose_matches_original_v13b_vertex_oracle":True,
        "perturbed_source_pose_matches_original_v13b_native_control":True,
        "source_active_faces_are_not_artist_drawables":True,
        "source_bone_mesh_material_names_as_drawable_labels":False,
        "original_source_action_name_as_model_feature":False,
        "rgba_and_visibility_only_not_actual_draw_order_proof":True,
        "root_sham":"CAMERA_NEGATIVE_TRANSLATION_EQUIVALENT_TO_POSITIVE_GLOBAL_SOURCE_TRANSLATION_NOT_AUTHORED_ROOT",
        "artist_same_drawable_pairs":0,"artist_different_drawable_pairs":0,
        "native_full_resolution_caa_authorized":False,
        "graphics_owner_authorized":False,"training_authorized":False,
        "source_sha256":"a"*64,"subject_id":"synthetic",
        "source_private_v13b_intervention_array_sha256":"d"*64,
        "source_private_v13b_intervention_manifest_sha256":"c"*64,
        "original_source_control_carrier_kinds_teacher_only":prior["native_control_carrier_types_teacher_only"]
    }
    return rgba,render,plan,prior,batch,public

def test_source_mechanics_raster_precommit_and_same_image_court_but_no_artist_owner():
    args=evidence()
    result=evaluate(*args)
    assert result["status"]=="V14_REAL_CAUSAL_SOURCE_MECHANICAL_TO_RGBA_JOIN_QUALIFIED__OWNER_UNKNOWN"
    assert result["baseline_source_vs_soup_and_root_noop_qualified_pose_views"]==12
    assert result["original_native_perturbation_vs_source_soup_qualified_camera_cases"]==48
    assert result["original_native_perturbation_vs_source_soup_total_camera_cases"]==48
    assert all(x>0 for x in result["whole_source_perturbation_effective_pixel_instances_by_selected_control"])
    assert result["artist_same_drawable_pairs"]==result["artist_different_drawable_pairs"]==0
    assert result["training_authorized"] is False
    assert result["graphics_owner_authorized"] is False

def test_blind_control_selection_reproducible_from_original_vertex_source_only():
    r,b,p=native()
    a=select(r,b,p,[0,1,2],2,[0,1,2])
    z=select(r,b,p,[0,1,2],2,[0,1,2])
    assert a==z
    assert a["raster_pixels_seen_during_control_selection"] is False
    assert a["controls"][0]["source_response_class_teacher_only"]=="LOCAL_NONRIGID"
    assert a["source_bone_or_object_identity_as_artist_target"] is False
    assert len(arms_for_plan(a)[0])==5+len(a["controls"])*7

@pytest.mark.parametrize("attack",[
    "wrong_source","wrong_manifest","false_training","false_owner",
    "forged_poses","camera_injection","artist_identity","source_raster_not_equal",
    "fake_parent","forged_controls","false_CAA","source_layer_name",
    "illegal_rgba","false_noop","wrong_root"
])
def test_v14_false_truth_forgery_and_lineage_fail_closed(attack):
    a=list(copy.deepcopy(evidence()))
    x,r,plan,prior,batch,public=a
    if attack=="wrong_source":r["source_sha256"]="b"*64
    elif attack=="wrong_manifest":r["source_private_v13b_intervention_manifest_sha256"]="b"*64
    elif attack=="false_training":r["training_authorized"]=True
    elif attack=="false_owner":r["graphics_owner_authorized"]=True
    elif attack=="forged_poses":r["rendered_pose_indices"]=[1,2,3]
    elif attack=="camera_injection":r["rendered_source_camera_indices"]=[0,1,2,3]
    elif attack=="artist_identity":r["artist_same_drawable_pairs"]=1
    elif attack=="source_raster_not_equal":
        x[:,:,r["rendered_arms"].index("SOUP_BASE"),25:140,30:150,0]=1
        result=evaluate(*a)
        assert result["all_original_source_camera_cases_passed"] is False
        return
    elif attack=="fake_parent":batch["independent_mechanics_court_sha256"]="f"*64
    elif attack=="forged_controls":plan["controls"][0]["training_authorized"]=True
    elif attack=="false_CAA":r["native_full_resolution_caa_authorized"]=True
    elif attack=="source_layer_name":r["source_bone_mesh_material_names_as_drawable_labels"]=True
    elif attack=="illegal_rgba":x[...,3]=float("nan")
    elif attack=="false_noop":
        x[:,:,r["rendered_arms"].index("SOUP_NOOP"),10:150,10:150,0]=1
        assert evaluate(*a)["all_original_source_camera_cases_passed"] is False
        return
    elif attack=="wrong_root":r["root_sham"]="AUTHOR_ROOT_PROVED"
    with pytest.raises((ValueError,AssertionError,KeyError)):
        evaluate(*a)

def test_no_real_visible_change_remains_abstain_not_train():
    args=list(evidence())
    x,render,plan,prior,batch,public=args
    for ci in range(2):
        for sign in ("PLUS","MINUS"):
            for t in ("SOURCE","SOUP","ACTIVE"):
                x[:,:,render["rendered_arms"].index(f"{t}_C{ci:02d}_{sign}")]=x[:,:,render["rendered_arms"].index("ORIGINAL_BASE")]
    result=evaluate(*args)
    assert result["all_original_source_camera_cases_passed"] is False
    assert result["training_authorized"] is False


def test_replay_blender_numpy_source_geometry_roundoff_keeps_opaque_control_ids():
    from tools.corpus.corpus370_native_raster_join_plan_v14 import verify_locked_source_selection
    rest,baseline,interventions=native()
    plan=select(rest,baseline,interventions,[0,1,2],2,[0,1,2])
    replay=copy.deepcopy(plan)
    replay["controls"][0]["source_global_rigid_residual"]+=7e-11
    assert verify_locked_source_selection(replay,plan)
    for mutation in ("control","class","training","numeric","control_count","source_readset"):
        bad=copy.deepcopy(replay)
        if mutation=="control":
            bad["controls"][0]["control_index_teacher_only"]+=1
        elif mutation=="class":
            bad["controls"][0]["source_response_class_teacher_only"]="ARTIST_LAYER"
        elif mutation=="training":
            bad["training_authorized"]=True
        elif mutation=="numeric":
            bad["controls"][0]["source_global_rigid_residual"]+=1e-4
        elif mutation=="control_count":
            bad["selected_control_count"]+=1
        elif mutation=="source_readset":
            bad["raster_pixels_seen_during_control_selection"]=True
        with pytest.raises(ValueError):
            verify_locked_source_selection(bad,plan)
