"""V8 synthetic independent native-pose group correspondence contract tests.

Do not turn these synthetic tests into real source/pose authorizations.
"""
import copy
import numpy as np
import pytest

from tools.corpus.court_corpus370_native_pose_partition_v8 import evaluate

S="a"*64
REST="b"*64
NATIVE="c"*64
RECEIPT="d"*64


def example():
    nfaces=32
    vertex=np.zeros((3*nfaces,3),dtype=np.float64)
    faces=np.arange(3*nfaces,dtype=np.int64).reshape(nfaces,3)
    for i in range(nfaces):
        x=(i%8)*2.
        y=(i//8)*2.
        vertex[3*i:3*i+3]=[[x,y,0],[x+0.8,y,0],[x,y+0.9,0]]
    poses=np.stack([vertex.copy() for _ in range(4)])
    poses[1,:,0]+=.15
    poses[2,:,1]+=.1
    poses[3,:,0]+=.2
    poses[3,4,2]+=.06
    original={"rest_vertices_source_world":vertex.copy(),"faces":faces.copy()}
    authored={"rest_vertices_source_world":vertex.copy(),"faces":faces.copy(),
              "posed_vertices_source_world":poses}
    proposal={
        "schema":"RealSaS.Corpus370AllViewGeometryOperationalGroups.v3",
        "status":"ALL_8_INPUT_VIEW_GEOMETRY_LAYER_HYPOTHESES__RENDER_OUTCOME_HELDOUT",
        "source_sha256":S,"source_face_count":nfaces,
        "training_authorized":False,"graphics_owner_truth_minted":False,
        "view4_rendered_edit_consequences_used_by_compiler":False,
        "candidates":[{
            "slot_count":size,"candidate":f"ALLVIEW_GEOMETRY_GROUPS_{size}",
            "face_slot_teacher_only":[i//(nfaces//size) for i in range(nfaces)],
            "source_faces_changed":False,"training_authorized":False,
        } for size in (8,16,32)]
    }
    pairs={
        "parents":{"source_sha256":S,"rest_geometry_sha256":REST,
                   "authored_teacher_sha256":NATIVE,
                   "authored_teacher_receipt_sha256":RECEIPT},
        "training_authorized":False,
        "rows":[
            {"view_index":0,"source_face_a_teacher_only":0,
             "source_face_b_teacher_only":8},
            {"view_index":4,"source_face_a_teacher_only":4,
             "source_face_b_teacher_only":12},
        ]
    }
    receipt={
        "source":{"sha256":S},"output":{"sha256":NATIVE},
        "teacher_only":True,
        "teacher_rig_skin_motion_allowed_as_inference_input":False,
        "sampled_frames":[{"action":"idle","frame":i} for i in (0,2)]+[
            {"action":"run","frame":i} for i in (0,3)],
        "native_motion_inventory_exported":["idle","run"],
        "sampled_frame_count":4,
    }
    court={
        "source_sha256":S,
        "status":"PASS_SOURCE_AUTHORED_TEMPORAL_GEOMETRY_CANDIDATE__CAUSAL_RESPONSE_SEPARATE",
        "teacher_arrays":{"sha256":NATIVE},
        "teacher_manifest":{"sha256":RECEIPT},
        "counts":{"sampled_frame_count":4},
        "training_authorized":False,
        "family_authority_minted":False,
    }
    return original,authored,proposal,pairs,receipt,court


def run(d):
    return evaluate(*d,rest_sha256=REST,teacher_sha256=NATIVE,
                    teacher_receipt_sha256=RECEIPT,source_sha256=S)


def test_native_pose_group_correspondence_keeps_unknown_labels():
    v=run(example())
    assert v["authored_pose_count"]==4
    assert v["authored_action_count"]==2
    assert v["same_take_temporal_edges"]==2
    assert v["source_authored_rest_exact_face_order"] is True
    assert v["available_candidate_sizes"]==[8,16,32]
    assert v["unavailable_requested_sizes"]==[64]
    assert v["by_candidate_size"]["8"]["group_count"]==8
    assert len(v["by_candidate_size"]["32"]["group_observations_teacher_only"])==32
    assert all(g["native_pose_group_source_face_ids_persistent"] is True
               for g in v["by_candidate_size"]["8"]["group_observations_teacher_only"])
    assert v["whole_source_max_vertex_motion_over_extent"]>0
    assert v["by_candidate_size"]["8"][
        "max_group_local_rigid_fit_rmse_over_source_extent"]>0
    assert v["animated_source_original_vs_group_soup_image_parity_courted"] is False
    assert v["native_resolution_caa_authorized"] is False
    assert v["same_drawable_positive_pairs"]==0
    assert v["different_drawable_negative_pairs"]==0
    assert v["training_authorized"] is False


@pytest.mark.parametrize("fault",[
    "source_changed","rest_faces_reindexed","teacher_faces_reindexed",
    "rest_drift","nan_pose","wrong_pose_shape",
    "fake_manifest_inference","forged_teacher_sha",
    "court_promoted","missing_action","duplicate_native_frame",
    "group_face_label_invalid","broken_nested_hierarchy","bad_rest_sha",
])
def test_all_foreign_or_forged_source_contracts_fail_closed(fault):
    d=list(copy.deepcopy(example()))
    if fault=="source_changed":d[2]["source_sha256"]="f"*64
    elif fault=="rest_faces_reindexed":d[0]["faces"][1]=d[0]["faces"][0]
    elif fault=="teacher_faces_reindexed":d[1]["faces"][1]=d[1]["faces"][0]
    elif fault=="rest_drift":d[1]["rest_vertices_source_world"][0,0]+=1.
    elif fault=="nan_pose":d[1]["posed_vertices_source_world"][0,0,0]=np.nan
    elif fault=="wrong_pose_shape":d[1]["posed_vertices_source_world"]=d[1]["posed_vertices_source_world"][:2]
    elif fault=="fake_manifest_inference":d[4]["teacher_rig_skin_motion_allowed_as_inference_input"]=True
    elif fault=="forged_teacher_sha":d[5]["teacher_arrays"]["sha256"]="f"*64
    elif fault=="court_promoted":d[5]["training_authorized"]=True
    elif fault=="missing_action":d[4]["native_motion_inventory_exported"]=["idle"]
    elif fault=="duplicate_native_frame":d[4]["sampled_frames"][1]["frame"]=0
    elif fault=="group_face_label_invalid":d[2]["candidates"][0]["face_slot_teacher_only"][0]=99
    elif fault=="broken_nested_hierarchy":
        labels=d[2]["candidates"][1]["face_slot_teacher_only"]
        labels[1],labels[12]=labels[12],labels[1]
    elif fault=="bad_rest_sha":d[3]["parents"]["rest_geometry_sha256"]="f"*64
    with pytest.raises(ValueError):
        run(tuple(d))


def test_source_pose_collapses_are_diagnostics_not_owner_negatives():
    d=list(example())
    # Actual geometry can collapse at a teacher-authored frame, even if it
    # originated from the same source and material; no false owner labels.
    f=d[1]["faces"][0]
    d[1]["posed_vertices_source_world"][1,f[2]] = d[1][
        "posed_vertices_source_world"][1,f[1]]
    v=run(tuple(d))
    assert v["whole_source_collapsed_face_pose_instances"]>0
    assert v["by_candidate_size"]["8"][
        "source_face_collapses_are_author_drawable_failures"] is False
    assert v["graphics_owner_authorized"] is False
