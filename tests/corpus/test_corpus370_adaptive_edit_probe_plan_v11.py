"""Synthetic heldout geometry-only V11 source edit query experiments."""
import copy
import numpy as np
import pytest
from tools.corpus.corpus370_adaptive_edit_probe_plan_v11 import build_adaptive_plan
from tools.corpus.corpus370_partition_edit_plan_v7 import build_plan

SOURCE="a"*64
REST="b"*64

def evidence():
    n=32
    proposal={
        "schema":"RealSaS.Corpus370AllViewGeometryOperationalGroups.v3",
        "status":"ALL_8_INPUT_VIEW_GEOMETRY_LAYER_HYPOTHESES__RENDER_OUTCOME_HELDOUT",
        "source_sha256":SOURCE,"source_face_count":n,
        "training_authorized":False,"graphics_owner_truth_minted":False,
        "view4_rendered_edit_consequences_used_by_compiler":False,
        "candidates":[{
            "slot_count":k,"candidate":f"ALLVIEW_GEOMETRY_GROUPS_{k}",
            "source_faces_changed":False,"training_authorized":False,
            "face_slot_teacher_only":[i//(n//k) for i in range(n)]
        } for k in (8,16,32)]}
    pairs={
        "parents":{"source_sha256":SOURCE,"rest_geometry_sha256":REST},
        "training_authorized":False,
        "rows":[
            {"view_index":0,"source_face_a_teacher_only":0,"source_face_b_teacher_only":8},
            {"view_index":4,"source_face_a_teacher_only":4,"source_face_b_teacher_only":12},
        ]}
    plan=build_plan(proposal,pairs)
    v9={
        "schema":"RealSaS.Corpus370NativeAnimatedPartitionSourceRender.v9",
        "subject_id":"synthetic-only","source_sha256":SOURCE,
        "plan":plan,"teacher_only":True,
        "graphics_owner_authorized":False,"training_authorized":False,
    }
    prior={
        "schema":"RealSaS.Corpus370NativeAnimatedEditResponseEquivalenceCourt.v10",
        "status":"REAL_NATIVE_EDIT_RESPONSE_PAIRWISE_EQUIVALENCE_MEASURED__ARTIST_OWNER_UNKNOWN",
        "subject_id":v9["subject_id"],"source_sha256":SOURCE,
        "candidate_sizes":[8,16,32],
        "artist_same_drawable_positive_pairs":0,
        "artist_different_drawable_negative_pairs":0,
        "graphics_owner_authorized":False,"training_authorized":False,
        "finest_partition_is_artist_truth":False,
        "pairwise_equivalence_classes_scoped_to_tested_source_edits":[{
            "candidate_a":a,"candidate_b":b,
            "total_source_edit_task_count":len(plan["tasks"]),
            "indistinguishable_source_edit_task_count":
                len(plan["tasks"]) if (a,b)==(16,32) else 0,
            "graphics_owner_authorized":False,"training_authorized":False
        } for a,b in ((8,16),(8,32),(16,32))]
    }
    vertices=np.zeros((n*3,3),np.float64)
    faces=np.arange(n*3,dtype=np.int64).reshape(n,3)
    for i in range(n):
        x=float(i%8);y=float(i//8)
        vertices[i*3:i*3+3]=[[x,y,0],[x+.8,y,0],[x,y+.7,0]]
    rest={"rest_vertices_source_world":vertices,"faces":faces}
    return proposal,pairs,v9,prior,rest

def run(data,budget=12):
    return build_adaptive_plan(*data,source_rest_sha256=REST,max_queries=budget)

def test_new_targets_are_deterministic_heldout_and_without_owner_labels():
    e=evidence()
    result=run(e)
    assert result["new_source_face_intervention_query_count"]==12
    assert result["source_finest_heldout_valid_leaf_count"]==28
    assert result["prior_finest_group_ids_excluded_count"]==4
    assert result["potential_splittable_parent_events"]==24
    assert result["selected_splittable_parent_events"]>0
    assert result["prior_edit_indistinguishable_candidate_size_pairs"]==[
        {"coarse_size":16,"fine_size":32}]
    already={e[0]["candidates"][-1]["face_slot_teacher_only"][x["source_face_teacher_only"]]
             for x in e[2]["plan"]["tasks"]}
    new=result["query_targets_source_teacher_only"]
    assert {x["finest_group_id_teacher_only"] for x in new}.isdisjoint(already)
    assert all(x["same_drawable"]=="UNKNOWN"
               and x["different_drawable"]=="UNKNOWN"
               and x["training_authorized"] is False for x in new)
    assert result["graphics_owner_authorized"] is False
    assert result["real_rgba_consequences_measured_for_heldout_queries"] is False
    assert result["training_authorized"] is False
    assert run(copy.deepcopy(e))["query_targets_source_teacher_only"]==new

@pytest.mark.parametrize("fault",[
    "source_mismatch","rest_sha_mismatch","fake_group_truth","fake_v10_training",
    "fake_v10_owner","fake_artist_pos","fake_artist_neg","fake_finest_truth",
    "missing_candidate_pair","invalid_edit_task_count","duplicate_candidate_pair",
    "non_nested_group","rest_face_order_changed","nonfinite_rest",
    "invalid_budget","v9_wrong_parent","unusable_source_triangle",
    "negative_equivalence_count","overfull_equivalence_count",
])
def test_forgeries_non_nested_source_and_unobservable_geometry_fail_closed(fault):
    e=list(copy.deepcopy(evidence()))
    budget=12
    if fault=="source_mismatch":e[0]["source_sha256"]="f"*64
    elif fault=="rest_sha_mismatch":e[1]["parents"]["rest_geometry_sha256"]="f"*64
    elif fault=="fake_group_truth":e[0]["graphics_owner_truth_minted"]=True
    elif fault=="fake_v10_training":e[3]["training_authorized"]=True
    elif fault=="fake_v10_owner":e[3]["graphics_owner_authorized"]=True
    elif fault=="fake_artist_pos":e[3]["artist_same_drawable_positive_pairs"]=1
    elif fault=="fake_artist_neg":e[3]["artist_different_drawable_negative_pairs"]=1
    elif fault=="fake_finest_truth":e[3]["finest_partition_is_artist_truth"]=True
    elif fault=="missing_candidate_pair":e[3]["pairwise_equivalence_classes_scoped_to_tested_source_edits"].pop()
    elif fault=="invalid_edit_task_count":e[3]["pairwise_equivalence_classes_scoped_to_tested_source_edits"][0]["total_source_edit_task_count"]=1
    elif fault=="duplicate_candidate_pair":e[3]["pairwise_equivalence_classes_scoped_to_tested_source_edits"].append(
        copy.deepcopy(e[3]["pairwise_equivalence_classes_scoped_to_tested_source_edits"][0]))
    elif fault=="non_nested_group":
        labels=e[0]["candidates"][1]["face_slot_teacher_only"]
        labels[2],labels[24]=labels[24],labels[2]
    elif fault=="rest_face_order_changed":e[4]["faces"][0]=100000
    elif fault=="nonfinite_rest":e[4]["rest_vertices_source_world"][1,1]=np.nan
    elif fault=="invalid_budget":budget=2
    elif fault=="v9_wrong_parent":e[2]["plan"]["source_sha256"]="z"*64
    elif fault=="unusable_source_triangle":e[4]["rest_vertices_source_world"][:]=0
    elif fault=="negative_equivalence_count":
        e[3]["pairwise_equivalence_classes_scoped_to_tested_source_edits"][0][
            "indistinguishable_source_edit_task_count"]=-1
    elif fault=="overfull_equivalence_count":
        e[3]["pairwise_equivalence_classes_scoped_to_tested_source_edits"][0][
            "indistinguishable_source_edit_task_count"]=len(e[2]["plan"]["tasks"])+1
    with pytest.raises(ValueError):
        run(tuple(e),budget)


def test_valid_unsplittable_leaf_is_noninformative_not_subject_failure():
    e=list(evidence())
    # Construct a strictly nested but uneven original geometric partition:
    # one coarse region has a single child and that child one finest leaf.
    # This is valid for arbitrary morphologies; a forced two-way branch prior
    # would incorrectly fail the entire source.
    e[0]["candidates"][1]["face_slot_teacher_only"][30]=14
    for i in (28,29,30):
        e[0]["candidates"][0]["face_slot_teacher_only"][i]=6
    # Source prior V9 plan is now rebound to the same altered hypothetical
    # ORIGINAL group tree, while V10 remains an explicit test-only prior.
    from tools.corpus.corpus370_partition_edit_plan_v7 import build_plan
    e[2]["plan"]=build_plan(e[0],e[1])
    for row in e[3]["pairwise_equivalence_classes_scoped_to_tested_source_edits"]:
        row["total_source_edit_task_count"]=len(e[2]["plan"]["tasks"])
        if (row["candidate_a"],row["candidate_b"])==(16,32):
            row["indistinguishable_source_edit_task_count"]=len(e[2]["plan"]["tasks"])
    result=run(tuple(e))
    assert result["heldout_valid_unsplittable_leaf_count"]>=1
    assert result["new_source_face_intervention_query_count"]==12
    assert 31 not in [t["finest_group_id_teacher_only"]
                      for t in result["query_targets_source_teacher_only"]]
    assert result["graphics_owner_authorized"] is False
    assert result["training_authorized"] is False
