"""V18q redteams evidence types vs artist owner labels and ungrounded merges."""
import numpy as np
import pytest

from tools.corpus.compile_corpus370_v18a_reactive_staining import (
    component_labels,face_edges,THRESHOLDS
)
from tools.corpus.compile_corpus370_v18q_graph_affinity_hypotheses import (
    SCHEMA,STATUS,graph_proposals
)


def original_source_fixture():
    # Two original source topology islands, each with one shared face edge.
    faces=np.asarray([[0,1,2],[2,1,3],[10,11,12],[12,11,13]],np.int64)
    edges=face_edges(faces)
    channels=np.zeros((4,48),np.float32)
    # Strong on one axis, weak pooled RMS; continuous motion can STILL be
    # one drawable. This must remain candidate evidence, not cannot-link.
    channels[1,:20]=.12
    axes=np.asarray([[1,0],[2,1]],np.int32)
    labels={t:component_labels(channels,edges,t) for t in THRESHOLDS}
    return faces,channels,edges,axes,labels


def test_local_mechanical_evidence_never_mints_graphic_relations():
    data=graph_proposals(*original_source_fixture())
    assert data["source_original_topology_island_count"]==2
    assert data["source_original_local_edge_count"]==2
    assert data["source_signed_response_dimension"]==48
    assert data["source_selected_axes_count"]==2
    assert data["source_artist_same_drawable_pairs_minted"]==0
    assert data["source_artist_different_drawable_pairs_minted"]==0
    for edge in data["source_local_graph_affinity_evidence_ir_teacher_private"]:
        assert edge["can_merge_artist_drawable"]=="UNKNOWN"
        assert edge["must_separate_artist_drawable"]=="UNKNOWN"
        assert edge["original_source_uv_material_boundary"]=="UNOBSERVED"
        assert edge["original_source_cross_view_raster_relation"]=="UNOBSERVED"


def test_orthogonal_axis_signal_not_silently_overwritten_by_pooled_average():
    data=graph_proposals(*original_source_fixture())
    e=data["source_local_graph_affinity_evidence_ir_teacher_private"][0]
    assert e["source_any_axis_above_original_threshold"] is True
    assert e["source_pooled_above_original_threshold"] is False
    variants={x["candidate_key"]:x["metrics"] for x in
              data["source_candidate_drawable_hypotheses_teacher_private"]}
    assert variants["V18A_POOLED_LOCAL_BASELINE"]["K"]==2
    assert variants["SOURCE_TOPOLOGY_ONLY_CONTROL"]["K"]==2
    assert variants["MULTI_AXIS_OR_CUT_STRESS_ONLY"]["K"]==3
    assert data["source_local_split_edges_admitted"]==0
    assert data["source_axis_cut_stress_not_a_calibrated_cannot_link"] is True


def test_lifted_nonlocal_same_response_proposal_never_implies_merge():
    data=graph_proposals(*original_source_fixture())
    assert data["source_original_lifted_candidate_count"]==1
    l=data["source_nonlocal_graph_affinity_hypotheses_teacher_private"][0]
    assert l["source_mean_mechanical_response_distance"]<.1
    assert l["can_merge_artist_drawable"]=="UNKNOWN"
    assert l["must_separate_artist_drawable"]=="UNKNOWN"
    assert l["source_appearance_independent_confirmation_required"] is True
    assert l["original_source_spatial_surface_proximity"]=="UNOBSERVED"
    assert data["source_nonlocal_merge_edges_admitted"]==0


def test_distant_response_islands_not_proposed_as_same_owner():
    faces,channels,edges,axes,_=original_source_fixture()
    channels[2:,:]=2.
    labels={t:component_labels(channels,edges,t) for t in THRESHOLDS}
    data=graph_proposals(faces,channels,edges,axes,labels)
    assert data["source_original_lifted_candidate_count"]==0
    assert data["source_artist_different_drawable_pairs_minted"]==0
    # No lifted similarity does NOT mean these two are different drawables.


def test_source_cannot_introduce_any_geometry_truth_not_present_in_input():
    data=graph_proposals(*original_source_fixture())
    faces=data["source_face_observation_ir_teacher_private"]
    assert faces["original_face_uv_material_addresses"]=="MISSING_IN_THIS_SOURCE_INPUT"
    assert faces["original_face_raster_visibility"]=="MISSING_IN_THIS_SOURCE_INPUT"
    assert faces["source_face_artist_owner"]=="UNKNOWN"


def test_graph_proposals_independent_of_any_heldout_edit_or_full_rgba():
    args=original_source_fixture()
    d1=graph_proposals(*args)
    d2=graph_proposals(*args)
    assert d1==d2
    assert "heldout" not in str(d1).lower()
    assert d1["student_target_admitted"] is False


def test_forged_v18a_labels_fail_closed():
    faces,features,edges,axes,labels=original_source_fixture()
    labels[.1]=np.arange(len(faces),dtype=np.int32)
    with pytest.raises(ValueError,match="V18Q_ORIGINAL_FIT_ONLY"):
        graph_proposals(faces,features,edges,axes,labels)


@pytest.mark.parametrize("attack",("wrong_graph","wrong_axis_dim","nan","face_census"))
def test_invalid_original_source_graph_refused(attack):
    faces,features,edges,axes,labels=original_source_fixture()
    if attack=="wrong_graph":edges=np.asarray([[0,1],[1,2]],np.int64)
    elif attack=="wrong_axis_dim":axes=axes[:1]
    elif attack=="nan":features[0,0]=np.nan
    else:faces=faces[:3]
    with pytest.raises(ValueError,match="V18Q_ORIGINAL_FIT_ONLY"):
        graph_proposals(faces,features,edges,axes,labels)


def test_schema_and_epistemic_status_persistent():
    assert SCHEMA.endswith(".v18q")
    assert STATUS.endswith("ARTIST_OWNER_UNKNOWN")

from tools.corpus.court_corpus370_v18q_graph_affinity_hypotheses import (
    verify_ir,
)


def valid_ir():
    graph=graph_proposals(*original_source_fixture())
    return {
      "schema":SCHEMA,"status":STATUS,
      "artist_graphics_ownership":"UNKNOWN",
      "compiler_graphics_proof":"NOT_QUALIFIED",
      "source_original_face_uv_material_relation":
           "NOT_PRESENT_IN_SHA_BOUND_V18A_ARRAYS",
      "source_original_multiview_visibility_relation":
           "NOT_PRESENT_IN_SHA_BOUND_V18A_ARRAYS",
      "same_drawable_training_pairs_qualified":0,
      "different_drawable_training_pairs_qualified":0,
      "graphics_owner_authorized":False,
      "iris_training_authorized":False,
      "tessa_training_authorized":False,
      "materialization_authorized":False,
      "training_authorized":False,
      "source_private_graph":graph,
      "independent_edit_court_observations_not_for_fitting":{
         "source_v18o_edit_outcome_used_to_rank_hypotheses":False,
         "source_edit_consequence_for_new_graph_hypotheses":"NOT_EVALUATED",
         "original_source_cross_pose_graphics_owner_stability":"UNKNOWN",
      },
    }


def test_independent_replay_and_no_owner_label_admit():
    original=valid_ir()
    assert verify_ir(original,original)


@pytest.mark.parametrize("attack",(
 "graph_mutation","promote_owner","invent_uv","claim_heldout",
 "cannot_link","make_lifted_same","edit_rank_leakage",
))
def test_independent_court_detects_claim_forgery(attack):
    import copy
    original=valid_ir()
    fake=copy.deepcopy(original)
    if attack=="graph_mutation":
        fake["source_private_graph"]["source_original_local_edge_count"]+=1
    elif attack=="promote_owner":
        fake["graphics_owner_authorized"]=True
    elif attack=="invent_uv":
        fake["source_original_face_uv_material_relation"]="VERIFIED"
    elif attack=="claim_heldout":
        fake["independent_edit_court_observations_not_for_fitting"][
           "source_edit_consequence_for_new_graph_hypotheses"]="PASS"
    elif attack=="cannot_link":
        fake["source_private_graph"][
           "source_local_graph_affinity_evidence_ir_teacher_private"][0][
           "must_separate_artist_drawable"]="TRUE"
    elif attack=="make_lifted_same":
        fake["source_private_graph"][
           "source_nonlocal_graph_affinity_hypotheses_teacher_private"][0][
           "can_merge_artist_drawable"]="TRUE"
    else:
        fake["independent_edit_court_observations_not_for_fitting"][
           "source_v18o_edit_outcome_used_to_rank_hypotheses"]=True
    with pytest.raises(ValueError,match="V18Q_INDEPENDENT_"):
        verify_ir(fake,original)
