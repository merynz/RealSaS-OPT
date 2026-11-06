"""V18p original source: no topology-only/motion-only artist-owner leakage."""
import numpy as np
import pytest

from tools.corpus.compile_corpus370_v18a_reactive_staining import (
    face_edges,component_labels,THRESHOLDS
)
from tools.corpus.court_corpus370_v18p_response_topology_identifiability import (
    SCHEMA,STATUS,source_audit,canonical_labels
)


def source(contrast=False):
    # Two disconnected source surface islands, each internally edge-connected.
    faces=np.asarray([
        [0,1,2],[2,1,3],[10,11,12],[12,11,13]
    ],dtype=np.int64)
    edges=face_edges(faces)
    assert np.array_equal(edges,np.asarray([[0,1],[2,3]]))
    features=np.zeros((4,48),dtype=np.float32) # two 20D control axes + 8D joint
    if contrast:
        features[1,:20]=1.
    labels={t:component_labels(features,edges,t) for t in THRESHOLDS}
    return faces,features,edges,labels


def test_disconnected_topology_floor_even_if_motion_identical():
    faces,f,e,labels=source(False)
    row=source_audit(faces,f,e,2,labels)
    assert row["original_source_topology_island_profile"]["component_count"]==2
    assert row["original_source_topology_island_pair_count"]==1
    assert all(k==1 for k in
        row["original_source_different_island_mean_response_near_pairs"].values())
    for profile in row["original_source_threshold_diagnostics"]:
        assert profile["original_V18a_candidates"]["component_count"]==2
        assert profile["original_candidate_count_above_topology_floor"]==0
        assert profile["original_fraction_candidate_count_from_topology_floor"]==1.
        assert profile["source_semantic_labels_used"] is False
        assert profile["treat_topology_island_boundary_as_art_drawable_edge"] is False


def test_multi_axis_response_cut_adds_components_above_topology_floor():
    faces,f,e,labels=source(True)
    row=source_audit(faces,f,e,2,labels)
    assert row["original_source_topology_island_profile"]["component_count"]==2
    for profile in row["original_source_threshold_diagnostics"]:
        assert profile["original_V18a_candidates"]["component_count"]==3
        assert profile["original_candidate_count_above_topology_floor"]==1
        assert profile["shared_source_edges_cut_by_response"]==1
        assert profile["source_edges_only_one_axis_exceeds_threshold"]==1
        assert profile["source_edges_all_axes_exceed_threshold"]==0
        assert profile["source_individual_axis_partition_counts"][0]["component_count"]==3
        assert profile["source_individual_axis_partition_counts"][1]["component_count"]==2


def test_response_observations_cannot_assert_artist_layer_identity():
    faces,f,e,labels=source(False)
    result=source_audit(faces,f,e,2,labels)
    assert result["original_source_topology_island_mean_response_proximity_is_not_artist_same_drawable"]
    assert result["original_source_artist_same_drawable_pairs_minted"]==0
    assert result["original_source_artist_different_drawable_pairs_minted"]==0
    assert result["original_source_teacher_axis_and_face_ids_never_model_inputs"]


def test_unrelated_source_face_edge_is_forged_graph_and_must_fail():
    faces,f,e,labels=source(False)
    fake=np.asarray([[0,1],[1,2]],dtype=np.int64)
    with pytest.raises(ValueError,match="V18P_SOURCE_ORIGINAL_TOPOLOGY"):
        source_audit(faces,f,fake,2,labels)


def test_replay_court_rejects_forged_response_partition():
    faces,f,e,labels=source(False)
    labels[.1]=np.arange(4,dtype=np.int32)
    with pytest.raises(ValueError,match="V18P_ORIGINAL_SOURCE_V18A_FACE_PARTITION_NOT_REPRODUCED"):
        source_audit(faces,f,e,2,labels)


def test_channel_count_or_nan_fail_closed():
    faces,f,e,labels=source(False)
    with pytest.raises(ValueError,match="V18P_SOURCE_ORIGINAL_TOPOLOGY"):
        source_audit(faces,f[:,:24],e,2,labels)
    f[0,0]=np.nan
    with pytest.raises(ValueError,match="V18P_SOURCE_ORIGINAL_TOPOLOGY"):
        source_audit(faces,f,e,2,labels)


def test_unrelated_subset_axis_can_have_same_response_as_both_islands():
    faces,f,e,labels=source(False)
    a=canonical_labels(len(faces),e)
    assert len(np.unique(a))==2
    assert np.array_equal(f[0],f[3])
    # Unequal artist drawables may share all response channels identically.
    assert STATUS.endswith("DRAWABLE_OWNER_UNKNOWN")
    assert SCHEMA.endswith(".v18p")
