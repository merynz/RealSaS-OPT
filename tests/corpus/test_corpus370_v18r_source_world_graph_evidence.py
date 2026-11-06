"""V18r source-world evidence: pose split, geometry is not ownership."""
import copy
import numpy as np
import pytest

from tools.corpus.compile_corpus370_v18r_source_world_graph_evidence import (
    SCHEMA,STATUS,GAPS,original_geometry_metrics
)

def evidence_example():
    # Each island has two adjacent faces; islands are separately addressed
    # even when their source world vertices are nearly coincident.
    faces=np.asarray([[0,1,2],[2,1,3],[4,5,6],[6,5,7]],np.int32)
    vertices=np.asarray([
        [0,0,0],[1,0,0],[0,1,0],[1,1,0],
        [1.005,0,0],[2.005,0,0],[1.005,1,0],[2.005,1,0]
    ],np.float64)
    pose0=vertices
    pose1=vertices.copy()
    pose1[4:,0]+=.002
    pose2=vertices.copy()
    pose2[4:,0]+=.50
    base=np.stack([pose0,pose1,pose2])
    local=[{
      "face_a_source_teacher_only":0,"face_b_source_teacher_only":1,
      "source_any_axis_above_original_threshold":False,
      "source_pooled_above_original_threshold":False,
    },{
      "face_a_source_teacher_only":2,"face_b_source_teacher_only":3,
      "source_any_axis_above_original_threshold":True,
      "source_pooled_above_original_threshold":False,
    }]
    lifted=[{
      "topology_island_a_source_teacher_only":0,
      "topology_island_b_source_teacher_only":1
    }]
    return faces,base,[0,0,1,1],local,lifted


def test_measured_source_world_vertex_gap_rejects_unrelated_far_heldout():
    q=original_geometry_metrics(*evidence_example())
    assert q["source_total_faces"]==4
    assert q["source_total_vertices"]==8
    assert q["source_topology_islands"]==2
    assert q["source_lifted_world_proximity_evidence_private"][0][
         "source_fit_both_pose_min_vertex_gap_le_0p02"] is True
    l=q["source_lifted_world_proximity_evidence_private"][0]
    assert l["source_heldout_min_vertex_gap_le_0p02_COURT_ONLY"] is False
    assert q["source_world_proximity_sensitivity_summary"][1][
        "heldout_pose_separates_count_COURT_ONLY"]==1
    assert q["source_world_proximity_sensitivity_summary"][1][
        "not_drawable_owner_merge_count"]==0


def test_heldout_pose_changes_court_not_fit_hypothesis():
    faces,base,topo,local,lifted=evidence_example()
    a=original_geometry_metrics(faces,base,topo,local,lifted)
    other=base.copy()
    other[2,4:,0]=100.
    b=original_geometry_metrics(faces,other,topo,local,lifted)
    assert a["source_lifted_world_proximity_evidence_private"][0][
      "fit_pose_gap_max_relative_extent"]==b[
      "source_lifted_world_proximity_evidence_private"][0][
      "fit_pose_gap_max_relative_extent"]
    assert a["source_lifted_world_proximity_evidence_private"][0][
      "heldout_pose_gap_relative_extent_COURT_ONLY"]!=b[
      "source_lifted_world_proximity_evidence_private"][0][
      "heldout_pose_gap_relative_extent_COURT_ONLY"]
    assert a["heldout_pose_not_used_to_generate_candidates"]


def test_geometry_does_not_mint_artist_labels():
    r=original_geometry_metrics(*evidence_example())
    assert r["source_geometric_proximity_only_is_not_drawable_proof"] is True
    assert r["original_source_artist_graphic_ownership"]=="UNKNOWN"
    assert r["no_hard_merge_or_cannot_link_minted"] is True
    assert r["source_CAA_or_UV_appearance_layer_evidence"]==(
        "NOT_PRESENT_IN_THIS_SOURCE_GEOMETRY_AUDIT")
    assert r["source_teacher_world_vertices_not_student_model_input"] is True


def test_local_normal_faces_are_measured_but_not_graphic_boundary():
    r=original_geometry_metrics(*evidence_example())
    assert r["source_local_crease_summary"]["source_adjacent_edge_count"]==2
    assert r["source_local_crease_summary"][
        "fit_source_normal_crease_30deg_edge_count"]==0
    assert all(z["source_geometric_crease_is_not_artist_boundary"] for z
               in r["source_local_geometry_edge_evidence_private"])


def test_face_normal_crease_changes_source_diagnostic_not_owner():
    faces,base,t,local,lifted=evidence_example()
    # Raise one triangle's final vertex to create a crease.
    base[:,3,2]=2.
    result=original_geometry_metrics(faces,base,t,local,lifted)
    e=result["source_local_geometry_edge_evidence_private"][0]
    assert e["fit_consistent_geometric_normal_crease_30deg"] is True
    assert e["source_graphic_owner_separation"]=="UNKNOWN"
    assert e["source_uv_material_relation"]=="UNOBSERVED"


@pytest.mark.parametrize("problem",[
    "bad_faces","bad_vertex_count","nan","bad_source_index","zero_extent",
    "lifted_missing_island","reversed_lifted","bad_local",
])
def test_source_geometry_invalid_arrays_or_unauthorized_addresses_fail(problem):
    faces,base,t,local,lifted=evidence_example()
    if problem=="bad_faces":faces=faces[:2]
    elif problem=="bad_vertex_count":base=base[:2]
    elif problem=="nan":base[0,0,0]=np.nan
    elif problem=="bad_source_index":faces[0,0]=500
    elif problem=="zero_extent":base[:]=0.
    elif problem=="lifted_missing_island":
        lifted[0]["topology_island_b_source_teacher_only"]=7
    elif problem=="reversed_lifted":
        lifted[0]["topology_island_a_source_teacher_only"]=1
        lifted[0]["topology_island_b_source_teacher_only"]=0
    else:local[0]["face_b_source_teacher_only"]=0
    if problem=="bad_faces":
        t=t[:2] # face census different, test still invalid local references
    with pytest.raises(ValueError,match="V18R_"):
        original_geometry_metrics(faces,base,t,local,lifted)


def test_status_and_thresholds_not_artist_gold():
    assert STATUS.endswith("NO_GRAPHIC_OWNER_PROOF")
    assert SCHEMA.endswith(".v18r")
    assert GAPS==(.005,.02,.05)

from tools.corpus.court_corpus370_v18r_source_world_graph_evidence import (
    check_payload,
)


def court_sample():
    geom=original_geometry_metrics(*evidence_example())
    return {
      "schema":SCHEMA,"status":STATUS,
      "source_graphics_compiler_drawable_owner":"UNKNOWN",
      "visual_surface_appearance_and_uv_boundary":"UNOBSERVED_NOT_FABRICATED",
      "source_geometry_proximity_hypotheses_are_not_graphics_owner_labels":True,
      "source_heldout_pose2_only_court_no_fit":True,
      "same_drawable_training_pairs_qualified":0,
      "different_drawable_training_pairs_qualified":0,
      "graphics_owner_authorized":False,"iris_training_authorized":False,
      "tessa_training_authorized":False,"materialization_authorized":False,
      "training_authorized":False,
      "original_source_world_evidence_teacher_private":geom,
    }


def test_independent_court_accepts_full_source_replay_only():
    original=court_sample()
    assert check_payload(original,original)


@pytest.mark.parametrize("attack",(
   "fake_uv","make_merge","forge_pose2","artist_owner",
   "training","forge_crease","fake_surface_distance",
))
def test_source_world_independent_court_catches_forged_authority(attack):
    original=court_sample()
    fake=copy.deepcopy(original)
    if attack=="fake_uv":
        fake["visual_surface_appearance_and_uv_boundary"]="VERIFIED"
    elif attack=="make_merge":
        fake["original_source_world_evidence_teacher_private"][
          "source_lifted_world_proximity_evidence_private"][0][
          "original_artist_owner_merge"]="TRUE"
    elif attack=="forge_pose2":
        fake["original_source_world_evidence_teacher_private"][
          "heldout_pose_not_used_to_generate_candidates"]=False
    elif attack=="artist_owner":
        fake["source_graphics_compiler_drawable_owner"]="VERIFIED"
    elif attack=="training":
        fake["iris_training_authorized"]=True
    elif attack=="forge_crease":
        fake["original_source_world_evidence_teacher_private"][
          "source_local_geometry_edge_evidence_private"][0][
          "source_graphic_owner_separation"]="TRUE"
    else:
        fake["original_source_world_evidence_teacher_private"][
          "source_lifted_world_proximity_evidence_private"][0][
          "source_surface_exact_gap"]="VERIFIED_FULL_TRIANGLE_GAP"
    with pytest.raises(ValueError,match="V18R_INDEPENDENT_"):
        check_payload(fake,original)
