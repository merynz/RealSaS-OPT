#!/usr/bin/env python3
"""Independent full-source V18q graph IR court; original NPZ replay is required.

The producer's typed mechanical hypotheses are NOT artist part ground truth.
Independent source V18p/V18o are checked by exact SHA; bytes are recompiled
with the same frozen original V18a source face/axis arrays and compared fully.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from tools.corpus.compile_corpus370_v18q_graph_affinity_hypotheses import (
    compile_original,SCHEMA as GRAPH_SCHEMA,STATUS as GRAPH_STATUS
)

SCHEMA="RealSaS.Corpus370IndependentTypedGraphAffinityHypothesisCourt.v18q"
STATUS="ORIGINAL_SOURCE_GRAPH_HYPOTHESES_REPLAYED__NO_OWNER_OR_STUDENT_GOLD"
FALSE_FIELDS=("graphics_owner_authorized","iris_training_authorized",
              "tessa_training_authorized","materialization_authorized",
              "training_authorized")

def require(ok,why):
    if not ok:raise ValueError(why)

def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()

def verify_ir(payload,recomputed):
    require(payload==recomputed
        and payload.get("schema")==GRAPH_SCHEMA
        and payload.get("status")==GRAPH_STATUS
        and payload.get("artist_graphics_ownership")=="UNKNOWN"
        and payload.get("compiler_graphics_proof")=="NOT_QUALIFIED"
        and payload.get("source_original_face_uv_material_relation")==
            "NOT_PRESENT_IN_SHA_BOUND_V18A_ARRAYS"
        and payload.get("source_original_multiview_visibility_relation")==
            "NOT_PRESENT_IN_SHA_BOUND_V18A_ARRAYS"
        and payload.get("same_drawable_training_pairs_qualified")==0
        and payload.get("different_drawable_training_pairs_qualified")==0
        and all(payload.get(k) is False for k in FALSE_FIELDS),
        "V18Q_INDEPENDENT_GRAPH_REPLAY_OR_AUTHORITY_FRAUD")
    g=payload["source_private_graph"]
    local=g["source_local_graph_affinity_evidence_ir_teacher_private"]
    lifted=g["source_nonlocal_graph_affinity_hypotheses_teacher_private"]
    variants=g["source_candidate_drawable_hypotheses_teacher_private"]
    require(len(local)==g["source_original_local_edge_count"]
        and len(lifted)==g["source_original_lifted_candidate_count"]
        and len(g["source_face_observation_ir_teacher_private"][
           "source_topology_island_per_face_teacher_private"])==
              g["source_total_faces"]
        and len(variants)==3
        and {x["candidate_key"] for x in variants}=={
           "V18A_POOLED_LOCAL_BASELINE",
           "SOURCE_TOPOLOGY_ONLY_CONTROL",
           "MULTI_AXIS_OR_CUT_STRESS_ONLY"
        }
        and g["source_nonlocal_merge_edges_admitted"]==0
        and g["source_local_split_edges_admitted"]==0
        and g["source_artist_same_drawable_pairs_minted"]==0
        and g["source_artist_different_drawable_pairs_minted"]==0
        and g["student_target_admitted"] is False
        and g["source_lifted_response_proximity_not_a_calibrated_must_link"] is True
        and g["source_axis_cut_stress_not_a_calibrated_cannot_link"] is True
        and all(e["can_merge_artist_drawable"]=="UNKNOWN" and
                e["must_separate_artist_drawable"]=="UNKNOWN" and
                len(e["source_mechanical_per_axis_distance"])==
                     g["source_selected_axes_count"] for e in local)
        and all(e["can_merge_artist_drawable"]=="UNKNOWN" and
                e["must_separate_artist_drawable"]=="UNKNOWN" and
                e["original_source_uv_or_material_continuity"]=="UNOBSERVED" and
                e["original_source_occlusion_and_visibility_continuity"]=="UNOBSERVED"
                for e in lifted),
        "V18Q_INDEPENDENT_GRAPH_TYPED_RELATION_OR_LABEL_LEAK")
    edit=payload["independent_edit_court_observations_not_for_fitting"]
    require(edit["source_v18o_edit_outcome_used_to_rank_hypotheses"] is False
        and edit["source_edit_consequence_for_new_graph_hypotheses"]==
            "NOT_EVALUATED"
        and edit["original_source_cross_pose_graphics_owner_stability"]=="UNKNOWN",
        "V18Q_INDEPENDENT_SOURCE_HELDOUT_EDIT_LEAKAGE")
    return True


def court(path,stain_manifest,stain_npz,stain_court,v18p_court,v18o_court,
          expected_stain_sha,expected_v18p_sha,expected_v18o_sha):
    d=json.loads(Path(path).read_text())
    raw=compile_original(stain_manifest,stain_npz,stain_court,
           v18p_court,v18o_court,expected_stain_sha,expected_v18p_sha,
           expected_v18o_sha)
    verify_ir(d,raw)
    g=raw["source_private_graph"]
    return {
        "schema":SCHEMA,"status":STATUS,
        "original_subject_id_private":raw["subject_id_private"],
        "original_source_sha256":raw["original_source_sha256"],
        "source_original_graph_hypothesis_ir_sha256":sha(path),
        "original_graph_parents_sha256":raw["parents_sha256"],
        "exact_source_original_face_count":g["source_total_faces"],
        "exact_source_original_local_edge_count":g["source_original_local_edge_count"],
        "source_topology_island_count":g["source_original_topology_island_count"],
        "source_lifted_response_only_unaccepted_hypotheses":
            g["source_original_lifted_candidate_count"],
        "source_original_control_axis_count":g["source_selected_axes_count"],
        "source_candidate_complexity_stress_comparison":
            g["source_candidate_drawable_hypotheses_teacher_private"],
        "candidate_graph_cut_outcome_heldout_editor_test":"NOT_EVALUATED",
        "source_original_appearance_boundary":"NOT_YET_BOUND",
        "source_original_occ_order_for_graph":"NOT_YET_BOUND",
        "original_artist_graphic_owner":"UNKNOWN",
        "no_semantic_part_names_or_fixed_K":True,
        "no_unverified_training_ground_truth":True,
        "graphics_owner_authorized":False,
        "iris_training_authorized":False,
        "tessa_training_authorized":False,
        "materialization_authorized":False,
        "training_authorized":False,
    }

def main():
    a=argparse.ArgumentParser()
    for k in ("graph_ir","stain_manifest","stain_npz","stain_court",
              "v18p_court","v18o_court","output"):
        a.add_argument("--"+k.replace("_","-"),required=True,type=Path)
    for k in ("expected_stain_sha","expected_v18p_sha","expected_v18o_sha"):
        a.add_argument("--"+k.replace("_","-"),required=True)
    v=a.parse_args()
    x=court(v.graph_ir,v.stain_manifest,v.stain_npz,v.stain_court,
            v.v18p_court,v.v18o_court,v.expected_stain_sha,
            v.expected_v18p_sha,v.expected_v18o_sha)
    v.output.parent.mkdir(parents=True,exist_ok=True)
    v.output.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({"status":x["status"],
         "source":x["original_subject_id_private"],
         "islands":x["source_topology_island_count"],
         "lifted_ambiguous":x["source_lifted_response_only_unaccepted_hypotheses"],
         "candidate_counts":x["source_candidate_complexity_stress_comparison"],
         "training_authorized":False},sort_keys=True),flush=True)


if __name__=="__main__":main()
