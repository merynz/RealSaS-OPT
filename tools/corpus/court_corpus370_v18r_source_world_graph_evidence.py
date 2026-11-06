#!/usr/bin/env python3
"""Independent original source-world V18r court: full SHA-bound replay.

The court does not equate physical vertex/world normal proximity with artist
drawable identity; never mints source-to-student training target. Full replay
catches a forged fit/heldout result, parent swap or unverified editor claim.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from tools.corpus.compile_corpus370_v18r_source_world_graph_evidence import (
    SCHEMA as IR_SCHEMA,STATUS as IR_STATUS,compile_source
)

SCHEMA="RealSaS.Corpus370IndependentSourceWorldGraphGeometryCourt.v18r"
STATUS="SOURCE_GEOMETRY_SPATIAL_AND_CREASE_COURTED__OWNER_UNKNOWN"
DENY=("graphics_owner_authorized","iris_training_authorized",
      "tessa_training_authorized","materialization_authorized",
      "training_authorized")

def require(ok,code):
    if not ok:raise ValueError(code)

def sha(p):
    with Path(p).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()

def check_payload(measured,recomputed):
    require(measured==recomputed
        and measured.get("schema")==IR_SCHEMA
        and measured.get("status")==IR_STATUS
        and measured.get("source_graphics_compiler_drawable_owner")=="UNKNOWN"
        and measured.get("visual_surface_appearance_and_uv_boundary")==
            "UNOBSERVED_NOT_FABRICATED"
        and measured.get("source_geometry_proximity_hypotheses_are_not_graphics_owner_labels") is True
        and measured.get("source_heldout_pose2_only_court_no_fit") is True
        and measured.get("same_drawable_training_pairs_qualified")==0
        and measured.get("different_drawable_training_pairs_qualified")==0
        and all(measured.get(f) is False for f in DENY),
        "V18R_INDEPENDENT_SOURCE_WORLD_IR_REPLAY_OR_AUTHORITY_FRAUD")
    evidence=measured["original_source_world_evidence_teacher_private"]
    rows=evidence["source_lifted_world_proximity_evidence_private"]
    local=evidence["source_local_geometry_edge_evidence_private"]
    require(evidence.get("source_fit_pose_ordinals")==[0,1]
        and evidence.get("source_original_heldout_pose_ordinal_COURT_ONLY")==2
        and evidence.get("heldout_pose_not_used_to_generate_candidates") is True
        and evidence.get("source_CAA_or_UV_appearance_layer_evidence")==
          "NOT_PRESENT_IN_THIS_SOURCE_GEOMETRY_AUDIT"
        and len(local)==evidence["source_local_crease_summary"][
          "source_adjacent_edge_count"]
        and all(z["source_graphic_owner_separation"]=="UNKNOWN"
                and z["source_uv_material_relation"]=="UNOBSERVED"
                and z["source_geometric_crease_is_not_artist_boundary"]
                for z in local)
        and all(z["original_artist_owner_merge"]=="UNKNOWN"
                and z["source_surface_exact_gap"]==
                   "NOT_MEASURED_VERTEX_PROXY_ONLY"
                and z["source_image_uv_material_appearance"]=="UNOBSERVED"
                and z["source_vertex_gap_does_not_establish_surface_or_uv_continuity"]
                for z in rows)
        and len(evidence["source_world_proximity_sensitivity_summary"])==3
        and all(x["not_drawable_owner_merge_count"]==0 for x
                in evidence["source_world_proximity_sensitivity_summary"]),
        "V18R_INDEPENDENT_GEOMETRY_PROXIMITY_AS_OWNER_LEAK")
    return True


def court(ir_path,v18q_ir,v18q_court,v18a_manifest,v18a_npz,
          v13b_manifest,v13b_array,v13b_court,
          expected_v18q_court_sha,expected_v13b_court_sha):
    supplied=json.loads(Path(ir_path).read_text())
    fresh=compile_source(v18q_ir,v18q_court,v18a_manifest,v18a_npz,
                         v13b_manifest,v13b_array,v13b_court,
                         expected_v18q_court_sha,expected_v13b_court_sha)
    check_payload(supplied,fresh)
    x=fresh["original_source_world_evidence_teacher_private"]
    return {
      "schema":SCHEMA,"status":STATUS,
      "subject_id_private":supplied["subject_id_private"],
      "original_source_sha256":supplied["original_source_sha256"],
      "source_world_graph_ir_sha256":sha(ir_path),
      "original_parent_source_sha256":supplied["parents_sha256"],
      "source_lifted_candidates_with_real_fit_pose3d_proximity":
          len(x["source_lifted_world_proximity_evidence_private"]),
      "source_original_topological_face_neighbors_with_normal_evidence":
          len(x["source_local_geometry_edge_evidence_private"]),
      "source_original_lifted_proximity_threshold_census":
          x["source_world_proximity_sensitivity_summary"],
      "source_original_local_normal_crease_census":
          x["source_local_crease_summary"],
      "original_artist_drawable_owner":"UNKNOWN",
      "original_source_material_and_uv_boundary_not_measured":True,
      "original_source_heldout_pose2_not_used_for_fit":True,
      "no_actual_artist_layer_edit_court_claim":True,
      "graphics_owner_authorized":False,
      "iris_training_authorized":False,
      "tessa_training_authorized":False,
      "materialization_authorized":False,
      "training_authorized":False,
    }

def main():
    p=argparse.ArgumentParser()
    for f in ("ir_path","v18q_ir","v18q_court","v18a_manifest","v18a_npz",
              "v13b_manifest","v13b_array","v13b_court","output"):
        p.add_argument("--"+f.replace("_","-"),type=Path,required=True)
    p.add_argument("--expected-v18q-court-sha",required=True)
    p.add_argument("--expected-v13b-court-sha",required=True)
    a=p.parse_args()
    x=court(a.ir_path,a.v18q_ir,a.v18q_court,a.v18a_manifest,
            a.v18a_npz,a.v13b_manifest,a.v13b_array,a.v13b_court,
            a.expected_v18q_court_sha,a.expected_v13b_court_sha)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({
       "status":x["status"],"source":x["subject_id_private"],
       "lifted":x["source_original_lifted_proximity_threshold_census"],
       "local":x["source_original_local_normal_crease_census"],
       "training_authorized":False},sort_keys=True),flush=True)

if __name__=="__main__":main()
