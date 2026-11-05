#!/usr/bin/env python3
"""Independent read-only recompile and full-source typed Evidence IR court.

Every source-private subpixel ray still has its original independent V18j
court SHA. Recompiles the whole output and compares all typed source-locus,
local visibility, unknown and no-owner fields. Does not promote IR to model
truth, drawable owner, or generic unseen success.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from tools.corpus.compile_corpus370_graphics_evidence_ir_v1 import (
    compile_ir, SCHEMA as IR_SCHEMA, STATUS as IR_STATUS, KNOWN_KINDS
)

SCHEMA="RealSaS.Corpus370IndependentGenericGraphicsEvidenceIRCourt.v1"
STATUS="SOURCE_TYPED_GRAPHICS_EVIDENCE_RECOMPILED__UNKNOWN_OWNER_AND_NO_TRAINING"


def require(ok,reason):
    if not ok:raise ValueError(reason)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def audit(ir_path,source_report_path,v18j_court_path,expected_parent_sha):
    ir=load(ir_path)
    require(ir.get("schema")==IR_SCHEMA
        and ir.get("status")==IR_STATUS
        and ir.get("evidence_atom_types")==list(KNOWN_KINDS)
        and ir.get("parent_v18j_proposal_sha256")==sha(source_report_path)
        and ir.get("parent_v18j_independent_source_court_sha256")==sha(v18j_court_path)
        and ir.get("evidence_granularity")=="SOURCE_POSE_VIEW_PIXEL_SUBSAMPLE"
        and ir.get("evidence_is_retrospective_not_blind_heldout") is True
        and ir.get("no_graph_signed_edges_minted") is True
        and ir.get("source_ray_order_is_not_global_drawable_order") is True
        and ir.get("no_anatomical_or_semantic_categories") is True
        and ir.get("original_artist_layer_truth")=="UNKNOWN"
        and ir.get("consumer_editable_component_truth")=="UNKNOWN"
        and all(ir.get(k) is False for k in (
            "graphics_owner_authorized","iris_training_authorized",
            "tessa_training_authorized","materialization_authorized",
            "training_authorized")),
        "GRAPHICS_IR_INDEPENDENT_FORGED_CLAIM_OR_SOURCE")
    actual=compile_ir(source_report_path,v18j_court_path,expected_parent_sha)
    require(ir==actual,"GRAPHICS_IR_INDEPENDENT_SOURCE_OBSERVATIONS_CHANGED")
    rows=actual["observations_teacher_private"]
    for obs in rows:
        x=obs["observational_atoms"]
        require(set(x)==set(KNOWN_KINDS)
            and obs["operational_part_affinity"]=="UNKNOWN"
            and obs["operational_part_repulsion"]=="UNKNOWN"
            and obs["source_artist_part_identity"]=="UNKNOWN"
            and obs["graphics_compiler_draw_order"]=="UNKNOWN"
            and obs["cannot_link_minted"] is False
            and obs["must_link_minted"] is False
            and obs["student_model_target_admitted"] is False
            and x["CONTROLLED_RGBA_COMPOSITING_RESIDUAL_CLASS"]
                ["alpha_or_rgb_class_does_not_imply_layer_boundary"] is True
            and x["SOURCE_RAY_GEOMETRIC_ORDER"]
                ["visibility_relation_is_local_to_original_pose_camera_sample"] is True,
            "GRAPHICS_IR_INDEPENDENT_RAY_ORDER_TO_ARTIST_LABEL_LEAKAGE")
    total=sum(v["original_source_pixels"] for v in actual[
        "per_original_pixel_class_observation_counts"].values())
    require(len(rows)==total,
            "GRAPHICS_IR_INDEPENDENT_SOURCE_PIXEL_CENSUS")
    return {
        "schema":SCHEMA,"status":STATUS,
        "original_source_subject_id_private":actual["source_subject_id_private"],
        "original_source_sha256":actual["original_source_sha256"],
        "original_source_typed_evidence_ir_sha256":sha(ir_path),
        "original_source_parent_v18j_proposal_sha256":sha(source_report_path),
        "original_source_parent_v18j_court_sha256":sha(v18j_court_path),
        "source_pixel_observations_recompiled":len(rows),
        "evidence_types_independently_courted":list(KNOWN_KINDS),
        "all_source_rays_rebound_to_exact_parent":True,
        "source_3d_geometry_order_is_not_author_layer_relation":True,
        "same_drawable_training_pairs_qualified":0,
        "different_drawable_training_pairs_qualified":0,
        "no_fixed_K_or_semantic_ontology_admitted":True,
        "model_training_target_count":0,
        "source_artist_drawable_identity":"UNKNOWN",
        "operational_editable_layer_identity":"UNKNOWN",
        "graphics_owner_authorized":False,
        "iris_training_authorized":False,
        "tessa_training_authorized":False,
        "materialization_authorized":False,
        "training_authorized":False,
    }


def main():
    p=argparse.ArgumentParser()
    for name in ("ir","v18j_proposal","v18j_court","output"):
        p.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    p.add_argument("--expected-v18j-court-sha",required=True)
    a=p.parse_args()
    out=audit(a.ir,a.v18j_proposal,a.v18j_court,a.expected_v18j_court_sha)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({
        "status":out["status"],
        "source_pixel_observations_recompiled":out["source_pixel_observations_recompiled"],
        "same_drawable_training_pairs_qualified":0,
        "training_authorized":False,
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
