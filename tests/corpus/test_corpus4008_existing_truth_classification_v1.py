"""Existing source evidence classification rejects false truth and split leaks."""
import csv
import hashlib
import json

import pytest

from tools.corpus.classify_corpus4008_existing_truth_v1 import classify


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_csv(p,rows):
    with p.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def data(tmp):
    s=[]
    q=[]
    for i in range(3):
        h=f"{i+1:064x}"
        aid=f"asset_{i+1:024x}"
        src={
            "canonical_asset_id":aid,
            "candidate_id":"objaverse_animated_originals:"+h,
            "source_sha256":h,
            "source_format":".glb",
            "source_shard":"0",
            "split":"FIT",
            "source_import_observation_status":
                "IMPORTED_SOURCE_OBSERVATIONS_ONLY",
            "source_byte_verified_at_import":"True",
            "head_admission_count":"0",
            "training_authorized":"False",
            "mesh_object_count":"2",
            "vertex_count":"1500",
            "armature_count":"1",
            "observed_bone_count":"40",
            "skin_rows_sampled":"1500",
            "skin_rows_zero_weights":"12",
            "skin_rows_bad_sum":"3",
            "observed_action_curve_count":"200",
            "observed_action_count":"1",
            "observed_ik_constraint_count":"0",
            "material_count":"2",
            "image_count":"1",
            "image_dependency_missing_count":"0",
            "mesh_boundary_edges":"10",
            "mesh_nonmanifold_edges":"0",
            "source_character_vs_prop":"UNKNOWN",
            "morphology":"UNKNOWN",
            "original_license_commercial_ml_rights":"UNKNOWN",
            "functional_native_ik_fk":"UNKNOWN",
            "source_skin_bind_deformation":"UNKNOWN",
            "source_semantic_regions":"UNKNOWN",
            "graphics_source_bound_2d_art_qualified":"UNKNOWN",
            "tessa_ready_topology_truth":"UNKNOWN",
            "axis_typed_controllable_rig_program_truth":"UNKNOWN",
            "mira_weights_truth":"UNKNOWN",
        }
        qa={
            "canonical_asset_id":aid,
            "original_source_sha256":h,
            "source_import_status":src["source_import_observation_status"],
            "source_review_queue":"RIG_AND_SKIN_DAMAGE_RISK",
            "training_authorized":"False",
            "materialization_authorized":"False",
        }
        for a,b in (
            ("mesh_object_count","source_observed_mesh_objects"),
            ("observed_bone_count","source_observed_bones"),
            ("observed_action_count","source_observed_actions"),
            ("observed_ik_constraint_count","source_observed_IK_constraints"),
            ("skin_rows_sampled","source_observed_skin_rows"),
            ("skin_rows_bad_sum","source_observed_skin_bad_sum"),
            ("skin_rows_zero_weights","source_observed_skin_no_weights"),
            ("mesh_boundary_edges","source_observed_boundary_edges"),
            ("mesh_nonmanifold_edges","source_observed_nonmanifold_edges"),
            ("image_dependency_missing_count",
             "source_observed_unresolved_external_images")):
            qa[b]=src[a]
        s.append(src);q.append(qa)
    sf=tmp/"SOURCE.csv"
    qf=tmp/"QA.csv"
    write_csv(sf,s)
    write_csv(qf,q)
    ss=tmp/"SOURCE_SUMMARY.json"
    ts=tmp/"QA_SUMMARY.json"
    ss.write_text(json.dumps({
        "schema":"RealSaS.Corpus4008Parallel1024ImportedSceneMasterLedger.v1",
        "unique_source_candidates_measured":3,
        "typed_source_matrix_sha256":digest(sf),
        "training_authorized":False,
    }))
    ts.write_text(json.dumps({
        "schema":"RealSaS.Corpus4008SourceCourtReviewTriage.v1",
        "source_count":3,
        "source_readback_summary_sha256":digest(ss),
        "source_readback_matrix_sha256":digest(sf),
        "triage_matrix_sha256":digest(qf),
        "training_authorized":False,
    }))
    return ss,sf,ts,qf,s,q


def run(paths,tmp):
    return classify(*paths,tmp/"out",expected_count=3)


def test_all_source_and_six_head_routes_never_mint_truth(tmp_path):
    *paths,s,q=data(tmp_path)
    r=run(paths,tmp_path)
    assert r["source_count"]==3
    assert r["head_route_count"]==18
    assert all(v==0 for v in r["training_admitted_by_head"].values())
    assert not r["source_truth_synthesized"]
    rows=list(csv.DictReader((tmp_path/"out/SOURCE_EXISTING_EVIDENCE_CLASSIFICATION_V1.csv").open()))
    assert all(x["evidence__native_IK"]=="NOT_OBSERVED_IN_IMPORT_SOURCE_UNKNOWN" for x in rows)
    assert all(x["evidence__graphics_2d_game_art"]=="UNREVIEWED_2D_STYLIZED_ART_REQUIRED" for x in rows)
    assert all(x["training_authorized"]=="False" for x in rows)
    heads=list(csv.DictReader((tmp_path/"out/SIX_HEAD_EXISTING_EVIDENCE_ROUTES_V1.csv").open()))
    assert len(heads)==18
    assert all(x["source_truth"]=="NOT_ADMITTED" for x in heads)


def test_tampered_qa_data_fails_sha(tmp_path):
    *paths,s,q=data(tmp_path)
    paths[3].write_text(paths[3].read_text().replace("1500","1501",1))
    with pytest.raises(ValueError,match="QA_LINEAGE_SHA"):
        run(paths,tmp_path)


def test_qa_counter_drift_fails_even_if_sha_resigned(tmp_path):
    *paths,s,q=data(tmp_path)
    q[0]["source_observed_skin_no_weights"]="13"
    write_csv(paths[3],q)
    d=json.loads(paths[2].read_text())
    d["triage_matrix_sha256"]=digest(paths[3])
    paths[2].write_text(json.dumps(d))
    with pytest.raises(ValueError,match="SOURCE_QA_DRIFT_skin_rows_zero_weights"):
        run(paths,tmp_path)


def test_holdout_leak_fails_even_after_hash_resigned(tmp_path):
    *paths,s,q=data(tmp_path)
    s[0]["split"]="EXTERNAL_HOLDOUT"
    write_csv(paths[1],s)
    d=json.loads(paths[0].read_text())
    d["typed_source_matrix_sha256"]=digest(paths[1])
    paths[0].write_text(json.dumps(d))
    v=json.loads(paths[2].read_text())
    v["source_readback_summary_sha256"]=digest(paths[0])
    v["source_readback_matrix_sha256"]=digest(paths[1])
    paths[2].write_text(json.dumps(v))
    with pytest.raises(ValueError,match="SOURCE_SPLIT_OR_AUTHORITY"):
        run(paths,tmp_path)


def test_forged_truth_promotion_fails_before_output(tmp_path):
    *paths,s,q=data(tmp_path)
    s[0]["functional_native_ik_fk"]="PASS"
    write_csv(paths[1],s)
    d=json.loads(paths[0].read_text())
    d["typed_source_matrix_sha256"]=digest(paths[1])
    paths[0].write_text(json.dumps(d))
    t=json.loads(paths[2].read_text())
    t["source_readback_summary_sha256"]=digest(paths[0])
    t["source_readback_matrix_sha256"]=digest(paths[1])
    paths[2].write_text(json.dumps(t))
    with pytest.raises(ValueError,match="SOURCE_UNKNOWN_PROMOTED_functional_native_ik_fk"):
        run(paths,tmp_path)
