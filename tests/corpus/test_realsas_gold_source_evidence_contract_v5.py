"""Synthetic-only fail-closed Gold228 V5 research contract checks.

The private per-source receipts are deliberately not committed to GitHub.
"""
import json
from copy import deepcopy
from pathlib import Path

import pytest

CONTRACT = (
    Path(__file__).resolve().parents[2]
    / "canonical/REALSAS_GOLD_SOURCE_EVIDENCE_CONTRACT_V5.json"
)


def audit(r):
    assert r["schema"] == "RealSaS.GoldSourceStaticEvidenceContract.v5"
    u = r["source_universe"]
    e = r["evidence_contract"]
    g = r["risk_census"]
    t = r["target_routing"]

    assert u["gold_228_selected"] == u["quaternius"] + u["kaykit"] == 228
    assert (u["historical_master_sha_receipts"] +
            u["new_full_source_byte_sha_receipts"]
            == u["master_sha_receipts_now"])
    assert (u["master_sha_receipts_now"] +
            u["master_pending_full_sha"] == u["master_total"] == 3993)
    assert sum(e["source_format_headers"].values()) == 228
    assert e["same_file_cross_corpus_pairs"] == 53
    assert e["protected_source_split_links"] == 16
    assert e["new_exact_core370_duplicates"] == 0
    assert e["protected_source_split_links"] <= e["same_file_cross_corpus_pairs"]
    assert g["obj_original_face_count"] == (
        g["obj_original_triangle_count"] +
        g["obj_original_quad_count"] +
        g["obj_original_ngon_gt4_count"])
    assert g["obj_original_face_count"] - g["obj_original_triangle_count"] == 156021
    assert g["obj_source_with_ngon_gt4_count"] <= g["obj_source_with_nontriangle_count"]
    assert g["unselected_gltf_are_selected_master_subjects"] is False

    assert t["head_count"] == 6
    assert t["targets_per_source"] == 37
    assert t["candidate_routes"] == u["gold_228_selected"] * t["targets_per_source"]
    assert t["loss_mask_valid_count"] == t["training_admitted_count"] == 0

    assert e["face_triangulation_equivalence_proven"] is False
    assert e["source_format_is_not_rig_skin_truth"] is True
    assert e["source_original_polygon_address_required"] is True
    for key in (
        "source_legal_ml_rights_courted",
        "source_native_pose_mesh_bind_deformation_parity_courted",
        "source_semantic_drawable_edit_consequence_courted",
        "source_bound_stylized_2d_graphics_courted",
        "model_training_authorized",
    ):
        assert e[key] is False
    assert r["no_original_bytes_or_private_hashes_committed_to_repo"] is True


def test_source_scope_and_truth_firewall():
    audit(json.loads(CONTRACT.read_text()))


@pytest.mark.parametrize("mutation", [
    lambda r: r["source_universe"].update(master_source_sha_verified_after_overlay=3993)
        or r["source_universe"].update(master_pending_full_sha=0),
    lambda r: r["evidence_contract"].update(model_training_authorized=True),
    lambda r: r["target_routing"].update(loss_mask_valid_count=1),
    lambda r: r["evidence_contract"].update(face_triangulation_equivalence_proven=True),
    lambda r: r["risk_census"].update(obj_original_quad_count=1),
    lambda r: r["evidence_contract"].update(protected_source_split_links=0),
    lambda r: r["risk_census"].update(unselected_gltf_are_selected_master_subjects=True),
    lambda r: r["evidence_contract"].update(source_legal_ml_rights_courted=True),
    lambda r: r["evidence_contract"].update(source_native_pose_mesh_bind_deformation_parity_courted=True),
])
def test_minted_or_inconsistent_evidence_fails(mutation):
    original = json.loads(CONTRACT.read_text())
    altered = deepcopy(original)
    mutation(altered)
    with pytest.raises(AssertionError):
        audit(altered)
