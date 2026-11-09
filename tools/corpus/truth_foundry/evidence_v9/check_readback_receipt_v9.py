#!/usr/bin/env python3
"""Validate an archived local evidence-readback witness; never claim a private replay."""
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

def validate() -> None:
    r = json.loads((HERE / "INDEPENDENT_READBACK_COURT_V9.json").read_text())
    summary = json.loads((HERE / "INVENTORY_MATRIX_V9_SUMMARY.json").read_text())
    gap = json.loads((HERE / "V9_GAP_RECONCILIATION_SUMMARY.json").read_text())
    assert r["schema"] == "RealSaS.TruthFoundryIndependentReadback.v9"
    assert r["integrity"] == "PASS_ALL_INDEPENDENT_RECOUNTS__EVIDENCE_ONLY"
    assert r["source_rows_validated"] == summary["candidate_count"] == 16_397
    assert r["facet_columns"] == summary["facet_count"] == 264
    assert r["original_facet_cells_preserved"] + r["new_observed_cells"] == 16_397 * 264
    assert r["new_observed_cells"] == summary["new_import_receipt_observations"] == 7_398
    assert set(r["new_observations_by_facet"]) == {
        "mesh_armature_bind_association",
        "modular_vs_welded_vs_multimesh",
        "primitive_face_to_scene_node",
    }
    assert all(v == 2_466 for v in r["new_observations_by_facet"].values())
    assert r["master_full_import_receipt_sha_crosswalk"] == 2_466
    assert r["master_mesh_objects_recount"] == 14_351
    assert r["core_partial_receipt_sha_crosswalk"] == 370
    assert r["core_family_rows_recount"] == 5_180
    assert r["master_missing_deep_receipt_queue_rows"] == 1_527
    for field in ("source_original_sha_rehash_pass_minted","new_training_admission",
                  "new_anatomical_or_2d_art_court_pass","new_lineage_split_authority",
                  "global_sdf_pass_minted"):
        assert r[field] == 0, field
    for name, digest in r["artifact_sha256"].items():
        assert not name.startswith("/") and "/" not in name
        assert re.fullmatch(r"[a-f0-9]{64}",digest), name
    print("PASS: V9 archived evidence receipt contract (not private source replay)")
    print(f"candidate={r['source_rows_validated']} facets={r['facet_columns']} new_evidence={r['new_observed_cells']}")
    print("new model/semantic/graphics/global-SDF training admission: 0")

if __name__ == "__main__":
    validate()
