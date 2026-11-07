"""Synthetic tests: original Master4008/4036 reconciliation and strict authority."""
from __future__ import annotations

import pytest

from tools.corpus.reconcile_realsas_all_cohorts_v2 import (
    validate_origins, run,
)


def make_example():
    selected = {
        "canonical_asset_id": "asset_selected",
        "candidate_id": "provider:asset-selected",
        "split": "EXTERNAL_HOLDOUT",
        "source_registry_id": "provider",
        "decision": "ADMIT",
    }
    qa = {
        "canonical_asset_id": "asset_qa",
        "candidate_id": "khronos_gltf_qa:asset-qa",
        "split": "QA_ONLY",
        "source_registry_id": "khronos_gltf_qa",
        "decision": "ADMIT_RIG_NO_SKIN",
    }
    rejected = {
        "candidate_id": "provider:rejected-candidate",
        "decision": "REJECT_TECHNICAL",
        "audit": {
            "geometry_reasons": ["too_many_degenerate_faces"],
            "skin_reasons": [], "rig_reasons": []
        },
    }
    inventory = [{
        "canonical_asset_id": selected["canonical_asset_id"],
        "candidate_id": selected["candidate_id"],
        "historical_split": selected["split"],
        "provider": selected["source_registry_id"],
    }]
    return [selected, qa], [selected.copy(), qa.copy(), rejected], inventory


def checked(original, processed, inventory):
    return validate_origins(
        original, processed, inventory,
        expected_master=1, expected_qa=1, expected_reject=1
    )


def test_historical_holdout_and_qa_identity_survive():
    original, processed, inventory = make_example()
    selected, qa, rejects = checked(original, processed, inventory)
    assert selected["asset_selected"]["split"] == "EXTERNAL_HOLDOUT"
    assert qa[0]["split"] == "QA_ONLY"
    assert rejects[0]["candidate_id"] == "provider:rejected-candidate"
    assert "canonical_asset_id" not in rejects[0]


def test_candidate_substitution_rejected():
    original, processed, inventory = make_example()
    processed[0] = dict(processed[0], candidate_id="provider:replaced")
    with pytest.raises(ValueError, match="PROCESSED_MUTATED_ORIGINAL"):
        checked(original, processed, inventory)


def test_qa_illicit_fit_promotion_rejected():
    original, processed, inventory = make_example()
    original[1] = dict(original[1], split="FIT")
    with pytest.raises(ValueError, match="QA_SPLIT_CENSUS"):
        checked(original, processed, inventory)


def test_reject_candidate_overlap_rejected():
    original, processed, inventory = make_example()
    processed[2] = dict(processed[2], candidate_id=original[0]["candidate_id"])
    with pytest.raises(ValueError, match="REJECT_DUPLICATE_CANDIDATE"):
        checked(original, processed, inventory)


def test_rejected_requires_actual_historical_reason():
    original, processed, inventory = make_example()
    processed[2] = dict(processed[2], audit=dict(processed[2]["audit"],
                                                geometry_reasons=[]))
    with pytest.raises(ValueError, match="REJECT_PROVENANCE"):
        checked(original, processed, inventory)


def test_original_master_inventory_split_must_match():
    original, processed, inventory = make_example()
    inventory[0]["historical_split"] = "FIT"
    with pytest.raises(ValueError, match="MASTER_PROVENANCE_OR_SPLIT"):
        checked(original, processed, inventory)


def test_fake_master_ledger_is_rejected_before_any_source_read(tmp_path):
    fake = tmp_path / "fake.jsonl"
    fake.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="MASTER_LEDGER_HASH"):
        run(fake, fake, fake, tmp_path, fake, tmp_path / "out")
    assert not (tmp_path / "out").exists()
