from __future__ import annotations

from tools.corpus.truth_family_inventory_v1 import (
    evidence_descriptor,
    inventory_subject,
    make_family_row,
    validate_inventory,
)


H1 = "1" * 64
H2 = "2" * 64


def descriptor(status="PASS_LEGACY_COURT"):
    return evidence_descriptor(
        "LEGACY_ARTIFACT",
        {"path": "/tmp/example.json", "sha256": H2},
        {"schema": "Legacy.Schema", "status": status, "qualification_minted": True},
    )


def test_legacy_pass_is_evidence_not_new_authority():
    inventory = inventory_subject(
        subject_id="subject",
        source_sha256=H1,
        family_evidence={"SURFACE_GEOMETRY": [descriptor()]},
    )
    validate_inventory(inventory)
    geometry = next(row for row in inventory["families"] if row["family"] == "SURFACE_GEOMETRY")
    assert geometry["evidence_status"] == "PRESENT_UNQUALIFIED"
    assert geometry["legacy_pass_like_claim_count"] == 1
    assert geometry["authority_state"] == "PENDING"
    assert inventory["authority_matrix"]["subject_disposition"] == "PENDING_AUTHORITY_AUDIT"
    assert inventory["training_authorized"] is False


def test_missing_family_stays_none_and_pending():
    inventory = inventory_subject(
        subject_id="subject",
        source_sha256=H1,
        family_evidence={},
    )
    row = next(item for item in inventory["families"] if item["family"] == "SKIN_WEIGHTS")
    assert row["evidence_status"] == "NONE"
    assert row["evidence"] == []
    assert row["authority_state"] == "PENDING"


def test_fail_like_legacy_claim_is_counted_without_becoming_drop():
    inventory = inventory_subject(
        subject_id="subject",
        source_sha256=H1,
        family_evidence={"SIGNED_DISTANCE_OCCUPANCY": [descriptor("GEOMETRY_REPAIR_INCOMPLETE")]},
    )
    row = next(item for item in inventory["families"] if item["family"] == "SIGNED_DISTANCE_OCCUPANCY")
    assert row["legacy_fail_like_claim_count"] == 1
    assert row["authority_state"] == "PENDING"
    assert inventory["authority_matrix"]["subject_disposition"] == "PENDING_AUTHORITY_AUDIT"


def test_unknown_family_is_rejected():
    try:
        inventory_subject(
            subject_id="subject",
            source_sha256=H1,
            family_evidence={"MAGIC_LABELS": [descriptor()]},
        )
    except ValueError as exc:
        assert "CORPUS_TRUTH_INVENTORY_UNKNOWN_FAMILY" in str(exc)
    else:
        raise AssertionError("unknown truth family was accepted")


def test_family_row_has_no_route_to_qualified_authority():
    row = make_family_row("SURFACE_GEOMETRY")
    assert row["authority_state"] == "PENDING"
    assert "coverage_fraction" not in row
    assert "validity_mask_sha256" not in row
