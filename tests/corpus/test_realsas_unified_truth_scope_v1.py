"""Global truth union: synthetic fail-closed lineage and admission tests."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from tools.corpus.build_realsas_unified_truth_scope_v1 import build

BLUEPRINT = (
    Path(__file__).resolve().parents[2]
    / "canonical/CORPUS4008_PAPER_GROUNDED_HEAD_TARGET_BLUEPRINT_V1.json"
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, records):
    with path.open("w", newline="", encoding="utf-8") as stream:
        w = csv.DictWriter(stream, list(records[0]))
        w.writeheader()
        w.writerows(records)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def fixture(root):
    root.mkdir()
    master = [{
        "canonical_asset_id": "asset_one",
        "source_recovery_sha_verified": "a" * 64,
        "expected_source_sha256": "a" * 64,
        "historical_split": "EXTERNAL_HOLDOUT",
        "training_authorized": "False",
    }]
    core = [{
        "subject_id": "corpus_one",
        "source_sha256": "b" * 64,
        "machine_report_run_id": "HISTORICAL_PARTIAL",
        "training_authorized": "False",
        "current_PR327_authority_revalidated": "False",
    }]
    master_families = [{
        "asset_id": "asset_one", "family": "MF_" + str(i),
        "evidence_state": "UNKNOWN", "authority": "PENDING_INDEPENDENT_FAMILY_COURT",
        "training_authorized": "False",
    } for i in range(13)]
    core_names = [
        "CAMERA_OBSERVATION", "RIG_HIERARCHY", "SURFACE_GEOMETRY_UDF",
        "SOURCE_IMPORT", "SURFACE_NORMALS", "SKIN_WEIGHTS",
        "ANIMATION_KINEMATICS", "MECHANICAL_SURFACE_TOPOLOGY",
        "VISIBILITY_OCCLUSION", "RAY_COMPOSITING"
    ] + ["CF_" + str(i) for i in range(4)]
    core_families = [{
        "subject_id": "corpus_one", "family": family,
        "historical_machine_state": "PASS" if family == "CAMERA_OBSERVATION" else "UNKNOWN",
        "court_sha256s": "recorded-court", "evidence_sha256s": "recorded-evidence",
        "current_PR327_machine_authority_revalidated": "False",
        "core370_training_authorized": "False",
    } for family in core_names]

    base_files = {
        "MASTER3993_SOURCE_TRUTH_INVENTORY_V1.csv": master,
        "CORE370_370_MACHINE_SUBJECTS_V1.csv": core,
        "MASTER3993_51909_TRUTH_FAMILY_CELLS_V1.csv": master_families,
        "CORE370_5180_MACHINE_TRUTH_FAMILY_CELLS_V1.csv": core_families,
    }
    for name, rows in base_files.items():
        write_csv(root / name, rows)

    def resign():
        names = list(base_files)
        master_names = (names[0], names[2])
        core_names = (names[1], names[3])
        (root / "GLOBAL_INVENTORY_SUMMARY_V1.json").write_text(
            json.dumps({"output_files": {
                name: {"sha256": digest(root / name)} for name in master_names}}))
        (root / "CORE370_MACHINE_PROJECTION_SUMMARY_V1.json").write_text(
            json.dumps({"output_files": {
                name: {"sha256": digest(root / name)} for name in core_names}}))

    resign()
    catalog = json.loads(BLUEPRINT.read_text())["head_target_catalog"]
    facet_names = set(t["source_facet"] for t in catalog)
    while len(facet_names) < 32:
        facet_names.add("SYNTHETIC_FACET_" + str(len(facet_names)))
    assert len(facet_names) == 32
    facets = root.parent / "facets.csv"
    write_csv(facets, [{
        "canonical_asset_id": "asset_one", "facet": name,
        "evidence_authority": "SOURCE_OBSERVATION_NOT_TRUTH",
        "training_target_status": "NOT_ADMITTED",
        "evidence_state": "UNKNOWN_OR_NOT_INSPECTED",
    } for name in sorted(facet_names)])
    return root, facets, resign


def test_both_source_universes_are_preserved_without_fake_pass(tmp_path):
    root, facets, _ = fixture(tmp_path / "input")
    output = tmp_path / "out"
    report = build(root, facets, BLUEPRINT, output,
                   master_count=1, core_count=1, qa_unbound=15)
    assert report["tracked_source_rows"] == 2
    assert report["unified_family_evidence_rows"] == 27
    assert report["unified_facet_rows"] == 64
    assert report["unified_head_target_route_rows"] == 74
    assert report["qa_only_records_not_identity_bound"] == 15
    assert report["exact_sha_overlap_in_verified_subset"] == 0
    assert not report["unique_characters_certified"]
    assert report["training_admitted_count"] == 0
    source_rows = read_csv(output / "UNIFIED4363_SOURCE_SCOPE_V1.csv")
    assert [r["namespace"] for r in source_rows] == ["MASTER3993", "CORE370"]
    assert source_rows[0]["split_state"] == "EXTERNAL_HOLDOUT"
    routes = read_csv(output / "UNIFIED161431_HEAD_TARGET_ROUTES_V1.csv")
    assert all(r["loss_mask_valid"] == "False" and
               r["training_authorized"] == "False" and
               r["negative_label_assigned"] == "False" for r in routes)
    assert any(r["namespace"] == "CORE370" and
               r["source_evidence_state"] == "HISTORICAL_PASS_NOT_CURRENT_AUTHORITY"
               for r in routes)


def test_tampered_master_sha_fails_without_rewriting_receipt(tmp_path):
    root, facets, _ = fixture(tmp_path / "input")
    source = root / "MASTER3993_SOURCE_TRUTH_INVENTORY_V1.csv"
    source.write_text(source.read_text().replace("asset_one", "asset_two"))
    with pytest.raises(ValueError, match="MASTER_MANIFEST_SHA"):
        build(root, facets, BLUEPRINT, tmp_path / "out",
              master_count=1, core_count=1)


def test_promoted_core_record_rejected_even_after_resigning(tmp_path):
    root, facets, resign = fixture(tmp_path / "input")
    path = root / "CORE370_370_MACHINE_SUBJECTS_V1.csv"
    rows = read_csv(path)
    rows[0]["current_PR327_authority_revalidated"] = "True"
    write_csv(path, rows)
    resign()
    with pytest.raises(ValueError, match="CURRENT_CORE_AUTHORITY"):
        build(root, facets, BLUEPRINT, tmp_path / "out",
              master_count=1, core_count=1)


def test_sha_match_is_not_automatic_identity_merger(tmp_path):
    root, facets, resign = fixture(tmp_path / "input")
    path = root / "CORE370_370_MACHINE_SUBJECTS_V1.csv"
    rows = read_csv(path)
    rows[0]["source_sha256"] = "a" * 64
    write_csv(path, rows)
    resign()
    report = build(root, facets, BLUEPRINT, tmp_path / "out",
                   master_count=1, core_count=1)
    assert report["exact_sha_overlap_in_verified_subset"] == 1
    assert report["tracked_source_rows"] == 2
    assert not report["unique_characters_certified"]
