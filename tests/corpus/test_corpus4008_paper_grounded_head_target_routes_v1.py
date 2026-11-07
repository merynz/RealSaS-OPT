"""Paper-target route safety tests with synthetic data only; no private assets."""
import csv
import hashlib
import json
from pathlib import Path

import pytest

from tools.corpus.build_corpus4008_head_target_routes_v1 import build

CATALOG = Path(__file__).resolve().parents[2] / "canonical" / "CORPUS4008_PAPER_GROUNDED_HEAD_TARGET_BLUEPRINT_V1.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path):
    catalog = json.loads(CATALOG.read_text())
    facets = sorted({t["source_facet"] for t in catalog["head_target_catalog"]})
    facets.extend("EXTRA_" + str(i) for i in range(32 - len(facets)))
    source = tmp_path / "MASTER3993_SOURCE_FACET_EVIDENCE_V1.csv"
    columns = (
        "canonical_asset_id", "source_sha256_verified", "historical_split",
        "facet", "domain", "evidence_state", "evidence_authority",
        "required_court", "potential_consumers", "training_target_status",
        "negative_label_allowed", "source_file_mutated"
    )
    rows = [{
        "canonical_asset_id": "asset_test", "source_sha256_verified": "a" * 64,
        "historical_split": "FIT", "facet": f, "domain": "TEST",
        "evidence_state": "IMPORTED_SOURCE_OBSERVATION_ONLY",
        "evidence_authority": "SOURCE_OBSERVATION_NOT_TRUTH",
        "required_court": "REVIEW", "potential_consumers": "ALL",
        "training_target_status": "NOT_ADMITTED",
        "negative_label_allowed": "False", "source_file_mutated": "False",
    } for f in facets]

    def write(items):
        with source.open("w", newline="") as file:
            w = csv.DictWriter(file, fieldnames=columns)
            w.writeheader()
            w.writerows(items)

    write(rows)
    summary = tmp_path / "FACET_COVERAGE_SUMMARY_V1.json"

    def resign():
        summary.write_text(json.dumps({
            "schema": "RealSaS.Corpus4008SourceFacetCoverage.v1",
            "master_source_count": 1, "source_facet_count_per_asset": 32,
            "training_authorized": False, "source_truth_admission_minted": 0,
            "output_sha256": {source.name: digest(source)},
        }))

    resign()
    return source, summary, rows, write, resign


def test_routing_has_all_six_heads_but_zero_teacher_labels(tmp_path):
    source, summary, _, _, _ = fixture(tmp_path)
    out = tmp_path / "out"
    report = build(source, summary, CATALOG, out, expected_subjects=1)
    assert report["target_catalog_count"] == 37
    assert report["route_count"] == 37
    assert report["training_target_admitted_count"] == 0
    assert report["all_loss_masks_false"]
    rows = list(csv.DictReader(
        (out / "MASTER3993_PAPER_GROUNDED_HEAD_TARGET_ROUTES_V1.csv").open()
    ))
    assert {r["head"] for r in rows} == {
        "IRIS_GEOMETRY", "IRIS_SEMANTIC", "IRIS_GRAPHICS", "TESSA", "AXIS", "MIRA"
    }
    assert all(r["teacher_authority"] == "PENDING_INDEPENDENT_COURT"
               and r["loss_mask_valid"] == "False" and
               r["negative_label_assigned"] == "False" for r in rows)


def test_minted_teacher_label_rejected_even_after_resigning(tmp_path):
    source, summary, rows, write, resign = fixture(tmp_path)
    rows[0]["training_target_status"] = "PASS"
    write(rows)
    resign()
    with pytest.raises(ValueError, match="EVIDENCE_PROMOTION"):
        build(source, summary, CATALOG, tmp_path / "out", expected_subjects=1)


def test_tampered_immutable_facet_source_rejected(tmp_path):
    source, summary, _, _, _ = fixture(tmp_path)
    source.write_text(source.read_text().replace(
        "IMPORTED_SOURCE_OBSERVATION_ONLY", "UNKNOWN_OR_NOT_INSPECTED", 1
    ))
    with pytest.raises(ValueError, match="SOURCE_FACET_SHA"):
        build(source, summary, CATALOG, tmp_path / "out", expected_subjects=1)


def test_missing_source_facet_rejected(tmp_path):
    source, summary, rows, write, resign = fixture(tmp_path)
    write(rows[1:])
    resign()
    with pytest.raises(ValueError, match="FACET_ROW_COUNT"):
        build(source, summary, CATALOG, tmp_path / "out", expected_subjects=1)
