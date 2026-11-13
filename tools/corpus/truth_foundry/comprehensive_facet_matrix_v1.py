#!/usr/bin/env python3
"""Exhaustive source-record x evidence-facet inventory (not training admission).

The catalog describes *questions to prove*; this reader only binds existing
source census observations and emits explicit independent-court gaps.
An imported count of 0 is a real importer observation, NOT proof that the
authoring source lacks controls, textures or the corresponding truth.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import io
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re

SCHEMA = "RealSaS.TruthFoundryComprehensiveEvidenceFacetMatrix.v1"
CATALOG_SCHEMA = "RealSaS.TruthFoundryComprehensiveEvidenceFacetCatalog.v1"
SHA64 = re.compile(r"^[0-9a-f]{64}$")
IMPORT = "IMPORTED_NUMERIC_OBSERVATION_ONLY"
UNKNOWN = "UNKNOWN_UNREVIEWED"
SHA_RECORDED = "RECORDED_SOURCE_IDENTITY_NOT_REHASHED_BY_THIS_BUILDER"
SPLIT_RECORDED = "HISTORICAL_SPLIT_ONLY__LINEAGE_COURT_PENDING"


def require(value, reason):
    if not value:
        raise ValueError("TRUTH_FACET_MATRIX_" + reason)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


@contextmanager
def gzip_text_writer(path):
    # Normalized empty filename and mtime=0 for reproducible SHA receipts.
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw,
                           compresslevel=6, mtime=0) as gz:
            with io.TextIOWrapper(gz, encoding="utf-8", newline="\n") as fp:
                yield fp


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as fp:
        return list(csv.DictReader(fp))


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields, extrasaction="raise")
        w.writeheader()
        w.writerows(rows)


def load_catalog(path):
    obj = json.loads(path.read_text(encoding="utf-8"))
    facets = obj.get("facets", [])
    require(obj.get("schema") == CATALOG_SCHEMA, "CATALOG_SCHEMA")
    require(obj.get("facet_count") == len(facets) and len(facets) >= 200, "CATALOG_SIZE")
    names = [f.get("id") for f in facets]
    require(all(isinstance(k, str) and re.fullmatch("[a-z][a-z0-9_]+", k) for k in names),
            "CATALOG_FACET_IDS")
    require(len(set(names)) == len(names), "CATALOG_DUPLICATE")
    groups = {f.get("group") for f in facets}
    require(len(groups) == obj["group_count"] and len(groups) >= 12, "CATALOG_GROUPS")
    for f in facets:
        require(f.get("default_state") == "UNKNOWN" and
                f.get("training_authorized") is False and
                f.get("label_authority") == "NONE_UNTIL_INDEPENDENT_COURT",
                "CATALOG_FALSE_AUTHORITY:" + f["id"])
        require(isinstance(f.get("consumer_heads"), list) and
                f["consumer_heads"] and isinstance(f.get("independent_court"), str),
                "CATALOG_UNOWNED:" + f["id"])
        require(f["domain"] in ("IDENTITY", "SOURCE", "SEMANTIC", "OBSERVATION",
                                "DERIVED", "BEHAVIOR", "PRODUCT", "COURT"),
                "CATALOG_DOMAIN:" + f["id"])
    return obj, facets


def run(source_csv: Path, catalog_json: Path, out: Path, *,
        expected_master=3993, expected_core=370, expected_source_sha=None):
    catalog, facets = load_catalog(catalog_json)
    source = read_csv(source_csv)
    require(len(source) == expected_master + expected_core, "SOURCE_ROW_COUNT")
    if expected_source_sha:
        require(digest(source_csv) == expected_source_sha, "FROZEN_SOURCE_MATRIX_SHA")
    seen = set()
    for row in source:
        cohort = row.get("corpus")
        require(cohort in ("MASTER3993", "CORE370"), "SOURCE_COHORT")
        key = (cohort, row.get("asset_id"))
        require(key[1] and key not in seen, "SOURCE_DUPLICATE_OR_BLANK")
        seen.add(key)
        require(row.get("training_authorized", "").lower() == "false" and
                row.get("materialization_authorized", "").lower() == "false",
                "INPUT_ALREADY_AUTHORIZED")
        source_sha = row.get("source_sha256", "")
        require(not source_sha or SHA64.fullmatch(source_sha), "INVALID_RECORDED_SHA")
    counts = Counter(cohort for cohort, _ in seen)
    require(counts["MASTER3993"] == expected_master and
            counts["CORE370"] == expected_core, "COHORT_BOUNDARY")
    out.mkdir(parents=True, exist_ok=True)
    observed = Counter()
    grouped = defaultdict(Counter)
    input_master_imported = 0
    rows_by_asset = []
    group_tasks = []
    group_index = defaultdict(list)
    for f in facets:
        group_index[f["group"]].append(f)
    # Stable grouped ordering guarantees deterministic text and digest receipts.
    with gzip_text_writer(out / "ASSET_FACET_EVIDENCE_MATRIX_V1.jsonl.gz") as handle:
        for row in source:
            cohort = row["corpus"]
            asset_id = row["asset_id"]
            is_master = cohort == "MASTER3993"
            phys = str(row.get("physical_census_loaded", "")).lower() == "true"
            if phys:
                require(is_master, "CORE_FALSE_PHYSICAL_CENSUS")
                input_master_imported += 1
            evidence = {}
            observed_by_asset = 0
            for f in facets:
                key = f["id"]
                numeric = f.get("preexisting_numeric_observation_field")
                if numeric and phys:
                    raw = row.get(numeric)
                    require(raw is not None and re.fullmatch(r"[0-9]+", str(raw)),
                            "MISSING_OR_NONNUMERIC_IMPORT:" + asset_id + ":" + key)
                    state = IMPORT
                    value = int(raw)
                    observed_by_asset += 1
                    grouped[(cohort, f["group"])][IMPORT] += 1
                elif key == "original_source_full_byte_sha" and row.get("source_sha256"):
                    state = SHA_RECORDED
                    value = row["source_sha256"]
                    grouped[(cohort, f["group"])][SHA_RECORDED] += 1
                elif key == "historical_split_source_firewall" and row.get("historical_split"):
                    state = SPLIT_RECORDED
                    value = row["historical_split"]
                    grouped[(cohort, f["group"])][SPLIT_RECORDED] += 1
                else:
                    state = UNKNOWN
                    value = None
                    grouped[(cohort, f["group"])][UNKNOWN] += 1
                observed[(cohort, key, state)] += 1
                evidence[key] = {
                    "observation_state": state,
                    "observed_value": value,
                    "independent_court_state": "UNKNOWN",
                    "court_receipt_sha256": None,
                    "supervision_status": "NOT_ADMITTED",
                    "training_mask_valid": False,
                }
            write = {
                "schema": SCHEMA,
                "corpus": cohort, "asset_id": asset_id,
                "source_sha256_recorded": row.get("source_sha256") or None,
                "original_source_sha_rehash_performed": False,
                "historical_split": row.get("historical_split") or "UNKNOWN",
                "facet_total": len(facets),
                "numeric_observation_slots": observed_by_asset,
                "independent_court_pass_count_from_this_builder": 0,
                "facet_evidence": evidence,
                "training_authorized": False,
                "materialization_authorized": False,
            }
            handle.write(json.dumps(write, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            rows_by_asset.append({
                "corpus": cohort, "asset_id": asset_id,
                "historical_split": row.get("historical_split", ""),
                "original_sha_recorded": bool(row.get("source_sha256")),
                "physical_import_summary_bound": phys,
                "facet_count": len(facets),
                "numeric_import_slots_not_truth": observed_by_asset,
                "remaining_facet_independent_courts": len(facets),
                "independent_court_pass_count": 0,
                "head_training_admitted": False,
                "unique_character_verified": False,
            })
            for group, entries in sorted(group_index.items()):
                group_tasks.append({
                    "corpus": cohort, "asset_id": asset_id, "group": group,
                    "facet_count": len(entries),
                    "numeric_import_hints": sum(e["preexisting_numeric_observation_field"] is not None and phys for e in entries),
                    "independent_court_state": "UNKNOWN",
                    "review_receipt_sha256": "",
                    "task_status": "OPEN__NOT_REVIEWED",
                    "required_courts": ";".join(sorted({e["independent_court"] for e in entries})),
                    "training_authorized": False,
                })
    write_csv(out / "ASSET_FACET_REVIEW_SUMMARY_V1.csv", rows_by_asset, list(rows_by_asset[0]))
    write_csv(out / "ASSET_GROUP_REVIEW_QUEUE_V1.csv", group_tasks, list(group_tasks[0]))
    coverage = []
    for f in facets:
        for cohort in ("MASTER3993", "CORE370"):
            states = {state: count for (co, key, state), count in observed.items()
                      if co == cohort and key == f["id"]}
            coverage.append({
                "corpus": cohort, "facet_id": f["id"], "group": f["group"],
                "domain": f["domain"], "applicability_scope": f["applicability_scope"],
                "head_consumers": ";".join(f["consumer_heads"]),
                "independent_court": f["independent_court"],
                "record_count": counts[cohort],
                "import_count_observed_only": states.get(IMPORT, 0),
                "historical_identifier_only": states.get(SHA_RECORDED, 0) + states.get(SPLIT_RECORDED, 0),
                "unobserved_unknown": states.get(UNKNOWN, 0),
                "independent_court_PASS": 0,
                "qualified_loss_mask_count": 0,
                "training_authorized": False,
            })
    write_csv(out / "FACET_COVERAGE_BY_COHORT_V1.csv", coverage, list(coverage[0]))
    group_report = []
    for (cohort, group), c in sorted(grouped.items()):
        group_report.append({
            "corpus": cohort, "group": group,
            "facet_slots": sum(c.values()),
            "import_count_observations": c[IMPORT],
            "recorded_identity_and_split": c[SHA_RECORDED] + c[SPLIT_RECORDED],
            "unknown_unobserved": c[UNKNOWN],
            "independent_court_PASS": 0,
            "qualified_loss_mask_count": 0,
        })
    write_csv(out / "GROUP_COVERAGE_BY_COHORT_V1.csv", group_report, list(group_report[0]))
    summary = {
        "schema": SCHEMA, "catalog_sha256": digest(catalog_json),
        "source_roster_sha256": digest(source_csv),
        "source_record_count": len(source),
        "master_count": expected_master, "core_count": expected_core,
        "facet_count": len(facets), "group_count": len(group_index),
        "asset_facet_slots": len(source)*len(facets),
        "group_review_tasks_open": len(group_tasks),
        "master_physical_census_bound": input_master_imported,
        "import_numeric_observations": sum(row["numeric_import_slots_not_truth"] for row in rows_by_asset),
        "independent_target_court_pass_minted": 0,
        "source_category_visual_court_pass_minted": 0,
        "source_face_semantic_labels_minted": 0,
        "graphics_drawable_pass_minted": 0,
        "training_authorized": False, "materialization_authorized": False,
        "verified_unique_characters": "UNKNOWN",
        "file_sha256": {p.name: digest(p) for p in sorted(out.iterdir()) if p.is_file()},
    }
    (out / "SUMMARY_V1.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-census", type=Path, required=True)
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--expected-master", type=int, default=3993)
    p.add_argument("--expected-core", type=int, default=370)
    p.add_argument("--expected-source-sha", default=None)
    a = p.parse_args()
    print(json.dumps(run(a.source_census, a.catalog, a.out,
                         expected_master=a.expected_master,
                         expected_core=a.expected_core,
                         expected_source_sha=a.expected_source_sha), indent=2))


if __name__ == "__main__":
    main()
