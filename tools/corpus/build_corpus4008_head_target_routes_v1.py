#!/usr/bin/env python3
"""Read-only source-facet -> model target planner.

Outputs must stay in private data storage: they contain source identities.
This tool NEVER mints teacher labels, target masks or training permission.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "RealSaS.Corpus4008PaperGroundedHeadTargetRoutes.v1"
FACET_SCHEMA = "RealSaS.Corpus4008SourceFacetCoverage.v1"
BLUEPRINT_SCHEMA = "RealSaS.Corpus4008PaperGroundedHeadTargetBlueprint.v1"
HEADS = {"IRIS_GEOMETRY", "IRIS_SEMANTIC", "IRIS_GRAPHICS", "TESSA", "AXIS", "MIRA"}


def require(ok, reason):
    if not ok:
        raise ValueError("HEAD_TARGET_ROUTE_" + reason)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    require(bool(rows), "EMPTY_RESULT")
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return sha(path)


def build(facets_path, summary_path, blueprint_path, out, *, expected_subjects=3993):
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    bp = json.loads(blueprint_path.read_text(encoding="utf-8"))
    require(summary.get("schema") == FACET_SCHEMA, "FACET_SCHEMA")
    require(summary["master_source_count"] == expected_subjects, "SUBJECT_COUNT")
    require(summary["training_authorized"] is False and
            summary["source_truth_admission_minted"] == 0, "SOURCE_AUTHORITY")
    require(sha(facets_path) == summary["output_sha256"][facets_path.name], "SOURCE_FACET_SHA")
    require(bp.get("schema") == BLUEPRINT_SCHEMA and
            bp["training_authorized"] is False and
            bp["materialization_authorized"] is False, "BLUEPRINT_AUTHORITY")
    targets = bp["head_target_catalog"]
    names = [(t["head"], t["target"]) for t in targets]
    require(len(set(names)) == len(names), "DUPLICATE_TARGET")
    require({t["head"] for t in targets} == HEADS, "HEAD_COVERAGE")
    require(all(t["training_admitted"] is False and
                t["loss_mask_if_unverified"] is True and
                t["source_observation_is_teacher"] is False for t in targets),
            "FORGED_TARGET_ADMISSION")

    records = read_csv(facets_path)
    nfacets = summary["source_facet_count_per_asset"]
    require(len(records) == expected_subjects * nfacets, "FACET_ROW_COUNT")
    grouped = defaultdict(dict)
    identities = {}
    for row in records:
        aid, facet = row["canonical_asset_id"], row["facet"]
        require(facet not in grouped[aid], "DUPLICATE_FACET")
        require(row["evidence_authority"] == "SOURCE_OBSERVATION_NOT_TRUTH" and
                row["training_target_status"] == "NOT_ADMITTED" and
                row["negative_label_allowed"] == "False" and
                row["source_file_mutated"] == "False", "EVIDENCE_PROMOTION")
        identity = (row["source_sha256_verified"], row["historical_split"])
        if aid in identities:
            require(identities[aid] == identity, "INCONSISTENT_IDENTITY")
        identities[aid] = identity
        grouped[aid][facet] = row
    require(len(grouped) == expected_subjects, "SUBJECT_UNIVERSE")
    require(all(len(f) == nfacets for f in grouped.values()), "MISSING_FACET")
    require(all(t["source_facet"] in next(iter(grouped.values())) for t in targets),
            "UNKNOWN_TARGET_FACET")

    rows = []
    hints = Counter()
    for aid in sorted(grouped):
        for target in targets:
            facet = grouped[aid][target["source_facet"]]
            state = facet["evidence_state"]
            hint = state in (
                "IMPORTED_SOURCE_OBSERVATION_ONLY",
                "SHA_RECOVERY_VERIFIED_ONLY",
                "NUMERIC_IMPORT_CONSISTENCY_ONLY",
            )
            hints[target["head"] + "|" + target["target"] + "|" + str(hint)] += 1
            rows.append({
                "canonical_asset_id": aid,
                "source_sha256_verified": identities[aid][0],
                "historical_split": identities[aid][1],
                "head": target["head"],
                "target": target["target"],
                "source_facet": target["source_facet"],
                "source_evidence_state": state,
                "import_observation_hint": hint,
                "teacher_producer_proposed": target["teacher_producer"],
                "required_independent_court": target["required_independent_court"],
                "proposed_loss": target["proposed_loss"],
                "research_reference": target["research_reference"],
                "teacher_authority": "PENDING_INDEPENDENT_COURT",
                "loss_mask_valid": False,
                "negative_label_assigned": False,
                "materialization_authorized": False,
                "training_authorized": False,
            })
    require(len(rows) == expected_subjects * len(targets), "ROUTE_COUNT")
    out.mkdir(parents=True, exist_ok=True)
    routes = out / "MASTER3993_PAPER_GROUNDED_HEAD_TARGET_ROUTES_V1.csv"
    output_sha = write_csv(routes, rows)
    report = {
        "schema": SCHEMA,
        "master_source_count": expected_subjects,
        "target_catalog_count": len(targets),
        "route_count": len(rows),
        "head_target_count": dict(sorted(Counter(t["head"] for t in targets).items())),
        "observation_hint_counts": dict(sorted(hints.items())),
        "source_facet_sha256": sha(facets_path),
        "source_facet_summary_sha256": sha(summary_path),
        "target_catalog_sha256": sha(blueprint_path),
        "routes_sha256": output_sha,
        "truth_courts_passed_by_this_join": 0,
        "training_target_admitted_count": 0,
        "all_loss_masks_false": True,
        "training_authorized": False,
        "materialization_authorized": False,
        "source_bytes_accessed": False,
        "new_render_or_rig_experiment": False,
    }
    (out / "HEAD_TARGET_ROUTE_SUMMARY_V1.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main():
    p = argparse.ArgumentParser()
    for item in ("facets", "facets-summary", "blueprint", "output"):
        p.add_argument("--" + item, type=Path, required=True)
    a = p.parse_args()
    report = build(a.facets, a.facets_summary, a.blueprint, a.output)
    print(json.dumps({k: report[k] for k in (
        "master_source_count", "target_catalog_count", "route_count",
        "truth_courts_passed_by_this_join", "training_authorized"
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
