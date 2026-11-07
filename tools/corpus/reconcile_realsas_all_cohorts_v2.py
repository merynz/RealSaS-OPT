#!/usr/bin/env python3
"""Federate *all known* candidate cohorts without minting source or head truth.

Inputs are private Google Drive original ledgers and the prior SHA-pinned
read-only unified matrix. Output remains PRIVATE, never checked into GitHub.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

MASTER_LEDGER_SHA = "475b12c6876a7ba91d8b1b32b6acac29536431134b44a96277a74c2493e86913"
PROCESSED_LEDGER_SHA = "97c0b5bb27947689bc9eaacf8a46f1dd1c55b855b49fffaf59361801ba230b49"
SCHEMA = "RealSaS.GlobalAllCohortTruthScope.v2"


def require(ok, tag):
    if not ok:
        raise ValueError("UNIFIED_V2_" + tag)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def csvrows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def emit(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, columns, extrasaction="raise")
        w.writeheader()
        w.writerows(rows)
    return sha(path)


def validate_origins(original, processed, selected_inventory,
                     *, expected_master=3993, expected_qa=15, expected_reject=28):
    """Original historical selection is evidence; not current training admission."""
    require(len(original) == expected_master + expected_qa, "MASTER_CENSUS")
    original_by_id = {r["canonical_asset_id"]: r for r in original}
    require(len(original_by_id) == len(original), "MASTER_DUPLICATE_ID")
    require(len({r["candidate_id"] for r in original}) == len(original), "MASTER_DUPLICATE_CANDIDATE")
    qa = sorted([r for r in original if r["split"] == "QA_ONLY"],
                key=lambda r: r["canonical_asset_id"])
    selected = {key: r for key, r in original_by_id.items() if r["split"] != "QA_ONLY"}
    require(len(qa) == expected_qa and len(selected) == expected_master, "QA_SPLIT_CENSUS")
    rows = {r["canonical_asset_id"]: r for r in selected_inventory}
    require(len(rows) == expected_master and set(rows) == set(selected), "MASTER_INVENTORY_IDENTITY")
    for asset_id, row in rows.items():
        old = selected[asset_id]
        require(row["candidate_id"] == old["candidate_id"]
                and row["historical_split"] == old["split"]
                and row["provider"] == old["source_registry_id"],
                "MASTER_PROVENANCE_OR_SPLIT")
    require(len(processed) == len(original) + expected_reject, "PROCESSED_COUNT")
    rejected = sorted([r for r in processed if r.get("decision") == "REJECT_TECHNICAL"],
                      key=lambda r: r["candidate_id"])
    accepted = [r for r in processed if r.get("decision") != "REJECT_TECHNICAL"]
    require(len(accepted) == len(original) and len(rejected) == expected_reject,
            "PROCESSED_DECISIONS")
    accepted_by_id = {r["canonical_asset_id"]: r for r in accepted}
    require(len(accepted_by_id) == len(original) and
            set(accepted_by_id) == set(original_by_id), "PROCESSED_DUPLICATE")
    for asset_id, row in accepted_by_id.items():
        first = original_by_id[asset_id]
        require(all(row[k] == first[k] for k in ("candidate_id", "split", "decision",
                                                "source_registry_id")),
                "PROCESSED_MUTATED_ORIGINAL")
    rejected_ids = [r["candidate_id"] for r in rejected]
    require(len(set(rejected_ids)) == expected_reject
            and not (set(rejected_ids) & {r["candidate_id"] for r in original}),
            "REJECT_DUPLICATE_CANDIDATE")
    require(all(not r.get("canonical_asset_id")
                and "too_many_degenerate_faces" in
                r.get("audit", {}).get("geometry_reasons", []) for r in rejected),
            "REJECT_PROVENANCE")
    return selected, qa, rejected


def run(original_path, processed_path, inventory_path, v1_dir, blueprint_path, output_dir):
    require(sha(original_path) == MASTER_LEDGER_SHA, "MASTER_LEDGER_HASH")
    require(sha(processed_path) == PROCESSED_LEDGER_SHA, "PROCESSED_LEDGER_HASH")
    snapshot_path = v1_dir / "UNIFIED_TRUTH_SCOPE_SUMMARY_V1.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    require(snapshot["schema"] == "RealSaS.UnifiedTruthScopeProvisional.v1"
            and snapshot["tracked_source_rows"] == 4363
            and snapshot["qa_only_records_not_identity_bound"] == 15
            and snapshot["training_authorized"] is False, "V1_SCOPE")
    for name, expected in snapshot["output_sha256"].items():
        require(sha(v1_dir / name) == expected, "V1_OUTPUT_HASH_" + name)
    require(sha(inventory_path) ==
            snapshot["input_sha256"]["MASTER3993_SOURCE_TRUTH_INVENTORY_V1.csv"],
            "ORIGINAL_MASTER_INVENTORY_HASH")
    selected, qa, rejects = validate_origins(
        jsonl(original_path), jsonl(processed_path), csvrows(inventory_path))
    source = csvrows(v1_dir / "UNIFIED4363_SOURCE_SCOPE_V1.csv")
    facet = csvrows(v1_dir / "UNIFIED139616_FACET_EVIDENCE_V1.csv")
    routes = csvrows(v1_dir / "UNIFIED161431_HEAD_TARGET_ROUTES_V1.csv")
    blueprint = json.loads(blueprint_path.read_text())
    targets = blueprint["head_target_catalog"]
    require(blueprint["training_authorized"] is False and
            blueprint["materialization_authorized"] is False and len(targets) == 37
            and all(t["training_admitted"] is False and t["loss_mask_if_unverified"] is True
                    for t in targets), "HEAD_CATALOG_TRUTH")
    require(len(source) == 4363 and len(facet) == 139616 and len(routes) == 161431,
            "UNIFIED_V1_COUNTS")
    require({r["source_local_id"] for r in source if r["namespace"] == "MASTER3993"}
            == set(selected), "MASTER_SOURCE_ID_CROSSWALK")
    require(sum(r["split_state"] == "EXTERNAL_HOLDOUT" for r in source) == 169,
            "PROTECTED_HOLDOUT_COUNT")
    require(all(r["training_authorized"] == "False" for r in source)
            and all(r["training_target_status"] == "NOT_ADMITTED" and
                    r["negative_label_allowed"] == "False" for r in facet)
            and all(r["training_authorized"] == "False" and
                    r["loss_mask_valid"] == "False" and
                    r["negative_label_assigned"] == "False" for r in routes),
            "V1_AUTHORITY_DRIFT")
    facet_names = sorted({r["facet"] for r in facet if r["namespace"] == "MASTER3993"})
    require(len(facet_names) == 32 and
            all(t["source_facet"] in facet_names for t in targets), "FACET_CONTRACT")

    source_cols = list(source[0]) + ["candidate_id", "historical_decision",
                                   "source_cohort_status", "origin_ledger_sha256"]
    all_sources = []
    for r in source:
        is_master = r["namespace"] == "MASTER3993"
        old = selected[r["source_local_id"]] if is_master else None
        all_sources.append(dict(r, candidate_id=old["candidate_id"] if old else "",
                                historical_decision=old["decision"] if old else "",
                                source_cohort_status="SELECTED_NOT_CURRENT_ADMITTED"
                                if is_master else "CORE370_HISTORICAL_UNREVALIDATED",
                                origin_ledger_sha256=MASTER_LEDGER_SHA if old else ""))
    qa_provenance = []
    for r in qa:
        key = "MASTER_QA_ONLY:" + r["canonical_asset_id"]
        item = dict(zip(source_cols, [
            key, "MASTER_QA_ONLY", r["canonical_asset_id"], "",
            "CANDIDATE_ID_NOT_ORIGINAL_SOURCE_SHA", "QA_ONLY",
            "IDENTITY_RIGHTS_AND_LEAKAGE_REVIEW_REQUIRED", "QA_ONLY_NOT_TRAINING",
            "MASTER4008_HISTORICAL", "False", r["candidate_id"], r["decision"],
            "QA_ONLY_NO_TRAINING", MASTER_LEDGER_SHA,
        ]))
        all_sources.append(item)
        qa_provenance.append({
            "global_record_key": key, "canonical_asset_id": r["canonical_asset_id"],
            "candidate_id": r["candidate_id"], "provider": r["source_registry_id"],
            "historical_decision_NOT_AUTHORITY": r["decision"], "split": "QA_ONLY",
            "physical_original_sha256": "", "training_authorized": False,
        })
        for name in facet_names:
            facet.append({
                "global_record_key": key, "namespace": "MASTER_QA_ONLY",
                "facet": name, "evidence_state": "QA_UNKNOWN_NOT_SOURCE_TRUTH",
                "historical_family": "", "source_authority_tier": "QA_CANDIDATE_ONLY",
                "training_target_status": "NOT_ADMITTED", "negative_label_allowed": False,
            })
        for t in targets:
            routes.append({
                "global_record_key": key, "namespace": "MASTER_QA_ONLY",
                "head": t["head"], "target": t["target"],
                "source_facet": t["source_facet"],
                "source_evidence_state": "QA_UNKNOWN_NOT_SOURCE_TRUTH",
                "required_independent_court": t["required_independent_court"],
                "proposed_loss": t["proposed_loss"],
                "teacher_producer_proposed": t["teacher_producer"],
                "historical_hint_not_training": False,
                "teacher_authority": "PENDING_QA_SOURCE_COURT",
                "loss_mask_valid": False, "negative_label_assigned": False,
                "training_authorized": False,
            })

    rejection_queue = []
    for r in rejects:
        key = "MASTER_TECHNICAL_REJECT:" + r["candidate_id"]
        all_sources.append(dict(zip(source_cols, [
            key, "MASTER_TECHNICAL_REJECT", r["candidate_id"], "",
            "CANDIDATE_ONLY_NOT_A_PHYSICAL_SHA", "NO_SPLIT_QUARANTINED",
            "PENDING_ORIGINAL_SOURCE_RECOVERY_AND_DUPLICATE_COURT",
            "REJECT_HISTORICAL_NOT_TRAINING", "PROCESSED4036_HISTORICAL",
            "False", r["candidate_id"], "REJECT_TECHNICAL",
            "QUARANTINE_REPAIR_REVIEW", PROCESSED_LEDGER_SHA,
        ])))
        audit = r["audit"]
        rejection_queue.append({
            "global_record_key": key, "candidate_id": r["candidate_id"],
            "historical_reasons": ";".join(audit["geometry_reasons"] +
                                            audit["skin_reasons"] + audit["rig_reasons"]),
            "original_nondegenerate_fraction":
                audit["geometry_metrics"].get("nondegenerate_fraction"),
            "historical_bone_count": audit["rig_metrics"].get("bone_count"),
            "physical_original_sha256": "", "canonical_asset_id": "",
            "current_rejection_courted": False, "training_authorized": False,
        })
    require(len(all_sources) == 4406 and
            len({r["global_record_key"] for r in all_sources}) == 4406,
            "FINAL_GLOBAL_CANDIDATES")
    require(len(facet) == 140096 and len(routes) == 161986,
            "FINAL_QA_HEAD_ROUTES")
    require(all(r["training_target_status"] == "NOT_ADMITTED" for r in facet)
            and all(r["training_authorized"] in (False, "False")
                    and r["loss_mask_valid"] in (False, "False") for r in routes),
            "FALSE_TRAINING_ADMISSION")
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, rows, columns in (
        ("UNIFIED4406_CANDIDATE_SOURCE_REGISTER_V2.csv", all_sources, source_cols),
        ("MASTER_QA15_ORIGINAL_PROVENANCE_V2.csv", qa_provenance, list(qa_provenance[0])),
        ("MASTER_REJECT28_REVIEW_QUEUE_V2.csv", rejection_queue, list(rejection_queue[0])),
        ("UNIFIED140096_FACET_SCOPE_V2.csv", facet, list(facet[0])),
        ("UNIFIED161986_HEAD_TARGET_ROUTES_V2.csv", routes, list(routes[0])),
    ):
        files[name] = emit(output_dir / name, rows, columns)
    family_input = v1_dir / "UNIFIED57089_FAMILY_EVIDENCE_V1.csv"
    family_dest = output_dir / "UNIFIED57089_FAMILY_EVIDENCE_V2.csv"
    shutil.copyfile(family_input, family_dest)
    files[family_dest.name] = sha(family_dest)
    report = {
        "schema": SCHEMA, "source_register_count": 4406,
        "canonical_id_count_including_QA": 4378,
        "master_selected": 3993, "core370": 370, "QA_only": 15,
        "technical_reject_candidate_only": 28,
        "family_evidence_rows_unchanged": 57089,
        "facet_rows": 140096, "proposed_target_routes": 161986,
        "qa_head_truth_admitted": 0, "reject_head_truth_admitted": 0,
        "all_training_targets_masked": True,
        "exact_original_master_match": True,
        "original_master_ledger_sha256": MASTER_LEDGER_SHA,
        "processed_master_ledger_sha256": PROCESSED_LEDGER_SHA,
        "v1_manifest_sha256": sha(snapshot_path),
        "blueprint_sha256": sha(blueprint_path),
        "historical_reject_reasons": dict(sorted(Counter(
            reason for r in rejects
            for reason in r["audit"]["geometry_reasons"] +
                          r["audit"]["skin_reasons"] + r["audit"]["rig_reasons"]).items())),
        "physical_sha_dedup_final": False,
        "all_historical_cohorts_discovered": False,
        "training_authorized": False, "materialization_authorized": False,
        "output_sha256": files,
    }
    (output_dir / "UNIFIED_ALL_COHORT_SCOPE_SUMMARY_V2.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    for arg in ("master-ledger", "processed-ledger", "master-inventory",
                "v1", "blueprint", "output"):
        parser.add_argument("--" + arg, required=True, type=Path)
    a = parser.parse_args()
    report = run(a.master_ledger, a.processed_ledger, a.master_inventory,
                 a.v1, a.blueprint, a.output)
    print(json.dumps({k: report[k] for k in (
        "source_register_count", "QA_only", "technical_reject_candidate_only",
        "proposed_target_routes", "training_authorized")}, sort_keys=True))


if __name__ == "__main__":
    main()
