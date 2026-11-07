#!/usr/bin/env python3
"""Read-only Master3993 + Core370 truth-scope federation; never dedup by guess.

The 15 QA-only candidates remain a separate unbound roster count until exact
source identities are available. This script does NOT authorize training.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "RealSaS.UnifiedTruthScopeProvisional.v1"
HEADS = {"IRIS_GEOMETRY", "IRIS_SEMANTIC", "IRIS_GRAPHICS", "TESSA", "AXIS", "MIRA"}
# Conservative historic family-to-facet *evidence hint*, never a teacher label.
HIST_FACET = {
    "SOURCE_CAMERA_PROJECTION": "CAMERA_OBSERVATION",
    "SOURCE_MESH_VERTICES": "SURFACE_GEOMETRY_UDF",
    "SOURCE_NORMAL_ORIENTATION": "SURFACE_NORMALS",
    "GEOMETRY_OCCUPANCY_UDF_TSDF_SIGN": "SURFACE_GEOMETRY_UDF",
    "SOURCE_TOPOLOGY_INCIDENCE": "MECHANICAL_SURFACE_TOPOLOGY",
    "JOINT_HIERARCHY": "RIG_HIERARCHY",
    "SOURCE_SKIN_INFLUENCES": "SKIN_WEIGHTS",
    "ACTION_CURVE_INVENTORY": "ANIMATION_KINEMATICS",
    "VISIBILITY_OCCLUSION_DRAW_ORDER": "VISIBILITY_OCCLUSION",
    "RAY_DEPTH_COMPOSITING": "RAY_COMPOSITING",
}


def require(condition, reason):
    if not condition:
        raise ValueError("UNIFIED_SCOPE_" + reason)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def records(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def csv_writer(path, columns):
    stream = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(stream, columns, extrasaction="ignore")
    writer.writeheader()
    return stream, writer


def build(inputs: Path, source_facets: Path, blueprint: Path, output: Path,
          *, master_count=3993, core_count=370, qa_unbound=15):
    snapshot = json.loads((inputs / "GLOBAL_INVENTORY_SUMMARY_V1.json").read_text())
    core_snapshot = json.loads((inputs / "CORE370_MACHINE_PROJECTION_SUMMARY_V1.json").read_text())
    for name in ("MASTER3993_SOURCE_TRUTH_INVENTORY_V1.csv",
                 "MASTER3993_51909_TRUTH_FAMILY_CELLS_V1.csv"):
        require(digest(inputs / name) == snapshot["output_files"][name]["sha256"],
                "MASTER_MANIFEST_SHA_" + name)
    for name in ("CORE370_370_MACHINE_SUBJECTS_V1.csv",
                 "CORE370_5180_MACHINE_TRUTH_FAMILY_CELLS_V1.csv"):
        require(digest(inputs / name) == core_snapshot["output_files"][name]["sha256"],
                "CORE_MANIFEST_SHA_" + name)

    master = records(inputs / "MASTER3993_SOURCE_TRUTH_INVENTORY_V1.csv")
    core = records(inputs / "CORE370_370_MACHINE_SUBJECTS_V1.csv")
    master_family = records(inputs / "MASTER3993_51909_TRUTH_FAMILY_CELLS_V1.csv")
    core_family = records(inputs / "CORE370_5180_MACHINE_TRUTH_FAMILY_CELLS_V1.csv")
    facets = records(source_facets)
    catalog = json.loads(blueprint.read_text())["head_target_catalog"]
    require(len(master) == master_count and len(core) == core_count, "SOURCE_COUNT")
    require(len(master_family) == 13 * master_count and
            len(core_family) == 14 * core_count, "FAMILY_COUNT")
    require(len(facets) == 32 * master_count, "SOURCE_FACET_COUNT")
    require(len(catalog) == 37 and {t["head"] for t in catalog} == HEADS and
            all(t["training_admitted"] is False for t in catalog),
            "TARGET_AUTHORITY")
    require(len({x["canonical_asset_id"] for x in master}) == len(master) and
            len({x["subject_id"] for x in core}) == len(core), "DUPLICATE_SUBJECT")
    require(len({(r["subject_id"], r["family"]) for r in core_family}) == len(core_family) and
            len({(r["asset_id"], r["family"]) for r in master_family}) == len(master_family),
            "DUPLICATE_FAMILY")
    require(all(r["training_authorized"] == "False" for r in master + core + master_family),
            "SOURCE_TRAINING_AUTHORITY")
    require(all(r["current_PR327_authority_revalidated"] == "False"
                for r in core), "CURRENT_CORE_AUTHORITY")
    require(all(r["current_PR327_machine_authority_revalidated"] == "False" and
                r["core370_training_authorized"] == "False"
                for r in core_family), "CORE_FAMILY_AUTHORITY")
    require(all(r["authority"] == "PENDING_INDEPENDENT_FAMILY_COURT"
                for r in master_family), "MASTER_FAMILY_AUTHORITY")

    facet_map = defaultdict(dict)
    for row in facets:
        asset_id, facet = row["canonical_asset_id"], row["facet"]
        require(facet not in facet_map[asset_id], "DUPLICATE_FACET")
        require(row["evidence_authority"] == "SOURCE_OBSERVATION_NOT_TRUTH" and
                row["training_target_status"] == "NOT_ADMITTED", "PROMOTED_SOURCE_FACET")
        facet_map[asset_id][facet] = row["evidence_state"]
    master_ids = {m["canonical_asset_id"] for m in master}
    require(set(facet_map) == master_ids, "SOURCE_FACET_IDS")
    facet_names = set(next(iter(facet_map.values())))
    require(all(set(v) == facet_names and len(v) == 32 for v in facet_map.values()),
            "MISSING_FACET")
    require(all(t["source_facet"] in facet_names for t in catalog), "UNKNOWN_TARGET_FACET")

    verified = {s["source_recovery_sha_verified"] for s in master
                if s["source_recovery_sha_verified"]}
    core_hash = {s["source_sha256"] for s in core}
    require(len(core_hash) == core_count, "CORE_SHA_DUPLICATES")
    exact_overlap = verified & core_hash
    historical = {(r["subject_id"], r["family"]): r for r in core_family}

    output.mkdir(parents=True, exist_ok=True)
    path_source = output / "UNIFIED4363_SOURCE_SCOPE_V1.csv"
    path_family = output / "UNIFIED57089_FAMILY_EVIDENCE_V1.csv"
    path_facets = output / "UNIFIED139616_FACET_EVIDENCE_V1.csv"
    path_routes = output / "UNIFIED161431_HEAD_TARGET_ROUTES_V1.csv"
    source_stream, source_writer = csv_writer(path_source, (
        "global_record_key", "namespace", "source_local_id",
        "source_sha256_reported", "sha_evidence_tier", "split_state",
        "identity_crosswalk_state", "current_truth_authority",
        "historical_report_run", "training_authorized"))
    family_stream, family_writer = csv_writer(path_family, (
        "global_record_key", "namespace", "family", "observed_or_historical_state",
        "evidence_tier", "court_hashes_or_refs", "current_authority",
        "training_authorized"))
    facet_stream, facet_writer = csv_writer(path_facets, (
        "global_record_key", "namespace", "facet", "evidence_state",
        "historical_family", "source_authority_tier",
        "training_target_status", "negative_label_allowed"))
    route_stream, route_writer = csv_writer(path_routes, (
        "global_record_key", "namespace", "head", "target", "source_facet",
        "source_evidence_state", "required_independent_court", "proposed_loss",
        "teacher_producer_proposed", "historical_hint_not_training",
        "teacher_authority", "loss_mask_valid",
        "negative_label_assigned", "training_authorized"))

    def source_record(namespace, sid, sha, sha_tier, split, run):
        state = ("EXACT_SHA_MATCH_REQUIRES_INDEPENDENT_RECHECK"
                 if sha and sha in exact_overlap else
                 "NOT_CERTIFIED_NO_MATCH_IN_VERIFIED_SUBSET")
        source_writer.writerow({
            "global_record_key": namespace + ":" + sid, "namespace": namespace,
            "source_local_id": sid, "source_sha256_reported": sha,
            "sha_evidence_tier": sha_tier, "split_state": split,
            "identity_crosswalk_state": state,
            "current_truth_authority": "NOT_REVALIDATED",
            "historical_report_run": run, "training_authorized": False})

    def facet_record(namespace, sid, facet, state, historical_family=""):
        facet_writer.writerow({
            "global_record_key": namespace + ":" + sid, "namespace": namespace,
            "facet": facet, "evidence_state": state,
            "historical_family": historical_family,
            "source_authority_tier": "IMPORTED_OBSERVATION_NOT_TRUTH"
            if namespace == "MASTER3993" else
            "HISTORICAL_PARTIAL_MACHINE_NOT_CURRENT",
            "training_target_status": "NOT_ADMITTED",
            "negative_label_allowed": False})

    def routes(namespace, sid, evidence):
        for t in catalog:
            state = evidence[t["source_facet"]]
            route_writer.writerow({
                "global_record_key": namespace + ":" + sid,
                "namespace": namespace, "head": t["head"], "target": t["target"],
                "source_facet": t["source_facet"],
                "source_evidence_state": state,
                "required_independent_court": t["required_independent_court"],
                "proposed_loss": t["proposed_loss"],
                "teacher_producer_proposed": t["teacher_producer"],
                "historical_hint_not_training": state.startswith("HISTORICAL_"),
                "teacher_authority": "PENDING_INDEPENDENT_COURT",
                "loss_mask_valid": False, "negative_label_assigned": False,
                "training_authorized": False})

    for row in master:
        sid = row["canonical_asset_id"]
        source_record("MASTER3993", sid,
                      row["source_recovery_sha_verified"] or row["expected_source_sha256"],
                      "VERIFIED_RECOVERY" if row["source_recovery_sha_verified"]
                      else "UNVERIFIED_OR_MISSING",
                      row["historical_split"], "MASTER3993_V1")
        for facet in sorted(facet_names):
            facet_record("MASTER3993", sid, facet, facet_map[sid][facet])
        routes("MASTER3993", sid, facet_map[sid])
    for row in core:
        sid = row["subject_id"]
        source_record("CORE370", sid, row["source_sha256"],
                      "HISTORICAL_MACHINE_SHA_NOT_REHASHED",
                      "CORE370_SPLIT_REVALIDATION_REQUIRED", row["machine_report_run_id"])
        evidence = {}
        for facet in sorted(facet_names):
            family = HIST_FACET.get(facet, "")
            old = historical.get((sid, family)) if family else None
            if old is not None:
                state = "HISTORICAL_" + old["historical_machine_state"] + "_NOT_CURRENT_AUTHORITY"
            elif facet == "ORIGINAL_SOURCE_BYTES":
                state = "HISTORICAL_SOURCE_SHA_REPORTED_NOT_REHASHED"
            else:
                state = "UNKNOWN_NOT_OBSERVED_OR_NOT_MAPPED"
            evidence[facet] = state
            facet_record("CORE370", sid, facet, state, family)
        routes("CORE370", sid, evidence)
    for row in master_family:
        family_writer.writerow({
            "global_record_key": "MASTER3993:" + row["asset_id"],
            "namespace": "MASTER3993", "family": row["family"],
            "observed_or_historical_state": row["evidence_state"],
            "evidence_tier": "IMPORTED_OBSERVATION_ONLY",
            "court_hashes_or_refs": "",
            "current_authority": "PENDING_INDEPENDENT_FAMILY_COURT",
            "training_authorized": False})
    for row in core_family:
        family_writer.writerow({
            "global_record_key": "CORE370:" + row["subject_id"],
            "namespace": "CORE370", "family": row["family"],
            "observed_or_historical_state": row["historical_machine_state"],
            "evidence_tier": "HISTORICAL_MACHINE_PARTIAL_DIAGNOSTIC_ONLY",
            "court_hashes_or_refs": row["court_sha256s"] + "|" + row["evidence_sha256s"],
            "current_authority": "PR327_REVALIDATION_REQUIRED",
            "training_authorized": False})
    for stream in (source_stream, family_stream, facet_stream, route_stream):
        stream.close()

    report = {
        "schema": SCHEMA, "status": "PROVISIONAL_UNIVERSE_NO_DEDUP_OR_HEAD_ADMISSION",
        "tracked_source_rows": len(master) + len(core),
        "tracked_source_rows_by_namespace": {"MASTER3993": len(master), "CORE370": len(core)},
        "qa_only_records_not_identity_bound": qa_unbound,
        "qa_roster_status": "PENDING_EXACT_QA_ONLY_IDENTITIES__NO_FAKE_ASSET_ROWS",
        "known_source_rows_plus_unbound_qa": len(master) + len(core) + qa_unbound,
        "unique_characters_certified": False,
        "cross_corpus_verified_master_sha_count": len(verified),
        "core_historical_source_sha_count": len(core_hash),
        "exact_sha_overlap_in_verified_subset": len(exact_overlap),
        "master_without_recovered_verified_source_sha": len(master) - len(verified),
        "cross_corpus_semantic_geometric_duplicate_court": "PENDING",
        "core370_machine_authority": "HISTORICAL_NOT_CURRENT_PR327",
        "master_family_evidence_rows": len(master_family),
        "core370_historical_family_evidence_rows": len(core_family),
        "unified_family_evidence_rows": len(master_family) + len(core_family),
        "unified_facet_rows": 32 * (len(master) + len(core)),
        "head_target_catalog_count": len(catalog),
        "unified_head_target_route_rows": len(catalog) * (len(master) + len(core)),
        "core_historical_family_states": dict(sorted(
            Counter(x["historical_machine_state"] for x in core_family).items())),
        "blocked_unjoined_evidence": [
            "CURRENT_PR327_MACHINE_AUTHORITY",
            "V18_PLUS_SOURCE_FACE_AND_EDIT_RELATIONS",
            "WAVE4_GLOBAL_RECEIPTS", "QA_ONLY_SOURCE_ROSTER",
            "OTHER_SOURCE_COHORT_DISCOVERY"],
        "training_admitted_count": 0, "all_loss_masks_false": True,
        "training_authorized": False, "source_raw_bytes_read": False,
        "new_source_imports_or_renderers": False,
        "input_sha256": {p.name: digest(p) for p in (
            inputs / "MASTER3993_SOURCE_TRUTH_INVENTORY_V1.csv",
            inputs / "CORE370_370_MACHINE_SUBJECTS_V1.csv",
            inputs / "MASTER3993_51909_TRUTH_FAMILY_CELLS_V1.csv",
            inputs / "CORE370_5180_MACHINE_TRUTH_FAMILY_CELLS_V1.csv",
            source_facets, blueprint)},
        "output_sha256": {p.name: digest(p)
                          for p in (path_source, path_family, path_facets, path_routes)}
    }
    (output / "UNIFIED_TRUTH_SCOPE_SUMMARY_V1.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing", type=Path, required=True)
    parser.add_argument("--facets", type=Path, required=True)
    parser.add_argument("--blueprint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.existing, args.facets, args.blueprint, args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
