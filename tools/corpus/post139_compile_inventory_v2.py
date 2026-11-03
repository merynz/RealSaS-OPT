"""Compile Post139 source/audit evidence into the corrected V2 global ledgers.

This is the semantic compiler for the source corpus.  It replaces the earlier
"count every audited row" assumption with an explicit subject policy while
keeping all source/audit evidence intact:

* Base139 rows are immutable, but publisher representation variants can be
  explicitly marked as non-Base139 reference variants.
* Post139 authored loadout/pose variants can be excluded from the subject
  denominator without deleting their audit evidence.
* appearance/mechanical aliases can collapse into an authored subject identity;
* structural identity, lineage, admission and materialization remain separate.

Output schemas intentionally remain V2 so downstream authority tooling consumes
one stable contract.  The policy digest is embedded additively in every root.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tools.corpus.post139_finalize_inventory_v2 import (
    CLEAN_PROVENANCE,
    REGISTRY_SCHEMA,
    UnionFind,
    digest,
    publisher_inventory_row,
    read_json,
    sha256_file,
    source_evidence,
    subject_rows,
    write_json,
)

POLICY_SCHEMA = "RealSaS.Post139.SubjectPolicy.v1"
PRIMARY_STATUSES = {"ADMIT_PRIMARY"}
AUXILIARY_STATUSES = {"AUXILIARY_STATIC_PRIMARY", "AUXILIARY_POLICY_PRIMARY"}
LINEAGE_STATUSES = {"REFERENCE_BASE139_ROW"} | PRIMARY_STATUSES | AUXILIARY_STATUSES


def load_policy(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "schema": POLICY_SCHEMA,
            "exclude_subject_ids": [],
            "alias_subject_ids": {},
            "reference_include_subject_ids_by_source": {},
        }
    payload = read_json(path)
    if payload.get("schema") != POLICY_SCHEMA:
        raise SystemExit("POST139_SUBJECT_POLICY_SCHEMA_MISMATCH")
    return payload


def semantic_state(row: dict[str, Any], policy: dict[str, Any]) -> tuple[str, str | None]:
    sid = row["subject_id"]
    excluded = set(policy.get("exclude_subject_ids", []))
    aliases = dict(policy.get("alias_subject_ids", {}))
    reference_selection = policy.get("reference_include_subject_ids_by_source", {})
    if sid in excluded:
        return "EXCLUDED_BY_SEMANTIC_POLICY", None
    if sid in aliases:
        return "SEMANTIC_ALIAS_COLLAPSE", aliases[sid]
    if row["role"] == "REFERENCE_BASE139":
        selected = reference_selection.get(row["source_id"])
        if selected is not None and sid not in set(selected):
            return "REFERENCE_VARIANT_NOT_IN_BASE139", None
    return "ELIGIBLE", None


def validate_policy(all_subjects: list[dict[str, Any]], registry: dict[str, Any], policy: dict[str, Any]) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    by_id = {row["subject_id"]: row for row in all_subjects}
    if len(by_id) != len(all_subjects):
        blockers.append({"code": "GLOBAL_SUBJECT_ID_COLLISION"})
    excluded = set(policy.get("exclude_subject_ids", []))
    aliases = dict(policy.get("alias_subject_ids", {}))
    overlap = sorted(excluded & set(aliases))
    if overlap:
        blockers.append({"code": "SUBJECT_POLICY_EXCLUDE_ALIAS_OVERLAP", "subject_ids": overlap})
    for sid in sorted(excluded):
        if sid not in by_id:
            blockers.append({"code": "SUBJECT_POLICY_UNKNOWN_EXCLUSION", "subject_id": sid})
    for sid, canonical in sorted(aliases.items()):
        if sid not in by_id or canonical not in by_id:
            blockers.append({"code": "SUBJECT_POLICY_UNKNOWN_ALIAS", "subject_id": sid, "canonical_subject_id": canonical})
            continue
        if sid == canonical:
            blockers.append({"code": "SUBJECT_POLICY_SELF_ALIAS", "subject_id": sid})
        if by_id[sid]["source_id"] != by_id[canonical]["source_id"]:
            blockers.append({
                "code": "SUBJECT_POLICY_ALIAS_CROSS_SOURCE",
                "subject_id": sid,
                "canonical_subject_id": canonical,
                "source_id": by_id[sid]["source_id"],
                "canonical_source_id": by_id[canonical]["source_id"],
            })
        if canonical in excluded:
            blockers.append({"code": "SUBJECT_POLICY_ALIAS_TARGET_EXCLUDED", "subject_id": sid, "canonical_subject_id": canonical})
        if canonical in aliases:
            blockers.append({"code": "SUBJECT_POLICY_ALIAS_CHAIN_FORBIDDEN", "subject_id": sid, "canonical_subject_id": canonical})
    registry_ids = {row["id"] for row in registry["sources"]}
    for source_id, selected in sorted(policy.get("reference_include_subject_ids_by_source", {}).items()):
        if source_id not in registry_ids:
            blockers.append({"code": "SUBJECT_POLICY_UNKNOWN_REFERENCE_SOURCE", "source_id": source_id})
            continue
        for sid in selected:
            row = by_id.get(sid)
            if row is None:
                blockers.append({"code": "SUBJECT_POLICY_UNKNOWN_REFERENCE_SUBJECT", "source_id": source_id, "subject_id": sid})
            elif row["source_id"] != source_id or row["role"] != "REFERENCE_BASE139":
                blockers.append({"code": "SUBJECT_POLICY_REFERENCE_SELECTION_SCOPE_MISMATCH", "source_id": source_id, "subject_id": sid})
    return blockers


def compile_inventory(registry: dict[str, Any], workspace: Path, policy: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    blockers: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    all_subjects: list[dict[str, Any]] = []
    publisher_sources: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []
    order = {row["id"]: index for index, row in enumerate(registry["sources"])}

    for source in registry["sources"]:
        source_id = source["id"]
        if source["role"] == "QUARANTINE":
            publisher_sources.append({
                "source_id": source_id,
                "source_family": source["source_family"],
                "pack": source["pack"],
                "role": source["role"],
                "provenance": source["provenance"],
                "license": source.get("license"),
                "status": "QUARANTINED_NOT_ACQUIRED",
            })
            source_summaries.append({"source_id": source_id, "role": source["role"], "status": "QUARANTINED"})
            continue
        evidence, source_blockers = source_evidence(workspace, source)
        blockers.extend(source_blockers)
        if evidence is None:
            source_summaries.append({"source_id": source_id, "role": source["role"], "status": "MISSING_EVIDENCE"})
            continue
        publisher_sources.append(publisher_inventory_row(source, evidence))
        rows = subject_rows(workspace, source, evidence)
        all_subjects.extend(rows)
        full_static = [row for row in rows if row["full_static_truth_ready"]]
        full_animated = [row for row in full_static if row["animation_truth"]]
        source_summaries.append({
            "source_id": source_id,
            "source_family": source["source_family"],
            "pack": source["pack"],
            "role": source["role"],
            "status": "AUDITED",
            "publisher_files": evidence["publisher"].get("file_count"),
            "acquired_files": evidence["ledger"].get("file_count"),
            "audited_rows": len(rows),
            "full_static_rows": len(full_static),
            "full_animated_rows": len(full_animated),
            "source_closure_ledger_sha256": evidence["ledger"].get("ledger_sha256"),
        })

    blockers.extend(validate_policy(all_subjects, registry, policy))
    by_id = {row["subject_id"]: row for row in all_subjects}
    policy_digest = digest(policy)

    # Source-level semantic count gates are applied after explicit policy, not to
    # every publisher representation that happened to be auditable.
    for source in registry["sources"]:
        if source["role"] == "QUARANTINE":
            continue
        rows = [row for row in all_subjects if row["source_id"] == source["id"]]
        included = [row for row in rows if semantic_state(row, policy)[0] == "ELIGIBLE"]
        full_static = [row for row in included if row["full_static_truth_ready"]]
        full_animated = [row for row in full_static if row["animation_truth"]]
        if source["role"] == "REFERENCE_BASE139":
            expected = int(source["expected_subject_count"])
            if len(full_animated) != expected:
                blockers.append({
                    "code": "BASE139_TRAINING_ROW_COUNT_MISMATCH",
                    "source_id": source["id"],
                    "expected": expected,
                    "observed": len(full_animated),
                    "included_subject_ids": [row["subject_id"] for row in full_animated],
                })
            structural = {row["fingerprints"].get("structural_subject") for row in full_animated if row["fingerprints"].get("structural_subject")}
            if len(structural) < len(full_animated):
                warnings.append({
                    "code": "BASE139_NONSTRUCTURAL_VARIANTS_PRESERVED",
                    "source_id": source["id"],
                    "training_rows": len(full_animated),
                    "structural_identities": len(structural),
                    "delta": len(full_animated) - len(structural),
                })
        elif source["role"] == "POST139_CANDIDATE" and source.get("checkpoint_candidate_subjects") is not None:
            use_static = source.get("checkpoint_truth", {}).get("animation") == "NO"
            observed = full_static if use_static else full_animated
            expected = int(source["checkpoint_candidate_subjects"])
            if len(observed) != expected:
                blockers.append({
                    "code": "POST139_SEMANTIC_ROW_COUNT_REVIEW",
                    "source_id": source["id"],
                    "checkpoint_expected": expected,
                    "observed_after_policy": len(observed),
                    "included_subject_ids": [row["subject_id"] for row in observed],
                })

    admissions: list[dict[str, Any]] = []
    preclassified: set[str] = set()
    for row in all_subjects:
        state, canonical = semantic_state(row, policy)
        if state == "ELIGIBLE":
            continue
        admissions.append({
            "subject_id": row["subject_id"],
            "source_id": row["source_id"],
            "source_family": row["source_family"],
            "pack": row["pack"],
            "status": state,
            "canonical_subject_id": canonical or row["subject_id"],
            "structural_subject": row["fingerprints"].get("structural_subject"),
            "appearance": row["fingerprints"].get("appearance"),
            "rig_hierarchy": row["fingerprints"].get("rig_hierarchy"),
            "animation_truth": row["animation_truth"],
        })
        preclassified.add(row["subject_id"])

    eligible_full = [
        row for row in all_subjects
        if row["subject_id"] not in preclassified
        and row["full_static_truth_ready"]
        and row["provenance"] in CLEAN_PROVENANCE
        and row["fingerprints"].get("structural_subject")
    ]
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible_full:
        groups[row["fingerprints"]["structural_subject"]].append(row)

    appearance_variant_groups = []
    for structural_hash, members in sorted(groups.items()):
        base = sorted([row for row in members if row["role"] == "REFERENCE_BASE139"], key=lambda row: (order[row["source_id"]], row["subject_id"]))
        post = sorted([row for row in members if row["role"] == "POST139_CANDIDATE"], key=lambda row: (order[row["source_id"]], row["subject_id"]))
        canonical = base[0] if base else (post[0] if post else members[0])
        appearances = sorted({row["fingerprints"].get("appearance") for row in members if row["fingerprints"].get("appearance")})
        if len(members) > 1 and len(appearances) > 1:
            appearance_variant_groups.append({
                "structural_subject": structural_hash,
                "canonical_subject_id": canonical["subject_id"],
                "member_subject_ids": sorted(row["subject_id"] for row in members),
                "appearance_identity_count": len(appearances),
            })
        for row in members:
            if row["role"] == "REFERENCE_BASE139":
                status = "REFERENCE_BASE139_ROW"
                canonical_id = row["subject_id"]
            elif base:
                status = "DUPLICATE_OF_BASE139_STRUCTURE"
                canonical_id = base[0]["subject_id"]
            elif row is canonical:
                if not row["animation_truth"]:
                    status = "AUXILIARY_STATIC_PRIMARY"
                elif not row["source_materialization_eligible"]:
                    status = "AUXILIARY_POLICY_PRIMARY"
                else:
                    status = "ADMIT_PRIMARY"
                canonical_id = row["subject_id"]
            else:
                status = "DUPLICATE_COLLAPSE"
                canonical_id = canonical["subject_id"]
            admissions.append({
                "subject_id": row["subject_id"],
                "source_id": row["source_id"],
                "source_family": row["source_family"],
                "pack": row["pack"],
                "status": status,
                "canonical_subject_id": canonical_id,
                "structural_subject": structural_hash,
                "appearance": row["fingerprints"].get("appearance"),
                "rig_hierarchy": row["fingerprints"].get("rig_hierarchy"),
                "animation_truth": row["animation_truth"],
            })

    admitted_ids = {row["subject_id"] for row in admissions}
    for row in all_subjects:
        if row["subject_id"] in admitted_ids:
            continue
        status = "QUARANTINE_PROVENANCE" if row["provenance"] not in CLEAN_PROVENANCE else "PARTIAL_OR_NONTRAINABLE_SOURCE_ROW"
        admissions.append({
            "subject_id": row["subject_id"],
            "source_id": row["source_id"],
            "source_family": row["source_family"],
            "pack": row["pack"],
            "status": status,
            "canonical_subject_id": row["subject_id"],
            "structural_subject": row["fingerprints"].get("structural_subject"),
            "appearance": row["fingerprints"].get("appearance"),
            "rig_hierarchy": row["fingerprints"].get("rig_hierarchy"),
            "animation_truth": row["animation_truth"],
        })
    admissions.sort(key=lambda row: row["subject_id"])
    admission_by_id = {row["subject_id"]: row for row in admissions}

    # Semantic aliases must target an admitted canonical row and may not form chains.
    aliases = dict(policy.get("alias_subject_ids", {}))
    for sid, target in aliases.items():
        target_admission = admission_by_id.get(target)
        if target in aliases:
            blockers.append({"code": "SUBJECT_POLICY_ALIAS_CHAIN_FORBIDDEN", "subject_id": sid, "canonical_subject_id": target})
        if target_admission is None or target_admission["status"] in {"EXCLUDED_BY_SEMANTIC_POLICY", "REFERENCE_VARIANT_NOT_IN_BASE139", "SEMANTIC_ALIAS_COLLAPSE"}:
            blockers.append({"code": "SUBJECT_POLICY_ALIAS_TARGET_NOT_ADMITTED", "subject_id": sid, "canonical_subject_id": target})

    lineage_rows = [row for row in all_subjects if admission_by_id.get(row["subject_id"], {}).get("status") in LINEAGE_STATUSES]
    uf = UnionFind([row["subject_id"] for row in lineage_rows])
    edges = []
    for field, relation in (("rig_hierarchy", "EXACT_RIG_HIERARCHY"), ("geometry_topology", "EXACT_CANONICAL_SURFACE")):
        buckets: defaultdict[str, list[str]] = defaultdict(list)
        for row in lineage_rows:
            value = row["fingerprints"].get(field)
            if value:
                buckets[value].append(row["subject_id"])
        for value, ids in sorted(buckets.items()):
            ids = sorted(ids)
            if len(ids) < 2:
                continue
            anchor = ids[0]
            for other in ids[1:]:
                uf.union(anchor, other)
                edges.append({"a": anchor, "b": other, "relation": relation, "fingerprint": value})
    component_members: defaultdict[str, list[str]] = defaultdict(list)
    for row in lineage_rows:
        component_members[uf.find(row["subject_id"])].append(row["subject_id"])
    components = []
    component_by_subject: dict[str, str] = {}
    for index, members in enumerate(sorted((sorted(values) for values in component_members.values()), key=lambda values: values[0]), start=1):
        component_id = f"LC{index:04d}"
        packs = sorted({f"{by_id[sid]['source_family']}::{by_id[sid]['pack']}" for sid in members})
        for sid in members:
            component_by_subject[sid] = component_id
        components.append({
            "component_id": component_id,
            "member_count": len(members),
            "members": members,
            "packs": packs,
            "cross_pack": len(packs) > 1,
        })

    publisher_by_source = {row["source_id"]: row for row in publisher_sources if row.get("status") != "QUARANTINED_NOT_ACQUIRED"}
    primary_rows: list[dict[str, Any]] = []
    auxiliary_rows: list[dict[str, Any]] = []
    for admission in admissions:
        if admission["status"] not in PRIMARY_STATUSES | AUXILIARY_STATUSES:
            continue
        row = by_id[admission["subject_id"]]
        source_inventory = publisher_by_source.get(row["source_id"])
        if source_inventory is None:
            blockers.append({"code": "MATERIALIZATION_SOURCE_INVENTORY_MISSING", "subject_id": row["subject_id"]})
            continue
        source_path = workspace / row["source_id"] / "raw" / (row["relative_source_path"] or "")
        if not source_path.is_file():
            blockers.append({"code": "MATERIALIZATION_MODEL_BYTE_MISSING", "subject_id": row["subject_id"], "path": str(source_path)})
            continue
        actual = sha256_file(source_path)
        if actual != row["source_sha256"]:
            blockers.append({"code": "MATERIALIZATION_MODEL_HASH_MISMATCH", "subject_id": row["subject_id"], "expected": row["source_sha256"], "actual": actual})
            continue
        materialization = {
            "subject_id": row["subject_id"],
            "source_id": row["source_id"],
            "source_family": row["source_family"],
            "pack": row["pack"],
            "truth_scope": "FULL_ANIMATED_TRUTH" if row["animation_truth"] else "STATIC_GEOMETRY_APPEARANCE_RIG_SKIN",
            "lineage_component_id": component_by_subject.get(row["subject_id"]),
            "workspace_relative_model": str(source_path.relative_to(workspace)),
            "model_sha256": row["source_sha256"],
            "source_closure_ledger_sha256": source_inventory["source_closure_ledger_sha256"],
            "publisher_tree_digest": source_inventory["publisher_tree_digest"],
            "structural_subject": row["fingerprints"].get("structural_subject"),
            "geometry_topology": row["fingerprints"].get("geometry_topology"),
            "rig_rest": row["fingerprints"].get("rig_rest"),
            "rig_hierarchy": row["fingerprints"].get("rig_hierarchy"),
            "skin_weights": row["fingerprints"].get("skin_weights"),
            "appearance": row["fingerprints"].get("appearance"),
            "animation_inventory": row["fingerprints"].get("animation_inventory"),
            "animation_names": row["animation_names"],
            "metrics": row["metrics"],
        }
        (primary_rows if admission["status"] == "ADMIT_PRIMARY" else auxiliary_rows).append(materialization)

    base_rows = [
        row for row in all_subjects
        if admission_by_id.get(row["subject_id"], {}).get("status") == "REFERENCE_BASE139_ROW"
        and row["full_static_truth_ready"] and row["animation_truth"]
    ]
    expected_base = sum(int(source.get("expected_subject_count", 0)) for source in registry["sources"] if source["role"] == "REFERENCE_BASE139")
    if len(base_rows) != expected_base:
        blockers.append({"code": "BASE139_GLOBAL_TRAINING_ROW_COUNT_MISMATCH", "expected": expected_base, "observed": len(base_rows)})
    base_structural = {row["fingerprints"].get("structural_subject") for row in base_rows if row["fingerprints"].get("structural_subject")}
    primary_structural = {row["structural_subject"] for row in primary_rows if row.get("structural_subject")}
    status_counts = dict(sorted(Counter(row["status"] for row in admissions).items()))

    common = {"semantic_policy_digest": policy_digest}
    publisher_payload = {
        "schema": "RealSaS.Post139.ExactPublisherInventory.v2",
        "status": "COMPLETE" if not any(blocker["code"] == "SOURCE_EVIDENCE_MISSING" for blocker in blockers) else "INCOMPLETE",
        "source_count": len(publisher_sources),
        "publisher_file_count": sum(int(row.get("publisher_file_count") or 0) for row in publisher_sources),
        "acquired_file_count": sum(int(row.get("acquired_file_count") or 0) for row in publisher_sources),
        "sources": publisher_sources,
        **common,
    }
    subject_payload = {
        "schema": "RealSaS.Post139.SubjectStructureInventory.v2",
        "subject_row_count": len(all_subjects),
        "subjects": sorted(all_subjects, key=lambda row: row["subject_id"]),
        **common,
    }
    admission_payload = {
        "schema": "RealSaS.Post139.GlobalAdmission.v2",
        "status_counts": status_counts,
        "appearance_variant_groups": appearance_variant_groups,
        "subjects": admissions,
        **common,
    }
    lineage_payload = {
        "schema": "RealSaS.Post139.LineageComponents.v2",
        "component_count": len(components),
        "cross_pack_component_count": sum(1 for row in components if row["cross_pack"]),
        "components": components,
        "edges": edges,
        **common,
    }
    roster_payload = {
        "schema": "RealSaS.Post139.FrozenRoster.v2",
        "status": "FROZEN" if not blockers else "BLOCKED",
        "base139_training_row_count": len(base_rows),
        "base139_structural_identity_count": len(base_structural),
        "post139_primary_materialization_count": len(primary_rows),
        "post139_primary_structural_identity_count": len(primary_structural),
        "expanded_full_training_row_count": len(base_rows) + len(primary_rows),
        "primary_subjects": sorted(primary_rows, key=lambda row: row["subject_id"]),
        "auxiliary_subjects": sorted(auxiliary_rows, key=lambda row: row["subject_id"]),
        "split_rule": "LINEAGE_COMPONENT_ID_IS_ATOMIC_FOR_FIT_UNSEEN_SPLITS",
        **common,
    }
    plan_payload = {
        "schema": "RealSaS.Post139.MaterializationPlan.v2",
        "status": "MATERIALIZATION_READY" if not blockers else "BLOCKED",
        "primary_subject_count": len(primary_rows),
        "auxiliary_subject_count": len(auxiliary_rows),
        "required_source_ids": sorted({row["source_id"] for row in primary_rows + auxiliary_rows}),
        "primary_subjects": sorted(primary_rows, key=lambda row: row["subject_id"]),
        "auxiliary_subjects": sorted(auxiliary_rows, key=lambda row: row["subject_id"]),
        **common,
    }
    report = {
        "schema": "RealSaS.Post139.FinalInventoryReport.v2",
        "status": "PASS_MATERIALIZATION_READY" if not blockers else "BLOCKED",
        "source_summaries": source_summaries,
        "base139_training_rows": len(base_rows),
        "base139_structural_identities": len(base_structural),
        "post139_primary_materialization_subjects": len(primary_rows),
        "post139_auxiliary_materialization_subjects": len(auxiliary_rows),
        "expanded_full_training_rows": len(base_rows) + len(primary_rows),
        "global_admission_counts": status_counts,
        "appearance_variant_group_count": len(appearance_variant_groups),
        "lineage_component_count": len(components),
        "cross_pack_lineage_component_count": sum(1 for row in components if row["cross_pack"]),
        "blocker_count": len(blockers),
        "blockers": blockers,
        "warning_count": len(warnings),
        "warnings": warnings,
        **common,
    }
    return publisher_payload, subject_payload, admission_payload, lineage_payload, roster_payload, plan_payload, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--semantic-policy")
    parser.add_argument("--allow-blockers", action="store_true")
    args = parser.parse_args()
    registry = read_json(Path(args.registry))
    if registry.get("schema") != REGISTRY_SCHEMA:
        raise SystemExit("POST139_COMPILE_REGISTRY_SCHEMA_MISMATCH")
    policy_path = Path(args.semantic_policy) if args.semantic_policy else None
    policy = load_policy(policy_path)
    payloads = compile_inventory(registry, Path(args.workspace).resolve(), policy)
    out = Path(args.out).resolve()
    names = (
        "POST139_EXACT_PUBLISHER_INVENTORY_V2.json",
        "POST139_SUBJECT_STRUCTURE_INVENTORY_V2.json",
        "POST139_GLOBAL_ADMISSION_V2.json",
        "POST139_LINEAGE_COMPONENTS_V2.json",
        "POST139_FROZEN_ROSTER_V2.json",
        "POST139_MATERIALIZATION_PLAN_V2.json",
        "POST139_FINAL_INVENTORY_REPORT_V2.json",
    )
    for name, payload in zip(names, payloads):
        write_json(out / name, payload)
    report = payloads[-1]
    print(
        "POST139_COMPILE_V2",
        report["status"],
        report["base139_training_rows"],
        report["post139_primary_materialization_subjects"],
        report["post139_auxiliary_materialization_subjects"],
        report["blocker_count"],
        report["semantic_policy_digest"],
    )
    if report["blockers"] and not args.allow_blockers:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
