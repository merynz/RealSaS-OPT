"""Finalize Post139 from immutable source-workspace artifacts.

V2 deliberately separates four concepts that were previously easy to conflate:

1. publisher inventory: every file/path/representation exposed by the author;
2. training rows: immutable Base139 rows plus admitted Post139 rows;
3. structural identity: geometry/topology + rest rig + spatial skin identity;
4. materialization source closure: exact acquired bytes and their ledger digest.

Base139 remains 139 immutable training rows even when two rows share the same
mechanical structure but differ in appearance (for example appearance variants).
Post139 structural duplicates do not inflate the new-subject denominator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

REGISTRY_SCHEMA = "RealSaS.Post139.SourceRegistry.v1"
CLEAN_PROVENANCE = {"CLEAN_CC0_HUMAN_AUTHORED", "CLEAN_PERMISSIVE_HUMAN_AUTHORED"}
FORMAT_LANE_PARTS = {
    "fbx", "gltf", "glb", "obj", "blend", "blends", "blender",
    "models-fbx", "models-gltf", "models-glb", "models-obj", "models-blend",
}
REPRESENTATION_VARIANT_PARTS = {"flat", "textured"}


def cb(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(cb(value)).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def slug(value: str) -> str:
    import re
    result = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return result or "item"


def authored_item_key(path: str) -> tuple[str, list[str]]:
    pure = PurePosixPath(path)
    tags: list[str] = []
    kept = []
    for part in pure.parts[:-1]:
        token = slug(part)
        if token in FORMAT_LANE_PARTS:
            continue
        if token in REPRESENTATION_VARIANT_PARTS:
            tags.append(token.upper())
            continue
        kept.append(token)
    kept.append(slug(pure.stem))
    return "/".join(kept), sorted(set(tags))


class UnionFind:
    def __init__(self, values: list[str]):
        self.parent = {value: value for value in values}
        self.rank = {value: 0 for value in values}

    def find(self, value: str) -> str:
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def source_evidence(workspace: Path, source: dict[str, Any]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    source_id = source["id"]
    root = workspace / source_id
    required = {
        "receipt": root / "ACQUISITION_RECEIPT.json",
        "publisher": root / "PUBLISHER_TREE.json",
        "ledger": root / "inventory" / "RAW_FILE_LEDGER.json",
        "inventory": root / "inventory" / "SOURCE_INVENTORY.json",
        "audit": root / "audit" / "POST139_AUDIT_INDEX.json",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        return None, [{"code": "SOURCE_EVIDENCE_MISSING", "source_id": source_id, "missing": missing}]
    evidence = {name: read_json(path) for name, path in required.items()}
    blockers = []
    if evidence["receipt"].get("status") != "ACQUIRED_VERIFIED":
        blockers.append({"code": "ACQUISITION_NOT_VERIFIED", "source_id": source_id, "status": evidence["receipt"].get("status")})
    if evidence["audit"].get("failure_count"):
        blockers.append({"code": "AUDIT_FAILURES", "source_id": source_id, "failures": evidence["audit"].get("failures", [])})
    if evidence["inventory"].get("raw_ledger_sha256") != evidence["ledger"].get("ledger_sha256"):
        blockers.append({"code": "LEDGER_BINDING_MISMATCH", "source_id": source_id})
    return evidence, blockers


def subject_rows(workspace: Path, source: dict[str, Any], evidence: dict[str, Any]) -> list[dict[str, Any]]:
    root = workspace / source["id"]
    audit_index = evidence["audit"]
    source_by_id = {row["subject_id"]: row["source"] for row in audit_index.get("subjects", [])}
    inventory_groups = {
        row["audit_source"]["path"]: row
        for row in evidence["inventory"].get("representation_groups", [])
        if row.get("audit_source")
    }
    rows = []
    subjects_root = root / "audit" / "subjects"
    for subject_dir in sorted(subjects_root.iterdir() if subjects_root.is_dir() else []):
        if not subject_dir.is_dir():
            continue
        structure_path = subject_dir / "source_structure.json"
        technical_path = subject_dir / "technical_audit.json"
        admission_path = subject_dir / "admission.json"
        if not (structure_path.is_file() and technical_path.is_file() and admission_path.is_file()):
            continue
        structure = read_json(structure_path)
        technical = read_json(technical_path)
        admission = read_json(admission_path)
        relative = source_by_id.get(subject_dir.name)
        group = inventory_groups.get(relative or "")
        rows.append({
            "subject_id": subject_dir.name,
            "source_id": source["id"],
            "source_family": source["source_family"],
            "pack": source["pack"],
            "role": source["role"],
            "provenance": source["provenance"],
            "license": source.get("license"),
            "candidate_policy": source.get("candidate_policy"),
            "source_materialization_eligible": bool(source.get("materialization_eligible")),
            "relative_source_path": relative,
            "source_sha256": structure.get("source", {}).get("sha256"),
            "source_size_bytes": structure.get("source", {}).get("size_bytes"),
            "source_extension": structure.get("source", {}).get("extension"),
            "capabilities": structure.get("capabilities", {}),
            "metrics": structure.get("metrics", {}),
            "fingerprints": structure.get("fingerprints", {}),
            "animation_names": [row.get("name") for row in structure.get("inventory", {}).get("animations", [])],
            "objects": structure.get("inventory", {}).get("objects", []),
            "technical_status": technical.get("status"),
            "full_static_truth_ready": bool(technical.get("full_static_truth_ready")),
            "animation_truth": bool(technical.get("animation_truth")),
            "within_pack_admission": admission.get("status"),
            "within_pack_duplicate_of": admission.get("duplicate_of"),
            "representation_key": group.get("representation_key") if group else None,
            "representations": group.get("representations", []) if group else [],
            "evidence_hashes": {
                "source_structure": sha256_file(structure_path),
                "technical_audit": sha256_file(technical_path),
                "admission": sha256_file(admission_path),
            },
        })
    return rows


def publisher_inventory_row(source: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    tree = evidence["publisher"]
    ledger = evidence["ledger"]
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    variant_tags: defaultdict[str, set[str]] = defaultdict(set)
    for row in tree.get("files", []):
        extension = row.get("extension", "")
        if extension not in {".gltf", ".glb", ".fbx", ".blend", ".obj"}:
            continue
        key, tags = authored_item_key(row["path"])
        grouped[key].append(row)
        variant_tags[key].update(tags)
    authored_groups = []
    for key, rows in sorted(grouped.items()):
        authored_groups.append({
            "authored_item_key": key,
            "representation_variant_tags": sorted(variant_tags[key]),
            "representation_count": len(rows),
            "format_counts": dict(sorted(Counter(row.get("extension") for row in rows).items())),
            "representations": sorted(rows, key=lambda row: row["path"]),
        })
    return {
        "source_id": source["id"],
        "source_family": source["source_family"],
        "pack": source["pack"],
        "role": source["role"],
        "provenance": source["provenance"],
        "license": source.get("license"),
        "acquisition": evidence["receipt"],
        "publisher_file_count": tree.get("file_count"),
        "publisher_extension_counts": tree.get("extension_counts", {}),
        "publisher_tree_digest": tree.get("tree_digest"),
        "publisher_files": tree.get("files", []),
        "authored_item_group_count": len(authored_groups),
        "authored_item_groups": authored_groups,
        "acquired_file_count": ledger.get("file_count"),
        "acquired_total_bytes": ledger.get("total_bytes"),
        "acquired_extension_counts": ledger.get("extension_counts", {}),
        "source_closure_ledger_sha256": ledger.get("ledger_sha256"),
        "acquired_files": ledger.get("files", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--semantic-overrides")
    parser.add_argument("--allow-blockers", action="store_true")
    args = parser.parse_args()

    registry = read_json(Path(args.registry))
    if registry.get("schema") != REGISTRY_SCHEMA:
        raise SystemExit("POST139_FINALIZE_REGISTRY_SCHEMA_MISMATCH")
    overrides = read_json(Path(args.semantic_overrides)) if args.semantic_overrides else {"exclude_subject_ids": [], "alias_subject_ids": {}}
    excluded = set(overrides.get("exclude_subject_ids", []))
    aliases = dict(overrides.get("alias_subject_ids", {}))
    workspace = Path(args.workspace).resolve()
    out = Path(args.out).resolve()
    blockers: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    all_subjects: list[dict[str, Any]] = []
    publisher_sources = []
    source_summaries = []

    order = {row["id"]: i for i, row in enumerate(registry["sources"])}
    for source in registry["sources"]:
        if source["role"] == "QUARANTINE":
            publisher_sources.append({
                "source_id": source["id"], "source_family": source["source_family"], "pack": source["pack"],
                "role": source["role"], "provenance": source["provenance"], "license": source.get("license"),
                "status": "QUARANTINED_NOT_ACQUIRED",
            })
            source_summaries.append({"source_id": source["id"], "role": source["role"], "status": "QUARANTINED"})
            continue
        evidence, source_blockers = source_evidence(workspace, source)
        blockers.extend(source_blockers)
        if evidence is None:
            source_summaries.append({"source_id": source["id"], "role": source["role"], "status": "MISSING_EVIDENCE"})
            continue
        publisher_sources.append(publisher_inventory_row(source, evidence))
        rows = subject_rows(workspace, source, evidence)
        all_subjects.extend(rows)
        full_static = [row for row in rows if row["full_static_truth_ready"]]
        full_animated = [row for row in full_static if row["animation_truth"]]
        structural = {row["fingerprints"].get("structural_subject") for row in full_static if row["fingerprints"].get("structural_subject")}
        summary = {
            "source_id": source["id"], "source_family": source["source_family"], "pack": source["pack"],
            "role": source["role"], "status": "AUDITED",
            "publisher_files": evidence["publisher"].get("file_count"),
            "acquired_files": evidence["ledger"].get("file_count"),
            "audited_rows": len(rows), "full_static_rows": len(full_static),
            "full_animated_rows": len(full_animated), "structural_identity_count": len(structural),
            "within_pack_statuses": dict(sorted(Counter(row["within_pack_admission"] for row in rows).items())),
            "source_closure_ledger_sha256": evidence["ledger"].get("ledger_sha256"),
        }
        if source["role"] == "REFERENCE_BASE139":
            expected = int(source["expected_subject_count"])
            if len(full_animated) != expected:
                blockers.append({
                    "code": "BASE139_TRAINING_ROW_COUNT_MISMATCH", "source_id": source["id"],
                    "expected": expected, "observed": len(full_animated),
                })
            if len(structural) < len(full_animated):
                warnings.append({
                    "code": "BASE139_APPEARANCE_OR_OTHER_NONSTRUCTURAL_VARIANTS",
                    "source_id": source["id"], "training_rows": len(full_animated),
                    "structural_identities": len(structural), "delta": len(full_animated) - len(structural),
                })
        elif source["role"] == "POST139_CANDIDATE":
            expected = source.get("checkpoint_candidate_subjects")
            if expected is not None:
                use_static = source.get("checkpoint_truth", {}).get("animation") == "NO"
                observed_rows = full_static if use_static else full_animated
                semantic_rows = [row for row in observed_rows if row["subject_id"] not in excluded and row["subject_id"] not in aliases]
                if len(semantic_rows) != int(expected):
                    blockers.append({
                        "code": "POST139_SEMANTIC_ROW_COUNT_REVIEW", "source_id": source["id"],
                        "checkpoint_expected": int(expected), "observed_before_overrides": len(observed_rows),
                        "observed_after_overrides": len(semantic_rows),
                        "full_row_subject_ids": [row["subject_id"] for row in observed_rows],
                    })
        source_summaries.append(summary)

    by_id = {row["subject_id"]: row for row in all_subjects}
    # Validate explicit aliases only after all evidence exists.
    for subject_id, canonical_id in aliases.items():
        if subject_id not in by_id or canonical_id not in by_id:
            blockers.append({"code": "SEMANTIC_ALIAS_UNKNOWN_SUBJECT", "subject_id": subject_id, "canonical_id": canonical_id})

    full_static = [row for row in all_subjects if row["full_static_truth_ready"] and row["provenance"] in CLEAN_PROVENANCE]
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in full_static:
        fingerprint = row["fingerprints"].get("structural_subject")
        if fingerprint:
            groups[fingerprint].append(row)

    admissions = []
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
                "member_subject_ids": [row["subject_id"] for row in members],
                "appearance_identity_count": len(appearances),
            })
        for row in members:
            if row["subject_id"] in excluded:
                status = "EXCLUDED_BY_SEMANTIC_POLICY"
                canonical_id = row["subject_id"]
            elif row["subject_id"] in aliases:
                status = "SEMANTIC_ALIAS_COLLAPSE"
                canonical_id = aliases[row["subject_id"]]
            elif row["role"] == "REFERENCE_BASE139":
                # Base139 rows are immutable even when multiple appearances share structure.
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
                "subject_id": row["subject_id"], "source_id": row["source_id"],
                "source_family": row["source_family"], "pack": row["pack"],
                "status": status, "canonical_subject_id": canonical_id,
                "structural_subject": structural_hash,
                "appearance": row["fingerprints"].get("appearance"),
                "rig_hierarchy": row["fingerprints"].get("rig_hierarchy"),
                "animation_truth": row["animation_truth"],
            })

    # Subjects without full static truth are still explicitly inventoried.
    admitted_ids = {row["subject_id"] for row in admissions}
    for row in all_subjects:
        if row["subject_id"] in admitted_ids:
            continue
        admissions.append({
            "subject_id": row["subject_id"], "source_id": row["source_id"],
            "source_family": row["source_family"], "pack": row["pack"],
            "status": "PARTIAL_OR_NONTRAINABLE_SOURCE_ROW",
            "canonical_subject_id": row["subject_id"],
            "structural_subject": row["fingerprints"].get("structural_subject"),
            "appearance": row["fingerprints"].get("appearance"),
            "rig_hierarchy": row["fingerprints"].get("rig_hierarchy"),
            "animation_truth": row["animation_truth"],
        })
    admission_by_id = {row["subject_id"]: row for row in admissions}

    # Leakage graph over immutable Base rows + primary Post rows. Exact hierarchy
    # or exact surface identity is sufficient to bind rows into one split component.
    lineage_rows = [
        row for row in all_subjects
        if admission_by_id.get(row["subject_id"], {}).get("status") in {
            "REFERENCE_BASE139_ROW", "ADMIT_PRIMARY", "AUXILIARY_STATIC_PRIMARY", "AUXILIARY_POLICY_PRIMARY"
        }
    ]
    uf = UnionFind([row["subject_id"] for row in lineage_rows])
    edges = []
    for field, relation in (("rig_hierarchy", "EXACT_RIG_HIERARCHY"), ("geometry_topology", "EXACT_CANONICAL_SURFACE")):
        buckets: defaultdict[str, list[str]] = defaultdict(list)
        for row in lineage_rows:
            value = row["fingerprints"].get(field)
            if value:
                buckets[value].append(row["subject_id"])
        for value, ids in buckets.items():
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
    component_by_subject = {}
    for index, members in enumerate(sorted((sorted(values) for values in component_members.values()), key=lambda values: values[0]), start=1):
        component_id = f"LC{index:04d}"
        packs = sorted({f"{by_id[sid]['source_family']}::{by_id[sid]['pack']}" for sid in members})
        for sid in members:
            component_by_subject[sid] = component_id
        components.append({
            "component_id": component_id, "member_count": len(members), "members": members,
            "packs": packs, "cross_pack": len(packs) > 1,
        })

    publisher_by_source = {row["source_id"]: row for row in publisher_sources if row.get("status") != "QUARANTINED_NOT_ACQUIRED"}
    primary_rows = []
    auxiliary_rows = []
    for admission in admissions:
        if admission["status"] not in {"ADMIT_PRIMARY", "AUXILIARY_STATIC_PRIMARY", "AUXILIARY_POLICY_PRIMARY"}:
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
            "subject_id": row["subject_id"], "source_id": row["source_id"],
            "source_family": row["source_family"], "pack": row["pack"],
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

    base_rows = [row for row in all_subjects if row["role"] == "REFERENCE_BASE139" and row["full_static_truth_ready"] and row["animation_truth"]]
    base_structural = {row["fingerprints"].get("structural_subject") for row in base_rows if row["fingerprints"].get("structural_subject")}
    primary_structural = {row["structural_subject"] for row in primary_rows if row.get("structural_subject")}
    status_counts = dict(sorted(Counter(row["status"] for row in admissions).items()))

    publisher_payload = {
        "schema": "RealSaS.Post139.ExactPublisherInventory.v2",
        "status": "COMPLETE" if not any(blocker["code"] == "SOURCE_EVIDENCE_MISSING" for blocker in blockers) else "INCOMPLETE",
        "source_count": len(publisher_sources),
        "publisher_file_count": sum(int(row.get("publisher_file_count") or 0) for row in publisher_sources),
        "acquired_file_count": sum(int(row.get("acquired_file_count") or 0) for row in publisher_sources),
        "sources": publisher_sources,
    }
    subject_payload = {
        "schema": "RealSaS.Post139.SubjectStructureInventory.v2",
        "subject_row_count": len(all_subjects),
        "subjects": sorted(all_subjects, key=lambda row: row["subject_id"]),
    }
    admission_payload = {
        "schema": "RealSaS.Post139.GlobalAdmission.v2",
        "status_counts": status_counts,
        "appearance_variant_groups": appearance_variant_groups,
        "subjects": sorted(admissions, key=lambda row: row["subject_id"]),
    }
    lineage_payload = {
        "schema": "RealSaS.Post139.LineageComponents.v2",
        "component_count": len(components),
        "cross_pack_component_count": sum(1 for row in components if row["cross_pack"]),
        "components": components,
        "edges": edges,
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
    }
    plan_payload = {
        "schema": "RealSaS.Post139.MaterializationPlan.v2",
        "status": "MATERIALIZATION_READY" if not blockers else "BLOCKED",
        "primary_subject_count": len(primary_rows),
        "auxiliary_subject_count": len(auxiliary_rows),
        "required_source_ids": sorted({row["source_id"] for row in primary_rows + auxiliary_rows}),
        "primary_subjects": sorted(primary_rows, key=lambda row: row["subject_id"]),
        "auxiliary_subjects": sorted(auxiliary_rows, key=lambda row: row["subject_id"]),
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
    }

    for name, payload in (
        ("POST139_EXACT_PUBLISHER_INVENTORY_V2.json", publisher_payload),
        ("POST139_SUBJECT_STRUCTURE_INVENTORY_V2.json", subject_payload),
        ("POST139_GLOBAL_ADMISSION_V2.json", admission_payload),
        ("POST139_LINEAGE_COMPONENTS_V2.json", lineage_payload),
        ("POST139_FROZEN_ROSTER_V2.json", roster_payload),
        ("POST139_MATERIALIZATION_PLAN_V2.json", plan_payload),
        ("POST139_FINAL_INVENTORY_REPORT_V2.json", report),
    ):
        write_json(out / name, payload)
    print("POST139_FINALIZE_V2", report["status"], len(base_rows), len(primary_rows), len(auxiliary_rows), len(blockers))
    if blockers and not args.allow_blockers:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
