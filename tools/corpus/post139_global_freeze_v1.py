"""Build a fail-closed global Post139 inventory, dedup and materialization freeze.

Workspace contract
------------------
<workspace>/<source-id>/
  ACQUISITION_RECEIPT.json
  raw/...
  inventory/RAW_FILE_LEDGER.json
  inventory/SOURCE_INVENTORY.json
  inventory/CANDIDATE_MANIFEST.json
  audit/POST139_AUDIT_INDEX.json
  audit/subjects/<subject-id>/{source_structure,technical_audit,admission}.json

The tool does not generate truth.  It proves that the source/audit inventory is
closed enough for downstream materialization and emits a frozen execution plan.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REGISTRY_SCHEMA = "RealSaS.Post139.SourceRegistry.v1"
OUTPUT_SCHEMAS = {
    "raw": "RealSaS.Post139.GlobalRawFileInventory.v1",
    "representation": "RealSaS.Post139.GlobalRepresentationInventory.v1",
    "subject": "RealSaS.Post139.GlobalSubjectStructureInventory.v1",
    "admission": "RealSaS.Post139.GlobalAdmission.v1",
    "lineage": "RealSaS.Post139.LineageComponents.v1",
    "roster": "RealSaS.Post139.FrozenRoster.v1",
    "plan": "RealSaS.Post139.MaterializationPlan.v1",
    "report": "RealSaS.Post139.FreezeReport.v1",
}
SUPPORTED_MATERIALIZATION_EXTENSIONS = {".gltf", ".glb", ".fbx"}
CLEAN_PROVENANCE = {"CLEAN_CC0_HUMAN_AUTHORED", "CLEAN_PERMISSIVE_HUMAN_AUTHORED"}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class UnionFind:
    def __init__(self, values: list[str]):
        self.parent = {value: value for value in values}
        self.rank = {value: 0 for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def load_registry(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("schema") != REGISTRY_SCHEMA:
        raise SystemExit("POST139_FREEZE_REGISTRY_SCHEMA_MISMATCH")
    return payload


def subject_source_map(audit_index: dict[str, Any]) -> dict[str, str]:
    return {row["subject_id"]: row["source"] for row in audit_index.get("subjects", [])}


def representation_by_path(inventory: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for group in inventory.get("representation_groups", []):
        audit_source = group.get("audit_source")
        if audit_source:
            result[audit_source["path"]] = group
    return result


def semantic_status(row: dict[str, Any]) -> str:
    if row["provenance"] not in CLEAN_PROVENANCE:
        return "QUARANTINE_PROVENANCE"
    if not row["full_static_truth_ready"]:
        return "AUXILIARY_PARTIAL_TRUTH"
    return "FULL_STATIC_TRUTH"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-not-ready", action="store_true")
    args = parser.parse_args()

    registry = load_registry(Path(args.registry))
    workspace = Path(args.workspace).resolve()
    out = Path(args.out).resolve()
    sources = registry["sources"]
    registry_order = {row["id"]: index for index, row in enumerate(sources)}
    source_meta = {row["id"]: row for row in sources}

    blockers: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    raw_sources = []
    representation_sources = []
    subjects: list[dict[str, Any]] = []
    source_summaries = []

    for source in sources:
        source_id = source["id"]
        root = workspace / source_id
        if source["role"] == "QUARANTINE":
            source_summaries.append({"source_id": source_id, "role": source["role"], "status": "QUARANTINED_NOT_REQUIRED"})
            continue
        receipt_path = root / "ACQUISITION_RECEIPT.json"
        inventory_path = root / "inventory" / "SOURCE_INVENTORY.json"
        ledger_path = root / "inventory" / "RAW_FILE_LEDGER.json"
        audit_index_path = root / "audit" / "POST139_AUDIT_INDEX.json"
        missing = [str(path.relative_to(workspace)) for path in (receipt_path, inventory_path, ledger_path, audit_index_path) if not path.is_file()]
        if missing:
            blockers.append({"code": "SOURCE_EVIDENCE_MISSING", "source_id": source_id, "paths": missing})
            source_summaries.append({"source_id": source_id, "role": source["role"], "status": "MISSING_EVIDENCE", "missing": missing})
            continue

        receipt = read_json(receipt_path)
        inventory = read_json(inventory_path)
        ledger = read_json(ledger_path)
        audit_index = read_json(audit_index_path)
        if receipt.get("status") != "ACQUIRED_VERIFIED":
            blockers.append({"code": "SOURCE_NOT_ACQUIRED_VERIFIED", "source_id": source_id, "status": receipt.get("status")})
        if audit_index.get("failure_count"):
            blockers.append({"code": "AUDIT_FAILURES", "source_id": source_id, "count": audit_index["failure_count"], "failures": audit_index.get("failures", [])})
        if inventory.get("raw_ledger_sha256") != ledger.get("ledger_sha256"):
            blockers.append({"code": "INVENTORY_LEDGER_HASH_MISMATCH", "source_id": source_id})

        raw_sources.append({
            "source_id": source_id, "source_family": source["source_family"], "pack": source["pack"],
            "role": source["role"], "acquisition": receipt,
            "raw_file_count": ledger["file_count"], "raw_total_bytes": ledger["total_bytes"],
            "raw_ledger_sha256": ledger["ledger_sha256"], "extension_counts": ledger["extension_counts"],
            "files": ledger["files"],
        })
        representation_sources.append({
            "source_id": source_id, "source_family": source["source_family"], "pack": source["pack"],
            "role": source["role"], "representation_group_count": inventory["representation_group_count"],
            "audit_candidate_count": inventory["audit_candidate_count"],
            "representation_groups": inventory["representation_groups"],
        })

        index_source = subject_source_map(audit_index)
        rep_source = representation_by_path(inventory)
        full_rows = 0
        full_animated_rows = 0
        full_structural_hashes = set()
        for subject_root in sorted((root / "audit" / "subjects").iterdir() if (root / "audit" / "subjects").is_dir() else []):
            if not subject_root.is_dir():
                continue
            structure_path = subject_root / "source_structure.json"
            technical_path = subject_root / "technical_audit.json"
            if not structure_path.is_file() or not technical_path.is_file():
                continue
            structure = read_json(structure_path)
            technical = read_json(technical_path)
            subject_id = subject_root.name
            relative_source = index_source.get(subject_id)
            group = rep_source.get(relative_source or "")
            capabilities = structure.get("capabilities", {})
            fingerprints = structure.get("fingerprints", {})
            full_static = bool(technical.get("full_static_truth_ready"))
            animation_truth = bool(technical.get("animation_truth"))
            if full_static:
                full_rows += 1
                if fingerprints.get("structural_subject"):
                    full_structural_hashes.add(fingerprints["structural_subject"])
            if full_static and animation_truth:
                full_animated_rows += 1
            subject_record = {
                "subject_id": subject_id, "source_id": source_id,
                "source_family": source["source_family"], "pack": source["pack"], "role": source["role"],
                "provenance": source["provenance"], "license": source.get("license"),
                "materialization_eligible_source": bool(source.get("materialization_eligible")),
                "relative_source_path": relative_source,
                "source_sha256": structure.get("source", {}).get("sha256"),
                "source_extension": structure.get("source", {}).get("extension"),
                "source_size_bytes": structure.get("source", {}).get("size_bytes"),
                "capabilities": capabilities, "metrics": structure.get("metrics", {}),
                "fingerprints": fingerprints,
                "full_static_truth_ready": full_static, "animation_truth": animation_truth,
                "technical_status": technical.get("status"),
                "semantic_status": None,
                "representation_key": group.get("representation_key") if group else None,
                "representations": group.get("representations", []) if group else [],
                "animation_names": [row.get("name") for row in structure.get("inventory", {}).get("animations", [])],
                "object_inventory": structure.get("inventory", {}).get("objects", []),
                "evidence": {
                    "source_structure_sha256": sha256_file(structure_path),
                    "technical_audit_sha256": sha256_file(technical_path),
                    "relative_evidence_root": str(subject_root.relative_to(workspace)),
                },
            }
            subject_record["semantic_status"] = semantic_status(subject_record)
            subjects.append(subject_record)

        if source["role"] == "REFERENCE_BASE139":
            expected = int(source["expected_subject_count"])
            if len(full_structural_hashes) != expected:
                blockers.append({
                    "code": "BASE139_REFERENCE_COUNT_MISMATCH", "source_id": source_id,
                    "expected_unique_full_truth": expected, "observed_unique_full_truth": len(full_structural_hashes),
                    "full_rows": full_rows, "full_animated_rows": full_animated_rows,
                })
        elif source["role"] == "POST139_CANDIDATE" and "checkpoint_candidate_subjects" in source:
            expected_rows = int(source["checkpoint_candidate_subjects"])
            comparison_count = full_rows if source.get("checkpoint_truth", {}).get("animation") == "NO" else full_animated_rows
            if comparison_count != expected_rows:
                blockers.append({
                    "code": "POST139_SEMANTIC_CANDIDATE_COUNT_REVIEW", "source_id": source_id,
                    "checkpoint_candidate_subjects": expected_rows, "observed_full_truth_rows": comparison_count,
                    "note": "Resolve authored-subject aliases/exclusions in the source registry; do not silently change denominator.",
                })
        source_summaries.append({
            "source_id": source_id, "role": source["role"], "status": "AUDITED",
            "raw_files": ledger["file_count"], "representations": inventory["representation_group_count"],
            "audit_candidates": inventory["audit_candidate_count"], "audited": audit_index.get("audited_count"),
            "audit_failures": audit_index.get("failure_count"), "full_static_rows": full_rows,
            "full_animated_rows": full_animated_rows, "unique_full_structural_hashes": len(full_structural_hashes),
        })

    # Global structural identity.  Base139 is immutable reference authority.
    full = [row for row in subjects if row["semantic_status"] == "FULL_STATIC_TRUTH" and row["fingerprints"].get("structural_subject")]
    structural_groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in full:
        structural_groups[row["fingerprints"]["structural_subject"]].append(row)
    global_admission = []
    canonical_rows = []
    for fingerprint, group in sorted(structural_groups.items()):
        base = [row for row in group if row["role"] == "REFERENCE_BASE139"]
        post = [row for row in group if row["role"] == "POST139_CANDIDATE"]
        if len(base) > 1:
            blockers.append({
                "code": "BASE139_STRUCTURAL_IDENTITY_COLLISION", "structural_subject": fingerprint,
                "subjects": [row["subject_id"] for row in base],
            })
        ordered = sorted(group, key=lambda row: (0 if row["role"] == "REFERENCE_BASE139" else 1, registry_order[row["source_id"]], row["subject_id"]))
        canonical = ordered[0]
        canonical_rows.append(canonical)
        for row in ordered:
            duplicate = row is not canonical
            if row["role"] == "REFERENCE_BASE139":
                status = "REFERENCE_BASE139"
            elif row["provenance"] not in CLEAN_PROVENANCE:
                status = "QUARANTINE_PROVENANCE"
            elif duplicate and canonical["role"] == "REFERENCE_BASE139":
                status = "DUPLICATE_OF_BASE139"
            elif duplicate:
                status = "DUPLICATE_COLLAPSE"
            elif not row["animation_truth"]:
                status = "AUXILIARY_STATIC_TRUTH"
            elif not row["materialization_eligible_source"]:
                status = "AUXILIARY_POLICY"
            else:
                status = "ADMIT_PRIMARY"
            global_admission.append({
                "subject_id": row["subject_id"], "source_id": row["source_id"],
                "source_family": row["source_family"], "pack": row["pack"],
                "status": status, "canonical_subject_id": canonical["subject_id"],
                "duplicate_of": canonical["subject_id"] if duplicate else None,
                "duplicate_scope": (
                    "BASE139" if duplicate and canonical["role"] == "REFERENCE_BASE139" else
                    "CROSS_PACK_OR_WITHIN_POST139" if duplicate else None
                ),
                "structural_subject": fingerprint,
                "appearance": row["fingerprints"].get("appearance"),
                "rig_hierarchy": row["fingerprints"].get("rig_hierarchy"),
            })
    admission_by_subject = {row["subject_id"]: row for row in global_admission}

    # Leakage components: exact rig hierarchy and exact canonical surface are hard
    # connections. Appearance is recorded separately but does not by itself join
    # components because generic shared materials would over-connect unrelated rigs.
    lineage_nodes = [row for row in subjects if admission_by_subject.get(row["subject_id"], {}).get("status") in {"REFERENCE_BASE139", "ADMIT_PRIMARY", "AUXILIARY_STATIC_TRUTH", "AUXILIARY_POLICY"}]
    uf = UnionFind([row["subject_id"] for row in lineage_nodes])
    relation_edges = []
    for field, relation in (("rig_hierarchy", "EXACT_RIG_HIERARCHY"), ("geometry_topology", "EXACT_CANONICAL_SURFACE")):
        buckets: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in lineage_nodes:
            value = row["fingerprints"].get(field)
            if value:
                buckets[value].append(row)
        for value, bucket in buckets.items():
            if len(bucket) < 2:
                continue
            anchor = bucket[0]["subject_id"]
            for other in bucket[1:]:
                uf.union(anchor, other["subject_id"])
                relation_edges.append({"a": anchor, "b": other["subject_id"], "relation": relation, "fingerprint": value})
    components: defaultdict[str, list[str]] = defaultdict(list)
    for row in lineage_nodes:
        components[uf.find(row["subject_id"])].append(row["subject_id"])
    component_rows = []
    component_id_by_subject = {}
    for index, members in enumerate(sorted((sorted(values) for values in components.values()), key=lambda values: values[0]), start=1):
        component_id = f"LC{index:04d}"
        for subject_id in members:
            component_id_by_subject[subject_id] = component_id
        packs = sorted({f"{next(row for row in lineage_nodes if row['subject_id']==sid)['source_family']}::{next(row for row in lineage_nodes if row['subject_id']==sid)['pack']}" for sid in members})
        component_rows.append({"component_id": component_id, "members": members, "member_count": len(members), "packs": packs, "cross_pack": len(packs) > 1})

    subject_by_id = {row["subject_id"]: row for row in subjects}
    primary = []
    for admission in global_admission:
        if admission["status"] != "ADMIT_PRIMARY":
            continue
        row = subject_by_id[admission["subject_id"]]
        extension = row["source_extension"]
        if extension not in SUPPORTED_MATERIALIZATION_EXTENSIONS:
            blockers.append({"code": "MATERIALIZATION_SOURCE_FORMAT_UNSUPPORTED", "subject_id": row["subject_id"], "extension": extension})
            continue
        source_path = workspace / row["source_id"] / "raw" / (row["relative_source_path"] or "")
        if not source_path.is_file():
            blockers.append({"code": "MATERIALIZATION_SOURCE_BYTE_MISSING", "subject_id": row["subject_id"], "path": str(source_path)})
            continue
        actual_hash = sha256_file(source_path)
        if actual_hash != row["source_sha256"]:
            blockers.append({"code": "MATERIALIZATION_SOURCE_HASH_MISMATCH", "subject_id": row["subject_id"], "expected": row["source_sha256"], "actual": actual_hash})
            continue
        primary.append({
            "subject_id": row["subject_id"], "source_id": row["source_id"],
            "source_family": row["source_family"], "pack": row["pack"],
            "lineage_component_id": component_id_by_subject.get(row["subject_id"]),
            "workspace_relative_source": str(source_path.relative_to(workspace)),
            "source_sha256": row["source_sha256"], "source_extension": extension,
            "structural_subject": row["fingerprints"]["structural_subject"],
            "geometry_topology": row["fingerprints"].get("geometry_topology"),
            "rig_rest": row["fingerprints"].get("rig_rest"),
            "rig_hierarchy": row["fingerprints"].get("rig_hierarchy"),
            "skin_weights": row["fingerprints"].get("skin_weights"),
            "appearance": row["fingerprints"].get("appearance"),
            "animation_inventory": row["fingerprints"].get("animation_inventory"),
            "animation_names": row["animation_names"],
            "metrics": row["metrics"],
        })
    primary.sort(key=lambda row: row["subject_id"])

    status_counts = Counter(row["status"] for row in global_admission)
    raw_payload = {"schema": OUTPUT_SCHEMAS["raw"], "sources": raw_sources}
    representation_payload = {"schema": OUTPUT_SCHEMAS["representation"], "sources": representation_sources}
    subject_payload = {"schema": OUTPUT_SCHEMAS["subject"], "subjects": sorted(subjects, key=lambda row: row["subject_id"])}
    admission_payload = {"schema": OUTPUT_SCHEMAS["admission"], "status_counts": dict(sorted(status_counts.items())), "subjects": sorted(global_admission, key=lambda row: row["subject_id"])}
    lineage_payload = {"schema": OUTPUT_SCHEMAS["lineage"], "components": component_rows, "edges": relation_edges}
    roster_payload = {
        "schema": OUTPUT_SCHEMAS["roster"], "status": "FROZEN" if not blockers else "NOT_FROZEN",
        "subject_count": len(primary), "subjects": primary,
        "split_rule": "LINEAGE_COMPONENT_ID_IS_ATOMIC_FOR_FIT_UNSEEN_SPLITS",
    }
    plan_payload = {
        "schema": OUTPUT_SCHEMAS["plan"], "status": "MATERIALIZATION_READY" if not blockers else "BLOCKED",
        "workspace_contract": "SOURCE_BYTES_PRESENT_AND_SHA256_VERIFIED",
        "subject_count": len(primary),
        "subjects": [
            {
                **row,
                "materialization_output_key": f"post139/{row['subject_id']}",
                "required_source_hash": row["source_sha256"],
            }
            for row in primary
        ],
    }
    report_payload = {
        "schema": OUTPUT_SCHEMAS["report"], "status": "PASS_MATERIALIZATION_READY" if not blockers else "BLOCKED",
        "registry_source_count": len(sources), "source_summaries": source_summaries,
        "raw_source_count": len(raw_sources), "audited_subject_count": len(subjects),
        "global_admission_counts": dict(sorted(status_counts.items())),
        "primary_materialization_subject_count": len(primary),
        "lineage_component_count": len(component_rows),
        "cross_pack_lineage_component_count": sum(1 for row in component_rows if row["cross_pack"]),
        "blocker_count": len(blockers), "blockers": blockers, "warnings": warnings,
    }

    write_json(out / "POST139_RAW_FILE_INVENTORY.json", raw_payload)
    write_json(out / "POST139_REPRESENTATION_INVENTORY.json", representation_payload)
    write_json(out / "POST139_SUBJECT_STRUCTURE_INVENTORY.json", subject_payload)
    write_json(out / "POST139_GLOBAL_ADMISSION.json", admission_payload)
    write_json(out / "POST139_LINEAGE_COMPONENTS.json", lineage_payload)
    write_json(out / "POST139_FROZEN_ROSTER.json", roster_payload)
    write_json(out / "POST139_MATERIALIZATION_PLAN.json", plan_payload)
    write_json(out / "POST139_FREEZE_REPORT.json", report_payload)
    print("POST139_GLOBAL_FREEZE", report_payload["status"], len(primary), len(blockers))
    if blockers and not args.allow_not_ready:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
