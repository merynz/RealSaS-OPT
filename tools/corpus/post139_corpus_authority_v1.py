"""Seal and verify the canonical Post139 corpus authority.

The V2 finalizer intentionally emits reviewable staging ledgers.  This tool is
where those ledgers become a single deterministic consumer authority.  It:

* verifies every acquired source byte against its RAW_FILE_LEDGER;
* validates publisher, inventory, audit and per-subject evidence bindings;
* strips runner-local paths from semantic evidence;
* emits stable source/identity/admission/lineage/roster/materialization ledgers;
* binds every ledger under one hash-chained CORPUS_AUTHORITY root;
* verifies the same bundle later without trusting the machine that produced it.

The authority never creates truth.  It only seals already-audited evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

REGISTRY_SCHEMA = "RealSaS.Post139.SourceRegistry.v1"
FINAL_SCHEMAS = {
    "publisher": "RealSaS.Post139.ExactPublisherInventory.v2",
    "subjects": "RealSaS.Post139.SubjectStructureInventory.v2",
    "admission": "RealSaS.Post139.GlobalAdmission.v2",
    "lineage": "RealSaS.Post139.LineageComponents.v2",
    "roster": "RealSaS.Post139.FrozenRoster.v2",
    "plan": "RealSaS.Post139.MaterializationPlan.v2",
    "report": "RealSaS.Post139.FinalInventoryReport.v2",
}
AUTHORITY_SCHEMAS = {
    "source": "RealSaS.Post139.SourceEvidenceLedger.v1",
    "identity": "RealSaS.Post139.SubjectIdentityLedger.v1",
    "admission": "RealSaS.Post139.AdmissionLedger.v1",
    "lineage": "RealSaS.Post139.LineageLedger.v1",
    "roster": "RealSaS.Post139.TrainingRosterLedger.v1",
    "materialization": "RealSaS.Post139.MaterializationLedger.v1",
    "root": "RealSaS.Post139.CorpusAuthority.v1",
}
AUTHORITY_FILENAMES = {
    "source": "POST139_SOURCE_EVIDENCE_LEDGER_V1.json",
    "identity": "POST139_SUBJECT_IDENTITY_LEDGER_V1.json",
    "admission": "POST139_ADMISSION_LEDGER_V1.json",
    "lineage": "POST139_LINEAGE_LEDGER_V1.json",
    "roster": "POST139_TRAINING_ROSTER_LEDGER_V1.json",
    "materialization": "POST139_MATERIALIZATION_LEDGER_V1.json",
}
FINAL_FILENAMES = {
    "publisher": "POST139_EXACT_PUBLISHER_INVENTORY_V2.json",
    "subjects": "POST139_SUBJECT_STRUCTURE_INVENTORY_V2.json",
    "admission": "POST139_GLOBAL_ADMISSION_V2.json",
    "lineage": "POST139_LINEAGE_COMPONENTS_V2.json",
    "roster": "POST139_FROZEN_ROSTER_V2.json",
    "plan": "POST139_MATERIALIZATION_PLAN_V2.json",
    "report": "POST139_FINAL_INVENTORY_REPORT_V2.json",
}
RUNTIME_KEYS = {"raw_root", "source_root"}
PRIMARY_STATUSES = {"ADMIT_PRIMARY"}
AUXILIARY_STATUSES = {"AUXILIARY_STATIC_PRIMARY", "AUXILIARY_POLICY_PRIMARY"}
LINEAGE_NODE_STATUSES = {"REFERENCE_BASE139_ROW"} | PRIMARY_STATUSES | AUXILIARY_STATUSES


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


def without_runtime_paths(value: Any) -> Any:
    """Return a canonical semantic projection with machine-local paths removed."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in RUNTIME_KEYS:
                continue
            if key == "path" and set(value).issuperset({"name", "extension", "sha256", "size_bytes"}):
                # source_structure.source.path is an absolute audit-time path.
                continue
            result[key] = without_runtime_paths(item)
        return result
    if isinstance(value, list):
        return [without_runtime_paths(item) for item in value]
    return value


def stable_evidence_hash(payload: Any) -> str:
    return digest(without_runtime_paths(payload))


def assert_schema(payload: dict[str, Any], expected: str, label: str) -> None:
    if payload.get("schema") != expected:
        raise ValueError(f"POST139_AUTHORITY_SCHEMA_MISMATCH:{label}:{payload.get('schema')}:{expected}")


def unique_by(rows: Iterable[dict[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"POST139_AUTHORITY_KEY_MISSING:{label}:{key}")
        if value in result:
            raise ValueError(f"POST139_AUTHORITY_DUPLICATE_KEY:{label}:{key}:{value}")
        result[value] = row
    return result


def publisher_tree_digest(tree: dict[str, Any]) -> str:
    rows = tree.get("files", [])
    if any("file_id" in row for row in rows):
        projection = [
            {"file_id": row.get("file_id"), "path": row.get("path"), "extension": row.get("extension")}
            for row in rows
        ]
    else:
        projection = rows
    return digest(projection)


def verify_raw_ledger(source_id: str, raw_root: Path, ledger: dict[str, Any]) -> dict[str, Any]:
    assert_schema(ledger, "RealSaS.Post139.RawFileLedger.v1", f"raw-ledger:{source_id}")
    if ledger.get("source_id") != source_id:
        raise ValueError(f"POST139_AUTHORITY_SOURCE_ID_MISMATCH:ledger:{source_id}:{ledger.get('source_id')}")
    rows = ledger.get("files", [])
    expected_digest = digest(rows)
    if ledger.get("ledger_sha256") != expected_digest:
        raise ValueError(f"POST139_AUTHORITY_RAW_LEDGER_DIGEST_MISMATCH:{source_id}")
    if int(ledger.get("file_count", -1)) != len(rows):
        raise ValueError(f"POST139_AUTHORITY_RAW_LEDGER_COUNT_MISMATCH:{source_id}")
    total_bytes = sum(int(row["size_bytes"]) for row in rows)
    if int(ledger.get("total_bytes", -1)) != total_bytes:
        raise ValueError(f"POST139_AUTHORITY_RAW_LEDGER_BYTES_MISMATCH:{source_id}")
    counts = dict(sorted(Counter(row.get("extension") for row in rows).items()))
    if ledger.get("extension_counts") != counts:
        raise ValueError(f"POST139_AUTHORITY_RAW_LEDGER_EXTENSION_COUNTS_MISMATCH:{source_id}")
    for row in rows:
        path = raw_root / row["path"]
        if not path.is_file():
            raise ValueError(f"POST139_AUTHORITY_RAW_FILE_MISSING:{source_id}:{row['path']}")
        if path.stat().st_size != int(row["size_bytes"]):
            raise ValueError(f"POST139_AUTHORITY_RAW_FILE_SIZE_MISMATCH:{source_id}:{row['path']}")
        if sha256_file(path) != row["sha256"]:
            raise ValueError(f"POST139_AUTHORITY_RAW_FILE_HASH_MISMATCH:{source_id}:{row['path']}")
    return {
        "file_count": len(rows),
        "total_bytes": total_bytes,
        "extension_counts": counts,
        "raw_ledger_sha256": expected_digest,
        "files": rows,
    }


def load_final(final_dir: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for key, filename in FINAL_FILENAMES.items():
        path = final_dir / filename
        if not path.is_file():
            raise ValueError(f"POST139_AUTHORITY_FINAL_LEDGER_MISSING:{filename}")
        payload = read_json(path)
        assert_schema(payload, FINAL_SCHEMAS[key], f"final:{key}")
        result[key] = payload
    if result["report"].get("status") != "PASS_MATERIALIZATION_READY":
        raise ValueError(f"POST139_AUTHORITY_FINALIZER_NOT_READY:{result['report'].get('status')}")
    if result["roster"].get("status") != "FROZEN":
        raise ValueError(f"POST139_AUTHORITY_ROSTER_NOT_FROZEN:{result['roster'].get('status')}")
    if result["plan"].get("status") != "MATERIALIZATION_READY":
        raise ValueError(f"POST139_AUTHORITY_PLAN_NOT_READY:{result['plan'].get('status')}")
    return result


def verify_source_workspace(registry: dict[str, Any], workspace: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]], list[dict[str, Any]]]:
    source_rows: list[dict[str, Any]] = []
    stable_subject_hashes: dict[str, dict[str, str]] = {}
    warnings: list[dict[str, Any]] = []
    for source in registry.get("sources", []):
        source_id = source["id"]
        if source.get("role") == "QUARANTINE":
            source_rows.append({
                "source_id": source_id,
                "source_family": source.get("source_family"),
                "pack": source.get("pack"),
                "role": source.get("role"),
                "status": "QUARANTINED_NOT_ACQUIRED",
            })
            continue
        root = workspace / source_id
        required = {
            "receipt": root / "ACQUISITION_RECEIPT.json",
            "publisher": root / "PUBLISHER_TREE.json",
            "ledger": root / "inventory" / "RAW_FILE_LEDGER.json",
            "inventory": root / "inventory" / "SOURCE_INVENTORY.json",
            "manifest": root / "inventory" / "CANDIDATE_MANIFEST.json",
            "audit": root / "audit" / "POST139_AUDIT_INDEX.json",
        }
        missing = [name for name, path in required.items() if not path.is_file()]
        if missing:
            raise ValueError(f"POST139_AUTHORITY_SOURCE_EVIDENCE_MISSING:{source_id}:{','.join(missing)}")
        receipt = read_json(required["receipt"])
        publisher = read_json(required["publisher"])
        ledger = read_json(required["ledger"])
        inventory = read_json(required["inventory"])
        manifest = read_json(required["manifest"])
        audit = read_json(required["audit"])
        assert_schema(receipt, "RealSaS.Post139.AcquisitionReceipt.v1", f"receipt:{source_id}")
        assert_schema(publisher, "RealSaS.Post139.PublisherTree.v1", f"publisher:{source_id}")
        assert_schema(inventory, "RealSaS.Post139.SourceInventory.v1", f"inventory:{source_id}")
        assert_schema(manifest, "RealSaS.Post139.CandidateManifest.v1", f"manifest:{source_id}")
        assert_schema(audit, "RealSaS.Post139.AuditIndex.v1", f"audit:{source_id}")
        for label, payload in (("receipt", receipt), ("publisher", publisher), ("inventory", inventory), ("manifest", manifest)):
            if payload.get("source_id") != source_id:
                raise ValueError(f"POST139_AUTHORITY_SOURCE_ID_MISMATCH:{label}:{source_id}:{payload.get('source_id')}")
        if receipt.get("status") != "ACQUIRED_VERIFIED":
            raise ValueError(f"POST139_AUTHORITY_ACQUISITION_NOT_VERIFIED:{source_id}:{receipt.get('status')}")
        if int(audit.get("failure_count", 0)):
            raise ValueError(f"POST139_AUTHORITY_AUDIT_FAILURES:{source_id}:{audit.get('failure_count')}")
        raw = verify_raw_ledger(source_id, root / "raw", ledger)
        if inventory.get("raw_ledger_sha256") != raw["raw_ledger_sha256"]:
            raise ValueError(f"POST139_AUTHORITY_INVENTORY_LEDGER_BINDING_MISMATCH:{source_id}")
        actual_publisher_digest = publisher_tree_digest(publisher)
        if publisher.get("tree_digest") != actual_publisher_digest:
            raise ValueError(f"POST139_AUTHORITY_PUBLISHER_TREE_DIGEST_MISMATCH:{source_id}")
        if inventory.get("publisher_tree_digest") not in {None, actual_publisher_digest}:
            raise ValueError(f"POST139_AUTHORITY_INVENTORY_PUBLISHER_BINDING_MISMATCH:{source_id}")

        # Historical V1 manifests hashed SOURCE_INVENTORY before source metadata was
        # appended.  The manifest is staging evidence, not consumer authority; keep
        # the mismatch visible but rebuild a stable semantic digest below.
        declared_inventory_sha = manifest.get("inventory_sha256")
        current_inventory_sha = digest(inventory)
        if declared_inventory_sha != current_inventory_sha:
            warnings.append({
                "code": "LEGACY_CANDIDATE_MANIFEST_INVENTORY_DIGEST",
                "source_id": source_id,
            })

        audit_subject_by_id = unique_by(audit.get("subjects", []), "subject_id", f"audit-index:{source_id}")
        subject_evidence = []
        subjects_root = root / "audit" / "subjects"
        for subject_id, index_row in sorted(audit_subject_by_id.items()):
            subject_root = subjects_root / subject_id
            paths = {
                "structure": subject_root / "source_structure.json",
                "technical": subject_root / "technical_audit.json",
                "admission": subject_root / "admission.json",
            }
            if index_row.get("status") == "AUDIT_FAILED":
                raise ValueError(f"POST139_AUTHORITY_AUDIT_FAILED_SUBJECT:{source_id}:{subject_id}")
            if any(not path.is_file() for path in paths.values()):
                raise ValueError(f"POST139_AUTHORITY_SUBJECT_EVIDENCE_MISSING:{source_id}:{subject_id}")
            structure = read_json(paths["structure"])
            technical = read_json(paths["technical"])
            admission = read_json(paths["admission"])
            assert_schema(structure, "RealSaS.Post139.SourceStructure.v1", f"structure:{subject_id}")
            assert_schema(technical, "RealSaS.Post139.TechnicalAudit.v1", f"technical:{subject_id}")
            assert_schema(admission, "RealSaS.Post139.Admission.v1", f"admission:{subject_id}")
            if admission.get("subject_id") != subject_id:
                raise ValueError(f"POST139_AUTHORITY_SUBJECT_ID_MISMATCH:{source_id}:{subject_id}")
            relative_source = index_row.get("source")
            if not relative_source:
                raise ValueError(f"POST139_AUTHORITY_SUBJECT_SOURCE_MISSING:{source_id}:{subject_id}")
            model_path = root / "raw" / relative_source
            source_meta = structure.get("source", {})
            if not model_path.is_file() or sha256_file(model_path) != source_meta.get("sha256"):
                raise ValueError(f"POST139_AUTHORITY_SUBJECT_MODEL_BINDING_MISMATCH:{source_id}:{subject_id}")
            hashes = {
                "source_structure_semantic_sha256": stable_evidence_hash(structure),
                "technical_audit_semantic_sha256": stable_evidence_hash(technical),
                "admission_semantic_sha256": stable_evidence_hash(admission),
            }
            hashes["subject_evidence_digest"] = digest(hashes)
            stable_subject_hashes[subject_id] = hashes
            subject_evidence.append({
                "subject_id": subject_id,
                "relative_source_path": relative_source,
                "model_sha256": source_meta.get("sha256"),
                **hashes,
            })

        stable_receipt = without_runtime_paths(receipt)
        stable_publisher = without_runtime_paths(publisher)
        stable_inventory = without_runtime_paths(inventory)
        stable_manifest = without_runtime_paths({k: v for k, v in manifest.items() if k != "inventory_sha256"})
        stable_audit = without_runtime_paths(audit)
        evidence_hashes = {
            "acquisition_receipt": digest(stable_receipt),
            "publisher_tree": digest(stable_publisher),
            "raw_file_ledger": raw["raw_ledger_sha256"],
            "source_inventory": digest(stable_inventory),
            "candidate_manifest": digest(stable_manifest),
            "audit_index": digest(stable_audit),
        }
        evidence_hashes["source_evidence_digest"] = digest(evidence_hashes)
        source_rows.append({
            "source_id": source_id,
            "source_family": source.get("source_family"),
            "pack": source.get("pack"),
            "role": source.get("role"),
            "provenance": source.get("provenance"),
            "license": source.get("license"),
            "materialization_eligible": bool(source.get("materialization_eligible")),
            "status": "VERIFIED",
            "publisher_tree_digest": actual_publisher_digest,
            "publisher_file_count": publisher.get("file_count"),
            "raw_file_count": raw["file_count"],
            "raw_total_bytes": raw["total_bytes"],
            "raw_extension_counts": raw["extension_counts"],
            "raw_ledger_sha256": raw["raw_ledger_sha256"],
            "raw_files": raw["files"],
            "acquisition": stable_receipt,
            "publisher_tree": stable_publisher,
            "subject_evidence": subject_evidence,
            "evidence_hashes": evidence_hashes,
        })
    return source_rows, stable_subject_hashes, warnings


def normalize_identity_ledger(final_subjects: dict[str, Any], stable_subject_hashes: dict[str, dict[str, str]]) -> dict[str, Any]:
    rows = []
    seen = set()
    for source_row in final_subjects.get("subjects", []):
        row = without_runtime_paths(source_row)
        subject_id = row["subject_id"]
        if subject_id in seen:
            raise ValueError(f"POST139_AUTHORITY_DUPLICATE_SUBJECT:{subject_id}")
        seen.add(subject_id)
        if subject_id not in stable_subject_hashes:
            raise ValueError(f"POST139_AUTHORITY_SUBJECT_NOT_IN_WORKSPACE:{subject_id}")
        row.pop("evidence_hashes", None)
        row["stable_evidence_hashes"] = stable_subject_hashes[subject_id]
        rows.append(row)
    rows.sort(key=lambda row: row["subject_id"])
    return {
        "schema": AUTHORITY_SCHEMAS["identity"],
        "subject_count": len(rows),
        "subjects": rows,
    }


def validate_cross_ledger(final: dict[str, dict[str, Any]], identity: dict[str, Any], source_rows: list[dict[str, Any]]) -> None:
    identities = unique_by(identity["subjects"], "subject_id", "identity")
    admissions = unique_by(final["admission"].get("subjects", []), "subject_id", "admission")
    sources = {row["source_id"]: row for row in source_rows if row.get("status") == "VERIFIED"}
    if set(admissions) != set(identities):
        missing = sorted(set(identities) - set(admissions))
        extra = sorted(set(admissions) - set(identities))
        raise ValueError(f"POST139_AUTHORITY_ADMISSION_IDENTITY_SET_MISMATCH:missing={missing}:extra={extra}")

    components = unique_by(final["lineage"].get("components", []), "component_id", "lineage")
    member_to_component: dict[str, str] = {}
    for component_id, component in components.items():
        for subject_id in component.get("members", []):
            if subject_id in member_to_component:
                raise ValueError(f"POST139_AUTHORITY_LINEAGE_MULTI_COMPONENT:{subject_id}")
            member_to_component[subject_id] = component_id
    required_lineage = {sid for sid, row in admissions.items() if row.get("status") in LINEAGE_NODE_STATUSES}
    missing_lineage = sorted(required_lineage - set(member_to_component))
    if missing_lineage:
        raise ValueError(f"POST139_AUTHORITY_LINEAGE_MISSING:{missing_lineage}")

    plan_primary = unique_by(final["plan"].get("primary_subjects", []), "subject_id", "plan-primary")
    plan_aux = unique_by(final["plan"].get("auxiliary_subjects", []), "subject_id", "plan-aux")
    expected_primary = {sid for sid, row in admissions.items() if row.get("status") in PRIMARY_STATUSES}
    expected_aux = {sid for sid, row in admissions.items() if row.get("status") in AUXILIARY_STATUSES}
    if set(plan_primary) != expected_primary:
        raise ValueError(f"POST139_AUTHORITY_PRIMARY_PLAN_MISMATCH:expected={sorted(expected_primary)}:actual={sorted(plan_primary)}")
    if set(plan_aux) != expected_aux:
        raise ValueError(f"POST139_AUTHORITY_AUX_PLAN_MISMATCH:expected={sorted(expected_aux)}:actual={sorted(plan_aux)}")
    for row in list(plan_primary.values()) + list(plan_aux.values()):
        sid = row["subject_id"]
        if row.get("lineage_component_id") != member_to_component.get(sid):
            raise ValueError(f"POST139_AUTHORITY_PLAN_LINEAGE_BINDING_MISMATCH:{sid}")
        identity_row = identities[sid]
        if row.get("model_sha256") != identity_row.get("source_sha256"):
            raise ValueError(f"POST139_AUTHORITY_PLAN_MODEL_HASH_MISMATCH:{sid}")
        if row.get("structural_subject") != identity_row.get("fingerprints", {}).get("structural_subject"):
            raise ValueError(f"POST139_AUTHORITY_PLAN_IDENTITY_MISMATCH:{sid}")
        source = sources.get(row.get("source_id"))
        if source is None:
            raise ValueError(f"POST139_AUTHORITY_PLAN_SOURCE_NOT_VERIFIED:{sid}:{row.get('source_id')}")
        if row.get("source_closure_ledger_sha256") != source.get("raw_ledger_sha256"):
            raise ValueError(f"POST139_AUTHORITY_PLAN_SOURCE_LEDGER_MISMATCH:{sid}")
        if row.get("publisher_tree_digest") != source.get("publisher_tree_digest"):
            raise ValueError(f"POST139_AUTHORITY_PLAN_PUBLISHER_DIGEST_MISMATCH:{sid}")
        expected_workspace_model = f"{row['source_id']}/raw/{identity_row.get('relative_source_path')}"
        if row.get("workspace_relative_model") != expected_workspace_model:
            raise ValueError(f"POST139_AUTHORITY_PLAN_MODEL_PATH_MISMATCH:{sid}")

    roster_primary = {row["subject_id"] for row in final["roster"].get("primary_subjects", [])}
    roster_aux = {row["subject_id"] for row in final["roster"].get("auxiliary_subjects", [])}
    if roster_primary != set(plan_primary) or roster_aux != set(plan_aux):
        raise ValueError("POST139_AUTHORITY_ROSTER_PLAN_SET_MISMATCH")
    if int(final["roster"].get("post139_primary_materialization_count", -1)) != len(plan_primary):
        raise ValueError("POST139_AUTHORITY_ROSTER_PRIMARY_COUNT_MISMATCH")
    if int(final["plan"].get("primary_subject_count", -1)) != len(plan_primary):
        raise ValueError("POST139_AUTHORITY_PLAN_PRIMARY_COUNT_MISMATCH")
    if int(final["plan"].get("auxiliary_subject_count", -1)) != len(plan_aux):
        raise ValueError("POST139_AUTHORITY_PLAN_AUX_COUNT_MISMATCH")


def build_authority_ledgers(
    registry: dict[str, Any],
    source_rows: list[dict[str, Any]],
    identity: dict[str, Any],
    final: dict[str, dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    validate_cross_ledger(final, identity, source_rows)
    source_ledger = {
        "schema": AUTHORITY_SCHEMAS["source"],
        "source_count": len(source_rows),
        "verified_source_count": sum(row.get("status") == "VERIFIED" for row in source_rows),
        "sources": sorted(source_rows, key=lambda row: row["source_id"]),
        "warnings": warnings,
    }
    admission = {
        "schema": AUTHORITY_SCHEMAS["admission"],
        "status_counts": final["admission"].get("status_counts", {}),
        "appearance_variant_groups": final["admission"].get("appearance_variant_groups", []),
        "subjects": sorted(final["admission"].get("subjects", []), key=lambda row: row["subject_id"]),
    }
    lineage = {
        "schema": AUTHORITY_SCHEMAS["lineage"],
        "component_count": final["lineage"].get("component_count"),
        "cross_pack_component_count": final["lineage"].get("cross_pack_component_count"),
        "components": final["lineage"].get("components", []),
        "edges": final["lineage"].get("edges", []),
        "split_rule": "LINEAGE_COMPONENT_ID_IS_ATOMIC_FOR_FIT_UNSEEN_SPLITS",
    }
    roster = dict(final["roster"])
    roster["schema"] = AUTHORITY_SCHEMAS["roster"]
    materialization = dict(final["plan"])
    materialization["schema"] = AUTHORITY_SCHEMAS["materialization"]
    return {
        "source": source_ledger,
        "identity": identity,
        "admission": admission,
        "lineage": lineage,
        "roster": roster,
        "materialization": materialization,
    }


def seal(args: argparse.Namespace) -> int:
    registry_path = Path(args.registry).resolve()
    workspace = Path(args.workspace).resolve()
    final_dir = Path(args.final_dir).resolve()
    out = Path(args.out).resolve()
    registry = read_json(registry_path)
    assert_schema(registry, REGISTRY_SCHEMA, "registry")
    ids = [row.get("id") for row in registry.get("sources", [])]
    if len(ids) != len(set(ids)):
        raise ValueError("POST139_AUTHORITY_REGISTRY_DUPLICATE_SOURCE_ID")
    final = load_final(final_dir)
    source_rows, stable_subject_hashes, warnings = verify_source_workspace(registry, workspace)
    identity = normalize_identity_ledger(final["subjects"], stable_subject_hashes)
    ledgers = build_authority_ledgers(registry, source_rows, identity, final, warnings)

    out.mkdir(parents=True, exist_ok=True)
    ledger_refs = {}
    for key, filename in AUTHORITY_FILENAMES.items():
        path = out / filename
        write_json(path, ledgers[key])
        ledger_refs[key] = {
            "file": filename,
            "schema": ledgers[key]["schema"],
            "sha256": sha256_file(path),
            "semantic_digest": digest(ledgers[key]),
        }
    semantic_arg = getattr(args, "semantic_policy", None) or getattr(args, "semantic_overrides", None)
    semantic_policy = Path(semantic_arg).resolve() if semantic_arg else None
    root = {
        "schema": AUTHORITY_SCHEMAS["root"],
        "status": "SEALED",
        "authority_rule": "THIS_ROOT_AND_ITS_HASH_BOUND_LEDGERS_ARE_THE_ONLY_POST139_CONSUMER_AUTHORITY",
        "staging_rule": "FINALIZER_V2_AND_PER_SOURCE_AUDIT_FILES_ARE_EVIDENCE_INPUTS_NOT_CONSUMER_AUTHORITY",
        "registry": {
            "schema": registry["schema"],
            "sha256": sha256_file(registry_path),
            "semantic_digest": digest(registry),
        },
        "semantic_policy": (
            {"sha256": sha256_file(semantic_policy), "semantic_digest": digest(read_json(semantic_policy))}
            if semantic_policy and semantic_policy.is_file() else None
        ),
        "ledgers": ledger_refs,
        "counts": {
            "registry_sources": len(registry.get("sources", [])),
            "verified_sources": ledgers["source"]["verified_source_count"],
            "subjects": ledgers["identity"]["subject_count"],
            "lineage_components": ledgers["lineage"]["component_count"],
            "base139_training_rows": ledgers["roster"].get("base139_training_row_count"),
            "post139_primary": ledgers["materialization"].get("primary_subject_count"),
            "post139_auxiliary": ledgers["materialization"].get("auxiliary_subject_count"),
            "expanded_full_training_rows": ledgers["roster"].get("expanded_full_training_row_count"),
        },
        "warning_count": len(warnings),
        "warnings": warnings,
    }
    root["authority_digest"] = digest({k: v for k, v in root.items() if k != "authority_digest"})
    write_json(out / "POST139_CORPUS_AUTHORITY_V1.json", root)
    print(
        "POST139_CORPUS_AUTHORITY_SEALED",
        root["authority_digest"],
        root["counts"]["subjects"],
        root["counts"]["post139_primary"],
        root["warning_count"],
    )
    return 0


def verify_authority_dir(authority_dir: Path) -> dict[str, Any]:
    root_path = authority_dir / "POST139_CORPUS_AUTHORITY_V1.json"
    if not root_path.is_file():
        raise ValueError("POST139_AUTHORITY_ROOT_MISSING")
    root = read_json(root_path)
    assert_schema(root, AUTHORITY_SCHEMAS["root"], "authority-root")
    expected_root_digest = digest({k: v for k, v in root.items() if k != "authority_digest"})
    if root.get("authority_digest") != expected_root_digest:
        raise ValueError("POST139_AUTHORITY_ROOT_DIGEST_MISMATCH")
    loaded = {}
    for key, ref in root.get("ledgers", {}).items():
        filename = ref["file"]
        path = authority_dir / filename
        if not path.is_file():
            raise ValueError(f"POST139_AUTHORITY_LEDGER_MISSING:{filename}")
        if sha256_file(path) != ref["sha256"]:
            raise ValueError(f"POST139_AUTHORITY_LEDGER_FILE_HASH_MISMATCH:{filename}")
        payload = read_json(path)
        assert_schema(payload, ref["schema"], f"authority-ledger:{key}")
        if digest(payload) != ref["semantic_digest"]:
            raise ValueError(f"POST139_AUTHORITY_LEDGER_SEMANTIC_DIGEST_MISMATCH:{filename}")
        loaded[key] = payload
    required = set(AUTHORITY_FILENAMES)
    if set(loaded) != required:
        raise ValueError(f"POST139_AUTHORITY_LEDGER_SET_MISMATCH:{sorted(set(loaded))}")
    # Re-run the consumer-facing cross-ledger checks without staging files.
    identities = unique_by(loaded["identity"].get("subjects", []), "subject_id", "authority-identity")
    admissions = unique_by(loaded["admission"].get("subjects", []), "subject_id", "authority-admission")
    if set(identities) != set(admissions):
        raise ValueError("POST139_AUTHORITY_VERIFY_IDENTITY_ADMISSION_SET_MISMATCH")
    primary = {row["subject_id"] for row in loaded["materialization"].get("primary_subjects", [])}
    expected_primary = {sid for sid, row in admissions.items() if row.get("status") in PRIMARY_STATUSES}
    if primary != expected_primary:
        raise ValueError("POST139_AUTHORITY_VERIFY_PRIMARY_SET_MISMATCH")
    auxiliary = {row["subject_id"] for row in loaded["materialization"].get("auxiliary_subjects", [])}
    expected_aux = {sid for sid, row in admissions.items() if row.get("status") in AUXILIARY_STATUSES}
    if auxiliary != expected_aux:
        raise ValueError("POST139_AUTHORITY_VERIFY_AUX_SET_MISMATCH")
    sources = {row["source_id"]: row for row in loaded["source"].get("sources", []) if row.get("status") == "VERIFIED"}
    for row in loaded["materialization"].get("primary_subjects", []) + loaded["materialization"].get("auxiliary_subjects", []):
        sid = row["subject_id"]
        identity = identities[sid]
        source = sources.get(row.get("source_id"))
        if source is None:
            raise ValueError(f"POST139_AUTHORITY_VERIFY_SOURCE_MISSING:{sid}")
        if row.get("source_closure_ledger_sha256") != source.get("raw_ledger_sha256"):
            raise ValueError(f"POST139_AUTHORITY_VERIFY_SOURCE_LEDGER_MISMATCH:{sid}")
        if row.get("publisher_tree_digest") != source.get("publisher_tree_digest"):
            raise ValueError(f"POST139_AUTHORITY_VERIFY_PUBLISHER_DIGEST_MISMATCH:{sid}")
        if row.get("model_sha256") != identity.get("source_sha256"):
            raise ValueError(f"POST139_AUTHORITY_VERIFY_MODEL_HASH_MISMATCH:{sid}")
    return root


def verify(args: argparse.Namespace) -> int:
    authority_dir = Path(args.authority_dir).resolve()
    root = verify_authority_dir(authority_dir)
    print("POST139_CORPUS_AUTHORITY_VERIFY_PASS", root["authority_digest"])
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_seal = sub.add_parser("seal")
    p_seal.add_argument("--registry", required=True)
    p_seal.add_argument("--workspace", required=True)
    p_seal.add_argument("--final-dir", required=True)
    p_seal.add_argument("--out", required=True)
    p_seal.add_argument("--semantic-policy", help="Canonical semantic subject policy JSON")
    p_seal.add_argument("--semantic-overrides", help=argparse.SUPPRESS)
    p_seal.set_defaults(func=seal)
    p_verify = sub.add_parser("verify")
    p_verify.add_argument("--authority-dir", required=True)
    p_verify.set_defaults(func=verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
