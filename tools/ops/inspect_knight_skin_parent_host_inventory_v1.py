#!/usr/bin/env python3
"""Read-only typed-parent inventory for current sealed Knight authority roots.

This tool never imports artifacts, runs a model, executes a Compiler capability,
mints qualification, promotes a model, activates product.skin, or rewrites host
state. It searches only the explicit current Knight roots named by
canonical/PRODUCT_NEAREST_KNIGHT_WITNESS_V1.json and accepts candidates only
when their exact typed schema decodes under the current contracts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

TARGETS = {
    "mesh": "RealSaS.CanonicalMeshIR.v1",
    "rig": "RealSaS.CanonicalRigIR.v1",
    "surface": "RealSaS.RiggingSurfaceIR.v1",
    "support": "RealSaS.TESSAMaterialSupportFieldIR.v1",
}
MAX_JSON_FILES = 20000
MAX_JSON_BYTES = 64 << 20


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _schema(payload: dict[str, Any]) -> str:
    return str(payload.get("schema_version") or payload.get("schema") or "")


def _load_decoders() -> dict[str, Callable[[dict[str, Any]], Any]]:
    from compiler.realsas_compiler_core.artifact_codec_v2 import rigging_surface_from_dict
    from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import canonical_mesh_from_dict_v1
    from compiler.realsas_compiler_core.canonical_rig_ir_v1 import canonical_rig_from_dict_v1
    from models.mira.inference_v55 import _material_support_rows

    return {
        "mesh": canonical_mesh_from_dict_v1,
        "rig": canonical_rig_from_dict_v1,
        "surface": rigging_surface_from_dict,
        "support": _material_support_rows,
    }


def _explicit_roots(repo_root: Path, authority_root: Path) -> list[dict[str, str]]:
    inventory_path = repo_root / "canonical" / "PRODUCT_NEAREST_KNIGHT_WITNESS_V1.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inventory.get("schema") != "RealSaS.ProductNearestWitnessInventory.v1":
        raise RuntimeError("KNIGHT_PARENT_HOST_INVENTORY_DESCRIPTOR_SCHEMA")
    names = dict(inventory.get("current_host_lineage_names") or {})
    required = {
        "caa_run": str(names.get("caa_run") or ""),
        "carrier_materialization_run": str(names.get("carrier_materialization_run") or ""),
    }
    if not all(required.values()):
        raise RuntimeError("KNIGHT_PARENT_HOST_INVENTORY_ROOT_NAME_MISSING")
    rows = []
    for role, name in required.items():
        root = (authority_root / "runs" / name).resolve()
        expected_parent = (authority_root / "runs").resolve()
        if not root.is_relative_to(expected_parent):
            raise RuntimeError("KNIGHT_PARENT_HOST_INVENTORY_ROOT_ESCAPE")
        if not root.is_dir():
            raise RuntimeError(f"KNIGHT_PARENT_HOST_INVENTORY_ROOT_MISSING:{role}:{root}")
        rows.append({"role": role, "run_name": name, "path": str(root)})
    return rows


def _read_candidates(roots: list[dict[str, str]]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    decoders = _load_decoders()
    by_schema = {schema: role for role, schema in TARGETS.items()}
    valid: dict[str, list[dict[str, Any]]] = {role: [] for role in TARGETS}
    diagnostics: list[dict[str, Any]] = []
    scanned = 0

    for root_row in roots:
        root = Path(root_row["path"])
        for path in sorted(root.rglob("*.json")):
            if not path.is_file():
                continue
            scanned += 1
            if scanned > MAX_JSON_FILES:
                raise RuntimeError("KNIGHT_PARENT_HOST_INVENTORY_JSON_FILE_LIMIT")
            size = path.stat().st_size
            if size > MAX_JSON_BYTES:
                diagnostics.append({
                    "status": "SKIPPED_OVERSIZED_JSON",
                    "root_role": root_row["role"],
                    "path": str(path),
                    "size_bytes": size,
                })
                continue
            try:
                raw = path.read_bytes()
                payload = json.loads(raw.decode("utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            schema = _schema(payload)
            role = by_schema.get(schema)
            if role is None:
                continue
            base = {
                "role": role,
                "schema": schema,
                "root_role": root_row["role"],
                "run_name": root_row["run_name"],
                "path": str(path),
                "content_sha256": _sha256(raw),
                "size_bytes": len(raw),
            }
            try:
                value = decoders[role](payload)
                record = dict(base)
                record["value"] = value
                valid[role].append(record)
                diagnostics.append({**base, "status": "VALID_TYPED_CANDIDATE"})
            except Exception as exc:
                diagnostics.append({
                    **base,
                    "status": "INCOMPATIBLE_TYPED_CANDIDATE",
                    "error": f"{type(exc).__name__}:{exc}",
                })
    return valid, diagnostics


def _support_covers_mesh(mesh: Any, support: Any) -> bool:
    by_id = {str(row.proposal_vertex_id) for row in support.rows}
    if len(by_id) != len(support.rows):
        return False
    for vertex in mesh.vertices:
        coefficients = tuple(vertex.support_binding.coefficients)
        if not coefficients:
            return False
        if abs(sum(float(weight) for _, weight in coefficients) - 1.0) > 2.0e-6:
            return False
        if any(str(proposal_id) not in by_id for proposal_id, _ in coefficients):
            return False
    return True


def _public(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "value"}


def _select_bundle(valid: dict[str, list[dict[str, Any]]]) -> dict[str, Any] | None:
    for mesh_row in valid["mesh"]:
        mesh = mesh_row["value"]
        for rig_row in valid["rig"]:
            rig = rig_row["value"]
            if rig.mesh_lineage_hash != mesh.mesh_lineage_hash:
                continue
            for surface_row in valid["surface"]:
                surface = surface_row["value"]
                if surface.geometry_lineage_hash != rig.feature_evidence_binding_hash:
                    continue
                if surface.geometry_lineage_hash != mesh.source_evidence_binding_hash:
                    continue
                for support_row in valid["support"]:
                    support = support_row["value"]
                    if support.surface_binding_hash != surface.geometry_lineage_hash:
                        continue
                    if support.partition_binding_hash != mesh.partition_binding_hash:
                        continue
                    if not _support_covers_mesh(mesh, support):
                        continue
                    return {
                        "artifacts": {
                            "mesh": _public(mesh_row),
                            "rig": _public(rig_row),
                            "surface": _public(surface_row),
                            "support": _public(support_row),
                        },
                        "mesh_lineage_hash": str(mesh.mesh_lineage_hash),
                        "rig_lineage_hash": str(rig.rig_lineage_hash),
                        "surface_geometry_lineage_hash": str(surface.geometry_lineage_hash),
                        "partition_binding_hash": str(mesh.partition_binding_hash),
                        "material_support_field_lineage_hash": str(support.field_lineage_hash),
                        "binding_status": "PASS_EXACT_TYPED_SKIN_INPUT_BINDINGS",
                    }
    return None


def inspect(repo_root: Path, authority_root: Path, source_code_sha: str) -> dict[str, Any]:
    roots = _explicit_roots(repo_root, authority_root)
    valid, diagnostics = _read_candidates(roots)
    bundle = _select_bundle(valid)
    counts = {role: len(rows) for role, rows in valid.items()}
    if bundle is not None:
        status = "EXACT_TYPED_BUNDLE_FOUND_ON_HOST"
        next_action = "IMPORT_EXISTING_TYPED_BUNDLE_TO_PLATFORM_CAS_AND_BIND_KNIGHT__NO_REEXECUTION"
    elif any(counts.values()):
        status = "PARTIAL_OR_INCOMPATIBLE_TYPED_PARENTS_ON_HOST"
        next_action = "EXECUTE_ONLY_MISSING_OR_INCOMPATIBLE_OWNER_RESPONSIBILITIES"
    else:
        status = "NO_EXACT_TYPED_PARENTS_IN_CURRENT_SEALED_ROOTS"
        next_action = "EXECUTE_CURRENT_MESH_AND_RIG_RESPONSIBILITIES"
    return {
        "schema": "RealSaS.KnightSkinParentHostInventory.v1",
        "status": status,
        "source_code_sha": source_code_sha,
        "authority_roots": roots,
        "required_schemas": TARGETS,
        "valid_candidate_counts": counts,
        "candidate_diagnostics": diagnostics,
        "compatible_typed_bundle": bundle,
        "decision": {"next_action": next_action},
        "host_scan_is_execution_authority": False,
        "legacy_payload_reinterpretation_performed": False,
        "artifact_import_performed": False,
        "model_inference_executed": False,
        "compiler_capability_executed": False,
        "qualification_minted": False,
        "model_promotion_performed": False,
        "product_skin_activated": False,
        "product_revision_minted": False,
        "genericity_claimed": False,
    }


def _failure(source_code_sha: str, exc: Exception) -> dict[str, Any]:
    return {
        "schema": "RealSaS.KnightSkinParentHostInventoryFailure.v1",
        "status": "FAILED_READ_ONLY_INSPECTION",
        "source_code_sha": source_code_sha,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "artifact_import_performed": False,
        "model_inference_executed": False,
        "compiler_capability_executed": False,
        "qualification_minted": False,
        "model_promotion_performed": False,
        "product_skin_activated": False,
        "product_revision_minted": False,
        "genericity_claimed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--authority-root", type=Path, required=True)
    parser.add_argument("--expected-main-sha", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    expected = args.expected_main_sha.strip().lower()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = inspect(args.repo_root.resolve(), args.authority_root.resolve(), expected)
        args.out.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": report["status"],
            "valid_candidate_counts": report["valid_candidate_counts"],
            "next_action": report["decision"]["next_action"],
        }, sort_keys=True), flush=True)
    except Exception as exc:
        report = _failure(expected, exc)
        args.out.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
