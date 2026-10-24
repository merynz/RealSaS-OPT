#!/usr/bin/env python3
"""Read-only current Knight Skin prerequisite inspection.

This command discovers typed artifacts only through Platform read APIs, verifies
bounded CAS bytes, decodes the exact current Mesh/Rig/surface/support contracts,
and reports whether an internally compatible Skin input bundle exists. It never
imports bytes, runs MIRA, mints qualification, promotes a model, or activates a
product capability.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

MIRA_ARCHITECTURE = "RealSaS.MIRA.DirectCarrierSimplex.Axis541.v5_5"
MIRA_RUNTIME_SHA256 = "b5ad2d56a9006058cba30c0b123d4e111f3eea3b78ed42f3542cbd095983139a"
REQUIRED = (
    ("mesh", "RealSaS.CanonicalMeshIR", "v1"),
    ("rig", "RealSaS.CanonicalRigIR", "v1"),
    ("surface", "RealSaS.RiggingSurfaceIR", "v1"),
    ("support", "RealSaS.TESSAMaterialSupportFieldIR", "v1"),
)


def _get_json(api_root: str, path: str) -> Any:
    request = Request(api_root.rstrip("/") + path, method="GET")
    try:
        with urlopen(request, timeout=30) as response:
            if response.status != 200:
                raise RuntimeError(f"SKIN_READINESS_HTTP_STATUS:{path}:{response.status}")
            return json.load(response)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"SKIN_READINESS_HTTP_{exc.code}:{path}:{detail}") from exc


def _get_bytes(api_root: str, artifact_id: str, expected_sha: str, expected_size: int) -> bytes:
    request = Request(api_root.rstrip("/") + f"/v1/artifacts/{artifact_id}/bytes", method="GET")
    with urlopen(request, timeout=60) as response:
        if response.status != 200:
            raise RuntimeError(f"SKIN_READINESS_BYTES_STATUS:{artifact_id}:{response.status}")
        header_sha = str(response.headers.get("X-Content-SHA256") or "")
        raw = response.read()
    actual_sha = hashlib.sha256(raw).hexdigest()
    if header_sha != expected_sha or actual_sha != expected_sha:
        raise RuntimeError(f"SKIN_READINESS_BYTES_SHA_DRIFT:{artifact_id}")
    if len(raw) != int(expected_size):
        raise RuntimeError(f"SKIN_READINESS_BYTES_SIZE_DRIFT:{artifact_id}")
    return raw


def _artifact_candidates(api_root: str, subject_id: str, artifact_type: str, schema: str) -> list[dict[str, Any]]:
    query = urlencode({"artifact_type": artifact_type, "schema_version": schema, "limit": 25})
    rows = _get_json(api_root, f"/v1/subjects/{subject_id}/artifacts?{query}")
    if not isinstance(rows, list):
        raise RuntimeError("SKIN_READINESS_SUBJECT_ARTIFACT_LIST_INVALID")
    seen: set[str] = set()
    out = []
    for row in rows:
        artifact_id = str(row.get("artifact_id") or "")
        if not artifact_id or artifact_id in seen:
            continue
        seen.add(artifact_id)
        out.append(dict(row))
    return out


def _decode_json_bytes(raw: bytes, artifact_id: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"SKIN_READINESS_JSON_DECODE:{artifact_id}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"SKIN_READINESS_JSON_OBJECT_REQUIRED:{artifact_id}")
    return value


def _load_valid_candidates(api_root: str, subject_id: str) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    from compiler.realsas_compiler_core.artifact_codec_v2 import rigging_surface_from_dict
    from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import canonical_mesh_from_dict_v1
    from compiler.realsas_compiler_core.canonical_rig_ir_v1 import canonical_rig_from_dict_v1
    from models.mira.inference_v55 import _material_support_rows

    decoders = {
        "mesh": canonical_mesh_from_dict_v1,
        "rig": canonical_rig_from_dict_v1,
        "surface": rigging_surface_from_dict,
        "support": _material_support_rows,
    }
    valid: dict[str, list[dict[str, Any]]] = {role: [] for role, _, _ in REQUIRED}
    diagnostics: list[dict[str, Any]] = []
    for role, artifact_type, schema in REQUIRED:
        rows = _artifact_candidates(api_root, subject_id, artifact_type, schema)
        if not rows:
            diagnostics.append({"role": role, "artifact_type": artifact_type, "status": "MISSING"})
            continue
        for row in rows:
            artifact_id = str(row["artifact_id"])
            try:
                metadata = _get_json(api_root, f"/v1/artifacts/{artifact_id}")
                if metadata.get("artifact_type") != artifact_type or metadata.get("schema_version") != schema:
                    raise RuntimeError("ARTIFACT_TYPE_METADATA_DRIFT")
                raw = _get_bytes(
                    api_root,
                    artifact_id,
                    str(metadata["content_sha256"]),
                    int(metadata["size_bytes"]),
                )
                payload = _decode_json_bytes(raw, artifact_id)
                value = decoders[role](payload)
                record = {
                    "role": role,
                    "artifact_id": artifact_id,
                    "artifact_type": artifact_type,
                    "schema_version": schema,
                    "content_sha256": str(metadata["content_sha256"]),
                    "semantic_sha256": str(metadata["semantic_sha256"]),
                    "size_bytes": int(metadata["size_bytes"]),
                    "producer_contract": str(metadata.get("producer_contract") or ""),
                    "implementation_sha256": str(metadata.get("implementation_sha256") or ""),
                    "policy_sha256": str(metadata.get("policy_sha256") or ""),
                    "source_kind": str(row.get("source_kind") or ""),
                    "source_id": str(row.get("source_id") or ""),
                    "source_role": str(row.get("role") or ""),
                    "source_origin": str(row.get("origin") or ""),
                    "value": value,
                }
                valid[role].append(record)
                diagnostics.append({
                    key: record[key]
                    for key in (
                        "role",
                        "artifact_id",
                        "artifact_type",
                        "content_sha256",
                        "source_kind",
                        "source_id",
                        "source_role",
                        "source_origin",
                    )
                } | {"status": "VALID_TYPED_ARTIFACT"})
            except Exception as exc:
                diagnostics.append({
                    "role": role,
                    "artifact_id": artifact_id,
                    "artifact_type": artifact_type,
                    "status": "INCOMPATIBLE",
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
        total = sum(float(weight) for _, weight in coefficients)
        if abs(total - 1.0) > 2.0e-6:
            return False
        if any(str(proposal_id) not in by_id for proposal_id, _ in coefficients):
            return False
    return True


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
                    selected = {}
                    for role, row in (
                        ("mesh", mesh_row),
                        ("rig", rig_row),
                        ("surface", surface_row),
                        ("support", support_row),
                    ):
                        selected[role] = {
                            key: row[key]
                            for key in (
                                "artifact_id",
                                "artifact_type",
                                "schema_version",
                                "content_sha256",
                                "semantic_sha256",
                                "size_bytes",
                                "producer_contract",
                                "implementation_sha256",
                                "policy_sha256",
                                "source_kind",
                                "source_id",
                                "source_role",
                                "source_origin",
                            )
                        }
                    return {
                        "artifacts": selected,
                        "mesh_lineage_hash": str(mesh.mesh_lineage_hash),
                        "rig_lineage_hash": str(rig.rig_lineage_hash),
                        "surface_geometry_lineage_hash": str(surface.geometry_lineage_hash),
                        "partition_binding_hash": str(mesh.partition_binding_hash),
                        "material_support_field_lineage_hash": str(support.field_lineage_hash),
                        "binding_status": "PASS_EXACT_TYPED_SKIN_INPUT_BINDINGS",
                        "reuse_eligibility_claimed": False,
                    }
    return None


def _mira_state(api_root: str) -> dict[str, Any]:
    releases = _get_json(api_root, "/v1/model-releases")
    channels = _get_json(api_root, "/v1/model-channels")
    exact = [
        row for row in releases
        if str(row.get("family") or "") == "MIRA"
        and str(row.get("architecture_id") or "") == MIRA_ARCHITECTURE
        and str(row.get("checkpoint_content_sha256") or "") == MIRA_RUNTIME_SHA256
    ]
    if len(exact) != 1:
        return {
            "status": "MISSING" if not exact else "AMBIGUOUS",
            "exact_release_count": len(exact),
            "runtime_identity_ready": False,
            "research_execution_admitted": False,
        }
    release = exact[0]
    release_id = str(release["id"])
    bound = [
        str(row.get("channel") or "")
        for row in channels
        if str(row.get("family") or "") == "MIRA"
        and str(row.get("model_release_id") or "") == release_id
    ]
    lifecycle = str(release.get("lifecycle_state") or "")
    research_admitted = any(channel in {"RESEARCH", "CANDIDATE"} for channel in bound)
    return {
        "status": "READY_RUNTIME_IDENTITY",
        "model_release_id": release_id,
        "name": str(release.get("name") or ""),
        "architecture_id": MIRA_ARCHITECTURE,
        "checkpoint_artifact_id": str(release.get("checkpoint_artifact_id") or ""),
        "checkpoint_content_sha256": MIRA_RUNTIME_SHA256,
        "lifecycle_state": lifecycle,
        "evaluations": list(release.get("evaluations") or []),
        "bound_channels": sorted(bound),
        "runtime_identity_ready": True,
        "research_execution_admitted": research_admitted,
        "draft_evaluation_required": lifecycle == "DRAFT" and not research_admitted,
    }


def inspect(api_root: str, expected_main_sha: str) -> dict[str, Any]:
    subject = _get_json(api_root, "/v1/subjects/by-slug/knight")
    if subject.get("schema") != "RealSaS.SubjectIdentity.v1":
        raise RuntimeError("SKIN_READINESS_SUBJECT_IDENTITY_SCHEMA_DRIFT")
    subject_id = str(subject.get("subject_id") or "")
    if not subject_id or str(subject.get("slug") or "") != "knight":
        raise RuntimeError("SKIN_READINESS_KNIGHT_SUBJECT_IDENTITY_DRIFT")

    valid, artifact_diagnostics = _load_valid_candidates(api_root, subject_id)
    bundle = _select_bundle(valid)
    mira = _mira_state(api_root)
    typed_bundle_ready = bundle is not None
    runtime_ready = bool(mira.get("runtime_identity_ready"))
    research_admitted = bool(mira.get("research_execution_admitted"))
    if typed_bundle_ready and runtime_ready and research_admitted:
        status = "READY_FOR_RESEARCH_SKIN_EXECUTION"
        next_action = "RUN_CURRENT_SKIN_ATTEMPT_THROUGH_ADMITTED_RESEARCH_MODEL_CHANNEL"
    elif typed_bundle_ready and runtime_ready and bool(mira.get("draft_evaluation_required")):
        status = "READY_FOR_DRAFT_MODEL_EVALUATION"
        next_action = "RUN_DRAFT_MIRA_CURRENT_SKIN_EVALUATION__DO_NOT_ACTIVATE_PRODUCT_SKIN"
    else:
        status = "NOT_READY"
        next_action = "CLOSE_MISSING_TYPED_SKIN_PREREQUISITES"

    return {
        "schema": "RealSaS.KnightSkinReadinessInspection.v1",
        "status": status,
        "source_code_sha": expected_main_sha,
        "subject": {
            "id": subject_id,
            "slug": str(subject.get("slug") or ""),
            "display_name": str(subject.get("display_name") or ""),
            "identity_source": "PLATFORM_SUBJECT_IDENTITY_READ_V1",
        },
        "required_artifacts": [
            {"role": role, "artifact_type": artifact_type, "schema_version": schema}
            for role, artifact_type, schema in REQUIRED
        ],
        "valid_candidate_counts": {role: len(rows) for role, rows in valid.items()},
        "artifact_diagnostics": artifact_diagnostics,
        "compatible_typed_bundle": bundle,
        "mira": mira,
        "decision": {
            "typed_bundle_ready": typed_bundle_ready,
            "runtime_identity_ready": runtime_ready,
            "research_execution_admitted": research_admitted,
            "draft_evaluation_required": bool(mira.get("draft_evaluation_required")),
            "next_action": next_action,
        },
        "scientific_pass_minted": False,
        "qualification_minted": False,
        "model_inference_executed": False,
        "platform_mutation_performed": False,
        "model_promotion_performed": False,
        "product_skin_activated": False,
        "product_revision_minted": False,
        "genericity_claimed": False,
    }


def _failure_report(expected_main_sha: str, exc: Exception) -> dict[str, Any]:
    return {
        "schema": "RealSaS.KnightSkinReadinessInspectionFailure.v1",
        "status": "FAILED_INSPECTION",
        "source_code_sha": expected_main_sha,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "scientific_pass_minted": False,
        "qualification_minted": False,
        "model_inference_executed": False,
        "platform_mutation_performed": False,
        "model_promotion_performed": False,
        "product_skin_activated": False,
        "product_revision_minted": False,
        "genericity_claimed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-root", default="http://127.0.0.1:8080")
    parser.add_argument("--expected-main-sha", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    expected_main_sha = args.expected_main_sha.strip().lower()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = inspect(args.api_root, expected_main_sha)
        args.out.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": report["status"],
            "typed_bundle_ready": report["decision"]["typed_bundle_ready"],
            "runtime_identity_ready": report["decision"]["runtime_identity_ready"],
            "research_execution_admitted": report["decision"]["research_execution_admitted"],
            "next_action": report["decision"]["next_action"],
        }, sort_keys=True), flush=True)
    except Exception as exc:
        failure = _failure_report(expected_main_sha, exc)
        args.out.write_text(json.dumps(failure, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": "FAILED_INSPECTION", "error": str(exc)}, sort_keys=True), flush=True)
        raise


if __name__ == "__main__":
    main()
