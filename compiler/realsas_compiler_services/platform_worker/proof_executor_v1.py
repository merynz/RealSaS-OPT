from __future__ import annotations

"""CAS-native product.proof executor for the modular product graph."""

import json
from typing import Any, Callable

PROOF_EXECUTOR_CONTRACT_V1 = "RealSaS.ProductProofCompile.v1"


def _typed(
    request: dict[str, Any], artifact_type: str, schema_version: str
) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in request.get("input_artifacts") or ()
        if str(row.get("artifact_type") or "") == artifact_type
        and str(row.get("schema_version") or "") == schema_version
    ]


def _one(request: dict[str, Any], artifact_type: str, schema_version: str) -> dict[str, Any]:
    rows = _typed(request, artifact_type, schema_version)
    if len(rows) != 1:
        raise RuntimeError(
            f"PROOF_INPUT_ARTIFACT_CARDINALITY:{artifact_type}:{schema_version}:{len(rows)}"
        )
    return rows[0]


def _many(request: dict[str, Any], artifact_type: str, schema_version: str) -> list[dict[str, Any]]:
    rows = _typed(request, artifact_type, schema_version)
    if not rows:
        raise RuntimeError(
            f"PROOF_INPUT_ARTIFACT_REQUIRED:{artifact_type}:{schema_version}"
        )
    rows.sort(key=lambda row: (str(row.get("content_sha256") or ""), str(row.get("role") or "")))
    return rows


def _json_payload(read_object, ref):
    try:
        value = json.loads(read_object(ref).decode("utf-8"))
    except Exception as exc:
        raise RuntimeError("PROOF_TYPED_JSON_DECODE_FAILED") from exc
    if not isinstance(value, dict):
        raise RuntimeError("PROOF_TYPED_JSON_OBJECT_REQUIRED")
    return value


def _json_bytes(value: Any) -> bytes:
    payload = value.to_dict() if hasattr(value, "to_dict") else value
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def execute_proof_capability_v1(
    request: dict[str, Any],
    *,
    read_object: Callable[[dict[str, Any]], bytes],
    put_bytes: Callable[[bytes], dict[str, Any]],
) -> dict[str, Any]:
    metadata = dict(request.get("capability_metadata") or {})
    if metadata.get("executor_contract") != PROOF_EXECUTOR_CONTRACT_V1:
        raise RuntimeError("PROOF_EXECUTOR_CONTRACT_DRIFT")
    if str(request.get("kind") or "") != "MODULE":
        raise RuntimeError("PROOF_CAPABILITY_KIND_DRIFT")
    if request.get("model_releases"):
        raise RuntimeError("PROOF_DIRECT_MODEL_RELEASE_FORBIDDEN")

    from compiler.realsas_compiler_core.canonical_graphics_ir_v1 import (
        canonical_graphics_from_dict_v1,
    )
    from compiler.realsas_compiler_core.canonical_mechanics_ir_v1 import (
        canonical_mechanics_from_dict_v1,
    )
    from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import (
        canonical_mesh_from_dict_v1,
    )
    from compiler.realsas_compiler_core.canonical_motion_ir_v2 import (
        canonical_motion_from_dict_v2,
    )
    from compiler.realsas_compiler_core.canonical_rig_ir_v2 import (
        canonical_rig_from_dict_v2,
    )
    from compiler.realsas_compiler_core.canonical_skin_ir_v2 import (
        canonical_skin_from_dict_v2,
    )
    from compiler.realsas_compiler_core.modular_product_proof_eval_v1 import (
        evaluate_modular_product_proof_v1,
    )

    mesh = canonical_mesh_from_dict_v1(
        _json_payload(read_object, _one(request, "RealSaS.CanonicalMeshIR", "v1"))
    )
    rig = canonical_rig_from_dict_v2(
        _json_payload(read_object, _one(request, "RealSaS.CanonicalRigIR", "v2"))
    )
    skin = canonical_skin_from_dict_v2(
        _json_payload(read_object, _one(request, "RealSaS.CanonicalSkinIR", "v2"))
    )
    mechanics = canonical_mechanics_from_dict_v1(
        _json_payload(read_object, _one(request, "RealSaS.CanonicalMechanicsIR", "v1"))
    )
    graphics = canonical_graphics_from_dict_v1(
        _json_payload(read_object, _one(request, "RealSaS.CanonicalGraphicsIR", "v1"))
    )
    motions = tuple(
        canonical_motion_from_dict_v2(_json_payload(read_object, ref))
        for ref in _many(request, "RealSaS.CanonicalMotionIR", "v2")
    )

    proof = evaluate_modular_product_proof_v1(
        mesh,
        rig,
        skin,
        mechanics,
        graphics,
        motions,
    )
    sealed = put_bytes(_json_bytes(proof))
    output = {
        "role": "product_proof",
        "artifact_type": "RealSaS.ProductProofBundleIR",
        "schema_version": "v1",
        "storage_key": sealed["storage_key"],
        "content_sha256": sealed["content_sha256"],
        "size_bytes": sealed["size_bytes"],
        "authority_class": "PRODUCT_PASS_AUTHORITY",
    }
    failed_domains = [
        report.proof_domain
        for report in proof.domain_reports
        if report.status != "PASS"
    ]
    diagnostics = {
        "schema": "RealSaS.ProductProofDiagnostics.v1",
        "product_state_hash": proof.source_product_state_hash,
        "proof_bundle_hash": proof.proof_bundle_hash,
        "overall_status": proof.overall_status,
        "required_domains": list(proof.required_domains),
        "failed_domains": failed_domains,
        "legacy_canonical_puppet_graph_projection_used": False,
        "product_revision_minted": False,
    }
    if proof.overall_status != "PASS":
        return {
            "capability_id": str(request.get("capability_id") or ""),
            "status": "FAIL",
            "outputs": [output],
            "diagnostics": diagnostics,
            "failure": {
                "code": "PRODUCT_PROOF_NOT_PASS",
                "class": "PRODUCT_PROOF",
                "reported_owner_module_id": "product.proof",
                "diagnostics": {
                    "failed_domains": failed_domains,
                    "proof_bundle_hash": proof.proof_bundle_hash,
                },
            },
        }
    return {
        "capability_id": str(request.get("capability_id") or ""),
        "status": "PASS",
        "outputs": [output],
        "diagnostics": diagnostics,
        "failure": None,
    }


__all__ = ["PROOF_EXECUTOR_CONTRACT_V1", "execute_proof_capability_v1"]
