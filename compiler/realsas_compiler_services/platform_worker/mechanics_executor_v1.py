from __future__ import annotations

"""CAS-native product.mechanics executor for control-ready Rig/Skin products."""

import json
from typing import Any, Callable

MECHANICS_EXECUTOR_CONTRACT_V1 = "RealSaS.ProductMechanicsCompile.v1"


def _one(request: dict[str, Any], artifact_type: str, schema_version: str) -> dict[str, Any]:
    rows = [dict(r) for r in request.get("input_artifacts") or ()
            if str(r.get("artifact_type") or "") == artifact_type
            and str(r.get("schema_version") or "") == schema_version]
    if len(rows) != 1:
        raise RuntimeError(f"MECHANICS_INPUT_ARTIFACT_CARDINALITY:{artifact_type}:{schema_version}:{len(rows)}")
    return rows[0]


def _json_payload(read_object, ref):
    try:
        value = json.loads(read_object(ref).decode("utf-8"))
    except Exception as exc:
        raise RuntimeError("MECHANICS_TYPED_JSON_DECODE_FAILED") from exc
    if not isinstance(value, dict):
        raise RuntimeError("MECHANICS_TYPED_JSON_OBJECT_REQUIRED")
    return value


def _json_bytes(value: Any) -> bytes:
    payload = value.to_dict() if hasattr(value, "to_dict") else value
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def execute_mechanics_capability_v1(
    request: dict[str, Any], *, read_object: Callable[[dict[str, Any]], bytes],
    put_bytes: Callable[[bytes], dict[str, Any]],
) -> dict[str, Any]:
    metadata = dict(request.get("capability_metadata") or {})
    if metadata.get("executor_contract") != MECHANICS_EXECUTOR_CONTRACT_V1:
        raise RuntimeError("MECHANICS_EXECUTOR_CONTRACT_DRIFT")
    if str(request.get("kind") or "") != "MODULE":
        raise RuntimeError("MECHANICS_CAPABILITY_KIND_DRIFT")
    if request.get("model_releases"):
        raise RuntimeError("MECHANICS_LEARNED_MODEL_RELEASE_FORBIDDEN")

    from compiler.realsas_compiler_core.canonical_mechanics_ir_v1 import mint_canonical_mechanics_v1
    from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import canonical_mesh_from_dict_v1
    from compiler.realsas_compiler_core.canonical_rig_ir_v2 import canonical_rig_from_dict_v2
    from compiler.realsas_compiler_core.canonical_skin_ir_v2 import canonical_skin_from_dict_v2

    mesh = canonical_mesh_from_dict_v1(_json_payload(read_object, _one(request, "RealSaS.CanonicalMeshIR", "v1")))
    rig = canonical_rig_from_dict_v2(_json_payload(read_object, _one(request, "RealSaS.CanonicalRigIR", "v2")))
    skin = canonical_skin_from_dict_v2(_json_payload(read_object, _one(request, "RealSaS.CanonicalSkinIR", "v2")))
    policy = dict(metadata.get("rest_qualification") or {})
    mechanics = mint_canonical_mechanics_v1(
        mesh, rig, skin,
        rest_position_tolerance=float(policy.get("max_position_drift", 1e-5)),
        rest_constraint_tolerance=float(policy.get("max_constraint_residual", 1e-4)),
    )
    sealed = put_bytes(_json_bytes(mechanics))
    output = {
        "role": "canonical_mechanics",
        "artifact_type": "RealSaS.CanonicalMechanicsIR",
        "schema_version": "v1",
        "storage_key": sealed["storage_key"],
        "content_sha256": sealed["content_sha256"],
        "size_bytes": sealed["size_bytes"],
        "authority_class": "COMPILER_CANONICAL_MECHANICS",
    }
    return {
        "capability_id": str(request.get("capability_id") or ""),
        "status": "PASS",
        "outputs": [output],
        "diagnostics": {
            "schema": "RealSaS.ProductMechanicsDiagnostics.v1",
            "mechanics_lineage_hash": mechanics.mechanics_lineage_hash,
            "rig_lineage_hash": mechanics.rig_lineage_hash,
            "skin_lineage_hash": mechanics.skin_lineage_hash,
            "constraint_schedule_count": len(mechanics.constraint_schedule),
            "learned_inference_performed": False,
            "rig_program_mutated": False,
            "product_revision_pass_claimed": False,
        },
        "failure": None,
    }


__all__ = ["MECHANICS_EXECUTOR_CONTRACT_V1", "execute_mechanics_capability_v1"]
