from __future__ import annotations

"""CAS-native product.motion executor for control-space CanonicalMotionIR.v2."""

import json
from typing import Any, Callable

MOTION_EXECUTOR_CONTRACT_V2 = "RealSaS.ProductMotionCompile.v2"


def _one(request: dict[str, Any], artifact_type: str, schema_version: str) -> dict[str, Any]:
    rows = [
        dict(row)
        for row in request.get("input_artifacts") or ()
        if str(row.get("artifact_type") or "") == artifact_type
        and str(row.get("schema_version") or "") == schema_version
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"MOTION_V2_INPUT_ARTIFACT_CARDINALITY:{artifact_type}:{schema_version}:{len(rows)}"
        )
    return rows[0]


def _optional_one(
    request: dict[str, Any], artifact_type: str, schema_version: str
) -> dict[str, Any] | None:
    rows = [
        dict(row)
        for row in request.get("input_artifacts") or ()
        if str(row.get("artifact_type") or "") == artifact_type
        and str(row.get("schema_version") or "") == schema_version
    ]
    if len(rows) > 1:
        raise RuntimeError(
            f"MOTION_V2_INPUT_ARTIFACT_CARDINALITY:{artifact_type}:{schema_version}:{len(rows)}"
        )
    return rows[0] if rows else None


def _json_payload(read_object, ref):
    try:
        value = json.loads(read_object(ref).decode("utf-8"))
    except Exception as exc:
        raise RuntimeError("MOTION_V2_TYPED_JSON_DECODE_FAILED") from exc
    if not isinstance(value, dict):
        raise RuntimeError("MOTION_V2_TYPED_JSON_OBJECT_REQUIRED")
    return value


def _json_bytes(value: Any) -> bytes:
    payload = value.to_dict() if hasattr(value, "to_dict") else value
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")


def execute_motion_capability_v2(
    request: dict[str, Any],
    *,
    read_object: Callable[[dict[str, Any]], bytes],
    put_bytes: Callable[[bytes], dict[str, Any]],
) -> dict[str, Any]:
    metadata = dict(request.get("capability_metadata") or {})
    if metadata.get("executor_contract") != MOTION_EXECUTOR_CONTRACT_V2:
        raise RuntimeError("MOTION_V2_EXECUTOR_CONTRACT_DRIFT")
    if str(request.get("kind") or "") != "MODULE":
        raise RuntimeError("MOTION_V2_CAPABILITY_KIND_DRIFT")
    if request.get("model_releases"):
        raise RuntimeError("MOTION_V2_LEARNED_MODEL_RELEASE_FORBIDDEN")

    from compiler.realsas_compiler_core.canonical_mechanics_ir_v1 import (
        canonical_mechanics_from_dict_v1,
    )
    from compiler.realsas_compiler_core.canonical_motion_ir_v2 import mint_canonical_motion_v2
    from compiler.realsas_compiler_core.canonical_rig_ir_v2 import canonical_rig_from_dict_v2
    from compiler.realsas_compiler_core.control_motion_ir_v1 import (
        control_motion_proposal_from_dict_v1,
        qualify_control_motion_v1,
    )
    from compiler.realsas_compiler_core.default_idle_provider_v1 import (
        DEFAULT_IDLE_PROVIDER_V1,
        build_default_idle_proposal_v1,
    )

    rig = canonical_rig_from_dict_v2(
        _json_payload(read_object, _one(request, "RealSaS.CanonicalRigIR", "v2"))
    )
    mechanics = canonical_mechanics_from_dict_v1(
        _json_payload(read_object, _one(request, "RealSaS.CanonicalMechanicsIR", "v1"))
    )
    proposal_ref = _optional_one(
        request, "RealSaS.ControlMotionClipProposalIR", "v1"
    )
    if proposal_ref is None:
        proposal = build_default_idle_proposal_v1(rig, mechanics)
        provider = DEFAULT_IDLE_PROVIDER_V1
        proposal_source = "PRODUCT_DEFAULT_PROVIDER"
    else:
        proposal = control_motion_proposal_from_dict_v1(
            _json_payload(read_object, proposal_ref)
        )
        provider = str(proposal.metadata.get("provider") or "EXTERNAL_CONTROL_MOTION_PROVIDER")
        proposal_source = "EXTERNAL_ARTIFACT"

    qualified = qualify_control_motion_v1(proposal, rig=rig, mechanics=mechanics)
    motion = mint_canonical_motion_v2(qualified)
    sealed = put_bytes(_json_bytes(motion))
    output = {
        "role": "canonical_motion",
        "artifact_type": "RealSaS.CanonicalMotionIR",
        "schema_version": "v2",
        "storage_key": sealed["storage_key"],
        "content_sha256": sealed["content_sha256"],
        "size_bytes": sealed["size_bytes"],
        "authority_class": "COMPILER_CANONICAL_MOTION",
    }
    return {
        "capability_id": str(request.get("capability_id") or ""),
        "status": "PASS",
        "outputs": [output],
        "diagnostics": {
            "schema": "RealSaS.ProductMotionDiagnostics.v2",
            "motion_lineage_hash": motion.motion_lineage_hash,
            "rig_lineage_hash": motion.rig_lineage_hash,
            "control_program_hash": motion.control_program_hash,
            "mechanics_lineage_hash": motion.mechanics_lineage_hash,
            "track_count": len(motion.control_tracks),
            "intent": motion.intent,
            "motion_provider": provider,
            "proposal_source": proposal_source,
            "default_idle_provider_used": proposal_ref is None,
            "editable_motion_space": "ANIMATOR_CONTROLS",
            "direct_joint_track_authority": False,
            "constraint_solver_executed_by_compiler": False,
            "compiler_retargeting_performed": False,
            "product_revision_pass_claimed": False,
        },
        "failure": None,
    }


__all__ = ["MOTION_EXECUTOR_CONTRACT_V2", "execute_motion_capability_v2"]
