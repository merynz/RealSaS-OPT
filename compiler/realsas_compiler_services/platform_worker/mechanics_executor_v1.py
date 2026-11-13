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
    outputs = [output]
    intervention = None
    intervention_policy = metadata.get("control_intervention_policy")
    if intervention_policy is not None:
        if (
            type(intervention_policy) is not dict
            or set(intervention_policy) != {
                "enabled", "max_probed_controls", "max_full_mesh_deformation_probes",
                "relative_translation", "max_constraint_residual",
            }
            or type(intervention_policy["enabled"]) is not bool
        ):
            raise RuntimeError("MECHANICS_INTERVENTION_RELEASE_POLICY_DRIFT")
        if intervention_policy["enabled"]:
            from compiler.realsas_compiler_core.mechanics_intervention_court_v1 import (
                compile_mechanics_intervention_court_v1,
            )
            intervention = compile_mechanics_intervention_court_v1(
                mesh, rig, skin, mechanics,
                max_probed_controls=intervention_policy["max_probed_controls"],
                max_full_mesh_deformation_probes=intervention_policy["max_full_mesh_deformation_probes"],
                relative_translation=intervention_policy["relative_translation"],
                max_constraint_residual=intervention_policy["max_constraint_residual"],
            )
            data = _json_bytes(intervention)
            sealed_intervention = put_bytes(data)
            from hashlib import sha256
            if (
                sealed_intervention.get("content_sha256") != sha256(data).hexdigest()
                or sealed_intervention.get("size_bytes") != len(data)
                or not sealed_intervention.get("storage_key")
            ):
                raise RuntimeError("MECHANICS_INTERVENTION_CAS_WRITE_DRIFT")
            outputs.append({
                "role": "mechanics_intervention_court",
                "artifact_type": "RealSaS.MechanicsInterventionCourt",
                "schema_version": "v1",
                "storage_key": sealed_intervention["storage_key"],
                "content_sha256": sealed_intervention["content_sha256"],
                "size_bytes": sealed_intervention["size_bytes"],
                "authority_class": "COMPILER_INTERVENTION_NOT_SOURCE_CONTACT_OR_PROOF",
            })
    # Dynamic geometry is a distinct, independently replayable court;
    # static LBS success never implies a moving mesh is collision-safe.
    integrity = None
    integrity_policy = metadata.get("dynamic_integrity_policy")
    if integrity_policy is not None:
        expected_policy = {
            "enabled", "max_faces", "max_controls",
            "relative_translation", "max_constraint_residual",
        }
        if (
            type(integrity_policy) is not dict
            or set(integrity_policy) != expected_policy
            or type(integrity_policy["enabled"]) is not bool
            or type(integrity_policy["max_faces"]) is not int
            or type(integrity_policy["max_controls"]) is not int
            or type(integrity_policy["relative_translation"]) not in (float, int)
            or type(integrity_policy["max_constraint_residual"]) not in (float, int)
            or integrity_policy["max_faces"] != 1024
            or integrity_policy["max_controls"] != 128
            or float(integrity_policy["relative_translation"]) != .02
            or float(integrity_policy["max_constraint_residual"]) != .001
        ):
            raise RuntimeError("MECHANICS_DYNAMIC_INTEGRITY_POLICY_DRIFT")
        if integrity_policy["enabled"]:
            from hashlib import sha256
            from compiler.realsas_compiler_core.mechanics_dynamic_integrity_court_v1 import (
                compile_mechanics_dynamic_integrity_court_v1,
            )
            integrity = compile_mechanics_dynamic_integrity_court_v1(
                mesh, rig, skin, mechanics,
                max_faces=integrity_policy["max_faces"],
                max_controls=integrity_policy["max_controls"],
                relative_translation=integrity_policy["relative_translation"],
                max_constraint_residual=integrity_policy["max_constraint_residual"],
            )
            raw = _json_bytes(integrity)
            ref = put_bytes(raw)
            if (
                ref.get("content_sha256") != sha256(raw).hexdigest()
                or ref.get("size_bytes") != len(raw)
                or not ref.get("storage_key")
            ):
                raise RuntimeError("MECHANICS_DYNAMIC_INTEGRITY_CAS_WRITE_DRIFT")
            outputs.append({
                "role": "mechanics_dynamic_integrity_court",
                "artifact_type": "RealSaS.MechanicsDynamicIntegrityCourt",
                "schema_version": "v1",
                "storage_key": ref["storage_key"],
                "content_sha256": ref["content_sha256"],
                "size_bytes": ref["size_bytes"],
                "authority_class": "MEASURED_NONREST_GEOMETRY_NO_SOURCE_CONTACT_CLAIM",
            })
    return {
        "capability_id": str(request.get("capability_id") or ""),
        "status": "PASS",
        "outputs": outputs,
        "diagnostics": {
            "schema": "RealSaS.ProductMechanicsDiagnostics.v1",
            "mechanics_lineage_hash": mechanics.mechanics_lineage_hash,
            "rig_lineage_hash": mechanics.rig_lineage_hash,
            "skin_lineage_hash": mechanics.skin_lineage_hash,
            "constraint_schedule_count": len(mechanics.constraint_schedule),
            "intervention_court_hash": (
                intervention.court_hash if intervention is not None else None
            ),
            "non_rest_control_probes_executed": (
                intervention.control_count if intervention is not None else 0
            ),
            "attachment_socket_pose_count": (
                intervention.anchor_count if intervention is not None else 0
            ),
            "full_mesh_deformation_probe_count": (
                intervention.full_mesh_deformation_probe_count
                if intervention is not None else 0
            ),
            "full_mesh_deformation_complete": (
                intervention.full_mesh_deformation_complete
                if intervention is not None else False
            ),
            "dynamic_integrity_court_hash": (
                integrity.court_hash if integrity is not None else None
            ),
            "dynamic_integrity_complete": bool(
                integrity is not None
                and integrity.complete_face_and_control_coverage
            ),
            "dynamic_new_self_intersection_count": (
                integrity.maximum_new_intersection_count
                if integrity is not None else None
            ),
            "geometric_collision_nonregression_qualified": bool(
                integrity is not None
                and integrity.complete_face_and_control_coverage
                and integrity.no_new_self_intersections
            ),
            "source_contact_qualified": False,
            "learned_secondary_motion_qualified": False,
            "edit_consequence_qualified": False,
            "learned_inference_performed": False,
            "rig_program_mutated": False,
            "product_revision_pass_claimed": False,
        },
        "failure": None,
    }


__all__ = ["MECHANICS_EXECUTOR_CONTRACT_V1", "execute_mechanics_capability_v1"]
