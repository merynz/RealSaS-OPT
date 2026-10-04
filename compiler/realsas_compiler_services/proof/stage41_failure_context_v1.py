from __future__ import annotations

"""Exact Stage41 failure context for controlled causal attribution.

This service binds one failed exact-motion measurement to the exact evaluator,
policy and product lineages that produced it. It does not infer a causal owner
and does not authorize repair.
"""

from typing import Any, Mapping, Sequence

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_services.proof.causal_attribution import (
    proof_probe_fingerprint_v1,
)


def build_stage41_failure_attribution_context_v1(
    *,
    measurements: Mapping[str, Any],
    bindings: Mapping[str, str],
    mesh_policy_hash: str,
    camera_binding_hashes: Sequence[str],
    observation_set_hash: str,
    evaluator_semantic_version: str,
) -> dict[str, Any]:
    bindings = {str(k): str(v) for k, v in dict(bindings).items()}
    required = {
        "mechanical_state_binding_hash",
        "skeleton_binding_hash",
        "mesh_binding_hash",
        "mesh_skin_binding_hash",
        "qualified_motion_binding_hash",
        "constraint_set_binding_hash",
        "presentation_binding_hash",
        "mechanical_carrier_evidence_hash",
        "mechanical_carrier_topology_hash",
        "mechanical_carrier_geometry_hash",
        "static_mesh_qualification_binding_hash",
    }
    missing = sorted(k for k in required if not bindings.get(k))
    if missing:
        raise ValueError(
            "STAGE41_ATTRIBUTION_BINDING_MISSING:" + ",".join(missing)
        )
    if not mesh_policy_hash:
        raise ValueError("STAGE41_ATTRIBUTION_POLICY_HASH_MISSING")
    cameras = tuple(map(str, camera_binding_hashes))
    if len(cameras) != 8 or any(not x for x in cameras):
        raise ValueError("STAGE41_ATTRIBUTION_CAMERA_BINDING_INVALID")
    if not observation_set_hash:
        raise ValueError("STAGE41_ATTRIBUTION_OBSERVATION_HASH_MISSING")
    if not evaluator_semantic_version:
        raise ValueError("STAGE41_ATTRIBUTION_EVALUATOR_VERSION_MISSING")

    probe_specification = {
        "schema": "RealSaS.Stage41ExactMotionProbeSpecification.v1",
        "proof_domain": "MOTION",
        "evaluator_semantic_version": str(evaluator_semantic_version),
        "sampling_contract": "UNIFORM17_PLUS_KEYFRAMES_PLUS_CONTACT_ENDPOINTS",
        "mechanical_state_binding_hash": bindings[
            "mechanical_state_binding_hash"
        ],
        "skeleton_binding_hash": bindings["skeleton_binding_hash"],
        "mesh_binding_hash": bindings["mesh_binding_hash"],
        "mesh_skin_binding_hash": bindings["mesh_skin_binding_hash"],
        "qualified_motion_binding_hash": bindings[
            "qualified_motion_binding_hash"
        ],
        "constraint_set_binding_hash": bindings[
            "constraint_set_binding_hash"
        ],
        "presentation_binding_hash": bindings[
            "presentation_binding_hash"
        ],
        "mechanical_carrier_evidence_hash": bindings[
            "mechanical_carrier_evidence_hash"
        ],
        "mechanical_carrier_topology_hash": bindings[
            "mechanical_carrier_topology_hash"
        ],
        "mechanical_carrier_geometry_hash": bindings[
            "mechanical_carrier_geometry_hash"
        ],
        "static_mesh_qualification_binding_hash": bindings[
            "static_mesh_qualification_binding_hash"
        ],
        "camera_binding_hashes": list(cameras),
        "observation_set_hash": str(observation_set_hash),
    }
    operator_policy_hashes = (
        str(mesh_policy_hash),
        str(evaluator_semantic_version),
    )
    probe_fingerprint = proof_probe_fingerprint_v1(
        operator_policy_hashes=operator_policy_hashes,
        probe_specification=probe_specification,
    )
    measurement_payload = {
        "schema": "RealSaS.Stage41ExactMotionFailureMeasurement.v1",
        "proof_domain": "MOTION",
        "measurements": dict(measurements),
        "bindings": bindings,
        "mesh_policy_hash": str(mesh_policy_hash),
        "probe_fingerprint": probe_fingerprint,
    }
    return {
        "schema": "RealSaS.Stage41FailureAttributionContext.v1",
        "proof_domain": "MOTION",
        "baseline_measurement_report_hash": content_sha256(
            measurement_payload
        ),
        "proof_probe_fingerprint": probe_fingerprint,
        "operator_policy_hashes": list(operator_policy_hashes),
        "probe_specification": probe_specification,
        "bindings": bindings,
        "causal_owner_attribution": "NOT_PERFORMED",
        "automatic_repair_eligible": False,
        "same_probe_counterfactual_required": True,
        "failure_signature_alone_is_not_owner_evidence": True,
    }


__all__ = ["build_stage41_failure_attribution_context_v1"]
