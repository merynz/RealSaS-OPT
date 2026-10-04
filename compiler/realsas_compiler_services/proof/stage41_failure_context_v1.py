from __future__ import annotations

"""Exact Stage41 probe/state identity for controlled causal attribution.

A causal court needs two different identities:
- the immutable evaluation protocol ("same probe");
- the mutable product/mechanical state being intervened on.

Conflating them makes real single-owner counterfactuals impossible, because an
owner mutation necessarily changes one or more product lineage hashes.
"""

from typing import Any, Mapping, Sequence

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_services.proof.causal_attribution import (
    proof_probe_fingerprint_v1,
)


_REQUIRED_BINDINGS = {
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
    "motion_source_set_binding_hash",
    "motion_source_seal_binding_hash",
}


def build_stage41_probe_receipt_v1(
    *,
    bindings: Mapping[str, str],
    mesh_policy_hash: str,
    camera_binding_hashes: Sequence[str],
    observation_set_hash: str,
    evaluator_semantic_version: str,
) -> dict[str, Any]:
    bindings = {str(k): str(v) for k, v in dict(bindings).items()}
    missing = sorted(k for k in _REQUIRED_BINDINGS if not bindings.get(k))
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
        "schema": "RealSaS.Stage41ExactMotionProbeSpecification.v2",
        "proof_domain": "MOTION",
        "evaluator_semantic_version": str(evaluator_semantic_version),
        "sampling_contract": "UNIFORM17_PLUS_KEYFRAMES_PLUS_CONTACT_ENDPOINTS",
        "motion_source_set_binding_hash": bindings[
            "motion_source_set_binding_hash"
        ],
        "motion_source_seal_binding_hash": bindings[
            "motion_source_seal_binding_hash"
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
    product_bindings_hash = content_sha256(
        {
            "schema": "RealSaS.Stage41ProductBindings.v1",
            "bindings": bindings,
        }
    )
    return {
        "schema": "RealSaS.Stage41ProbeReceipt.v1",
        "proof_domain": "MOTION",
        "proof_probe_fingerprint": probe_fingerprint,
        "operator_policy_hashes": list(operator_policy_hashes),
        "probe_specification": probe_specification,
        "bindings": bindings,
        "product_bindings_hash": product_bindings_hash,
        "probe_identity_excludes_mutable_product_bindings": True,
    }


def build_stage41_failure_attribution_context_v1(
    *,
    measurements: Mapping[str, Any],
    bindings: Mapping[str, str],
    mesh_policy_hash: str,
    camera_binding_hashes: Sequence[str],
    observation_set_hash: str,
    evaluator_semantic_version: str,
) -> dict[str, Any]:
    receipt = build_stage41_probe_receipt_v1(
        bindings=bindings,
        mesh_policy_hash=mesh_policy_hash,
        camera_binding_hashes=camera_binding_hashes,
        observation_set_hash=observation_set_hash,
        evaluator_semantic_version=evaluator_semantic_version,
    )
    measurement_payload = {
        "schema": "RealSaS.Stage41ExactMotionFailureMeasurement.v1",
        "proof_domain": "MOTION",
        "measurements": dict(measurements),
        "bindings": dict(receipt["bindings"]),
        "mesh_policy_hash": str(mesh_policy_hash),
        "probe_fingerprint": receipt["proof_probe_fingerprint"],
    }
    return {
        "schema": "RealSaS.Stage41FailureAttributionContext.v1",
        "proof_domain": "MOTION",
        "baseline_measurement_report_hash": content_sha256(
            measurement_payload
        ),
        "proof_probe_fingerprint": receipt["proof_probe_fingerprint"],
        "operator_policy_hashes": list(receipt["operator_policy_hashes"]),
        "probe_specification": dict(receipt["probe_specification"]),
        "bindings": dict(receipt["bindings"]),
        "baseline_product_bindings_hash": receipt["product_bindings_hash"],
        "probe_receipt": receipt,
        "probe_identity_excludes_mutable_product_bindings": True,
        "causal_owner_attribution": "NOT_PERFORMED",
        "automatic_repair_eligible": False,
        "same_probe_counterfactual_required": True,
        "failure_signature_alone_is_not_owner_evidence": True,
    }


__all__ = [
    "build_stage41_probe_receipt_v1",
    "build_stage41_failure_attribution_context_v1",
]
