from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping

from .hashing import content_sha256
from .types import QualificationError

Json = dict[str, Any]
ALLOWED_CAA_BACKENDS = {"DETERMINISTIC_V1", "LEARNED_V2", "IM2SURFTEX_RESEARCH_ONLY"}


@dataclass(frozen=True)
class CAACompilePreregistrationIR:
    backend_id: str
    shipping_eligible: bool
    contract_sha256: str
    observation_set_binding_hash: str
    output_direction_set_binding_hash: str
    candidate_mesh_binding_hash: str
    surface_addressing_binding_hash: str
    appearance_domain_binding_hash: str
    static_mesh_qualification_binding_hash: str
    source_lock_policy: Json
    completion_quality_policy: Json
    preregistration_hash: str
    schema_version: str = "RealSaS.CAACompilePreregistrationIR.v1"
    metadata: Json = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def caa_preregistration_hash(value: CAACompilePreregistrationIR) -> str:
    payload = value.to_dict()
    payload.pop("preregistration_hash", None)
    return content_sha256(payload)


def build_caa_preregistration(
    *,
    backend_id: str,
    contract_sha256: str,
    observation_set_hash: str,
    output_direction_set_hash: str,
    candidate_mesh_hash: str,
    surface_addressing_hash: str,
    appearance_domain_hash: str,
    static_mesh_qualification_hash: str,
    source_lock_policy: Json,
    completion_quality_policy: Json,
) -> CAACompilePreregistrationIR:
    backend = str(backend_id)
    if backend not in ALLOWED_CAA_BACKENDS:
        raise QualificationError("CAA_BACKEND_UNSUPPORTED")
    value = CAACompilePreregistrationIR(
        backend_id=backend,
        shipping_eligible=backend != "IM2SURFTEX_RESEARCH_ONLY",
        contract_sha256=str(contract_sha256),
        observation_set_binding_hash=str(observation_set_hash),
        output_direction_set_binding_hash=str(output_direction_set_hash),
        candidate_mesh_binding_hash=str(candidate_mesh_hash),
        surface_addressing_binding_hash=str(surface_addressing_hash),
        appearance_domain_binding_hash=str(appearance_domain_hash),
        static_mesh_qualification_binding_hash=str(static_mesh_qualification_hash),
        source_lock_policy=dict(source_lock_policy),
        completion_quality_policy=dict(completion_quality_policy),
        preregistration_hash="",
        metadata={
            "source_wins": True,
            "runtime_generation_forbidden": True,
            "presentation_warp_v1_forbidden": True,
            "canonical_internal_alpha": "PREMULTIPLIED",
            "im2surftex_research_only": backend == "IM2SURFTEX_RESEARCH_ONLY",
        },
    )
    return replace(value, preregistration_hash=caa_preregistration_hash(value))


def caa_preregistration_from_dict(payload: Mapping[str, Any]) -> CAACompilePreregistrationIR:
    value = CAACompilePreregistrationIR(
        str(payload["backend_id"]),
        bool(payload["shipping_eligible"]),
        str(payload["contract_sha256"]),
        str(payload["observation_set_binding_hash"]),
        str(payload["output_direction_set_binding_hash"]),
        str(payload["candidate_mesh_binding_hash"]),
        str(payload["surface_addressing_binding_hash"]),
        str(payload["appearance_domain_binding_hash"]),
        str(payload["static_mesh_qualification_binding_hash"]),
        dict(payload.get("source_lock_policy") or {}),
        dict(payload.get("completion_quality_policy") or {}),
        str(payload["preregistration_hash"]),
        schema_version=str(payload.get("schema_version") or "RealSaS.CAACompilePreregistrationIR.v1"),
        metadata=dict(payload.get("metadata") or {}),
    )
    if value.backend_id not in ALLOWED_CAA_BACKENDS:
        raise QualificationError("CAA_BACKEND_UNSUPPORTED")
    if value.preregistration_hash != caa_preregistration_hash(value):
        raise QualificationError("CAA_PREREG_HASH_MISMATCH")
    return value
