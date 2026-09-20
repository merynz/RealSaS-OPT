from __future__ import annotations

from pathlib import Path

from compiler.realsas_compiler_core.complete_appearance_authority_v1 import (
    build_caa_preregistration,
)
from compiler.realsas_compiler_core.output_presentation_v1 import (
    output_direction_set_from_dict,
)
from compiler.realsas_compiler_core.product_artifact_codec_v1 import (
    canonical_mesh_candidate_from_dict,
)
from compiler.realsas_compiler_core.surface_addressing_v1 import (
    appearance_domain_from_dict,
    static_mesh_qualification_from_dict,
    surface_addressing_from_dict,
)
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_services.orchestrator.adapters.product_mesh_v1 import (
    _sha256,
    _stage_output_payload,
    _write_ir,
)


def preregister_caa_backend_stage(ctx: dict) -> dict:
    cfg = dict(ctx["run_manifest"].get("appearance") or {})
    backend = str(cfg.get("backend") or "")
    if not backend:
        return {
            "status": "BLOCKED",
            "blockers": ["CAA_BACKEND_MUST_BE_EXPLICIT"],
            "diagnostics": {},
        }

    contract_path = (
        Path(ctx["repo_root"])
        / "canonical"
        / "COMPLETE_APPEARANCE_AUTHORITY_V1_20260920.json"
    )
    if not contract_path.is_file():
        raise QualificationError("CAA_CANONICAL_CONTRACT_MISSING")
    contract_sha = _sha256(contract_path)

    observation_payload = _stage_output_payload(
        ctx,
        "07_OBSERVATION_CONTRACT_QUALIFIED",
        "RealSaS.QualifiedObservationSetIR.v1",
    )
    directions = output_direction_set_from_dict(
        _stage_output_payload(
            ctx,
            "16_OUTPUT_PRESENTATION_DIRECTIONS_SEALED",
            "RealSaS.OutputPresentationDirectionSetIR.v1",
        )
    )
    candidate = canonical_mesh_candidate_from_dict(
        _stage_output_payload(
            ctx,
            "18_CANONICAL_MESH_ADDRESSING_BUILD",
            "RealSaS.CanonicalMeshCandidateIR.v1",
        )
    )
    addressing = surface_addressing_from_dict(
        _stage_output_payload(
            ctx,
            "18_CANONICAL_MESH_ADDRESSING_BUILD",
            "RealSaS.SurfaceAddressingIR.v1",
        )
    )
    domain = appearance_domain_from_dict(
        _stage_output_payload(
            ctx,
            "18_CANONICAL_MESH_ADDRESSING_BUILD",
            "RealSaS.AppearanceDomainIR.v1",
        )
    )
    static_mesh = static_mesh_qualification_from_dict(
        _stage_output_payload(
            ctx,
            "19_STATIC_CANONICAL_MESH_QUALIFIED",
            "RealSaS.StaticCanonicalMeshQualificationIR.v1",
        )
    )

    prereg = build_caa_preregistration(
        backend_id=backend,
        contract_sha256=contract_sha,
        observation_set_hash=str(observation_payload["observation_set_hash"]),
        output_direction_set_hash=directions.direction_set_hash,
        candidate_mesh_hash=candidate.candidate_lineage_hash,
        surface_addressing_hash=addressing.addressing_hash,
        appearance_domain_hash=domain.domain_hash,
        static_mesh_qualification_hash=static_mesh.qualification_hash,
        source_lock_policy=dict(cfg.get("source_lock_policy") or {}),
        completion_quality_policy=dict(cfg.get("completion_quality_policy") or {}),
    )
    root = ctx["run_root"] / "artifacts" / ctx["stage"]["id"]
    return {
        "status": "PASS",
        "outputs": [
            _write_ir(
                root / "caa_backend_preregistration.json",
                prereg,
                authority_class="CAA_BACKEND_PREREGISTRATION",
            )
        ],
        "diagnostics": {
            "backend_id": backend,
            "shipping_eligible": prereg.shipping_eligible,
            "contract_sha256": contract_sha,
            "preregistration_hash": prereg.preregistration_hash,
        },
    }
