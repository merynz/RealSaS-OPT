from __future__ import annotations

from dataclasses import replace
import math

from compiler.realsas_compiler_core.output_presentation_v1 import (
    build_output_direction_set,
    output_direction_set_from_dict,
)
from compiler.realsas_compiler_core.preproduct_authority_v1 import (
    rest_reprojection_geometry_gate_from_dict,
)
from compiler.realsas_compiler_core.product_artifact_codec_v1 import (
    canonical_mesh_candidate_from_dict,
    mechanical_partition_from_dict,
    qualified_camera_set_from_dict,
    read_json,
)
from compiler.realsas_compiler_core.surface_addressing_v1 import (
    StaticCanonicalMeshQualificationIR,
    appearance_domain_from_dict,
    build_appearance_domain,
    build_surface_addressing,
    static_mesh_qualification_hash,
    surface_addressing_from_dict,
)
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_services.orchestrator.adapters.product_mesh_v1 import (
    _stage_output_payload,
    _write_ir,
    build_canonical_mesh_candidate_stage,
)


def seal_output_presentation_directions_stage(ctx: dict) -> dict:
    cameras = qualified_camera_set_from_dict(
        _stage_output_payload(
            ctx, "05_CAMERA_CONTRACT_SOLVED", "RealSaS.QualifiedCameraSetIR.v1"
        )
    )
    value = build_output_direction_set(cameras)
    root = ctx["run_root"] / "artifacts" / ctx["stage"]["id"]
    return {
        "status": "PASS",
        "outputs": [
            _write_ir(
                root / "output_presentation_directions.json",
                value,
                authority_class="OUTPUT_PRESENTATION_DIRECTION_AUTHORITY",
            )
        ],
        "diagnostics": {
            "direction_set_hash": value.direction_set_hash,
            "output_direction_count": 8,
            "source_observation_authority_separate": True,
        },
    }


def build_canonical_mesh_addressing_stage(ctx: dict) -> dict:
    base = build_canonical_mesh_candidate_stage(ctx)
    if str(base.get("status")) != "PASS":
        return base
    by_schema = {str(o.get("schema")): o for o in base.get("outputs") or ()}
    candidate = canonical_mesh_candidate_from_dict(
        read_json(by_schema["RealSaS.CanonicalMeshCandidateIR.v1"]["path"])
    )
    directions = output_direction_set_from_dict(
        _stage_output_payload(
            ctx,
            "16_OUTPUT_PRESENTATION_DIRECTIONS_SEALED",
            "RealSaS.OutputPresentationDirectionSetIR.v1",
        )
    )
    addressing = build_surface_addressing(candidate)
    domain = build_appearance_domain(
        candidate, addressing, directions.direction_set_hash
    )
    root = ctx["run_root"] / "artifacts" / ctx["stage"]["id"]
    outputs = list(base["outputs"]) + [
        _write_ir(
            root / "surface_addressing.json",
            addressing,
            authority_class="CANONICAL_SURFACE_ADDRESSING",
        ),
        _write_ir(
            root / "appearance_domain.json",
            domain,
            authority_class="APPEARANCE_DOMAIN",
        ),
    ]
    diagnostics = dict(base.get("diagnostics") or {})
    diagnostics.update(
        {
            "surface_addressing_hash": addressing.addressing_hash,
            "appearance_domain_hash": domain.domain_hash,
            "appearance_domain_face_count": domain.renderable_face_count,
        }
    )
    return {"status": "PASS", "outputs": outputs, "diagnostics": diagnostics}


def _double_area(a, b, c) -> float:
    ux, uy, uz = (b[i] - a[i] for i in range(3))
    vx, vy, vz = (c[i] - a[i] for i in range(3))
    cx = uy * vz - uz * vy
    cy = uz * vx - ux * vz
    cz = ux * vy - uy * vx
    return math.sqrt(cx * cx + cy * cy + cz * cz)


def qualify_static_canonical_mesh_stage(ctx: dict) -> dict:
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
    geometry = rest_reprojection_geometry_gate_from_dict(
        _stage_output_payload(
            ctx,
            "13_REST_REPROJECTION_GEOMETRY_GATE",
            "RealSaS.RestReprojectionGeometryGateIR.v1",
        )
    )
    partition = mechanical_partition_from_dict(
        _stage_output_payload(
            ctx,
            "17_MECHANICAL_PARTITION_QUALIFIED",
            "RealSaS.MechanicalPartitionIR.v1",
        )
    )
    if (
        addressing.candidate_mesh_binding_hash != candidate.candidate_lineage_hash
        or domain.candidate_mesh_binding_hash != candidate.candidate_lineage_hash
    ):
        raise QualificationError("STATIC_MESH_DOMAIN_BINDING_DRIFT")
    if domain.surface_addressing_binding_hash != addressing.addressing_hash:
        raise QualificationError("STATIC_MESH_ADDRESSING_DOMAIN_DRIFT")

    vertices = {
        str(v.candidate_vertex_id): tuple(map(float, v.P)) for v in candidate.vertices
    }
    if not vertices or not candidate.faces:
        raise QualificationError("STATIC_MESH_EMPTY")
    if any(not all(math.isfinite(x) for x in p) for p in vertices.values()):
        raise QualificationError("STATIC_MESH_NONFINITE")

    degenerate = 0
    for face in candidate.faces:
        a, b, c = (vertices[str(x)] for x in face)
        if _double_area(a, b, c) <= 1e-12:
            degenerate += 1
    if degenerate:
        raise QualificationError(f"STATIC_MESH_DEGENERATE_FACE_COUNT:{degenerate}")
    if len(addressing.vertex_addresses) != len(candidate.vertices):
        raise QualificationError("STATIC_MESH_VERTEX_ADDRESSABILITY_INCOMPLETE")
    if len(addressing.face_address_ids) != len(candidate.faces):
        raise QualificationError("STATIC_MESH_FACE_ADDRESSABILITY_INCOMPLETE")
    if not all(view.passed for view in geometry.views):
        raise QualificationError("STATIC_MESH_GEOMETRY_GATE_NOT_PASS")

    report = {
        "status": "PASS_STATIC_CANONICAL_CARRIER",
        "vertex_count": len(candidate.vertices),
        "face_count": len(candidate.faces),
        "degenerate_face_count": 0,
        "surface_addressability_fraction": 1.0,
        "stage13_geometry_gate_inherited": True,
        "silhouette_is_geometry_authority_not_caa": True,
        "unknown_boundary_policy": (
            "CONSERVATIVE_UNTIL_STAGE35__NO_POST_SKIN_PARTITION_MUTATION_V1"
        ),
    }
    value = StaticCanonicalMeshQualificationIR(
        candidate_mesh_binding_hash=candidate.candidate_lineage_hash,
        surface_addressing_binding_hash=addressing.addressing_hash,
        appearance_domain_binding_hash=domain.domain_hash,
        geometry_gate_binding_hash=geometry.geometry_gate_hash,
        partition_binding_hash=partition.partition_lineage_hash,
        qualification_report=report,
        qualification_hash="",
        metadata={
            "mesh_frozen_for_branching": True,
            "repair_on_stage35_failure": "NEW_STAGE18_LINEAGE",
        },
    )
    value = replace(value, qualification_hash=static_mesh_qualification_hash(value))
    root = ctx["run_root"] / "artifacts" / ctx["stage"]["id"]
    return {
        "status": "PASS",
        "outputs": [
            _write_ir(
                root / "static_canonical_mesh_qualification.json",
                value,
                authority_class="STATIC_CANONICAL_MESH_QUALIFICATION",
            )
        ],
        "diagnostics": {"qualification_hash": value.qualification_hash, **report},
    }
