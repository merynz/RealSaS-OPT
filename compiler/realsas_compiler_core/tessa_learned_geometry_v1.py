from __future__ import annotations

"""Typed Compiler contract for TESSA learned mechanical geometry.

TESSA learned vertex positions and GSA material/mechanical support are two
separate authorities:

* learned ``P`` / topology is proposal geometry and can become carrier geometry
  only after the exact Stage19 candidate passes source-fidelity and static
  quality qualification;
* ``TESSAMaterialSupportFieldIR`` is component-safe continuous field lineage for
  mechanics/skin correspondence.  Its coefficients are deliberately *not* a
  claim that learned ``P`` equals the convex GSA support lift.

The support mode introduced here is intentionally unknown to the legacy
QualifiedMesh G1 positional-support validator.  Therefore a TESSA candidate can
be measured at Stage19 but cannot accidentally enter the legacy Stage35 product
path until a hash-bound learned-geometry admission is explicitly consumed.
"""

from dataclasses import asdict, dataclass, field, replace
import math
from typing import Any

import numpy as np

from .hashing import content_sha256
from .product_authority_v1 import (
    CanonicalMeshCandidateIR,
    CanonicalMeshVertexCandidateIR,
    ComponentCarrierPolicyIR,
    MechanicalPartitionIR,
    canonical_mesh_candidate_lineage_hash,
    validate_component_carrier_policy,
    validate_mechanical_partition,
)
from .surface_addressing_v1 import (
    StaticCanonicalMeshQualificationIR,
    static_mesh_qualification_hash,
)
from .tessa_surface_field_v1 import (
    TESSAMaterialSupportFieldIR,
    validate_tessa_material_support_field_v1,
)
from .types import QualificationError, RiggingSurfaceIR, SurfaceSupportBinding

Json = dict[str, Any]

TESSA_FIELD_SUPPORT_MODE_V1 = "TESSA_MATERIAL_FIELD_SUPPORT_ONLY_V1"
TESSA_CANDIDATE_PRODUCER_V1 = "RealSaS.TESSA.LearnedGeometryCandidate.v1"
TESSA_GEOMETRY_AUTHORITY_PENDING_V1 = "PENDING_STAGE19_ACTUAL_CANDIDATE_QUALIFICATION"
TESSA_GEOMETRY_AUTHORITY_QUALIFIED_V1 = (
    "STAGE19_ACTUAL_CANDIDATE_SOURCE_FIDELITY_PLUS_STATIC_QUALITY"
)
TESSA_FIELD_AUTHORITY_V1 = "COMPONENT_SAFE_GSA_CONTINUOUS_FIELD_LINEAGE_V1"


@dataclass(frozen=True)
class TESSALearnedGeometryAdmissionIR:
    candidate_mesh_binding_hash: str
    static_qualification_binding_hash: str
    material_support_field_binding_hash: str
    surface_binding_hash: str
    partition_binding_hash: str
    carrier_policy_binding_hash: str
    topology_sequence_hash: str
    geometry_authority: str
    field_support_authority: str
    admission_hash: str
    schema_version: str = "RealSaS.TESSALearnedGeometryAdmissionIR.v1"
    metadata: Json = field(default_factory=dict)

    def to_dict(self) -> Json:
        return asdict(self)


def tessa_learned_geometry_admission_hash(
    value: TESSALearnedGeometryAdmissionIR,
) -> str:
    payload = value.to_dict()
    payload.pop("admission_hash", None)
    return content_sha256(payload)


def _edge(a: str, b: str) -> tuple[str, str]:
    a = str(a)
    b = str(b)
    if not a or not b or a == b:
        raise QualificationError("TESSA_LEARNED_GEOMETRY_EDGE_INVALID")
    return (a, b) if a < b else (b, a)


def _validate_world_vertices(vertices_world: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices_world, dtype=np.float64)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or len(vertices) < 3
        or not np.isfinite(vertices).all()
    ):
        raise QualificationError("TESSA_LEARNED_GEOMETRY_VERTEX_ARRAY_INVALID")
    return vertices


def _validate_faces(faces: np.ndarray, *, vertex_count: int) -> np.ndarray:
    value = np.asarray(faces, dtype=np.int64)
    if value.ndim != 2 or value.shape[1] != 3 or len(value) < 1:
        raise QualificationError("TESSA_LEARNED_GEOMETRY_FACE_ARRAY_INVALID")
    if np.any(value < 0) or np.any(value >= int(vertex_count)):
        raise QualificationError("TESSA_LEARNED_GEOMETRY_FACE_INDEX_INVALID")
    if any(len(set(map(int, row))) != 3 for row in value.tolist()):
        raise QualificationError("TESSA_LEARNED_GEOMETRY_DEGENERATE_INDEX_FACE")
    keys = [tuple(sorted(map(int, row))) for row in value.tolist()]
    if len(keys) != len(set(keys)):
        raise QualificationError("TESSA_LEARNED_GEOMETRY_DUPLICATE_FACE")
    return value


def build_tessa_learned_geometry_candidate_v1(
    *,
    vertices_world: np.ndarray,
    faces: np.ndarray,
    material_support_field: TESSAMaterialSupportFieldIR,
    surface: RiggingSurfaceIR,
    partition: MechanicalPartitionIR,
    carrier_policy: ComponentCarrierPolicyIR,
    model_provenance: str,
) -> CanonicalMeshCandidateIR:
    """Build a proposal-only canonical candidate without positional support fiction.

    The returned candidate is intentionally *not* legacy-G1 admissible.  Its
    ``SurfaceSupportBinding`` rows use ``TESSA_FIELD_SUPPORT_MODE_V1`` and carry
    GSA continuous-field provenance only.  Stage19 must qualify learned ``P``.
    """

    validate_mechanical_partition(partition, surface)
    validate_component_carrier_policy(carrier_policy, partition)
    vertices = _validate_world_vertices(vertices_world)
    face_array = _validate_faces(faces, vertex_count=len(vertices))

    if material_support_field.surface_binding_hash != surface.geometry_lineage_hash:
        raise QualificationError("TESSA_LEARNED_GEOMETRY_FIELD_SURFACE_DRIFT")
    if material_support_field.partition_binding_hash != partition.partition_lineage_hash:
        raise QualificationError("TESSA_LEARNED_GEOMETRY_FIELD_PARTITION_DRIFT")
    if not material_support_field.topology_sequence_hash:
        raise QualificationError("TESSA_LEARNED_GEOMETRY_TOPOLOGY_HASH_MISSING")

    proposal_vertex_ids = tuple(
        str(row.proposal_vertex_id) for row in material_support_field.rows
    )
    if len(proposal_vertex_ids) != len(vertices):
        raise QualificationError("TESSA_LEARNED_GEOMETRY_FIELD_VERTEX_COUNT_DRIFT")
    if len(set(proposal_vertex_ids)) != len(proposal_vertex_ids):
        raise QualificationError("TESSA_LEARNED_GEOMETRY_FIELD_VERTEX_ID_DUPLICATE")

    validate_tessa_material_support_field_v1(
        material_support_field,
        surface=surface,
        partition=partition,
        expected_vertex_ids=proposal_vertex_ids,
    )
    row_by_id = {
        str(row.proposal_vertex_id): row for row in material_support_field.rows
    }
    known_components = {str(row.component_id) for row in partition.components}

    vertex_rows = []
    for index, proposal_vertex_id in enumerate(proposal_vertex_ids):
        row = row_by_id[proposal_vertex_id]
        component_id = str(row.mechanical_component_id)
        if component_id not in known_components:
            raise QualificationError("TESSA_LEARNED_GEOMETRY_COMPONENT_INVALID")
        support = SurfaceSupportBinding(
            mode=TESSA_FIELD_SUPPORT_MODE_V1,
            coefficients=tuple(
                (str(surface_id), float(weight))
                for surface_id, weight in row.coefficients
            ),
            metadata={
                "authority_class": "MATERIAL_SUPPORT_ONLY",
                "geometry_position_derived_from_support": False,
                "field_lineage_hash": material_support_field.field_lineage_hash,
                "topology_sequence_hash": material_support_field.topology_sequence_hash,
                "proposal_vertex_id": proposal_vertex_id,
                "mechanical_component_id": component_id,
                "decoded_component_index": int(row.decoded_component_index),
            },
        )
        vertex_rows.append(
            CanonicalMeshVertexCandidateIR(
                candidate_vertex_id=proposal_vertex_id,
                support_binding=support,
                component_id=component_id,
                P=tuple(map(float, vertices[index].tolist())),
                refinement=None,
                metadata={
                    "geometry_position_authority": TESSA_GEOMETRY_AUTHORITY_PENDING_V1,
                    "field_support_authority": TESSA_FIELD_AUTHORITY_V1,
                    "model_provenance": str(model_provenance),
                },
            )
        )

    candidate_faces = tuple(
        tuple(proposal_vertex_ids[int(index)] for index in row)
        for row in face_array.tolist()
    )
    component_by_vertex = {
        row.candidate_vertex_id: row.component_id for row in vertex_rows
    }
    for face in candidate_faces:
        if len({component_by_vertex[vertex_id] for vertex_id in face}) != 1:
            raise QualificationError("TESSA_LEARNED_GEOMETRY_FACE_CROSSES_COMPONENT")

    candidate_edges = tuple(
        sorted(
            {
                _edge(face[0], face[1])
                for face in candidate_faces
            }
            | {
                _edge(face[1], face[2])
                for face in candidate_faces
            }
            | {
                _edge(face[2], face[0])
                for face in candidate_faces
            }
        )
    )
    producer_policy_hash = content_sha256(
        {
            "schema": "RealSaS.TESSALearnedGeometryCandidatePolicy.v1",
            "surface_binding_hash": surface.geometry_lineage_hash,
            "partition_binding_hash": partition.partition_lineage_hash,
            "carrier_policy_binding_hash": carrier_policy.carrier_policy_lineage_hash,
            "material_support_field_binding_hash": material_support_field.field_lineage_hash,
            "topology_sequence_hash": material_support_field.topology_sequence_hash,
            "model_provenance": str(model_provenance),
            "geometry_position_authority": TESSA_GEOMETRY_AUTHORITY_PENDING_V1,
            "field_support_authority": TESSA_FIELD_AUTHORITY_V1,
            "legacy_g1_positional_support_claimed": False,
        }
    )
    candidate = CanonicalMeshCandidateIR(
        vertices=tuple(vertex_rows),
        faces=candidate_faces,
        edges=candidate_edges,
        surface_binding_hash=surface.geometry_lineage_hash,
        partition_binding_hash=partition.partition_lineage_hash,
        carrier_policy_binding_hash=carrier_policy.carrier_policy_lineage_hash,
        producer_id=TESSA_CANDIDATE_PRODUCER_V1,
        producer_policy_hash=producer_policy_hash,
        candidate_lineage_hash="",
        metadata={
            "authority_class": "LEARNED_GEOMETRY_PROPOSAL_ONLY",
            "geometry_position_authority": TESSA_GEOMETRY_AUTHORITY_PENDING_V1,
            "field_support_authority": TESSA_FIELD_AUTHORITY_V1,
            "material_support_field_binding_hash": material_support_field.field_lineage_hash,
            "tessa_topology_sequence_hash": material_support_field.topology_sequence_hash,
            "legacy_g1_positional_support_claimed": False,
            "stage19_exact_candidate_qualification_required": True,
            "model_provenance": str(model_provenance),
        },
    )
    return replace(
        candidate,
        candidate_lineage_hash=canonical_mesh_candidate_lineage_hash(candidate),
    )


def _validate_static_qualification_for_candidate(
    qualification: StaticCanonicalMeshQualificationIR,
    *,
    candidate: CanonicalMeshCandidateIR,
    partition: MechanicalPartitionIR,
) -> None:
    if qualification.qualification_hash != static_mesh_qualification_hash(qualification):
        raise QualificationError("TESSA_LEARNED_GEOMETRY_STATIC_HASH_MISMATCH")
    if qualification.candidate_mesh_binding_hash != candidate.candidate_lineage_hash:
        raise QualificationError("TESSA_LEARNED_GEOMETRY_STATIC_CANDIDATE_DRIFT")
    if qualification.partition_binding_hash != partition.partition_lineage_hash:
        raise QualificationError("TESSA_LEARNED_GEOMETRY_STATIC_PARTITION_DRIFT")
    report = dict(qualification.qualification_report or {})
    if str(report.get("status") or "") != "PASS_STATIC_CANONICAL_CARRIER":
        raise QualificationError("TESSA_LEARNED_GEOMETRY_STATIC_STATUS_NOT_PASS")
    if report.get("static_quality_policy_passed") is not True:
        raise QualificationError("TESSA_LEARNED_GEOMETRY_STATIC_QUALITY_NOT_PASS")
    if report.get("actual_candidate_source_fidelity_passed") is not True:
        raise QualificationError("TESSA_LEARNED_GEOMETRY_SOURCE_FIDELITY_NOT_PASS")
    if report.get("source_fidelity_qualification_passed") is not True:
        raise QualificationError("TESSA_LEARNED_GEOMETRY_SOURCE_FIDELITY_SCOPE_DRIFT")
    if report.get("demo_geometry_lineage") is not False:
        raise QualificationError("TESSA_LEARNED_GEOMETRY_DEMO_STATIC_FORBIDDEN")
    if report.get("demo_only_source_fidelity_admission") is not False:
        raise QualificationError("TESSA_LEARNED_GEOMETRY_DEMO_ADMISSION_FORBIDDEN")


def validate_tessa_learned_geometry_admission_v1(
    value: TESSALearnedGeometryAdmissionIR,
    *,
    candidate: CanonicalMeshCandidateIR,
    material_support_field: TESSAMaterialSupportFieldIR,
    static_qualification: StaticCanonicalMeshQualificationIR,
    surface: RiggingSurfaceIR,
    partition: MechanicalPartitionIR,
    carrier_policy: ComponentCarrierPolicyIR,
) -> None:
    validate_mechanical_partition(partition, surface)
    validate_component_carrier_policy(carrier_policy, partition)
    if candidate.candidate_lineage_hash != canonical_mesh_candidate_lineage_hash(candidate):
        raise QualificationError("TESSA_LEARNED_GEOMETRY_CANDIDATE_HASH_MISMATCH")
    if candidate.producer_id != TESSA_CANDIDATE_PRODUCER_V1:
        raise QualificationError("TESSA_LEARNED_GEOMETRY_PRODUCER_INVALID")
    if candidate.surface_binding_hash != surface.geometry_lineage_hash:
        raise QualificationError("TESSA_LEARNED_GEOMETRY_CANDIDATE_SURFACE_DRIFT")
    if candidate.partition_binding_hash != partition.partition_lineage_hash:
        raise QualificationError("TESSA_LEARNED_GEOMETRY_CANDIDATE_PARTITION_DRIFT")
    if candidate.carrier_policy_binding_hash != carrier_policy.carrier_policy_lineage_hash:
        raise QualificationError("TESSA_LEARNED_GEOMETRY_CANDIDATE_CARRIER_DRIFT")

    metadata = dict(candidate.metadata or {})
    if metadata.get("geometry_position_authority") != TESSA_GEOMETRY_AUTHORITY_PENDING_V1:
        raise QualificationError("TESSA_LEARNED_GEOMETRY_PENDING_AUTHORITY_DRIFT")
    if metadata.get("field_support_authority") != TESSA_FIELD_AUTHORITY_V1:
        raise QualificationError("TESSA_LEARNED_GEOMETRY_FIELD_AUTHORITY_DRIFT")
    if metadata.get("legacy_g1_positional_support_claimed") is not False:
        raise QualificationError("TESSA_LEARNED_GEOMETRY_LEGACY_G1_CLAIM_FORBIDDEN")
    if metadata.get("stage19_exact_candidate_qualification_required") is not True:
        raise QualificationError("TESSA_LEARNED_GEOMETRY_STAGE19_REQUIREMENT_MISSING")
    if metadata.get("material_support_field_binding_hash") != material_support_field.field_lineage_hash:
        raise QualificationError("TESSA_LEARNED_GEOMETRY_FIELD_BINDING_DRIFT")
    if metadata.get("tessa_topology_sequence_hash") != material_support_field.topology_sequence_hash:
        raise QualificationError("TESSA_LEARNED_GEOMETRY_TOPOLOGY_BINDING_DRIFT")

    expected_vertex_ids = tuple(vertex.candidate_vertex_id for vertex in candidate.vertices)
    validate_tessa_material_support_field_v1(
        material_support_field,
        surface=surface,
        partition=partition,
        expected_vertex_ids=expected_vertex_ids,
    )
    row_by_id = {
        str(row.proposal_vertex_id): row for row in material_support_field.rows
    }
    for vertex in candidate.vertices:
        row = row_by_id.get(str(vertex.candidate_vertex_id))
        if row is None:
            raise QualificationError("TESSA_LEARNED_GEOMETRY_FIELD_VERTEX_MISSING")
        if vertex.support_binding.mode != TESSA_FIELD_SUPPORT_MODE_V1:
            raise QualificationError("TESSA_LEARNED_GEOMETRY_SUPPORT_MODE_DRIFT")
        if vertex.refinement is not None:
            raise QualificationError("TESSA_LEARNED_GEOMETRY_FAKE_REFINEMENT_FORBIDDEN")
        if tuple(vertex.support_binding.coefficients) != tuple(row.coefficients):
            raise QualificationError("TESSA_LEARNED_GEOMETRY_SUPPORT_COEFFICIENT_DRIFT")
        support_metadata = dict(vertex.support_binding.metadata or {})
        if support_metadata.get("geometry_position_derived_from_support") is not False:
            raise QualificationError("TESSA_LEARNED_GEOMETRY_POSITION_SUPPORT_CLAIM_FORBIDDEN")
        if support_metadata.get("field_lineage_hash") != material_support_field.field_lineage_hash:
            raise QualificationError("TESSA_LEARNED_GEOMETRY_VERTEX_FIELD_HASH_DRIFT")
        if str(vertex.component_id) != str(row.mechanical_component_id):
            raise QualificationError("TESSA_LEARNED_GEOMETRY_VERTEX_COMPONENT_DRIFT")
        if not all(math.isfinite(float(x)) for x in vertex.P):
            raise QualificationError("TESSA_LEARNED_GEOMETRY_VERTEX_POSITION_INVALID")

    _validate_static_qualification_for_candidate(
        static_qualification,
        candidate=candidate,
        partition=partition,
    )

    expected = {
        "candidate_mesh_binding_hash": candidate.candidate_lineage_hash,
        "static_qualification_binding_hash": static_qualification.qualification_hash,
        "material_support_field_binding_hash": material_support_field.field_lineage_hash,
        "surface_binding_hash": surface.geometry_lineage_hash,
        "partition_binding_hash": partition.partition_lineage_hash,
        "carrier_policy_binding_hash": carrier_policy.carrier_policy_lineage_hash,
        "topology_sequence_hash": material_support_field.topology_sequence_hash,
        "geometry_authority": TESSA_GEOMETRY_AUTHORITY_QUALIFIED_V1,
        "field_support_authority": TESSA_FIELD_AUTHORITY_V1,
    }
    for field_name, expected_value in expected.items():
        if getattr(value, field_name) != expected_value:
            raise QualificationError(
                f"TESSA_LEARNED_GEOMETRY_ADMISSION_BINDING_DRIFT:{field_name}"
            )
    if value.metadata.get("threshold_relaxation_performed") is not False:
        raise QualificationError("TESSA_LEARNED_GEOMETRY_THRESHOLD_RELAXATION_FORBIDDEN")
    if value.metadata.get("legacy_g1_positional_support_claimed") is not False:
        raise QualificationError("TESSA_LEARNED_GEOMETRY_ADMISSION_LEGACY_G1_CLAIM_FORBIDDEN")
    if value.admission_hash != tessa_learned_geometry_admission_hash(value):
        raise QualificationError("TESSA_LEARNED_GEOMETRY_ADMISSION_HASH_MISMATCH")


def build_tessa_learned_geometry_admission_v1(
    *,
    candidate: CanonicalMeshCandidateIR,
    material_support_field: TESSAMaterialSupportFieldIR,
    static_qualification: StaticCanonicalMeshQualificationIR,
    surface: RiggingSurfaceIR,
    partition: MechanicalPartitionIR,
    carrier_policy: ComponentCarrierPolicyIR,
) -> TESSALearnedGeometryAdmissionIR:
    value = TESSALearnedGeometryAdmissionIR(
        candidate_mesh_binding_hash=candidate.candidate_lineage_hash,
        static_qualification_binding_hash=static_qualification.qualification_hash,
        material_support_field_binding_hash=material_support_field.field_lineage_hash,
        surface_binding_hash=surface.geometry_lineage_hash,
        partition_binding_hash=partition.partition_lineage_hash,
        carrier_policy_binding_hash=carrier_policy.carrier_policy_lineage_hash,
        topology_sequence_hash=material_support_field.topology_sequence_hash,
        geometry_authority=TESSA_GEOMETRY_AUTHORITY_QUALIFIED_V1,
        field_support_authority=TESSA_FIELD_AUTHORITY_V1,
        admission_hash="",
        metadata={
            "authority_class": "COMPILER_QUALIFIED_LEARNED_GEOMETRY_ADMISSION",
            "stage19_exact_candidate_qualification_required": True,
            "threshold_relaxation_performed": False,
            "legacy_g1_positional_support_claimed": False,
            "geometry_position_derived_from_gsa_support": False,
            "compiler_is_sole_promotion_authority": True,
        },
    )
    value = replace(
        value,
        admission_hash=tessa_learned_geometry_admission_hash(value),
    )
    validate_tessa_learned_geometry_admission_v1(
        value,
        candidate=candidate,
        material_support_field=material_support_field,
        static_qualification=static_qualification,
        surface=surface,
        partition=partition,
        carrier_policy=carrier_policy,
    )
    return value
