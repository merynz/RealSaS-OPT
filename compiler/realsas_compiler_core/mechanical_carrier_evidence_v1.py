from __future__ import annotations

"""Carrier-bound model evidence derived from the statically qualified mechanical mesh.

GSA/RiggingSurfaceIR remains observation-grounded perception evidence. This module
defines the mechanical coordinate system consumed by carrier-first learned mechanics.
The neural models are not required to consume triangle faces directly; faces are
sealed here so XYZ/normals/query evidence cannot silently drift to another
interpolation basis.
"""

from dataclasses import asdict, dataclass, field, replace
import math
from typing import Any, Mapping

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.types import QualificationError

Json = dict[str, Any]
Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class MechanicalCarrierEvidenceIR:
    candidate_mesh_binding_hash: str
    static_mesh_qualification_binding_hash: str
    surface_addressing_binding_hash: str
    source_surface_binding_hash: str
    ordered_vertex_ids: tuple[str, ...]
    positions: tuple[Vec3, ...]
    normals: tuple[Vec3, ...]
    normal_valid: tuple[bool, ...]
    face_vertex_indices: tuple[tuple[int, int, int], ...]
    topology_hash: str
    geometry_hash: str
    carrier_evidence_hash: str
    schema_version: str = "RealSaS.MechanicalCarrierEvidenceIR.v1"
    metadata: Json = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _hash_without(value: MechanicalCarrierEvidenceIR, field_name: str) -> str:
    payload = value.to_dict()
    payload.pop(field_name, None)
    return content_sha256(payload)


def mechanical_carrier_evidence_hash(value: MechanicalCarrierEvidenceIR) -> str:
    return _hash_without(value, "carrier_evidence_hash")


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _finite_vec3(value) -> Vec3:
    out = tuple(float(x) for x in value)
    if len(out) != 3 or not all(math.isfinite(x) for x in out):
        raise QualificationError("MECHANICAL_CARRIER_EVIDENCE_VEC3_INVALID")
    return out  # type: ignore[return-value]


def build_mechanical_carrier_evidence_v1(
    candidate,
    *,
    static_qualification,
    surface_addressing,
) -> MechanicalCarrierEvidenceIR:
    if (
        static_qualification.candidate_mesh_binding_hash
        != candidate.candidate_lineage_hash
    ):
        raise QualificationError("MECHANICAL_CARRIER_STATIC_BINDING_DRIFT")
    if (
        surface_addressing.candidate_mesh_binding_hash
        != candidate.candidate_lineage_hash
        or static_qualification.surface_addressing_binding_hash
        != surface_addressing.addressing_hash
    ):
        raise QualificationError("MECHANICAL_CARRIER_ADDRESSING_BINDING_DRIFT")

    ordered_vertices = tuple(
        sorted(candidate.vertices, key=lambda row: str(row.candidate_vertex_id))
    )
    if not ordered_vertices:
        raise QualificationError("MECHANICAL_CARRIER_EMPTY_VERTICES")

    vertex_ids = tuple(str(row.candidate_vertex_id) for row in ordered_vertices)
    if len(set(vertex_ids)) != len(vertex_ids):
        raise QualificationError("MECHANICAL_CARRIER_DUPLICATE_VERTEX_ID")

    positions = tuple(_finite_vec3(row.P) for row in ordered_vertices)
    index_by_id = {vertex_id: index for index, vertex_id in enumerate(vertex_ids)}

    faces: list[tuple[int, int, int]] = []
    normal_acc = [[0.0, 0.0, 0.0] for _ in vertex_ids]
    for raw_face in candidate.faces:
        ids = tuple(str(x) for x in raw_face)
        if len(ids) != 3 or len(set(ids)) != 3:
            raise QualificationError("MECHANICAL_CARRIER_FACE_INVALID")
        try:
            face = tuple(index_by_id[x] for x in ids)
        except KeyError as exc:
            raise QualificationError("MECHANICAL_CARRIER_FACE_VERTEX_UNKNOWN") from exc
        a, b, c = (positions[index] for index in face)
        n = _cross(_sub(b, a), _sub(c, a))
        area2 = math.sqrt(sum(x * x for x in n))
        if not math.isfinite(area2) or area2 <= 1e-12:
            raise QualificationError("MECHANICAL_CARRIER_DEGENERATE_FACE")
        for index in face:
            normal_acc[index][0] += n[0]
            normal_acc[index][1] += n[1]
            normal_acc[index][2] += n[2]
        faces.append(face)  # type: ignore[arg-type]

    if not faces:
        raise QualificationError("MECHANICAL_CARRIER_EMPTY_FACES")

    normals: list[Vec3] = []
    normal_valid: list[bool] = []
    for row in normal_acc:
        length = math.sqrt(sum(float(x) * float(x) for x in row))
        valid = bool(math.isfinite(length) and length > 1e-12)
        normal_valid.append(valid)
        if valid:
            normals.append(
                (
                    float(row[0]) / length,
                    float(row[1]) / length,
                    float(row[2]) / length,
                )
            )
        else:
            normals.append((0.0, 0.0, 0.0))

    topology_hash = content_sha256(
        {
            "schema": "RealSaS.MechanicalCarrierTopologyBasis.v1",
            "ordered_vertex_ids": vertex_ids,
            "face_vertex_indices": tuple(faces),
        }
    )
    geometry_hash = content_sha256(
        {
            "schema": "RealSaS.MechanicalCarrierGeometryBasis.v1",
            "ordered_vertex_ids": vertex_ids,
            "positions": positions,
            "normals": tuple(normals),
            "normal_valid": tuple(normal_valid),
        }
    )

    value = MechanicalCarrierEvidenceIR(
        candidate_mesh_binding_hash=str(candidate.candidate_lineage_hash),
        static_mesh_qualification_binding_hash=str(
            static_qualification.qualification_hash
        ),
        surface_addressing_binding_hash=str(surface_addressing.addressing_hash),
        source_surface_binding_hash=str(candidate.surface_binding_hash),
        ordered_vertex_ids=vertex_ids,
        positions=positions,
        normals=tuple(normals),
        normal_valid=tuple(normal_valid),
        face_vertex_indices=tuple(faces),
        topology_hash=topology_hash,
        geometry_hash=geometry_hash,
        carrier_evidence_hash="",
        metadata={
            "authority": "STATIC_QUALIFIED_MECHANICAL_CARRIER",
            "positions_owner": "STAGE18_CANONICAL_MESH",
            "normals_owner": "DERIVED_FROM_EXACT_CARRIER_FACES",
            "faces_are_neural_input_required": False,
            "gsa_is_auxiliary_observation_evidence": True,
            "invalid_normal_count": int(sum(not x for x in normal_valid)),
        },
    )
    return replace(value, carrier_evidence_hash=mechanical_carrier_evidence_hash(value))


def mechanical_carrier_evidence_from_dict(
    payload: Mapping[str, Any],
) -> MechanicalCarrierEvidenceIR:
    schema = str(payload.get("schema_version") or payload.get("schema") or "")
    if schema != "RealSaS.MechanicalCarrierEvidenceIR.v1":
        raise QualificationError("MECHANICAL_CARRIER_EVIDENCE_SCHEMA_DRIFT")
    value = MechanicalCarrierEvidenceIR(
        candidate_mesh_binding_hash=str(payload["candidate_mesh_binding_hash"]),
        static_mesh_qualification_binding_hash=str(
            payload["static_mesh_qualification_binding_hash"]
        ),
        surface_addressing_binding_hash=str(
            payload["surface_addressing_binding_hash"]
        ),
        source_surface_binding_hash=str(payload["source_surface_binding_hash"]),
        ordered_vertex_ids=tuple(map(str, payload.get("ordered_vertex_ids") or ())),
        positions=tuple(
            _finite_vec3(row) for row in payload.get("positions") or ()
        ),
        normals=tuple(_finite_vec3(row) for row in payload.get("normals") or ()),
        normal_valid=tuple(bool(x) for x in payload.get("normal_valid") or ()),
        face_vertex_indices=tuple(
            tuple(int(x) for x in row)
            for row in payload.get("face_vertex_indices") or ()
        ),
        topology_hash=str(payload["topology_hash"]),
        geometry_hash=str(payload["geometry_hash"]),
        carrier_evidence_hash=str(payload["carrier_evidence_hash"]),
        schema_version=schema,
        metadata=dict(payload.get("metadata") or {}),
    )
    n = len(value.ordered_vertex_ids)
    if (
        len(value.positions) != n
        or len(value.normals) != n
        or len(value.normal_valid) != n
        or n == 0
    ):
        raise QualificationError("MECHANICAL_CARRIER_EVIDENCE_SHAPE_DRIFT")
    if any(
        len(face) != 3
        or min(face) < 0
        or max(face) >= n
        or len(set(face)) != 3
        for face in value.face_vertex_indices
    ):
        raise QualificationError("MECHANICAL_CARRIER_EVIDENCE_FACE_DRIFT")
    expected_topology = content_sha256(
        {
            "schema": "RealSaS.MechanicalCarrierTopologyBasis.v1",
            "ordered_vertex_ids": value.ordered_vertex_ids,
            "face_vertex_indices": value.face_vertex_indices,
        }
    )
    if value.topology_hash != expected_topology:
        raise QualificationError("MECHANICAL_CARRIER_TOPOLOGY_HASH_MISMATCH")
    expected_geometry = content_sha256(
        {
            "schema": "RealSaS.MechanicalCarrierGeometryBasis.v1",
            "ordered_vertex_ids": value.ordered_vertex_ids,
            "positions": value.positions,
            "normals": value.normals,
            "normal_valid": value.normal_valid,
        }
    )
    if value.geometry_hash != expected_geometry:
        raise QualificationError("MECHANICAL_CARRIER_GEOMETRY_HASH_MISMATCH")
    if value.carrier_evidence_hash != mechanical_carrier_evidence_hash(value):
        raise QualificationError("MECHANICAL_CARRIER_EVIDENCE_HASH_MISMATCH")
    return value


__all__ = [
    "MechanicalCarrierEvidenceIR",
    "build_mechanical_carrier_evidence_v1",
    "mechanical_carrier_evidence_from_dict",
    "mechanical_carrier_evidence_hash",
]
