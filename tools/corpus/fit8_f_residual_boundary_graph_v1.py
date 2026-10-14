from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from math import factorial
from typing import Mapping, Sequence

from tools.corpus.fit8_cube_rotation_symmetry_v1 import proper_cube_rotations_v1
from tools.corpus.fit8_face_strand_boundary_incidence_v1 import (
    FaceStrandBoundaryIncidenceQualificationV1,
)
from tools.corpus.fit8_face_strand_teacher_v1 import FaceStrandTeacherV1
from tools.corpus.fit8_f_topology_residual_court_v1 import evaluate_face_topology_residual_v1
from tools.corpus.fit8_local_patch_teacher_v1 import ExtractionCellTeacherV1

SCHEMA_VERSION = "RealSaS.FResidualBoundaryGraph.v1"
CANONICAL = "CANONICAL"
NO_RESIDUAL = "NO_RESIDUAL"
ABSTAIN = "ABSTAIN"
STATUSES = {CANONICAL, NO_RESIDUAL, ABSTAIN}


@dataclass(frozen=True)
class ResidualBoundaryGraphCanonicalizationPolicyV1:
    max_exact_patch_count: int
    provenance: str

    def validate(self) -> None:
        if self.max_exact_patch_count <= 0:
            raise ValueError("F_RESIDUAL_BOUNDARY_GRAPH_PATCH_CAP_INVALID")
        if not self.provenance:
            raise ValueError("F_RESIDUAL_BOUNDARY_GRAPH_POLICY_PROVENANCE_MISSING")


@dataclass(frozen=True)
class ResidualBoundaryGraphResultV1:
    schema_version: str
    status: str
    patch_count: int
    residual_edge_event_count: int
    residual_face_count: int
    canonical_code: tuple | None
    abstention_reason: str | None
    rotation_count_considered: int
    patch_permutation_count_considered: int

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("F_RESIDUAL_BOUNDARY_GRAPH_SCHEMA_INVALID")
        if self.status not in STATUSES:
            raise ValueError("F_RESIDUAL_BOUNDARY_GRAPH_STATUS_INVALID")
        for name in (
            "patch_count",
            "residual_edge_event_count",
            "residual_face_count",
            "rotation_count_considered",
            "patch_permutation_count_considered",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"F_RESIDUAL_BOUNDARY_GRAPH_NEGATIVE:{name}")
        if self.status == CANONICAL:
            if self.canonical_code is None or self.abstention_reason is not None:
                raise ValueError("F_RESIDUAL_BOUNDARY_GRAPH_CANONICAL_PAYLOAD_INVALID")
        elif self.status == NO_RESIDUAL:
            if self.canonical_code is not None or self.abstention_reason is not None:
                raise ValueError("F_RESIDUAL_BOUNDARY_GRAPH_NO_RESIDUAL_PAYLOAD_INVALID")
            if self.residual_edge_event_count or self.residual_face_count:
                raise ValueError("F_RESIDUAL_BOUNDARY_GRAPH_NO_RESIDUAL_COUNT_INVALID")
        else:
            if self.canonical_code is not None or not self.abstention_reason:
                raise ValueError("F_RESIDUAL_BOUNDARY_GRAPH_ABSTENTION_PAYLOAD_INVALID")


def _triangle_to_patch(cell: ExtractionCellTeacherV1) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for patch in cell.local_patch_lower_bound:
        for triangle_id in patch.source_triangle_ids:
            triangle_id = int(triangle_id)
            if triangle_id in mapping and mapping[triangle_id] != patch.patch_local_id:
                raise ValueError("F_RESIDUAL_BOUNDARY_GRAPH_TRIANGLE_MULTI_PATCH")
            mapping[triangle_id] = int(patch.patch_local_id)
    return mapping


def _face_maps(
    faces: Sequence[FaceStrandTeacherV1],
    qualifications: Sequence[FaceStrandBoundaryIncidenceQualificationV1],
) -> tuple[dict[int, FaceStrandTeacherV1], dict[int, FaceStrandBoundaryIncidenceQualificationV1]]:
    face_map = {int(face.face_index): face for face in faces}
    qualification_map = {int(row.face_index): row for row in qualifications}
    if sorted(face_map) != list(range(6)):
        raise ValueError("F_RESIDUAL_BOUNDARY_GRAPH_FACE_SET_INCOMPLETE")
    if sorted(qualification_map) != list(range(6)):
        raise ValueError("F_RESIDUAL_BOUNDARY_GRAPH_QUALIFICATION_SET_INCOMPLETE")
    if len(face_map) != len(faces) or len(qualification_map) != len(qualifications):
        raise ValueError("F_RESIDUAL_BOUNDARY_GRAPH_DUPLICATE_FACE")
    for face in face_map.values():
        face.validate()
    for qualification in qualification_map.values():
        qualification.validate()
        face = face_map[qualification.face_index]
        if len(qualification.strand_incidence) != len(face.strands):
            raise ValueError("F_RESIDUAL_BOUNDARY_GRAPH_STRAND_QUALIFICATION_COUNT_MISMATCH")
    return face_map, qualification_map


def _edge_residual_events(
    cell: ExtractionCellTeacherV1,
    triangle_to_patch: Mapping[int, int],
) -> tuple[tuple[int, int, int], ...]:
    """(cube_edge, patch_local_id, crossing_multiplicity) for capacity beyond F-B."""
    counts: dict[tuple[int, int], int] = {}
    for crossing in cell.edge_crossings:
        patch_ids = {
            triangle_to_patch[int(triangle_id)]
            for triangle_id in crossing.source_triangle_ids
            if int(triangle_id) in triangle_to_patch
        }
        if not patch_ids:
            raise ValueError("F_RESIDUAL_BOUNDARY_GRAPH_EDGE_PATCH_UNRESOLVED")
        for patch_id in patch_ids:
            key = (int(crossing.edge_index), int(patch_id))
            counts[key] = counts.get(key, 0) + 1
    return tuple(sorted((edge, patch, count) for (edge, patch), count in counts.items() if count > 1))


def _face_residual_events(
    cell: ExtractionCellTeacherV1,
    face_map: Mapping[int, FaceStrandTeacherV1],
    qualification_map: Mapping[int, FaceStrandBoundaryIncidenceQualificationV1],
    triangle_to_patch: Mapping[int, int],
) -> tuple[tuple, ...]:
    rows = []
    for face_index in range(6):
        face = face_map[face_index]
        qualification = qualification_map[face_index]
        report = evaluate_face_topology_residual_v1(cell, face)
        if not report.topology_residual_required:
            continue
        strand_rows = []
        for strand, boundary in zip(face.strands, qualification.strand_incidence):
            if strand.strand_local_id != boundary.strand_local_id:
                raise ValueError("F_RESIDUAL_BOUNDARY_GRAPH_STRAND_ID_MISMATCH")
            patch_ids = tuple(sorted({
                triangle_to_patch[int(triangle_id)]
                for triangle_id in strand.source_triangle_ids
                if int(triangle_id) in triangle_to_patch
            }))
            if not patch_ids:
                raise ValueError("F_RESIDUAL_BOUNDARY_GRAPH_FACE_PATCH_UNRESOLVED")
            strand_rows.append((
                strand.kind,
                int(strand.boundary_endpoint_count),
                int(strand.interior_endpoint_count),
                int(strand.max_graph_degree),
                patch_ids,
                tuple(tuple(map(int, edge_set)) for edge_set in boundary.boundary_endpoint_cube_edge_sets),
            ))
        rows.append((
            face_index,
            tuple(sorted(report.topology_residual_reasons)),
            tuple(sorted(strand_rows)),
        ))
    return tuple(rows)


def _rotated_code(
    patch_count: int,
    edge_events: Sequence[tuple[int, int, int]],
    face_events: Sequence[tuple],
    rotation,
    new_label_by_old: Sequence[int],
) -> tuple:
    rotated_edges = tuple(sorted(
        (
            int(rotation.edge_permutation[edge_index]),
            int(new_label_by_old[patch_id]),
            int(multiplicity),
        )
        for edge_index, patch_id, multiplicity in edge_events
    ))

    rotated_faces = []
    for face_index, reasons, strands in face_events:
        rotated_strands = []
        for kind, boundary_count, interior_count, max_degree, patch_ids, endpoint_edge_sets in strands:
            rotated_patch_ids = tuple(sorted(int(new_label_by_old[patch_id]) for patch_id in patch_ids))
            rotated_endpoint_sets = tuple(sorted(
                tuple(sorted(int(rotation.edge_permutation[edge_id]) for edge_id in edge_set))
                for edge_set in endpoint_edge_sets
            ))
            rotated_strands.append((
                kind,
                boundary_count,
                interior_count,
                max_degree,
                rotated_patch_ids,
                rotated_endpoint_sets,
            ))
        rotated_faces.append((
            int(rotation.face_permutation[face_index]),
            reasons,
            tuple(sorted(rotated_strands)),
        ))

    return (
        "F_RESIDUAL_BOUNDARY_GRAPH",
        int(patch_count),
        rotated_edges,
        tuple(sorted(rotated_faces)),
    )


def canonicalize_residual_boundary_graph_v1(
    cell: ExtractionCellTeacherV1,
    faces: Sequence[FaceStrandTeacherV1],
    qualifications: Sequence[FaceStrandBoundaryIncidenceQualificationV1],
    *,
    policy: ResidualBoundaryGraphCanonicalizationPolicyV1,
) -> ResidualBoundaryGraphResultV1:
    """Canonicalize residual boundary topology under proper cube rotations and unlabeled patches.

    Reflection equivalence is deliberately excluded. Patch-local IDs are teacher
    bookkeeping only, so exact canonicalization enumerates their permutations.
    If patch cardinality exceeds the explicit census policy, the function abstains
    instead of approximating graph isomorphism.

    This artifact describes cell-boundary residual topology only. It must not be
    promoted to a whole-cell/interior-topology authority until a separate interior
    incidence court closes tunnel/interior connectivity.
    """
    cell.validate()
    policy.validate()
    face_map, qualification_map = _face_maps(faces, qualifications)
    triangle_to_patch = _triangle_to_patch(cell)
    patch_count = len(cell.local_patch_lower_bound)
    edge_events = _edge_residual_events(cell, triangle_to_patch)
    face_events = _face_residual_events(
        cell,
        face_map,
        qualification_map,
        triangle_to_patch,
    )

    if not edge_events and not face_events:
        result = ResidualBoundaryGraphResultV1(
            schema_version=SCHEMA_VERSION,
            status=NO_RESIDUAL,
            patch_count=patch_count,
            residual_edge_event_count=0,
            residual_face_count=0,
            canonical_code=None,
            abstention_reason=None,
            rotation_count_considered=0,
            patch_permutation_count_considered=0,
        )
        result.validate()
        return result

    if patch_count > policy.max_exact_patch_count:
        result = ResidualBoundaryGraphResultV1(
            schema_version=SCHEMA_VERSION,
            status=ABSTAIN,
            patch_count=patch_count,
            residual_edge_event_count=len(edge_events),
            residual_face_count=len(face_events),
            canonical_code=None,
            abstention_reason=(
                "EXACT_PATCH_PERMUTATION_CAP_EXCEEDED:"
                f"{patch_count}>{policy.max_exact_patch_count}"
            ),
            rotation_count_considered=0,
            patch_permutation_count_considered=0,
        )
        result.validate()
        return result

    rotations = proper_cube_rotations_v1()
    patch_permutations = tuple(permutations(range(patch_count))) if patch_count else ((),)
    candidates = []
    for rotation in rotations:
        for patch_permutation in patch_permutations:
            candidates.append(
                _rotated_code(
                    patch_count,
                    edge_events,
                    face_events,
                    rotation,
                    patch_permutation,
                )
            )
    canonical_code = min(candidates)
    result = ResidualBoundaryGraphResultV1(
        schema_version=SCHEMA_VERSION,
        status=CANONICAL,
        patch_count=patch_count,
        residual_edge_event_count=len(edge_events),
        residual_face_count=len(face_events),
        canonical_code=canonical_code,
        abstention_reason=None,
        rotation_count_considered=len(rotations),
        patch_permutation_count_considered=factorial(patch_count),
    )
    result.validate()
    return result
