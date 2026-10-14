from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from tools.corpus.fit8_face_strand_teacher_v1 import FaceStrandTeacherV1
from tools.corpus.fit8_f_topology_residual_court_v1 import evaluate_face_topology_residual_v1
from tools.corpus.fit8_local_patch_teacher_v1 import ExtractionCellTeacherV1

SCHEMA_VERSION = "RealSaS.FResidualSignature.v1"


@dataclass(frozen=True, order=True)
class EdgeResidualSignatureV1:
    """Patch-label-invariant crossing multiplicity on one grid edge."""

    crossing_multiplicity_multiset: tuple[int, ...]

    def validate(self) -> None:
        if not self.crossing_multiplicity_multiset:
            raise ValueError("F_RESIDUAL_SIGNATURE_EDGE_EMPTY")
        if any(value <= 0 for value in self.crossing_multiplicity_multiset):
            raise ValueError("F_RESIDUAL_SIGNATURE_EDGE_MULTIPLICITY_INVALID")
        if max(self.crossing_multiplicity_multiset) <= 1:
            raise ValueError("F_RESIDUAL_SIGNATURE_EDGE_NOT_RESIDUAL")


@dataclass(frozen=True, order=True)
class StrandTopologyProfileV1:
    kind: str
    boundary_endpoint_count: int
    interior_endpoint_count: int
    max_graph_degree: int
    patch_cardinality: int

    def validate(self) -> None:
        if self.boundary_endpoint_count < 0 or self.interior_endpoint_count < 0:
            raise ValueError("F_RESIDUAL_SIGNATURE_STRAND_ENDPOINT_INVALID")
        if self.max_graph_degree < 0 or self.patch_cardinality <= 0:
            raise ValueError("F_RESIDUAL_SIGNATURE_STRAND_PROFILE_INVALID")


@dataclass(frozen=True, order=True)
class FaceResidualSignatureV1:
    """Orientation/patch-label invariant topology signature for one residual face.

    Geometry, source IDs, triangle counts and face orientation are deliberately not
    part of this signature. They are continuous/provenance parameters, not local
    topology-template identity.
    """

    topology_reasons: tuple[str, ...]
    strand_profiles: tuple[StrandTopologyProfileV1, ...]
    patch_strand_multiplicity_multiset: tuple[int, ...]

    def validate(self) -> None:
        if not self.topology_reasons:
            raise ValueError("F_RESIDUAL_SIGNATURE_FACE_NOT_RESIDUAL")
        if not self.strand_profiles:
            raise ValueError("F_RESIDUAL_SIGNATURE_FACE_STRANDS_EMPTY")
        for profile in self.strand_profiles:
            profile.validate()
        if any(value <= 0 for value in self.patch_strand_multiplicity_multiset):
            raise ValueError("F_RESIDUAL_SIGNATURE_FACE_PATCH_MULTIPLICITY_INVALID")


@dataclass(frozen=True)
class ResidualSignatureHistogramV1:
    schema_version: str
    event_count: int
    unique_signature_count: int
    counts: tuple[tuple[object, int], ...]

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("F_RESIDUAL_SIGNATURE_HISTOGRAM_SCHEMA_INVALID")
        if self.event_count < 0 or self.unique_signature_count < 0:
            raise ValueError("F_RESIDUAL_SIGNATURE_HISTOGRAM_NEGATIVE")
        if self.unique_signature_count != len(self.counts):
            raise ValueError("F_RESIDUAL_SIGNATURE_HISTOGRAM_UNIQUE_MISMATCH")
        if sum(count for _, count in self.counts) != self.event_count:
            raise ValueError("F_RESIDUAL_SIGNATURE_HISTOGRAM_TOTAL_MISMATCH")
        if any(count <= 0 for _, count in self.counts):
            raise ValueError("F_RESIDUAL_SIGNATURE_HISTOGRAM_COUNT_INVALID")


def _triangle_to_patch(cell: ExtractionCellTeacherV1) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for patch in cell.local_patch_lower_bound:
        for triangle_id in patch.source_triangle_ids:
            if triangle_id in mapping and mapping[triangle_id] != patch.patch_local_id:
                raise ValueError("F_RESIDUAL_SIGNATURE_TRIANGLE_MULTI_PATCH")
            mapping[triangle_id] = patch.patch_local_id
    return mapping


def derive_edge_residual_signature_v1(
    cell: ExtractionCellTeacherV1,
    edge_index: int,
) -> EdgeResidualSignatureV1 | None:
    """Return a label-invariant signature only when F-B edge capacity is exceeded."""
    cell.validate()
    if edge_index < 0 or edge_index >= 12:
        raise ValueError("F_RESIDUAL_SIGNATURE_EDGE_INDEX_INVALID")
    triangle_to_patch = _triangle_to_patch(cell)
    patch_counts: dict[int, int] = {}
    for crossing in cell.edge_crossings:
        if crossing.edge_index != edge_index:
            continue
        patch_ids = {
            triangle_to_patch[triangle_id]
            for triangle_id in crossing.source_triangle_ids
            if triangle_id in triangle_to_patch
        }
        if not patch_ids:
            raise ValueError("F_RESIDUAL_SIGNATURE_EDGE_PATCH_UNRESOLVED")
        for patch_id in patch_ids:
            patch_counts[patch_id] = patch_counts.get(patch_id, 0) + 1
    if not patch_counts or max(patch_counts.values()) <= 1:
        return None
    signature = EdgeResidualSignatureV1(tuple(sorted(patch_counts.values())))
    signature.validate()
    return signature


def derive_face_residual_signature_v1(
    cell: ExtractionCellTeacherV1,
    face: FaceStrandTeacherV1,
) -> FaceResidualSignatureV1 | None:
    """Return topology-only residual signature; canonicalization noise is excluded."""
    cell.validate()
    face.validate()
    report = evaluate_face_topology_residual_v1(cell, face)
    if not report.topology_residual_required:
        return None

    triangle_to_patch = _triangle_to_patch(cell)
    patch_strand_counts: Counter[int] = Counter()
    profiles: list[StrandTopologyProfileV1] = []
    for strand in face.strands:
        patch_ids = {
            triangle_to_patch[triangle_id]
            for triangle_id in strand.source_triangle_ids
            if triangle_id in triangle_to_patch
        }
        if not patch_ids:
            raise ValueError("F_RESIDUAL_SIGNATURE_FACE_PATCH_UNRESOLVED")
        for patch_id in patch_ids:
            patch_strand_counts[patch_id] += 1
        profiles.append(
            StrandTopologyProfileV1(
                kind=strand.kind,
                boundary_endpoint_count=strand.boundary_endpoint_count,
                interior_endpoint_count=strand.interior_endpoint_count,
                max_graph_degree=strand.max_graph_degree,
                patch_cardinality=len(patch_ids),
            )
        )

    signature = FaceResidualSignatureV1(
        topology_reasons=tuple(sorted(report.topology_residual_reasons)),
        strand_profiles=tuple(sorted(profiles)),
        patch_strand_multiplicity_multiset=tuple(sorted(patch_strand_counts.values())),
    )
    signature.validate()
    return signature


def build_residual_signature_histogram_v1(
    signatures: Iterable[object | None],
) -> ResidualSignatureHistogramV1:
    counter = Counter(signature for signature in signatures if signature is not None)
    rows = tuple(sorted(counter.items(), key=lambda row: (-row[1], repr(row[0]))))
    report = ResidualSignatureHistogramV1(
        schema_version=SCHEMA_VERSION,
        event_count=sum(counter.values()),
        unique_signature_count=len(counter),
        counts=rows,
    )
    report.validate()
    return report
