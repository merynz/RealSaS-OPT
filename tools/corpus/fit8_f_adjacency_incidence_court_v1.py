from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from tools.corpus.fit8_f_oracle_assembly_court_v1 import (
    PatchContinuationV1,
    PatchInstanceKeyV1,
    PatchInstanceV1,
)

SCHEMA_VERSION = "RealSaS.FAdjacencyIncidenceCourt.v1"


@dataclass(frozen=True)
class FAdjacencyIncidenceReportV1:
    schema_version: str
    source_triangle_count: int
    source_adjacency_count: int
    witnessed_intra_patch_count: int
    witnessed_cross_cell_count: int
    missing_source_adjacencies: tuple[tuple[int, int], ...]
    missing_source_triangles: tuple[int, ...]
    continuation_edge_count: int
    unjustified_continuations: tuple[tuple[PatchInstanceKeyV1, PatchInstanceKeyV1], ...]

    @property
    def exact_incidence_coverage(self) -> bool:
        return (
            not self.missing_source_adjacencies
            and not self.missing_source_triangles
            and not self.unjustified_continuations
            and self.witnessed_intra_patch_count + self.witnessed_cross_cell_count
            == self.source_adjacency_count
        )

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("F_ADJACENCY_INCIDENCE_SCHEMA_INVALID")
        for name in (
            "source_triangle_count",
            "source_adjacency_count",
            "witnessed_intra_patch_count",
            "witnessed_cross_cell_count",
            "continuation_edge_count",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"F_ADJACENCY_INCIDENCE_NEGATIVE:{name}")
        if self.witnessed_intra_patch_count + self.witnessed_cross_cell_count > self.source_adjacency_count:
            raise ValueError("F_ADJACENCY_INCIDENCE_OVERCLAIM")


def _canonical_source_adjacency(
    source_triangle_adjacency: Mapping[int, Sequence[int]],
) -> tuple[set[int], set[tuple[int, int]]]:
    triangles = {int(tid) for tid in source_triangle_adjacency}
    pairs: set[tuple[int, int]] = set()
    for a_raw, neighbors in source_triangle_adjacency.items():
        a = int(a_raw)
        for b_raw in neighbors:
            b = int(b_raw)
            if a == b:
                raise ValueError("F_ADJACENCY_INCIDENCE_SOURCE_SELF_ADJACENCY")
            triangles.add(b)
            pair = (a, b) if a < b else (b, a)
            pairs.add(pair)
    if not triangles:
        raise ValueError("F_ADJACENCY_INCIDENCE_SOURCE_EMPTY")
    return triangles, pairs


def evaluate_oracle_adjacency_incidence_v1(
    source_triangle_adjacency: Mapping[int, Sequence[int]],
    patches: Sequence[PatchInstanceV1],
    continuations: Sequence[PatchContinuationV1],
) -> FAdjacencyIncidenceReportV1:
    """Measure whether local-patch assembly can carry authored triangle incidence.

    This is intentionally weaker than a candidate-mesh topology PASS. Source
    adjacency remains authority. The court only asks whether every authored
    triangle-adjacency pair has a representation witness and whether every
    supplied cross-cell continuation has source-lineage justification.

    A witness is either:
      * both source triangles coexist in one local patch; or
      * the two triangles are carried by local patches connected by a qualified
        cross-cell continuation.

    Patch co-membership is never interpreted as a clique of new adjacencies, so
    this court does not claim absence of false candidate-mesh edges. That requires
    an actual assembled surface and a stronger primal-topology court.
    """
    source_triangles, source_pairs = _canonical_source_adjacency(source_triangle_adjacency)

    patch_by_key: dict[PatchInstanceKeyV1, PatchInstanceV1] = {}
    triangle_seen: set[int] = set()
    for patch in patches:
        patch.validate()
        if patch.key in patch_by_key:
            raise ValueError("F_ADJACENCY_INCIDENCE_DUPLICATE_PATCH_KEY")
        unknown = set(patch.source_triangle_ids) - source_triangles
        if unknown:
            raise ValueError("F_ADJACENCY_INCIDENCE_PATCH_TRIANGLE_UNKNOWN")
        patch_by_key[patch.key] = patch
        triangle_seen.update(patch.source_triangle_ids)

    continuation_pairs: list[tuple[set[int], set[int]]] = []
    unjustified: list[tuple[PatchInstanceKeyV1, PatchInstanceKeyV1]] = []
    for continuation in continuations:
        continuation.validate()
        if continuation.a not in patch_by_key or continuation.b not in patch_by_key:
            raise ValueError("F_ADJACENCY_INCIDENCE_CONTINUATION_PATCH_UNKNOWN")
        a_ids = set(patch_by_key[continuation.a].source_triangle_ids)
        b_ids = set(patch_by_key[continuation.b].source_triangle_ids)
        continuation_pairs.append((a_ids, b_ids))
        shared = bool(a_ids & b_ids)
        adjacent = any(
            ((a, b) if a < b else (b, a)) in source_pairs
            for a in a_ids
            for b in b_ids
            if a != b
        )
        if not shared and not adjacent:
            unjustified.append((continuation.a, continuation.b))

    intra_witnessed: set[tuple[int, int]] = set()
    for patch in patches:
        ids = set(patch.source_triangle_ids)
        for pair in source_pairs:
            if pair[0] in ids and pair[1] in ids:
                intra_witnessed.add(pair)

    cross_witnessed: set[tuple[int, int]] = set()
    remaining = source_pairs - intra_witnessed
    for pair in remaining:
        a, b = pair
        for left, right in continuation_pairs:
            if (a in left and b in right) or (b in left and a in right):
                cross_witnessed.add(pair)
                break

    missing_pairs = tuple(sorted(source_pairs - intra_witnessed - cross_witnessed))
    missing_triangles = tuple(sorted(source_triangles - triangle_seen))
    report = FAdjacencyIncidenceReportV1(
        schema_version=SCHEMA_VERSION,
        source_triangle_count=len(source_triangles),
        source_adjacency_count=len(source_pairs),
        witnessed_intra_patch_count=len(intra_witnessed),
        witnessed_cross_cell_count=len(cross_witnessed),
        missing_source_adjacencies=missing_pairs,
        missing_source_triangles=missing_triangles,
        continuation_edge_count=len(continuations),
        unjustified_continuations=tuple(sorted(unjustified)),
    )
    report.validate()
    return report
