from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from tools.corpus.fit8_f_sheet_chart_contract_v1 import (
    FSheetChartV1,
    OrientedSheetContinuationV1,
    SheetPatchKeyV1,
    SheetPatchRecordV1,
)

SCHEMA_VERSION = "RealSaS.FSheetChartContract.v2"


@dataclass(frozen=True)
class ResidualBoundaryContinuationV2:
    manifold_patch: SheetPatchKeyV1
    residual_patch: SheetPatchKeyV1
    orientation_transport: int
    provenance: str

    def validate(self) -> None:
        self.manifold_patch.validate()
        self.residual_patch.validate()
        if self.manifold_patch == self.residual_patch:
            raise ValueError("F_SHEET_CHART_V2_RESIDUAL_SELF_CONTINUATION")
        if self.orientation_transport not in {-1, 1}:
            raise ValueError("F_SHEET_CHART_V2_RESIDUAL_ORIENTATION_INVALID")
        if not self.provenance:
            raise ValueError("F_SHEET_CHART_V2_RESIDUAL_PROVENANCE_MISSING")


@dataclass(frozen=True)
class FSheetChartPartitionV2:
    schema_version: str
    base_charts: tuple[FSheetChartV1, ...]
    singular_patch_keys: tuple[SheetPatchKeyV1, ...]
    topology_residual_patch_keys: tuple[SheetPatchKeyV1, ...]
    residual_boundary_continuations: tuple[ResidualBoundaryContinuationV2, ...]

    def validate(self, all_patch_keys: set[SheetPatchKeyV1]) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("F_SHEET_CHART_V2_SCHEMA_INVALID")
        chart_keys = {
            key
            for chart in self.base_charts
            for key in chart.patches
        }
        singular = set(self.singular_patch_keys)
        residual = set(self.topology_residual_patch_keys)
        if chart_keys & singular:
            raise ValueError("F_SHEET_CHART_V2_SINGULAR_PATCH_IN_BASE_CHART")
        if chart_keys | singular != all_patch_keys:
            raise ValueError("F_SHEET_CHART_V2_PATCH_COVERAGE_MISMATCH")
        if not singular <= residual:
            raise ValueError("F_SHEET_CHART_V2_SINGULAR_NOT_RESIDUAL")
        if not residual <= all_patch_keys:
            raise ValueError("F_SHEET_CHART_V2_RESIDUAL_PATCH_UNKNOWN")
        for chart in self.base_charts:
            chart.validate()
            if chart.has_authored_nonmanifold_incidence:
                raise ValueError("F_SHEET_CHART_V2_BASE_CHART_CONTAINS_NONMANIFOLD")
        for row in self.residual_boundary_continuations:
            row.validate()
            if row.manifold_patch not in chart_keys:
                raise ValueError("F_SHEET_CHART_V2_RESIDUAL_BOUNDARY_BASE_PATCH_UNKNOWN")
            if row.residual_patch not in singular:
                raise ValueError("F_SHEET_CHART_V2_RESIDUAL_BOUNDARY_SINGULAR_PATCH_UNKNOWN")


class _UnionFind:
    def __init__(self, values: Sequence[SheetPatchKeyV1]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: SheetPatchKeyV1) -> SheetPatchKeyV1:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, a: SheetPatchKeyV1, b: SheetPatchKeyV1) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if rb < ra:
            ra, rb = rb, ra
        self.parent[rb] = ra


def build_sheet_chart_partition_v2(
    patches: Sequence[SheetPatchRecordV1],
    continuations: Sequence[OrientedSheetContinuationV1],
) -> FSheetChartPartitionV2:
    """Build manifold chart-local scalar domains and localize topology residuals.

    Evidence from the Bat physical oracle showed that authored non-manifold support
    is sparse and local. V1 conservatively made an entire connected chart abstain
    if any member patch was singular. V2 instead removes singular patches from the
    base scalar graph, preserves their incidence to neighboring manifold charts,
    and marks only the singular support as mandatory explicit topology residual.

    Authored open-boundary patches remain in orientable base charts because a
    chart-local oriented field is still useful for their geometry. They are also
    marked as topology-residual patches: an ordinary scalar zero-set cannot end in
    the interior of its domain without explicit support/boundary semantics.
    """
    if not patches:
        raise ValueError("F_SHEET_CHART_V2_PATCHES_EMPTY")

    patch_map: dict[SheetPatchKeyV1, SheetPatchRecordV1] = {}
    for patch in patches:
        patch.validate()
        if patch.key in patch_map:
            raise ValueError("F_SHEET_CHART_V2_DUPLICATE_PATCH_KEY")
        patch_map[patch.key] = patch

    singular = {
        key for key, patch in patch_map.items()
        if patch.authored_nonmanifold_incidence
    }
    base_keys = tuple(sorted(set(patch_map) - singular))
    uf = _UnionFind(base_keys)
    adjacency: dict[SheetPatchKeyV1, list[tuple[SheetPatchKeyV1, int]]] = {
        key: [] for key in base_keys
    }
    residual_boundary_rows: list[ResidualBoundaryContinuationV2] = []
    seen_edges: set[tuple[SheetPatchKeyV1, SheetPatchKeyV1]] = set()

    for continuation in continuations:
        continuation.validate()
        if continuation.a not in patch_map or continuation.b not in patch_map:
            raise ValueError("F_SHEET_CHART_V2_CONTINUATION_PATCH_UNKNOWN")
        edge = tuple(sorted((continuation.a, continuation.b)))
        if edge in seen_edges:
            raise ValueError("F_SHEET_CHART_V2_DUPLICATE_CONTINUATION")
        seen_edges.add(edge)
        a_record = patch_map[continuation.a]
        b_record = patch_map[continuation.b]
        if a_record.source_component_id != b_record.source_component_id:
            raise ValueError("F_SHEET_CHART_V2_CONTINUATION_MERGES_SOURCE_COMPONENTS")

        a_singular = continuation.a in singular
        b_singular = continuation.b in singular
        if a_singular and b_singular:
            # Singular-to-singular connectivity belongs entirely to explicit
            # topology residual state and is deliberately not projected into the
            # scalar chart graph.
            continue
        if a_singular != b_singular:
            manifold = continuation.b if a_singular else continuation.a
            residual = continuation.a if a_singular else continuation.b
            residual_boundary_rows.append(
                ResidualBoundaryContinuationV2(
                    manifold_patch=manifold,
                    residual_patch=residual,
                    orientation_transport=continuation.orientation_transport,
                    provenance=continuation.provenance,
                )
            )
            continue

        uf.union(continuation.a, continuation.b)
        adjacency[continuation.a].append(
            (continuation.b, continuation.orientation_transport)
        )
        adjacency[continuation.b].append(
            (continuation.a, continuation.orientation_transport)
        )

    groups: dict[SheetPatchKeyV1, list[SheetPatchKeyV1]] = {}
    for key in base_keys:
        groups.setdefault(uf.find(key), []).append(key)
    ordered_groups = sorted(
        (tuple(sorted(values)) for values in groups.values()),
        key=lambda values: values[0],
    )

    charts: list[FSheetChartV1] = []
    for chart_id, keys in enumerate(ordered_groups):
        components = {patch_map[key].source_component_id for key in keys}
        if len(components) != 1:
            raise ValueError("F_SHEET_CHART_V2_GROUP_MERGES_SOURCE_COMPONENTS")

        signs: dict[SheetPatchKeyV1, int] = {}
        consistent = True
        for seed in keys:
            if seed in signs:
                continue
            signs[seed] = 1
            stack = [seed]
            while stack:
                current = stack.pop()
                for neighbor, transport in adjacency[current]:
                    expected = signs[current] * transport
                    if neighbor in signs:
                        if signs[neighbor] != expected:
                            consistent = False
                    else:
                        signs[neighbor] = expected
                        stack.append(neighbor)

        has_open = any(patch_map[key].authored_open_boundary for key in keys)
        chart = FSheetChartV1(
            chart_id=chart_id,
            patches=keys,
            patch_orientation_signs=tuple(sorted(signs.items())),
            source_component_id=next(iter(components)),
            orientation_consistent=consistent,
            has_authored_open_boundary=has_open,
            has_authored_nonmanifold_incidence=False,
            base_sheet_scalar_admitted=consistent,
            admission_reason=(
                "ORIENTATION_TRANSPORT_CONFLICT"
                if not consistent
                else (
                    "OPEN_ORIENTABLE_CHART_LOCAL_SIGN_ADMITTED"
                    if has_open
                    else "ORIENTABLE_CHART_LOCAL_SIGN_ADMITTED"
                )
            ),
        )
        chart.validate()
        charts.append(chart)

    residual = {
        key for key, patch in patch_map.items()
        if patch.authored_nonmanifold_incidence or patch.authored_open_boundary
    }
    result = FSheetChartPartitionV2(
        schema_version=SCHEMA_VERSION,
        base_charts=tuple(charts),
        singular_patch_keys=tuple(sorted(singular)),
        topology_residual_patch_keys=tuple(sorted(residual)),
        residual_boundary_continuations=tuple(sorted(
            residual_boundary_rows,
            key=lambda row: (row.manifold_patch, row.residual_patch),
        )),
    )
    result.validate(set(patch_map))
    return result


__all__ = [
    "FSheetChartPartitionV2",
    "ResidualBoundaryContinuationV2",
    "SCHEMA_VERSION",
    "build_sheet_chart_partition_v2",
]
