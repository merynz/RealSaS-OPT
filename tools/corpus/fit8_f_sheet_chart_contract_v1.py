from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

SCHEMA_VERSION = "RealSaS.FSheetChartContract.v1"
AREA_SUPPORT = "AREA_SUPPORT"


@dataclass(frozen=True, order=True)
class SheetPatchKeyV1:
    cell_xyz: tuple[int, int, int]
    patch_local_id: int

    def validate(self) -> None:
        if len(self.cell_xyz) != 3:
            raise ValueError("F_SHEET_CHART_CELL_INVALID")
        if self.patch_local_id < 0:
            raise ValueError("F_SHEET_CHART_PATCH_ID_INVALID")


@dataclass(frozen=True)
class SheetPatchRecordV1:
    key: SheetPatchKeyV1
    source_component_id: int
    support_dimension: str
    authored_open_boundary: bool
    authored_nonmanifold_incidence: bool
    provenance: str

    def validate(self) -> None:
        self.key.validate()
        if self.source_component_id < 0:
            raise ValueError("F_SHEET_CHART_COMPONENT_ID_INVALID")
        if self.support_dimension != AREA_SUPPORT:
            raise ValueError("F_SHEET_CHART_REQUIRES_AREA_SUPPORT_PATCH")
        if not self.provenance:
            raise ValueError("F_SHEET_CHART_PATCH_PROVENANCE_MISSING")


@dataclass(frozen=True)
class OrientedSheetContinuationV1:
    a: SheetPatchKeyV1
    b: SheetPatchKeyV1
    orientation_transport: int  # +1 same chart gauge, -1 flipped local gauge
    provenance: str

    def validate(self) -> None:
        self.a.validate()
        self.b.validate()
        if self.a == self.b:
            raise ValueError("F_SHEET_CHART_SELF_CONTINUATION")
        if self.orientation_transport not in {-1, 1}:
            raise ValueError("F_SHEET_CHART_ORIENTATION_TRANSPORT_INVALID")
        delta = tuple(abs(x - y) for x, y in zip(self.a.cell_xyz, self.b.cell_xyz))
        if sum(delta) != 1:
            raise ValueError("F_SHEET_CHART_CONTINUATION_NOT_FACE_NEIGHBOR")
        if not self.provenance:
            raise ValueError("F_SHEET_CHART_CONTINUATION_PROVENANCE_MISSING")


@dataclass(frozen=True)
class FSheetChartV1:
    chart_id: int
    patches: tuple[SheetPatchKeyV1, ...]
    patch_orientation_signs: tuple[tuple[SheetPatchKeyV1, int], ...]
    source_component_id: int
    orientation_consistent: bool
    has_authored_open_boundary: bool
    has_authored_nonmanifold_incidence: bool
    base_sheet_scalar_admitted: bool
    admission_reason: str

    def validate(self) -> None:
        if self.chart_id < 0:
            raise ValueError("F_SHEET_CHART_ID_INVALID")
        if not self.patches:
            raise ValueError("F_SHEET_CHART_EMPTY")
        if tuple(sorted(set(self.patches))) != self.patches:
            raise ValueError("F_SHEET_CHART_PATCH_SET_NOT_CANONICAL")
        sign_map = dict(self.patch_orientation_signs)
        if set(sign_map) != set(self.patches):
            raise ValueError("F_SHEET_CHART_ORIENTATION_COVERAGE_MISMATCH")
        if any(value not in {-1, 1} for value in sign_map.values()):
            raise ValueError("F_SHEET_CHART_ORIENTATION_SIGN_INVALID")
        if self.source_component_id < 0:
            raise ValueError("F_SHEET_CHART_COMPONENT_ID_INVALID")
        expected_admission = self.orientation_consistent and not self.has_authored_nonmanifold_incidence
        if self.base_sheet_scalar_admitted != expected_admission:
            raise ValueError("F_SHEET_CHART_ADMISSION_STATE_INVALID")
        if not self.admission_reason:
            raise ValueError("F_SHEET_CHART_ADMISSION_REASON_MISSING")


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


def build_sheet_charts_v1(
    patches: Sequence[SheetPatchRecordV1],
    continuations: Sequence[OrientedSheetContinuationV1],
) -> tuple[FSheetChartV1, ...]:
    """Build chart-local oriented scalar domains from qualified local sheets.

    A chart sign is a local orientation gauge, not a global inside/outside claim.
    Open orientable surfaces are therefore admissible. Non-manifold incidence or
    orientation transport inconsistency makes the base sheet-scalar path abstain;
    those regions require F-R/F-C explicit topology handling.
    """
    if not patches:
        raise ValueError("F_SHEET_CHART_PATCHES_EMPTY")
    patch_map: dict[SheetPatchKeyV1, SheetPatchRecordV1] = {}
    for patch in patches:
        patch.validate()
        if patch.key in patch_map:
            raise ValueError("F_SHEET_CHART_DUPLICATE_PATCH_KEY")
        patch_map[patch.key] = patch

    uf = _UnionFind(tuple(patch_map))
    continuation_map: dict[SheetPatchKeyV1, list[tuple[SheetPatchKeyV1, int]]] = {
        key: [] for key in patch_map
    }
    seen_edges: set[tuple[SheetPatchKeyV1, SheetPatchKeyV1]] = set()
    for continuation in continuations:
        continuation.validate()
        if continuation.a not in patch_map or continuation.b not in patch_map:
            raise ValueError("F_SHEET_CHART_CONTINUATION_PATCH_UNKNOWN")
        edge = tuple(sorted((continuation.a, continuation.b)))
        if edge in seen_edges:
            raise ValueError("F_SHEET_CHART_DUPLICATE_CONTINUATION")
        seen_edges.add(edge)
        if patch_map[continuation.a].source_component_id != patch_map[continuation.b].source_component_id:
            raise ValueError("F_SHEET_CHART_CONTINUATION_MERGES_SOURCE_COMPONENTS")
        uf.union(continuation.a, continuation.b)
        continuation_map[continuation.a].append(
            (continuation.b, continuation.orientation_transport)
        )
        continuation_map[continuation.b].append(
            (continuation.a, continuation.orientation_transport)
        )

    groups: dict[SheetPatchKeyV1, list[SheetPatchKeyV1]] = {}
    for key in patch_map:
        groups.setdefault(uf.find(key), []).append(key)
    ordered_groups = sorted(
        (tuple(sorted(values)) for values in groups.values()),
        key=lambda values: values[0],
    )

    charts: list[FSheetChartV1] = []
    for chart_id, keys in enumerate(ordered_groups):
        component_ids = {patch_map[key].source_component_id for key in keys}
        if len(component_ids) != 1:
            raise ValueError("F_SHEET_CHART_GROUP_MERGES_SOURCE_COMPONENTS")

        signs: dict[SheetPatchKeyV1, int] = {}
        orientation_consistent = True
        for seed in keys:
            if seed in signs:
                continue
            signs[seed] = 1
            stack = [seed]
            while stack:
                current = stack.pop()
                for neighbor, transport in continuation_map[current]:
                    if neighbor not in keys:
                        continue
                    expected = signs[current] * transport
                    if neighbor in signs:
                        if signs[neighbor] != expected:
                            orientation_consistent = False
                    else:
                        signs[neighbor] = expected
                        stack.append(neighbor)

        has_open = any(patch_map[key].authored_open_boundary for key in keys)
        has_nonmanifold = any(
            patch_map[key].authored_nonmanifold_incidence for key in keys
        )
        admitted = orientation_consistent and not has_nonmanifold
        if not orientation_consistent:
            reason = "ORIENTATION_TRANSPORT_CONFLICT"
        elif has_nonmanifold:
            reason = "AUTHORED_NONMANIFOLD_REQUIRES_EXPLICIT_TOPOLOGY"
        elif has_open:
            reason = "OPEN_ORIENTABLE_CHART_LOCAL_SIGN_ADMITTED"
        else:
            reason = "ORIENTABLE_CHART_LOCAL_SIGN_ADMITTED"

        chart = FSheetChartV1(
            chart_id=chart_id,
            patches=keys,
            patch_orientation_signs=tuple(sorted(signs.items())),
            source_component_id=next(iter(component_ids)),
            orientation_consistent=orientation_consistent,
            has_authored_open_boundary=has_open,
            has_authored_nonmanifold_incidence=has_nonmanifold,
            base_sheet_scalar_admitted=admitted,
            admission_reason=reason,
        )
        chart.validate()
        charts.append(chart)

    return tuple(charts)


__all__ = [
    "AREA_SUPPORT",
    "FSheetChartV1",
    "OrientedSheetContinuationV1",
    "SCHEMA_VERSION",
    "SheetPatchKeyV1",
    "SheetPatchRecordV1",
    "build_sheet_charts_v1",
]
