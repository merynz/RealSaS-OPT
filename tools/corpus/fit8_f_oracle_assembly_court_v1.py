from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

SCHEMA_VERSION = "RealSaS.FOracleAssemblyCourt.v1"


@dataclass(frozen=True, order=True)
class PatchInstanceKeyV1:
    cell_xyz: tuple[int, int, int]
    patch_local_id: int

    def validate(self) -> None:
        if len(self.cell_xyz) != 3:
            raise ValueError("F_ORACLE_ASSEMBLY_CELL_COORD_INVALID")
        if self.patch_local_id < 0:
            raise ValueError("F_ORACLE_ASSEMBLY_PATCH_ID_INVALID")


@dataclass(frozen=True)
class PatchInstanceV1:
    key: PatchInstanceKeyV1
    source_triangle_ids: tuple[int, ...]

    def validate(self) -> None:
        self.key.validate()
        if not self.source_triangle_ids:
            raise ValueError("F_ORACLE_ASSEMBLY_PATCH_EMPTY")
        if tuple(sorted(set(self.source_triangle_ids))) != self.source_triangle_ids:
            raise ValueError("F_ORACLE_ASSEMBLY_TRIANGLE_IDS_NOT_CANONICAL")


@dataclass(frozen=True)
class PatchContinuationV1:
    a: PatchInstanceKeyV1
    b: PatchInstanceKeyV1

    def validate(self) -> None:
        self.a.validate()
        self.b.validate()
        if self.a == self.b:
            raise ValueError("F_ORACLE_ASSEMBLY_SELF_CONTINUATION")


@dataclass(frozen=True)
class FOracleAssemblyReportV1:
    schema_version: str
    source_component_count: int
    representation_component_count: int
    false_merge_components: int
    split_source_components: int
    missing_source_triangles: int
    max_representation_pieces_per_source_component: int

    @property
    def exact_component_partition(self) -> bool:
        return (
            self.false_merge_components == 0
            and self.split_source_components == 0
            and self.missing_source_triangles == 0
            and self.source_component_count == self.representation_component_count
        )

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("F_ORACLE_ASSEMBLY_SCHEMA_INVALID")
        for name in (
            "source_component_count",
            "representation_component_count",
            "false_merge_components",
            "split_source_components",
            "missing_source_triangles",
            "max_representation_pieces_per_source_component",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"F_ORACLE_ASSEMBLY_NEGATIVE:{name}")


class _UnionFind:
    def __init__(self, count: int) -> None:
        self.parent = list(range(count))
        self.rank = [0] * count

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, a: int, b: int) -> None:
        a = self.find(a)
        b = self.find(b)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def evaluate_oracle_patch_assembly_v1(
    source_triangle_component: Mapping[int, int],
    patches: Sequence[PatchInstanceV1],
    continuations: Sequence[PatchContinuationV1],
) -> FOracleAssemblyReportV1:
    """Measure only global connected-component partition closure.

    Cell-local patch IDs are not global identities. The supplied continuations are
    the qualified cross-cell lineage relations. This court checks whether their
    transitive closure reconstructs the exact source component partition without
    split, false merge or triangle loss.

    It deliberately does *not* claim Euler/genus/boundary-loop or geometric mesh
    fidelity; those require stronger topology courts.
    """
    if not source_triangle_component:
        raise ValueError("F_ORACLE_ASSEMBLY_SOURCE_EMPTY")

    key_to_index: dict[PatchInstanceKeyV1, int] = {}
    triangle_seen: set[int] = set()
    for index, patch in enumerate(patches):
        patch.validate()
        if patch.key in key_to_index:
            raise ValueError("F_ORACLE_ASSEMBLY_DUPLICATE_PATCH_KEY")
        key_to_index[patch.key] = index
        for triangle_id in patch.source_triangle_ids:
            if triangle_id not in source_triangle_component:
                raise ValueError("F_ORACLE_ASSEMBLY_TRIANGLE_UNKNOWN")
            triangle_seen.add(triangle_id)

    union_find = _UnionFind(len(patches))
    for continuation in continuations:
        continuation.validate()
        if continuation.a not in key_to_index or continuation.b not in key_to_index:
            raise ValueError("F_ORACLE_ASSEMBLY_CONTINUATION_PATCH_UNKNOWN")
        union_find.union(key_to_index[continuation.a], key_to_index[continuation.b])

    roots: dict[int, list[PatchInstanceV1]] = {}
    for index, patch in enumerate(patches):
        roots.setdefault(union_find.find(index), []).append(patch)

    source_components = set(map(int, source_triangle_component.values()))
    source_to_roots: dict[int, set[int]] = {component: set() for component in source_components}
    false_merge_components = 0
    for root, root_patches in roots.items():
        root_components = {
            int(source_triangle_component[triangle_id])
            for patch in root_patches
            for triangle_id in patch.source_triangle_ids
        }
        if len(root_components) > 1:
            false_merge_components += 1
        for component in root_components:
            source_to_roots[component].add(root)

    split_source_components = sum(len(root_set) > 1 for root_set in source_to_roots.values())
    max_pieces = max((len(root_set) for root_set in source_to_roots.values()), default=0)
    missing_source_triangles = len(set(source_triangle_component) - triangle_seen)

    report = FOracleAssemblyReportV1(
        schema_version=SCHEMA_VERSION,
        source_component_count=len(source_components),
        representation_component_count=len(roots),
        false_merge_components=false_merge_components,
        split_source_components=split_source_components,
        missing_source_triangles=missing_source_triangles,
        max_representation_pieces_per_source_component=max_pieces,
    )
    report.validate()
    return report
