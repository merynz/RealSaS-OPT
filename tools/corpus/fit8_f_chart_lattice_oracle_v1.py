from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence

from tools.corpus.fit8_f_sheet_chart_contract_v1 import FSheetChartV1, SheetPatchKeyV1

SCHEMA_VERSION = "RealSaS.FChartLatticeOracle.v1"

CUBE_CORNER_OFFSETS = (
    (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
    (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1),
)
CUBE_EDGES = (
    (0, 1), (2, 3), (4, 5), (6, 7),
    (0, 2), (1, 3), (4, 6), (5, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
)

BASE_COMPATIBLE = "BASE_COMPATIBLE"
RESIDUAL_REQUIRED = "RESIDUAL_REQUIRED"
SCALAR_INCOMPATIBLE = "SCALAR_INCOMPATIBLE"


@dataclass(frozen=True, order=True)
class ChartLatticeVertexKeyV1:
    chart_id: int
    lattice_ijk: tuple[int, int, int]

    def validate(self) -> None:
        if self.chart_id < 0:
            raise ValueError("F_CHART_LATTICE_CHART_ID_INVALID")
        if len(self.lattice_ijk) != 3:
            raise ValueError("F_CHART_LATTICE_VERTEX_IJK_INVALID")


@dataclass(frozen=True)
class ChartLatticeVertexSampleV1:
    key: ChartLatticeVertexKeyV1
    oriented_scalar: float
    unsigned_distance: float
    orientation_valid: bool
    source_binding: str
    provenance: str

    def validate(self) -> None:
        self.key.validate()
        if not isfinite(self.oriented_scalar):
            raise ValueError("F_CHART_LATTICE_SCALAR_NONFINITE")
        if not isfinite(self.unsigned_distance) or self.unsigned_distance < 0.0:
            raise ValueError("F_CHART_LATTICE_UDF_INVALID")
        if not self.source_binding:
            raise ValueError("F_CHART_LATTICE_SOURCE_BINDING_MISSING")
        if not self.provenance:
            raise ValueError("F_CHART_LATTICE_SAMPLE_PROVENANCE_MISSING")


@dataclass(frozen=True)
class ChartCellScalarTruthV1:
    chart_id: int
    cell_xyz: tuple[int, int, int]
    patch_local_ids: tuple[int, ...]
    source_component_id: int
    teacher_edge_crossing_counts: tuple[int, ...]  # 12 entries, chart-local
    topology_residual_reasons: tuple[str, ...]
    provenance: str

    def validate(self) -> None:
        if self.chart_id < 0:
            raise ValueError("F_CHART_LATTICE_CELL_CHART_ID_INVALID")
        if len(self.cell_xyz) != 3:
            raise ValueError("F_CHART_LATTICE_CELL_XYZ_INVALID")
        if not self.patch_local_ids:
            raise ValueError("F_CHART_LATTICE_CELL_PATCHES_EMPTY")
        if tuple(sorted(set(self.patch_local_ids))) != self.patch_local_ids:
            raise ValueError("F_CHART_LATTICE_CELL_PATCH_IDS_NOT_CANONICAL")
        if self.source_component_id < 0:
            raise ValueError("F_CHART_LATTICE_COMPONENT_ID_INVALID")
        if len(self.teacher_edge_crossing_counts) != len(CUBE_EDGES):
            raise ValueError("F_CHART_LATTICE_EDGE_COUNT_ARITY_INVALID")
        if any(value < 0 for value in self.teacher_edge_crossing_counts):
            raise ValueError("F_CHART_LATTICE_EDGE_COUNT_NEGATIVE")
        if tuple(sorted(set(self.topology_residual_reasons))) != self.topology_residual_reasons:
            raise ValueError("F_CHART_LATTICE_RESIDUAL_REASONS_NOT_CANONICAL")
        if not self.provenance:
            raise ValueError("F_CHART_LATTICE_CELL_PROVENANCE_MISSING")

    def expected_corner_keys(self) -> tuple[ChartLatticeVertexKeyV1, ...]:
        x, y, z = self.cell_xyz
        return tuple(
            ChartLatticeVertexKeyV1(
                self.chart_id,
                (x + dx, y + dy, z + dz),
            )
            for dx, dy, dz in CUBE_CORNER_OFFSETS
        )


@dataclass(frozen=True)
class ChartCellRepresentabilityV1:
    chart_id: int
    cell_xyz: tuple[int, int, int]
    classification: str
    reasons: tuple[str, ...]
    teacher_crossing_count: int
    scalar_crossing_count: int
    max_teacher_crossings_per_edge: int


@dataclass(frozen=True)
class FChartLatticeReportV1:
    schema_version: str
    chart_id: int
    cell_count: int
    base_compatible_cells: int
    residual_required_cells: int
    scalar_incompatible_cells: int
    teacher_crossing_count: int
    scalar_crossing_count: int
    max_teacher_crossings_per_edge: int
    rows: tuple[ChartCellRepresentabilityV1, ...]

    @property
    def scalar_field_matches_all_base_eligible_crossings(self) -> bool:
        return self.scalar_incompatible_cells == 0

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("F_CHART_LATTICE_REPORT_SCHEMA_INVALID")
        counts = (
            self.base_compatible_cells,
            self.residual_required_cells,
            self.scalar_incompatible_cells,
        )
        if any(value < 0 for value in counts):
            raise ValueError("F_CHART_LATTICE_REPORT_NEGATIVE_COUNT")
        if sum(counts) != self.cell_count:
            raise ValueError("F_CHART_LATTICE_REPORT_CELL_COUNT_MISMATCH")
        if len(self.rows) != self.cell_count:
            raise ValueError("F_CHART_LATTICE_REPORT_ROW_COUNT_MISMATCH")


@dataclass(frozen=True)
class FChartLatticeOracleV1:
    chart: FSheetChartV1
    extraction_pitch: float
    zero_epsilon: float
    vertex_samples: tuple[ChartLatticeVertexSampleV1, ...]
    cells: tuple[ChartCellScalarTruthV1, ...]
    provenance: str
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("F_CHART_LATTICE_SCHEMA_INVALID")
        self.chart.validate()
        if not self.chart.base_sheet_scalar_admitted:
            raise ValueError("F_CHART_LATTICE_REQUIRES_ADMITTED_BASE_CHART")
        if not isfinite(self.extraction_pitch) or self.extraction_pitch <= 0.0:
            raise ValueError("F_CHART_LATTICE_PITCH_INVALID")
        if not isfinite(self.zero_epsilon) or self.zero_epsilon < 0.0:
            raise ValueError("F_CHART_LATTICE_ZERO_EPSILON_INVALID")
        if not self.provenance:
            raise ValueError("F_CHART_LATTICE_PROVENANCE_MISSING")

        sample_map: dict[ChartLatticeVertexKeyV1, ChartLatticeVertexSampleV1] = {}
        for sample in self.vertex_samples:
            sample.validate()
            if sample.key.chart_id != self.chart.chart_id:
                raise ValueError("F_CHART_LATTICE_SAMPLE_CHART_DRIFT")
            if sample.key in sample_map:
                raise ValueError("F_CHART_LATTICE_DUPLICATE_VERTEX_SAMPLE")
            sample_map[sample.key] = sample

        chart_patches = set(self.chart.patches)
        seen_cells: set[tuple[int, int, int]] = set()
        for cell in self.cells:
            cell.validate()
            if cell.chart_id != self.chart.chart_id:
                raise ValueError("F_CHART_LATTICE_CELL_CHART_DRIFT")
            if cell.source_component_id != self.chart.source_component_id:
                raise ValueError("F_CHART_LATTICE_COMPONENT_DRIFT")
            if cell.cell_xyz in seen_cells:
                raise ValueError("F_CHART_LATTICE_DUPLICATE_CELL")
            seen_cells.add(cell.cell_xyz)
            expected_patch_keys = {
                SheetPatchKeyV1(cell.cell_xyz, patch_id)
                for patch_id in cell.patch_local_ids
            }
            if not expected_patch_keys <= chart_patches:
                raise ValueError("F_CHART_LATTICE_CELL_PATCH_NOT_IN_CHART")
            for key in cell.expected_corner_keys():
                if key not in sample_map:
                    raise ValueError("F_CHART_LATTICE_CORNER_SAMPLE_MISSING")

        # One chart-local grid vertex has exactly one scalar sample. A different
        # chart at the same physical ijk is intentionally a different identity.
        used = {
            key
            for cell in self.cells
            for key in cell.expected_corner_keys()
        }
        if used != set(sample_map):
            raise ValueError("F_CHART_LATTICE_UNUSED_OR_MISSING_SAMPLE")


def _edge_has_scalar_crossing(a: float, b: float, epsilon: float) -> tuple[bool, bool]:
    """Return (crossing, ambiguous_zero_edge)."""
    az = abs(a) <= epsilon
    bz = abs(b) <= epsilon
    if az and bz:
        return True, True
    if az or bz:
        return True, False
    return (a < 0.0) != (b < 0.0), False


def evaluate_chart_lattice_oracle_v1(
    oracle: FChartLatticeOracleV1,
) -> FChartLatticeReportV1:
    """Test whether one chart-local scalar lattice can express teacher crossings.

    The court is intentionally representation-limited. It does not claim that a
    matching edge crossing pattern guarantees correct face/cell topology. Explicit
    teacher topology residual reasons (closed loop, open endpoint, branch,
    authored boundary, etc.) therefore force `RESIDUAL_REQUIRED` even when all
    corner signs match. More than one teacher crossing on a chart edge also forces
    residual capacity because a single scalar interpolation cannot preserve both.

    A sign mismatch on an otherwise base-eligible edge is `SCALAR_INCOMPATIBLE`,
    not an excuse to fabricate residual topology. This distinguishes a bad chart
    field from a genuine representation-capacity exception.
    """
    oracle.validate()
    sample_map = {sample.key: sample for sample in oracle.vertex_samples}

    rows: list[ChartCellRepresentabilityV1] = []
    total_teacher = 0
    total_scalar = 0
    max_teacher = 0

    for cell in sorted(oracle.cells, key=lambda row: row.cell_xyz):
        samples = [sample_map[key] for key in cell.expected_corner_keys()]
        reasons: list[str] = list(cell.topology_residual_reasons)
        incompatible: list[str] = []
        scalar_crossings = 0

        if any(not sample.orientation_valid for sample in samples):
            incompatible.append("ORIENTATION_VALIDITY_MISSING")

        for edge_index, (a_index, b_index) in enumerate(CUBE_EDGES):
            teacher_count = int(cell.teacher_edge_crossing_counts[edge_index])
            total_teacher += teacher_count
            max_teacher = max(max_teacher, teacher_count)
            scalar_crossing, ambiguous = _edge_has_scalar_crossing(
                samples[a_index].oriented_scalar,
                samples[b_index].oriented_scalar,
                oracle.zero_epsilon,
            )
            scalar_crossings += int(scalar_crossing)
            if ambiguous:
                reasons.append(f"AMBIGUOUS_ZERO_EDGE:{edge_index}")
            if teacher_count > 1:
                reasons.append(f"MULTI_CROSSING_EDGE:{edge_index}:{teacher_count}")
            elif teacher_count == 1 and not scalar_crossing:
                incompatible.append(f"MISSING_SCALAR_CROSSING:{edge_index}")
            elif teacher_count == 0 and scalar_crossing:
                incompatible.append(f"SPURIOUS_SCALAR_CROSSING:{edge_index}")

        total_scalar += scalar_crossings
        reasons = sorted(set(reasons))
        incompatible = sorted(set(incompatible))
        if incompatible:
            classification = SCALAR_INCOMPATIBLE
            final_reasons = tuple(incompatible + reasons)
        elif reasons:
            classification = RESIDUAL_REQUIRED
            final_reasons = tuple(reasons)
        else:
            classification = BASE_COMPATIBLE
            final_reasons = ()

        rows.append(
            ChartCellRepresentabilityV1(
                chart_id=cell.chart_id,
                cell_xyz=cell.cell_xyz,
                classification=classification,
                reasons=final_reasons,
                teacher_crossing_count=sum(cell.teacher_edge_crossing_counts),
                scalar_crossing_count=scalar_crossings,
                max_teacher_crossings_per_edge=max(cell.teacher_edge_crossing_counts),
            )
        )

    report = FChartLatticeReportV1(
        schema_version=SCHEMA_VERSION,
        chart_id=oracle.chart.chart_id,
        cell_count=len(rows),
        base_compatible_cells=sum(row.classification == BASE_COMPATIBLE for row in rows),
        residual_required_cells=sum(row.classification == RESIDUAL_REQUIRED for row in rows),
        scalar_incompatible_cells=sum(row.classification == SCALAR_INCOMPATIBLE for row in rows),
        teacher_crossing_count=total_teacher,
        scalar_crossing_count=total_scalar,
        max_teacher_crossings_per_edge=max_teacher,
        rows=tuple(rows),
    )
    report.validate()
    return report


@dataclass(frozen=True)
class MultiChartLatticeBundleV1:
    oracles: tuple[FChartLatticeOracleV1, ...]

    def validate(self) -> None:
        if not self.oracles:
            raise ValueError("F_CHART_LATTICE_BUNDLE_EMPTY")
        chart_ids: set[int] = set()
        identities: set[ChartLatticeVertexKeyV1] = set()
        for oracle in self.oracles:
            oracle.validate()
            if oracle.chart.chart_id in chart_ids:
                raise ValueError("F_CHART_LATTICE_BUNDLE_DUPLICATE_CHART")
            chart_ids.add(oracle.chart.chart_id)
            for sample in oracle.vertex_samples:
                if sample.key in identities:
                    raise ValueError("F_CHART_LATTICE_BUNDLE_IDENTITY_ALIAS")
                identities.add(sample.key)

    @property
    def physical_grid_vertex_overlap_count(self) -> int:
        self.validate()
        by_ijk: dict[tuple[int, int, int], int] = {}
        for oracle in self.oracles:
            for sample in oracle.vertex_samples:
                by_ijk[sample.key.lattice_ijk] = by_ijk.get(sample.key.lattice_ijk, 0) + 1
        return sum(count > 1 for count in by_ijk.values())


__all__ = [
    "BASE_COMPATIBLE",
    "CUBE_CORNER_OFFSETS",
    "CUBE_EDGES",
    "ChartCellRepresentabilityV1",
    "ChartCellScalarTruthV1",
    "ChartLatticeVertexKeyV1",
    "ChartLatticeVertexSampleV1",
    "FChartLatticeOracleV1",
    "FChartLatticeReportV1",
    "MultiChartLatticeBundleV1",
    "RESIDUAL_REQUIRED",
    "SCALAR_INCOMPATIBLE",
    "SCHEMA_VERSION",
    "evaluate_chart_lattice_oracle_v1",
]
