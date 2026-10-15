from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from tools.corpus.fit8_f_flexicubes_reference_adapter_v1 import FlexiCubesDenseGridV1
from tools.corpus.fit8_f_qualified_fa_bundle_v1 import (
    FQualifiedFABuildReportV1,
    QualifiedFAAuthorityRefsV1,
    build_qualified_fa_bundle_v1,
)
from tools.corpus.fit8_mesh_field_teacher_v1 import MeshFieldTeacherPolicyV1
from tools.corpus.fit8_static_self_intersection_qualification_v1 import StaticSelfIntersectionPolicyV1

SCHEMA_VERSION = "RealSaS.FQualifiedFABundleBuilder.v2"
ARRAY_CONTENT_HASH = "CANONICAL_ARRAY_CONTENT_SHA256_V1"
SOURCE_CARRIER_HASH = "CANONICAL_SOURCE_CARRIER_SHA256_V1"


def _canonical_array(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise ValueError("F_QUALIFIED_FA_V2_OBJECT_DTYPE_FORBIDDEN")
    if array.dtype.itemsize > 1:
        dtype = array.dtype.newbyteorder("<")
        if array.dtype != dtype or not array.dtype.isnative:
            array = array.astype(dtype, copy=False)
        elif array.dtype.byteorder == "=":
            array = array.astype(dtype, copy=False)
    return np.ascontiguousarray(array)


def canonical_array_content_sha256_v1(arrays: Mapping[str, np.ndarray]) -> str:
    """Hash semantic ndarray content independently of NPZ/ZIP compression bytes."""
    if not arrays:
        raise ValueError("F_QUALIFIED_FA_V2_ARRAYS_EMPTY")
    digest = hashlib.sha256()
    for name in sorted(arrays):
        if not name or "\x00" in name:
            raise ValueError("F_QUALIFIED_FA_V2_ARRAY_NAME_INVALID")
        value = _canonical_array(np.asarray(arrays[name]))
        dtype = value.dtype.str
        shape = json.dumps(list(value.shape), separators=(",", ":"), ensure_ascii=True)
        for text in (name, dtype, shape):
            payload = text.encode("utf-8")
            digest.update(len(payload).to_bytes(8, "little"))
            digest.update(payload)
        raw = value.tobytes(order="C")
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def canonical_source_carrier_sha256_v1(vertices: np.ndarray, faces: np.ndarray) -> str:
    xyz = np.asarray(vertices, dtype=np.float64)
    tri = np.asarray(faces, dtype=np.int64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError("F_QUALIFIED_FA_V2_SOURCE_VERTICES_INVALID")
    if tri.ndim != 2 or tri.shape[1] != 3 or not len(tri) or np.any(tri < 0) or np.any(tri >= len(xyz)):
        raise ValueError("F_QUALIFIED_FA_V2_SOURCE_FACES_INVALID")
    return canonical_array_content_sha256_v1({
        "faces_i64": tri.astype("<i8", copy=False),
        "vertices_f64": xyz.astype("<f8", copy=False),
    })


@dataclass(frozen=True)
class ExtractionVolumeQualificationV2:
    minimum_source_margin_cells: float
    measured_min_margin_cells: float
    boundary_vertex_count: int
    boundary_inside_count: int
    boundary_nonpositive_count: int
    qualified: bool
    provenance: str
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("F_QUALIFIED_FA_V2_VOLUME_SCHEMA_INVALID")
        if not np.isfinite(self.minimum_source_margin_cells) or self.minimum_source_margin_cells <= 0.0:
            raise ValueError("F_QUALIFIED_FA_V2_MARGIN_POLICY_INVALID")
        if not np.isfinite(self.measured_min_margin_cells):
            raise ValueError("F_QUALIFIED_FA_V2_MARGIN_MEASUREMENT_INVALID")
        if self.boundary_vertex_count < 1:
            raise ValueError("F_QUALIFIED_FA_V2_BOUNDARY_VERTEX_COUNT_INVALID")
        if self.boundary_inside_count < 0 or self.boundary_nonpositive_count < 0:
            raise ValueError("F_QUALIFIED_FA_V2_BOUNDARY_COUNT_INVALID")
        expected = bool(
            self.measured_min_margin_cells >= self.minimum_source_margin_cells
            and self.boundary_inside_count == 0
            and self.boundary_nonpositive_count == 0
        )
        if self.qualified != expected:
            raise ValueError("F_QUALIFIED_FA_V2_VOLUME_QUALIFICATION_MISMATCH")
        if not self.provenance:
            raise ValueError("F_QUALIFIED_FA_V2_VOLUME_PROVENANCE_MISSING")


@dataclass(frozen=True)
class FQualifiedFABuildReportV2:
    parent: FQualifiedFABuildReportV1
    source_carrier_sha256: str
    array_content_sha256: str
    transport_npz_sha256: str
    volume: ExtractionVolumeQualificationV2
    final_manifest_sha256: str
    provenance: str
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("F_QUALIFIED_FA_V2_REPORT_SCHEMA_INVALID")
        self.parent.validate()
        self.volume.validate()
        for value in (
            self.source_carrier_sha256,
            self.array_content_sha256,
            self.transport_npz_sha256,
            self.final_manifest_sha256,
        ):
            if len(value) != 64 or set(value) == {"0"}:
                raise ValueError("F_QUALIFIED_FA_V2_REPORT_SHA_INVALID")
        if self.transport_npz_sha256 != self.parent.bundle_npz_sha256:
            raise ValueError("F_QUALIFIED_FA_V2_TRANSPORT_SHA_DRIFT")
        if not self.volume.qualified:
            raise ValueError("F_QUALIFIED_FA_V2_VOLUME_NOT_QUALIFIED")
        if not self.provenance:
            raise ValueError("F_QUALIFIED_FA_V2_REPORT_PROVENANCE_MISSING")


def _grid_bounds(grid: FlexiCubesDenseGridV1) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(grid.vertex_xyz, dtype=np.float64)
    return xyz.min(axis=0), xyz.max(axis=0)


def _measured_margin_cells(
    vertices: np.ndarray,
    grid: FlexiCubesDenseGridV1,
) -> float:
    xyz = np.asarray(vertices, dtype=np.float64)
    lo, hi = _grid_bounds(grid)
    source_lo = xyz.min(axis=0)
    source_hi = xyz.max(axis=0)
    distances = np.concatenate((source_lo - lo, hi - source_hi))
    return float(np.min(distances) / float(grid.extraction_pitch))


def _boundary_mask(grid: FlexiCubesDenseGridV1) -> np.ndarray:
    ijk = np.asarray(grid.lattice_ijks, dtype=np.int64)
    lo = np.asarray(grid.cell_min_ijk, dtype=np.int64)
    hi = np.asarray(grid.cell_max_exclusive_ijk, dtype=np.int64)
    return np.any((ijk == lo[None, :]) | (ijk == hi[None, :]), axis=1)


def build_qualified_fa_bundle_v2(
    *,
    subject_key: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    grid: FlexiCubesDenseGridV1,
    authority_refs: QualifiedFAAuthorityRefsV1,
    mesh_field_policy: MeshFieldTeacherPolicyV1,
    self_intersection_policy: StaticSelfIntersectionPolicyV1,
    minimum_source_margin_cells: float,
    output_npz: Path,
    output_manifest: Path,
    output_grid_authority: Path,
    provenance: str,
) -> FQualifiedFABuildReportV2:
    """Add volume/shell and semantic-content qualification to the v1 F_A builder."""
    if not np.isfinite(minimum_source_margin_cells) or minimum_source_margin_cells <= 0.0:
        raise ValueError("F_QUALIFIED_FA_V2_MARGIN_POLICY_INVALID")
    margin = _measured_margin_cells(vertices, grid)
    if margin < minimum_source_margin_cells:
        raise ValueError(
            f"F_QUALIFIED_FA_V2_SOURCE_MARGIN_INSUFFICIENT:{margin:.12g}"
        )

    parent = build_qualified_fa_bundle_v1(
        subject_key=subject_key,
        vertices=vertices,
        faces=faces,
        grid=grid,
        authority_refs=authority_refs,
        mesh_field_policy=mesh_field_policy,
        self_intersection_policy=self_intersection_policy,
        output_npz=output_npz,
        output_manifest=output_manifest,
        output_grid_authority=output_grid_authority,
        provenance=f"{provenance};V1_PARENT",
    )

    with np.load(output_npz, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    content_sha = canonical_array_content_sha256_v1(arrays)
    carrier_sha = canonical_source_carrier_sha256_v1(vertices, faces)

    boundary = _boundary_mask(grid)
    inside = np.asarray(arrays["teacher_inside"], dtype=bool)
    oriented = np.asarray(arrays["teacher_oriented_level_set_f64"], dtype=np.float64)
    if inside.shape != boundary.shape or oriented.shape != boundary.shape:
        raise ValueError("F_QUALIFIED_FA_V2_BOUNDARY_ARRAY_SHAPE_INVALID")
    boundary_count = int(np.count_nonzero(boundary))
    boundary_inside = int(np.count_nonzero(inside[boundary]))
    boundary_nonpositive = int(
        np.count_nonzero(oriented[boundary] <= float(mesh_field_policy.spatial_tolerance))
    )
    volume = ExtractionVolumeQualificationV2(
        minimum_source_margin_cells=float(minimum_source_margin_cells),
        measured_min_margin_cells=margin,
        boundary_vertex_count=boundary_count,
        boundary_inside_count=boundary_inside,
        boundary_nonpositive_count=boundary_nonpositive,
        qualified=bool(
            margin >= minimum_source_margin_cells
            and boundary_inside == 0
            and boundary_nonpositive == 0
        ),
        provenance=f"{provenance};EXTRACTION_VOLUME_AND_OUTSIDE_SHELL",
    )
    volume.validate()
    if not volume.qualified:
        raise ValueError("F_QUALIFIED_FA_V2_EXTRACTION_VOLUME_NOT_QUALIFIED")

    manifest_raw = json.loads(output_manifest.read_text(encoding="utf-8"))
    metadata = dict(manifest_raw.get("metadata", {}))
    metadata.update({
        "qualified_fa_builder_v2_schema": SCHEMA_VERSION,
        "source_carrier_sha256": carrier_sha,
        "array_content_sha256": content_sha,
        "array_content_hash_semantics": ARRAY_CONTENT_HASH,
        "source_carrier_hash_semantics": SOURCE_CARRIER_HASH,
        "transport_npz_sha256": parent.bundle_npz_sha256,
        "minimum_source_margin_cells": volume.minimum_source_margin_cells,
        "measured_min_source_margin_cells": volume.measured_min_margin_cells,
        "boundary_vertex_count": volume.boundary_vertex_count,
        "boundary_inside_count": volume.boundary_inside_count,
        "boundary_nonpositive_count": volume.boundary_nonpositive_count,
        "outside_shell_qualified": True,
    })
    manifest_raw["metadata"] = metadata
    manifest_text = json.dumps(manifest_raw, indent=2, sort_keys=True) + "\n"
    output_manifest.write_text(manifest_text, encoding="utf-8")
    final_manifest_sha = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()

    report = FQualifiedFABuildReportV2(
        parent=parent,
        source_carrier_sha256=carrier_sha,
        array_content_sha256=content_sha,
        transport_npz_sha256=parent.bundle_npz_sha256,
        volume=volume,
        final_manifest_sha256=final_manifest_sha,
        provenance=provenance,
    )
    report.validate()
    return report


__all__ = [
    "ARRAY_CONTENT_HASH",
    "ExtractionVolumeQualificationV2",
    "FQualifiedFABuildReportV2",
    "SCHEMA_VERSION",
    "SOURCE_CARRIER_HASH",
    "build_qualified_fa_bundle_v2",
    "canonical_array_content_sha256_v1",
    "canonical_source_carrier_sha256_v1",
]
