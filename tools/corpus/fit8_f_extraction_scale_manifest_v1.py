from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from math import isfinite

from tools.corpus.fit8_f_flexicubes_reference_adapter_v1 import (
    FlexiCubesDenseGridV1,
    build_dense_flexicubes_grid_v1,
)

SCHEMA_VERSION = "RealSaS.FExtractionScaleManifest.v1"
ZERO_SHA256 = "0" * 64


def _sha(name: str, value: str) -> None:
    if len(value) != 64:
        raise ValueError(f"F_EXTRACTION_SCALE_SHA_INVALID:{name}")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"F_EXTRACTION_SCALE_SHA_NONHEX:{name}") from exc
    if value == ZERO_SHA256:
        raise ValueError(f"F_EXTRACTION_SCALE_ZERO_SHA_FORBIDDEN:{name}")


@dataclass(frozen=True)
class FExtractionScaleLevelV1:
    scale_id: str
    cell_min_ijk: tuple[int, int, int]
    cell_max_exclusive_ijk: tuple[int, int, int]
    lattice_origin_xyz: tuple[float, float, float]
    extraction_pitch: float
    provenance: str

    @property
    def res(self) -> tuple[int, int, int]:
        return tuple(
            int(self.cell_max_exclusive_ijk[i]) - int(self.cell_min_ijk[i])
            for i in range(3)
        )

    @property
    def cell_count(self) -> int:
        rx, ry, rz = self.res
        return int(rx * ry * rz)

    def validate(self) -> None:
        if not self.scale_id or not self.provenance:
            raise ValueError("F_EXTRACTION_SCALE_LEVEL_IDENTITY_MISSING")
        if len(self.cell_min_ijk) != 3 or len(self.cell_max_exclusive_ijk) != 3:
            raise ValueError("F_EXTRACTION_SCALE_LEVEL_BOUNDS_INVALID")
        if any(v <= 0 for v in self.res):
            raise ValueError("F_EXTRACTION_SCALE_LEVEL_RES_INVALID")
        if len(self.lattice_origin_xyz) != 3 or any(not isfinite(float(v)) for v in self.lattice_origin_xyz):
            raise ValueError("F_EXTRACTION_SCALE_LEVEL_ORIGIN_INVALID")
        if not isfinite(float(self.extraction_pitch)) or self.extraction_pitch <= 0.0:
            raise ValueError("F_EXTRACTION_SCALE_LEVEL_PITCH_INVALID")

    def dense_grid(self) -> FlexiCubesDenseGridV1:
        self.validate()
        return build_dense_flexicubes_grid_v1(
            cell_min_ijk=self.cell_min_ijk,
            cell_max_exclusive_ijk=self.cell_max_exclusive_ijk,
            lattice_origin_xyz=self.lattice_origin_xyz,
            extraction_pitch=self.extraction_pitch,
        )


@dataclass(frozen=True)
class FExtractionScaleManifestV1:
    manifest_id: str
    normalization_frame_sha256: str
    minimum_source_margin_cells: float
    levels: tuple[FExtractionScaleLevelV1, ...]
    provenance: str
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("F_EXTRACTION_SCALE_SCHEMA_INVALID")
        if not self.manifest_id or not self.provenance:
            raise ValueError("F_EXTRACTION_SCALE_IDENTITY_MISSING")
        _sha("normalization_frame_sha256", self.normalization_frame_sha256)
        if not isfinite(float(self.minimum_source_margin_cells)) or self.minimum_source_margin_cells <= 0.0:
            raise ValueError("F_EXTRACTION_SCALE_MARGIN_INVALID")
        if not self.levels:
            raise ValueError("F_EXTRACTION_SCALE_LEVELS_EMPTY")
        ids: set[str] = set()
        geometry_keys: set[tuple] = set()
        for level in self.levels:
            level.validate()
            if level.scale_id in ids:
                raise ValueError("F_EXTRACTION_SCALE_DUPLICATE_ID")
            ids.add(level.scale_id)
            geometry_key = (
                level.cell_min_ijk,
                level.cell_max_exclusive_ijk,
                tuple(float(v) for v in level.lattice_origin_xyz),
                float(level.extraction_pitch),
            )
            if geometry_key in geometry_keys:
                raise ValueError("F_EXTRACTION_SCALE_DUPLICATE_GEOMETRY")
            geometry_keys.add(geometry_key)

    def level(self, scale_id: str) -> FExtractionScaleLevelV1:
        self.validate()
        rows = [row for row in self.levels if row.scale_id == scale_id]
        if len(rows) != 1:
            raise ValueError("F_EXTRACTION_SCALE_ID_UNKNOWN")
        return rows[0]

    def canonical_payload(self) -> dict:
        self.validate()
        payload = asdict(self)
        payload["levels"] = [
            {
                **asdict(level),
                "cell_min_ijk": list(level.cell_min_ijk),
                "cell_max_exclusive_ijk": list(level.cell_max_exclusive_ijk),
                "lattice_origin_xyz": [float(v) for v in level.lattice_origin_xyz],
            }
            for level in self.levels
        ]
        return payload

    def sha256(self) -> str:
        blob = json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


__all__ = [
    "FExtractionScaleLevelV1",
    "FExtractionScaleManifestV1",
    "SCHEMA_VERSION",
]
