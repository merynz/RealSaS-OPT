from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from math import isfinite
from typing import Any, Mapping

import numpy as np

from tools.corpus.fit8_f_flexicubes_reference_adapter_v1 import (
    FlexiCubesDenseGridV1,
    build_dense_flexicubes_grid_v1,
)

SCHEMA_VERSION = "RealSaS.FReferenceDenseGridAuthority.v1"
CUBE_ORDERING = "XYZ_LEXICOGRAPHIC_Z_FASTEST"
SCALAR_SIGN_CONVENTION = "NEGATIVE_INSIDE"
REGULAR_LATTICE = "REGULAR_LATTICE"


@dataclass(frozen=True)
class FReferenceDenseGridAuthorityV1:
    cell_min_ijk: tuple[int, int, int]
    cell_max_exclusive_ijk: tuple[int, int, int]
    lattice_origin_xyz: tuple[float, float, float]
    extraction_pitch: float
    cube_ordering: str
    scalar_sign_convention: str
    vertex_position_mode: str
    bundle_npz_sha256: str
    provenance: str
    metadata: Mapping[str, Any]
    schema_version: str = SCHEMA_VERSION

    @property
    def res(self) -> tuple[int, int, int]:
        return tuple(
            int(self.cell_max_exclusive_ijk[i]) - int(self.cell_min_ijk[i])
            for i in range(3)
        )

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("F_REFERENCE_GRID_AUTHORITY_SCHEMA_INVALID")
        if len(self.cell_min_ijk) != 3 or len(self.cell_max_exclusive_ijk) != 3:
            raise ValueError("F_REFERENCE_GRID_AUTHORITY_BOUNDS_INVALID")
        if any(v <= 0 for v in self.res):
            raise ValueError("F_REFERENCE_GRID_AUTHORITY_RES_INVALID")
        if len(self.lattice_origin_xyz) != 3 or any(not isfinite(float(v)) for v in self.lattice_origin_xyz):
            raise ValueError("F_REFERENCE_GRID_AUTHORITY_ORIGIN_INVALID")
        if not isfinite(float(self.extraction_pitch)) or self.extraction_pitch <= 0.0:
            raise ValueError("F_REFERENCE_GRID_AUTHORITY_PITCH_INVALID")
        if self.cube_ordering != CUBE_ORDERING:
            raise ValueError("F_REFERENCE_GRID_AUTHORITY_CUBE_ORDERING_INVALID")
        if self.scalar_sign_convention != SCALAR_SIGN_CONVENTION:
            raise ValueError("F_REFERENCE_GRID_AUTHORITY_SIGN_CONVENTION_INVALID")
        if self.vertex_position_mode != REGULAR_LATTICE:
            raise ValueError("F_REFERENCE_GRID_AUTHORITY_POSITION_MODE_UNSUPPORTED")
        if len(self.bundle_npz_sha256) != 64:
            raise ValueError("F_REFERENCE_GRID_AUTHORITY_BUNDLE_SHA_INVALID")
        try:
            int(self.bundle_npz_sha256, 16)
        except ValueError as exc:
            raise ValueError("F_REFERENCE_GRID_AUTHORITY_BUNDLE_SHA_NONHEX") from exc
        if not self.provenance:
            raise ValueError("F_REFERENCE_GRID_AUTHORITY_PROVENANCE_MISSING")

    def expected_grid(self) -> FlexiCubesDenseGridV1:
        self.validate()
        return build_dense_flexicubes_grid_v1(
            cell_min_ijk=self.cell_min_ijk,
            cell_max_exclusive_ijk=self.cell_max_exclusive_ijk,
            lattice_origin_xyz=self.lattice_origin_xyz,
            extraction_pitch=self.extraction_pitch,
        )

    def canonical_payload(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["cell_min_ijk"] = list(self.cell_min_ijk)
        payload["cell_max_exclusive_ijk"] = list(self.cell_max_exclusive_ijk)
        payload["lattice_origin_xyz"] = list(self.lattice_origin_xyz)
        return payload

    def sha256(self) -> str:
        blob = json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


def validate_dense_grid_bundle_arrays_v1(
    authority: FReferenceDenseGridAuthorityV1,
    arrays: Mapping[str, np.ndarray],
    *,
    manifest_res: tuple[int, int, int],
) -> None:
    authority.validate()
    if tuple(map(int, manifest_res)) != authority.res:
        raise ValueError("F_REFERENCE_GRID_AUTHORITY_MANIFEST_RES_DRIFT")
    if "x_nx3" not in arrays or "cube_fx8" not in arrays:
        raise ValueError("F_REFERENCE_GRID_AUTHORITY_ARRAY_MISSING")

    expected = authority.expected_grid()
    x = np.asarray(arrays["x_nx3"])
    cubes = np.asarray(arrays["cube_fx8"])
    expected_x = np.asarray(expected.vertex_xyz, dtype=x.dtype)
    expected_cubes = np.asarray(expected.cube_fx8, dtype=cubes.dtype)

    if x.shape != expected_x.shape or not np.array_equal(x, expected_x):
        raise ValueError("F_REFERENCE_GRID_AUTHORITY_VERTEX_ORDER_OR_POSITION_DRIFT")
    if cubes.shape != expected_cubes.shape or not np.array_equal(cubes, expected_cubes):
        raise ValueError("F_REFERENCE_GRID_AUTHORITY_CUBE_ORDER_DRIFT")


__all__ = [
    "CUBE_ORDERING",
    "FReferenceDenseGridAuthorityV1",
    "REGULAR_LATTICE",
    "SCALAR_SIGN_CONVENTION",
    "SCHEMA_VERSION",
    "validate_dense_grid_bundle_arrays_v1",
]
